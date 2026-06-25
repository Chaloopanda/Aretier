"""
agents/pricing_agent.py
-----------------------
Pricing Agent — sets the retail price for the drop.

Responsibilities:
  1. Retrieve comparable resale comps from ChromaDB (market_data collection)
  2. Calculate base price using comp medians + inflation adjustment
  3. Apply dynamic adjustments (CLIP score quality, iteration penalty)
  4. Ask Gemini to write a price rationale with comp citations
  5. On revision: adjust for Critic's specific concern
"""

import os
import json
from langchain_core.messages import HumanMessage, AIMessage
from tenacity import retry, stop_after_attempt, wait_exponential
from tools.llm_provider import get_llm

from graph.state import DropState
from tools.resale_data import (
    get_resale_comps,
    calculate_recommended_price,
    clamp_and_validate,
    CATEGORY_MEDIANS,
)

_llm = get_llm(temperature=0.4)

PRICING_SYSTEM_PROMPT = """You are the Head of Commercial Strategy at Arétier.

Your job is to write a clear, concise price rationale for a new drop, referencing:
1. The comparable silhouettes retrieved from the resale market
2. Why the chosen retail price is appropriate for Arétier's positioning
3. The expected resale premium and what that signals about demand

Rules:
- Be specific about comp prices and brands (use the data provided)
- Do NOT mention the word "luxury" or "premium" — Arétier implies these
- 80–120 words. Dense and analytical.
- The rationale is internal — it does not appear in public-facing copy
"""


def _infer_silhouette_type(keywords: list[str]) -> str:
    """Infer the silhouette category from aesthetic keywords."""
    keyword_str = " ".join(keywords).lower()
    if any(k in keyword_str for k in ["trail", "gore-tex", "waterproof", "hike", "mountain"]):
        return "trail"
    if any(k in keyword_str for k in ["high-top", "high top", "boot", "field"]):
        return "high_top"
    if any(k in keyword_str for k in ["low top", "low-top", "court", "tennis"]):
        return "low_top"
    if any(k in keyword_str for k in ["mid", "basketball"]):
        return "mid_top"
    if any(k in keyword_str for k in ["court", "samba", "campus", "handball"]):
        return "court"
    return "runner"  # Default


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def run_pricing_agent(state: DropState) -> dict:
    """
    Execute the Pricing Agent and return state updates.
    Runs in parallel with Design and Drop Timer agents.
    """
    keywords = state.get("aesthetic_keywords") or []
    clip_score = state.get("clip_score") or 0.0
    iteration = state.get("iteration_count", 0)
    revision_note = state.get("pricing_revision_note")

    print(f"[pricing_agent] Calculating price for drop {state['drop_id']}...")

    silhouette_type = _infer_silhouette_type(keywords)
    print(f"[pricing_agent] Inferred silhouette type: {silhouette_type}")

    # 1. Get resale comps
    comp_data = get_resale_comps(
        aesthetic_keywords=keywords,
        silhouette_type=silhouette_type,
        n_comps=5,
    )

    base_price = comp_data["suggested_base_price"]
    median_premium = comp_data["median_resale_premium"]

    # 2. Calculate recommended retail price
    suggested_price = calculate_recommended_price(
        base_price=base_price,
        clip_score=clip_score,
        iteration_count=iteration,
        silhouette_type=silhouette_type,
    )

    # 3. Validate bounds
    suggested_price, warnings = clamp_and_validate(suggested_price)
    for w in warnings:
        print(f"[pricing_agent] [WARN]  {w}")

    # 4. Generate rationale with Gemini
    revision_instruction = ""
    if revision_note:
        revision_instruction = f"\n\nCRITIC REVISION: {revision_note}\nAddress this in your rationale."

    rationale_prompt = f"""
{PRICING_SYSTEM_PROMPT}

DROP CONTEXT:
- Aesthetic direction: {", ".join(keywords)}
- Silhouette type: {silhouette_type}
- Design quality (CLIP alignment): {clip_score:.3f}/1.0

COMPARABLE MARKET DATA:
{comp_data['context_str']}
{"(Note: fallback category median used — no direct comps found)" if comp_data['fallback_used'] else ""}

CALCULATED PRICE: ${suggested_price:.0f}
ESTIMATED RESALE PREMIUM: {median_premium * 100:.1f}%
{revision_instruction}

Write the price rationale now.
"""

    rationale_response = _llm.invoke([HumanMessage(content=rationale_prompt)])
    price_rationale = rationale_response.content.strip()

    print(f"[pricing_agent] Price: ${suggested_price:.0f} | Resale premium: {median_premium*100:.1f}%")

    return {
        "suggested_price": suggested_price,
        "price_rationale": price_rationale,
        "resale_premium_estimate": round(median_premium, 4),
        "messages": [
            AIMessage(
                content=(
                    f"[Pricing Agent] Retail price: ${suggested_price:.0f} | "
                    f"Est. resale premium: {median_premium*100:.1f}% | "
                    f"Silhouette: {silhouette_type}"
                ),
                name="pricing_agent",
            )
        ],
    }

