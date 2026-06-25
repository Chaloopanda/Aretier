"""
agents/copywriter_agent.py
--------------------------
Copywriter Agent — writes all public-facing text for the drop.

Responsibilities:
  1. RAG-retrieve brand voice guidelines from ChromaDB
  2. Generate a 100-150 word product description following Arétier's formula
  3. Generate 3 distinct tweet drafts with varied structures
  4. Post-process outputs to scrub banned openers, price mentions, and vocabulary
  5. On revision: rewrite specifically the parts flagged by Critic

Runs AFTER Design, Pricing, and Drop Timer agents complete.
"""

import os
import re
import json
from langchain_core.messages import HumanMessage, AIMessage
from tenacity import retry, stop_after_attempt, wait_exponential
from tools.llm_provider import get_llm

from graph.state import DropState
from tools.rag_retriever import retrieve, format_results_as_context

_llm = get_llm(temperature=0.85)

# ── Post-processing rules ─────────────────────────────────────────────────────
BANNED_OPENERS = [
    "introducing", "excited to", "we are pleased", "we are thrilled",
    "pleased to announce", "happy to announce", "meet the", "say hello to"
]

BANNED_VOCAB = {
    "luxury": "refined",
    "premium": "built",
    "game-changer": "shift",
    "next-level": "evolved",
    "elevate": "earn",
    "limited edition": "this drop",
    "for the culture": "",
}


def _scrub_banned_opener(text: str) -> str:
    """Return the text unchanged but flag if it starts with a banned opener."""
    text_lower = text.lower().strip()
    for opener in BANNED_OPENERS:
        if text_lower.startswith(opener):
            return None  # Signal to regenerate
    return text


def _scrub_price_mentions(text: str) -> str:
    """Remove any price mentions (e.g. $185, 185 USD)."""
    text = re.sub(r"\$\d+(\.\d{2})?", "[PRICE REDACTED]", text)
    text = re.sub(r"\d+\s*(USD|usd|dollars?)", "[PRICE REDACTED]", text)
    return text


def _replace_banned_vocab(text: str) -> str:
    """Replace banned vocabulary with Arétier-appropriate alternatives."""
    for banned, replacement in BANNED_VOCAB.items():
        if replacement:
            text = re.sub(re.escape(banned), replacement, text, flags=re.IGNORECASE)
        else:
            text = re.sub(re.escape(banned), "", text, flags=re.IGNORECASE)
    return text


COPYWRITER_SYSTEM_PROMPT = """You are the Lead Copywriter at Arétier.

BRAND ESSENCE: Arétier (Ah-ray-tee-ay) — "The Maker of Excellence."
Rooted in Arete (Greek: earned excellence, valour) + -ier (French artisan suffix).
Core brand pillars: Valour. Strength. Persistence.

PRODUCT DESCRIPTION FORMULA:
1. Provenance line (1 sentence) — What idea, terrain, or resistance does this silhouette answer?
2. Construction statement (2-3 sentences) — What is actually built? Materials, structure, key engineering.
3. The customer truth (1-2 sentences) — Who is this for without naming them?
Target: 100-150 words. Be dense. No fluff. Write at least 100 words.

TWEET STRUCTURE OPTIONS (use a DIFFERENT structure for each tweet):
Option A: "[Terrain/context line.] [Product name] [Drop date]."
Option B: "[A truth about the customer.] [Product name] ->"
Option C: "Drop alert [ALERT] [Product name] [drops/hits/arrives] [Day Time]."

HARD RULES:
- Never start with: Introducing / Excited / We are pleased / Meet the / Say hello to
- No exclamation marks in product description
- No price mentions anywhere
- No hashtags in tweet body (put them at the end or omit)
- End product description with a period
- Each tweet must start with a DIFFERENT word
"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def run_copywriter_agent(state: DropState) -> dict:
    """
    Execute the Copywriter Agent and return state updates.
    Runs sequentially after Design, Pricing, and Drop Timer complete.
    """
    keywords = state.get("aesthetic_keywords") or []
    design_brief = state.get("design_brief", "")
    drop_time = state.get("recommended_drop_time", "TBD")
    revision_note = state.get("copy_revision_note")
    iteration = state.get("iteration_count", 0)

    print(f"[copywriter_agent] Writing copy for drop {state['drop_id']}...")

    # 1. Retrieve brand voice context
    brand_voice_results = retrieve(
        "Arétier brand voice tone vocabulary formula product description",
        collection="brand_knowledge",
        n=3,
    )
    brand_voice_context = format_results_as_context(brand_voice_results)

    # 2. Retrieve previous product descriptions for consistency
    history_results = retrieve(
        "product description sneaker copy",
        collection="brand_knowledge",
        n=2,
        where={"source": "drop_history"},
    )
    history_context = format_results_as_context(history_results)

    # 3. Format drop time for copy
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(drop_time)
        drop_time_readable = dt.strftime("%A %B %d, %I:%M %p EST").replace(" 0", " ")
    except (ValueError, TypeError):
        drop_time_readable = "Coming Soon"

    revision_instruction = ""
    if revision_note:
        revision_instruction = f"\n\nCRITIC REVISION: {revision_note}\nAddress this issue specifically in your rewrite."

    # 4. Generate copy
    prompt = f"""
{COPYWRITER_SYSTEM_PROMPT}

BRAND VOICE REFERENCE:
{brand_voice_context or "Use Arétier brand voice: direct, earned, grounded in craft."}

PREVIOUS ARÉTIER COPY (for consistency):
{history_context or "First drop — establish the voice from scratch."}

THIS DROP DETAILS:
- Aesthetic direction: {", ".join(keywords)}
- Design brief: {design_brief[:400]}...
- Drop date/time: {drop_time_readable}

{revision_instruction}

Generate the product description and 3 tweet drafts in this exact JSON format:
{{
    "product_description": "Your 120-180 word product description here...",
    "tweet_drafts": [
        "First tweet (terrain/context structure)...",
        "Second tweet (customer truth structure)...",
        "Third tweet (drop alert structure)..."
    ]
}}

Return ONLY valid JSON.
"""

    response = _llm.invoke([HumanMessage(content=prompt)])
    raw_content = response.content.strip()

    # Clean markdown if present
    if raw_content.startswith("```"):
        raw_content = raw_content.split("```")[1]
        if raw_content.startswith("json"):
            raw_content = raw_content[4:]

    try:
        parsed = json.loads(raw_content.strip())
        product_description = parsed.get("product_description", "")
        tweet_drafts = parsed.get("tweet_drafts", [])
    except json.JSONDecodeError:
        print("[copywriter_agent] JSON parse failed — extracting manually.")
        product_description = raw_content[:600]
        tweet_drafts = ["Drop coming soon.", "Stay tuned.", "Arétier ->"]

    # 5. Post-process: scrub banned openers, price mentions, bad vocabulary
    product_description = _scrub_price_mentions(product_description)
    product_description = _replace_banned_vocab(product_description)

    # Check opener and note for Critic if needed
    scrubbed = _scrub_banned_opener(product_description)
    if scrubbed is None:
        print("[copywriter_agent] [WARN]  Description starts with banned opener — Critic will flag.")

    cleaned_tweets = []
    for tweet in tweet_drafts[:3]:
        tweet = _scrub_price_mentions(tweet)
        tweet = _replace_banned_vocab(tweet)
        cleaned_tweets.append(tweet)

    # Pad to 3 tweets if LLM returned fewer
    while len(cleaned_tweets) < 3:
        cleaned_tweets.append(f"Arétier. Built for those who earn it. Drop coming {drop_time_readable} ->")

    print(f"[copywriter_agent] Description: {len(product_description.split())} words | {len(cleaned_tweets)} tweets")

    return {
        "product_description": product_description,
        "tweet_drafts": cleaned_tweets,
        "messages": [
            AIMessage(
                content=(
                    f"[Copywriter Agent] Description written ({len(product_description.split())} words). "
                    f"{len(cleaned_tweets)} tweet drafts generated."
                ),
                name="copywriter_agent",
            )
        ],
    }

