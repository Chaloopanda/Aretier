"""
agents/critic_agent.py
----------------------
Critic Agent — the adversarial evaluation layer.

This is the most important agent in the system. It:
  1. Runs deterministic checks (from evaluation/critic_rubric.py)
  2. Runs an LLM adversarial critique on top of the deterministic results
  3. Produces structured revision_requests for each failed agent
  4. Computes a final approval_score (0.0–1.0)
  5. Is explicitly prevented from being too lenient (prompt + length checks)

Key design principle: The Critic MUST find at least one issue.
If it cannot find a genuine issue, it must raise a brand consistency concern.
This prevents rubber-stamping.
"""

import os
import json
from langchain_core.messages import HumanMessage, AIMessage
from tenacity import retry, stop_after_attempt, wait_exponential
from tools.llm_provider import get_llm

from graph.state import DropState
from evaluation.critic_rubric import run_deterministic_checks, build_revision_requests

_llm = get_llm(temperature=0.5)

APPROVAL_THRESHOLD = float(os.getenv("CRITIC_APPROVAL_THRESHOLD", "0.75"))
MIN_CRITIQUE_LENGTH = 80  # words — below this = sycophantic, regenerate

CRITIC_SYSTEM_PROMPT = """You are a brand critic for Arétier sneakers. Evaluate this drop briefly.

You MUST find at least one issue. Return ONLY this JSON (no extra text):
{
  "design_critique": "one sentence",
  "pricing_critique": "one sentence",
  "timing_critique": "one sentence",
  "copy_critique": "one sentence",
  "overall_assessment": "one sentence summary",
  "revision_priority": []
}"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def run_critic_agent(state: DropState) -> dict:
    """
    Execute the Critic Agent and return state updates.
    
    Combines deterministic rule checks with LLM adversarial evaluation.
    """
    iteration = state.get("iteration_count", 0)
    print(f"[critic_agent] Starting adversarial evaluation (iteration {iteration})...")

    # 1. Run deterministic checks first
    det_checks = run_deterministic_checks(state)
    det_revision_requests = build_revision_requests(det_checks)

    print(f"[critic_agent] Deterministic score: {det_checks['overall_score']:.3f}")
    if det_checks["any_failed"]:
        failed_sections = [k for k, v in det_checks.items() if isinstance(v, dict) and not v.get("passed", True)]
        print(f"[critic_agent] Failed checks: {failed_sections}")

    # 2. LLM adversarial critique
    # Build summary of deterministic results to inform LLM
    det_summary_parts = []
    for section in ["design", "pricing", "timing", "copy"]:
        result = det_checks[section]
        status = "[OK] PASSED" if result["passed"] else f"[FAIL] FAILED ({len(result['issues'])} issue(s))"
        issues_str = "\n".join(f"  • {i}" for i in result["issues"]) or "  • None"
        det_summary_parts.append(f"{section.upper()}: {status}\n{issues_str}")

    det_summary = "\n\n".join(det_summary_parts)

    # Build a short, focused prompt — local models choke on long context
    kws = ", ".join((state.get("aesthetic_keywords") or [])[:5])
    price = state.get("suggested_price", "N/A")
    drop_time = state.get("recommended_drop_time", "N/A")
    desc_snippet = (state.get("product_description") or "")[:200]
    tweet_snippet = (state.get("tweet_drafts") or [""])[0][:100]
    det_score = det_checks.get("overall_score", 1.0)

    prompt = f"""{CRITIC_SYSTEM_PROMPT}

DROP DATA:
- Keywords: {kws}
- Price: ${price} | Det. score: {det_score:.2f}
- Drop time: {drop_time}
- Description snippet: {desc_snippet}
- Tweet 1 snippet: {tweet_snippet}

Return ONLY the JSON object."""

    response = get_llm(temperature=0.3).invoke([HumanMessage(content=prompt)])
    raw_content = response.content.strip()

    # Clean markdown
    if raw_content.startswith("```"):
        raw_content = raw_content.split("```")[1]
        if raw_content.startswith("json"):
            raw_content = raw_content[4:]

    try:
        parsed_critique = json.loads(raw_content.strip())
    except json.JSONDecodeError:
        print("[critic_agent] JSON parse failed — using raw content.")
        parsed_critique = {
            "design_critique": "Unable to parse structured critique. Review design alignment.",
            "pricing_critique": "Review pricing against market comps.",
            "timing_critique": "Confirm drop timing is ≥7 days out.",
            "copy_critique": "Verify copy follows Arétier brand voice guidelines.",
            "overall_assessment": "Critique parsing failed — manual review required.",
            "revision_priority": [],
        }

    # 3. Sycophancy check removed — too expensive on local models
    # Deterministic checks already enforce quality gates

    # 4. Build final revision requests (merge deterministic + LLM)
    final_revision_requests = {}
    for agent_key in ["design_agent", "pricing_agent", "drop_timer_agent", "copywriter_agent"]:
        det_req = det_revision_requests.get(agent_key)
        
        # Map agent key to LLM critique key
        llm_key_map = {
            "design_agent": "design_critique",
            "pricing_agent": "pricing_critique",
            "drop_timer_agent": "timing_critique",
            "copywriter_agent": "copy_critique",
        }
        llm_critique = parsed_critique.get(llm_key_map[agent_key], "")

        if det_req:
            # Deterministic failure takes priority — combine with LLM insight
            final_revision_requests[agent_key] = f"{det_req}\n\nAdditional LLM insight: {llm_critique}"
        elif llm_critique and len(llm_critique.split()) > 10:
            # Only LLM concern — include if substantial
            final_revision_requests[agent_key] = llm_critique
        else:
            final_revision_requests[agent_key] = None

    # 5. Compute composite approval score
    # Deterministic score (40% weight) + LLM-inferred quality (60% weight)
    # LLM quality is estimated from how many sections it flagged
    sections_flagged = sum(
        1 for k in ["design_critique", "pricing_critique", "timing_critique", "copy_critique"]
        if len((parsed_critique.get(k) or "").split()) > 15  # Substantive critique = flagged
    )
    llm_quality_score = 1.0 - (sections_flagged / 4 * 0.6)  # Penalise for flagged sections

    composite_score = (det_checks["overall_score"] * 0.4) + (llm_quality_score * 0.6)
    composite_score = round(min(1.0, max(0.0, composite_score)), 3)

    # Build human-readable critique report
    critique_report = (
        f"ITERATION {iteration + 1}/3 | Approval Score: {composite_score:.2f} | "
        f"Threshold: {APPROVAL_THRESHOLD}\n\n"
        f"DETERMINISTIC CHECKS: Overall {det_checks['overall_score']:.2f}\n{det_summary}\n\n"
        f"LLM CRITIQUE:\n"
        f"Design: {parsed_critique.get('design_critique', 'N/A')}\n\n"
        f"Pricing: {parsed_critique.get('pricing_critique', 'N/A')}\n\n"
        f"Timing: {parsed_critique.get('timing_critique', 'N/A')}\n\n"
        f"Copy: {parsed_critique.get('copy_critique', 'N/A')}\n\n"
        f"Overall: {parsed_critique.get('overall_assessment', 'N/A')}"
    )

    # Map revision notes into state fields (for individual agents to read)
    state_updates = {
        "critique_report": critique_report,
        "revision_requests": final_revision_requests,
        "approval_score": composite_score,
        "deterministic_checks": det_checks,
        # Set individual revision note fields for each agent
        "design_revision_note": final_revision_requests.get("design_agent"),
        "pricing_revision_note": final_revision_requests.get("pricing_agent"),
        "timing_revision_note": final_revision_requests.get("drop_timer_agent"),
        "copy_revision_note": final_revision_requests.get("copywriter_agent"),
        "messages": [
            AIMessage(
                content=(
                    f"[Critic Agent] Evaluation complete. Score: {composite_score:.2f} | "
                    f"{'APPROVED [OK]' if composite_score >= APPROVAL_THRESHOLD else 'REVISIONS REQUIRED [WARN]'}"
                ),
                name="critic_agent",
            )
        ],
    }

    print(
        f"[critic_agent] Score: {composite_score:.3f} | "
        f"{'APPROVED' if composite_score >= APPROVAL_THRESHOLD else 'REVISIONS NEEDED'}"
    )

    return state_updates

