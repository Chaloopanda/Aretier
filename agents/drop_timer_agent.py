"""
agents/drop_timer_agent.py
--------------------------
Drop Timer Agent — recommends the optimal drop date and time.

Responsibilities:
  1. RAG-query historical drop performance data (drop_timing collection)
  2. Apply heuristics: day-of-week, time-of-day, competitor avoidance
  3. Ask Gemini to synthesise a drop window recommendation
  4. Validate the output date is ≥ 7 days in the future
  5. On revision: adjust per Critic's instruction

Runs in PARALLEL with Design and Pricing agents (LangGraph fan-out).
"""

import os
import json
from datetime import datetime, timezone, timedelta
from langchain_core.messages import HumanMessage, AIMessage
from tenacity import retry, stop_after_attempt, wait_exponential
from tools.llm_provider import get_llm

from graph.state import DropState
from tools.rag_retriever import retrieve, format_results_as_context

_llm = get_llm(temperature=0.3)

# Heuristics (from historical_drops.csv analysis)
PREFERRED_DAYS = ["Thursday", "Friday", "Saturday"]
PREFERRED_HOUR_EST = 11  # 11:00 AM EST — one hour after Nike's standard 10am
MIN_LEAD_DAYS = 7
MAX_LEAD_DAYS = 45

DROP_TIMER_SYSTEM_PROMPT = """You are the Release Strategy Manager at Arétier.

You must recommend a specific drop date and time based on:
1. Historical drop performance data (what day/time maximises sell-out speed for comparable drops)
2. Current season and demand signals
3. Avoiding competitor major drops if known

Rules:
- The drop must be at least 7 days from today
- Prefer Thursday, Friday, or Saturday
- Prefer 11:00 AM EST (one hour after Nike's standard, giving us a news cycle window)
- Avoid December 24–Jan 2 (holiday blackout)
- Provide a specific date and time, not a range

Return your answer in this exact JSON format:
{
    "recommended_drop_time": "YYYY-MM-DDTHH:MM:00-05:00",
    "timing_rationale": "Your 80-120 word rationale here..."
}

Return ONLY valid JSON.
"""


def _get_default_drop_time() -> str:
    """Conservative fallback: next Thursday at 11:00 AM EST."""
    now = datetime.now(timezone.utc)
    # Find next Thursday (weekday 3)
    days_until_thursday = (3 - now.weekday()) % 7
    if days_until_thursday < MIN_LEAD_DAYS:
        days_until_thursday += 7  # Skip to the following Thursday
    drop_dt = now + timedelta(days=max(days_until_thursday, MIN_LEAD_DAYS))
    drop_dt = drop_dt.replace(hour=16, minute=0, second=0, microsecond=0)  # 11am EST = 16:00 UTC
    return drop_dt.strftime("%Y-%m-%dT11:00:00-05:00")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def run_drop_timer_agent(state: DropState) -> dict:
    """
    Execute the Drop Timer Agent and return state updates.
    Runs in parallel with Design and Pricing agents.
    """
    keywords = state.get("aesthetic_keywords") or []
    season = state.get("season", "Current Season")
    revision_note = state.get("timing_revision_note")

    print(f"[drop_timer_agent] Calculating optimal drop window for drop {state['drop_id']}...")

    # 1. RAG-retrieve comparable historical drops
    query = f"{' '.join(keywords)} {season} drop timing performance"
    timing_results = retrieve(query, collection="drop_timing", n=6)
    timing_context = format_results_as_context(timing_results)

    if not timing_context or timing_context.startswith("No relevant"):
        timing_context = (
            "No historical drop data available yet. Using conservative defaults: "
            "Thursday 11AM EST, minimum 7 days lead time."
        )

    # 2. Prepare revision instruction if applicable
    revision_instruction = ""
    if revision_note:
        revision_instruction = f"\n\nCRITIC REVISION: {revision_note}\nAdjust your recommendation accordingly."

    # 3. Ask Gemini for recommendation
    today_str = datetime.now(timezone.utc).strftime("%A, %B %d, %Y")

    prompt = f"""
{DROP_TIMER_SYSTEM_PROMPT}

TODAY'S DATE: {today_str}
SEASON: {season}
AESTHETIC DIRECTION: {", ".join(keywords)}

HISTORICAL DROP PERFORMANCE DATA:
{timing_context}

{revision_instruction}

Provide your drop timing recommendation now.
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
        drop_time = parsed.get("recommended_drop_time", "")
        rationale = parsed.get("timing_rationale", "")

        # 4. Validate the date is in the future with sufficient lead time
        drop_dt = datetime.fromisoformat(drop_time)
        if drop_dt.tzinfo is None:
            drop_dt = drop_dt.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        days_until = (drop_dt - now).days

        if days_until < MIN_LEAD_DAYS:
            print(f"[drop_timer_agent] Recommended time too soon ({days_until} days). Using safe fallback.")
            drop_time = _get_default_drop_time()
            rationale += f" (Note: original recommendation adjusted — insufficient lead time.)"

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"[drop_timer_agent] Parse error: {e}. Using safe fallback.")
        drop_time = _get_default_drop_time()
        rationale = (
            "Defaulting to conservative drop window: Thursday 11:00 AM EST, "
            f"minimum {MIN_LEAD_DAYS} days lead time. "
            "Historical data suggests Thursday morning drops outperform weekday alternatives."
        )

    print(f"[drop_timer_agent] Recommended drop time: {drop_time}")

    return {
        "recommended_drop_time": drop_time,
        "timing_rationale": rationale,
        "messages": [
            AIMessage(
                content=f"[Drop Timer Agent] Recommended: {drop_time}",
                name="drop_timer_agent",
            )
        ],
    }
