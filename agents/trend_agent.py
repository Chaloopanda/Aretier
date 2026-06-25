"""
agents/trend_agent.py
---------------------
Trend Research Agent — the first node in the drop pipeline.

Responsibilities:
  1. Scrape Hypebeast RSS + Reddit for current sneaker/fashion trends
  2. RAG-check drop history to avoid repetitive aesthetics
  3. Ask Gemini to extract concrete aesthetic keywords for the Design Agent
  4. Cache all raw data to ChromaDB for offline/rate-limited fallback
"""

import os
import json
from langchain_core.messages import HumanMessage, AIMessage
from tenacity import retry, stop_after_attempt, wait_exponential
from tools.llm_provider import get_llm

from graph.state import DropState
from tools.web_scraper import (
    fetch_hypebeast_trends,
    fetch_reddit_trends,
    fetch_highsnobiety_trends,
    compile_trend_summary,
)
from tools.rag_retriever import retrieve, format_results_as_context

_llm = get_llm(temperature=0.7)

TREND_SYSTEM_PROMPT = """You are the Trend Research Director for Arétier, a high-end sneaker brand 
rooted in the philosophy of Arete — earned excellence, valour, and persistence.

Your job is to analyse current sneaker and fashion trend signals and identify a coherent 
aesthetic direction for the next drop. You must:

1. Identify the dominant aesthetic movements in the current data
2. Check if the proposed direction clashes with recent Arétier drops (to avoid repetition)
3. Extract 5–8 concrete, specific aesthetic keywords — NOT vague ones like "cool" or "modern"
   Good examples: "gorpcore", "earth tones", "ripstop upper", "trail silhouette", "muted sage"
4. Write a concise trend report (150–200 words) that explains the direction and why it fits Arétier

Brand values to keep in mind: Valour, Strength, Persistence. Anti-maximalist. Anti-neon.
Materials: technical fabrics, natural textures. Silhouettes: trail runners, field boots, low trainers.
"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def run_trend_agent(state: DropState) -> dict:
    """
    Execute the Trend Agent and return state updates.
    
    Returns only the fields this agent is responsible for.
    LangGraph merges these into the shared DropState.
    """
    print(f"[trend_agent] Starting trend research for drop {state['drop_id']}...")

    # 1. Scrape live trend data (3 sources, no API keys needed)
    hypebeast_items = fetch_hypebeast_trends(max_items=15)
    reddit_items = fetch_reddit_trends(max_posts=25)
    highsnobiety_items = fetch_highsnobiety_trends(max_items=10)
    trend_raw = compile_trend_summary(hypebeast_items, reddit_items, highsnobiety_items)

    # 2. Retrieve drop history to check for aesthetic repetition
    drop_history_results = retrieve(
        "recent drop aesthetic keywords silhouette",
        collection="brand_knowledge",
        n=5,
        where={"source": "drop_history"},
    )
    history_context = format_results_as_context(drop_history_results)

    # 3. Ask Gemini to synthesise trend report + keywords
    prompt = f"""
{TREND_SYSTEM_PROMPT}

CURRENT TREND DATA:
{trend_raw}

RECENT ARÉTIER DROP HISTORY:
{history_context or "No previous drops — this is the first Arétier drop."}

SEASON: {state.get('season', 'Current Season')}

Based on the above, provide your analysis in the following JSON format:
{{
    "trend_report": "Your 150-200 word trend narrative here...",
    "aesthetic_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
    "avoided_aesthetics": ["any aesthetics too similar to recent drops"]
}}

Return ONLY valid JSON. No markdown, no explanation outside the JSON.
"""

    response = _llm.invoke([HumanMessage(content=prompt)])
    raw_content = response.content.strip()

    # Parse JSON response — handle LLM wrapping in markdown code blocks
    if raw_content.startswith("```"):
        raw_content = raw_content.split("```")[1]
        if raw_content.startswith("json"):
            raw_content = raw_content[4:]

    try:
        parsed = json.loads(raw_content.strip())
    except json.JSONDecodeError as e:
        print(f"[trend_agent] JSON parse failed: {e}. Using fallback.")
        parsed = {
            "trend_report": raw_content[:500],
            "aesthetic_keywords": ["trail silhouette", "earth tones", "technical fabric",
                                   "muted palette", "utilitarian"],
        }

    trend_report = parsed.get("trend_report", "")
    aesthetic_keywords = parsed.get("aesthetic_keywords", [])[:8]  # Cap at 8

    print(f"[trend_agent] Keywords: {aesthetic_keywords}")

    return {
        "trend_report": trend_report,
        "aesthetic_keywords": aesthetic_keywords,
        "messages": [
            AIMessage(
                content=f"[Trend Agent] Completed. Keywords: {aesthetic_keywords}",
                name="trend_agent",
            )
        ],
    }
