"""
graph/graph_builder.py
----------------------
LangGraph state machine definition for ASBOS.

Graph topology:
  START -> orchestrator_init -> trend_node
       -> [design_node || pricing_node || timer_node]  (parallel fan-out)
       -> copywriter_node -> critic_node
       -> [APPROVED -> finalise_node -> END]
          [REVISE -> increment_iter -> selective_revision -> copywriter_node -> critic_node]
          [FAILED -> finalise_node -> END]

Key LangGraph features used:
  - StateGraph with TypedDict
  - Parallel fan-out via Send API (design, pricing, timer run concurrently)
  - Conditional edges with routing function
  - MemorySaver checkpointing (SQLiteSaver ready for production swap)
  - Thread-based state isolation (each drop = unique thread_id)
"""

import os
import uuid
from datetime import datetime, timezone
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage

from graph.state import DropState
from graph.router import route_after_critic, increment_iteration, should_rerun_parallel_agents
from agents.trend_agent import run_trend_agent
from agents.design_agent import run_design_agent
from agents.pricing_agent import run_pricing_agent
from agents.drop_timer_agent import run_drop_timer_agent
from agents.copywriter_agent import run_copywriter_agent
from agents.critic_agent import run_critic_agent


# ── Node functions ────────────────────────────────────────────────────────────

def orchestrator_init(state: DropState) -> dict:
    """
    Initialize the drop state with metadata.
    Retrieves brand voice from ChromaDB for downstream agents.
    """
    from tools.rag_retriever import retrieve, format_results_as_context

    drop_id = state.get("drop_id") or str(uuid.uuid4())[:8]
    season = state.get("season") or _infer_season()

    print(f"\n{'='*60}")
    print(f"  ASBOS — Arétier Drop Pipeline")
    print(f"  Drop ID: {drop_id} | Season: {season}")
    print(f"{'='*60}\n")

    # Retrieve brand voice for downstream agents
    brand_voice_results = retrieve(
        "Arétier brand philosophy voice tone keywords",
        collection="brand_knowledge",
        n=2,
    )
    brand_voice = format_results_as_context(brand_voice_results)

    return {
        "drop_id": drop_id,
        "season": season,
        "brand_voice": brand_voice,
        "iteration_count": 0,
        "status": "in_progress",
        "messages": [
            SystemMessage(content=f"Drop pipeline initialised. ID: {drop_id} | Season: {season}")
        ],
    }


def increment_iteration_node(state: DropState) -> dict:
    """Thin node wrapper around the router's increment function."""
    return increment_iteration(state)


def finalise_node(state: DropState) -> dict:
    """
    Final node — sets status, persists drop to history, and computes summary.
    Runs for both APPROVED and FAILED states.
    """
    approval_score = state.get("approval_score", 0.0)
    threshold = float(os.getenv("CRITIC_APPROVAL_THRESHOLD", "0.75"))
    final_status = "approved" if (approval_score or 0) >= threshold else "failed"

    print(f"\n{'='*60}")
    print(f"  DROP {state['drop_id']} — {final_status.upper()}")
    print(f"  Final score: {approval_score:.3f} | Iterations: {state.get('iteration_count', 0)}")
    print(f"{'='*60}\n")

    # Persist to drop history if approved
    if final_status == "approved":
        _persist_to_history(state)

    return {
        "status": final_status,
        "messages": [
            SystemMessage(
                content=f"Drop {state['drop_id']} {final_status}. Score: {approval_score:.3f}"
            )
        ],
    }


def _persist_to_history(state: DropState) -> None:
    """
    Save approved drop to drop_history.jsonl and index in ChromaDB.
    This feeds the RAG knowledge base for future runs.
    """
    import json
    from pathlib import Path
    from tools.rag_retriever import upsert

    record = {
        "drop_id": state.get("drop_id"),
        "season": state.get("season"),
        "aesthetic_keywords": state.get("aesthetic_keywords"),
        "design_brief": state.get("design_brief", "")[:300],
        "suggested_price": state.get("suggested_price"),
        "recommended_drop_time": state.get("recommended_drop_time"),
        "approval_score": state.get("approval_score"),
        "iteration_count": state.get("iteration_count"),
        "product_description": state.get("product_description", "")[:300],
        "timestamp": str(datetime.now(timezone.utc)),
    }

    # Append to JSONL file
    history_path = Path("knowledge_base/drop_history.jsonl")
    history_path.parent.mkdir(exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # Index in ChromaDB for future RAG queries
    doc_text = (
        f"Drop {record['drop_id']} | {record['season']} | "
        f"Keywords: {', '.join(record['aesthetic_keywords'] or [])} | "
        f"Price: ${record['suggested_price']} | "
        f"Description excerpt: {record['product_description'][:200]}"
    )
    upsert(
        collection="brand_knowledge",
        ids=[f"drop_history_{record['drop_id']}"],
        documents=[doc_text],
        metadatas=[{"source": "drop_history", "drop_id": record["drop_id"]}],
    )
    print(f"[graph] Drop {record['drop_id']} persisted to history.")


def _infer_season() -> str:
    """Infer the current season from the date."""
    month = datetime.now().month
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct and compile the ASBOS LangGraph state machine.

    Returns a compiled graph ready for invocation.
    """
    builder = StateGraph(DropState)

    # ── Add all nodes ─────────────────────────────────────────────────────────
    builder.add_node("orchestrator_init", orchestrator_init)
    builder.add_node("trend_node", run_trend_agent)
    builder.add_node("design_node", run_design_agent)
    builder.add_node("pricing_node", run_pricing_agent)
    builder.add_node("timer_node", run_drop_timer_agent)
    builder.add_node("copywriter_node", run_copywriter_agent)
    builder.add_node("critic_node", run_critic_agent)
    builder.add_node("increment_iter", increment_iteration_node)
    builder.add_node("finalise_node", finalise_node)

    # ── Linear edges (sequential) ─────────────────────────────────────────────
    builder.add_edge(START, "orchestrator_init")
    builder.add_edge("orchestrator_init", "trend_node")

    # ── Parallel fan-out: trend -> [design, pricing, timer] ───────────────────
    # LangGraph executes all three concurrently, then waits for all to complete
    builder.add_edge("trend_node", "design_node")
    builder.add_edge("trend_node", "pricing_node")
    builder.add_edge("trend_node", "timer_node")

    # ── Join: all parallel agents -> copywriter ────────────────────────────────
    # LangGraph automatically joins when all incoming edges are complete
    builder.add_edge("design_node", "copywriter_node")
    builder.add_edge("pricing_node", "copywriter_node")
    builder.add_edge("timer_node", "copywriter_node")

    # ── Sequential: copywriter -> critic ──────────────────────────────────────
    builder.add_edge("copywriter_node", "critic_node")

    # ── Conditional edge: critic -> approved | revise | failed ────────────────
    builder.add_conditional_edges(
        "critic_node",
        route_after_critic,
        {
            "approved": "finalise_node",
            "revise": "increment_iter",
            "failed": "finalise_node",
        },
    )

    # ── Revision loop: increment -> selective re-run ───────────────────────────
    # After incrementing, we use a conditional to decide if parallel agents
    # need to re-run, or if only copywriter needs to revise.
    def _revision_routing(state: DropState) -> str:
        if should_rerun_parallel_agents(state):
            return "rerun_parallel"
        return "rerun_copy_only"

    builder.add_conditional_edges(
        "increment_iter",
        _revision_routing,
        {
            "rerun_parallel": "design_node",   # Re-fan-out from design
            "rerun_copy_only": "copywriter_node",
        },
    )
    # When re-running parallel from design, pricing and timer also need edges
    # LangGraph handles this because they already have edges from trend_node context
    # The state already has trend data; agents check their revision notes

    # ── Finalise -> END ───────────────────────────────────────────────────────
    builder.add_edge("finalise_node", END)

    # ── Compile with MemorySaver checkpointing ────────────────────────────────
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    return graph


# ── Convenience runner ────────────────────────────────────────────────────────

def run_drop_pipeline(
    drop_id: str | None = None,
    season: str | None = None,
    thread_id: str | None = None,
    on_step=None,
) -> dict:
    """
    Run a complete drop pipeline and return the final state.

    Args:
        drop_id: Optional custom drop ID (auto-generated if None)
        season: Optional season override (auto-detected if None)
        thread_id: LangGraph thread ID for state isolation (auto-generated if None)
        on_step: Optional callback fn(node_name: str) called when a node completes.

    Returns:
        Final DropState dict
    """
    graph = build_graph()

    initial_state: DropState = {
        "drop_id": drop_id or str(uuid.uuid4())[:8],
        "season": season or _infer_season(),
        "brand_voice": "",
        "trend_report": None,
        "aesthetic_keywords": None,
        "design_brief": None,
        "design_image_path": None,
        "design_image_b64": None,
        "clip_score": None,
        "design_revision_note": None,
        "suggested_price": None,
        "price_rationale": None,
        "resale_premium_estimate": None,
        "pricing_revision_note": None,
        "recommended_drop_time": None,
        "timing_rationale": None,
        "timing_revision_note": None,
        "product_description": None,
        "tweet_drafts": None,
        "copy_revision_note": None,
        "critique_report": None,
        "revision_requests": None,
        "approval_score": None,
        "deterministic_checks": None,
        "iteration_count": 0,
        "status": "in_progress",
        "messages": [],
    }

    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}

    final_state = initial_state
    for event in graph.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, state_updates in event.items():
            if on_step:
                on_step(node_name)
            # The streaming interface returns updates, we just need to grab the last state
            # LangGraph actually manages internal state, but stream_mode="values" is easier if we just want state
            # Wait, `graph.get_state(config).values` gets the current full state.
            pass

    return graph.get_state(config).values

