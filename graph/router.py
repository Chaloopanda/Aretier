"""
graph/router.py
---------------
Routing and conditional edge logic for the LangGraph state machine.

This module defines:
  1. route_after_critic() — the main conditional edge after Critic runs
  2. route_revision()     — determines which agents need to re-run
  3. The parallel fan-out configuration for Design/Pricing/Timer agents
"""

import os
from typing import Literal

from graph.state import DropState

APPROVAL_THRESHOLD = float(os.getenv("CRITIC_APPROVAL_THRESHOLD", "0.75"))
MAX_ITERATIONS = int(os.getenv("MAX_REVISION_ITERATIONS", "3"))


def route_after_critic(
    state: DropState,
) -> Literal["approved", "revise_design", "revise_pricing", "revise_timing", "revise_copy", "revise_all", "failed"]:
    """
    Main routing function called after the Critic Agent completes.

    Decision tree:
      1. If approval_score >= threshold -> APPROVED (go to END node)
      2. If iteration_count >= MAX_ITERATIONS -> FAILED (go to END node with failure status)
      3. Otherwise -> determine which agent(s) need to revise

    Returns a routing key that maps to the next node(s) in the graph.
    Note: For multi-agent revision, we route to a "selective_revision" node
    that re-dispatches only the failing agents.
    """
    score = state.get("approval_score", 0.0)
    iteration = state.get("iteration_count", 0)
    revision_requests = state.get("revision_requests") or {}

    if score >= APPROVAL_THRESHOLD:
        print(f"[router] [OK] APPROVED (score: {score:.3f})")
        return "approved"

    if iteration >= MAX_ITERATIONS:
        print(f"[router] [FAIL] FAILED — max iterations ({MAX_ITERATIONS}) reached. Score: {score:.3f}")
        return "failed"

    print(f"[router] [WARN]  Score {score:.3f} < threshold {APPROVAL_THRESHOLD}. Routing to revision...")
    return "revise"


def get_agents_needing_revision(state: DropState) -> list[str]:
    """
    Returns the list of agent node names that need to re-run based on revision_requests.
    Only agents with non-None revision instructions are included.
    
    Used by the graph builder to construct selective revision edges.
    """
    revision_requests = state.get("revision_requests") or {}
    
    agent_node_map = {
        "design_agent": "design_node",
        "pricing_agent": "pricing_node",
        "drop_timer_agent": "timer_node",
        "copywriter_agent": "copywriter_node",
    }
    
    return [
        agent_node_map[agent]
        for agent, request in revision_requests.items()
        if request is not None and agent in agent_node_map
    ]


def should_rerun_parallel_agents(state: DropState) -> bool:
    """
    Returns True if any of the parallel agents (Design/Pricing/Timer) need revision.
    This determines whether we re-trigger the parallel fan-out or go straight to copywriter.
    """
    revision_requests = state.get("revision_requests") or {}
    parallel_agents = ["design_agent", "pricing_agent", "drop_timer_agent"]
    return any(
        revision_requests.get(agent) is not None
        for agent in parallel_agents
    )


def increment_iteration(state: DropState) -> dict:
    """
    Increment the iteration counter.
    Called as a preprocessing step before entering the revision loop.
    """
    new_count = state.get("iteration_count", 0) + 1
    print(f"[router] Starting revision iteration {new_count}/{MAX_ITERATIONS}")
    return {"iteration_count": new_count}

