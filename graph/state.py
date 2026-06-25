"""
graph/state.py
--------------
Defines the central DropState TypedDict — the single shared state object
that all agents read from and write to in the LangGraph state machine.

Design principles:
  - All fields explicitly typed and Optional where an agent hasn't run yet
  - Uses LangGraph's add_messages reducer for the message history (append-only)
  - Avoids storing large blobs (images stored on disk; state holds path only)
  - iteration_count and status are the orchestrator's control levers
"""

from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages


class DropState(TypedDict):
    # ── Identity ──────────────────────────────────────────────────────────────
    drop_id: str                             # Unique ID per drop cycle (uuid4)
    season: str                              # e.g. "Summer 2025"
    brand_voice: str                         # RAG-retrieved brand_voice.md excerpt

    # ── Trend Agent outputs ───────────────────────────────────────────────────
    trend_report: Optional[str]              # Summarised trend signals narrative
    aesthetic_keywords: Optional[list[str]]  # e.g. ["gorpcore", "earth tones"]

    # ── Design Agent outputs ──────────────────────────────────────────────────
    design_brief: Optional[str]              # Full text description of the silhouette
    design_image_path: Optional[str]         # Absolute path to saved image on disk
    design_image_b64: Optional[str]          # Base64 of image for Streamlit display
    clip_score: Optional[float]              # CLIP text-image alignment (0.0–1.0)
    design_revision_note: Optional[str]      # Instruction from Critic for revision

    # ── Pricing Agent outputs ─────────────────────────────────────────────────
    suggested_price: Optional[float]         # Recommended retail price (USD)
    price_rationale: Optional[str]           # Explanation with resale comp citations
    resale_premium_estimate: Optional[float] # Predicted % above retail on resale mkt
    pricing_revision_note: Optional[str]     # Instruction from Critic for revision

    # ── Drop Timer Agent outputs ──────────────────────────────────────────────
    recommended_drop_time: Optional[str]     # ISO 8601 datetime string
    timing_rationale: Optional[str]          # Why this window was chosen
    timing_revision_note: Optional[str]      # Instruction from Critic for revision

    # ── Copywriter Agent outputs ──────────────────────────────────────────────
    product_description: Optional[str]       # 120–180 word product page copy
    tweet_drafts: Optional[list[str]]        # 3 tweet options for launch day
    copy_revision_note: Optional[str]        # Instruction from Critic for revision

    # ── Critic Agent outputs ──────────────────────────────────────────────────
    critique_report: Optional[str]           # Full adversarial critique narrative
    revision_requests: Optional[dict]        # {agent_name: instruction | None}
    approval_score: Optional[float]          # Aggregate quality score (0.0–1.0)
    deterministic_checks: Optional[dict]     # Results of hard rule checks per agent

    # ── Orchestrator control ──────────────────────────────────────────────────
    iteration_count: int                     # Revision loop counter (hard cap: 3)
    status: str                              # "in_progress" | "approved" | "failed"

    # ── Full message history (append-only via add_messages reducer) ───────────
    messages: Annotated[list, add_messages]
