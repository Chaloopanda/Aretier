"""
app/streamlit_app.py
---------------------
ASBOS — Autonomous Sneaker Brand Operating System
Arétier Drop Studio

Main Streamlit entry point. Drop Studio page.
"""

import os
import sys
import uuid
import json
import base64
import time
from pathlib import Path
from datetime import datetime

# Ensure project root in path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Arétier — ASBOS Drop Studio",
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0a0a;
    color: #e8e5e0;
  }

  .main { background-color: #0a0a0a; }
  .block-container { padding-top: 2rem; max-width: 1400px; }

  /* Header */
  .aretier-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid #2a2a2a;
    margin-bottom: 2rem;
  }
  .aretier-wordmark {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #e8e5e0;
    margin: 0;
  }
  .aretier-sub {
    font-size: 0.72rem;
    letter-spacing: 0.25em;
    color: #666;
    text-transform: uppercase;
    margin-top: 0.3rem;
  }

  /* Agent cards */
  .agent-card {
    background: #141414;
    border: 1px solid #222;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
  }
  .agent-card.active { border-color: #8b7355; }
  .agent-card.done { border-color: #2d4a2d; }
  .agent-card.failed { border-color: #4a2d2d; }

  .agent-name {
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 0.3rem;
  }
  .agent-status {
    font-size: 0.9rem;
    color: #e8e5e0;
  }

  /* Score gauge */
  .score-display {
    font-size: 3rem;
    font-weight: 700;
    font-family: 'Playfair Display', serif;
    color: #c4a97d;
    text-align: center;
  }
  .score-label {
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #666;
    text-align: center;
  }

  /* Drop card */
  .drop-card {
    background: #141414;
    border: 1px solid #222;
    border-radius: 8px;
    padding: 1.5rem;
  }
  .price-tag {
    font-size: 2.2rem;
    font-weight: 600;
    color: #e8e5e0;
  }
  .price-sub {
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #888;
  }

  /* Tweet card */
  .tweet-card {
    background: #0f0f0f;
    border: 1px solid #1e1e1e;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #d4d0cb;
  }

  /* Status badge */
  .status-approved {
    display: inline-block;
    background: #1a2e1a;
    color: #6abf6a;
    border: 1px solid #2d4a2d;
    border-radius: 4px;
    padding: 0.2rem 0.8rem;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
  }
  .status-failed {
    display: inline-block;
    background: #2e1a1a;
    color: #bf6a6a;
    border: 1px solid #4a2d2d;
    border-radius: 4px;
    padding: 0.2rem 0.8rem;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
  }
  .status-running {
    display: inline-block;
    background: #1e1a0e;
    color: #c4a97d;
    border: 1px solid #3a2e18;
    border-radius: 4px;
    padding: 0.2rem 0.8rem;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
  }

  /* Section labels */
  .section-label {
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e1e1e;
  }

  /* CLIP score bar */
  .clip-bar-container {
    background: #1e1e1e;
    border-radius: 4px;
    height: 6px;
    margin-top: 0.4rem;
  }

  /* Streamlit overrides */
  .stButton > button {
    background: #1e1e1e;
    color: #e8e5e0;
    border: 1px solid #333;
    border-radius: 4px;
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.6rem 2rem;
    transition: all 0.2s;
    width: 100%;
  }
  .stButton > button:hover {
    background: #8b7355;
    border-color: #8b7355;
    color: #0a0a0a;
  }
  .stButton > button:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  div[data-testid="stSelectbox"] label,
  div[data-testid="stTextInput"] label {
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #666;
  }

  .stProgress > div > div { background-color: #8b7355; }

  hr { border-color: #1e1e1e; }
</style>
""", unsafe_allow_html=True)

def _load_drop_history() -> list[dict]:
    history_path = Path("knowledge_base/drop_history.jsonl")
    if not history_path.exists():
        return []
    drops = []
    with open(history_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and line != "[]":
                try:
                    drops.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return drops

# ── Session state init ────────────────────────────────────────────────────────
if "drop_result" not in st.session_state:
    st.session_state.drop_result = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "drop_history" not in st.session_state:
    st.session_state.drop_history = _load_drop_history()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="aretier-header">
  <p class="aretier-wordmark">Arétier</p>
  <p class="aretier-sub">Autonomous Brand Operating System &nbsp;·&nbsp; Drop Studio</p>
</div>
""", unsafe_allow_html=True)

# ── Controls ──────────────────────────────────────────────────────────────────
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 1])

with col_ctrl1:
    season_options = ["Spring", "Summer", "Fall", "Winter"]
    current_month = datetime.now().month
    default_season = (
        "Winter" if current_month in [12, 1, 2] else
        "Spring" if current_month in [3, 4, 5] else
        "Summer" if current_month in [6, 7, 8] else "Fall"
    )
    season = st.selectbox(
        "Season",
        options=season_options,
        index=season_options.index(default_season),
        key="season_select",
    )

with col_ctrl2:
    drop_id_input = st.text_input(
        "Drop ID (optional)",
        placeholder="auto-generated",
        key="drop_id_input",
    )

with col_ctrl3:
    st.markdown("<div style='margin-top:1.6rem;'></div>", unsafe_allow_html=True)
    trigger_btn = st.button(
        "⬛  Trigger New Drop",
        disabled=st.session_state.is_running,
        key="trigger_btn",
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.markdown('<p class="section-label">Agent Pipeline</p>', unsafe_allow_html=True)

    agent_statuses = {
        "Orchestrator": "idle",
        "Trend Research": "idle",
        "Design": "idle",
        "Pricing": "idle",
        "Drop Timer": "idle",
        "Copywriter": "idle",
        "Critic": "idle",
    }

    agent_status_placeholder = st.empty()

    def render_agent_cards(statuses: dict):
        STATUS_ICONS = {
            "idle": "○",
            "running": "◐",
            "done": "●",
            "failed": "✕",
            "skipped": "—",
        }
        STATUS_CLASSES = {
            "idle": "",
            "running": "active",
            "done": "done",
            "failed": "failed",
            "skipped": "",
        }
        html = ""
        for agent, status in statuses.items():
            icon = STATUS_ICONS.get(status, "○")
            cls = STATUS_CLASSES.get(status, "")
            html += f"""
            <div class="agent-card {cls}">
              <div class="agent-name">{agent}</div>
              <div class="agent-status">{icon} {status.capitalize()}</div>
            </div>"""
        agent_status_placeholder.markdown(html, unsafe_allow_html=True)

    render_agent_cards(agent_statuses)

    # Iteration counter
    iter_placeholder = st.empty()
    iter_placeholder.markdown(
        '<p style="font-size:0.7rem; color:#444; margin-top:0.5rem;">Revision iterations: 0 / 3</p>',
        unsafe_allow_html=True
    )


with right_col:
    result_placeholder = st.empty()

    def render_idle_state():
        result_placeholder.markdown("""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; 
                    height:400px; border:1px dashed #222; border-radius:8px; color:#333;">
          <p style="font-size:2rem; margin-bottom:0.5rem;">⬛</p>
          <p style="font-size:0.75rem; letter-spacing:0.2em; text-transform:uppercase;">
            Trigger a drop to begin
          </p>
        </div>
        """, unsafe_allow_html=True)

    def render_result(result: dict):
        """Render the full drop result in the right column."""
        status = result.get("status", "failed")
        approval_score = result.get("approval_score", 0.0) or 0.0
        iterations = result.get("iteration_count", 0)
        drop_id = result.get("drop_id", "—")
        keywords = result.get("aesthetic_keywords") or []
        design_brief = result.get("design_brief", "")
        image_b64 = result.get("design_image_b64")
        clip_score = result.get("clip_score") or 0.0
        price = result.get("suggested_price")
        resale_premium = result.get("resale_premium_estimate") or 0.0
        drop_time = result.get("recommended_drop_time", "")
        description = result.get("product_description", "")
        tweets = result.get("tweet_drafts") or []
        critique = result.get("critique_report", "")

        status_badge = (
            '<span class="status-approved">Approved</span>' if status == "approved"
            else '<span class="status-failed">Failed</span>'
        )

        with result_placeholder.container():
            # ── Top bar ───────────────────────────────────────────────────────
            top_left, top_right = st.columns([3, 1])
            with top_left:
                st.markdown(
                    f'<p style="font-size:0.68rem; letter-spacing:0.2em; color:#555; '
                    f'text-transform:uppercase; margin-bottom:0.3rem;">Drop ID: {drop_id}</p>'
                    f'{status_badge}',
                    unsafe_allow_html=True
                )
            with top_right:
                score_pct = f"{approval_score * 100:.0f}"
                score_color = "#6abf6a" if approval_score >= 0.75 else "#c4a97d" if approval_score >= 0.5 else "#bf6a6a"
                st.markdown(
                    f'<div class="score-display" style="color:{score_color};">{score_pct}</div>'
                    f'<div class="score-label">Approval Score</div>',
                    unsafe_allow_html=True
                )

            st.markdown("<hr style='margin:0.8rem 0;'>", unsafe_allow_html=True)

            # ── Design section ────────────────────────────────────────────────
            st.markdown('<p class="section-label">Design</p>', unsafe_allow_html=True)
            img_col, brief_col = st.columns([1, 1.3])

            with img_col:
                if image_b64:
                    st.markdown(
                        f'<img src="data:image/png;base64,{image_b64}" '
                        f'style="width:100%; border-radius:6px; border:1px solid #1e1e1e;">',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div style="background:#141414; border:1px solid #222; border-radius:6px; '
                        'height:200px; display:flex; align-items:center; justify-content:center; '
                        'color:#333; font-size:0.8rem;">No image generated</div>',
                        unsafe_allow_html=True
                    )

                # CLIP score bar
                clip_pct = min(100, int(clip_score * 250))  # Scale for visibility
                clip_color = "#6abf6a" if clip_score >= 0.35 else "#c4a97d" if clip_score >= 0.28 else "#bf6a6a"
                st.markdown(
                    f'<p style="font-size:0.65rem; letter-spacing:0.15em; text-transform:uppercase; '
                    f'color:#555; margin-top:0.6rem; margin-bottom:0.2rem;">CLIP Alignment</p>'
                    f'<div class="clip-bar-container">'
                    f'<div style="background:{clip_color}; width:{clip_pct}%; height:100%; border-radius:4px;"></div>'
                    f'</div>'
                    f'<p style="font-size:0.75rem; color:{clip_color}; margin-top:0.3rem;">{clip_score:.3f}</p>',
                    unsafe_allow_html=True
                )

            with brief_col:
                st.markdown(
                    f'<p style="font-size:0.65rem; letter-spacing:0.2em; text-transform:uppercase; '
                    f'color:#555; margin-bottom:0.5rem;">Aesthetic Direction</p>',
                    unsafe_allow_html=True
                )
                # Keywords as pills
                pills_html = " ".join(
                    f'<span style="background:#1e1a0e; color:#c4a97d; border:1px solid #3a2e18; '
                    f'border-radius:3px; padding:0.15rem 0.5rem; font-size:0.72rem; '
                    f'margin-right:0.3rem;">{kw}</span>'
                    for kw in keywords
                )
                st.markdown(pills_html, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    f'<p style="font-size:0.82rem; line-height:1.7; color:#c8c4be;">{design_brief[:450]}...</p>',
                    unsafe_allow_html=True
                )

            st.markdown("<hr style='margin:0.8rem 0;'>", unsafe_allow_html=True)

            # ── Pricing + Timing ──────────────────────────────────────────────
            price_col, time_col = st.columns(2)

            with price_col:
                st.markdown('<p class="section-label">Pricing</p>', unsafe_allow_html=True)
                price_str = f"${price:.0f}" if price else "—"
                st.markdown(
                    f'<div class="drop-card">'
                    f'<p class="price-sub">Retail Price</p>'
                    f'<p class="price-tag">{price_str}</p>'
                    f'<p class="price-sub" style="margin-top:0.8rem;">Est. Resale Premium</p>'
                    f'<p style="font-size:1.4rem; color:#c4a97d; font-weight:600;">'
                    f'{resale_premium*100:.0f}%</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with time_col:
                st.markdown('<p class="section-label">Drop Window</p>', unsafe_allow_html=True)
                try:
                    dt = datetime.fromisoformat(drop_time)
                    day_str = dt.strftime("%A")
                    date_str = dt.strftime("%B %d, %Y")
                    time_str = dt.strftime("%I:%M %p EST").lstrip("0")
                except (ValueError, TypeError):
                    day_str, date_str, time_str = "—", "—", "—"

                st.markdown(
                    f'<div class="drop-card">'
                    f'<p class="price-sub">Day</p>'
                    f'<p style="font-size:1.3rem; font-weight:600;">{day_str}</p>'
                    f'<p class="price-sub" style="margin-top:0.6rem;">Date & Time</p>'
                    f'<p style="font-size:0.9rem; color:#c8c4be;">{date_str}</p>'
                    f'<p style="font-size:0.9rem; color:#c4a97d;">{time_str}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown("<hr style='margin:0.8rem 0;'>", unsafe_allow_html=True)

            # ── Copy ─────────────────────────────────────────────────────────
            st.markdown('<p class="section-label">Product Copy</p>', unsafe_allow_html=True)
            st.markdown(
                f'<p style="font-size:0.88rem; line-height:1.8; color:#c8c4be; '
                f'background:#141414; padding:1.2rem; border-radius:6px; '
                f'border:1px solid #1e1e1e;">{description}</p>',
                unsafe_allow_html=True
            )

            st.markdown('<p class="section-label" style="margin-top:1rem;">Tweet Drafts</p>', unsafe_allow_html=True)
            for tweet in tweets:
                st.markdown(f'<div class="tweet-card">{tweet}</div>', unsafe_allow_html=True)

            st.markdown("<hr style='margin:0.8rem 0;'>", unsafe_allow_html=True)

            # ── Critic Report ─────────────────────────────────────────────────
            if critique:
                with st.expander("▸  Critic Agent Report", expanded=False):
                    st.markdown(
                        f'<pre style="font-size:0.75rem; color:#888; white-space:pre-wrap; '
                        f'font-family:monospace;">{critique}</pre>',
                        unsafe_allow_html=True
                    )

    render_idle_state()


# ── Run pipeline on trigger ───────────────────────────────────────────────────
if trigger_btn and not st.session_state.is_running:
    st.session_state.is_running = True
    st.session_state.drop_result = None

    # Check API key
    if not os.getenv("GEMINI_API_KEY"):
        st.error("[FAIL] GEMINI_API_KEY not set. Copy .env.example to .env and add your key.")
        st.session_state.is_running = False
        st.stop()

    drop_id = drop_id_input.strip() or str(uuid.uuid4())[:8]

    # ── Streaming agent status updates ────────────────────────────────────────
    AGENT_SEQUENCE = [
        ("Orchestrator", "Initialising drop pipeline..."),
        ("Trend Research", "Scraping Hypebeast & Reddit, extracting aesthetic keywords..."),
        ("Design", "Writing design brief, generating image, scoring with CLIP..."),
        ("Pricing", "Retrieving resale comps, calculating optimal retail price..."),
        ("Drop Timer", "Analysing historical drops, selecting optimal window..."),
        ("Copywriter", "Writing product description and tweet drafts..."),
        ("Critic", "Running adversarial evaluation..."),
    ]

    statuses = {agent: "idle" for agent, _ in AGENT_SEQUENCE}
    progress_bar = st.progress(0, text="Initialising...")

    try:
        from graph.graph_builder import run_drop_pipeline

        # We can't truly stream LangGraph internals to Streamlit without
        # async streaming, so we update status for each "phase" visually.
        total_steps = len(AGENT_SEQUENCE)

        for i, (agent_name, status_msg) in enumerate(AGENT_SEQUENCE):
            statuses[agent_name] = "running"
            render_agent_cards(statuses)
            progress_bar.progress(
                int((i / total_steps) * 85),
                text=f"{agent_name}: {status_msg}"
            )

            # Actually run the pipeline on the first agent step
            # (LangGraph runs the whole thing synchronously)
            if i == 0:
                try:
                    result = run_drop_pipeline(
                        drop_id=drop_id,
                        season=season,
                        thread_id=str(uuid.uuid4()),
                    )
                except Exception as e:
                    error_msg = str(e)
                    if hasattr(e, "last_attempt") and e.last_attempt is not None:
                        try:
                            error_msg += " " + str(e.last_attempt.exception())
                        except:
                            pass
                    
                    if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "Quota exceeded" in error_msg:
                        st.error("[WARN] **Gemini Free Tier Limit Hit!** You've exceeded 15 requests per minute. Please wait 60 seconds before triggering another drop.")
                        st.stop()
                    else:
                        st.error(f"[WARN] **Pipeline Error:** {error_msg}")
                        st.stop()

            # Simulate per-agent status update (graph has already completed)
            time.sleep(0.3)
            statuses[agent_name] = "done"
            render_agent_cards(statuses)

        progress_bar.progress(100, text="Drop pipeline complete.")
        iter_placeholder.markdown(
            f'<p style="font-size:0.7rem; color:#555; margin-top:0.5rem;">'
            f'Revision iterations: {result.get("iteration_count", 0)} / 3</p>',
            unsafe_allow_html=True
        )

        st.session_state.drop_result = result
        st.session_state.drop_history = _load_drop_history()
        render_result(result)

        # Show toast
        final_status = result.get("status", "failed")
        if final_status == "approved":
            st.toast("[OK] Drop approved and added to Arétier history.", icon="⬛")
        else:
            st.toast("[WARN] Drop failed critic evaluation after max iterations.", icon="[WARN]")

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        for agent in statuses:
            if statuses[agent] == "running":
                statuses[agent] = "failed"
        render_agent_cards(statuses)

    finally:
        st.session_state.is_running = False
        progress_bar.empty()

elif st.session_state.drop_result:
    render_result(st.session_state.drop_result)

