"""Critic Dashboard — radar chart of agent scores per drop."""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Arétier — Critic Dashboard", page_icon="⬛", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:wght@700&display=swap');
html, body, [class*="css"] { font-family:'Inter',sans-serif; background:#0a0a0a; color:#e8e5e0; }
.main { background:#0a0a0a; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<p style="font-family:'Playfair Display',serif; font-size:1.8rem; letter-spacing:0.05em; 
   margin-bottom:0.2rem;">Arétier</p>
<p style="font-size:0.65rem; letter-spacing:0.25em; text-transform:uppercase; color:#555; 
   margin-bottom:2rem;">Critic Dashboard</p>
""", unsafe_allow_html=True)

if "drop_result" not in st.session_state or not st.session_state.get("drop_result"):
    st.markdown(
        '<p style="color:#444; font-size:0.9rem;">Run a drop first to see critic scores.</p>',
        unsafe_allow_html=True
    )
else:
    result = st.session_state.drop_result
    det_checks = result.get("deterministic_checks") or {}

    categories = ["Design", "Pricing", "Timing", "Copy"]
    key_map = {"Design": "design", "Pricing": "pricing", "Timing": "timing", "Copy": "copy"}
    scores = [det_checks.get(key_map[c], {}).get("score", 0.0) * 100 for c in categories]

    # Close the radar loop
    categories_loop = categories + [categories[0]]
    scores_loop = scores + [scores[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores_loop,
        theta=categories_loop,
        fill='toself',
        fillcolor='rgba(139, 115, 85, 0.15)',
        line=dict(color='#8b7355', width=2),
        name='Current Drop',
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='#141414',
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix="%",
                gridcolor='#2a2a2a',
                linecolor='#2a2a2a',
                tickfont=dict(color='#666', size=10),
            ),
            angularaxis=dict(
                gridcolor='#2a2a2a',
                linecolor='#2a2a2a',
                tickfont=dict(color='#aaa', size=12),
            ),
        ),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        font=dict(color='#e8e5e0', family='Inter'),
        margin=dict(t=40, b=40, l=40, r=40),
        height=400,
        showlegend=False,
    )

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p style="font-size:0.65rem; letter-spacing:0.2em; text-transform:uppercase; '
                    'color:#555; margin-bottom:1rem;">Issue Breakdown</p>', unsafe_allow_html=True)

        for cat in categories:
            key = key_map[cat]
            check = det_checks.get(key, {})
            passed = check.get("passed", True)
            score = check.get("score", 1.0)
            issues = check.get("issues", [])

            status_icon = "●" if passed else "○"
            status_color = "#6abf6a" if passed else "#bf6a6a"

            st.markdown(
                f'<div style="margin-bottom:0.8rem; padding:0.8rem; background:#141414; '
                f'border:1px solid #1e1e1e; border-radius:6px;">'
                f'<p style="font-size:0.68rem; letter-spacing:0.15em; text-transform:uppercase; '
                f'color:{status_color}; margin:0 0 0.3rem;">{status_icon} {cat} — {score*100:.0f}%</p>'
                + (
                    "".join(f'<p style="font-size:0.78rem; color:#888; margin:0.2rem 0;">• {issue}</p>'
                            for issue in issues)
                    if issues else
                    '<p style="font-size:0.78rem; color:#555; margin:0;">No issues flagged.</p>'
                )
                + '</div>',
                unsafe_allow_html=True
            )

    # Revision requests summary
    revision_requests = result.get("revision_requests") or {}
    revisions_made = {k: v for k, v in revision_requests.items() if v}
    if revisions_made:
        st.markdown('<hr style="border-color:#1e1e1e; margin:1.5rem 0;">', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.65rem; letter-spacing:0.2em; text-transform:uppercase; '
                    'color:#555; margin-bottom:1rem;">Revision Instructions Issued</p>', unsafe_allow_html=True)
        for agent, instruction in revisions_made.items():
            with st.expander(f"▸  {agent}"):
                st.markdown(f'<p style="font-size:0.82rem; color:#888; white-space:pre-wrap;">{instruction}</p>',
                            unsafe_allow_html=True)
