"""Drop History page — all past Arétier drops with scores and iteration counts."""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Arétier — Drop History", page_icon="⬛", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:wght@700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background:#0a0a0a; color:#e8e5e0; }
.main { background:#0a0a0a; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<p style="font-family:'Playfair Display',serif; font-size:1.8rem; letter-spacing:0.05em; 
   margin-bottom:0.2rem;">Arétier</p>
<p style="font-size:0.65rem; letter-spacing:0.25em; text-transform:uppercase; color:#555; 
   margin-bottom:2rem;">Drop History</p>
""", unsafe_allow_html=True)

history_path = Path("knowledge_base/drop_history.jsonl")

if not history_path.exists() or history_path.stat().st_size < 5:
    st.markdown(
        '<p style="color:#444; font-size:0.9rem;">No drops completed yet. '
        'Trigger your first drop from the Drop Studio.</p>',
        unsafe_allow_html=True
    )
else:
    drops = []
    with open(history_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and line != "[]":
                try:
                    drops.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not drops:
        st.info("No approved drops yet.")
    else:
        df = pd.DataFrame(drops)

        # Clean up columns for display
        display_cols = {
            "drop_id": "Drop ID",
            "season": "Season",
            "suggested_price": "Retail Price",
            "approval_score": "Approval Score",
            "iteration_count": "Iterations",
            "recommended_drop_time": "Drop Time",
            "timestamp": "Created",
        }

        display_df = df[[c for c in display_cols if c in df.columns]].rename(columns=display_cols)

        if "Retail Price" in display_df.columns:
            display_df["Retail Price"] = display_df["Retail Price"].apply(
                lambda x: f"${x:.0f}" if pd.notna(x) else "—"
            )
        if "Approval Score" in display_df.columns:
            display_df["Approval Score"] = display_df["Approval Score"].apply(
                lambda x: f"{x*100:.0f}%" if pd.notna(x) else "—"
            )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        # Detail view
        st.markdown('<hr style="border-color:#1e1e1e; margin:2rem 0;">', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.65rem; letter-spacing:0.2em; text-transform:uppercase; '
                    'color:#555;">Drop Detail</p>', unsafe_allow_html=True)

        selected_id = st.selectbox(
            "Select a drop to view",
            options=[d.get("drop_id", "?") for d in drops],
            key="history_select",
        )

        selected = next((d for d in drops if d.get("drop_id") == selected_id), None)
        if selected:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Keywords:** {', '.join(selected.get('aesthetic_keywords') or [])}")
                st.markdown(f"**Retail:** ${selected.get('suggested_price', '—')}")
                st.markdown(f"**Score:** {(selected.get('approval_score') or 0)*100:.0f}%")
                st.markdown(f"**Iterations:** {selected.get('iteration_count', 0)}")
            with col2:
                st.markdown(f"**Season:** {selected.get('season', '—')}")
                st.markdown(f"**Drop Time:** {selected.get('recommended_drop_time', '—')}")
                desc = selected.get("product_description", "")
                if desc:
                    st.markdown(f"**Description excerpt:** {desc[:200]}...")
