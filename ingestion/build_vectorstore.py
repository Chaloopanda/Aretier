"""
ingestion/build_vectorstore.py
-------------------------------
One-time script to build all ChromaDB collections from local data files.

Run this BEFORE starting the Streamlit app for the first time:
  python ingestion/build_vectorstore.py

Collections built:
  1. brand_knowledge  ← brand_voice.md (chunked)
  2. drop_timing      ← historical_drops.csv (one doc per row)
  3. market_data      ← market comps from historical_drops.csv (derived)
  4. trend_archive    ← empty (populated live by web_scraper)
"""

import os
import sys
import csv
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from tools.rag_retriever import upsert, get_or_create_collection


def ingest_brand_voice():
    """Chunk and index brand_voice.md into brand_knowledge collection."""
    brand_voice_path = Path("knowledge_base/brand_voice.md")
    if not brand_voice_path.exists():
        print("❌ brand_voice.md not found. Skipping.")
        return

    text = brand_voice_path.read_text(encoding="utf-8")

    # Split on ## headings for semantic chunking
    chunks = []
    current_chunk = ""
    for line in text.split("\n"):
        if line.startswith("## ") and current_chunk.strip():
            chunks.append(current_chunk.strip())
            current_chunk = line
        else:
            current_chunk += "\n" + line
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    ids = [f"brand_voice_chunk_{i}" for i in range(len(chunks))]
    metas = [{"source": "brand_voice", "chunk_index": str(i)} for i in range(len(chunks))]

    upsert("brand_knowledge", ids=ids, documents=chunks, metadatas=metas)
    print(f"✅ brand_knowledge: {len(chunks)} chunks indexed from brand_voice.md")


def ingest_historical_drops():
    """
    Parse historical_drops.csv and index into:
      - drop_timing  : for the Drop Timer Agent
      - market_data  : for the Pricing Agent (resale comp derivation)
    """
    csv_path = Path("data/historical_drops.csv")
    if not csv_path.exists():
        print("❌ historical_drops.csv not found. Skipping.")
        return

    timing_ids, timing_docs, timing_metas = [], [], []
    market_ids, market_docs, market_metas = [], [], []

    # Hype tier → estimated resale premium mapping
    HYPE_PREMIUM = {"S": 0.85, "A": 0.45, "B": 0.25, "C": 0.10}
    # Silhouette → estimated retail price range
    SILHOUETTE_RETAIL = {
        "high_top": 210, "runner": 180, "trail": 165,
        "low_top": 160, "mid_top": 185, "court": 140,
        "slide": 85, "other": 150,
    }

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            drop_name = row.get("drop_name", "").strip()
            brand = row.get("brand", "").strip()
            day = row.get("day_of_week", "").strip()
            time_est = row.get("time_est", "").strip()
            sell_out = row.get("sell_out_minutes", "").strip()
            hype = row.get("hype_tier", "B").strip()
            silhouette = row.get("silhouette_type", "runner").strip()
            season = row.get("season", "").strip()
            date = row.get("drop_date", "").strip()

            row_id = f"drop_{i:04d}"

            # ── Timing document ──────────────────────────────────────────────
            sell_out_int = int(sell_out) if sell_out.isdigit() else 999
            performance = (
                "instant sell-out" if sell_out_int <= 5
                else f"sold out in {sell_out_int} minutes" if sell_out_int < 60
                else f"sold out in {sell_out_int//60} hour(s)"
            )
            timing_doc = (
                f"{drop_name} by {brand}. "
                f"Dropped on {day} at {time_est} EST. "
                f"Season: {season}. Silhouette: {silhouette}. "
                f"Hype tier: {hype}. Result: {performance}."
            )
            timing_ids.append(row_id + "_timing")
            timing_docs.append(timing_doc)
            timing_metas.append({
                "source": "historical_drops",
                "day_of_week": day,
                "time_est": time_est,
                "hype_tier": hype,
                "silhouette_type": silhouette,
                "season": season,
                "sell_out_minutes": str(sell_out_int),
            })

            # ── Market data document ─────────────────────────────────────────
            retail_estimate = SILHOUETTE_RETAIL.get(silhouette, 170)
            resale_premium = HYPE_PREMIUM.get(hype, 0.20)
            resale_estimate = retail_estimate * (1 + resale_premium)

            market_doc = (
                f"{drop_name} ({brand}). "
                f"Silhouette: {silhouette}. "
                f"Estimated retail: ${retail_estimate}. "
                f"Estimated resale: ${resale_estimate:.0f} ({resale_premium*100:.0f}% premium). "
                f"Hype tier: {hype}. Season: {season}."
            )
            market_ids.append(row_id + "_market")
            market_docs.append(market_doc)
            market_metas.append({
                "source": "historical_drops",
                "brand": brand,
                "silhouette_type": silhouette,
                "hype_tier": hype,
                "retail_price": str(retail_estimate),
                "resale_premium": str(resale_premium),
                "season": season,
            })

    upsert("drop_timing", ids=timing_ids, documents=timing_docs, metadatas=timing_metas)
    upsert("market_data", ids=market_ids, documents=market_docs, metadatas=market_metas)

    print(f"✅ drop_timing: {len(timing_ids)} rows indexed")
    print(f"✅ market_data: {len(market_ids)} rows indexed")


def ensure_trend_archive():
    """Create the trend_archive collection so it exists for web_scraper to write to."""
    get_or_create_collection("trend_archive")
    print("✅ trend_archive: collection ready (will be populated live)")


if __name__ == "__main__":
    print("\n[ASBOS] Building ChromaDB vector store...\n")
    ingest_brand_voice()
    ingest_historical_drops()
    ensure_trend_archive()
    print("\n[ASBOS] Vector store build complete. Run the Streamlit app next.\n")
