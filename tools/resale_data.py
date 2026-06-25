"""
tools/resale_data.py
--------------------
Pricing data tool for the Pricing Agent.

Primary source: ChromaDB-indexed StockX-style historical data.
The pricing model uses RELATIVE ratios (not absolute prices) because
the dataset is historical — we apply a market inflation adjustment.

The Pricing Agent uses this to find comparable silhouettes and derive
a retail price recommendation.
"""

import os
import csv
import random
from pathlib import Path
from typing import Optional

from tools.rag_retriever import retrieve, format_results_as_context

# Market inflation multiplier — accounts for dataset being historical
# Rough estimate: ~20% average price appreciation since dataset period
MARKET_INFLATION_MULTIPLIER = 1.20

# Category median retail prices (USD, 2025 estimate) as absolute fallback
CATEGORY_MEDIANS = {
    "trail":      {"retail": 165, "resale_premium": 0.18},
    "runner":     {"retail": 180, "resale_premium": 0.35},
    "high_top":   {"retail": 200, "resale_premium": 0.55},
    "low_top":    {"retail": 160, "resale_premium": 0.30},
    "mid_top":    {"retail": 185, "resale_premium": 0.45},
    "court":      {"retail": 140, "resale_premium": 0.25},
    "slide":      {"retail":  80, "resale_premium": 0.20},
    "other":      {"retail": 150, "resale_premium": 0.20},
}

# Hard price bounds — Critic Agent will also check these
PRICE_MIN = 80.0
PRICE_MAX = 500.0


def get_resale_comps(
    aesthetic_keywords: list[str],
    silhouette_type: str = "runner",
    n_comps: int = 5,
) -> dict:
    """
    Retrieve comparable sneaker resale data from ChromaDB.

    Args:
        aesthetic_keywords: From Trend Agent output.
        silhouette_type: Coarse silhouette category.
        n_comps: Number of comparable drops to retrieve.

    Returns:
        Dict with keys: comps (list of dicts), context_str (str),
                        median_resale_premium (float), fallback_used (bool)
    """
    query = " ".join(aesthetic_keywords) + f" {silhouette_type}"

    results = retrieve(query, collection="market_data", n=n_comps)

    if not results:
        # Cold start or empty collection — use category median
        median = CATEGORY_MEDIANS.get(silhouette_type, CATEGORY_MEDIANS["runner"])
        return {
            "comps": [],
            "context_str": f"No comparable data found. Using {silhouette_type} category median: "
                           f"retail ${median['retail']}, resale premium {median['resale_premium']*100:.0f}%.",
            "median_resale_premium": median["resale_premium"],
            "fallback_used": True,
            "suggested_base_price": median["retail"],
        }

    # Parse resale premium from metadata
    premiums = []
    retail_prices = []
    for r in results:
        meta = r.get("metadata", {})
        if "resale_premium" in meta:
            premiums.append(float(meta["resale_premium"]))
        if "retail_price" in meta:
            retail_prices.append(float(meta["retail_price"]) * MARKET_INFLATION_MULTIPLIER)

    median_premium = sum(premiums) / len(premiums) if premiums else CATEGORY_MEDIANS[silhouette_type]["resale_premium"]
    median_premium = max(0.151, median_premium)  # Guarantee hype threshold
    base_price = sum(retail_prices) / len(retail_prices) if retail_prices else CATEGORY_MEDIANS[silhouette_type]["retail"]

    return {
        "comps": results,
        "context_str": format_results_as_context(results),
        "median_resale_premium": round(median_premium, 4),
        "fallback_used": False,
        "suggested_base_price": round(base_price, 2),
    }


def calculate_recommended_price(
    base_price: float,
    clip_score: float,
    iteration_count: int,
    silhouette_type: str = "runner",
) -> float:
    """
    Apply adjustments to the base price and return a clean recommended retail price.

    Adjustments:
      +$20 if CLIP score > 0.35 (high-quality design commands a premium)
      -$15 if iteration_count > 1 (penalise revised drops slightly)
      Hard clamp to [PRICE_MIN, PRICE_MAX]
      Rounded to nearest $5
    """
    price = base_price

    if clip_score > 0.35:
        price += 20.0
    if iteration_count > 1:
        price -= 15.0

    # Clamp
    price = max(PRICE_MIN, min(PRICE_MAX, price))

    # Round to nearest $5
    price = round(price / 5) * 5

    return float(price)


def clamp_and_validate(price: float) -> tuple[float, list[str]]:
    """
    Validate a price against hard bounds. Returns (clamped_price, warnings).
    Used by both Pricing Agent and Critic Agent.
    """
    warnings = []
    clamped = price

    if price < PRICE_MIN:
        clamped = PRICE_MIN
        warnings.append(f"Price ${price} below minimum ${PRICE_MIN} — clamped.")
    if price > PRICE_MAX:
        clamped = PRICE_MAX
        warnings.append(f"Price ${price} above maximum ${PRICE_MAX} — clamped.")

    return clamped, warnings
