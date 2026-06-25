"""
evaluation/critic_rubric.py
----------------------------
Deterministic scoring rules for the Critic Agent.

These checks run BEFORE the LLM critique step.
Deterministic checks are cheap, fast, and not hallucinable.
The LLM critique layer adds qualitative depth on top.

Usage:
    from evaluation.critic_rubric import run_deterministic_checks

    checks = run_deterministic_checks(state)
    # checks = {
    #   "design": {"passed": bool, "score": float, "issues": list[str]},
    #   "pricing": {...},
    #   "timing": {...},
    #   "copy": {...},
    # }
"""

import re
from datetime import datetime, timezone
from typing import Optional

from graph.state import DropState

# ── Thresholds (sourced from .env or defaults) ────────────────────────────────
import os
from dotenv import load_dotenv
load_dotenv()

CLIP_THRESHOLD = float(os.getenv("CLIP_SCORE_THRESHOLD", "0.28"))
PRICE_MIN = 80.0
PRICE_MAX = 500.0
RESALE_PREMIUM_MIN = 0.15
DROP_DAYS_MIN = 7
DESCRIPTION_WORDS_MIN = 100
DESCRIPTION_WORDS_MAX = 200
TWEET_COUNT_REQUIRED = 3

BANNED_OPENERS = [
    "introducing",
    "excited to",
    "we are pleased",
    "we are thrilled",
    "pleased to announce",
    "happy to announce",
]

BANNED_WORDS_IN_DESCRIPTION = [
    "luxury",
    "premium",
    "game-changer",
    "next-level",
    "elevate",
    "for the culture",
    "limited edition",
]


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_design(state: DropState) -> dict:
    issues = []
    score = 1.0

    clip = state.get("clip_score")
    if clip is None:
        issues.append("CLIP score missing — Design Agent may not have completed.")
        score -= 0.5
    elif clip < CLIP_THRESHOLD:
        issues.append(
            f"CLIP score {clip:.3f} is below threshold {CLIP_THRESHOLD}. "
            "Image does not sufficiently match design brief."
        )
        score -= 0.4

    brief = state.get("design_brief", "") or ""
    keywords = state.get("aesthetic_keywords") or []
    if keywords:
        covered = 0
        for kw in keywords:
            words = [w.lower() for w in kw.split() if len(w) > 3]
            if not words:
                words = [kw.lower()]
            if any(w in brief.lower() for w in words) or kw.lower() in brief.lower():
                covered += 1

        coverage = covered / len(keywords)
        if coverage < 0.5:
            issues.append(
                f"Design brief only covers {coverage*100:.0f}% of trend keywords. "
                f"Missing concepts from: {[kw for kw in keywords if not any(w.lower() in brief.lower() for w in kw.split() if len(w) > 3) and kw.lower() not in brief.lower()]}"
            )
            score -= 0.2

    if not state.get("design_image_path"):
        issues.append("No design image generated.")
        score -= 0.3

    return {
        "passed": len(issues) == 0,
        "score": round(max(0.0, score), 3),
        "issues": issues,
    }


def _check_pricing(state: DropState) -> dict:
    issues = []
    score = 1.0

    price = state.get("suggested_price")
    if price is None:
        issues.append("Suggested price is missing.")
        score -= 0.5
    else:
        if price < PRICE_MIN:
            issues.append(f"Price ${price} is below minimum ${PRICE_MIN}.")
            score -= 0.3
        if price > PRICE_MAX:
            issues.append(f"Price ${price} exceeds maximum ${PRICE_MAX}.")
            score -= 0.3

    premium = state.get("resale_premium_estimate")
    if premium is None:
        issues.append("Resale premium estimate is missing.")
        score -= 0.2
    elif premium < RESALE_PREMIUM_MIN:
        issues.append(
            f"Resale premium estimate {premium*100:.1f}% is below minimum {RESALE_PREMIUM_MIN*100:.0f}%. "
            "Drop may not generate sufficient hype."
        )
        score -= 0.2

    if not state.get("price_rationale"):
        issues.append("Price rationale missing — no comp citations provided.")
        score -= 0.1

    return {
        "passed": len(issues) == 0,
        "score": round(max(0.0, score), 3),
        "issues": issues,
    }


def _check_timing(state: DropState) -> dict:
    issues = []
    score = 1.0

    drop_time_str = state.get("recommended_drop_time")
    if not drop_time_str:
        issues.append("Drop time is missing.")
        score -= 0.5
    else:
        try:
            drop_dt = datetime.fromisoformat(drop_time_str)
            # Make timezone-aware if naive
            if drop_dt.tzinfo is None:
                drop_dt = drop_dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            days_until = (drop_dt - now).days

            if days_until < DROP_DAYS_MIN:
                issues.append(
                    f"Drop is only {days_until} day(s) away — minimum is {DROP_DAYS_MIN}. "
                    "Insufficient lead time for a proper marketing build-up."
                )
                score -= 0.3

            # Check if it's a good drop day (Thu/Fri/Sat are premium)
            day_name = drop_dt.strftime("%A")
            if day_name in ["Monday", "Tuesday", "Wednesday"]:
                issues.append(
                    f"Drop scheduled on {day_name} — Thu/Fri/Sat historically outperform by 34%."
                )
                score -= 0.1

        except (ValueError, TypeError) as e:
            issues.append(f"Could not parse drop time '{drop_time_str}': {e}")
            score -= 0.4

    if not state.get("timing_rationale"):
        issues.append("Timing rationale missing.")
        score -= 0.1

    return {
        "passed": len(issues) == 0,
        "score": round(max(0.0, score), 3),
        "issues": issues,
    }


def _check_copy(state: DropState) -> dict:
    issues = []
    score = 1.0

    description = state.get("product_description", "") or ""

    # Word count
    word_count = len(description.split())
    if word_count < DESCRIPTION_WORDS_MIN:
        issues.append(
            f"Description is {word_count} words — minimum is {DESCRIPTION_WORDS_MIN}."
        )
        score -= 0.25
    elif word_count > DESCRIPTION_WORDS_MAX:
        issues.append(
            f"Description is {word_count} words — maximum is {DESCRIPTION_WORDS_MAX}."
        )
        score -= 0.1

    # Banned openers
    desc_lower = description.lower().strip()
    for opener in BANNED_OPENERS:
        if desc_lower.startswith(opener):
            issues.append(
                f"Description starts with banned opener: '{opener}'. "
                "Rewrite the opening line."
            )
            score -= 0.3
            break

    # Banned vocabulary in description
    for banned in BANNED_WORDS_IN_DESCRIPTION:
        if banned in desc_lower:
            issues.append(f"Description contains banned word: '{banned}'.")
            score -= 0.1

    # Price mentions (not allowed in copy)
    if re.search(r"\$\d+", description):
        issues.append("Product description contains a price mention — remove it.")
        score -= 0.2

    # Tweet drafts
    tweets = state.get("tweet_drafts") or []
    if len(tweets) < TWEET_COUNT_REQUIRED:
        issues.append(
            f"Only {len(tweets)} tweet draft(s) provided — need {TWEET_COUNT_REQUIRED}."
        )
        score -= 0.2

    # Check tweet openers
    for i, tweet in enumerate(tweets):
        tweet_lower = tweet.lower().strip()
        for opener in BANNED_OPENERS:
            if tweet_lower.startswith(opener):
                issues.append(f"Tweet {i+1} starts with banned opener: '{opener}'.")
                score -= 0.1
                break

    # All tweets starting the same way is a sign of low creativity
    if len(tweets) >= 2:
        first_words = [t.split()[0].lower() if t.split() else "" for t in tweets]
        if len(set(first_words)) == 1:
            issues.append(
                "All tweet drafts start with the same word — lacks variety."
            )
            score -= 0.1

    return {
        "passed": len(issues) == 0,
        "score": round(max(0.0, score), 3),
        "issues": issues,
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def run_deterministic_checks(state: DropState) -> dict:
    """
    Run all deterministic checks and return a structured report.

    Returns:
        {
            "design": {"passed": bool, "score": float, "issues": list[str]},
            "pricing": {...},
            "timing": {...},
            "copy": {...},
            "overall_score": float,  # Weighted average
            "any_failed": bool,
        }
    """
    design_result = _check_design(state)
    pricing_result = _check_pricing(state)
    timing_result = _check_timing(state)
    copy_result = _check_copy(state)

    # Weighted average: design and copy are weighted higher (brand impact)
    weights = {"design": 0.30, "pricing": 0.25, "timing": 0.20, "copy": 0.25}
    overall = (
        design_result["score"] * weights["design"]
        + pricing_result["score"] * weights["pricing"]
        + timing_result["score"] * weights["timing"]
        + copy_result["score"] * weights["copy"]
    )

    return {
        "design": design_result,
        "pricing": pricing_result,
        "timing": timing_result,
        "copy": copy_result,
        "overall_score": round(overall, 3),
        "any_failed": not all(
            [design_result["passed"], pricing_result["passed"],
             timing_result["passed"], copy_result["passed"]]
        ),
    }


def build_revision_requests(checks: dict) -> dict:
    """
    Convert check results into structured revision instructions for each agent.
    Returns None for agents that don't need revision.
    """
    def _format(check_result: dict, agent_name: str) -> Optional[str]:
        if check_result["passed"]:
            return None
        issues_str = "\n".join(f"  - {i}" for i in check_result["issues"])
        return f"[{agent_name}] Revision required due to:\n{issues_str}"

    return {
        "design_agent": _format(checks["design"], "Design Agent"),
        "pricing_agent": _format(checks["pricing"], "Pricing Agent"),
        "drop_timer_agent": _format(checks["timing"], "Drop Timer Agent"),
        "copywriter_agent": _format(checks["copy"], "Copywriter Agent"),
    }
