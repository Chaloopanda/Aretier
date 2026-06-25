"""
agents/design_agent.py
----------------------
Design Agent — generates the sneaker concept and image.

Responsibilities:
  1. Use Gemini to convert aesthetic keywords -> detailed design brief
  2. Generate sneaker image via Gemini Imagen (tools/image_gen.py)
  3. Score alignment with CLIP (tools/clip_scorer.py)
  4. Retry image generation if CLIP score < threshold (max 2 retries)
  5. On revision: incorporate Critic Agent's specific instruction
"""

import os
import json
from langchain_core.messages import HumanMessage, AIMessage
from tenacity import retry, stop_after_attempt, wait_exponential
from tools.llm_provider import get_llm

from graph.state import DropState
from tools.image_gen import generate_sneaker_image, get_placeholder_image
from tools.clip_scorer import score_image_text_alignment, interpret_clip_score

_llm = get_llm(temperature=0.8)

CLIP_THRESHOLD = float(os.getenv("CLIP_SCORE_THRESHOLD", "0.28"))
MAX_IMAGE_RETRIES = 2

DESIGN_SYSTEM_PROMPT = """You are the Creative Director at Arétier — a brand rooted in Arete (earned 
excellence), craftsmanship, and athletic resilience.

Your task is to write a detailed, technically precise design brief for a new sneaker silhouette.

The brief must include:
1. Silhouette type (trail runner / low trainer / field boot / high-top)
2. Upper material (specific: ripstop, Cordura, full-grain suede, etc.)
3. Colourway (specific: muted sage green, raw umber, slate grey — not "earthy")
4. Sole design (lightweight trail lug / clean cupsole / technical midsole)
5. Key design detail that makes this Arétier (not a Nike clone)
6. One sentence on the terrain/context this silhouette answers

Style rules:
- Anti-maximalist. Anti-chunky. Anti-neon.
- "Looks strong because it IS strong"
- Restraint as a design philosophy

Length: 120-150 words. Dense, specific, no fluff.
"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def run_design_agent(state: DropState) -> dict:
    """
    Execute the Design Agent and return state updates.
    Handles the image generation + CLIP scoring loop internally.
    """
    keywords = state.get("aesthetic_keywords") or []
    revision_note = state.get("design_revision_note")
    iteration = state.get("iteration_count", 0)
    drop_id = state["drop_id"]

    print(f"[design_agent] Starting design generation (iteration {iteration})...")

    # 1. Generate design brief
    revision_instruction = ""
    if revision_note:
        revision_instruction = f"\n\nCRITIC REVISION INSTRUCTION: {revision_note}\nAddress this issue specifically."

    brief_prompt = f"""
{DESIGN_SYSTEM_PROMPT}

AESTHETIC KEYWORDS TO INCORPORATE: {", ".join(keywords)}
SEASON: {state.get('season', 'Current Season')}
{revision_instruction}

Write the design brief now.
CRITICAL: Output ONLY the 1-2 paragraph visual description. DO NOT include any conversational text, pleasantries, or formatting (like 'Here is the revision:').
"""

    brief_response = _llm.invoke([HumanMessage(content=brief_prompt)])
    design_brief = brief_response.content.strip()

    print(f"[design_agent] Design brief generated ({len(design_brief.split())} words).")

    # 2. Generate image with CLIP scoring loop
    image_path = None
    image_b64 = None
    clip_score = 0.0
    generation_attempt = 1

    while generation_attempt <= MAX_IMAGE_RETRIES + 1:
        try:
            print(f"[design_agent] Generating image (attempt {generation_attempt})...")
            image_path, image_b64 = generate_sneaker_image(
                design_brief=design_brief,
                aesthetic_keywords=keywords,
                drop_id=drop_id,
                attempt=generation_attempt,
            )

            clip_score = score_image_text_alignment(
                image_path=image_path,
                text=design_brief,
            )

            print(f"[design_agent] CLIP score: {clip_score:.3f} ({interpret_clip_score(clip_score)})")

            if clip_score >= CLIP_THRESHOLD:
                break  # Acceptable — stop retrying

            if generation_attempt <= MAX_IMAGE_RETRIES:
                print(f"[design_agent] Score below threshold, refining prompt...")
                # Refine the brief to be more concrete for Imagen
                refinement = _llm.invoke([HumanMessage(content=
                    f"The following sneaker design brief produced a poor image. "
                    f"Rewrite it with MORE CONCRETE visual details — specific colours, "
                    f"textures, and structural features that an image model can render.\n"
                    f"CRITICAL: Output ONLY the revised visual description. DO NOT include any conversational text, pleasantries, or formatting (like 'Here is the revision:').\n\n"
                    f"{design_brief}"
                )])
                design_brief = refinement.content.strip()

            generation_attempt += 1

        except Exception as e:
            print(f"[design_agent] Image generation failed: {e}")
            if generation_attempt > MAX_IMAGE_RETRIES:
                print("[design_agent] All retries exhausted — using placeholder.")
                image_path, image_b64 = get_placeholder_image()
                clip_score = 0.0
            generation_attempt += 1

    return {
        "design_brief": design_brief,
        "design_image_path": image_path,
        "design_image_b64": image_b64,
        "clip_score": clip_score,
        "messages": [
            AIMessage(
                content=(
                    f"[Design Agent] Brief written. Image generated. "
                    f"CLIP score: {clip_score:.3f} — {interpret_clip_score(clip_score)}"
                ),
                name="design_agent",
            )
        ],
    }

