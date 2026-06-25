"""
tools/clip_scorer.py
--------------------
Local CLIP evaluation tool for the Design Agent.

Computes cosine similarity between a text prompt and a generated image.
This gives us a quantitative "alignment score" — how well the generated
image actually matches the design brief.

Threshold: 0.28 (below this = regenerate)

Hardware note: clip-vit-base-patch32 uses ~300MB VRAM — safe on 4GB RTX 3050.
Auto-falls back to CPU if GPU is occupied.
"""

import base64
import os
from pathlib import Path
from io import BytesIO
from typing import Optional

_model = None
_processor = None
_device = None


def _load_model():
    """Lazy-load CLIP model on first call to avoid startup overhead."""
    global _model, _processor, _device

    if _model is not None:
        return _model, _processor, _device

    import torch
    from transformers import CLIPProcessor, CLIPModel

    # Try GPU first; fallback to CPU
    if torch.cuda.is_available():
        try:
            test = torch.zeros(1).cuda()
            _device = "cuda"
        except RuntimeError:
            _device = "cpu"
    else:
        _device = "cpu"

    print(f"[clip_scorer] Loading CLIP on {_device}...")
    _model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", use_safetensors=True).to(_device)
    _processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
    _model.eval()
    print("[clip_scorer] CLIP loaded.")

    return _model, _processor, _device


def score_image_text_alignment(
    image_path: Optional[str] = None,
    image_b64: Optional[str] = None,
    text: str = "",
) -> float:
    """
    Compute CLIP cosine similarity between an image and a text description.

    Args:
        image_path: Path to image file on disk (preferred).
        image_b64: Base64-encoded image string (fallback).
        text: The design brief or description to score against.

    Returns:
        Float in range [0.0, 1.0]. Higher = better alignment.
        Returns 0.0 on any failure (triggers regeneration).
    """
    if not image_path and not image_b64:
        print("[clip_scorer] No image provided — returning 0.0")
        return 0.0

    try:
        import torch
        from PIL import Image

        model, processor, device = _load_model()

        # Load image
        if image_path and Path(image_path).exists():
            image = Image.open(image_path).convert("RGB")
        elif image_b64:
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        else:
            print("[clip_scorer] Image file not found — returning 0.0")
            return 0.0

        # Truncate text to CLIP's max token length (77 tokens)
        text_input = text[:300]

        inputs = processor(
            text=[text_input],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Normalised embeddings
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            # Cosine similarity
            similarity = (image_embeds * text_embeds).sum().item()

        # CLIP similarity is in [-1, 1]; map to [0, 1]
        score = (similarity + 1) / 2
        return round(float(score), 4)

    except Exception as e:
        print(f"[clip_scorer] Error during scoring: {e}")
        return 0.0


def interpret_clip_score(score: float) -> str:
    """Human-readable interpretation of a CLIP score for the UI."""
    if score >= 0.40:
        return "Excellent alignment [OK]"
    elif score >= 0.28:
        return "Acceptable alignment [WARN]"
    else:
        return "Poor alignment [FAIL] — Regenerating"

