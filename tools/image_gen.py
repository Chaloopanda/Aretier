"""
tools/image_gen.py
------------------
No-API-Key image generation using Pollinations.ai.
This provides high-quality sneaker generation without requiring a Gemini API key
or heavy local VRAM usage.
"""

import os
import base64
import urllib.parse
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageDraw
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

_ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "./artifacts"))
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_sneaker_image(
    design_brief: str,
    aesthetic_keywords: list[str],
    drop_id: str,
    attempt: int = 1,
) -> tuple[str, str]:
    """
    Generates an image via Pollinations.ai (free, no API key).
    
    Returns:
        Tuple of (image_path: str, image_b64: str)
    """
    drop_dir = _ARTIFACTS_DIR / drop_id
    drop_dir.mkdir(parents=True, exist_ok=True)
    image_path = drop_dir / "design.png"

    keyword_str = ", ".join(aesthetic_keywords) if aesthetic_keywords else ""
    
    # Truncate strings to avoid HTTP 414 URI Too Long
    safe_brief = design_brief[:400] if design_brief else "sleek design"
    safe_kws = keyword_str[:100]
    
    prompt = (
        f"Professional product photography of a sneaker. "
        f"{safe_brief} "
        f"Clean studio background, soft directional lighting, "
        f"high-detail shot showing sole, upper, and lacing. "
        f"Editorial style. {safe_kws}. "
        f"No text, no logos, no people."
    )
    
    encoded_prompt = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true&model=flux"
    
    try:
        with httpx.Client(timeout=45.0, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            
            with open(image_path, "wb") as f:
                f.write(resp.content)
    except Exception as e:
        print(f"[image_gen] Pollinations generation failed: {e}")
        raise e
            
    # Encode to base64 for Streamlit/Frontend
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    
    return str(image_path), b64


def get_placeholder_image() -> tuple[str, str]:
    """
    Returns a placeholder when image generation is unavailable.
    """
    try:
        img = Image.new("RGB", (512, 512), color=(20, 20, 20))
        draw = ImageDraw.Draw(img)
        draw.rectangle([196, 216, 316, 236], fill=(40, 40, 40))
        draw.text((172, 244), "Arétier", fill=(139, 115, 85))
        draw.text((160, 264), "[Image Pending]", fill=(80, 80, 80))
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        placeholder_path = str(_ARTIFACTS_DIR / "placeholder.png")
        img.save(placeholder_path)
        return placeholder_path, b64
    except Exception:
        # 1x1 transparent PNG fallback
        b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        return "", b64
