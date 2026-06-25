"""
tools/llm_provider.py
---------------------
Unified LLM provider for ASBOS.

Auto-detects whether Ollama is running locally.
- If Ollama is UP -> use local model (no API calls, no rate limits, free forever)
- If Ollama is DOWN -> fall back to Gemini API

Set LLM_PROVIDER=gemini in .env to force Gemini.
Set LLM_PROVIDER=ollama in .env to force Ollama (fails if not running).
Leave unset for auto-detect (recommended).

Local model: gemma3:4b (fits in 4GB VRAM, handles JSON well)
Cloud model: gemini-2.5-flash
"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

_ollama_available: bool | None = None  # Cached after first check


def _check_ollama() -> bool:
    """Quick health-check: is Ollama running?"""
    global _ollama_available
    if _ollama_available is not None:
        return _ollama_available
    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
        _ollama_available = r.status_code == 200
    except Exception:
        _ollama_available = False
    return _ollama_available


def get_llm(temperature: float = 0.7):
    """
    Return the best available LLM — lazily instantiated on each call.

    Priority:
      1. LLM_PROVIDER=ollama  -> always use local Ollama
      2. LLM_PROVIDER=gemini  -> always use Gemini API
      3. (auto) Ollama up     -> use local Ollama
      4. (auto) Ollama down   -> use Gemini API

    Usage:
        from tools.llm_provider import get_llm
        llm = get_llm(temperature=0.7)
        response = llm.invoke([HumanMessage(content=prompt)])
    """
    provider = os.getenv("LLM_PROVIDER", "auto").lower()

    use_ollama = False
    if provider == "ollama":
        use_ollama = True
    elif provider == "gemini":
        use_ollama = False
    else:  # auto
        use_ollama = _check_ollama()

    if use_ollama:
        from langchain_ollama import ChatOllama
        print(f"[llm_provider] Using LOCAL Ollama -> {OLLAMA_MODEL}")
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
            num_predict=1024,
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No GEMINI_API_KEY found and Ollama is not running. "
                "Either start Ollama or add your API key to .env"
            )
        print(f"[llm_provider] Using Gemini API -> {GEMINI_MODEL}")
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=temperature,
        )


def get_provider_name() -> str:
    """Human-readable name of the active provider, for UI display."""
    provider = os.getenv("LLM_PROVIDER", "auto").lower()
    if provider == "ollama":
        return f"Local Ollama ({OLLAMA_MODEL})"
    elif provider == "gemini":
        return f"Gemini API ({GEMINI_MODEL})"
    else:
        if _check_ollama():
            return f"Local Ollama ({OLLAMA_MODEL})"
        return f"Gemini API ({GEMINI_MODEL})"

