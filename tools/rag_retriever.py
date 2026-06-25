"""
tools/rag_retriever.py
----------------------
ChromaDB wrapper for all RAG retrieval operations across agents.

Collections:
  - "brand_knowledge"  : brand_voice.md + drop_history.jsonl
  - "market_data"      : StockX-style resale comps indexed by keywords
  - "drop_timing"      : historical_drops.csv indexed by day/time/season
  - "trend_archive"    : Cached Hypebeast/Reddit trend summaries

Usage:
  from tools.rag_retriever import retrieve

  results = retrieve("gorpcore trail runner", collection="brand_knowledge", n=3)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Singleton client & embedding function ─────────────────────────────────────
_client: Optional[chromadb.PersistentClient] = None
_emb_fn: Optional[ONNXMiniLM_L6_V2] = None

VALID_COLLECTIONS = {
    "brand_knowledge",
    "market_data",
    "drop_timing",
    "trend_archive",
}


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=_PERSIST_DIR)
    return _client


def _get_emb_fn() -> ONNXMiniLM_L6_V2:
    global _emb_fn
    if _emb_fn is None:
        _emb_fn = ONNXMiniLM_L6_V2()
    return _emb_fn


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_or_create_collection(name: str) -> chromadb.Collection:
    """Get an existing collection or create it with the correct embedding fn."""
    if name not in VALID_COLLECTIONS:
        raise ValueError(f"Unknown collection '{name}'. Valid: {VALID_COLLECTIONS}")
    return _get_client().get_or_create_collection(
        name=name,
        embedding_function=_get_emb_fn(),
        metadata={"hnsw:space": "cosine"},
    )


def retrieve(
    query: str,
    collection: str,
    n: int = 4,
    where: Optional[dict] = None,
) -> list[dict]:
    """
    Query a ChromaDB collection and return top-n results as dicts.

    Returns:
        List of {"id": str, "document": str, "metadata": dict, "distance": float}
    """
    coll = get_or_create_collection(collection)

    # If collection is empty, return gracefully
    if coll.count() == 0:
        return []

    kwargs: dict = {"query_texts": [query], "n_results": min(n, coll.count())}
    if where:
        kwargs["where"] = where

    results = coll.query(**kwargs)

    output = []
    for i, doc in enumerate(results["documents"][0]):
        output.append(
            {
                "id": results["ids"][0][i],
                "document": doc,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else 0.0,
            }
        )
    return output


def upsert(
    collection: str,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict] | None = None,
) -> None:
    """Add or update documents in a collection."""
    coll = get_or_create_collection(collection)
    coll.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas or [{} for _ in ids],
    )


def format_results_as_context(results: list[dict]) -> str:
    """
    Format RAG results into a clean context string for LLM injection.
    Each result is numbered and separated clearly.
    """
    if not results:
        return "No relevant documents found in knowledge base."

    parts = []
    for i, r in enumerate(results, 1):
        source = r["metadata"].get("source", "unknown")
        parts.append(f"[{i}] (source: {source})\n{r['document']}")
    return "\n\n---\n\n".join(parts)
