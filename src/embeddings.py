from __future__ import annotations
from typing import List
from functools import lru_cache
import numpy as np
from .config import EMBEDDING_MODEL, USE_OPENAI, OPENAI_API_KEY

try:  # always try local model availability
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

if USE_OPENAI:
    try:  # import openai client lazily
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    except Exception:  # pragma: no cover
        openai_client = None
else:
    openai_client = None


@lru_cache(maxsize=1)
def get_local_model():
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed; cannot use local embeddings")
    return SentenceTransformer(EMBEDDING_MODEL)


def _embed_openai(texts: List[str]) -> np.ndarray:
    vectors = []
    for t in texts:
        resp = openai_client.embeddings.create(model="text-embedding-3-small", input=t[:8000])
        vectors.append(resp.data[0].embedding)
    return np.array(vectors, dtype="float32")


def embed_texts(texts: List[str]) -> np.ndarray:
    """Return embeddings; gracefully fallback to local model if OpenAI not configured."""
    if USE_OPENAI and OPENAI_API_KEY and openai_client is not None:
        try:
            return _embed_openai(texts)
        except Exception as e:  # pragma: no cover
            print(f"[yellow]OpenAI embedding failed ({e}); falling back to local model.")
    elif USE_OPENAI and not OPENAI_API_KEY:
        print("[yellow]OPENAI_API_KEY not set while USE_OPENAI=true; falling back to local model.")
    # local fallback
    model = get_local_model()
    v = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return v.astype("float32")


def embed_query(q: str) -> np.ndarray:
    return embed_texts([q])[0]
