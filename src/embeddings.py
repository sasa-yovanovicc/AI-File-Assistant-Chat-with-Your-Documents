from __future__ import annotations
from typing import List
from functools import lru_cache
import numpy as np
from .config import EMBEDDING_MODEL, USE_OPENAI, OPENAI_API_KEY
from .exceptions import EmbeddingError, ConfigurationError
from .error_handler import handle_errors, log_error
from .logging_config import get_logger

logger = get_logger(__name__)

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
@handle_errors(exception_type=EmbeddingError, reraise=True)
def get_local_model():
    if SentenceTransformer is None:
        raise EmbeddingError(
            message="sentence-transformers not installed; cannot use local embeddings",
            details={'required_package': 'sentence-transformers'}
        )
    
    logger.info(f"Loading local embedding model: {EMBEDDING_MODEL}")
    try:
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        raise EmbeddingError(
            message=f"Failed to load embedding model: {EMBEDDING_MODEL}",
            details={'model_name': EMBEDDING_MODEL, 'error': str(e)}
        ) from e


@handle_errors(exception_type=EmbeddingError, reraise=True)
def _embed_openai(texts: List[str]) -> np.ndarray:
    if not openai_client:
        raise EmbeddingError(
            message="OpenAI client not initialized",
            details={'USE_OPENAI': USE_OPENAI, 'OPENAI_API_KEY_set': bool(OPENAI_API_KEY)}
        )
    
    vectors = []
    for i, text in enumerate(texts):
        try:
            resp = openai_client.embeddings.create(model="text-embedding-3-small", input=text[:8000])
            vectors.append(resp.data[0].embedding)
            logger.debug(f"Generated OpenAI embedding for text {i+1}/{len(texts)}")
        except Exception as e:
            raise EmbeddingError(
                message=f"OpenAI embedding failed for text {i+1}",
                details={'text_length': len(text), 'text_preview': text[:100], 'error': str(e)}
            ) from e
    
    return np.array(vectors, dtype="float32")


@handle_errors(default_return=np.array([]), exception_type=EmbeddingError)
def embed_texts(texts: List[str]) -> np.ndarray:
    """Return embeddings; gracefully fallback to local model if OpenAI not configured."""
    if not texts:
        logger.warning("Empty text list provided for embedding")
        return np.array([])
    
    logger.info(f"Generating embeddings for {len(texts)} texts")
    
    if USE_OPENAI and OPENAI_API_KEY and openai_client is not None:
        try:
            logger.info("Using OpenAI embeddings")
            return _embed_openai(texts)
        except EmbeddingError as e:
            logger.warning(f"OpenAI embedding failed: {e.message}. Falling back to local model.")
        except Exception as e:
            logger.warning(f"OpenAI embedding unexpected error: {str(e)}. Falling back to local model.")
    elif USE_OPENAI and not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set while USE_OPENAI=true; falling back to local model.")
    
    # Local fallback
    logger.info("Using local embedding model")
    try:
        model = get_local_model()
        v = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        logger.info(f"Successfully generated {len(v)} local embeddings")
        return v.astype("float32")
    except Exception as e:
        raise EmbeddingError(
            message="Both OpenAI and local embedding generation failed",
            details={
                'text_count': len(texts),
                'local_error': str(e),
                'USE_OPENAI': USE_OPENAI,
                'OPENAI_API_KEY_set': bool(OPENAI_API_KEY)
            }
        ) from e


@handle_errors(exception_type=EmbeddingError, reraise=True)
def embed_query(q: str) -> np.ndarray:
    if not q.strip():
        raise EmbeddingError(
            message="Empty query provided for embedding",
            details={'query': q}
        )
    
    logger.debug(f"Generating embedding for query: {q[:50]}...")
    return embed_texts([q])[0]
