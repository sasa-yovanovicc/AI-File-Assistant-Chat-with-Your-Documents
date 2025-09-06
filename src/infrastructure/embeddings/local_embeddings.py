"""Local embeddings implementation using sentence-transformers."""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from ...domain.repositories import EmbeddingRepository
from ...exceptions import EmbeddingError
from ...error_handler import handle_errors
from ...logging_config import get_logger

logger = get_logger(__name__)


class LocalEmbeddingClient(EmbeddingRepository):
    """Local sentence-transformers implementation of EmbeddingRepository."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        self._model = None
    
    def _get_model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"Loading local embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            # Update dimension based on actual model
            test_embedding = self._model.encode("test")
            self.dimension = len(test_embedding)
        return self._model
    
    def get_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.dimension
    
    @handle_errors(default_return=[], exception_type=EmbeddingError)
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not text.strip():
            raise EmbeddingError(message="Cannot embed empty text")
        
        logger.info("Generating local embedding for 1 text")
        
        model = self._get_model()
        embedding = model.encode(text)
        
        return embedding.tolist()
    
    @handle_errors(default_return=[], exception_type=EmbeddingError)
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        
        # Filter empty texts
        non_empty_texts = [t for t in texts if t.strip()]
        if not non_empty_texts:
            raise EmbeddingError(message="Cannot embed all empty texts")
        
        logger.info(f"Generating local embeddings for {len(non_empty_texts)} texts")
        
        model = self._get_model()
        embeddings = model.encode(non_empty_texts)
        
        return [embedding.tolist() for embedding in embeddings]
