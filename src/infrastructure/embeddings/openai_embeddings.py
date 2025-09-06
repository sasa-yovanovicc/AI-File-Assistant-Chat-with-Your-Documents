"""OpenAI embeddings implementation."""

from typing import List
import openai
import numpy as np

from ...domain.repositories import EmbeddingRepository
from ...config import USE_OPENAI, OPENAI_API_KEY
from ...exceptions import EmbeddingError
from ...error_handler import handle_errors
from ...logging_config import get_logger

logger = get_logger(__name__)


class OpenAIEmbeddingClient(EmbeddingRepository):
    """OpenAI implementation of EmbeddingRepository."""
    
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.model = "text-embedding-3-small"
        self.dimension = 1536
        
        if self.api_key:
            openai.api_key = self.api_key
    
    def get_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.dimension
    
    @handle_errors(default_return=[], exception_type=EmbeddingError)
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not text.strip():
            raise EmbeddingError(message="Cannot embed empty text")
        
        if not USE_OPENAI or not self.api_key:
            raise EmbeddingError(message="OpenAI embeddings not available")
        
        logger.info("Generating OpenAI embedding for 1 text")
        
        response = openai.embeddings.create(
            model=self.model,
            input=[text]
        )
        
        return response.data[0].embedding
    
    @handle_errors(default_return=[], exception_type=EmbeddingError)
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        
        # Filter empty texts
        non_empty_texts = [t for t in texts if t.strip()]
        if not non_empty_texts:
            raise EmbeddingError(message="Cannot embed all empty texts")
        
        if not USE_OPENAI or not self.api_key:
            raise EmbeddingError(message="OpenAI embeddings not available")
        
        logger.info(f"Generating OpenAI embeddings for {len(non_empty_texts)} texts")
        
        response = openai.embeddings.create(
            model=self.model,
            input=non_empty_texts
        )
        
        return [data.embedding for data in response.data]
