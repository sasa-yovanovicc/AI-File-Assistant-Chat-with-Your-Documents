"""Missing repository interface for embeddings."""

from abc import ABC, abstractmethod
from typing import List

class EmbeddingRepository(ABC):
    """Repository interface for text embeddings."""
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this repository."""
        pass
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
