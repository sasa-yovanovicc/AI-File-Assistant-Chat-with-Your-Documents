"""Abstract LLM repository interface."""

from abc import ABC, abstractmethod
from typing import List


class LLMRepository(ABC):
    """Abstract interface for LLM operations."""
    
    @abstractmethod
    def generate_answer(self, prompt: str, **kwargs) -> str:
        """Generate an answer using the LLM."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM is available."""
        pass


class EmbeddingRepository(ABC):
    """Abstract interface for embedding operations."""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings."""
        pass
