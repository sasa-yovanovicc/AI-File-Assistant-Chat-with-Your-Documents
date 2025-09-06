"""Abstract repository interfaces."""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from ..entities import Document, Chunk, SearchResult


class VectorRepository(ABC):
    """Abstract interface for vector storage operations."""
    
    @abstractmethod
    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks with their embeddings to the vector store."""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[SearchResult]:
        """Search for similar chunks using vector similarity."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Get total number of chunks in the store."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Clear all data from the vector store."""
        pass
