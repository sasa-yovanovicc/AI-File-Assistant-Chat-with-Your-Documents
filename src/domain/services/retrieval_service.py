"""Retrieval domain service."""

from typing import List
import numpy as np

from ..entities import Query, Chunk, SearchResult
from ..repositories import VectorRepository, EmbeddingRepository


class RetrievalService:
    """Domain service for document retrieval."""
    
    def __init__(
        self, 
        vector_repository: VectorRepository,
        embedding_repository: EmbeddingRepository
    ):
        self._vector_repo = vector_repository
        self._embedding_repo = embedding_repository
    
    def retrieve_chunks(self, query: Query) -> List[SearchResult]:
        """Retrieve relevant chunks for a query."""
        # Generate query embedding
        query_embedding = self._embedding_repo.embed_text(query.text)
        query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # Search vector store
        results = self._vector_repo.search(query_vector, k=query.k)
        
        # Filter by minimum score if specified
        if query.min_score is not None:
            results = [r for r in results if r.score >= query.min_score]
        
        return results
