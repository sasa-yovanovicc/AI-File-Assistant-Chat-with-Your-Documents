"""Retrieval domain service."""

import re
from typing import List
import numpy as np

from ..entities import Query, Chunk, SearchResult
from ..repositories import VectorRepository, EmbeddingRepository
from ...config import LEXICAL_RERANK_WEIGHT, MIN_KEYWORD_LENGTH


# Minimal Serbian/English stopword subset for keyword extraction
STOPWORDS = {
    'je','sam','si','su','smo','the','a','and','i','u','na','za','da','to','od','se','of','in','koji','koja','koje','sa','kao','ali','pa','ili'
}


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
        
        # Search vector store with extra results for reranking
        initial_k = max(query.k, 8)
        results = self._vector_repo.search(query_vector, k=initial_k)
        
        # Apply lexical reranking if we have results
        if results:
            results = self._lexical_rerank(query.text, results)
        
        # Take top k after reranking
        results = results[:query.k]
        
        # Filter by minimum score if specified
        if query.min_score is not None:
            results = [r for r in results if r.score >= query.min_score]
        
        return results
    
    def _extract_keywords(self, question: str) -> List[str]:
        """Extract keywords from question, removing stopwords."""
        tokens = re.findall(r'[A-Za-zÀ-ž0-9]+', question.lower())
        return [t for t in tokens if t not in STOPWORDS and len(t) > MIN_KEYWORD_LENGTH]
    
    def _lexical_rerank(self, question: str, results: List[SearchResult]) -> List[SearchResult]:
        """Re-rank results by combining semantic and lexical scores."""
        keywords = self._extract_keywords(question)
        if not keywords:
            return results
        
        # Calculate blended scores
        for result in results:
            text_lower = result.chunk.content.lower()
            overlap = sum(1 for k in keywords if k in text_lower)
            # Combine semantic score with lexical overlap
            result.blended_score = result.score + LEXICAL_RERANK_WEIGHT * overlap
        
        # Sort by blended score
        return sorted(results, key=lambda x: x.blended_score, reverse=True)
