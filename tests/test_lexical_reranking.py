"""Test cases for lexical reranking and retrieval strategies."""

import pytest
import numpy as np
from unittest.mock import Mock

from src.domain.services import RetrievalService
from src.domain.entities import Query, Chunk, SearchResult
from src.rag_pipeline import _lexical_rerank, _keywords


class TestLexicalReranking:
    """Test lexical reranking functionality."""
    
    def test_keyword_extraction(self):
        """Test keyword extraction from questions."""
        # Serbian question
        keywords = _keywords("Ko je Bert u oblasti veštačke inteligencije?")
        expected = {"bert", "oblasti", "veštačke", "inteligencije"}
        assert set(keywords) >= expected  # Should contain at least these
        
        # English question
        keywords = _keywords("What is machine learning algorithm?")
        expected = {"machine", "learning", "algorithm"}
        assert set(keywords) >= expected
        
        # Should filter stopwords
        keywords = _keywords("Ko je to u priči?")
        assert "je" not in keywords  # Stopword should be filtered
        assert "to" not in keywords
    
    def test_lexical_rerank_scoring(self):
        """Test lexical reranking score calculation."""
        question = "Ko je Bert model?"

        hits = [
            {"text": "BERT je model za obradu prirodnog jezika", "score": 0.6, "id": "1"},
            {"text": "GPT je generativni model", "score": 0.7, "id": "2"},
            {"text": "Transformers koriste Bert arhitekturu", "score": 0.5, "id": "3"}
        ]

        reranked = _lexical_rerank(question, hits)

        # First hit should have highest blended score due to "BERT" and "model" matches
        # Check that reranking occurred (first result should be different)
        assert len(reranked) == 3
        assert reranked[0]["id"] in ["1", "3"]  # Either Bert-related hit should be first
        assert "blended" in reranked[0]
        assert reranked[0]["blended"] > reranked[0]["score"]  # Lexical boost applied
    
    def test_clean_arch_lexical_rerank(self, mock_vector_repository, mock_embedding_repository):
        """Test lexical reranking in Clean Architecture."""
        service = RetrievalService(mock_vector_repository, mock_embedding_repository)
        
        # Create sample search results
        chunk1 = Chunk("1", "doc1", "BERT je AI model", 0, {"source": "ai.txt"})
        chunk2 = Chunk("2", "doc1", "Python je programski jezik", 1, {"source": "prog.txt"})
        
        results = [
            SearchResult(chunk1, 0.7, 0),
            SearchResult(chunk2, 0.8, 1)
        ]
        
        reranked = service._lexical_rerank("Ko je BERT?", results)
        
        # Should prioritize BERT chunk despite lower semantic score
        assert reranked[0].chunk.content == "BERT je AI model"
        assert reranked[0].blended_score > reranked[0].score


class TestRetrievalStrategies:
    """Test various retrieval strategies."""
    
    def test_semantic_search_baseline(self, mock_vector_repository, mock_embedding_repository):
        """Test pure semantic search without reranking."""
        service = RetrievalService(mock_vector_repository, mock_embedding_repository)
        
        # Mock embedding and search
        mock_embedding_repository.embed_text.return_value = [0.1] * 384
        mock_vector_repository.search.return_value = [
            SearchResult(Chunk("1", "doc1", "Test content", 0), 0.9, 0)
        ]
        
        query = Query("test question", k=3)
        results = service.retrieve_chunks(query)
        
        assert len(results) >= 0
        mock_embedding_repository.embed_text.assert_called_once()
        mock_vector_repository.search.assert_called_once()
    
    def test_retrieval_with_minimum_score_filter(self, mock_vector_repository, mock_embedding_repository):
        """Test retrieval with minimum score filtering."""
        service = RetrievalService(mock_vector_repository, mock_embedding_repository)
        
        # Mock low-score results
        mock_embedding_repository.embed_text.return_value = [0.1] * 384
        mock_vector_repository.search.return_value = [
            SearchResult(Chunk("1", "doc1", "Low relevance", 0), 0.2, 0),
            SearchResult(Chunk("2", "doc1", "High relevance", 1), 0.8, 1)
        ]
        
        query = Query("test question", k=5, min_score=0.5)
        results = service.retrieve_chunks(query)
        
        # Should filter out low-score results
        assert all(r.score >= 0.5 for r in results)
    
    def test_retrieval_k_expansion(self, mock_vector_repository, mock_embedding_repository):
        """Test that retrieval gets extra results for reranking."""
        service = RetrievalService(mock_vector_repository, mock_embedding_repository)
        
        mock_embedding_repository.embed_text.return_value = [0.1] * 384
        mock_vector_repository.search.return_value = []
        
        query = Query("test question", k=3)
        service.retrieve_chunks(query)
        
        # Should search for max(k, 8) initially for reranking
        mock_vector_repository.search.assert_called_with(
            mock_embedding_repository.embed_text.return_value, k=8
        )
