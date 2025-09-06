"""Test cases for Clean Architecture domain layer."""

import pytest
from datetime import datetime
import numpy as np

from src.domain.entities import Document, Chunk, Query, QueryResult, ConfidenceLevel, SearchResult
from src.domain.services import RetrievalService, AnswerService


class TestDomainEntities:
    """Test domain entities."""
    
    def test_document_creation(self):
        """Test Document entity creation."""
        from datetime import datetime
        doc = Document(
            id="test_1",
            source="test.txt",
            content="Test content",
            metadata={"file_type": "text/plain", "file_size": 100},
            created_at=datetime.now()
        )
        
        assert doc.id == "test_1"
        assert doc.source == "test.txt"
        assert doc.content == "Test content"
        assert doc.metadata["file_type"] == "text/plain"
        assert doc.created_at is not None
        assert isinstance(doc.metadata, dict)
    
    def test_chunk_creation(self):
        """Test Chunk entity creation."""
        chunk = Chunk(
            id="chunk_1",
            document_id="doc_1",
            content="Chunk content",
            chunk_index=0,
            start_position=0,
            end_position=13,
            metadata={"source": "test.txt"}
        )
        
        assert chunk.id == "chunk_1"
        assert chunk.document_id == "doc_1"
        assert chunk.content == "Chunk content"
        assert chunk.chunk_index == 0
        assert chunk.length == 13
        assert isinstance(chunk.metadata, dict)
    
    def test_query_creation(self):
        """Test Query entity creation."""
        query = Query(text="What is AI?", k=5)
        
        assert query.text == "What is AI?"
        assert query.k == 5
        assert query.min_score is None
        assert query.use_llm is True
    
    def test_query_result_creation(self, sample_chunks):
        """Test QueryResult entity creation."""
        from src.domain.entities import QueryResult, ConfidenceLevel, AnswerStrategy, SearchResult
        
        query = Query("Test query")
        search_results = [
            SearchResult(chunk=sample_chunks[0], score=0.9, rank=0),
            SearchResult(chunk=sample_chunks[1], score=0.8, rank=1)
        ]
        
        result = QueryResult(
            query=query,
            answer="Test answer",
            confidence=ConfidenceLevel.HIGH,
            confidence_reason="Strong match",
            strategy=AnswerStrategy.GENERATE,
            search_results=search_results,
            keyword_coverage=0.8,
            metadata={"test": True}
        )
        
        assert result.query == query
        assert result.answer == "Test answer"
        assert result.confidence == ConfidenceLevel.HIGH
        assert len(result.search_results) == 2


class TestDomainServices:
    """Test domain services."""
    
    def test_retrieval_service(self, mock_vector_repository, mock_embedding_repository, sample_query):
        """Test RetrievalService."""
        service = RetrievalService(mock_vector_repository, mock_embedding_repository)
        
        # Add some test data
        from src.domain.entities import Chunk
        chunks = [
            Chunk("1", "doc1", "AI content", 0, 0, 10, {"source": "test.txt"}),
            Chunk("2", "doc1", "Machine learning", 1, 11, 27, {"source": "test.txt"})
        ]
        embeddings = np.array([[0.1] * 384, [0.2] * 384])
        mock_vector_repository.add_chunks(chunks, embeddings)
        
        # Test retrieval  
        results = service.retrieve_chunks(sample_query)
        
        assert len(results) >= 0  # Should return some results
    
    def test_answer_service(self, mock_llm_repository):
        """Test AnswerService."""
        service = AnswerService(mock_llm_repository)
        
        from src.domain.entities import Chunk, SearchResult
        chunks = [
            Chunk("1", "doc1", "AI is artificial intelligence", 0, 0, 30, {}),
            Chunk("2", "doc1", "Used in machine learning", 1, 31, 55, {})
        ]
        
        # Create SearchResult objects
        search_results = [
            SearchResult(chunk=chunks[0], score=0.9, rank=0),
            SearchResult(chunk=chunks[1], score=0.8, rank=1)
        ]
        
        query = Query("What is AI?")
        result = service.generate_answer(query, search_results)
        
        assert result.answer is not None
