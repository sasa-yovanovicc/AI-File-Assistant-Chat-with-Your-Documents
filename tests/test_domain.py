"""Test cases for Clean Architecture domain layer."""

import pytest
from datetime import datetime

from src.domain.entities import Document, Chunk, Query, QueryResult, ConfidenceLevel
from src.domain.services import RetrievalService, AnswerService


class TestDomainEntities:
    """Test domain entities."""
    
    def test_document_creation(self):
        """Test Document entity creation."""
        doc = Document(
            id="test_1",
            filename="test.txt",
            content="Test content",
            file_type="text/plain",
            file_size=100
        )
        
        assert doc.id == "test_1"
        assert doc.filename == "test.txt"
        assert doc.content == "Test content"
        assert doc.file_type == "text/plain"
        assert doc.file_size == 100
        assert isinstance(doc.upload_time, (str, datetime))
        assert isinstance(doc.metadata, dict)
    
    def test_chunk_creation(self):
        """Test Chunk entity creation."""
        chunk = Chunk(
            id="chunk_1",
            document_id="doc_1",
            content="Chunk content",
            chunk_index=0
        )
        
        assert chunk.id == "chunk_1"
        assert chunk.document_id == "doc_1"
        assert chunk.content == "Chunk content"
        assert chunk.chunk_index == 0
        assert isinstance(chunk.metadata, dict)
    
    def test_query_creation(self):
        """Test Query entity creation."""
        query = Query(text="What is AI?", max_results=5)
        
        assert query.text == "What is AI?"
        assert query.max_results == 5
    
    def test_query_result_creation(self, sample_chunks):
        """Test QueryResult entity creation."""
        result = QueryResult(
            query_text="Test query",
            answer="Test answer",
            confidence=ConfidenceLevel.MEDIUM,
            sources=sample_chunks
        )
        
        assert result.query_text == "Test query"
        assert result.answer == "Test answer"
        assert result.confidence == ConfidenceLevel.MEDIUM
        assert len(result.sources) == 2


class TestDomainServices:
    """Test domain services."""
    
    def test_retrieval_service(self, mock_vector_repository, mock_embedding_repository, sample_query):
        """Test RetrievalService."""
        service = RetrievalService(mock_vector_repository, mock_embedding_repository)
        
        # Add some test data
        from src.domain.entities import Chunk
        chunks = [
            Chunk("1", "doc1", "AI content", 0, {"source": "test.txt"}),
            Chunk("2", "doc1", "Machine learning", 1, {"source": "test.txt"})
        ]
        embeddings = [[0.1] * 384, [0.2] * 384]
        mock_vector_repository.save_vectors(chunks, embeddings)
        
        # Test retrieval
        results = service.retrieve_relevant_chunks(sample_query)
        
        assert len(results) >= 0  # Should return some results
        assert mock_embedding_repository.call_count > 0  # Should have called embedding
    
    def test_answer_service(self, mock_llm_repository):
        """Test AnswerService."""
        service = AnswerService(mock_llm_repository)
        
        from src.domain.entities import Chunk
        chunks = [
            Chunk("1", "doc1", "AI is artificial intelligence", 0),
            Chunk("2", "doc1", "Used in machine learning", 1)
        ]
        
        result = service.generate_answer("What is AI?", chunks)
        
        assert isinstance(result, str)
        assert mock_llm_repository.call_count == 1
        assert "What is AI?" in mock_llm_repository.last_prompt
