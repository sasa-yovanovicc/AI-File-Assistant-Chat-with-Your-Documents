"""Test cases for Clean Architecture application layer."""

import pytest

from src.application.use_cases import ChatUseCase
from src.domain.entities import Query, ConfidenceLevel
from src.container import Container


class TestChatUseCase:
    """Test ChatUseCase application service."""
    
    def test_chat_use_case_execution(self, test_container, sample_query):
        """Test complete chat use case execution."""
        # Get use case from container
        chat_use_case = test_container.chat_use_case()
        
        # Add some test data
        from src.domain.entities import Chunk
        chunks = [
            Chunk("1", "doc1", "Artificial intelligence is a technology", 0, 0, 40, {"source": "ai.txt"}),
            Chunk("2", "doc1", "That simulates human intelligence", 41, 41, 71, {"source": "ai.txt"})
        ]
        embeddings = [[0.1] * 384, [0.2] * 384]
        test_container.vector_repository().save_vectors(chunks, embeddings)
        
        # Execute use case
        result = chat_use_case.execute(sample_query)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert "query_text" in result
        assert "answer" in result
        assert "confidence" in result
        assert "sources" in result
        
        assert result["query_text"] == sample_query.text
        assert isinstance(result["answer"], str)
        assert result["confidence"] in [level.value for level in ConfidenceLevel]
        assert isinstance(result["sources"], list)
    
    def test_chat_use_case_with_no_results(self, test_container):
        """Test chat use case when no relevant chunks found."""
        chat_use_case = test_container.chat_use_case()
        
        query = Query("Completely unrelated query")
        result = chat_use_case.execute(query)
        
        # Should still return a valid response
        assert isinstance(result, dict)
        assert "answer" in result
        assert result["confidence"] == ConfidenceLevel.NONE.value
    
    def test_chat_use_case_dependency_injection(self, test_container):
        """Test that use case properly uses injected dependencies."""
        chat_use_case = test_container.chat_use_case()
        
        # Verify dependencies are injected
        assert chat_use_case._retrieval_service is not None
        assert chat_use_case._answer_service is not None
        
        # Verify they're the same instances from container
        assert chat_use_case._retrieval_service == test_container.retrieval_service()
        assert chat_use_case._answer_service == test_container.answer_service()
