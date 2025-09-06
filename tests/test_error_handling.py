"""Test cases for error handling and exception scenarios."""

"""Test error handling functionality."""

import pytest
from unittest.mock import Mock, patch

from src.error_handler import handle_errors, log_error, safe_execute
from src.exceptions import (
    AIFileAssistantError, 
    DocumentProcessingError, 
    LLMError, 
    VectorStoreError,
    EmbeddingError
)
from src.domain.services import AnswerService, RetrievalService


class TestCustomExceptions:
    """Test custom exception hierarchy."""
    
    def test_base_exception_structure(self):
        """Test AIFileAssistantError base exception."""
        error = AIFileAssistantError(
            message="Test error",
            error_code="TEST_ERROR",
            details={"key": "value"}
        )
        
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.details == {"key": "value"}
    
    def test_specific_exceptions(self):
        """Test specific exception types."""
        # EmbeddingError
        emb_error = EmbeddingError("Embedding failed")
        assert isinstance(emb_error, AIFileAssistantError)
        assert emb_error.error_code == "EmbeddingError"
        
        # VectorStoreError
        vec_error = VectorStoreError("Vector store failed")
        assert isinstance(vec_error, AIFileAssistantError)
        
        # LLMError
        llm_error = LLMError("LLM failed")
        assert isinstance(llm_error, AIFileAssistantError)
    
    def test_exception_with_details(self):
        """Test exceptions with detailed error information."""
        error = VectorStoreError(
            message="FAISS index corrupted",
            error_code="FAISS_CORRUPTED",
            details={
                "index_path": "/path/to/index",
                "dimension_mismatch": True,
                "expected_dim": 384,
                "actual_dim": 512
            }
        )
        
        assert "FAISS index corrupted" in str(error)
        assert error.details["dimension_mismatch"] == True
        assert error.details["expected_dim"] == 384


class TestErrorHandlerDecorator:
    """Test error handling decorator functionality."""
    
    def test_successful_function_execution(self):
        """Test decorator doesn't interfere with successful execution."""
        @handle_errors(default_return="default", exception_type=Exception)
        def successful_function(x, y):
            return x + y
        
        result = successful_function(2, 3)
        assert result == 5
    
    def test_exception_handling_with_default(self):
        """Test decorator handles exceptions and returns default."""
        @handle_errors(default_return="error_default", exception_type=ValueError)
        def failing_function():
            raise ValueError("Something went wrong")
        
        result = failing_function()
        assert result == "error_default"
    
    def test_exception_type_filtering(self):
        """Test decorator only catches specified exception types."""
        @handle_errors(default_return="handled", exception_type=DocumentProcessingError, reraise=True)
        def selective_handler():
            raise TypeError("Wrong type")  # Different exception type

        # Should not catch TypeError, should re-raise as DocumentProcessingError 
        with pytest.raises(DocumentProcessingError):  # Will be wrapped in DocumentProcessingError
            selective_handler()

    def test_custom_exception_transformation(self):
        """Test decorator transforms exceptions to custom types."""
        @handle_errors(default_return=[], exception_type=VectorStoreError)
        def vector_operation():
            raise Exception("Generic error")
        
        # Should transform to VectorStoreError and return default
        result = vector_operation()
        assert result == []


class TestServiceErrorHandling:
    """Test error handling in domain services."""
    
    def test_retrieval_service_embedding_error(self, mock_vector_repository):
        """Test RetrievalService handles embedding errors gracefully."""
        mock_embedding_repo = Mock()
        mock_embedding_repo.embed_text.side_effect = EmbeddingError("Embedding failed")
        
        service = RetrievalService(mock_vector_repository, mock_embedding_repo)
        
        from src.domain.entities import Query
        query = Query("test question")
        
        # Should handle embedding error gracefully
        with pytest.raises(EmbeddingError):
            service.retrieve_chunks(query)
    
    def test_vector_store_error(self):
        """Test vector store errors are handled properly."""
        mock_vector_repo = Mock()
        mock_vector_repo.search.side_effect = VectorStoreError("Vector store unavailable")
        
        mock_embedding_repo = Mock()
        mock_embedding_repo.embed_text.return_value = [0.1] * 384  # Mock embedding
        from src.domain.services import RetrievalService
        service = RetrievalService(mock_vector_repo, mock_embedding_repo)
        
        from src.domain.entities import Query
        query = Query("test question")
        
        # Should handle vector store error
        with pytest.raises(VectorStoreError):
            service.retrieve_chunks(query)

    def test_answer_service_llm_error(self):
        """Test AnswerService handles LLM errors gracefully."""
        mock_llm_repo = Mock()
        mock_llm_repo.generate_answer.side_effect = LLMError("LLM unavailable")
        
        service = AnswerService(mock_llm_repo)
        
        from src.domain.entities import Query, Chunk, SearchResult
        query = Query("test question")
        chunk = Chunk("1", "doc1", "test content", 0, 0, 12, {})
        search_results = [SearchResult(chunk=chunk, score=0.9, rank=0)]
        
        # Should handle LLM error
        with pytest.raises(LLMError):
            service.generate_answer(query, search_results)


class TestErrorLogging:
    """Test error logging functionality."""
    
    @patch('src.error_handler.logger')
    def test_error_logging(self, mock_logger):
        """Test log_error function."""
        test_error = Exception("Test error")
        context = {"user_id": "123", "query": "test"}

        from src.error_handler import log_error
        log_error(test_error, "Test operation failed", context)

        # Should log error with context
        assert mock_logger.log.called
        # Check that context information is included
        log_call_args = mock_logger.log.call_args[0][1] 
        assert "Test operation failed" in log_call_args
    
    def test_error_details_preservation(self):
        """Test that error details are preserved through handling."""
        original_details = {"operation": "search", "k": 5, "timeout": True}
        
        error = VectorStoreError(
            message="Search timeout",
            details=original_details
        )
        
        # Details should be preserved
        assert error.details == original_details
        assert error.details["timeout"] == True
        assert error.details["k"] == 5


class TestEdgeCaseErrorScenarios:
    """Test edge case error scenarios."""
    
    def test_empty_embedding_handling(self):
        """Test handling of empty embeddings."""
        with pytest.raises(EmbeddingError):
            from src.embeddings import embed_query
            embed_query("")  # Empty query should raise error
    
    def test_invalid_configuration_handling(self):
        """Test handling of invalid configuration."""
        # This would test configuration validation
        # Could test invalid paths, negative values, etc.
        pass
    
    def test_memory_error_handling(self):
        """Test handling of memory-related errors."""
        # Test very large content processing
        # This is more of an integration test
        pass
    
    def test_concurrent_access_errors(self):
        """Test handling of concurrent access errors."""
        # Test database locking, file access conflicts
        # This would need more complex setup
        pass
