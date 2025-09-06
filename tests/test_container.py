"""Test cases for dependency injection container."""

import pytest

from src.container import Container
from src.domain.repositories import VectorRepository, LLMRepository, DocumentRepository, EmbeddingRepository
from src.domain.services import RetrievalService, AnswerService
from src.application.use_cases import ChatUseCase


class TestContainer:
    """Test dependency injection container."""
    
    def test_container_creation(self):
        """Test container can be created."""
        container = Container()
        
        assert container is not None
        assert container._vector_repository is None
        assert container._llm_repository is None
    
    def test_singleton_behavior(self):
        """Test that container returns same instances."""
        container = Container()
        
        # Get instances twice
        vector_repo1 = container.vector_repository()
        vector_repo2 = container.vector_repository()
        
        # Should be same instance
        assert vector_repo1 is vector_repo2
        
        llm_repo1 = container.llm_repository()
        llm_repo2 = container.llm_repository()
        
        assert llm_repo1 is llm_repo2
    
    def test_repository_interfaces(self):
        """Test that repositories implement correct interfaces."""
        container = Container()
        
        vector_repo = container.vector_repository()
        assert isinstance(vector_repo, VectorRepository)
        
        llm_repo = container.llm_repository()
        assert isinstance(llm_repo, LLMRepository)
        
        doc_repo = container.document_repository()
        assert isinstance(doc_repo, DocumentRepository)
        
        embedding_repo = container.embedding_repository()
        assert isinstance(embedding_repo, EmbeddingRepository)
    
    def test_service_creation(self):
        """Test that services are properly created and injected."""
        container = Container()
        
        retrieval_service = container.retrieval_service()
        assert isinstance(retrieval_service, RetrievalService)
        
        answer_service = container.answer_service()
        assert isinstance(answer_service, AnswerService)
    
    def test_use_case_creation(self):
        """Test that use cases are properly created."""
        container = Container()
        
        chat_use_case = container.chat_use_case()
        assert isinstance(chat_use_case, ChatUseCase)
        
        # Verify dependencies are injected
        assert chat_use_case._retrieval_service is not None
        assert chat_use_case._answer_service is not None
    
    def test_dependency_chain(self):
        """Test that dependency chain is properly constructed."""
        container = Container()
        
        # Get final use case
        chat_use_case = container.chat_use_case()
        
        # Verify entire chain
        assert chat_use_case._retrieval_service._vector_repo is container.vector_repository()
        assert chat_use_case._retrieval_service._embedding_repo is container.embedding_repository()
        assert chat_use_case._answer_service._llm_repo is container.llm_repository()
    
    def test_container_reset(self):
        """Test container reset functionality."""
        container = Container()
        
        # Create instances
        vector_repo1 = container.vector_repository()
        chat_use_case1 = container.chat_use_case()
        
        # Reset container
        container.reset()
        
        # Get new instances
        vector_repo2 = container.vector_repository()
        chat_use_case2 = container.chat_use_case()
        
        # Should be different instances
        assert vector_repo1 is not vector_repo2
        assert chat_use_case1 is not chat_use_case2
    
    def test_configuration_based_injection(self):
        """Test that container chooses implementations based on configuration."""
        container = Container()
        
        # Get LLM repository (should be based on USE_OPENAI config)
        llm_repo = container.llm_repository()
        
        # Verify it's the expected implementation
        from src.config import USE_OPENAI
        if USE_OPENAI:
            from src.infrastructure.llm import OpenAILLMClient
            assert isinstance(llm_repo, OpenAILLMClient)
        else:
            from src.infrastructure.llm import OllamaLLMClient
            assert isinstance(llm_repo, OllamaLLMClient)
