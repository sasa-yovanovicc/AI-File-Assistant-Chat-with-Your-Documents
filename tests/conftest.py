"""Test configuration and fixtures for Clean Architecture testing."""

import pytest
import tempfile
import os
from typing import Generator
from unittest.mock import Mock

from src.domain.repositories import VectorRepository, LLMRepository, DocumentRepository, EmbeddingRepository
from src.domain.services import RetrievalService, AnswerService
from src.application.use_cases import ChatUseCase
from src.domain.entities import Document, Query, Chunk, QueryResult, ConfidenceLevel
from src.container import Container


class MockVectorRepository(VectorRepository):
    """Mock implementation of VectorRepository for testing."""
    
    def __init__(self):
        self._vectors = {}
        self._chunks = {}
    
    def save_vectors(self, chunks: list, embeddings: list) -> bool:
        for chunk, embedding in zip(chunks, embeddings):
            self._vectors[chunk.id] = embedding
            self._chunks[chunk.id] = chunk
        return True
    
    def search(self, query_embedding: list, k: int = 5) -> list:
        # Simple mock search - return first k chunks
        return list(self._chunks.values())[:k]
    
    def delete_by_source(self, source: str) -> bool:
        to_delete = [chunk_id for chunk_id, chunk in self._chunks.items() 
                    if chunk.metadata.get("source") == source]
        for chunk_id in to_delete:
            del self._chunks[chunk_id]
            del self._vectors[chunk_id]
        return True
    
    def count(self) -> int:
        return len(self._chunks)
    
    def clear(self) -> bool:
        self._vectors.clear()
        self._chunks.clear()
        return True


class MockLLMRepository(LLMRepository):
    """Mock implementation of LLMRepository for testing."""
    
    def __init__(self, mock_response: str = "Mock LLM response"):
        self.mock_response = mock_response
        self.call_count = 0
        self.last_prompt = None
    
    def generate_response(self, prompt: str, system_message: str = None, 
                         temperature: float = 0.3, max_tokens: int = None) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        return self.mock_response


class MockEmbeddingRepository(EmbeddingRepository):
    """Mock implementation of EmbeddingRepository for testing."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.call_count = 0
    
    def get_dimension(self) -> int:
        return self.dimension
    
    def embed_text(self, text: str) -> list:
        self.call_count += 1
        # Return mock embedding vector
        return [0.1] * self.dimension
    
    def embed_texts(self, texts: list) -> list:
        self.call_count += len(texts)
        return [[0.1] * self.dimension for _ in texts]


class MockDocumentRepository(DocumentRepository):
    """Mock implementation of DocumentRepository for testing."""
    
    def __init__(self):
        self._documents = {}
        self._chunks = {}
    
    def save_document(self, document: Document) -> str:
        if not document.id:
            document.id = f"doc_{len(self._documents)}"
        self._documents[document.id] = document
        return document.id
    
    def get_document(self, document_id: str):
        return self._documents.get(document_id)
    
    def list_documents(self):
        return list(self._documents.values())
    
    def delete_document(self, document_id: str) -> bool:
        if document_id in self._documents:
            del self._documents[document_id]
            return True
        return False
    
    def save_chunks(self, chunks: list) -> list:
        chunk_ids = []
        for chunk in chunks:
            if not chunk.id:
                chunk.id = f"chunk_{len(self._chunks)}"
            self._chunks[chunk.id] = chunk
            chunk_ids.append(chunk.id)
        return chunk_ids
    
    def get_chunks_by_document(self, document_id: str):
        return [chunk for chunk in self._chunks.values() 
                if chunk.document_id == document_id]


@pytest.fixture
def mock_vector_repository():
    """Provide mock vector repository."""
    return MockVectorRepository()


@pytest.fixture
def mock_llm_repository():
    """Provide mock LLM repository."""
    return MockLLMRepository()


@pytest.fixture
def mock_embedding_repository():
    """Provide mock embedding repository."""
    return MockEmbeddingRepository()


@pytest.fixture
def mock_document_repository():
    """Provide mock document repository."""
    return MockDocumentRepository()


@pytest.fixture
def test_container(mock_vector_repository, mock_llm_repository, 
                  mock_embedding_repository, mock_document_repository):
    """Provide container with mocked dependencies."""
    container = Container()
    
    # Override with mocks
    container._vector_repository = mock_vector_repository
    container._llm_repository = mock_llm_repository
    container._embedding_repository = mock_embedding_repository
    container._document_repository = mock_document_repository
    
    return container


@pytest.fixture
def sample_document():
    """Provide sample document for testing."""
    return Document(
        id="test_doc_1",
        filename="test.txt",
        content="This is a test document with some content about artificial intelligence.",
        file_type="text/plain",
        file_size=1024,
        metadata={"source": "test"}
    )


@pytest.fixture
def sample_chunks():
    """Provide sample chunks for testing."""
    return [
        Chunk(
            id="chunk_1",
            document_id="test_doc_1",
            content="This is a test document",
            chunk_index=0,
            metadata={"source": "test.txt", "score": 0.9}
        ),
        Chunk(
            id="chunk_2", 
            document_id="test_doc_1",
            content="with some content about artificial intelligence",
            chunk_index=1,
            metadata={"source": "test.txt", "score": 0.8}
        )
    ]


@pytest.fixture
def sample_query():
    """Provide sample query for testing."""
    return Query(
        text="What is artificial intelligence?",
        max_results=5
    )


@pytest.fixture
def temp_storage_dir() -> Generator[str, None, None]:
    """Provide temporary directory for storage tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir
