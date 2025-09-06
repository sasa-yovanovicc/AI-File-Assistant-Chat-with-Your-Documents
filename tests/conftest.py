"""Test configuration and fixtures."""

import pytest
import tempfile
import shutil
from typing import List, Generator
import numpy as np

from src.domain.entities import Document, Chunk, Query, SearchResult
from src.domain.repositories.vector_repository import VectorRepository 
from src.domain.repositories.llm_repository import LLMRepository, EmbeddingRepository
from src.domain.repositories.document_repository import DocumentRepository
from src.container import Container


class MockVectorRepository(VectorRepository):
    """Mock vector repository for testing."""
    
    def __init__(self):
        self.chunks = []
        self.embeddings = []
    
    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks with their embeddings."""
        self.chunks.extend(chunks)
        if len(self.embeddings) == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
    
    def save_vectors(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Save vectors (alias for add_chunks)."""
        self.add_chunks(chunks, np.array(embeddings))
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[SearchResult]:
        """Search for similar chunks."""
        if not self.chunks:
            return []
        
        # Simple mock search - return first k chunks with decreasing scores
        results = []
        for i, chunk in enumerate(self.chunks[:k]):
            score = max(0.9 - i * 0.1, 0.1)  # Decreasing scores
            results.append(SearchResult(
                chunk=chunk,
                score=score,
                rank=i
            ))
        return results
    
    def count(self) -> int:
        """Get total number of chunks."""
        return len(self.chunks)
    
    def reset(self) -> None:
        """Clear all data."""
        self.chunks.clear()
        self.embeddings = []


class MockLLMRepository(LLMRepository):
    """Mock LLM repository for testing."""
    
    def __init__(self):
        self.available = True
        self.responses = {}
    
    def generate_answer(self, prompt: str, **kwargs) -> str:
        """Generate mock answer."""
        if "Bert" in prompt or "BERT" in prompt:
            return "BERT je model za obradu prirodnog jezika koji je razvio Google."
        return "Ovo je test odgovor na vaše pitanje."
    
    def is_available(self) -> bool:
        """Check availability."""
        return self.available


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
    from datetime import datetime
    return Document(
        id="test_doc_1",
        source="test.txt",
        content="This is a test document with some content about artificial intelligence.",
        metadata={"file_type": "text/plain", "file_size": 1024},
        created_at=datetime.now()
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
            start_position=0,
            end_position=24,
            metadata={"source": "test.txt", "score": 0.9}
        ),
        Chunk(
            id="chunk_2",
            document_id="test_doc_1",
            content="with some content about artificial intelligence",
            chunk_index=1,
            start_position=25,
            end_position=72,
            metadata={"source": "test.txt", "score": 0.8}
        )
    ]
@pytest.fixture
def sample_query():
    """Provide sample query for testing."""
    return Query(
        text="What is artificial intelligence?",
        k=5
    )


@pytest.fixture
def temp_storage_dir() -> Generator[str, None, None]:
    """Provide temporary directory for storage tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir
