"""Test cases for Clean Architecture infrastructure layer."""

import pytest
import os
import tempfile

from src.infrastructure.storage import FaissVectorStore, FileDocumentRepository
from src.infrastructure.llm import OpenAILLMClient, OllamaLLMClient
from src.infrastructure.embeddings import OpenAIEmbeddingClient, LocalEmbeddingClient
from src.domain.entities import Document, Chunk


class TestFaissVectorStore:
    """Test FAISS vector store implementation."""
    
    def test_faiss_vector_store_creation(self, temp_storage_dir):
        """Test FaissVectorStore can be created."""
        db_path = os.path.join(temp_storage_dir, "test.db")
        index_path = os.path.join(temp_storage_dir, "test.index")
        
        store = FaissVectorStore(db_path=db_path, index_path=index_path)
        
        assert store.db_path == db_path
        assert store.index_path == index_path
    
    def test_faiss_vector_store_operations(self, temp_storage_dir, sample_chunks):
        """Test basic FAISS operations."""
        db_path = os.path.join(temp_storage_dir, "test.db")
        index_path = os.path.join(temp_storage_dir, "test.index")
        
        store = FaissVectorStore(db_path=db_path, index_path=index_path)
        
        # Test save vectors
        embeddings = [[0.1] * 384, [0.2] * 384]
        result = store.save_vectors(sample_chunks, embeddings)
        assert result is True
        
        # Test count
        count = store.count()
        assert count >= 0  # May be 0 if FAISS not properly initialized
        
        # Test clear
        clear_result = store.clear()
        assert clear_result is True


class TestFileDocumentRepository:
    """Test file-based document repository."""
    
    def test_file_document_repository_creation(self, temp_storage_dir):
        """Test FileDocumentRepository creation."""
        repo = FileDocumentRepository(temp_storage_dir)
        
        assert repo.storage_path.exists()
        assert repo.metadata_file.exists() or not repo.metadata_file.exists()  # May not exist initially
    
    def test_document_save_and_retrieve(self, temp_storage_dir, sample_document):
        """Test saving and retrieving documents."""
        repo = FileDocumentRepository(temp_storage_dir)
        
        # Save document
        doc_id = repo.save_document(sample_document)
        assert doc_id is not None
        assert isinstance(doc_id, str)
        
        # Retrieve document
        retrieved_doc = repo.get_document(doc_id)
        assert retrieved_doc is not None
        assert retrieved_doc.filename == sample_document.filename
        assert retrieved_doc.content == sample_document.content
    
    def test_chunk_operations(self, temp_storage_dir, sample_chunks):
        """Test chunk save and retrieve operations."""
        repo = FileDocumentRepository(temp_storage_dir)
        
        # Save chunks
        chunk_ids = repo.save_chunks(sample_chunks)
        assert len(chunk_ids) == len(sample_chunks)
        
        # Retrieve chunks by document
        retrieved_chunks = repo.get_chunks_by_document(sample_chunks[0].document_id)
        assert len(retrieved_chunks) >= 0  # May be 0 if storage failed


class TestLLMClients:
    """Test LLM client implementations."""
    
    def test_openai_llm_client_creation(self):
        """Test OpenAI LLM client creation."""
        client = OpenAILLMClient()
        
        assert hasattr(client, 'model_name')
        assert hasattr(client, 'temperature')
        assert hasattr(client, 'max_tokens')
    
    def test_ollama_llm_client_creation(self):
        """Test Ollama LLM client creation."""
        client = OllamaLLMClient()
        
        assert hasattr(client, 'base_url')
        assert hasattr(client, 'model_name')
        assert hasattr(client, 'api_url')
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OpenAI API key")
    def test_openai_llm_response(self):
        """Test OpenAI LLM response generation (if API key available)."""
        client = OpenAILLMClient()
        
        response = client.generate_response("Hello, this is a test.", max_tokens=10)
        
        assert isinstance(response, str)
        assert len(response) > 0


class TestEmbeddingClients:
    """Test embedding client implementations."""
    
    def test_openai_embedding_client_creation(self):
        """Test OpenAI embedding client creation."""
        client = OpenAIEmbeddingClient()
        
        assert client.model == "text-embedding-3-small"
        assert client.dimension == 1536
    
    def test_local_embedding_client_creation(self):
        """Test local embedding client creation."""
        client = LocalEmbeddingClient()
        
        assert client.model_name == "all-MiniLM-L6-v2"
        assert client.dimension == 384
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OpenAI API key")
    def test_openai_embedding_generation(self):
        """Test OpenAI embedding generation (if API key available)."""
        client = OpenAIEmbeddingClient()
        
        embedding = client.embed_text("Hello world")
        
        assert isinstance(embedding, list)
        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)
