"""Integration tests for complete system workflows."""

import pytest
import tempfile
import os
from unittest.mock import patch, Mock

from src.container import Container
from src.domain.entities import Document, Chunk, Query


class TestEndToEndWorkflows:
    """Test complete end-to-end system workflows."""
    
    def test_complete_rag_pipeline_legacy(self):
        """Test complete RAG pipeline using legacy architecture."""
        # This would test the full flow:
        # Question -> Embedding -> Vector Search -> Reranking -> LLM -> Answer
        pass
    
    def test_complete_rag_pipeline_clean_arch(self, test_container):
        """Test complete RAG pipeline using Clean Architecture."""
        # Mock the full pipeline
        chat_use_case = test_container.chat_use_case()
        
        # Add test data
        from src.domain.entities import Chunk
        chunks = [
            Chunk("1", "doc1", "BERT je AI model za obradu prirodnog jezika", 0, 0, 45, {}),
            Chunk("2", "doc1", "Transformeri su arhitektura koja se koristi u BERT-u", 46, 46, 96, {})
        ]
        embeddings = [[0.1] * 384, [0.2] * 384]
        
        # Mock vector store with data
        vector_repo = test_container.vector_repository()
        vector_repo.save_vectors(chunks, embeddings)
        
        # Execute query
        result = chat_use_case.execute("Ko je BERT?")
        
        # Verify complete pipeline execution
        assert isinstance(result, dict)
        assert "query_text" in result
        assert "answer" in result
        assert "confidence" in result
        assert "sources" in result
    
    def test_document_ingestion_workflow(self, temp_storage_dir):
        """Test complete document ingestion workflow."""
        # Create test document
        test_doc_path = os.path.join(temp_storage_dir, "test.txt")
        with open(test_doc_path, 'w', encoding='utf-8') as f:
            f.write("BERT je model za obradu prirodnog jezika. Koristi se za različite NLP zadatke.")
        
        # This would test:
        # File -> Text Extraction -> Chunking -> Embedding -> Storage
        # For now, just verify the infrastructure exists
        from src.infrastructure.storage import FileDocumentRepository
        doc_repo = FileDocumentRepository(temp_storage_dir)
        
        # Create and save document
        from datetime import datetime
        doc = Document(
            id="test_doc",
            source="test.txt",
            content="BERT je model za obradu prirodnog jezika. Koristi se za različite NLP zadatke.",
            metadata={"file_type": "txt"},
            created_at=datetime.now()
        )
        
        doc_id = doc_repo.save_document(doc)
        assert doc_id is not None
        
        # Verify retrieval
        retrieved = doc_repo.get_document(doc_id)
        assert retrieved is not None
        assert retrieved.content == doc.content


class TestSystemIntegration:
    """Test integration between system components."""
    
    def test_container_integration(self):
        """Test that container properly wires all components."""
        container = Container()
        
        # Test that all components can be created
        vector_repo = container.vector_repository()
        doc_repo = container.document_repository()
        llm_repo = container.llm_repository()
        embedding_repo = container.embedding_repository()
        
        # Test services
        retrieval_service = container.retrieval_service()
        answer_service = container.answer_service()
        
        # Test use case
        chat_use_case = container.chat_use_case()
        
        # Verify dependency injection worked
        assert retrieval_service._vector_repo is vector_repo
        assert retrieval_service._embedding_repo is embedding_repo
        assert answer_service._llm_repo is llm_repo
        assert chat_use_case._retrieval_service is retrieval_service
        assert chat_use_case._answer_service is answer_service
    
    def test_dual_architecture_compatibility(self):
        """Test that both legacy and Clean Architecture work with same data."""
        # This would test that both /chat and /v2/chat endpoints
        # work with the same underlying data and give consistent results
        pass
    
    @patch('src.api._try_with_fallback_k')
    def test_api_integration(self, mock_fallback):
        """Test API integration with fallback strategy."""
        from fastapi.testclient import TestClient
        from src.api import app
        
        client = TestClient(app)
        
        # Mock fallback response
        mock_fallback.return_value = {
            "question": "Ko je BERT?",
            "answer": "BERT je AI model",
            "confidence": "high",
            "sources": [],
            "k_used": 3,
            "fallback_attempted": False
        }
        
        # Test legacy endpoint
        response = client.post("/chat", json={"question": "Ko je BERT?"})
        assert response.status_code == 200
        
        data = response.json()
        assert "answer" in data
        assert mock_fallback.called
    
    def test_error_propagation(self):
        """Test that errors propagate correctly through layers."""
        container = Container()
        
        # Mock embedding service to raise error
        embedding_repo = container.embedding_repository()
        with patch.object(embedding_repo, 'embed_text', side_effect=Exception("Embedding failed")):
            
            retrieval_service = container.retrieval_service()
            
            # Error should propagate through the layers
            with pytest.raises(Exception):
                query = Query("test question")
                retrieval_service.retrieve_chunks(query)


class TestPerformanceIntegration:
    """Test performance-related integration scenarios."""
    
    def test_large_document_handling(self):
        """Test system behavior with large documents."""
        # Create large document content
        large_content = "BERT model. " * 1000  # Large document
        
        # Test chunking and processing
        # This would verify memory usage stays reasonable
        pass
    
    def test_many_documents_handling(self):
        """Test system behavior with many documents."""
        # Test with 100+ documents to verify scalability
        pass
    
    def test_concurrent_requests(self):
        """Test system behavior under concurrent load."""
        # Test multiple simultaneous API requests
        pass


class TestConfigurationIntegration:
    """Test different configuration scenarios."""
    
    def test_openai_vs_local_embeddings(self):
        """Test switching between OpenAI and local embeddings."""
        # Test both embedding providers work
        pass
    
    def test_different_chunk_sizes(self):
        """Test system with different chunking configurations."""
        # Test various CHUNK_SIZE and CHUNK_OVERLAP values
        pass
    
    def test_different_retrieval_parameters(self):
        """Test system with different retrieval parameters."""
        # Test various k values, min_score thresholds
        pass


class TestDataMigration:
    """Test data migration scenarios."""
    
    def test_legacy_to_clean_migration(self):
        """Test migration from legacy to Clean Architecture data."""
        # Test that data created with legacy system
        # works with Clean Architecture
        pass
    
    def test_centralized_storage_migration(self):
        """Test migration to centralized data/uploads/ structure."""
        # Test migration scripts work correctly
        pass
    
    def test_backwards_compatibility(self):
        """Test backwards compatibility with old data formats."""
        # Ensure old data can still be read
        pass
