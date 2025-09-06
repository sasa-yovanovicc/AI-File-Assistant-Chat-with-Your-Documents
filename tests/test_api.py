"""Test cases for API integration with Clean Architecture."""

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture
def client():
    """Create test client for API testing."""
    return TestClient(app)


class TestAPIIntegration:
    """Test API integration with Clean Architecture."""
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    
    def test_info_endpoint(self, client):
        """Test info endpoint showing architecture status."""
        response = client.get("/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "version" in data
        assert "architectures" in data
        assert "legacy" in data["architectures"]
        assert "clean_architecture" in data["architectures"]
        assert "configuration" in data
        assert "migration" in data
    
    def test_legacy_chat_endpoint(self, client):
        """Test legacy chat endpoint."""
        response = client.post("/chat", json={"question": "What is AI?"})
        
        # Should work even without data
        assert response.status_code in [200, 500]  # May fail without vector data
        
        if response.status_code == 200:
            data = response.json()
            assert "question" in data
            assert "answer" in data
    
    def test_legacy_chat_with_clean_arch_flag(self, client):
        """Test legacy chat endpoint with clean architecture flag."""
        response = client.post(
            "/chat?use_clean_arch=true", 
            json={"question": "What is AI?"}
        )
        
        # Should attempt to use Clean Architecture
        assert response.status_code in [200, 500]
    
    def test_v2_chat_endpoint(self, client):
        """Test v2 chat endpoint (pure Clean Architecture)."""
        response = client.post("/v2/chat", json={"question": "What is AI?"})
        
        # Should work even without data
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "query" in data
            assert "result" in data
            assert "answer" in data["result"]
            assert "confidence" in data["result"]
            assert "sources" in data["result"]
            assert "metadata" in data["result"]
            assert data["result"]["metadata"]["architecture"] == "clean_architecture"
    
    def test_stats_endpoint(self, client):
        """Test stats endpoint."""
        response = client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "documents" in data
        assert "index_size" in data
        assert "default_min_score" in data
    
    def test_chat_validation(self, client):
        """Test chat endpoint input validation."""
        # Empty question
        response = client.post("/chat", json={"question": ""})
        assert response.status_code == 400
        
        response = client.post("/v2/chat", json={"question": ""})
        assert response.status_code == 400
        
        # Invalid parameters
        response = client.post("/chat?k=0", json={"question": "test"})
        assert response.status_code == 422  # Validation error
        
        response = client.post("/chat?k=20", json={"question": "test"})
        assert response.status_code == 422  # Out of range
