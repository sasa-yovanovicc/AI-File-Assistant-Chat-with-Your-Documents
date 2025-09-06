"""Test cases for domain entity validation and edge cases."""

import pytest
from datetime import datetime

from src.domain.entities import Document, Chunk, SearchResult, Query, QueryResult, ConfidenceLevel


class TestEntityValidation:
    """Test domain entity validation logic."""
    
    def test_document_validation(self):
        """Test Document entity validation."""
        # Valid document should work
        doc = Document(
            id="valid_id",
            source="test.txt", 
            content="Valid content",
            metadata={},
            created_at=datetime.now()
        )
        assert doc.id == "valid_id"
        
        # Invalid document should raise errors
        with pytest.raises(ValueError, match="Document ID cannot be empty"):
            Document(
                id="",
                source="test.txt",
                content="Content", 
                metadata={},
                created_at=datetime.now()
            )
        
        with pytest.raises(ValueError, match="Document source cannot be empty"):
            Document(
                id="valid_id",
                source="",
                content="Content",
                metadata={},
                created_at=datetime.now()
            )
    
    def test_chunk_validation(self):
        """Test Chunk entity validation."""
        # Valid chunk
        chunk = Chunk(
            id="chunk_1",
            document_id="doc_1", 
            content="Valid chunk content",
            chunk_index=0,
            start_position=0,
            end_position=20,
            metadata={}
        )
        assert chunk.length == len("Valid chunk content")
        
        # Invalid chunks should raise errors
        with pytest.raises(ValueError):
            Chunk(
                id="",  # Empty ID
                document_id="doc_1",
                content="Content",
                chunk_index=0,
                start_position=0,
                end_position=10,
                metadata={}
            )
        
        with pytest.raises(ValueError):
            Chunk(
                id="chunk_1",
                document_id="",  # Empty document_id
                content="Content", 
                chunk_index=0,
                start_position=0,
                end_position=10,
                metadata={}
            )
        
        with pytest.raises(ValueError):
            Chunk(
                id="chunk_1",
                document_id="doc_1",
                content="Content",
                chunk_index=-1,  # Negative index
                start_position=0,
                end_position=10,
                metadata={}
            )
    
    def test_search_result_validation(self):
        """Test SearchResult entity validation."""
        chunk = Chunk(
            id="chunk_1",
            document_id="doc_1",
            content="Test content",
            chunk_index=0,
            start_position=0,
            end_position=12,
            metadata={}
        )
        
        # Valid search result
        result = SearchResult(
            chunk=chunk,
            score=0.85,
            rank=0
        )
        assert result.blended_score == 0.85  # Should default to score
        
        # Invalid score range
        with pytest.raises(ValueError, match="Score must be between 0 and 1"):
            SearchResult(
                chunk=chunk,
                score=1.5,  # Invalid score
                rank=0
            )
        
        # Invalid rank
        with pytest.raises(ValueError, match="Rank must be non-negative"):
            SearchResult(
                chunk=chunk,
                score=0.8,
                rank=-1  # Invalid rank
            )
        
        # Test blended_score override
        result_with_blend = SearchResult(
            chunk=chunk,
            score=0.7,
            rank=0,
            blended_score=0.9
        )
        assert result_with_blend.blended_score == 0.9
    
    def test_query_validation(self):
        """Test Query entity validation and defaults."""
        # Basic query
        query = Query(text="What is AI?")
        assert query.text == "What is AI?"
        assert query.k == 5  # Should have default
        assert query.min_score is None  # Should be None by default
        assert query.use_llm == True  # Should default to True
        
        # Query with custom parameters
        custom_query = Query(
            text="Custom question",
            k=10,
            min_score=0.7,
            use_llm=False
        )
        assert custom_query.k == 10
        assert custom_query.min_score == 0.7
        assert custom_query.use_llm == False
    
    def test_confidence_level_enum(self):
        """Test ConfidenceLevel enum values."""
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.LOW.value == "low"
        assert ConfidenceLevel.NONE.value == "none"
        assert ConfidenceLevel.NONE.value == "none"
        
        # Test enum can be used in comparisons
        confidence = ConfidenceLevel.HIGH
        assert confidence == ConfidenceLevel.HIGH
        assert confidence != ConfidenceLevel.LOW


class TestEntityEdgeCases:
    """Test edge cases for domain entities."""
    
    def test_empty_content_handling(self):
        """Test handling of empty or whitespace content."""
        # Document with empty content should work but may not be useful
        doc = Document(
            id="empty_doc",
            source="empty.txt",
            content="",
            metadata={},
            created_at=datetime.now()
        )
        assert doc.content == ""
        
        # Chunk with empty content
        chunk = Chunk(
            id="empty_chunk",
            document_id="doc_1",
            content="",
            chunk_index=0,
            start_position=0,
            end_position=0,
            metadata={}
        )
        assert chunk.length == 0
    
    def test_unicode_content_handling(self):
        """Test handling of Unicode and special characters."""
        unicode_content = "Testiranje ћирилице and émojis 🤖 and symbols ∑∞"
        
        chunk = Chunk(
            id="unicode_chunk",
            document_id="doc_1", 
            content=unicode_content,
            chunk_index=0,
            start_position=0,
            end_position=len(unicode_content),
            metadata={}
        )
        
        assert chunk.content == unicode_content
        assert chunk.length == len(unicode_content)
    
    def test_large_content_handling(self):
        """Test handling of very large content."""
        large_content = "x" * 10000  # 10KB of content
        
        chunk = Chunk(
            id="large_chunk",
            document_id="doc_1",
            content=large_content,
            chunk_index=0,
            start_position=0,
            end_position=len(large_content),
            metadata={}
        )
        
        assert chunk.length == 10000
        assert chunk.content == large_content
    
    def test_metadata_flexibility(self):
        """Test that metadata can contain various types."""
        complex_metadata = {
            "source": "test.pdf",
            "page": 42,
            "confidence": 0.95,
            "tags": ["AI", "ML", "NLP"],
            "processing_time": 1.23,
            "nested": {"key": "value"}
        }
        
        chunk = Chunk(
            id="meta_chunk",
            document_id="doc_1",
            content="Content",
            chunk_index=0,
            start_position=0,
            end_position=7,
            metadata=complex_metadata
        )
        
        assert chunk.metadata == complex_metadata
        assert chunk.metadata["page"] == 42
        assert "AI" in chunk.metadata["tags"]
