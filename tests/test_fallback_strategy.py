"""Test cases for k-fallback strategy and answer quality detection."""

import pytest
from unittest.mock import Mock, patch

from src.api import _try_with_fallback_k, _is_poor_answer, _generate_k_fallbacks
from src.domain.entities import Query


class TestKFallbackStrategy:
    """Test intelligent k-fallback strategy."""
    
    def test_generate_k_fallbacks(self):
        """Test fallback k value generation."""
        # Test normal case
        fallbacks = _generate_k_fallbacks(5)
        expected = [6, 4, 3]  # k+1, k-1, k-2
        assert fallbacks == expected
        
        # Test edge cases
        assert _generate_k_fallbacks(1) == [2]  # Only k+1 possible
        assert _generate_k_fallbacks(2) == [3, 1]  # k+1, k-1
        assert _generate_k_fallbacks(15) == [14, 13]  # Only k-1, k-2 (can't go above 15)
    
    def test_is_poor_answer_detection(self):
        """Test poor answer quality detection."""
        # Short answers should be marked as poor (< 80 chars)
        assert _is_poor_answer("Yes.", "Who is Bert?") == True
        assert _is_poor_answer("I don't know.", "What is quantum physics?") == True

        # Generic responses should be poor
        assert _is_poor_answer("Not enough information in the local documents.", "Test question") == True
        assert _is_poor_answer("Cannot find relevant information in documents.", "Test question") == True
        
        # Good answers should not be marked as poor (> 80 chars + contains "is")
        long_answer = "Bert is mentioned as a colleague of the narrator's father in Baghdad, who the narrator was eager to go running with during a summer visit in 1978."
        assert _is_poor_answer(long_answer, "Who is Bert?") == False
        assert _is_poor_answer("Quantum physics is a branch of physics that studies the behavior of matter at the quantum level.", "What is quantum physics?") == False
    
    @patch('src.api.answer_question')
    def test_fallback_strategy_execution(self, mock_answer):
        """Test complete fallback strategy execution."""
        # Mock poor initial answer
        mock_answer.side_effect = [
            {"answer": "I don't know.", "confidence": "low", "sources": []},  # Poor initial
            {"answer": "Bert is mentioned as a colleague of the narrator's father in Baghdad, who the narrator was eager to go running with during a summer visit in 1978.", "confidence": "medium", "sources": [{"source": "story.txt"}]}  # Good fallback (>80 chars)
        ]
        
        result = _try_with_fallback_k("Who is Bert?", original_k=3, use_clean_arch=False)
        
        # Should have tried fallback and succeeded
        assert result["k_used"] != 3  # Different k was used
        assert result["fallback_attempted"] == True
        assert "colleague" in result["answer"]
        assert mock_answer.call_count >= 2  # Initial + fallback
    
    @patch('src.container.container.chat_use_case')
    def test_clean_arch_fallback(self, mock_chat_use_case):
        """Test fallback strategy with Clean Architecture."""
        mock_use_case = Mock()
        mock_chat_use_case.return_value = mock_use_case
        
        # Mock poor then good response
        mock_use_case.execute.side_effect = [
            {"answer": "No.", "confidence": "low", "sources": []},
            {"answer": "Bert is a colleague mentioned in the story who worked with the narrator's father.", "confidence": "high", "sources": []}
        ]
        
        result = _try_with_fallback_k("Who is Bert?", original_k=5, use_clean_arch=True)
        
        assert result["fallback_attempted"] == True
        assert mock_use_case.execute.call_count >= 2


class TestAnswerQualityMetrics:
    """Test answer quality assessment."""
    
    def test_confidence_level_mapping(self):
        """Test confidence level assessment."""
        # This would test _assess_confidence if we extract it
        pass
    
    def test_answer_length_thresholds(self):
        """Test answer length quality thresholds."""
        short_answer = "Yes"
        medium_answer = "Bert is a colleague of the narrator's father who worked in Baghdad and was planning to go running with the narrator."
        long_answer = "Bert is mentioned as a colleague of the narrator's father in Baghdad, who the narrator was eager to go running with during a summer visit in 1978..."
        
        assert _is_poor_answer(short_answer, "Who is Bert?") == True
        assert _is_poor_answer(medium_answer, "Who is Bert?") == False  # Over 80 chars + contains "is"
        assert _is_poor_answer(long_answer, "Who is Bert?") == False
    
    def test_generic_response_detection(self):
        """Test detection of generic/template responses."""
        generic_responses = [
            "I don't have enough information",
            "Not enough information in the local documents",
            "Cannot find relevant information",
            "Unable to answer the question",
            "Sorry, I cannot answer"
        ]
        
        for response in generic_responses:
            assert _is_poor_answer(response, "Test question") == True
