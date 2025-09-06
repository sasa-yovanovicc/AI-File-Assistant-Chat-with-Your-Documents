"""Chat use case implementation."""

from typing import Dict, Any

from ..domain.entities import Query, QueryResult
from ..domain.services import RetrievalService, AnswerService


class ChatUseCase:
    """Use case for handling chat queries."""
    
    def __init__(
        self,
        retrieval_service: RetrievalService, 
        answer_service: AnswerService
    ):
        self._retrieval_service = retrieval_service
        self._answer_service = answer_service
    
    def execute(self, query_text: str, **kwargs) -> Dict[str, Any]:
        """Execute the chat use case."""
        # Create query entity
        query = Query(
            text=query_text,
            k=kwargs.get('k', 5),
            min_score=kwargs.get('min_score'),
            use_llm=kwargs.get('use_llm', True)
        )
        
        # Retrieve relevant chunks
        search_results = self._retrieval_service.retrieve_chunks(query)
        
        # Generate answer
        result = self._answer_service.generate_answer(query, search_results)
        
        # Convert to API response format (backward compatibility)
        return self._to_api_response(result)
    
    def _to_api_response(self, result: QueryResult) -> Dict[str, Any]:
        """Convert QueryResult to API response format."""
        return {
            "question": result.query.text,
            "answer": result.answer,
            "confidence": result.confidence.value,
            "reason": result.confidence_reason,
            "strategy": result.strategy.value,
            "sources": [
                {
                    "id": sr.chunk.id,
                    "source": sr.chunk.document_id,  # Simplified
                    "chunk_index": sr.chunk.chunk_index,
                    "text": sr.chunk.content,
                    "score": sr.score
                }
                for sr in result.search_results
            ],
            "kw_coverage": result.keyword_coverage,
            "metadata": result.metadata
        }
