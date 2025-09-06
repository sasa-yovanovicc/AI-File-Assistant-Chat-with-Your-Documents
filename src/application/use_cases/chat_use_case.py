"""Chat use case implementation."""

from typing import Dict, Any

from ...domain.entities import Query, QueryResult
from ...domain.services import RetrievalService, AnswerService


class ChatUseCase:
    """Use case for handling chat queries."""
    
    def __init__(
        self,
        retrieval_service: RetrievalService, 
        answer_service: AnswerService
    ):
        self._retrieval_service = retrieval_service
        self._answer_service = answer_service
    
    def execute(self, query, **kwargs) -> Dict[str, Any]:
        """Execute the chat use case."""
        # Handle both string and Query object
        if isinstance(query, str):
            query_text = query
            k = kwargs.get('k', 5)
        else:
            # Query object
            query_text = query.text
            k = getattr(query, 'max_results', kwargs.get('k', 5))
        
        # Create query entity for processing
        query_obj = Query(
            text=query_text,
            k=k,
            min_score=kwargs.get('min_score'),
            use_llm=kwargs.get('use_llm', True)
        )
        
        # Retrieve relevant chunks
        search_results = self._retrieval_service.retrieve_chunks(query_obj)
        
        # Generate answer
        result = self._answer_service.generate_answer(query_obj, search_results)
        
        # Convert to API response format (backward compatibility)
        return self._to_api_response(result, query_text)
    
    def _to_api_response(self, result: QueryResult, query_text: str) -> Dict[str, Any]:
        """Convert QueryResult to API response format."""
        return {
            "query_text": query_text,
            "answer": result.answer,
            "confidence": result.confidence.value,
            "sources": [
                {
                    "source": getattr(chunk, 'metadata', {}).get("source", "unknown"),
                    "chunk_index": getattr(chunk, 'chunk_index', 0),
                    "content": getattr(chunk, 'content', '')[:200] + "..." if len(getattr(chunk, 'content', '')) > 200 else getattr(chunk, 'content', ''),
                    "score": getattr(chunk, 'metadata', {}).get("score", 0.0)
                }
                for chunk in result.search_results
            ]
        }
