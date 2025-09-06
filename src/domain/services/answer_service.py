"""Answer generation domain service."""

from typing import List

from ..entities import Query, QueryResult, SearchResult, ConfidenceLevel, AnswerStrategy
from ..repositories import LLMRepository


class AnswerService:
    """Domain service for answer generation."""
    
    def __init__(self, llm_repository: LLMRepository):
        self._llm_repo = llm_repository
    
    def generate_answer(
        self, 
        query: Query, 
        search_results: List[SearchResult]
    ) -> QueryResult:
        """Generate an answer based on query and search results."""
        
        # Assess confidence and determine strategy
        confidence, reason, strategy = self._evaluate_quality(query, search_results)
        
        # Generate answer based on strategy
        if strategy == AnswerStrategy.NONE:
            answer = "Not enough information in the local documents."
        elif strategy == AnswerStrategy.EXTRACT:
            answer = self._extract_answer(query, search_results)
        else:  # GENERATE
            answer = self._generate_llm_answer(query, search_results)
        
        # Calculate keyword coverage
        coverage = self._calculate_keyword_coverage(query, search_results)
        
        return QueryResult(
            query=query,
            answer=answer,
            confidence=confidence,
            confidence_reason=reason,
            strategy=strategy,
            search_results=search_results,
            keyword_coverage=coverage,
            metadata={}
        )
    
    def _evaluate_quality(
        self, 
        query: Query, 
        results: List[SearchResult]
    ) -> tuple[ConfidenceLevel, str, AnswerStrategy]:
        """Evaluate retrieval quality and determine strategy."""
        if not results:
            return ConfidenceLevel.NONE, "no_results", AnswerStrategy.NONE
        
        top_score = results[0].score if results else 0.0
        min_score = query.min_score or 0.45
        
        if top_score < min_score:
            return ConfidenceLevel.LOW, f"score<{min_score:.2f}", AnswerStrategy.GENERATE
        
        return ConfidenceLevel.HIGH, "ok", AnswerStrategy.GENERATE
    
    def _extract_answer(self, query: Query, results: List[SearchResult]) -> str:
        """Extract answer using heuristic methods."""
        # Simple implementation - return first chunk
        if results:
            return results[0].chunk.content[:400]
        return "No relevant information found."
    
    def _generate_llm_answer(self, query: Query, results: List[SearchResult]) -> str:
        """Generate answer using LLM."""
        if not query.use_llm or not self._llm_repo.is_available():
            return self._extract_answer(query, results)
        
        # Build context from search results
        contexts = [result.chunk.content for result in results[:6]]
        prompt = self._build_prompt(query.text, contexts)
        
        return self._llm_repo.generate_answer(prompt)
    
    def _build_prompt(self, question: str, contexts: List[str]) -> str:
        """Build prompt for LLM generation."""
        ctx_joined = "\n\n".join(f"[DOC {i+1}]\n{c}" for i, c in enumerate(contexts))
        return (
            "Answer in an encyclopedic style using ONLY the provided context. "
            "Be precise, factual, and concise like a Wikipedia entry. "
            "If the answer is missing respond exactly 'Not enough information in the local documents.'\n"
            f"Question: {question}\n\nContext:\n{ctx_joined}\n\nAnswer:"
        )
    
    def _calculate_keyword_coverage(
        self, 
        query: Query, 
        results: List[SearchResult]
    ) -> float:
        """Calculate keyword coverage for the results."""
        # Simplified implementation
        if not results:
            return 0.0
        
        # Extract simple keywords from query
        query_words = set(word.lower() for word in query.text.split() if len(word) > 2)
        if not query_words:
            return 0.0
        
        # Check coverage in results
        all_content = " ".join(result.chunk.content.lower() for result in results)
        found_words = sum(1 for word in query_words if word in all_content)
        
        return found_words / len(query_words)
