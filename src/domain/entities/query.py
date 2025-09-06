"""Query domain entities."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class ConfidenceLevel(Enum):
    """Confidence levels for query results."""
    NONE = "none"
    LOW = "low"
    HIGH = "high"


class AnswerStrategy(Enum):
    """Strategy for generating answers."""
    NONE = "none"
    EXTRACT = "extract"
    GENERATE = "generate"


@dataclass
class Query:
    """Represents a user query."""
    text: str
    k: int = 5
    min_score: Optional[float] = None
    use_llm: bool = True
    
    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Query text cannot be empty")
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.min_score is not None and not 0 <= self.min_score <= 1:
            raise ValueError("min_score must be between 0 and 1")


@dataclass 
class QueryResult:
    """Represents the result of a query."""
    query: Query
    answer: str
    confidence: ConfidenceLevel
    confidence_reason: str
    strategy: AnswerStrategy
    search_results: List[Any]  # Will be SearchResult from document.py
    keyword_coverage: float
    metadata: Dict[str, Any]
    
    def __post_init__(self) -> None:
        if not 0 <= self.keyword_coverage <= 1:
            raise ValueError("keyword_coverage must be between 0 and 1")
