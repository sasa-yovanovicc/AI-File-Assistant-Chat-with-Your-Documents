"""Domain entities for document processing."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime


@dataclass
class Document:
    """Represents a document in the system."""
    id: str
    source: str
    content: str
    metadata: Dict[str, Any]
    created_at: datetime
    
    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Document ID cannot be empty")
        if not self.source:
            raise ValueError("Document source cannot be empty")


@dataclass
class Chunk:
    """Represents a chunk of a document."""
    id: str
    document_id: str
    content: str
    chunk_index: int
    start_position: int
    end_position: int
    metadata: Dict[str, Any]
    
    @property
    def length(self) -> int:
        return len(self.content)
    
    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Chunk ID cannot be empty")
        if not self.document_id:
            raise ValueError("Document ID cannot be empty")
        if self.chunk_index < 0:
            raise ValueError("Chunk index must be non-negative")


@dataclass
class SearchResult:
    """Represents a search result with relevance score."""
    chunk: Chunk
    score: float
    rank: int
    
    def __post_init__(self) -> None:
        if not 0 <= self.score <= 1:
            raise ValueError("Score must be between 0 and 1")
        if self.rank < 0:
            raise ValueError("Rank must be non-negative")
