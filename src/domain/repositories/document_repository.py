"""Document repository interface."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..entities import Document, Chunk


class DocumentRepository(ABC):
    """Abstract interface for document storage operations."""
    
    @abstractmethod
    def save_document(self, document: Document) -> str:
        """Save document and return its ID."""
        pass
    
    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID."""
        pass
    
    @abstractmethod
    def list_documents(self) -> List[Document]:
        """List all documents."""
        pass
    
    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """Delete document and return success status."""
        pass
    
    @abstractmethod
    def save_chunks(self, chunks: List[Chunk]) -> List[str]:
        """Save document chunks and return their IDs."""
        pass
    
    @abstractmethod
    def get_chunks_by_document(self, document_id: str) -> List[Chunk]:
        """Get all chunks for a document."""
        pass
