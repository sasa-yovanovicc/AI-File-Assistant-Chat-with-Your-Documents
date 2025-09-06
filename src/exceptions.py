"""Custom exceptions for the AI File Assistant application."""

from typing import Optional


class AIFileAssistantError(Exception):
    """Base exception for all AI File Assistant errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}


class EmbeddingError(AIFileAssistantError):
    """Raised when embedding generation fails."""
    pass


class VectorStoreError(AIFileAssistantError):
    """Raised when vector store operations fail."""
    pass


class DocumentProcessingError(AIFileAssistantError):
    """Raised when document processing fails."""
    pass


class DocumentError(AIFileAssistantError):
    """Raised when document storage operations fail."""
    pass


class LLMError(AIFileAssistantError):
    """Raised when LLM operations fail."""
    pass


class ConfigurationError(AIFileAssistantError):
    """Raised when configuration is invalid."""
    pass


class APIError(AIFileAssistantError):
    """Raised when API operations fail."""
    pass


class FileReadError(DocumentProcessingError):
    """Raised when file reading fails."""
    pass


class ChunkingError(DocumentProcessingError):
    """Raised when text chunking fails."""
    pass


class RetrievalError(AIFileAssistantError):
    """Raised when document retrieval fails."""
    pass
