"""Storage infrastructure module."""

from .document_storage import FileDocumentRepository
from .faiss_vector_store import FaissVectorStore

__all__ = ['FileDocumentRepository', 'FaissVectorStore']
