"""Dependency injection container for Clean Architecture."""

from typing import Optional
import os

from .domain.repositories import VectorRepository, LLMRepository, DocumentRepository, EmbeddingRepository
from .infrastructure.storage import FaissVectorStore, FileDocumentRepository
from .infrastructure.llm import OpenAILLMClient, OllamaLLMClient
from .infrastructure.embeddings import OpenAIEmbeddingClient, LocalEmbeddingClient
from .domain.services import RetrievalService, AnswerService
from .application.use_cases import ChatUseCase
from .config import USE_OPENAI, UPLOADS_DIR
from .logging_config import get_logger

logger = get_logger(__name__)


class Container:
    """Dependency injection container."""
    
    def __init__(self):
        self._vector_repository: Optional[VectorRepository] = None
        self._llm_repository: Optional[LLMRepository] = None
        self._document_repository: Optional[DocumentRepository] = None
        self._embedding_repository: Optional[EmbeddingRepository] = None
        self._retrieval_service: Optional[RetrievalService] = None
        self._answer_service: Optional[AnswerService] = None
        self._chat_use_case: Optional[ChatUseCase] = None
    
    def vector_repository(self) -> VectorRepository:
        """Get vector repository instance."""
        if self._vector_repository is None:
            logger.info("Creating FaissVectorStore instance")
            # Ensure directories exist
            faiss_dir = os.path.join(UPLOADS_DIR, "faiss")
            vector_db_dir = os.path.join(UPLOADS_DIR, "vector_db")
            os.makedirs(faiss_dir, exist_ok=True)
            os.makedirs(vector_db_dir, exist_ok=True)
            
            self._vector_repository = FaissVectorStore()
        return self._vector_repository
    
    def llm_repository(self) -> LLMRepository:
        """Get LLM repository instance."""
        if self._llm_repository is None:
            if USE_OPENAI:
                logger.info("Creating OpenAI LLM client")
                self._llm_repository = OpenAILLMClient()
            else:
                logger.info("Creating Ollama LLM client")
                self._llm_repository = OllamaLLMClient()
        return self._llm_repository
    
    def document_repository(self) -> DocumentRepository:
        """Get document repository instance."""
        if self._document_repository is None:
            storage_path = os.path.join(UPLOADS_DIR, "documents")
            logger.info(f"Creating FileDocumentRepository with path: {storage_path}")
            self._document_repository = FileDocumentRepository(storage_path)
        return self._document_repository
    
    def embedding_repository(self) -> EmbeddingRepository:
        """Get embedding repository instance."""
        if self._embedding_repository is None:
            if USE_OPENAI:
                logger.info("Creating OpenAI embedding client")
                self._embedding_repository = OpenAIEmbeddingClient()
            else:
                logger.info("Creating local embedding client")
                self._embedding_repository = LocalEmbeddingClient()
        return self._embedding_repository
    
    def retrieval_service(self) -> RetrievalService:
        """Get retrieval service instance."""
        if self._retrieval_service is None:
            self._retrieval_service = RetrievalService(
                vector_repository=self.vector_repository(),
                embedding_repository=self.embedding_repository()
            )
        return self._retrieval_service
    
    def answer_service(self) -> AnswerService:
        """Get answer service instance."""
        if self._answer_service is None:
            self._answer_service = AnswerService(
                llm_repository=self.llm_repository()
            )
        return self._answer_service
    
    def chat_use_case(self) -> ChatUseCase:
        """Get chat use case instance."""
        if self._chat_use_case is None:
            self._chat_use_case = ChatUseCase(
                retrieval_service=self.retrieval_service(),
                answer_service=self.answer_service()
            )
        return self._chat_use_case
    
    def reset(self):
        """Reset all instances (useful for testing)."""
        self._vector_repository = None
        self._llm_repository = None
        self._document_repository = None
        self._embedding_repository = None
        self._retrieval_service = None
        self._answer_service = None
        self._chat_use_case = None
        logger.info("Container reset")


# Global container instance
container = Container()
