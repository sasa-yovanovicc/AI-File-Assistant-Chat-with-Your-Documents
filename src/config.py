import os
from dotenv import load_dotenv

load_dotenv()

# Basic settings
DATA_DIR = os.path.abspath(os.getenv("DATA_DIR", "data"))
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
# Updated paths to use uploads directory for centralized storage
DB_PATH = os.path.join(UPLOADS_DIR, "vector_db", "vectors.db")
FAISS_INDEX_PATH = os.path.join(UPLOADS_DIR, "faiss", "faiss.index")
# Multilingual default model (covers >50 languages). Users can override in .env
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Tuned smaller defaults for better definition retrieval (can override via .env)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))

# Local LLM (Ollama) configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# OpenAI configuration (for chat completion, not just embeddings)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# RAG Pipeline Configuration
RETRIEVAL_MIN_SCORE = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.45"))
KEYWORD_COVERAGE_THRESHOLD = float(os.getenv("KEYWORD_COVERAGE_THRESHOLD", "0.25"))
STRICT_MODE_ENABLED = os.getenv("STRICT_MODE_ENABLED", "false").lower() == "true"

# Answer Extraction Parameters
LEXICAL_COVERAGE_WEIGHT = float(os.getenv("LEXICAL_COVERAGE_WEIGHT", "0.3"))
SHORT_SENTENCE_PENALTY = float(os.getenv("SHORT_SENTENCE_PENALTY", "0.5"))
MIN_ANSWER_SCORE = float(os.getenv("MIN_ANSWER_SCORE", "0.2"))
LEXICAL_RERANK_WEIGHT = float(os.getenv("LEXICAL_RERANK_WEIGHT", "0.08"))

# Definition Extraction Parameters
MIN_DEFINITION_LENGTH = int(os.getenv("MIN_DEFINITION_LENGTH", "15"))
MAX_DEFINITION_LENGTH = int(os.getenv("MAX_DEFINITION_LENGTH", "400"))
DEFINITION_NAME_WEIGHT = float(os.getenv("DEFINITION_NAME_WEIGHT", "0.3"))

# Text Processing Parameters
MIN_KEYWORD_LENGTH = int(os.getenv("MIN_KEYWORD_LENGTH", "2"))
MAX_ANSWER_LENGTH = int(os.getenv("MAX_ANSWER_LENGTH", "1200"))

# Retrieval Defaults
DEFAULT_RETRIEVAL_K = int(os.getenv("DEFAULT_RETRIEVAL_K", "5"))
DEFAULT_ANSWER_K = int(os.getenv("DEFAULT_ANSWER_K", "3"))

# LLM Parameters
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "400"))

# API Configuration
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
FRONTEND_VITE_PORT = int(os.getenv("FRONTEND_VITE_PORT", "5173"))
FRONTEND_REACT_PORT = int(os.getenv("FRONTEND_REACT_PORT", "3000"))

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
# Create subdirectories for centralized storage
os.makedirs(os.path.join(UPLOADS_DIR, "vector_db"), exist_ok=True)
os.makedirs(os.path.join(UPLOADS_DIR, "faiss"), exist_ok=True)
os.makedirs(os.path.join(UPLOADS_DIR, "documents"), exist_ok=True)
os.makedirs(os.path.join(UPLOADS_DIR, "logs"), exist_ok=True)
