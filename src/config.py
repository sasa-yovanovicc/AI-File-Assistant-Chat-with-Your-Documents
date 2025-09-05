import os
from dotenv import load_dotenv

load_dotenv()

# Basic settings
DATA_DIR = os.path.abspath(os.getenv("DATA_DIR", "data"))
DB_PATH = os.path.join(DATA_DIR, "vectors.db")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
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

os.makedirs(DATA_DIR, exist_ok=True)
