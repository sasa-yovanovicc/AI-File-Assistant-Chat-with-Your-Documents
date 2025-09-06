"""FastAPI backend for AI File Assistant (API only).

Endpoints:
  POST /chat   -> JSON {question:str} returns {question, answer, sources:[{source, chunk_index}]}
  GET  /       -> basic info JSON (frontend is a separate React app)
  GET  /health -> {'status':'ok'}
  GET  /stats  -> basic store stats
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .rag_pipeline import answer_question, MIN_SCORE as DEFAULT_MIN_SCORE
from .config import (
    API_HOST, API_PORT, FRONTEND_VITE_PORT, FRONTEND_REACT_PORT,
    CHUNK_SIZE, CHUNK_OVERLAP, USE_OPENAI, DEFAULT_RETRIEVAL_K
)
from .vector_store import store
from .exceptions import AIFileAssistantError, VectorStoreError, EmbeddingError, LLMError
from .error_handler import log_error
from .logging_config import get_logger

logger = get_logger(__name__)

app = FastAPI(title="AI File Assistant")

# Global exception handler for our custom exceptions
@app.exception_handler(AIFileAssistantError)
async def ai_file_assistant_exception_handler(request, exc: AIFileAssistantError):
    log_error(exc, f"API error in {request.url.path}")
    
    # Map exception types to HTTP status codes
    status_code = 500
    if isinstance(exc, (EmbeddingError, VectorStoreError)):
        status_code = 503  # Service Unavailable
    elif isinstance(exc, LLMError):
        status_code = 502  # Bad Gateway
    
    return JSONResponse(
        status_code=status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details
        }
    )

app.add_middleware(
  CORSMiddleware,
  allow_origins=[
        f"http://127.0.0.1:{FRONTEND_VITE_PORT}",
        "http://localhost:5173",
        f"http://127.0.0.1:{FRONTEND_REACT_PORT}",
        "http://localhost:3000",
    ],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


class ChatRequest(BaseModel):
  question: str


@app.post("/chat")
def chat(req: ChatRequest,
         k: int = Query(3, ge=1, le=15, description="Retrieval depth (top-k chunks)"),
         min_score: float = Query(None, description="Override minimal similarity score"),
         llm: bool = Query(True, description="Use local LLM via Ollama if available")):
    """Answer a question using RAG pipeline with proper error handling."""
    q = req.question.strip()
    if not q:
        return JSONResponse({"error": "empty question"}, status_code=400)
    
    try:
        logger.info(f"Chat request: {q[:50]}... (k={k}, llm={llm})")
        
        # Allow temporary override of MIN_SCORE without modifying module global
        if min_score is not None:
            from . import rag_pipeline as rp  # local import to avoid circular issues
            old = rp.MIN_SCORE
            rp.MIN_SCORE = min_score
            try:
                resp = answer_question(q, k=k, use_local_llm=llm)
            finally:
                rp.MIN_SCORE = old
        else:
            resp = answer_question(q, k=k, use_local_llm=llm)
        
        logger.info(f"Chat response: confidence={resp.get('confidence', 'unknown')}")
        return resp
        
    except AIFileAssistantError:
        # These are handled by the global exception handler
        raise
    except Exception as e:
        # Wrap unexpected errors
        log_error(e, "Unexpected error in chat endpoint", {
            'question': q,
            'k': k,
            'llm': llm
        })
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError", 
                "message": "An unexpected error occurred while processing your request"
            }
        )


@app.get("/health")
def health():
  return {"status": "ok"}


@app.get("/stats")
def stats():
  try:
    doc_count = store.count()
    index_size = store.index_size()
  except Exception:  # pragma: no cover
    doc_count = None
    index_size = None
  return {"documents": doc_count, "index_size": index_size, "default_min_score": DEFAULT_MIN_SCORE}


@app.post("/admin/refresh")
def refresh_index():
  store.refresh()
  return {"status": "reloaded", "index_size": store.index_size()}


@app.post("/admin/reset")
def reset_vector_store():
    """Reset/clear the vector store and metadata."""
    try:
        store.reset()
        return {
            "status": "success", 
            "message": "Vector store reset successfully",
            "index_size": store.count()
        }
    except Exception as e:
        from .exceptions import AIFileAssistantError
        raise AIFileAssistantError(
            message=f"Failed to reset vector store: {str(e)}",
            details={"operation": "reset", "error": str(e)}
        )


@app.get("/admin/stats")
def get_admin_stats():
    """Get detailed admin statistics."""
    try:
        # Get metadata count
        metadata_count = "N/A"
        try:
            with store.conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM documents")
                metadata_count = cur.fetchone()[0]
        except Exception:
            pass
            
        return {
            "vector_store": {
                "total_chunks": store.count(),
                "index_size": store.index_size(),
                "has_index": store.index is not None
            },
            "metadata": {
                "total_documents": metadata_count
            },
            "configuration": {
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "embedding_model": "OpenAI" if USE_OPENAI else "sentence-transformers",
                "retrieval_k": DEFAULT_RETRIEVAL_K
            }
        }
    except Exception as e:
        from .exceptions import AIFileAssistantError
        raise AIFileAssistantError(
            message=f"Failed to get admin stats: {str(e)}",
            details={"operation": "admin_stats", "error": str(e)}
        )


@app.get("/")
def root():
  return {
    "name": "AI File Assistant API",
    "endpoints": ["POST /chat", "GET /stats", "GET /health", "POST /admin/refresh", "POST /admin/reset", "GET /admin/stats"],
  "frontend": "React dev server runs separately (see frontend folder).",
  "notes": "Use POST /admin/refresh after ingest from another process to load FAISS index."
  }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host=API_HOST, port=API_PORT, reload=True)
