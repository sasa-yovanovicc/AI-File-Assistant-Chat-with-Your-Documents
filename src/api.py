"""FastAPI backend for AI File Assistant (API only).

Endpoints:
  POST /chat   -> JSON {question:str} returns {question, answer, sources:[{source, chunk_index}]}
  GET  /       -> basic info JSON (frontend is a separate React app)
  GET  /health -> {'status':'ok'}
  GET  /stats  -> basic store stats
"""

from typing import Dict, Any, List
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .rag_pipeline import answer_question, MIN_SCORE as DEFAULT_MIN_SCORE
from .container import container  # Clean Architecture DI container
from .config import (
    API_HOST, API_PORT, FRONTEND_VITE_PORT, FRONTEND_REACT_PORT,
    CHUNK_SIZE, CHUNK_OVERLAP, USE_OPENAI, DEFAULT_RETRIEVAL_K
)
from .vector_store import store
from .exceptions import AIFileAssistantError, VectorStoreError, EmbeddingError, LLMError
from .error_handler import log_error
from .logging_config import get_logger

logger = get_logger(__name__)


def _is_poor_answer(answer: str, question: str = "") -> bool:
    """Check if answer indicates insufficient information or is just an excerpt."""
    poor_indicators = [
        "not enough information",
        "local documents",
        "cannot find",
        "insufficient data",
        "nema podataka",
        "nedostaju informacije"
    ]
    answer_lower = answer.lower()
    question_lower = question.lower()
    
    # Check for explicit poor answer indicators
    if any(indicator in answer_lower for indicator in poor_indicators):
        return True
    
    # Check if answer is too short (likely excerpt rather than comprehensive answer)
    if len(answer) < 80:
        return True
    
    # For "ko je" questions, check if answer actually defines the person
    if question_lower.startswith("ko je ") or "who is" in question_lower:
        person_name = ""
        if question_lower.startswith("ko je "):
            person_name = question_lower.replace("ko je ", "").strip()
        elif "who is" in question_lower:
            person_name = question_lower.split("who is")[-1].strip()
        
        if person_name:
            # Good answer should contain "je" (is) or similar defining words
            if " je " not in answer_lower and " is " not in answer_lower:
                return True
            # Answer should mention the person's name or role
            if person_name not in answer_lower and "kolega" not in answer_lower and "colleague" not in answer_lower:
                return True
    
    return False


def _generate_k_fallbacks(original_k: int) -> List[int]:
    """Generate fallback k values to try if original k fails."""
    fallbacks = []
    
    # Try one above
    if original_k < 15:
        fallbacks.append(original_k + 1)
    
    # Try one or two below
    if original_k > 1:
        fallbacks.append(original_k - 1)
    if original_k > 2:
        fallbacks.append(original_k - 2)
    
    # Remove duplicates and invalid values
    fallbacks = [k for k in fallbacks if 1 <= k <= 15 and k != original_k]
    
    return fallbacks


def _try_with_fallback_k(question: str, original_k: int, min_score: float = None, 
                        llm: bool = True, vector_store=None, use_clean_arch: bool = False) -> Dict[str, Any]:
    """Try answering with different k values if the first attempt gives poor results."""
    
    # First attempt with original k
    if use_clean_arch:
        chat_use_case = container.chat_use_case()
        result = chat_use_case.execute(question, k=original_k)
        resp = {
            "question": question,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "sources": result["sources"],
            "kw_coverage": result.get("kw_coverage", 0.0),
            "reason": result.get("reason", ""),
            "k_used": original_k,
            "fallback_attempted": False
        }
    else:
        from . import rag_pipeline as rp
        old_score = None
        if min_score is not None:
            old_score = rp.MIN_SCORE
            rp.MIN_SCORE = min_score
        try:
            resp = answer_question(question, k=original_k, use_local_llm=llm, vector_store=vector_store)
            resp["k_used"] = original_k
            resp["fallback_attempted"] = False
        finally:
            if old_score is not None:
                rp.MIN_SCORE = old_score
    
    # Check if answer is poor and we should try fallbacks
    is_poor = _is_poor_answer(resp["answer"], question) or resp.get("confidence") == "none"
    if not is_poor:
        return resp
    
    # Try fallback k values
    fallback_ks = _generate_k_fallbacks(original_k)
    logger.info(f"Poor answer with k={original_k} (confidence={resp.get('confidence')}), trying fallbacks: {fallback_ks}")
    
    best_resp = resp  # Keep original as backup
    
    for fallback_k in fallback_ks:
        logger.info(f"Trying fallback k={fallback_k}")
        
        try:
            if use_clean_arch:
                chat_use_case = container.chat_use_case()
                result = chat_use_case.execute(question, k=fallback_k)
                fallback_resp = {
                    "question": question,
                    "answer": result["answer"],
                    "confidence": result["confidence"],
                    "sources": result["sources"],
                    "kw_coverage": result.get("kw_coverage", 0.0),
                    "reason": result.get("reason", ""),
                    "k_used": fallback_k,
                    "fallback_attempted": True
                }
            else:
                from . import rag_pipeline as rp
                old_score = None
                if min_score is not None:
                    old_score = rp.MIN_SCORE
                    rp.MIN_SCORE = min_score
                try:
                    fallback_resp = answer_question(question, k=fallback_k, use_local_llm=llm, vector_store=vector_store)
                    fallback_resp["k_used"] = fallback_k
                    fallback_resp["fallback_attempted"] = True
                finally:
                    if old_score is not None:
                        rp.MIN_SCORE = old_score
            
            # If this fallback gives a better answer, use it
            is_fallback_poor = _is_poor_answer(fallback_resp["answer"], question) or fallback_resp.get("confidence") == "none"
            if not is_fallback_poor:
                logger.info(f"Fallback k={fallback_k} succeeded")
                return fallback_resp
            
            # Keep track of the best answer so far (longer is often better)
            if len(fallback_resp["answer"]) > len(best_resp["answer"]):
                best_resp = fallback_resp
                
        except Exception as e:
            logger.warning(f"Fallback k={fallback_k} failed: {e}")
            continue
    
    logger.info(f"All fallbacks failed, returning best attempt (k={best_resp['k_used']})")
    return best_resp

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
         llm: bool = Query(True, description="Use local LLM via Ollama if available"),
         use_clean_arch: bool = Query(False, description="Use Clean Architecture (experimental)")) -> Dict[str, Any]:
    """Answer a question using RAG pipeline with proper error handling and k-fallback strategy."""
    q = req.question.strip()
    if not q:
        return JSONResponse({"error": "empty question"}, status_code=400)
    
    try:
        logger.info(f"Chat request: {q[:50]}... (k={k}, llm={llm}, clean_arch={use_clean_arch})")
        
        # Use fallback strategy for both architectures
        resp = _try_with_fallback_k(
            question=q,
            original_k=k,
            min_score=min_score,
            llm=llm,
            vector_store=store,
            use_clean_arch=use_clean_arch
        )
        
        logger.info(f"Chat response: k_used={resp.get('k_used', k)}, fallback={resp.get('fallback_attempted', False)}")
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
def health() -> Dict[str, str]:
  return {"status": "ok"}


@app.post("/v2/chat") 
def chat_v2(req: ChatRequest,
            k: int = Query(3, ge=1, le=15, description="Retrieval depth (top-k chunks)")) -> Dict[str, Any]:
    """Modern chat endpoint using Clean Architecture (v2 API) with k-fallback strategy."""
    q = req.question.strip()
    if not q:
        return JSONResponse({"error": "empty question"}, status_code=400)
    
    try:
        logger.info(f"Chat v2 request: {q[:50]}... (k={k})")
        
        # Use fallback strategy with Clean Architecture
        legacy_result = _try_with_fallback_k(
            question=q,
            original_k=k,
            min_score=None,
            llm=True,
            vector_store=store,
            use_clean_arch=True
        )
        
        # Convert legacy format to v2 API format
        resp = {
            "query": {
                "text": q,
                "max_results": legacy_result.get("k_used", k)
            },
            "result": {
                "answer": legacy_result["answer"],
                "confidence": {
                    "level": legacy_result["confidence"],
                    "description": legacy_result.get("reason", "")
                },
                "sources": legacy_result["sources"],
                "metadata": {
                    "retrieval_strategy": legacy_result.get("strategy", "semantic_search"),
                    "architecture": "clean_architecture",
                    "keyword_coverage": legacy_result.get("kw_coverage", 0.0),
                    "k_used": legacy_result.get("k_used", k),
                    "fallback_attempted": legacy_result.get("fallback_attempted", False)
                }
            }
        }
        
        logger.info(f"Chat v2 response: k_used={legacy_result.get('k_used', k)}, fallback={legacy_result.get('fallback_attempted', False)}")
        return resp
        
    except AIFileAssistantError:
        raise
    except Exception as e:
        log_error(e, "Unexpected error in chat v2 endpoint", {'question': q, 'k': k})
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "An unexpected error occurred while processing your request"
            }
        )


@app.get("/stats")
def stats() -> Dict[str, Any]:
  try:
    doc_count = store.count()
    index_size = store.index_size()
  except Exception:  # pragma: no cover
    doc_count = None
    index_size = None
  return {"documents": doc_count, "index_size": index_size, "default_min_score": DEFAULT_MIN_SCORE}


@app.get("/info")
def info() -> Dict[str, Any]:
    """Get information about available architectures and components."""
    try:
        # Test Clean Architecture availability
        chat_use_case = container.chat_use_case()
        clean_arch_available = True
        clean_arch_error = None
    except Exception as e:
        clean_arch_available = False
        clean_arch_error = str(e)
    
    return {
        "version": "2.0",
        "architectures": {
            "legacy": {
                "available": True,
                "endpoint": "/chat",
                "description": "Original RAG pipeline with proven stability"
            },
            "clean_architecture": {
                "available": clean_arch_available,
                "endpoint": "/v2/chat",
                "description": "Modern Clean Architecture with dependency injection",
                "error": clean_arch_error if not clean_arch_available else None
            }
        },
        "configuration": {
            "use_openai": USE_OPENAI,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "default_k": DEFAULT_RETRIEVAL_K
        },
        "migration": {
            "status": "completed",
            "data_location": "data/uploads/",
            "backward_compatible": True
        }
    }


@app.post("/admin/refresh")
def refresh_index() -> Dict[str, Any]:
  store.refresh()
  return {"status": "reloaded", "index_size": store.index_size()}


@app.post("/admin/reset")
def reset_vector_store() -> Dict[str, Any]:
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
def get_admin_stats() -> Dict[str, Any]:
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
def root() -> Dict[str, Any]:
  return {
    "name": "AI File Assistant API",
    "endpoints": ["POST /chat", "GET /stats", "GET /health", "POST /admin/refresh", "POST /admin/reset", "GET /admin/stats"],
  "frontend": "React dev server runs separately (see frontend folder).",
  "notes": "Use POST /admin/refresh after ingest from another process to load FAISS index."
  }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host=API_HOST, port=API_PORT, reload=True)
