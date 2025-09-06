"""FastAPI backend for AI File Assistant (API only).

Endpoints:
  POST /chat   -> JSON {question:str} returns {question, answer, sources:[{source, chunk_index}]}
  GET  /       -> basic info JSON (frontend is a separate React app)
  GET  /health -> {'status':'ok'}
  GET  /stats  -> basic store stats
"""

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .rag_pipeline import answer_question, MIN_SCORE as DEFAULT_MIN_SCORE
from .vector_store import store
from .config import API_HOST, API_PORT, FRONTEND_VITE_PORT, FRONTEND_REACT_PORT

app = FastAPI(title="AI File Assistant")

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
  q = req.question.strip()
  if not q:
    return JSONResponse({"error": "empty question"}, status_code=400)
  # Allow temporary override of MIN_SCORE without modifying module global
  if min_score is not None:
    from . import rag_pipeline as rp  # local import to avoid circular issues
    old = rp.MIN_SCORE
    rp.MIN_SCORE = min_score
    try:
      resp = answer_question(q, k=k, use_local_llm=llm)
    finally:
      rp.MIN_SCORE = old
    return resp
  return answer_question(q, k=k, use_local_llm=llm)


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


@app.get("/")
def root():
  return {
    "name": "AI File Assistant API",
    "endpoints": ["POST /chat", "GET /stats", "GET /health"],
  "frontend": "React dev server runs separately (see frontend folder).",
  "notes": "Use POST /admin/refresh after ingest from another process to load FAISS index."
  }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host=API_HOST, port=API_PORT, reload=True)
