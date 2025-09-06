from __future__ import annotations
import os
import json
import sqlite3
import threading
from typing import List, Dict, Tuple
import numpy as np
import faiss
from .config import DB_PATH, FAISS_INDEX_PATH
from .exceptions import VectorStoreError, ConfigurationError
from .error_handler import handle_errors, log_error
from .logging_config import get_logger

logger = get_logger(__name__)

META_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,
  source TEXT,
  chunk_index INTEGER,
  text TEXT
);
"""

class VectorStore:
    def __init__(self, db_path: str = DB_PATH, index_path: str = FAISS_INDEX_PATH, dim: int | None = None):
        self.db_path = db_path
        self.index_path = index_path
        # Thread-safe SQLite connection (allow use across threads)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.RLock()
        self._ensure_meta()
        self.index = None
        self.dim = dim
        if os.path.exists(self.index_path):
            self._load_index()

    def _ensure_meta(self):
        cur = self.conn.cursor()
        cur.execute(META_TABLE_SQL)
        self.conn.commit()

    def _load_index(self):
        self.index = faiss.read_index(self.index_path)
        self.dim = self.index.d

    def _create_index(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)  # cosine if normalized
        self.dim = dim

    def add(self, docs: List[Dict], embeddings: np.ndarray):
        with self._lock:
            if self.index is None:
                self._create_index(embeddings.shape[1])
            if embeddings.shape[1] != self.dim:
                raise ValueError("Embedding dimension mismatch")
            cur = self.conn.cursor()
            for doc in docs:
                cur.execute("INSERT OR REPLACE INTO documents(id, source, chunk_index, text) VALUES(?,?,?,?)",
                            (doc['id'], doc['source'], doc['chunk_index'], doc['text']))
            self.conn.commit()
            self.index.add(embeddings)
            faiss.write_index(self.index, self.index_path)

    @handle_errors(default_return=[], exception_type=VectorStoreError)
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict]:
        with self._lock:
            if self.index is None:
                logger.warning("Search attempted on empty index")
                return []
            
            q = query_vector.reshape(1, -1)
            
            # Check dimension compatibility before search
            if q.shape[1] != self.index.d:
                error_msg = f"Query vector dimension ({q.shape[1]}) does not match index dimension ({self.index.d})"
                logger.error(error_msg)
                raise VectorStoreError(
                    message=error_msg,
                    details={
                        'query_dim': q.shape[1],
                        'index_dim': self.index.d,
                        'suggestion': 'Re-ingest documents with consistent embedding model'
                    }
                )
            
            try:
                scores, idxs = self.index.search(q, k)
                logger.debug(f"Vector search returned {len(idxs[0])} results")
            except Exception as e:
                raise VectorStoreError(
                    message="FAISS search failed",
                    details={'k': k, 'query_shape': q.shape, 'error': str(e)}
                ) from e
            cur = self.conn.cursor()
            cur.execute("SELECT id, source, chunk_index, text FROM documents")
            rows = cur.fetchall()
            results = []
            for rank, i in enumerate(idxs[0]):
                if i < 0 or i >= len(rows):
                    continue
                rid, source, chunk_index, text = rows[i]
                results.append({"id": rid, "source": source, "chunk_index": chunk_index, "text": text, "score": float(scores[0][rank])})
            return results

    def count(self) -> int:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM documents")
            return cur.fetchone()[0]

    def index_size(self) -> int:
        with self._lock:
            if self.index is None:
                return 0
            return self.index.ntotal

    def reset(self):
        """Completely reset the store: delete DB + FAISS index and recreate empty structures."""
        with self._lock:
            try:
                self.conn.close()
            except Exception:
                pass
            if os.path.exists(self.db_path):
                try:
                    os.remove(self.db_path)
                except OSError:
                    pass
            if os.path.exists(self.index_path):
                try:
                    os.remove(self.index_path)
                except OSError:
                    pass
            # Recreate fresh DB connection + meta
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._ensure_meta()
            self.index = None
            self.dim = None

    def refresh(self):
        """Reload FAISS index from disk (use after external ingest process)."""
        with self._lock:
            if os.path.exists(self.index_path):
                self._load_index()


store = VectorStore()
