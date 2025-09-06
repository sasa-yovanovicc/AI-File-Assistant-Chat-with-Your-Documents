"""FAISS vector store implementation."""

from typing import List
import numpy as np
import sqlite3
import threading
import os
import faiss

from ...domain.entities import Chunk, SearchResult
from ...domain.repositories import VectorRepository
from ...config import UPLOADS_DIR
from ...exceptions import VectorStoreError
from ...error_handler import handle_errors
from ...logging_config import get_logger

logger = get_logger(__name__)


class FaissVectorStore(VectorRepository):
    """FAISS-based implementation of VectorRepository."""
    
    def __init__(self, db_path: str = None, index_path: str = None):
        # Use new Clean Architecture paths
        self.db_path = db_path or os.path.join(UPLOADS_DIR, "vector_db", "vectors.db")
        self.index_path = index_path or os.path.join(UPLOADS_DIR, "faiss", "faiss.index")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.RLock()
        self.index = None
        self.dim = None
        
        self._ensure_meta()
        if os.path.exists(self.index_path):
            self._load_index()
    
    def _ensure_meta(self) -> None:
        """Ensure metadata table exists."""
        cur = self.conn.cursor()
        # Use existing documents table structure for compatibility
        cur.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                source TEXT,
                chunk_index INTEGER,
                text TEXT
            );
        ''')
        self.conn.commit()
    
    def _load_index(self) -> None:
        """Load FAISS index from disk."""
        self.index = faiss.read_index(self.index_path)
        self.dim = self.index.d
    
    def _create_index(self, dim: int) -> None:
        """Create new FAISS index."""
        self.index = faiss.IndexFlatIP(dim)
        self.dim = dim
    
    @handle_errors(default_return=None, exception_type=VectorStoreError)
    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks with their embeddings to the vector store."""
        with self._lock:
            if self.index is None:
                self._create_index(embeddings.shape[1])
            
            if embeddings.shape[1] != self.dim:
                raise VectorStoreError(
                    message="Embedding dimension mismatch",
                    details={'expected': self.dim, 'got': embeddings.shape[1]}
                )
            
            # Store in documents table for compatibility
            cur = self.conn.cursor()
            for chunk in chunks:
                # Use chunk.metadata.get("source") or derive from document_id
                source = chunk.metadata.get("source", chunk.document_id)
                cur.execute('''
                    INSERT OR REPLACE INTO documents 
                    (id, source, chunk_index, text)
                    VALUES (?, ?, ?, ?)
                ''', (
                    chunk.id,
                    source, 
                    chunk.chunk_index,
                    chunk.content
                ))
            
            self.conn.commit()
            
            # Add to FAISS index
            self.index.add(embeddings)
            faiss.write_index(self.index, self.index_path)
    
    @handle_errors(default_return=[], exception_type=VectorStoreError)
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[SearchResult]:
        """Search for similar chunks using vector similarity."""
        with self._lock:
            if self.index is None:
                logger.warning("Search attempted on empty index")
                return []
            
            q = query_embedding.reshape(1, -1)
            
            if q.shape[1] != self.index.d:
                raise VectorStoreError(
                    message=f"Query vector dimension ({q.shape[1]}) does not match index dimension ({self.index.d})"
                )
            
            # FAISS search
            scores, indices = self.index.search(q, k)
            
            # Get metadata from documents table
            cur = self.conn.cursor()
            cur.execute('SELECT id, source, chunk_index, text FROM documents')
            rows = cur.fetchall()
            
            results = []
            for rank, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(rows):
                    continue
                
                row = rows[idx]
                
                # Create Chunk from documents table with source in metadata
                chunk = Chunk(
                    id=row[0],
                    document_id=row[1],  # source as document_id for compatibility
                    content=row[3],      # text as content
                    chunk_index=row[2],
                    start_position=0,    # Default value
                    end_position=len(row[3]),  # Text length
                    metadata={"source": row[1]}  # Add source to metadata
                )
                
                result = SearchResult(
                    chunk=chunk,
                    score=float(scores[0][rank]),
                    rank=rank
                )
                results.append(result)
            
            return results
    
    def count(self) -> int:
        """Get total number of chunks in the store."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM documents")
            return cur.fetchone()[0]
    
    def reset(self) -> None:
        """Clear all data from the vector store."""
        with self._lock:
            # Clear database
            cur = self.conn.cursor()
            cur.execute("DELETE FROM documents")
            self.conn.commit()
            
            # Clear FAISS index
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            
            self.index = None
            self.dim = None
