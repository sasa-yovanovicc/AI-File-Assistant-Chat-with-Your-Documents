"""Document infrastructure implementations."""

from typing import List, Dict, Any, Optional
import os
from pathlib import Path
import hashlib
import json
from datetime import datetime

from ...domain.entities import Document, Chunk
from ...domain.repositories import DocumentRepository
from ...exceptions import DocumentError
from ...error_handler import handle_errors
from ...logging_config import get_logger

logger = get_logger(__name__)


class FileDocumentRepository(DocumentRepository):
    """File-based implementation of DocumentRepository."""
    
    def __init__(self, storage_path: str = "data/documents"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_path / "metadata.json"
        self._metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load document metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save document metadata to file."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self._metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Could not save metadata: {e}")
    
    def _generate_document_id(self, filename: str, content: str) -> str:
        """Generate unique ID for document."""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
        return f"{Path(filename).stem}_{content_hash}"
    
    @handle_errors(default_return=None, exception_type=DocumentError)
    def save_document(self, document: Document) -> str:
        """Save document to storage."""
        # Generate ID if not provided
        if not document.id:
            document.id = self._generate_document_id(document.source, document.content)
        
        # Save document content
        doc_path = self.storage_path / f"{document.id}.txt"
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(document.content)
        
        # Save metadata
        self._metadata[document.id] = {
            'filename': document.source,
            'file_type': document.metadata.get('file_type', 'unknown'),
            'file_size': document.metadata.get('file_size', len(document.content)),
            'upload_time': document.created_at.isoformat(),
            'metadata': document.metadata
        }
        self._save_metadata()
        
        logger.info(f"Saved document: {document.id}")
        return document.id
    
    @handle_errors(default_return=None, exception_type=DocumentError)
    def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve document by ID."""
        if document_id not in self._metadata:
            return None
        
        doc_path = self.storage_path / f"{document_id}.txt"
        if not doc_path.exists():
            logger.warning(f"Document file not found: {doc_path}")
            return None
        
        metadata = self._metadata[document_id]
        
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return Document(
            id=document_id,
            source=metadata['filename'],
            content=content,
            metadata=metadata.get('metadata', {}),
            created_at=datetime.fromisoformat(metadata['upload_time'])
        )
    
    @handle_errors(default_return=[], exception_type=DocumentError)
    def list_documents(self) -> List[Document]:
        """List all documents."""
        documents = []
        for doc_id in self._metadata.keys():
            doc = self.get_document(doc_id)
            if doc:
                documents.append(doc)
        return documents
    
    @handle_errors(default_return=False, exception_type=DocumentError)
    def delete_document(self, document_id: str) -> bool:
        """Delete document by ID."""
        if document_id not in self._metadata:
            return False
        
        # Remove file
        doc_path = self.storage_path / f"{document_id}.txt"
        if doc_path.exists():
            doc_path.unlink()
        
        # Remove metadata
        del self._metadata[document_id]
        self._save_metadata()
        
        logger.info(f"Deleted document: {document_id}")
        return True
    
    @handle_errors(default_return=[], exception_type=DocumentError)
    def save_chunks(self, chunks: List[Chunk]) -> List[str]:
        """Save document chunks."""
        chunk_ids = []
        for chunk in chunks:
            if not chunk.id:
                chunk.id = f"chunk_{hashlib.md5(chunk.content.encode()).hexdigest()[:8]}"
            
            chunk_path = self.storage_path / "chunks" / f"{chunk.id}.json"
            chunk_path.parent.mkdir(exist_ok=True)
            
            chunk_data = {
                'id': chunk.id,
                'document_id': chunk.document_id,
                'content': chunk.content,
                'chunk_index': chunk.chunk_index,
                'metadata': chunk.metadata
            }
            
            with open(chunk_path, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, indent=2, ensure_ascii=False)
            
            chunk_ids.append(chunk.id)
        
        logger.info(f"Saved {len(chunks)} chunks")
        return chunk_ids
    
    @handle_errors(default_return=[], exception_type=DocumentError)
    def get_chunks_by_document(self, document_id: str) -> List[Chunk]:
        """Get all chunks for a document."""
        chunks_dir = self.storage_path / "chunks"
        if not chunks_dir.exists():
            return []
        
        chunks = []
        for chunk_file in chunks_dir.glob("*.json"):
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                
                if chunk_data['document_id'] == document_id:
                    chunk = Chunk(
                        id=chunk_data['id'],
                        document_id=chunk_data['document_id'],
                        content=chunk_data['content'],
                        chunk_index=chunk_data['chunk_index'],
                        metadata=chunk_data.get('metadata', {})
                    )
                    chunks.append(chunk)
            except Exception as e:
                logger.warning(f"Could not load chunk {chunk_file}: {e}")
        
        return sorted(chunks, key=lambda x: x.chunk_index)
