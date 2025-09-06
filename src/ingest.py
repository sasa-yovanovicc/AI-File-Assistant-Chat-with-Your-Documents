from __future__ import annotations
import argparse
import os
from typing import List, Any
try:
    from tqdm import tqdm
except ImportError:  # graceful fallback if dependency not installed
    def tqdm(it, **kwargs) -> Any:  # type: ignore
        return it
    print("[yellow]tqdm not installed – install with 'pip install tqdm' for progress bars.[/]")
from rich import print
from .config import CHUNK_SIZE, CHUNK_OVERLAP
from .chunking import chunk_text, build_docs
from .embeddings import embed_texts
from .vector_store import store
from .error_handler import handle_errors
from .exceptions import DocumentProcessingError, AIFileAssistantError

import PyPDF2
import docx  # python-docx

# Base supported extensions (lower-case). Added '.doc' placeholder (no native parsing; will warn & skip).
BASE_SUPPORTED_EXT = {".pdf", ".txt", ".docx", ".doc"}


def sniff_is_pdf(path: str) -> bool:
    try:
        with open(path, 'rb') as f:
            head = f.read(5)
        return head.startswith(b'%PDF-')
    except Exception:
        return False


@handle_errors(default_return="", exception_type=DocumentProcessingError)
def _read_pdf_file(path: str) -> str:
    """Extract text from PDF file."""
    text_parts: list[str] = []
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        if getattr(reader, 'is_encrypted', False):
            try:
                reader.decrypt("")
            except Exception:
                pass
        for page in reader.pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                text_parts.append("")
    return "\n".join(text_parts)

@handle_errors(default_return="", exception_type=DocumentProcessingError)
def _read_text_file(path: str, max_bytes: int | None = None) -> str:
    """Read plain text file with size limit."""
    if max_bytes is not None and os.path.getsize(path) > max_bytes:
        return ""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

@handle_errors(default_return="", exception_type=DocumentProcessingError)
def _read_docx_file(path: str) -> str:
    """Extract text from DOCX file."""
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)

@handle_errors(default_return="", exception_type=DocumentProcessingError)
def read_file(path: str, force_text: bool = False, max_bytes: int | None = None) -> str:
    """Return extracted text or empty string if unsupported/empty.

    force_text is preserved but by default we are strict: only .pdf / .txt / .docx.
    Legacy .doc is skipped with a warning (conversion requires external tools like antiword)."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == '.pdf' or sniff_is_pdf(path):
            return _read_pdf_file(path)
        elif ext == '.txt' or force_text:
            return _read_text_file(path, max_bytes)
        elif ext == '.docx':
            return _read_docx_file(path)
        elif ext == '.doc':
            print(f"[yellow]Skipping legacy .doc (convert manually to .docx): {path}[/]")
            return ""
    except Exception as e:  # pragma: no cover
        print(f"[red]Read error {path}: {e}[/]")
    return ""


def gather_files(root: str, include_ext: List[str], only_ext: List[str] | None = None) -> List[str]:
    """Return list of candidate file paths.

    Behaviour:
    * By default include base supported + any --include-ext.
    * If --only-ext is provided at least once, ignore base set and include ONLY those extensions.
      (Extensions are case-insensitive; with or without leading dot.)
    """
    allowed_extra = {e.lower() if e.startswith('.') else f".{e.lower()}" for e in include_ext}
    if only_ext:
        only = {e.lower() if e.startswith('.') else f".{e.lower()}" for e in only_ext}
    else:
        only = None
    all_files: List[str] = []
    for base, _dirs, files in os.walk(root):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if only is not None:
                if ext in only:
                    all_files.append(os.path.join(base, fname))
            else:
                if ext in BASE_SUPPORTED_EXT or ext in allowed_extra:
                    all_files.append(os.path.join(base, fname))
    return all_files


@handle_errors(default_return=[], exception_type=DocumentProcessingError)
def process_file(path: str, force_text: bool, max_bytes: int | None) -> dict:
    text = read_file(path, force_text=force_text, max_bytes=max_bytes)
    if not text.strip():
        return []
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    docs = build_docs(chunks, path)
    return docs


def _parse_arguments():
    """Parse command line arguments for ingestion."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Root folder to scan recursively')
    ap.add_argument('--batch', type=int, default=64, help='Embedding batch size')
    ap.add_argument('--exclude', action='append', default=[], help='Substring filter; if present in path skip (can repeat)')
    ap.add_argument('--include-ext', action='append', default=[], help='Additional extensions to include (e.g. .md)')
    ap.add_argument('--only-ext', action='append', default=[], help='Restrict ingestion ONLY to these extensions (repeatable, e.g. --only-ext pdf --only-ext docx)')
    ap.add_argument('--force-text', action='store_true', help='Treat unknown extensions as plain UTF-8 text')
    ap.add_argument('--max-bytes', type=int, default=2_000_000, help='Skip files larger than this when force-text (default 2MB)')
    ap.add_argument('--min-chars', type=int, default=40, help='Skip chunks/files with extracted text shorter than this')
    ap.add_argument('--debug', action='store_true', help='Verbose listing of discovered / skipped files')
    ap.add_argument('--reset', action='store_true', help='Reset (delete) existing vector index & DB before ingest')
    return ap.parse_args()

def _discover_and_filter_files(args) -> List[str]:
    """Discover files and apply exclusion filters."""
    if args.debug:
        print(f"[cyan]Scanning root:[/] {args.input}")
        import os as _os
        print(f"[cyan]Path exists?[/] {_os.path.exists(args.input)}  [cyan]Is dir?[/] {_os.path.isdir(args.input)}")
    
    files = gather_files(args.input, include_ext=args.include_ext, only_ext=args.only_ext if args.only_ext else None)
    if args.debug:
        print(f"[cyan]Discovered raw files count:[/] {len(files)}")
    
    # Apply substring exclusions
    if args.exclude:
        before = len(files)
        files = [f for f in files if not any(ex.lower() in f.lower() for ex in args.exclude)]
        if args.debug:
            print(f"[yellow]Excluded by patterns:[/] {before - len(files)} (remaining {len(files)})")
    
    print(f"[cyan]Found candidate files:[/] {len(files)}")
    if args.debug and len(files) == 0:
        print("[red]No candidates found.[/] Debug suggestions:\n - Verify extensions are .pdf / .txt / .docx\n - Check OneDrive sync status (cloud-only placeholders?)\n - PowerShell: Get-ChildItem -Recurse <path> -Include *.pdf,*.txt,*.docx | Measure")
    
    return files

def _process_files_batch(files: List[str], args) -> List[dict]:
    """Process files and extract documents with filtering."""
    all_docs = []
    counters = {"files": 0, "empty": 0, "errored": 0, "accepted": 0, "chunks": 0}
    
    for fp in tqdm(files, desc="Files"):
        if args.debug:
            print(f"[cyan]Processing:[/] {fp}")
        counters["files"] += 1
        try:
            docs = process_file(fp, force_text=args.force_text, max_bytes=args.max_bytes if args.force_text else None)
        except Exception:  # pragma: no cover
            counters["errored"] += 1
            continue
        # Filter tiny chunks
        filtered = [d for d in docs if len(d['text']) >= args.min_chars]
        if not filtered:
            counters["empty"] += 1
            continue
        counters["accepted"] += 1
        counters["chunks"] += len(filtered)
        all_docs.extend(filtered)

    print(f"[green]Summary:[/] processed={counters['files']} accepted_files={counters['accepted']} empty/too_small={counters['empty']} errors={counters['errored']} total_chunks={counters['chunks']}")
    return all_docs

def _embed_and_store_documents(all_docs: List[dict], batch_size: int) -> None:
    """Generate embeddings and store documents in vector store."""
    for i in tqdm(range(0, len(all_docs), batch_size), desc="Embeddings"):
        batch_docs = all_docs[i:i+batch_size]
        vectors = embed_texts([d['text'] for d in batch_docs])
        store.add(batch_docs, vectors)

def main() -> None:
    args = _parse_arguments()

    if args.reset:
        print('[yellow]Resetting existing vector store...[/]')
        store.reset()

    files = _discover_and_filter_files(args)

    all_docs = _process_files_batch(files, args)
    if not all_docs:
        print("[yellow]No documents to ingest after filtering.[/]")
        return

    _embed_and_store_documents(all_docs, args.batch)
    print(f"[green]Done.[/] Vector store now holds {store.count()} chunks (added {len(all_docs)}).")


if __name__ == '__main__':  # pragma: no cover
    main()
