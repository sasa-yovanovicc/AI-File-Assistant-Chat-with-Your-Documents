from __future__ import annotations
from typing import List, Dict
import re


def clean_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " \n ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def simple_sentence_split(text: str) -> List[str]:
    # naive split, can improve later
    parts = re.split(r'(?<=[.!?]) +', text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    sentences = simple_sentence_split(clean_text(text))
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sent in sentences:
        tokens = sent.split()
        if current_len + len(tokens) > chunk_size and current:
            chunks.append(" ".join(current))
            # overlap logic
            if overlap > 0:
                overlap_tokens = " ".join(current).split()[-overlap:]
                current = overlap_tokens.copy()
                current_len = len(current)
            else:
                current = []
                current_len = 0
        current.append(sent)
        current_len += len(tokens)
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_docs(chunks: List[str], source_path: str) -> List[Dict]:
    return [
        {"id": f"{source_path}::chunk_{i}", "text": c, "source": source_path, "chunk_index": i}
        for i, c in enumerate(chunks)
    ]
