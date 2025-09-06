from __future__ import annotations
"""Local LLM integration via Ollama HTTP API.

Dependencies: user must have Ollama installed and model pulled, e.g.:
  ollama pull mistral

We keep this lightweight: a single function generate_answer(model, prompt, stream=False)
that talks to http://localhost:11434/api/generate .
If Ollama is not running or request fails, we raise RuntimeError and caller can fallback.
"""
import json
import os
import http.client
from typing import Iterator, Optional

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")


def _make_ollama_request(payload: dict):
    """Make HTTP request to Ollama API and return connection and response."""
    conn = http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT, timeout=300)
    conn.request("POST", "/api/generate", body=json.dumps(payload), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        conn.close()
        raise RuntimeError(f"Ollama error status={resp.status} {resp.read()[:200]}")
    return conn, resp

def _parse_ollama_stream(resp: http.client.HTTPResponse, stream: bool):
    """Parse Ollama response stream."""
    if not stream:
        # Non-stream: accumulate until done
        full = []
        while True:
            line = resp.readline()
            if not line:
                break
            try:
                evt = json.loads(line.decode('utf-8'))
            except Exception:
                continue
            if 'response' in evt:
                full.append(evt['response'])
            if evt.get('done'):
                break
        return ''.join(full).strip()
    else:
        # Stream: return generator
        def gen():
            while True:
                line = resp.readline()
                if not line:
                    break
                try:
                    evt = json.loads(line.decode('utf-8'))
                except Exception:
                    continue
                if 'response' in evt:
                    yield evt['response']
                if evt.get('done'):
                    break
        return gen()

def generate_answer(prompt: str, model: Optional[str] = None, stream: bool = False) -> str | Iterator[str]:
    model = model or OLLAMA_MODEL
    payload = {"model": model, "prompt": prompt, "stream": stream}
    
    conn, resp = _make_ollama_request(payload)
    try:
        return _parse_ollama_stream(resp, stream)
    finally:
        conn.close()
