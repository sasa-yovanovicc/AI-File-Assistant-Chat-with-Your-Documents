from __future__ import annotations
from typing import List
import re
from .embeddings import embed_query
from .config import (
    OLLAMA_MODEL, USE_OPENAI, OPENAI_API_KEY, OPENAI_MODEL,
    RETRIEVAL_MIN_SCORE, KEYWORD_COVERAGE_THRESHOLD, STRICT_MODE_ENABLED,
    LEXICAL_COVERAGE_WEIGHT, SHORT_SENTENCE_PENALTY, MIN_ANSWER_SCORE,
    LEXICAL_RERANK_WEIGHT, MIN_DEFINITION_LENGTH, MAX_DEFINITION_LENGTH,
    DEFINITION_NAME_WEIGHT, MIN_KEYWORD_LENGTH, MAX_ANSWER_LENGTH,
    DEFAULT_RETRIEVAL_K, DEFAULT_ANSWER_K, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS
)
try:  # local LLM optional
    from .llm_local import generate_answer as _ollama_generate
except Exception:  # pragma: no cover
    _ollama_generate = None
try:  # OpenAI optional
    import openai
except Exception:  # pragma: no cover
    openai = None
from .vector_store import store
from rapidfuzz import fuzz

SYSTEM_PROMPT = (
    "Answer in an encyclopedic style using ONLY the provided context text chunks. Be precise, factual, and concise like a Wikipedia entry. If the answer is not present, reply: 'Not enough information in the local documents.'"
)

# Minimal similarity threshold (cosine) to accept a hit as relevant.
# Cosine similarities for unrelated chunks are often < 0.15; adjust if needed.
MIN_SCORE = RETRIEVAL_MIN_SCORE # base threshold (can be overridden per request)
STRICT_MODE = STRICT_MODE_ENABLED  # if True, enforce keyword coverage guard to reduce hallucinations
MIN_KEYWORD_COVERAGE = KEYWORD_COVERAGE_THRESHOLD  # proportion of distinct query keywords that must appear across contexts

# Minimal Serbian/English stopword subset for naive keyword scoring
STOPWORDS = {
    'je','sam','si','su','smo','the','a','and','i','u','na','za','da','to','od','se','of','in','koji','koja','koje','sa','kao','ali','pa','ili'
}

# Basic Serbian diacritic normalization map
_DIACRITICS = str.maketrans({'š':'s','č':'c','ć':'c','ž':'z','đ':'dj','Š':'S','Č':'C','Ć':'C','Ž':'Z','Đ':'Dj'})

def _norm(s: str) -> str:
    return s.translate(_DIACRITICS)

def _proper_names(question: str) -> List[str]:
    # Very naive: tokens starting with uppercase (Latin) and not stopwords
    tokens = re.findall(r'[A-Za-zÀ-žĐđČčĆćŠšŽž]+', question)
    res = []
    for t in tokens:
        if len(t) < 2:
            continue
        if t.lower() in STOPWORDS:
            continue
        if t[0].isupper():
            res.append(t)
    return res

def _sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def _keywords(q: str) -> List[str]:
    tokens = re.findall(r'[A-Za-zÀ-ž0-9]+', q.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > MIN_KEYWORD_LENGTH]

def extract_answer(question: str, contexts: List[str]) -> str:
    """Very small heuristic extractor: choose best scoring sentence among top contexts.

    Improvements:
    * Lexical score = (# keyword hits) + LEXICAL_COVERAGE_WEIGHT * coverage_ratio
    * Penalize sentences that are extremely short (< 30 chars)
    * Return fallback if nothing scores
    """
    kw = _keywords(question)
    if not kw:
        return contexts[0][:350]
    best_sent = None
    best_score = 0.0
    for ctx in contexts[:4]:  # top 4 chunks
        for s in _sentences(ctx):
            low = s.lower()
            hits = [k for k in kw if k in low]
            if not hits:
                continue
            coverage = len(set(hits)) / (len(set(kw)) or 1)
            score = len(hits) + LEXICAL_COVERAGE_WEIGHT * coverage
            if len(s) < 30:
                score -= SHORT_SENTENCE_PENALTY
            if score > best_score:
                best_score = score
                best_sent = s
    if best_sent and best_score > MIN_ANSWER_SCORE:
        return best_sent[:400]
    return contexts[0][:350]

def _definitional_extract(question: str, contexts: List[str]) -> str | None:
    names = _proper_names(question)
    if not names:
        return None
    name_vars = set()
    for n in names:
        name_vars.add(n)
        name_vars.add(_norm(n))
    q_lower = question.lower()
    is_def_q = q_lower.startswith('ko je ') or q_lower.startswith('who is ')
    if not is_def_q:
        return None
    best = None
    best_score = 0.0
    for ctx in contexts[:6]:
        for sent in _sentences(ctx):
            low = sent.lower()
            norm_low = _norm(sent.lower())
            if ' je ' not in low and ' is ' not in low:
                continue
            hit = False
            for v in name_vars:
                if v.lower() in norm_low:
                    hit = True
                    break
            if not hit:
                continue
            # Score: length penalty + position of ' je '
            length = len(sent)
            if length < MIN_DEFINITION_LENGTH or length > MAX_DEFINITION_LENGTH:
                continue
            pos = norm_low.find(' je ')
            if pos == -1:
                pos = norm_low.find(' is ')
            score = 1.0
            if pos != -1:
                score += max(0, 60 - pos) / 60  # earlier definition slightly better
            score += sum(1 for v in name_vars if v.lower() in norm_low) * DEFINITION_NAME_WEIGHT
            if score > best_score:
                best_score = score
                best = sent
    if best:
        return best[:400]
    return None

def build_prompt(question: str, contexts: List[str]) -> str:
    ctx_block = "\n\n".join(f"[DEO {i+1}]\n{c}" for i, c in enumerate(contexts))
    return f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{ctx_block}\n\nQuestion: {question}\nAnswer:"  # noqa


def _lexical_rerank(question: str, hits: List[dict]) -> List[dict]:
    """Re-rank by simple keyword overlap to push obviously relevant Serbian terms up."""
    kw = _keywords(question)
    if not kw:
        return hits
    for h in hits:
        text_low = h['text'].lower()
        overlap = sum(1 for k in kw if k in text_low)
        # blend semantic (h['score']) and lexical overlap
        h['blended'] = h['score'] + LEXICAL_RERANK_WEIGHT * overlap
    return sorted(hits, key=lambda x: x.get('blended', x['score']), reverse=True)

def retrieve(question: str, k: int = DEFAULT_RETRIEVAL_K):
    qv = embed_query(question)
    hits = store.search(qv, k=max(k, 8))  # pull a few more for reranking
    if not hits:
        return []
    hits = _lexical_rerank(question, hits)
    return hits[:k]

# For now, just echo context + pretend LLM step (LLM integration to add later)

def _clean_answer(ans: str) -> str:
    # Remove common hallucination preambles or code-like noise
    lines = [l for l in ans.splitlines() if l.strip()]
    if not lines:
        return ans.strip()
    # Drop lines that look like program steps from unrelated languages if majority are like that
    cleaned = []
    for l in lines:
        if re.match(r"^(var |int |for\s*\(|while\s*\(|function )", l):
            continue
        cleaned.append(l)
    ans2 = "\n".join(cleaned).strip()
    # Truncate if extremely long
    if len(ans2) > MAX_ANSWER_LENGTH:
        ans2 = ans2[:MAX_ANSWER_LENGTH] + '...'
    return ans2


def answer_question(question: str, k: int = DEFAULT_ANSWER_K, use_local_llm: bool = True) -> dict:
    hits = retrieve(question, k=k)
    contexts = [h['text'] for h in hits]
    prompt = build_prompt(question, contexts)

    kw_all = set(_keywords(question))
    # Compute keyword coverage across retrieved contexts
    coverage = 0.0
    if kw_all and contexts:
        concat_low = " \n ".join(contexts).lower()
        present = {kw for kw in kw_all if kw in concat_low}
        coverage = len(present) / max(1, len(kw_all))

    confidence = "high"
    reason = "ok"
    if not hits:
        answer = "Not enough information in the local documents."
        confidence = "none"
        reason = "no_hits"
    elif hits[0]['score'] < MIN_SCORE and coverage == 0:
        answer = "Not enough information in the local documents."
        confidence = "none"
        reason = "below_min_score_and_no_coverage"
    elif STRICT_MODE and kw_all and coverage < MIN_KEYWORD_COVERAGE:
        answer = "Not enough information in the local documents."
        confidence = "none"
        reason = "low_keyword_coverage"
    else:
        low_conf = hits[0]['score'] < MIN_SCORE
        if low_conf:
            confidence = "low"
            reason = f"score<{MIN_SCORE:.2f}"  # still try to extract
        # Build LLM prompt and choose provider
        if use_local_llm:
            ctx_joined = "\n\n".join(f"[DOC {i+1}]\n{c}" for i, c in enumerate(contexts[:6]))
            prompt = (
                "Answer in an encyclopedic style using ONLY the provided context. Be precise, factual, and concise like a Wikipedia entry. If the answer is missing respond exactly 'Not enough information in the local documents.'\n"
                f"Question: {question}\n\nContext:\n{ctx_joined}\n\nAnswer:"
            )
            
            # Try OpenAI first if configured, fallback to Ollama
            try:
                if USE_OPENAI and OPENAI_API_KEY and openai:
                    # Use OpenAI GPT for completion
                    openai.api_key = OPENAI_API_KEY
                    response = openai.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=OPENAI_MAX_TOKENS,
                        temperature=OPENAI_TEMPERATURE
                    )
                    gen = response.choices[0].message.content
                elif _ollama_generate is not None:
                    # Fallback to Ollama
                    gen = _ollama_generate(prompt, model=OLLAMA_MODEL, stream=False)
                else:
                    # No LLM available, use heuristic
                    gen = None
                
                if isinstance(gen, str) and gen.strip():
                    # Optional post-LLM guard: if LLM answer contains none of the keywords, fallback
                    cleaned = _clean_answer(gen)
                    low_ans = cleaned.lower()
                    if kw_all and not any(kw in low_ans for kw in kw_all):
                        answer = extract_answer(question, contexts) if not low_conf else extract_answer(question, contexts)
                        confidence = "low"
                        reason = "llm_no_keyword_overlap"
                    else:
                        answer = cleaned
                else:
                    answer = extract_answer(question, contexts)
            except Exception as e:
                # Fallback to heuristic if LLM fails
                answer = extract_answer(question, contexts)
                print(f"LLM error: {e}")  # for debugging
        else:
            # Try definitional pattern first (e.g., "Ko je <ime>?")
            defin = _definitional_extract(question, contexts)
            if defin:
                answer = defin
            else:
                answer = extract_answer(question, contexts)

    # Highlight matched keywords in sources (lightweight UI hint)
    kw = _keywords(question)
    if kw:
        for h in hits:
            txt = h['text']
            for kword in kw[:6]:  # limit to a few to avoid explosion
                pattern = re.compile(re.escape(kword), re.IGNORECASE)
                txt = pattern.sub(lambda m: f"**{m.group(0)}**", txt)
            h['preview'] = txt[:220]
    return {
        "question": question,
        "answer": answer,
        "prompt_preview": prompt[:500],
        "sources": hits,
        "kw_coverage": coverage,
        "confidence": confidence,
        "reason": reason,
    }

def fuzzy_keywords(question: str) -> list[str]:
    # Extract all words longer than 2 characters, ignore stopwords
    tokens = _keywords(question)
    # Normalize diacritics
    return [_norm(t) for t in tokens]

def extract_relevant_chunks(chunks, question, threshold=80):
    # Fuzzy match: use rapidfuzz for comparing keywords with chunk text
    keywords = fuzzy_keywords(question)
    hits = []
    for chunk in chunks:
        txt_norm = _norm(chunk['text'].lower())
        for kw in keywords:
            # Fuzzy match: score > threshold (e.g. 80/100)
            if any(fuzz.partial_ratio(kw, part) > threshold for part in txt_norm.split()):
                hits.append(chunk)
                break
    return hits if hits else chunks
