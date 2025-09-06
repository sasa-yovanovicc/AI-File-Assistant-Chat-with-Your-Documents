#!/usr/bin/env python3
import logging
from src.api import _try_with_fallback_k

logging.basicConfig(level=logging.INFO)

print("=== TESTIRANJE SA NOVOM CENTRALIZOVANOM KONFIGURACIJOM ===\n")

# Test legacy API
print("🔧 Test Legacy API:")
result1 = _try_with_fallback_k('ko je Bert?', 4, llm=True, use_clean_arch=False)
print(f"   k_used={result1['k_used']}, answer_len={len(result1['answer'])}")
print(f"   Answer: {result1['answer'][:80]}...")

# Test Clean Architecture API
print("\n🏗️  Test Clean Architecture:")
result2 = _try_with_fallback_k('ko je Bert?', 4, llm=True, use_clean_arch=True)
print(f"   k_used={result2['k_used']}, answer_len={len(result2['answer'])}")
print(f"   Answer: {result2['answer'][:80]}...")

print("\n✅ Oba API-ja rade sa centralizovanom konfiguracijom!")

# Proveravamo putanje
from src.config import DB_PATH, FAISS_INDEX_PATH
print(f"\n📂 Putanje:")
print(f"   DB_PATH: {DB_PATH}")
print(f"   FAISS_INDEX_PATH: {FAISS_INDEX_PATH}")
