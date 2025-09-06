#!/usr/bin/env python3
from src.api import _try_with_fallback_k

print("🧹 TEST POSLE BRISANJA STARIH FAJLOVA")
result = _try_with_fallback_k('ko je Bert?', 3, llm=True, use_clean_arch=False)
print(f"✅ Legacy API: k_used={result['k_used']}, answer_len={len(result['answer'])}")

result2 = _try_with_fallback_k('ko je Bert?', 3, llm=True, use_clean_arch=True)
print(f"✅ Clean Architecture: k_used={result2['k_used']}, answer_len={len(result2['answer'])}")

print("\n🎉 SVI SISTEMI RADE SA CENTRALIZOVANOM STRUKTUROM!")

from src.config import DB_PATH, FAISS_INDEX_PATH
import os
print(f"\n📂 Aktuelne putanje:")
print(f"   DB exists: {os.path.exists(DB_PATH)} - {DB_PATH}")
print(f"   FAISS exists: {os.path.exists(FAISS_INDEX_PATH)} - {FAISS_INDEX_PATH}")
