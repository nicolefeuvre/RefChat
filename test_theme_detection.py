"""
Test de la détection sémantique de thèmes.
Lance avec : python test_theme_detection.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# ── Chargement config + DB ────────────────────────────────────────────────────
from refchat_config import DB_PATH, EMBEDDING_MODEL

print("⏳ Chargement de la base ChromaDB...")
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
db = Chroma(persist_directory=DB_PATH, embedding_function=emb)

# ── Récupération des thèmes ───────────────────────────────────────────────────
print("🔍 Récupération des thèmes depuis la DB...")
from refchat_llm import lister_themes, detecter_theme_query
themes = lister_themes(db)
print(f"\n📂 Thèmes disponibles ({len(themes)}) :")
for t in themes:
    print(f"   • {t}")

# ── Détection sémantique ──────────────────────────────────────────────────────
import numpy as np

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def detect_semantic(query, themes, threshold=0.55):
    if not themes:
        return None, {}
    # E5 prefix
    q = f"query: {query}" if "e5" in EMBEDDING_MODEL.lower() else query
    t_texts = [f"passage: {t}" if "e5" in EMBEDDING_MODEL.lower() else t for t in themes]

    q_emb   = emb.embed_query(q)
    t_embs  = emb.embed_documents(t_texts)

    scores = {t: cosine(q_emb, t_embs[i]) for i, t in enumerate(themes)}
    best_t = max(scores, key=scores.get)
    best_s = scores[best_t]

    return (best_t if best_s >= threshold else None), scores

# ── Questions test ────────────────────────────────────────────────────────────
queries = [
    "comment l'eau réagit avec les roches ultramafiques ?",
    "quels sont les mécanismes de production d'H2 naturel ?",
    "explique moi la serpentinisation",
    "production d'hydrogène par radiolyse de l'eau",
    "les failles et la tectonique des plaques",
    "quel est le rôle des bactéries dans les écosystèmes profonds ?",
    "montre moi des articles sur la géochimie isotopique",
    "tell me about natural hydrogen seeps",
    "où trouve-t-on de l'H2 dans la croûte continentale ?",
]

THRESHOLD = 0.55

print(f"\n{'─'*70}")
print(f"🧪 TEST DÉTECTION SÉMANTIQUE (seuil={THRESHOLD})")
print(f"{'─'*70}")

for q in queries:
    # Ancienne méthode
    old = detecter_theme_query(q, themes)
    # Nouvelle méthode
    new, scores = detect_semantic(q, themes, THRESHOLD)

    top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
    top3_str = " | ".join(f"{t[:25]}={s:.3f}" for t, s in top3)

    print(f"\n❓ {q}")
    print(f"   🔴 Ancienne : {old or '(aucun)'}")
    print(f"   🟢 Nouvelle : {new or f'(sous seuil → DB entière)'}")
    print(f"   📊 Top3     : {top3_str}")

print(f"\n{'─'*70}")
print("✅ Test terminé.")
