import warnings
import os
warnings.filterwarnings("ignore")

# ── OPTIMISATION CPU ──
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
try:
    import torch
    torch.set_num_threads(8)
except ImportError:
    pass

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# MODIFICATION ICI : Appel de refchat_config
from refchat_config import EMBEDDING_MODEL, OLLAMA_TEMPERATURE, PERSONAL_DATA, get as config_get
from refchat_config import get_ollama_num_ctx_local, get_ollama_temperature
DB_PATH = config_get('db_path', '')

try:
    from refchat_config import MISTRAL_API_KEY
except ImportError:
    MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")

MOTS_CLES_RESUME = [
    "résumé", "resume", "résume", "résumer",
    "synthèse", "synthétise", "synthétiser", "fais un point",
    "que disent les articles", "que disent les auteurs",
    "présente les articles", "overview", "summarize", "summary"
]

MOTS_CLES_AUTEUR = [
    "articles de", "travaux de", "étude de", "papier de",
    "publication de", "article de", "que dit", "qu'écrit",
    "articles by", "work by", "paper by",
]

MOTS_CLES_REFERENCE = [
    "référence", "references", "cite", "cites", "citation",
    "donne moi les ref", "quelles références", "quels articles traitent",
    "quels articles parlent", "trouve les articles", "liste les articles",
    "which articles", "what articles",
]


# ── Ordre canonique des sections pour le tri ──────────────────────────────────
_SECTIONS_ORDER = {
    "Abstract": 0, "Introduction": 1, "Geological Setting": 2,
    "Methods": 3, "Results": 4, "Results and Discussion": 4,
    "Discussion": 5, "Conclusion": 6
}
_SECTIONS_PRIORITAIRES = {
    "Abstract": 0, "Introduction": 1,
    "Results and Discussion": 2, "Results": 2,
    "Conclusion": 3, "Geological Setting": 4,
    "Hydrogeology": 4, "Methods": 5,
}

def detecter_mode(query):
    q = query.lower()
    if any(mot in q for mot in MOTS_CLES_RESUME):
        return "resume"
    if any(mot in q for mot in MOTS_CLES_REFERENCE):
        return "reference"
    if any(mot in q for mot in MOTS_CLES_AUTEUR):
        return "auteur"
    return "question"

def extraire_nom_auteur(query):
    import re
    q = query.strip()
    for trigger in MOTS_CLES_AUTEUR:
        if trigger in q.lower():
            idx   = q.lower().find(trigger) + len(trigger)
            reste = q[idx:].strip()
            nom   = reste.split()[0].rstrip(".,?!") if reste else ""
            if nom:
                return nom.capitalize()
    mots_maj = re.findall(r'\b[A-ZÉÈÀÂÊÎÔÛÄËÏÖÜ][a-zéèàâêîôûäëïöü]{2,}\b', q)
    return mots_maj[0] if mots_maj else ""

def expand_query(query):
    expansions = {
        "hydrogène":        "hydrogen H2 dihydrogen native hydrogen natural hydrogen geological",
        "hydrogen":         "H2 dihydrogen hydrogène natif natural hydrogen geological",
        "h2":               "hydrogen dihydrogen hydrogène natif natural geological",
        "pyrénées":         "Pyrenees Pyrenean Iberian plate Axial Zone pyrenean thrust",
        "pyrenees":         "Pyrénées Pyrenean Iberian Axial Zone thrust fault",
        "serpentinisation": "serpentinization ophiolite ultramafic peridotite mantle",
        "serpentinization": "serpentinisation ophiolite ultramafic peridotite mantle",
        "gaz naturel":      "natural gas methane CH4 hydrocarbon",
        "faille":           "fault thrust fault strike-slip shear zone fracture",
        "fault":            "faille décrochement chevauchement shear zone fracture",
        "fluide":           "fluid brine water geothermal hydrothermal pore",
        "fluid":            "fluide eau saumure hydrothermal géothermal pore",
        "aquifère":         "aquifer groundwater chalk limestone karst porosity",
        "aquifer":          "aquifère eau souterraine craie calcaire karst porosité",
    }
    q_lower = query.lower()
    extras  = []
    for mot, expansion in expansions.items():
        if mot in q_lower:
            extras.append(expansion)
    enriched = query + " " + " ".join(extras) if extras else query

    # E5 exige le préfixe "query: " pour les requêtes
    if "e5" in EMBEDDING_MODEL.lower():
        enriched = f"query: {enriched}"

    return enriched

def format_docs(docs):
    parties = []
    for doc in docs:
        section  = doc.metadata.get("section", "?")
        filename = doc.metadata.get("filename",
                   doc.metadata.get("source", "?").split("\\")[-1].split("/")[-1])
        auteur   = doc.metadata.get("auteur", "")
        annee    = doc.metadata.get("annee", "")
        if auteur and annee:
            entete = f"[Article : {filename} | Auteur : {auteur} | Année : {annee} | Section : {section}]"
        else:
            entete = f"[Article : {filename} | Section : {section}]"
        parties.append(f"{entete}\n{doc.page_content}")
    return "\n\n---\n\n".join(parties)

def chercher_par_auteur(db, nom_auteur):
    from langchain_core.documents import Document
    nom_lower = nom_auteur.lower()
    total     = db._collection.count()

    tous_docs  = []
    tous_metas = []
    for offset in range(0, total, 2000):
        r = db._collection.get(
            limit=2000, offset=offset,
            include=["documents", "metadatas"]
        )
        tous_docs.extend(r["documents"])
        tous_metas.extend(r["metadatas"])

    par_article = {}
    for doc, meta in zip(tous_docs, tous_metas):
        champs_meta = " ".join([
            meta.get("auteur",   ""),
            meta.get("auteurs",  ""),
            meta.get("filename", ""),
        ]).lower()
  
        if nom_lower in champs_meta:
            filename = meta.get("filename", "?")
            if filename not in par_article:
                par_article[filename] = []
            par_article[filename].append((doc, meta))

    if not par_article:
        return [], []

    articles_info  = []
    chunks_selects = []

    for filename, chunks in par_article.items():
        meta0 = chunks[0][1]
        articles_info.append({
            "filename":  filename,
            "auteur":    meta0.get("auteur", ""),
            "annee":     meta0.get("annee", ""),
            "titre":     meta0.get("titre", ""),
            "nb_chunks": len(chunks),
        })
        chunks_tries = sorted(
            chunks,
            key=lambda x: _SECTIONS_PRIORITAIRES.get(x[1].get("section", ""), 99)
        )
        nb_total  = len(chunks_tries)
        nb_garder = max(2, min(6, nb_total // 4))
        pas       = max(1, nb_total // nb_garder)
        selec     = [chunks_tries[i] for i in range(0, nb_total, pas)][:nb_garder]
        for doc_text, meta in selec:
            chunks_selects.append(Document(page_content=doc_text, metadata=meta))

    return articles_info, chunks_selects



# ── THEMES ────────────────────────────────────────────────────────────────────

def lister_themes(db):
    try:
        total = db._collection.count()
        themes = set()
        for offset in range(0, total, 2000):
            r = db._collection.get(limit=2000, offset=offset, include=["metadatas"])
            for meta in r["metadatas"]:
                t = meta.get("theme", "")
                if t:
                    themes.add(t)
        return sorted(themes)
    except Exception:
        return []


def recuperer_articles_par_theme(db, theme, max_articles=6, max_chunks_par_article=10):
    from langchain_core.documents import Document
    theme_lower = theme.lower()
    total = db._collection.count()
    tous_docs, tous_metas = [], []
    for offset in range(0, total, 2000):
        r = db._collection.get(limit=2000, offset=offset, include=["documents", "metadatas"])
        tous_docs.extend(r["documents"])
        tous_metas.extend(r["metadatas"])
    par_article = {}
    for doc, meta in zip(tous_docs, tous_metas):
        t = meta.get("theme", "")
        if theme_lower in t.lower():
            fname = meta.get("filename", "?")
            if fname not in par_article:
                par_article[fname] = []
            par_article[fname].append((doc, meta))
    if not par_article:
        return [], []
    articles_info = []
    chunks_selects = []
    for filename, chunks in list(par_article.items())[:max_articles]:
        meta0 = chunks[0][1]
        articles_info.append({
            "filename": filename,
            "auteur":   meta0.get("auteur", ""),
            "annee":    meta0.get("annee", ""),
            "titre":    meta0.get("titre", ""),
            "nb_chunks": len(chunks),
            "theme":    meta0.get("theme", ""),
        })
        chunks_tries = sorted(chunks, key=lambda x: _SECTIONS_PRIORITAIRES.get(x[1].get("section", ""), 99))
        nb_garder = max(2, min(max_chunks_par_article, len(chunks_tries)))
        pas = max(1, len(chunks_tries) // nb_garder)
        selec = [chunks_tries[i] for i in range(0, len(chunks_tries), pas)][:nb_garder]
        for doc_text, meta in selec:
            chunks_selects.append(Document(page_content=doc_text, metadata=meta))
    return articles_info, chunks_selects


def detecter_theme_query(query, themes_disponibles):
    if not themes_disponibles:
        return None
    import re
    patterns = [r"th[e\xe8]me\s+\S*([^\s?!.]+)", r"cat[e\xe9]gorie\s+\S*([^\s?!.]+)"]
    q_lower = query.lower()
    for pat in patterns:
        m = re.search(pat, q_lower)
        if m:
            candidate = m.group(1).strip()
            for t in themes_disponibles:
                if candidate[:10].lower() in t.lower() or t.lower()[:10] in candidate.lower():
                    return t
    for t in themes_disponibles:
        first_word = t.split(" - ")[0].lower()
        if len(first_word) > 4 and first_word in q_lower:
            return t
    return None

MAX_CHUNKS_THESIS = 5   # plafond chunks pour thèses (vs max_chunks_par_article pour articles)
MAX_THESIS_SLOTS  = 1   # nb max de thèses parmi les articles sélectionnés

# ── BM25 HYBRID SEARCH ────────────────────────────────────────────────────────

import pickle as _pickle
import time   as _time_bm25
from pathlib import Path as _PathBM25

_BM25_CACHE     = PERSONAL_DATA / "bm25_index.pkl"
_TRACKING_FILE  = PERSONAL_DATA / "refchat_index_db.json"


def charger_bm25(db, bm25_cache=None, tracking_file=None):
    """Build or load BM25 retriever from all ChromaDB chunks.

    Uses a .pkl cache to avoid rebuilding at every startup (~8-15s).
    Automatically rebuilds if refchat_index_db.json is newer than the cache
    (i.e. after a re-indexation).
    Returns None if rank-bm25 is not installed.
    """
    bm25_cache    = _PathBM25(bm25_cache    or _BM25_CACHE)
    tracking_file = _PathBM25(tracking_file or _TRACKING_FILE)

    # ── 1. Try loading from cache ──────────────────────────────────────────
    if bm25_cache.exists() and tracking_file.exists():
        if bm25_cache.stat().st_mtime > tracking_file.stat().st_mtime:
            try:
                with open(bm25_cache, "rb") as f:
                    retriever = _pickle.load(f)
                print(f"✅ BM25 index loaded from cache ({bm25_cache.name})")
                return retriever
            except Exception as e:
                print(f"⚠️  BM25 cache corrupted ({e}), rebuilding…")

    # ── 2. Check dependency ────────────────────────────────────────────────
    try:
        from langchain_community.retrievers import BM25Retriever
    except ImportError:
        print("⚠️  BM25 unavailable — install: pip install rank-bm25")
        return None

    # ── 3. Build from ChromaDB ─────────────────────────────────────────────
    total = db._collection.count()
    if total == 0:
        print("⚠️  BM25: database is empty, skipping.")
        return None

    print(f"⏳ Building BM25 index from {total} chunks…")
    t0 = _time_bm25.time()

    tous_docs, tous_metas = [], []
    for offset in range(0, total, 2000):
        r = db._collection.get(limit=2000, offset=offset, include=["documents", "metadatas"])
        tous_docs.extend(r["documents"])
        tous_metas.extend(r["metadatas"])

    from langchain_core.documents import Document as _Doc
    documents = [_Doc(page_content=d, metadata=m) for d, m in zip(tous_docs, tous_metas)]

    retriever = BM25Retriever.from_documents(documents, k=20)

    # ── 4. Save to disk ────────────────────────────────────────────────────
    try:
        with open(bm25_cache, "wb") as f:
            _pickle.dump(retriever, f)
        print(f"✅ BM25 index built ({len(documents)} chunks, {_time_bm25.time()-t0:.1f}s) — cached to {bm25_cache.name}")
    except Exception as e:
        print(f"⚠️  BM25 cache save failed: {e}")

    return retriever


def _rrf_select_articles(docs_dense, docs_bm25, n_candidates, bm25_weight, rrf_k=60):
    """Reciprocal Rank Fusion: merge dense and BM25 ranked doc lists.

    Returns top n_candidates filenames by combined RRF score.
    Thesis slot limits and final max_articles selection are applied by the caller.
    Score per article = (1-bm25_weight)*dense_rrf + bm25_weight*bm25_rrf
    """
    dense_scores = {}
    for rank, doc in enumerate(docs_dense):
        fname = doc.metadata.get("filename")
        if fname:
            dense_scores[fname] = dense_scores.get(fname, 0) + 1 / (rank + rrf_k)

    bm25_scores = {}
    for rank, doc in enumerate(docs_bm25):
        fname = doc.metadata.get("filename")
        if fname:
            bm25_scores[fname] = bm25_scores.get(fname, 0) + 1 / (rank + rrf_k)

    all_fnames = set(dense_scores) | set(bm25_scores)
    combined = {
        fname: (1 - bm25_weight) * dense_scores.get(fname, 0)
              + bm25_weight       * bm25_scores.get(fname, 0)
        for fname in all_fnames
    }

    return sorted(combined, key=lambda f: combined[f], reverse=True)[:n_candidates]


def recuperer_articles_complets(db, query_enrichie, bm25_retriever=None, bm25_weight=0.3,
                                 k_initial=8, max_articles=4, max_chunks_par_article=20,
                                 theme_filter=None, reranker=None):
    # Build optional theme pre-filter for ChromaDB
    # When a theme is detected, restrict the vector search to that theme's chunks only
    chroma_filter_abstract = {"section": "Abstract"}
    chroma_filter_fulltext = {"section": "Full text"}   # covers GROBID-fallback docs
    chroma_filter_full     = None
    if theme_filter:
        chroma_filter_abstract = {"$and": [{"section": {"$eq": "Abstract"}}, {"theme": {"$eq": theme_filter}}]}
        chroma_filter_fulltext = {"$and": [{"section": {"$eq": "Full text"}}, {"theme": {"$eq": theme_filter}}]}
        chroma_filter_full     = {"theme": {"$eq": theme_filter}}

    # Large-k search across ALL candidates before selecting top N.
    # This ensures globally-best articles are ranked, not just the first k found.
    # "Full text" search catches documents indexed without GROBID (no "Abstract" chunk).
    k_candidates   = max(k_initial * 15, 150)
    docs_abstracts = db.similarity_search(query_enrichie, k=k_candidates, filter=chroma_filter_abstract)
    docs_fulltext  = db.similarity_search(query_enrichie, k=k_candidates, filter=chroma_filter_fulltext)

    # Merge: build per-filename rank map (lower = more relevant).
    # Abstract hits take priority; Full text fills the gaps for fallback-indexed docs.
    rank_map      = {}  # fname → best rank position
    doc_types_map = {}  # fname → doc_type
    repr_docs     = {}  # fname → one representative doc (for RRF input)
    ordered_fnames = []

    for rank, doc in enumerate(docs_abstracts):
        fname = doc.metadata.get("filename")
        if not fname:
            continue
        if fname not in rank_map:
            rank_map[fname]      = rank
            doc_types_map[fname] = doc.metadata.get("doc_type", "article")
            repr_docs[fname]     = doc
            ordered_fnames.append(fname)

    fulltext_offset = len(docs_abstracts)
    for rank, doc in enumerate(docs_fulltext):
        fname = doc.metadata.get("filename")
        if not fname or fname in rank_map:
            continue
        rank_map[fname]      = fulltext_offset + rank
        doc_types_map[fname] = doc.metadata.get("doc_type", "article")
        repr_docs[fname]     = doc
        ordered_fnames.append(fname)

    # Sort all unique candidates by best relevance rank
    sorted_fnames = sorted(ordered_fnames, key=lambda f: rank_map[f])

    # Number of candidates to feed the reranker (4× final target, min 20)
    n_candidates = min(max(max_articles * 4, 20), len(sorted_fnames))

    if bm25_retriever is not None:
        # ── Hybrid: BM25 + Dense via RRF ──────────────────────────────────
        bm25_retriever.k = k_initial * 4
        docs_bm25 = bm25_retriever.invoke(query_enrichie)
        if theme_filter:
            docs_bm25 = [d for d in docs_bm25 if d.metadata.get("theme", "") == theme_filter]
        # Supplement repr_docs / doc_types_map with BM25-only hits
        for doc in docs_bm25:
            fname = doc.metadata.get("filename")
            if fname and fname not in repr_docs:
                repr_docs[fname]     = doc
                doc_types_map[fname] = doc.metadata.get("doc_type", "article")
        # One representative doc per file (dense-ranked) feeds the RRF scorer
        docs_dense_repr  = [repr_docs[f] for f in sorted_fnames if f in repr_docs]
        candidate_fnames = _rrf_select_articles(docs_dense_repr, docs_bm25, n_candidates, bm25_weight)
    else:
        # ── Dense uniquement ───────────────────────────────────────────────
        candidate_fnames = sorted_fnames[:n_candidates]

    # ── Cross-encoder reranking (optional) ────────────────────────────────
    if reranker is not None and len(candidate_fnames) > max_articles:
        fnames_to_rank = [f for f in candidate_fnames if f in repr_docs]
        pairs  = [(query_enrichie, repr_docs[f].page_content) for f in fnames_to_rank]
        scores = reranker.predict(pairs)
        candidate_fnames = [f for _, f in sorted(zip(scores, fnames_to_rank), reverse=True)]

    # ── Sélection finale : thesis slot limit + top max_articles ───────────
    articles_pertinents = []
    nb_thesis_selected  = 0
    for fname in candidate_fnames:
        doc_type = doc_types_map.get(fname, "article")
        if doc_type == "thesis" and nb_thesis_selected >= MAX_THESIS_SLOTS:
            continue
        if doc_type == "thesis":
            nb_thesis_selected += 1
        articles_pertinents.append(fname)
        if len(articles_pertinents) == max_articles:
            break

    if not articles_pertinents:
        return [], []

    # Final chunk retrieval — scoped to detected theme if active
    final_filter = {"filename": {"$in": articles_pertinents}}
    if chroma_filter_full:
        final_filter = {"$and": [{"filename": {"$in": articles_pertinents}}, {"theme": {"$eq": theme_filter}}]}

    docs_finaux = db.similarity_search(
        query_enrichie,
        k=max_articles * max_chunks_par_article,
        filter=final_filter
    )

    articles_info = []
    chunks_tries = []

    for fname in articles_pertinents:
        chunks_du_fichier = [d for d in docs_finaux if d.metadata.get("filename") == fname]
        
        if not chunks_du_fichier:
            continue

        meta0    = chunks_du_fichier[0].metadata
        doc_type = meta0.get("doc_type", "article")
        articles_info.append({
            "filename": fname,
            "auteur":   meta0.get("auteur", ""),
            "annee":    meta0.get("annee", ""),
            "titre":    meta0.get("titre", ""),
            "nb_chunks": len(chunks_du_fichier),
            "score":    len(chunks_du_fichier),
            "doc_type": doc_type,
        })

        chunk_limit = MAX_CHUNKS_THESIS if doc_type == "thesis" else max_chunks_par_article
        chunks_du_fichier.sort(key=lambda x: _SECTIONS_ORDER.get(x.metadata.get("section", ""), 10))

        chunks_tries.extend(chunks_du_fichier[:chunk_limit])

    return articles_info, chunks_tries


# ══ EXTRACTION DE MOTS-CLÉS VIA LLM ══════════════════════════════════════════

def extraire_mots_cles_llm(query, llm):
    """
    Utilise le LLM pour extraire des mots-clés scientifiques en anglais
    depuis une question en langage naturel (toute langue).
    Retourne une chaîne de mots-clés propre, prête pour Semantic Scholar.
    Fallback sur la query originale en cas d'échec.
    """
    prompt = (
        "Your task is to extract scientific keywords from the following user question.\n"
        "Rules:\n"
        "- Output ONLY the keywords, separated by spaces, nothing else\n"
        "- Translate everything to English\n"
        "- Remove conversational words (I want, tell me, explain, what is, etc.)\n"
        "- Keep only scientifically meaningful terms (max 6 keywords)\n"
        "- Do not add any explanation or punctuation\n\n"
        f"Question: {query}\n"
        "Keywords:"
    )
    try:
        result = llm.invoke(prompt)
        # Compatibilité LangChain : result peut être str ou objet avec .content
        keywords = result.content if hasattr(result, "content") else str(result)
        keywords = keywords.strip().splitlines()[0].strip()  # Prendre uniquement la 1ère ligne
        if keywords:
            print(f"🔑 Mots-clés extraits : « {keywords} »")
            return keywords
    except Exception as e:
        print(f"⚠️  Extraction mots-clés LLM échouée ({e}), fallback query originale")
    return query


# ══ RECHERCHE WEB — SEMANTIC SCHOLAR ══════════════════════════════════════════
# Stratégie anti rate-limit (3 couches) :
#   1. Cache disque 24h  → évite les doublons de requêtes
#   2. Délai minimum 2s  → respecte la limite 1 req/s de l'API publique
#   3. Retry sur 429     → récupération automatique si on dépasse quand même
# Optionnel : clé API gratuite (semantic_scholar_api_key dans refchat_config)
#   → https://www.semanticscholar.org/product/api#api-key-form

import hashlib as _hashlib
import json    as _json_ss
import time    as _time_ss
from pathlib import Path as _Path

_SS_CACHE_DIR  = PERSONAL_DATA / "semantic_cache"
_SS_CACHE_TTL  = 86400   # 24 heures en secondes
_SS_MIN_DELAY  = 3.0     # secondes minimum entre deux appels réseau (augmenté de 2→3)
_SS_LAST_CALL  = 0.0     # timestamp du dernier appel (variable globale module)
_SS_RETRY_DELAYS = [10, 20, 40]  # backoff exponentiel : 3 tentatives

def _ss_cache_key(query, limit):
    """Clé de cache déterministe basée sur query + limit."""
    raw = f"{query.strip().lower()}|{limit}"
    return _hashlib.md5(raw.encode()).hexdigest()

def _ss_cache_get(key):
    """Retourne les données cachées si elles existent et sont fraîches."""
    _SS_CACHE_DIR.mkdir(exist_ok=True)
    f = _SS_CACHE_DIR / f"{key}.json"
    if f.exists() and (_time_ss.time() - f.stat().st_mtime) < _SS_CACHE_TTL:
        try:
            return _json_ss.loads(f.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None

def _ss_cache_set(key, data):
    """Sauvegarde les données en cache."""
    _SS_CACHE_DIR.mkdir(exist_ok=True)
    (_SS_CACHE_DIR / f"{key}.json").write_text(
        _json_ss.dumps(data, ensure_ascii=False), encoding="utf-8"
    )

def chercher_semantic_scholar(query, limit=5):
    """
    Interroge Semantic Scholar avec 3 couches de protection anti rate-limit :
    cache 24h + délai minimum 2s + retry sur 429.
    Supporte une clé API optionnelle (semantic_scholar_api_key dans config).
    Retourne (docs, nb_total).
    """
    global _SS_LAST_CALL
    import requests
    from langchain_core.documents import Document

    # ── 1. Strip du préfixe E5 "query:" ──
    query_clean = query.strip()
    if query_clean.lower().startswith("query:"):
        query_clean = query_clean[6:].strip()

    # ── 2. Vérification du cache ──
    cache_key    = _ss_cache_key(query_clean, limit)
    cached       = _ss_cache_get(cache_key)
    if cached is not None:
        print(f"🌐 Semantic Scholar [CACHE] : {len(cached['docs'])} abstracts pour « {query_clean[:50]} »")
        docs_web = [
            Document(page_content=d["content"], metadata=d["metadata"])
            for d in cached["docs"]
        ]
        return docs_web, cached["nb_total"]

    # ── 3. Délai minimum entre requêtes ──
    elapsed = _time_ss.time() - _SS_LAST_CALL
    if elapsed < _SS_MIN_DELAY:
        wait = _SS_MIN_DELAY - elapsed
        print(f"⏳ Semantic Scholar : attente {wait:.1f}s (rate limit préventif)…")
        _time_ss.sleep(wait)

    # ── 4. Clé API optionnelle ──
    try:
        from refchat_config import get as _cfg_get
        ss_api_key = _cfg_get("semantic_scholar_api_key", "")
    except Exception:
        ss_api_key = ""

    headers = {"x-api-key": ss_api_key} if ss_api_key else {}

    url    = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query":  query_clean,
        "limit":  limit,
        "fields": "title,authors,year,abstract,url,externalIds"
    }

    def _do_request():
        return requests.get(url, params=params, headers=headers, timeout=12)

    try:
        _SS_LAST_CALL = _time_ss.time()
        response      = _do_request()

        # ── 5. Retry avec backoff exponentiel sur 429 ──
        if response.status_code == 429:
            for attempt, wait_sec in enumerate(_SS_RETRY_DELAYS, start=1):
                print(f"⚠️  Semantic Scholar 429 — tentative {attempt}/{len(_SS_RETRY_DELAYS)}, attente {wait_sec}s…")
                _time_ss.sleep(wait_sec)
                _SS_LAST_CALL = _time_ss.time()
                response = _do_request()
                if response.status_code != 429:
                    break
            else:
                print("❌ Semantic Scholar : rate-limit persistant après 3 tentatives.")
                print("💡 Conseil : obtenez une clé API gratuite sur https://www.semanticscholar.org/product/api")
                return [], 0

        if response.status_code != 200:
            print(f"⚠️  Semantic Scholar HTTP {response.status_code}")
            return [], 0

        data     = response.json()
        nb_total = data.get("total", 0)
        docs_web = []
        cache_docs = []

        for paper in data.get("data", []):
            titre = paper.get("title", "")
            if not titre:
                continue
            abstract   = paper.get("abstract") or "[Abstract non disponible via l'API — consultez le lien source]"
            doi        = (paper.get("externalIds") or {}).get("DOI", "")
            url_papier = paper.get("url") or (f"https://doi.org/{doi}" if doi else "")
            content    = f"TITRE: {titre}\nABSTRACT: {abstract}"
            metadata   = {
                "filename": "Semantic Scholar",
                "auteur":   paper["authors"][0]["name"] if paper.get("authors") else "Inconnu",
                "annee":    str(paper.get("year") or "n.d."),
                "section":  "Web Search",
                "source":   url_papier,
                "titre":    titre,
                "has_abstract": bool(paper.get("abstract")),
            }
            docs_web.append(Document(page_content=content, metadata=metadata))
            cache_docs.append({"content": content, "metadata": metadata})

        nb_avec_abstract = sum(1 for d in cache_docs if d["metadata"].get("has_abstract"))

        # ── 6. Mise en cache du résultat ──
        _ss_cache_set(cache_key, {"docs": cache_docs, "nb_total": nb_total})

        src = "avec clé API" if ss_api_key else "sans clé API"
        print(f"🌐 Semantic Scholar [{src}] : {len(docs_web)} articles ({nb_avec_abstract} avec abstract, total dispo : {nb_total:,}) pour « {query_clean[:50]} »")
        return docs_web, nb_total

    except requests.exceptions.Timeout:
        print("❌ Semantic Scholar : timeout (>12s)")
    except Exception as e:
        print(f"❌ Semantic Scholar error : {e}")
    return [], 0


def vider_vram_ollama():
    import urllib.request, json
    try:
        req = urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=3)
        data = json.loads(req.read())
        for model in data.get("models", []):
            nom = model.get("name", "")
            if nom:
                payload = json.dumps({
                    "model": nom,
                    "keep_alive": 0
                }).encode()
                req2 = urllib.request.Request(
                    "http://127.0.0.1:11434/api/generate",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                urllib.request.urlopen(req2, timeout=5)
    except Exception:
        pass 

def _hw(key: str, fallback: int) -> int:
    """Read a hardware param saved by the user via the web UI. Falls back to a safe default."""
    try:
        v = config_get(key, None)
        return int(v) if v is not None and str(v).strip() != "" else fallback
    except Exception:
        return fallback


def charger_llm(choix):
    vider_vram_ollama()

    if choix == "api":
        if not MISTRAL_API_KEY:
            raise ValueError(
                "Mistral API key missing.\n"
                "Add it to config.py:\n"
                "   MISTRAL_API_KEY = 'your_key_here'\n"
                "Get your key at: https://console.mistral.ai/"
            )
        try:
            from langchain_mistralai import ChatMistralAI
        except ImportError:
            raise ImportError("Install the package: pip install langchain-mistralai")
        llm = ChatMistralAI(
            model="mistral-large-latest",
            api_key=MISTRAL_API_KEY,
            # num_ctx et temperature non applicables : tourne sur les serveurs Mistral
            max_tokens=8096,
        )
        return llm, "Mistral API (mistral-large-latest)"

    elif choix == "mixtral":
        from langchain_ollama import OllamaLLM
        t = _hw("num_thread", 8); g = _hw("num_gpu", 99); b = _hw("num_batch", 512)
        print(f"\u2699\ufe0f  Mixtral  threads={t}  gpu_layers={g}  batch={b}")
        llm = OllamaLLM(
            model="mixtral",
            temperature=get_ollama_temperature(),
            num_ctx=get_ollama_num_ctx_local(),
            max_tokens=4096,
            num_gpu=g, num_thread=t, num_batch=b,
        )
        return llm, "Mixtral 8x7B (local)"

    elif choix == "mistral-light":
        from langchain_ollama import OllamaLLM
        t = _hw("num_thread", 4); g = _hw("num_gpu", 99); b = _hw("num_batch", 512)
        print(f"\u2699\ufe0f  Mistral 7B Q4  threads={t}  gpu_layers={g}  batch={b}")
        llm = OllamaLLM(
            model="mistral:7b-instruct-q4_0",
            temperature=get_ollama_temperature(),
            num_ctx=get_ollama_num_ctx_local(),
            num_predict=4096,
            max_tokens=4096,
            num_gpu=g, num_thread=t, num_batch=b,
        )
        return llm, "Mistral 7B Q4 GPU \u26a1"

    else:
        from langchain_ollama import OllamaLLM
        t = _hw("num_thread", 4); g = _hw("num_gpu", 99); b = _hw("num_batch", 512)
        print(f"\u2699\ufe0f  Mistral 7B  threads={t}  gpu_layers={g}  batch={b}")
        llm = OllamaLLM(
            model="mistral",
            temperature=get_ollama_temperature(),
            num_ctx=get_ollama_num_ctx_local(),
            num_predict=4096,
            num_gpu=g, num_thread=t, num_batch=b,
        )
        return llm, "Mistral 7B (local)"


PROMPT_QUESTION = ChatPromptTemplate.from_messages([
    ("system",
     "RESPONSE LANGUAGE: ALWAYS respond in the same language as the user's question. "
     "French question → French answer. English question → English answer. Never deviate.\n\n"
     "You are a documentary research expert adopting the NotebookLM style: fluid, narrative, extremely rigorous.\n\n"
     "ABSOLUTE RULE — DOCUMENTS ARE GROUND TRUTH:\n"
     "If an excerpt comes from a non-scientific section (Acknowledgements, Remerciements, Table of Contents, Bibliography, Dedication), IGNORE it completely and do not use it in your answer."
     "Your answer MUST be based EXCLUSIVELY on the documents provided in the context. "
     "If a document explicitly states a finding, you MUST report it faithfully — even if it contradicts your training knowledge. "
     "NEVER deny, minimize or contradict any result that is written in the provided documents. "
     "Your training knowledge does NOT override the documents. When in doubt, trust the documents.\n\n"
     "WRITING RULES:\n"
     "1. INTRODUCTION: A natural prose paragraph setting the context before going into details.\n"
     "2. IN-TEXT CITATIONS: Integrate sources directly into your narrative as (Author et al., Year).\n"
     "3. EXHAUSTIVE COVERAGE: You are OBLIGATED to use information from EVERY author mentioned in the context.\n"
     "4. VISUAL STRUCTURE: Titles in UPPERCASE with thematic emoji. Underline EACH title with long dashes: ────────────────────────. NEVER use '##'.\n"
     "5. CONTENT: Write EXCLUSIVELY in continuous narrative prose. ABSOLUTE PROHIBITION on bullet points (-, *, •) anywhere in the response. Numbered lists (1. 2. 3.) are allowed ONLY for ranked steps or structured plans. Measurements and data must be integrated into sentences, not listed.\n"
     "6. STRICT SOURCES: Only cite authors explicitly present in the provided context. ABSOLUTE PROHIBITION on using training knowledge to add references. If an author is not in the context, they do not exist for this answer.\n"
     "7. WEB DISTINCTION: Only if an excerpt's section is 'Web Search', add '[Web Source]' after the citation. Never add it on your own initiative.\n"
     "8. END OF RESPONSE: Always finish with '📚 SOURCES USED' and list all references on a SINGLE horizontal line: (Author, Year ; Author, Year). ABSOLUTE PROHIBITION on bullet lists or line breaks between authors.\n\n"
     "Context:\n{context}"),
    ("human", "{input}"),
])

PROMPT_RESUME = ChatPromptTemplate.from_messages([
    ("system",
     "RESPONSE LANGUAGE: ALWAYS respond in the same language as the user's question. "
     "French question → French answer. English question → English answer. Never deviate.\n\n"
     "You are a documentary research expert. Produce a LARGE THEMATIC SYNTHESIS (Study Guide).\n\n"
     "ABSOLUTE RULE — DOCUMENTS ARE GROUND TRUTH:\n"
     "If an excerpt comes from a non-scientific section (Acknowledgements, Remerciements, Table of Contents, Bibliography, Dedication), IGNORE it completely and do not use it in your answer."
     "Your answer MUST be based EXCLUSIVELY on the documents provided in the context. "
     "If a document explicitly states a finding, report it faithfully — even if it contradicts your training knowledge. "
     "NEVER deny, minimize or contradict any result that is written in the provided documents. "
     "The documents are the ground truth. Your training knowledge does NOT override them.\n\n"
     "GOLDEN RULES:\n"
     "1. SYNTHESIS: Do not summarize article by article. Weave a logical narrative where ideas respond to each other.\n"
     "2. CITATIONS: Use the format (Author et al., Year) at the heart of your sentences to attribute findings.\n"
     "3. VISUAL: Titles in UPPERCASE + Emoji + Separator line: ────────────────────────. No '##'. Numbered sections (1. 2. 3.) are encouraged for structure.\n"
     "4. RICHNESS: Preserve all technical measurements and geological structure names.\n"
     "4b. PROSE ONLY: Write in continuous narrative prose. ABSOLUTE PROHIBITION on bullet points (-, *, •) anywhere. Data and measurements must be integrated into sentences, not listed.\n"
     "5. STRICT SOURCES: Only cite authors present in the provided context. ABSOLUTE PROHIBITION on adding references from training knowledge.\n"
     "6. WEB DISTINCTION: Only if an excerpt's section is 'Web Search', add '[Web Source]' after the citation. Never invent it.\n"
     "7. REFERENCES: Always finish with '📚 SOURCES USED' and group all articles on a SINGLE horizontal line: (Author, Year ; Author, Year). Strictly forbidden to use dashes or line breaks between sources.\n\n"
     "Context:\n{context}"),
    ("human", "{input}"),
])

PROMPT_REFERENCE = ChatPromptTemplate.from_messages([
    ("system",
     "RESPONSE LANGUAGE: ALWAYS respond in the same language as the user's question. "
     "French question → French answer. English question → English answer. Never deviate.\n\n"
     "You are an expert scientific research assistant.\n"
     "Each excerpt is prefixed by [Article: NAME | Author: AUTHOR | Year: YEAR | Section: SECTION].\n\n"
     "ABSOLUTE RULE — DOCUMENTS ARE GROUND TRUTH:\n"
     "If an excerpt comes from a non-scientific section (Acknowledgements, Remerciements, Table of Contents, Bibliography, Dedication), IGNORE it completely and do not use it in your answer."
     "Your answer MUST be based EXCLUSIVELY on the documents provided in the context. "
     "If a document explicitly states a finding, report it faithfully. "
     "NEVER deny, minimize or contradict any result that is written in the provided documents. "
     "The documents are the ground truth. Your training knowledge does NOT override them.\n\n"
     "STYLE — inspired by NotebookLM:\n"
     "• Start with a short introductory sentence announcing the results ('Among the available articles, X sources directly address this topic.').\n"
     "• Then list each relevant article in this format:\n\n"
     "   **Author (Year) — Title**\n"
     "   Describe in 2-3 fluid sentences why this article is relevant. Cite the section in parentheses. If the section is 'Web Search', add 🌐 **[Web Source]** at the end.\n\n"
     "• End with a short synthetic conclusion if several articles converge.\n"
     "• If no article addresses the topic, say so clearly in one sentence.\n"
     "• ABSOLUTE RULE: Only cite articles present in the context. No reference from training knowledge is allowed.\n\n"
     "Context:\n{context}"),
    ("human", "{input}"),
])
