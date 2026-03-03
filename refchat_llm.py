import warnings
import os
warnings.filterwarnings("ignore")
import sys
import subprocess
import importlib.util

def install_if_missing():
    """Vérifie et installe les dépendances manquantes au démarrage."""
    dependencies = {
        "fitz": "pymupdf",
        "langchain": "langchain",
        "langchain_community": "langchain-community",
        "langchain_huggingface": "langchain-huggingface",
        "langchain_chroma": "langchain-chroma",
        "langchain_ollama": "langchain-ollama",
        "langchain_mistralai": "langchain-mistralai",
        "langchain_text_splitters": "langchain-text-splitters",
        "chromadb": "chromadb",
        "torch": "torch",
        "tqdm": "tqdm",
        "sentence_transformers": "sentence-transformers"
    }
    
    missing = []
    for module_name, package_name in dependencies.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)
    
    if missing:
        print(f"📦 Missing libraries detected : {', '.join(missing)}")
        print("⏳ Installation automatique en cours...")
        try:
            # 1. Installation
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            print("✅ Installation terminée. Relance automatique du script...")
            
            # 2. Relance le script actuel pour prendre en compte les nouveaux paquets
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            print(f"❌ Installation error : {e}")
            sys.exit(1)

# Lancer l'installation avant d'exécuter le reste du code
install_if_missing()

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
from refchat_config import EMBEDDING_MODEL, OLLAMA_TEMPERATURE, get as config_get
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
    try:
        from refchat_config import EMBEDDING_MODEL
        if "e5" in EMBEDDING_MODEL.lower():
            enriched = f"query: {enriched}"
    except ImportError:
        pass

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

def recuperer_articles_complets(db, query_enrichie, k_initial=8, max_articles=4, max_chunks_par_article=20):
    from langchain_core.documents import Document

    # CORRECTION : Utilisation de k_initial pour scanner plus d'abstracts au départ
    retriever_abstracts = db.as_retriever(
        search_kwargs={
            "k": k_initial, 
            "filter": {"section": "Abstract"} 
        }
    )
    docs_abstracts = retriever_abstracts.invoke(query_enrichie)

    articles_pertinents = []
    for doc in docs_abstracts:
        fname = doc.metadata.get("filename")
        if fname and fname not in articles_pertinents:
            articles_pertinents.append(fname)
            if len(articles_pertinents) == max_articles:
                break

    if not articles_pertinents:
        docs_fallback = db.similarity_search(query_enrichie, k=k_initial)
        for doc in docs_fallback:
            fname = doc.metadata.get("filename")
            if fname and fname not in articles_pertinents:
                articles_pertinents.append(fname)
                if len(articles_pertinents) == max_articles:
                    break

    if not articles_pertinents:
        return [], []

    # CORRECTION : Le nombre de chunks récupérés correspond maintenant à max_articles * max_chunks_par_article
    docs_finaux = db.similarity_search(
        query_enrichie,
        k=max_articles * max_chunks_par_article, 
        filter={"filename": {"$in": articles_pertinents}} 
    )

    articles_info = []
    chunks_tries = []

    for fname in articles_pertinents:
        chunks_du_fichier = [d for d in docs_finaux if d.metadata.get("filename") == fname]
        
        if not chunks_du_fichier:
            continue

        meta0 = chunks_du_fichier[0].metadata
        articles_info.append({
            "filename": fname,
            "auteur": meta0.get("auteur", ""),
            "annee": meta0.get("annee", ""),
            "titre": meta0.get("titre", ""),
            "nb_chunks": len(chunks_du_fichier),
            "score": len(chunks_du_fichier) 
        })

        chunks_du_fichier.sort(key=lambda x: _SECTIONS_ORDER.get(x.metadata.get("section", ""), 10))
        
        chunks_tries.extend(chunks_du_fichier[:max_chunks_par_article])

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

_SS_CACHE_DIR  = _Path(__file__).parent / ".semantic_cache"
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
     "5. CONTENT: Alternate between narrative prose and bullet points for numerical data.\n"
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
     "3. VISUAL: Titles in UPPERCASE + Emoji + Separator line: ────────────────────────. No '##'.\n"
     "4. RICHNESS: Preserve all technical measurements and geological structure names.\n"
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
