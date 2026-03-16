"""
RefChat — Thématisation automatique de la bibliothèque
=======================================================
Ce script analyse les embeddings déjà présents dans ChromaDB,
identifie des thèmes par clustering (BERTopic), et écrit le champ
`theme` dans les métadonnées de chaque chunk — SANS réindexer.

Usage CLI :
    python refchat_theme.py
    python refchat_theme.py --dry-run --topics 60 --show

API programmatique (utilisé par refchat_web.py) :
    from refchat_theme import run_clustering_preview, apply_theme_mapping
    preview = run_clustering_preview(db=STATE["db"], n_topics=40, min_docs=5, log_cb=...)
    apply_theme_mapping(preview["filename_to_theme"], db=STATE["db"], log_cb=...)
"""

import os, sys, subprocess, importlib.util, argparse, json, time
from pathlib import Path

_BASE = Path(__file__).parent.resolve()

def _install_if_missing():
    needed = {
        "bertopic":              "bertopic",
        "umap":                  "umap-learn",
        "hdbscan":               "hdbscan",
        "sklearn":               "scikit-learn",
        "chromadb":              "chromadb",
        "langchain_chroma":      "langchain-chroma",
        "langchain_huggingface": "langchain-huggingface",
        "torch":                 "torch",
        "tqdm":                  "tqdm",
    }
    missing = [pkg for mod, pkg in needed.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print(f"Packages manquants : {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        print("Installation terminee.")
        print("=> Relance RefChat via RefChat.bat pour prendre en compte les nouveaux paquets.")

_install_if_missing()

import numpy as np
from tqdm import tqdm
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import torch
    torch.set_num_threads(8)
except ImportError:
    pass

try:
    from refchat_config import EMBEDDING_MODEL, PERSONAL_DATA, get as cfg_get
    DB_PATH = cfg_get("db_path", str(_BASE / "chroma_db"))
except ImportError:
    EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
    DB_PATH         = str(_BASE / "chroma_db")
    PERSONAL_DATA   = _BASE / "personal_data"

SEP = "─" * 58

# ── Constantes qualité ────────────────────────────────────────────────────────

# Fragments de label indiquant un thème parasite (remerciements, TOC, etc.)
_PARASITIC_PATTERNS = {
    "remerciem", "acknowledg", "bibliograph", "table of content",
    "sommaire", "remerci", "dédicace", "dedicace", "dedication",
    "copyright", "droits réservés", "all rights", "table des matières",
}

# Mots trop génériques pour être le mot principal d'un thème
_OVERLY_GENERIC = {
    "hydrogen", "hydrogène", "water", "fluid", "rock", "study",
    "analysis", "result", "method", "data", "model", "system",
    "field", "formation", "process", "gas", "flow",
}

# Tokens parasites dans les labels BERTopic
_LABEL_NOISE = {
    "thse","thèse","these","merci","remercier","remerciements","ont","remercions",
    "dt","ter","per","der","nov","gm","dd","tem","ocr","utc","une","les","des",
    "sont","mes","travail","galement","mavoir","jai","souhaite","tenu","remercie",
    "trs","davoir","partir",
}


# ── Helpers internes ──────────────────────────────────────────────────────────

def _log(msg):
    try:    tqdm.write(str(msg))
    except: print(str(msg), flush=True)


def _load_chroma():
    _log("Chargement du modele d'embedding...")
    try:
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    emb = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )
    _log(f"   Modele charge ({device.upper()})")
    db = Chroma(persist_directory=DB_PATH, embedding_function=emb)
    return db


def _fetch_all(db):
    total = db._collection.count()
    _log(f"   {total} chunks dans la base")
    ids, docs, metas, embeds = [], [], [], []
    BATCH = 2000
    for offset in tqdm(range(0, total, BATCH), desc="   Lecture ChromaDB", leave=False):
        r = db._collection.get(
            limit=BATCH, offset=offset,
            include=["documents", "metadatas", "embeddings"]
        )
        ids.extend(r["ids"])
        docs.extend(r["documents"])
        metas.extend(r["metadatas"])
        embeds.extend(r["embeddings"])
    return ids, docs, metas, np.array(embeds, dtype=np.float32)


def _aggregate_by_article(ids, docs, metas):
    articles = {}
    for id_, doc, meta in zip(ids, docs, metas):
        fname = meta.get("filename", "unknown")
        if fname not in articles:
            articles[fname] = {"ids": [], "texts": [], "meta": meta}
        articles[fname]["ids"].append(id_)
        articles[fname]["texts"].append(doc)
    return articles


def _representative_text(texts, max_chars=2000):
    abstracts = [t for t in texts if "abstract" in t.lower()[:50] or "resume" in t.lower()[:50]]
    pool = abstracts if abstracts else texts
    combined = " ".join(pool)
    return combined[:max_chars]


def _load_stopwords(base_path: Path) -> list:
    """
    Stopwords intégrés FR + EN + artefacts documentaires,
    complétés par refchat_stopwords.txt si présent (un mot par ligne, # = commentaire).
    """
    _SW_FR = [
        "le","la","les","un","une","de","du","des","en","et","est","au","aux","par",
        "sur","dans","avec","pour","qui","que","à","ce","se","sa","son","ses",
        "ou","si","il","ils","elle","elles","nous","vous","on","y","ne","pas",
        "plus","mais","donc","or","ni","car","leur","leurs","tout","tous","cette",
        "cet","ces","mon","ton","votre","notre","je","tu","me","te","lui","dont",
        "très","aussi","bien","même","encore","après","avant","entre","sous","vers",
        "été","avoir","faire","peut","ainsi","comme","lors","puis","quand","sont",
        "mes","travail","également","jai","souhaite","tenu","ensemble","remercie",
        "mavoir","galement","trs","davoir","partir",
    ]
    _SW_EN = [
        "the","a","an","of","in","and","is","to","for","with","on","at","by",
        "from","are","was","were","has","have","been","be","that","this","as",
        "it","its","or","not","but","also","we","they","their","our","which",
        "all","can","may","between","among","than","thus","such","these","those",
        "however","therefore","although","while","since","both","each","other",
        "used","using","based","results","data","study","studies","paper","show",
        "new","high","low","large","small","present","different","two","three",
        "first","second","well","shows","shown","found","find","known",
    ]
    _SW_DOC = [
        "merci","remerciements","thèse","these","thse","université","universite",
        "docteur","directeur","laboratoire","travaux","présente","presente",
        "figure","tableau","chapitre","annexe","références","references",
        "introduction","conclusion","résumé","resume","keywords",
        "remercier","ont","remercions",
        "acknowledgements","acknowledgments","university","department","professor",
        "thesis","dissertation","chapter","appendix","table","fig",
        "submitted","degree","faculty","copyright","rights","reserved",
        "dt","ter","per","der","nov","gm","dd","tem","ocr","utc",
    ]
    builtin   = set(_SW_FR + _SW_EN + _SW_DOC)
    user_words = set()
    sw_file   = base_path / "refchat_stopwords.txt"

    if not sw_file.exists():
        _SW_TEMPLATE = (
            "# ============================================================\n"
            "#  RefChat — User stopwords for topic clustering\n"
            "# ============================================================\n"
            "#  Add one word per line. Lines starting with # are ignored.\n"
            "#  These words are excluded from topic LABELS only — they do\n"
            "#  NOT affect your ChromaDB content or search results.\n"
            "#\n"
            "#  TIP: always run a dry-run first to check results before\n"
            "#  writing to the database:\n"
            "#    python refchat_theme.py --dry-run --topics 60 --show\n"
            "#\n"
            "# ------------------------------------------------------------\n"
            "#  Common corpus-specific words to consider adding:\n"
            "# ------------------------------------------------------------\n"
            "\n"
            "# Too generic in geoscience bibliographies:\n"
            "# basin\n# basins\n# crust\n# fault\n# water\n# fluid\n"
            "\n"
            "# Parasitic label tokens detected in topic labels:\n"
            "# moments\n# peu\n"
            "\n"
            "# ============================================================\n"
            "#  Add your own words below this line:\n"
            "# ============================================================\n\n"
        )
        try:
            sw_file.write_text(_SW_TEMPLATE, encoding="utf-8")
            _log("   [INFO] refchat_stopwords.txt created — edit it to add corpus-specific stopwords.")
        except Exception as e:
            _log(f"   [WARN] Could not create refchat_stopwords.txt : {e}")
    else:
        with open(sw_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    user_words.add(line.lower())
        if user_words:
            _log(f"   User stopwords: {len(user_words)} word(s) from refchat_stopwords.txt")

    return list(builtin | user_words)


def _build_topic_label(topic_model, topic_id):
    if topic_id == -1:
        return "Non classifie"
    try:
        words = topic_model.get_topic(topic_id)
        if words:
            clean = [w for w, _ in words
                     if w.lower() not in _LABEL_NOISE and len(w) > 2]
            label_words = clean[:4] if clean else [w for w, _ in words[:4]]
            return " - ".join(label_words[:3]).capitalize()
    except Exception:
        pass
    return f"Theme {topic_id}"


def _run_bertopic(article_texts, article_embeddings, n_topics, min_topic_size):
    _log(f"   BERTopic : {len(article_texts)} articles -> ~{n_topics or 'auto'} themes")
    _log(f"   Embeddings pre-calcules : {article_embeddings.shape}")

    n = len(article_texts)

    umap_model = UMAP(
        n_neighbors=min(10, max(2, n // 8)),
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=True,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=max(2, min_topic_size),
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="eom",
        cluster_selection_epsilon=0.3,
        prediction_data=True,
    )

    stopwords = _load_stopwords(_BASE)

    vectorizer = CountVectorizer(
        stop_words=stopwords,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.80,
        max_features=5000,
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        top_n_words=10,
        nr_topics=None,
        verbose=False,
        calculate_probabilities=False,
    )

    topics, _ = topic_model.fit_transform(article_texts, embeddings=article_embeddings)

    n_found = len(set(t for t in topics if t != -1))
    if n_topics and n_found > n_topics:
        _log(f"   {n_found} clusters trouves -> fusion vers {n_topics} themes...")
        topic_model.reduce_topics(article_texts, nr_topics=n_topics)
        topics = [topic_model.topics_[i] for i in range(len(article_texts))]

    return topic_model, topics


def _update_chroma_metadata(db, article_map, filename_to_theme):
    _log(f"\n{SEP}")
    _log("  ETAPE 4/4 -- Ecriture des themes dans ChromaDB")
    _log(SEP)

    all_ids, all_docs, all_metas = [], [], []

    for fname, theme_label in tqdm(filename_to_theme.items(), desc="   Preparation"):
        if fname not in article_map:
            continue
        article = article_map[fname]
        for id_, doc in zip(article["ids"], article["texts"]):
            raw_meta = article["meta"]
            new_meta = {
                k: (str(v) if v is not None else "")
                for k, v in raw_meta.items()
                if isinstance(k, str) and not isinstance(v, (list, dict))
            }
            new_meta["theme"] = theme_label
            all_ids.append(id_)
            all_docs.append(doc)
            all_metas.append(new_meta)

    BATCH_SIZE = 500
    updated = 0
    for i in tqdm(range(0, len(all_ids), BATCH_SIZE), desc="   Ecriture ChromaDB"):
        db._collection.update(
            ids=all_ids[i:i+BATCH_SIZE],
            documents=all_docs[i:i+BATCH_SIZE],
            metadatas=all_metas[i:i+BATCH_SIZE],
        )
        updated += len(all_ids[i:i+BATCH_SIZE])

    _log(f"   {updated} chunks mis a jour")


def _save_theme_map(filename_to_theme, output_path):
    theme_to_articles = {}
    for fname, theme in filename_to_theme.items():
        theme_to_articles.setdefault(theme, []).append(fname)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"theme_map": filename_to_theme, "themes": theme_to_articles},
                  f, ensure_ascii=False, indent=2)
    _log(f"   Mapping sauvegarde : {output_path}")


def _print_theme_table(theme_to_articles):
    _log("")
    _log("=" * 58)
    _log("  THEMES IDENTIFIES")
    _log("=" * 58)
    for theme, articles in sorted(theme_to_articles.items(), key=lambda x: -len(x[1])):
        _log(f"\n  [{theme}]  ({len(articles)} article(s))")
        for a in sorted(articles)[:8]:
            short = a.replace(".pdf", "")[:70]
            _log(f"     - {short}")
        if len(articles) > 8:
            _log(f"     ... +{len(articles)-8} autres")


# ── API qualité ───────────────────────────────────────────────────────────────

def quality_check(theme_to_articles: dict, n_min: int = 5, n_max: int = 25) -> dict:
    """
    Analyse la qualité des thèmes détectés.

    Returns:
        dict {theme_name: [{"type": str, "msg": str}, ...]}
        Types possibles : "parasitic", "small", "large", "generic"
    """
    issues = {}
    for theme, articles in theme_to_articles.items():
        if theme == "Non classifie":
            continue
        probs = []
        t_low = theme.lower()

        # Thème parasite (remerciements, bibliographie, TOC…)
        if any(p in t_low for p in _PARASITIC_PATTERNS):
            probs.append({
                "type": "parasitic",
                "msg":  "Label parasite (remerciements / bibliographie / TOC)"
            })

        # Trop peu d'articles
        if len(articles) < n_min:
            probs.append({
                "type": "small",
                "msg":  f"Trop peu d'articles ({len(articles)} < {n_min} min)"
            })

        # Trop d'articles — sous-clustering recommandé
        if len(articles) > n_max:
            probs.append({
                "type": "large",
                "msg":  f"Thème trop large ({len(articles)} articles) — sous-clustering recommandé"
            })

        # Premier mot trop générique
        first_word = t_low.split(" - ")[0].strip()
        if first_word in _OVERLY_GENERIC:
            probs.append({
                "type": "generic",
                "msg":  f"Premier mot trop générique ('{first_word}')"
            })

        if probs:
            issues[theme] = probs

    return issues


def _suggest_rename(topic_model, topic_id: int, current_label: str) -> str | None:
    """Propose un meilleur label en s'appuyant sur les mots BERTopic filtrés."""
    if topic_id == -1:
        return None
    try:
        words = topic_model.get_topic(topic_id)
        if not words:
            return None
        clean = [
            w for w, _ in words
            if w.lower() not in _LABEL_NOISE
            and w.lower() not in _OVERLY_GENERIC
            and len(w) > 3
        ]
        if len(clean) >= 2:
            candidate = " - ".join(clean[:3]).capitalize()
            if candidate != current_label:
                return candidate
    except Exception:
        pass
    return None


def _compute_article_embeddings(filenames, article_map, ids, embeddings):
    """Calcule l'embedding moyen de chaque article depuis les vecteurs ChromaDB."""
    fname_to_idx = {id_: i for i, id_ in enumerate(ids)}
    result = []
    for fname in filenames:
        article_ids = article_map[fname]["ids"]
        vecs = [embeddings[fname_to_idx[i]] for i in article_ids if i in fname_to_idx]
        if vecs:
            result.append(np.mean(vecs, axis=0))
        else:
            result.append(np.zeros(embeddings.shape[1]))
    return np.array(result, dtype=np.float32)


def _redistribute_small_themes(
    filenames: list,
    filename_to_theme: dict,
    article_embeddings: np.ndarray,
    n_min: int,
) -> dict:
    """
    Redistribue les articles des thèmes sous-peuplés (< n_min)
    vers le thème sémantiquement le plus proche (par cosine sur centroides).
    Si aucun thème cible valide → "Non classifie".

    Returns:
        Nouveau dict filename_to_theme
    """
    theme_to_articles = {}
    for f, t in filename_to_theme.items():
        theme_to_articles.setdefault(t, []).append(f)

    small_themes = {
        t for t, arts in theme_to_articles.items()
        if len(arts) < n_min and t not in ("Non classifie",)
    }
    if not small_themes:
        return dict(filename_to_theme)

    large_themes = {
        t: arts for t, arts in theme_to_articles.items()
        if t not in small_themes and t != "Non classifie"
    }
    if not large_themes:
        # Tout est petit → garder tel quel (pas de cible disponible)
        return dict(filename_to_theme)

    fname_to_emb = {f: article_embeddings[i] for i, f in enumerate(filenames)}

    # Centroides des thèmes cibles
    centroids = {}
    for t, arts in large_themes.items():
        vecs = [fname_to_emb[f] for f in arts if f in fname_to_emb]
        if vecs:
            centroids[t] = np.mean(vecs, axis=0)

    new_mapping = dict(filename_to_theme)

    for theme in small_themes:
        for fname in theme_to_articles[theme]:
            if fname not in fname_to_emb or not centroids:
                new_mapping[fname] = "Non classifie"
                continue
            emb = fname_to_emb[fname]
            norm_emb = np.linalg.norm(emb)
            best_t, best_score = "Non classifie", -1.0
            for t, centroid in centroids.items():
                denom = norm_emb * np.linalg.norm(centroid) + 1e-9
                score = float(np.dot(emb, centroid) / denom)
                if score > best_score:
                    best_score, best_t = score, t
            new_mapping[fname] = best_t

    n_moved = sum(1 for f, t in new_mapping.items()
                  if filename_to_theme[f] in small_themes)
    if n_moved:
        _log(f"   Redistribution : {n_moved} articles déplacés depuis {len(small_themes)} thèmes sous le seuil")

    return new_mapping


# ── API programmatique ────────────────────────────────────────────────────────

def run_clustering_preview(
    db=None,
    n_topics: int | None = None,
    min_docs: int = 5,
    log_cb=None,
) -> dict:
    """
    Lance le pipeline BERTopic complet SANS écrire dans ChromaDB.
    Retourne un dict de prévisualisation pour l'UI de validation.

    Args:
        db:        instance Chroma existante (optionnel — recharge si None)
        n_topics:  nombre cible de thèmes (None = auto)
        min_docs:  seuil minimum d'articles par thème
        log_cb:    callable(str) pour les messages de progression

    Returns:
        {
          "themes": [{"name", "count", "articles": [{"fname","auteur","annee","titre"}],
                      "issues": [], "suggested_rename": str|None}],
          "filename_to_theme": {fname: theme},
          "stats": {"n_articles","n_themes","n_unclassified","n_issues"},
          "error": str|None  (absent si succès)
        }
    """
    def log(msg):
        m = str(msg)
        _log(m)
        if log_cb:
            try: log_cb(m)
            except Exception: pass

    log("📂 Connexion ChromaDB...")
    own_db = db is None
    if own_db:
        db = _load_chroma()

    log("📖 Lecture des chunks...")
    ids, docs, metas, embeddings = _fetch_all(db)
    if not ids:
        return {"error": "Base vide — lance d'abord l'indexation."}

    article_map = _aggregate_by_article(ids, docs, metas)
    filenames   = list(article_map.keys())
    n_articles  = len(filenames)
    log(f"📄 {n_articles} articles distincts")

    if n_articles < 3:
        return {"error": f"Trop peu d'articles ({n_articles} < 3)."}

    log("🔢 Calcul des embeddings moyens par article...")
    article_embeddings = _compute_article_embeddings(filenames, article_map, ids, embeddings)

    repr_texts = [_representative_text(article_map[f]["texts"]) for f in filenames]

    log(f"🤖 BERTopic ({n_articles} articles, ~{n_topics or 'auto'} thèmes)...")
    try:
        topic_model, topic_ids = _run_bertopic(
            repr_texts, article_embeddings, n_topics, min_docs
        )
    except Exception as e:
        return {"error": f"Erreur BERTopic : {e}"}

    unique_topic_ids = set(topic_ids)
    topic_labels     = {tid: _build_topic_label(topic_model, tid) for tid in unique_topic_ids}

    filename_to_theme = {
        fname: topic_labels[tid]
        for fname, tid in zip(filenames, topic_ids)
    }

    # Redistribution des thèmes sous le seuil
    log(f"🔄 Redistribution des thèmes < {min_docs} articles...")
    filename_to_theme = _redistribute_small_themes(
        filenames, filename_to_theme, article_embeddings, min_docs
    )

    # Recalcul theme_to_articles après redistribution
    theme_to_articles: dict[str, list] = {}
    for fname, label in filename_to_theme.items():
        theme_to_articles.setdefault(label, []).append(fname)

    # Vérification qualité
    issues_by_theme = quality_check(theme_to_articles, n_min=min_docs)

    # Suggestions de renommage
    tid_for_label: dict[str, int] = {}
    for fname, tid in zip(filenames, topic_ids):
        label = filename_to_theme.get(fname)
        if label and label not in tid_for_label and tid != -1:
            tid_for_label[label] = tid

    suggestions: dict[str, str] = {}
    for theme, issues in issues_by_theme.items():
        if any(i["type"] in ("generic", "parasitic") for i in issues):
            tid = tid_for_label.get(theme)
            if tid is not None:
                s = _suggest_rename(topic_model, tid, theme)
                if s:
                    suggestions[theme] = s

    # Construction de la sortie
    themes_out = []
    for theme, articles_fnames in sorted(
        theme_to_articles.items(), key=lambda x: -len(x[1])
    ):
        art_details = []
        for fname in articles_fnames:
            meta = article_map[fname]["meta"]
            art_details.append({
                "fname":  fname,
                "auteur": meta.get("auteur", ""),
                "annee":  meta.get("annee",  ""),
                "titre":  meta.get("titre",  ""),
            })
        themes_out.append({
            "name":             theme,
            "count":            len(articles_fnames),
            "articles":         art_details,
            "issues":           issues_by_theme.get(theme, []),
            "suggested_rename": suggestions.get(theme),
        })

    n_unclassified = sum(1 for t in filename_to_theme.values() if t == "Non classifie")
    n_issues       = sum(len(v) for v in issues_by_theme.values())

    log(f"✅ Prévisualisation prête : {len(theme_to_articles)} thèmes, "
        f"{n_unclassified} non classifiés, {n_issues} problème(s) qualité")

    return {
        "themes":            themes_out,
        "filename_to_theme": filename_to_theme,
        "stats": {
            "n_articles":     n_articles,
            "n_themes":       len(theme_to_articles),
            "n_unclassified": n_unclassified,
            "n_issues":       n_issues,
        },
    }


def apply_theme_mapping(
    filename_to_theme: dict,
    db=None,
    log_cb=None,
) -> dict:
    """
    Écrit un mapping filename→thème validé dans ChromaDB et sauvegarde
    refchat_themes.json.

    Args:
        filename_to_theme: {fname: theme_name} validé par l'utilisateur
        db:       instance Chroma existante (recharge si None)
        log_cb:   callable(str) pour les messages de progression

    Returns:
        {"n_themes": int, "n_articles": int}
    """
    def log(msg):
        m = str(msg)
        _log(m)
        if log_cb:
            try: log_cb(m)
            except Exception: pass

    log("📂 Connexion ChromaDB...")
    if db is None:
        db = _load_chroma()

    log("📖 Lecture des chunks...")
    ids, docs, metas, _ = _fetch_all(db)
    article_map = _aggregate_by_article(ids, docs, metas)

    _update_chroma_metadata(db, article_map, filename_to_theme)

    # Sauvegarde JSON
    try:
        map_path = PERSONAL_DATA / "refchat_themes.json"
        _save_theme_map(filename_to_theme, map_path)
    except Exception as e:
        log(f"⚠️  Could not save refchat_themes.json : {e}")

    theme_to_articles: dict[str, list] = {}
    for fname, theme in filename_to_theme.items():
        theme_to_articles.setdefault(theme, []).append(fname)

    n_themes   = len(theme_to_articles)
    n_articles = len(filename_to_theme)
    log(f"✅ {n_themes} thèmes écrits, {n_articles} articles mis à jour")
    return {"n_themes": n_themes, "n_articles": n_articles}


# ── CLI main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Thematisation automatique RefChat")
    parser.add_argument("--topics",   type=int, default=None,
                        help="Nombre cible de thèmes (défaut : auto)")
    parser.add_argument("--min-docs", type=int, default=5,
                        help="Nombre minimum d'articles par thème (défaut : 5)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Affiche les thèmes sans modifier la base")
    parser.add_argument("--show",     action="store_true",
                        help="Affiche le tableau thème → articles")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("   RefChat -- Thematisation automatique")
    print("=" * 60)
    print(f"  Base ChromaDB : {DB_PATH}")
    print(f"  Embedding     : {EMBEDDING_MODEL}")
    if args.dry_run:
        print("  MODE DRY-RUN : aucune ecriture dans la base")
    print("")

    # Utilise run_clustering_preview pour le pipeline complet
    result = run_clustering_preview(
        db=None,
        n_topics=args.topics,
        min_docs=args.min_docs,
    )

    if "error" in result:
        print(f"ERREUR : {result['error']}")
        return

    filename_to_theme = result["filename_to_theme"]
    stats             = result["stats"]

    theme_to_articles: dict[str, list] = {}
    for fname, label in filename_to_theme.items():
        theme_to_articles.setdefault(label, []).append(fname)

    print(f"\n   {stats['n_themes']} thèmes identifiés")
    print(f"   {stats['n_articles'] - stats['n_unclassified']} articles classifiés")
    if stats["n_unclassified"]:
        print(f"   {stats['n_unclassified']} articles non classifiés")
    if stats["n_issues"]:
        print(f"   ⚠️  {stats['n_issues']} problème(s) qualité détecté(s)")
        for t in result["themes"]:
            for issue in t.get("issues", []):
                print(f"      [{t['name']}] {issue['msg']}")
            if t.get("suggested_rename"):
                print(f"      [{t['name']}] 💡 Suggestion : {t['suggested_rename']}")

    if args.show or args.dry_run:
        _print_theme_table(theme_to_articles)

    if not args.dry_run:
        apply_theme_mapping(filename_to_theme)
    else:
        print("\n   (dry-run : ChromaDB non modifié)")

    elapsed = int(time.time() - t0)
    dur_str = f"{elapsed//60}m{elapsed%60:02d}s" if elapsed >= 60 else f"{elapsed}s"
    print(f"\n{'=' * 60}")
    print(f"   Terminé en {dur_str}")
    print(f"{'=' * 60}")
    if not args.dry_run:
        print("   Les thèmes sont maintenant disponibles dans RefChat.")
        print("   Tu peux filtrer par thème dans tes questions.")


if __name__ == "__main__":
    main()
