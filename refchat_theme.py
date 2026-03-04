"""
RefChat — Thématisation automatique de la bibliothèque
=======================================================
Ce script analyse les embeddings déjà présents dans ChromaDB,
identifie des thèmes par clustering (BERTopic), et écrit le champ
`theme` dans les métadonnées de chaque chunk — SANS réindexer.

Usage :
    python refchat_theme.py

Options :
    --topics N      Nombre de thèmes cibles (défaut : auto)
    --min-docs N    Nbre minimum d'articles par thème (défaut : 2)
    --dry-run       Affiche les thèmes sans modifier la base
    --show          Affiche le tableau thème → articles à la fin
"""

import os, sys, subprocess, importlib.util, argparse, json, time
from pathlib import Path

_BASE = Path(__file__).parent.resolve()

def _install_if_missing():
    needed = {
        "bertopic":     "bertopic",
        "umap":         "umap-learn",
        "hdbscan":      "hdbscan",
        "sklearn":      "scikit-learn",
        "chromadb":     "chromadb",
        "langchain_chroma": "langchain-chroma",
        "langchain_huggingface": "langchain-huggingface",
        "torch":        "torch",
        "tqdm":         "tqdm",
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
    from refchat_config import EMBEDDING_MODEL, get as cfg_get
    DB_PATH = cfg_get("db_path", str(_BASE / "chroma_db"))
except ImportError:
    EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
    DB_PATH = str(_BASE / "chroma_db")

SEP = "─" * 58

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
        # Thèses françaises
        "merci","remerciements","thèse","these","thse","université","universite",
        "docteur","directeur","laboratoire","travaux","présente","presente",
        "figure","tableau","chapitre","annexe","références","references",
        "introduction","conclusion","résumé","resume","keywords",
        "remercier","ont","remercions",
        # Thèses anglophones
        "acknowledgements","acknowledgments","university","department","professor",
        "thesis","dissertation","chapter","appendix","table","fig",
        "submitted","degree","faculty","copyright","rights","reserved",
        # Artefacts OCR
        "dt","ter","per","der","nov","gm","dd","tem","ocr","utc",
    ]
    builtin = set(_SW_FR + _SW_EN + _SW_DOC)
    user_words = set()
    sw_file = base_path / "refchat_stopwords.txt"

    if not sw_file.exists():
        # Auto-generate template on first run
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
            "#  (uncomment or add your own below)\n"
            "# ------------------------------------------------------------\n"
            "\n"
            "# Too generic in geoscience bibliographies:\n"
            "# basin\n"
            "# basins\n"
            "# crust\n"
            "# fault\n"
            "# water\n"
            "# fluid\n"
            "\n"
            "# Parasitic label tokens detected in topic labels:\n"
            "# moments\n"
            "# peu\n"
            "\n"
            "# ============================================================\n"
            "#  Add your own words below this line:\n"
            "# ============================================================\n"
            "\n"
        )
        try:
            sw_file.write_text(_SW_TEMPLATE, encoding="utf-8")
            _log("   [INFO] refchat_stopwords.txt created — edit it to add corpus-specific stopwords.")
            _log("         Run --dry-run first to validate results before writing to the database.")
        except Exception as e:
            _log(f"   [WARN] Could not create refchat_stopwords.txt : {e}")
    else:
        with open(sw_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    user_words.add(line.lower())
        if user_words:
            _log(f"   User stopwords: {len(user_words)} word(s) loaded from refchat_stopwords.txt")
        else:
            _log("   refchat_stopwords.txt found but no custom words active (all commented out)")

    return list(builtin | user_words)


# Tokens qui signalent un label parasite — filtrés à l'affichage uniquement
_LABEL_NOISE = {
    "thse","thèse","these","merci","remercier","remerciements","ont","remercions",
    "dt","ter","per","der","nov","gm","dd","tem","ocr","utc","une","les","des",
    "sont","mes","travail","galement","mavoir","jai","souhaite","tenu","remercie",
    "trs","davoir","partir",
}


def _run_bertopic(article_texts, article_embeddings, n_topics, min_topic_size):
    _log(f"   BERTopic : {len(article_texts)} articles -> ~{n_topics or 'auto'} themes")
    _log(f"   Embeddings pre-calcules : {article_embeddings.shape}")

    n = len(article_texts)

    umap_model = UMAP(
        n_neighbors=min(10, max(2, n // 8)),   # voisinage local -> clusters fins
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
        cluster_selection_epsilon=0.3,         # évite la sur-fusion des clusters proches
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
        nr_topics=None,          # HDBSCAN découpe librement d'abord
        verbose=False,
        calculate_probabilities=False,
    )

    topics, _ = topic_model.fit_transform(article_texts, embeddings=article_embeddings)

    # Fusion post-hoc jusqu'au nombre cible si --topics fourni
    n_found = len(set(t for t in topics if t != -1))
    if n_topics and n_found > n_topics:
        _log(f"   {n_found} clusters trouves -> fusion vers {n_topics} themes...")
        topic_model.reduce_topics(article_texts, nr_topics=n_topics)
        topics = [topic_model.topics_[i] for i in range(len(article_texts))]

    return topic_model, topics

def _build_topic_label(topic_model, topic_id):
    if topic_id == -1:
        return "Non classifie"
    try:
        words = topic_model.get_topic(topic_id)
        if words:
            clean = [w for w, _ in words
                     if w.lower() not in _LABEL_NOISE and len(w) > 2]
            # Dédoublonner : si le mot apparaît déjà dans un autre label
            # on prend jusqu'à 4 mots pour avoir plus de chance de différencier
            label_words = clean[:4] if clean else [w for w, _ in words[:4]]
            return " - ".join(label_words[:3]).capitalize()
    except Exception:
        pass
    return f"Theme {topic_id}"

def _update_chroma_metadata(db, article_map, filename_to_theme):
    _log(f"\n{SEP}")
    _log("  ETAPE 4/4 -- Ecriture des themes dans ChromaDB")
    _log(SEP)

    all_ids, all_docs, all_metas = [], [], []

    for fname, theme_label in tqdm(filename_to_theme.items(), desc="   Preparation"):
        article = article_map[fname]
        for id_, doc in zip(article["ids"], article["texts"]):
            # ChromaDB rejette les valeurs None, listes, et types non-scalaires
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

def main():
    parser = argparse.ArgumentParser(description="Thematisation automatique RefChat")
    parser.add_argument("--topics",   type=int, default=None)
    parser.add_argument("--min-docs", type=int, default=2)
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--show",     action="store_true")
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

    print(SEP)
    print("  ETAPE 1/4 -- Connexion a ChromaDB")
    print(SEP)
    db = _load_chroma()

    print(f"\n{SEP}")
    print("  ETAPE 2/4 -- Lecture des chunks")
    print(SEP)
    ids, docs, metas, embeddings = _fetch_all(db)

    if len(ids) == 0:
        print("ERREUR : Base vide. Lance d'abord refchat_ingest.py")
        return

    print(f"\n{SEP}")
    print("  ETAPE 3/4 -- Clustering BERTopic")
    print(SEP)

    article_map = _aggregate_by_article(ids, docs, metas)
    n_articles  = len(article_map)
    print(f"   {n_articles} articles distincts dans la base\n")

    if n_articles < 3:
        print("AVERTISSEMENT : Moins de 3 articles, impossible de clusteriser.")
        return

    filenames   = list(article_map.keys())
    repr_texts  = [_representative_text(article_map[f]["texts"]) for f in filenames]

    # Calcul des embeddings moyens par article depuis les vecteurs ChromaDB
    # (evite que BERTopic recharge son propre modele et ecrase les embeddings)
    _log("   Calcul des embeddings moyens par article...")
    fname_to_idx = {id_: i for i, (id_, _, _) in enumerate(zip(ids, docs, metas))}
    article_embeddings = []
    for fname in filenames:
        article_ids = article_map[fname]["ids"]
        vecs = []
        for id_ in article_ids:
            idx = fname_to_idx.get(id_)
            if idx is not None:
                vecs.append(embeddings[idx])
        if vecs:
            article_embeddings.append(np.mean(vecs, axis=0))
        else:
            article_embeddings.append(np.zeros(embeddings.shape[1]))
    article_embeddings = np.array(article_embeddings, dtype=np.float32)

    try:
        topic_model, topic_ids = _run_bertopic(repr_texts, article_embeddings, args.topics, args.min_docs)
    except Exception as e:
        print(f"ERREUR BERTopic : {e}")
        print("   Essaie avec --topics N pour forcer un nombre de themes")
        return

    unique_topic_ids = set(topic_ids)
    topic_labels = {tid: _build_topic_label(topic_model, tid) for tid in unique_topic_ids}

    filename_to_theme = {
        fname: topic_labels[tid]
        for fname, tid in zip(filenames, topic_ids)
    }

    theme_to_articles = {}
    for fname, label in filename_to_theme.items():
        theme_to_articles.setdefault(label, []).append(fname)

    n_classified   = sum(1 for t in topic_ids if t != -1)
    n_unclassified = sum(1 for t in topic_ids if t == -1)

    print(f"\n   {len(theme_to_articles)} themes identifies")
    print(f"   {n_classified} articles classifies")
    if n_unclassified:
        print(f"   {n_unclassified} articles non classifies (theme 'Non classifie')")

    if args.show or args.dry_run:
        _print_theme_table(theme_to_articles)

    if not args.dry_run:
        _update_chroma_metadata(db, article_map, filename_to_theme)
        map_path = _BASE / "refchat_themes.json"
        _save_theme_map(filename_to_theme, map_path)
    else:
        print("\n   (dry-run : ChromaDB non modifie)")

    elapsed = int(time.time() - t0)
    dur_str = f"{elapsed//60}m{elapsed%60:02d}s" if elapsed >= 60 else f"{elapsed}s"
    print(f"\n{'=' * 60}")
    print(f"   Termine en {dur_str}")
    print(f"{'=' * 60}")
    if not args.dry_run:
        print("   Les themes sont maintenant disponibles dans RefChat.")
        print("   Tu peux filtrer par theme dans tes questions.")

if __name__ == "__main__":
    main()
