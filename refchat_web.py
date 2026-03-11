"""
RefChat — Interface Web
Lance avec : python refchat_main.py (ou via RefChat.bat)
"""
import warnings, os, sys, threading, webbrowser, time, json, pathlib, contextlib
os.environ["PYTHONUNBUFFERED"] = "1"
warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"]       = "8"
os.environ["MKL_NUM_THREADS"]       = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
try:
    import torch; torch.set_num_threads(8)
except ImportError:
    pass

from flask import Flask, request, jsonify, render_template_string, Response, stream_with_context
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

sys.path.insert(0, os.path.dirname(__file__))
from refchat_llm import (
    PROMPT_QUESTION, PROMPT_RESUME, PROMPT_REFERENCE,
    detecter_mode, extraire_nom_auteur,
    chercher_par_auteur, recuperer_articles_complets,
    chercher_semantic_scholar,
    format_docs, expand_query, charger_llm, charger_bm25,
    extraire_mots_cles_llm,
    lister_themes, recuperer_articles_par_theme, detecter_theme_query,
)
import refchat_config as cfg

app = Flask(__name__)

MAX_HISTORY          = 3
conversation_history = []

STATE = {
    "db": None, "llm": None, "nom_llm": None,
    "ready": False, "error": None, "modele": "api",
    "MAX_ARTICLES": 3, "MAX_CHUNKS_ARTICLE": 5, "K_INITIAL": 8,
    "bm25": None, "reranker": None,
    "memory_enabled": False,
    # Ingestion
    "ingest_running": False,
    "ingest_log":     [],
    "ingest_done":    False,
    "ingest_error":   None,
    "theme_running": False,
    "theme_log":     [],
    "theme_done":    False,
    "theme_error":   None,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_db_path():
    return cfg.get("db_path", str(cfg.PERSONAL_DATA / "chroma_db"))

def _detect_device_and_batch():
    """Auto-detect best compute device + safe batch size for the embedding model.
    Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU.
    CUDA batch size is scaled to available VRAM to avoid OOM on small GPUs.
    """
    try:
        import torch
        if torch.cuda.is_available():
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if   vram_gb >= 8: batch = 512
                elif vram_gb >= 6: batch = 256
                elif vram_gb >= 4: batch = 128
                else:              batch = 64
            except Exception:
                batch = 128
            return "cuda", batch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", 128   # Apple Silicon — unified memory, conservative default
    except ImportError:
        pass
    return "cpu", 256


def get_embedding():
    device, batch_size = _detect_device_and_batch()
    if device != "cpu":
        print(f"⚡ Embeddings: {device.upper()} — batch_size={batch_size}")
    return HuggingFaceEmbeddings(
        model_name=cfg.EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
    )

def init_system(modele="api"):
    global conversation_history
    conversation_history = []
    try:
        STATE["modele"] = modele
        
        if modele == "api":
            # API Cloud : Calibré pour ~50k - 60k tokens
            STATE["MAX_ARTICLES"] = 10;  STATE["MAX_CHUNKS_ARTICLE"] = 20; STATE["K_INITIAL"] = 30
        elif modele == "mixtral":
            STATE["MAX_ARTICLES"] = 3;  STATE["MAX_CHUNKS_ARTICLE"] = 5;  STATE["K_INITIAL"] = 8
        else:
            # Local Mistral (32 GB RAM): heavier config (~15k–20k tokens)
            STATE["MAX_ARTICLES"] = 8;  STATE["MAX_CHUNKS_ARTICLE"] = 10;  STATE["K_INITIAL"] = 20

        db_path = get_db_path()
        STATE["db"]  = Chroma(persist_directory=db_path, embedding_function=get_embedding())
        STATE["llm"], STATE["nom_llm"] = charger_llm(modele)
        STATE["bm25"] = charger_bm25(STATE["db"])

        try:
            from sentence_transformers import CrossEncoder
            # Multilingual MS MARCO cross-encoder (~120 MB) — reranks top candidates
            # for maximum precision on FR/EN mixed libraries.
            STATE["reranker"] = CrossEncoder("nreimers/mmarco-mMiniLMv2-L12-H384-v1")
        except Exception:
            STATE["reranker"] = None  # optional — falls back to RRF ranking

        if modele != "api":
            import urllib.request as _ur
            try: _ur.urlopen("http://127.0.0.1:11434/", timeout=3)
            except Exception as e: raise RuntimeError(f"Ollama non accessible : {e}")

        STATE["ready"] = True;  STATE["error"] = None
        return True
    except Exception as e:
        STATE["error"] = str(e);  STATE["ready"] = False
        return False

def build_prompt_with_history(base_prompt_template, history):
    if not history:
        return base_prompt_template
    history_text = "\n\nHISTORIQUE DE LA CONVERSATION :\n"
    for msg in history:
        role = "Utilisateur" if msg["role"] == "user" else "Assistant"
        history_text += f"{role} : {msg['content']}\n"
    history_text += "\n(Tiens compte du contexte ci-dessus.)\n"
    messages = base_prompt_template.messages
    new_messages = []
    for m in messages:
        if hasattr(m, 'prompt') and "{context}" in str(m.prompt.template):
            from langchain_core.prompts import SystemMessagePromptTemplate, PromptTemplate
            new_messages.append(SystemMessagePromptTemplate(prompt=PromptTemplate(
                template=history_text + m.prompt.template,
                input_variables=m.prompt.input_variables
            )))
        else:
            new_messages.append(m)
    return ChatPromptTemplate.from_messages(new_messages)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/api/status")
def status():
    db    = STATE["db"]
    count = db._collection.count() if db else 0
    return jsonify({"ready": STATE["ready"], "error": STATE["error"],
                    "modele": STATE["nom_llm"], "nb_extraits": count,
                    "memory_enabled": STATE["memory_enabled"]})


@app.route("/api/themes")
def api_themes():
    """Retourne la liste des themes disponibles dans la base."""
    db = STATE["db"]
    if not db:
        return jsonify({"themes": [], "error": "Base non initialisee"})
    themes = lister_themes(db)
    return jsonify({"themes": themes, "count": len(themes)})

@app.route("/api/theme/start", methods=["POST"])
def api_theme_start():
    if STATE["theme_running"]:
        return jsonify({"success": False, "error": "Already running"})
    data = request.get_json() or {}
    STATE.update({"theme_running": True, "theme_log": [],
                  "theme_done": False, "theme_error": None})
    n_topics = data.get("n_topics", None)
    min_docs = data.get("min_docs", 2)
    def run():
        import sys as _sys, importlib, contextlib
        try:
            sys.path.insert(0, str(pathlib.Path(__file__).parent))
            import refchat_theme as _rt
            importlib.reload(_rt)
            class Cap:
                def write(self, msg):
                    if msg.strip(): STATE["theme_log"].append(msg.rstrip())
                def flush(self): pass
                def isatty(self): return False
            old = _sys.argv[:]
            _sys.argv = ["refchat_theme.py", "--show"]
            if n_topics: _sys.argv += ["--topics", str(n_topics)]
            _sys.argv += ["--min-docs", str(min_docs)]
            try:
                with contextlib.redirect_stdout(Cap()): _rt.main()
            finally:
                _sys.argv = old
            STATE["theme_done"] = True
        except Exception as e:
            STATE["theme_error"] = str(e); STATE["theme_done"] = True
        finally:
            STATE["theme_running"] = False
    threading.Thread(target=run, daemon=True).start()
    return jsonify({"success": True})


@app.route("/api/theme/status")
def api_theme_status():
    return jsonify({
        "running": STATE["theme_running"],
        "done":    STATE["theme_done"],
        "error":   STATE["theme_error"],
        "log":     STATE["theme_log"][-100:],
    })



@app.route("/api/hardware/detect")
def api_hardware_detect():
    """Detect CPU cores, RAM, GPU and return recommended Ollama parameters."""
    import platform, multiprocessing, subprocess as _sp
    cpu = multiprocessing.cpu_count()
    ram_gb = 0
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / 1073741824)
    except ImportError:
        try:
            if platform.system() == "Windows":
                import ctypes
                class MEM(ctypes.Structure):
                    _fields_ = [("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),
                                 ("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),
                                 ("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),
                                 ("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),
                                 ("sullAvailExtendedVirtual",ctypes.c_ulonglong)]
                m = MEM(); m.dwLength = ctypes.sizeof(m)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
                ram_gb = round(m.ullTotalPhys / 1073741824)
            else:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal"):
                            ram_gb = round(int(line.split()[1]) / 1048576); break
        except Exception:
            pass
    has_gpu = False; gpu_name = ""; gpu_vram = 0
    try:
        r = _sp.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split(",")
            gpu_name = parts[0].strip(); gpu_vram = round(int(parts[1].strip())/1024); has_gpu = True
    except Exception:
        pass
    rec_threads = max(2, int(cpu * 0.75))
    rec_gpu     = 99 if has_gpu else 0
    rec_batch   = 1024 if (has_gpu and gpu_vram >= 8) else (512 if ram_gb >= 32 else (256 if ram_gb >= 16 else 128))
    # num_ctx LOCAL : limité par la VRAM (goulot réel quand GPU actif)
    # num_ctx API   : pas de limite hardware, on profite du contexte long
    if has_gpu and gpu_vram > 0:
        rec_num_ctx_local = 8192 if gpu_vram >= 8 else (4096 if gpu_vram >= 4 else 2048)
    else:
        # CPU only : limité par la RAM
        rec_num_ctx_local = 16384 if ram_gb >= 32 else (8192 if ram_gb >= 16 else (4096 if ram_gb >= 8 else 2048))
    c = cfg._load()
    # Sauvegarde automatique si pas encore défini
    if not c.get("ollama_num_ctx_local"):
        cfg.save({"ollama_num_ctx_local": rec_num_ctx_local})
    return jsonify({
        "cpu_count": cpu, "ram_gb": ram_gb,
        "has_gpu": has_gpu, "gpu_name": gpu_name, "gpu_vram_gb": gpu_vram,
        "rec_threads": rec_threads, "rec_gpu": rec_gpu, "rec_batch": rec_batch,
        "rec_num_ctx_local": rec_num_ctx_local,
        "saved_num_thread":  c.get("num_thread",""),
        "saved_num_gpu":     c.get("num_gpu",""),
        "saved_num_batch":   c.get("num_batch",""),
        "saved_num_ctx_local": c.get("ollama_num_ctx_local", rec_num_ctx_local),
    })


@app.route("/api/ollama/check")
def api_ollama_check():
    import shutil, subprocess as _sp
    import urllib.request as _ur
    ollama_installed = shutil.which("ollama") is not None
    ollama_running = False
    if ollama_installed:
        try:
            _ur.urlopen("http://127.0.0.1:11434/", timeout=2)
            ollama_running = True
        except Exception:
            pass
    models_available = []
    if ollama_running:
        try:
            result = _sp.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
            lines = result.stdout.strip().splitlines()[1:]
            for line in lines:
                name = line.split()[0] if line.split() else ""
                if name: models_available.append(name)
        except Exception:
            pass
    needed = ["mistral:7b-instruct-q4_0", "mistral"]
    missing = [m for m in needed if not any(m.split(":")[0] in a for a in models_available)]
    return jsonify({"ollama_installed": ollama_installed, "ollama_running": ollama_running,
                    "models_available": models_available, "models_missing": missing,
                    "all_ready": ollama_installed and ollama_running and len(missing) == 0})

@app.route("/api/ollama/pull", methods=["POST"])
def api_ollama_pull():
    import subprocess as _sp
    data  = request.get_json() or {}
    model = data.get("model", "").strip()
    if not model:
        return jsonify({"success": False, "error": "Model not specified"})
    try:
        result = _sp.run(["ollama", "pull", model], capture_output=True, text=True, timeout=1800)
        if result.returncode == 0:
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": result.stderr.strip() or "Unknown error"})
    except _sp.TimeoutExpired:
        return jsonify({"success": False, "error": "Timeout — download took too long"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/setup/check")
def api_setup_check():
    zotero = cfg.get("zotero_path", "")
    db     = get_db_path()
    has_db = os.path.isdir(db) and bool(os.listdir(db)) if os.path.isdir(db) else False
    auto_detected = ""
    if not zotero:
        try:
            import refchat_ingest
            auto_detected = refchat_ingest._detecter_zotero_storage()
        except Exception:
            pass
    return jsonify({
        "first_launch":   not zotero and not has_db,
        "has_zotero":     bool(zotero),
        "has_db":         has_db,
        "zotero_path":    zotero or auto_detected,
        "auto_detected":  bool(auto_detected and not zotero),
        "db_path":        db,
        "nb_chunks":      STATE["db"]._collection.count() if STATE["db"] else 0,
    })

@app.route("/api/config", methods=["GET"])
def api_config_get():
    c      = cfg._load()
    zotero = c.get("zotero_path", "")
    if not zotero:
        try:
            import refchat_ingest
            zotero = refchat_ingest._detecter_zotero_storage()
        except Exception:
            pass
    return jsonify({
        "zotero_path":              zotero,
        "db_path":                  c.get("db_path", get_db_path()),
        "llm_model":                c.get("llm_model", "mistral-light"),
        "mistral_api_key":          c.get("mistral_api_key", ""),
        "semantic_scholar_api_key": c.get("semantic_scholar_api_key", ""),
        "num_thread":               c.get("num_thread", ""),
        "num_gpu":                  c.get("num_gpu", ""),
        "num_batch":                c.get("num_batch", ""),
        "ollama_num_ctx_local":     c.get("ollama_num_ctx_local", 4096),
        "ollama_temperature":       c.get("ollama_temperature", 0.1),
        "bm25_weight":              c.get("bm25_weight", 0.3),
    })

@app.route("/api/config", methods=["POST"])
def api_config_save():
    data = request.get_json() or {}
    try:
        allowed = ("zotero_path","db_path","llm_model","mistral_api_key",
                   "semantic_scholar_api_key","num_thread","num_gpu","num_batch",
                   "ollama_num_ctx_local","ollama_temperature","bm25_weight")
        updates = {k: data[k] for k in allowed if k in data}
        for k in ("num_thread","num_gpu","num_batch","ollama_num_ctx_local"):
            if k in updates and str(updates[k]).strip() != "":
                updates[k] = int(updates[k])
        if "ollama_temperature" in updates and str(updates["ollama_temperature"]).strip() != "":
            updates["ollama_temperature"] = float(updates["ollama_temperature"])
        if "bm25_weight" in updates and str(updates["bm25_weight"]).strip() != "":
            updates["bm25_weight"] = float(updates["bm25_weight"])
        cfg.save(updates)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/ingest/scan")
def api_ingest_scan():
    import json as _json
    zotero = cfg.get("zotero_path", "")
    if not zotero or not os.path.isdir(zotero):
        return jsonify({"nouveaux": 0, "total": 0, "erreur": "Zotero not configured"})
    try:
        import refchat_ingest
        tous = refchat_ingest._scan_zotero_pdfs(zotero)
        suivi_path = cfg.PERSONAL_DATA / "refchat_index_db.json"
        # Already processed = indexed OR explicitly rejected (chunks=0 means rejected/blacklisted)
        deja = set()
        if suivi_path.exists():
            try:
                with open(suivi_path, encoding="utf-8") as f:
                    raw = _json.load(f)
                    # raw is a dict {path: {nb_chunks, ...}} — exclude nothing, all processed paths are "done"
                    deja = set(raw.keys()) if isinstance(raw, dict) else set(raw)
            except Exception:
                pass
        # Also load blacklist (filenames only)
        blacklist = set()
        ignore_path = pathlib.Path(__file__).parent / "refchat_ignore.txt"
        if ignore_path.exists():
            try:
                with open(ignore_path, encoding="utf-8") as f:
                    blacklist = {l.strip() for l in f if l.strip()}
            except Exception:
                pass
        nouveaux = [p for p in tous if p not in deja and os.path.basename(p) not in blacklist]
        return jsonify({
            "nouveaux": len(nouveaux),
            "total":    len(tous),
            "liste":    [os.path.basename(p) for p in nouveaux[:5]],
        })
    except Exception as e:
        return jsonify({"nouveaux": 0, "total": 0, "erreur": str(e)})

@app.route("/api/blacklist/add", methods=["POST"])
def api_blacklist_add():
    data = request.get_json() or {}
    filenames = data.get("filenames", [])
    if not filenames:
        return jsonify({"success": False, "error": "No filenames provided"})
    try:
        ignore_path = pathlib.Path(__file__).parent / "refchat_ignore.txt"
        existing = set()
        if ignore_path.exists():
            with open(ignore_path, encoding="utf-8") as f:
                existing = {l.strip() for l in f if l.strip()}
        new_entries = [fn for fn in filenames if fn not in existing]
        if new_entries:
            with open(ignore_path, "a", encoding="utf-8") as f:
                for fn in new_entries:
                    f.write(fn + "\n")
        return jsonify({"success": True, "added": len(new_entries)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/ingest/scan/full")
def api_ingest_scan_full():
    """Like /api/ingest/scan but returns full paths and filenames for the modal."""
    import json as _json
    zotero = cfg.get("zotero_path", "")
    if not zotero or not os.path.isdir(zotero):
        return jsonify({"nouveaux": [], "erreur": "Zotero not configured"})
    try:
        import refchat_ingest
        tous = refchat_ingest._scan_zotero_pdfs(zotero)
        suivi_path = cfg.PERSONAL_DATA / "refchat_index_db.json"
        deja = set()
        if suivi_path.exists():
            try:
                with open(suivi_path, encoding="utf-8") as f:
                    raw = _json.load(f)
                    deja = set(raw.keys()) if isinstance(raw, dict) else set(raw)
            except Exception:
                pass
        blacklist = set()
        ignore_path = pathlib.Path(__file__).parent / "refchat_ignore.txt"
        if ignore_path.exists():
            try:
                with open(ignore_path, encoding="utf-8") as f:
                    blacklist = {l.strip() for l in f if l.strip()}
            except Exception:
                pass
        nouveaux = [
            {"path": p, "filename": os.path.basename(p)}
            for p in tous
            if p not in deja and os.path.basename(p) not in blacklist
        ]
        return jsonify({"nouveaux": nouveaux})
    except Exception as e:
        return jsonify({"nouveaux": [], "erreur": str(e)})

@app.route("/api/db/articles")
def api_db_articles():
    """List all currently indexed articles with metadata (from tracking file + ChromaDB)."""
    db = STATE["db"]
    if not db:
        return jsonify({"articles": [], "total": 0, "error": "DB not loaded"})
    try:
        suivi_path = cfg.PERSONAL_DATA / "refchat_index_db.json"
        suivi = {}
        if suivi_path.exists():
            with open(suivi_path, encoding="utf-8") as f:
                suivi = json.load(f)

        # Build base info from tracking file (full path → info)
        fname_info = {}
        for path, info in suivi.items():
            if not isinstance(info, dict):
                continue
            chunks = info.get("chunks", 0)
            if chunks == 0:
                continue  # blacklisted / rejected
            fn = os.path.basename(path)
            if fn not in fname_info:
                fname_info[fn] = {
                    "filename": fn,
                    "nb_chunks": chunks,
                    "date": info.get("date", ""),
                    "auteur": "", "annee": "", "titre": "", "doc_type": "article",
                }

        if fname_info:
            # Enrich with ChromaDB metadata (metadatas only, no documents — fast)
            total = db._collection.count()
            enriched = 0
            for offset in range(0, total, 2000):
                r = db._collection.get(limit=2000, offset=offset, include=["metadatas"])
                for meta in r["metadatas"]:
                    fn = meta.get("filename", "")
                    if fn in fname_info and not fname_info[fn]["auteur"]:
                        fname_info[fn]["auteur"]   = meta.get("auteur", "")
                        fname_info[fn]["annee"]    = meta.get("annee", "")
                        fname_info[fn]["titre"]    = meta.get("titre", "")
                        fname_info[fn]["doc_type"] = meta.get("doc_type", "article")
                        enriched += 1
                if enriched >= len(fname_info):
                    break  # all articles enriched, stop early

        articles = sorted(fname_info.values(), key=lambda x: (x.get("auteur") or x["filename"]).lower())
        return jsonify({"articles": articles, "total": len(articles)})
    except Exception as e:
        return jsonify({"articles": [], "total": 0, "error": str(e)})


@app.route("/api/db/delete", methods=["POST"])
def api_db_delete():
    """Delete articles from ChromaDB + add to blacklist + remove from tracking file."""
    data = request.get_json() or {}
    filenames = data.get("filenames", [])
    if not filenames:
        return jsonify({"success": False, "error": "No filenames provided"})
    db = STATE["db"]
    if not db:
        return jsonify({"success": False, "error": "Database not loaded"})

    errors = []

    # 1. Delete all chunks from ChromaDB
    for fname in filenames:
        try:
            db._collection.delete(where={"filename": fname})
        except Exception as e:
            errors.append(f"{fname}: {e}")

    # 2. Add to blacklist (refchat_ignore.txt stays next to scripts)
    ignore_path = pathlib.Path(__file__).parent / "refchat_ignore.txt"
    existing = set()
    if ignore_path.exists():
        with open(ignore_path, encoding="utf-8") as f:
            existing = {l.strip() for l in f if l.strip()}
    new_entries = [fn for fn in filenames if fn not in existing]
    if new_entries:
        with open(ignore_path, "a", encoding="utf-8") as f:
            for fn in new_entries:
                f.write(fn + "\n")

    # 3. Remove from tracking file
    suivi_path = cfg.PERSONAL_DATA / "refchat_index_db.json"
    if suivi_path.exists():
        try:
            with open(suivi_path, encoding="utf-8") as f:
                tracking = json.load(f)
            if isinstance(tracking, dict):
                fnames_set = set(filenames)
                keys_to_remove = [k for k in tracking if os.path.basename(k) in fnames_set]
                for k in keys_to_remove:
                    del tracking[k]
                with open(suivi_path, "w", encoding="utf-8") as f:
                    json.dump(tracking, f, ensure_ascii=False, indent=2)
        except Exception as e:
            errors.append(f"tracking update: {e}")

    # 4. Invalidate BM25 cache (will be rebuilt on next query)
    bm25_cache = cfg.PERSONAL_DATA / "bm25_index.pkl"
    if bm25_cache.exists():
        try:
            bm25_cache.unlink()
            STATE["bm25"] = None  # force rebuild in memory too
        except Exception:
            pass

    return jsonify({
        "success": len(errors) == 0,
        "deleted": len(filenames),
        "blacklisted": len(new_entries),
        "errors": errors,
    })


@app.route("/api/ingest/start", methods=["POST"])
def api_ingest_start():
    if STATE["ingest_running"]:
        return jsonify({"success": False, "error": "Indexing already in progress"})
    data        = request.get_json() or {}
    zotero_path = data.get("zotero_path", "").strip()
    if not zotero_path or not os.path.isdir(zotero_path):
        return jsonify({"success": False, "error": f"Dossier introuvable : {zotero_path}"})

    cfg.save({"zotero_path": zotero_path})
    STATE.update({"ingest_running": True, "ingest_log": [],
                  "ingest_done": False, "ingest_error": None})

    def run():
        try:
            import refchat_ingest
            refchat_ingest.ZOTERO_PATH = zotero_path

            class Cap:
                def write(self, msg):
                    if msg.strip():
                        STATE["ingest_log"].append(msg.rstrip())
                def flush(self): pass
                def isatty(self): return False

            with contextlib.redirect_stdout(Cap()):
                refchat_ingest.main()
            STATE["ingest_done"] = True
        except Exception as e:
            STATE["ingest_error"] = str(e)
            STATE["ingest_done"]  = True
        finally:
            STATE["ingest_running"] = False

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"success": True})

@app.route("/api/ingest/status")
def api_ingest_status():
    return jsonify({
        "running": STATE["ingest_running"],
        "done":    STATE["ingest_done"],
        "error":   STATE["ingest_error"],
        "log":     STATE["ingest_log"][-150:],
    })

@app.route("/api/init", methods=["POST"])
def api_init():
    data   = request.json or {}
    modele = data.get("modele", "api")
    ok     = init_system(modele)
    return jsonify({"success": ok, "error": STATE["error"], "nom_llm": STATE["nom_llm"]})

@app.route("/api/clear_history", methods=["POST"])
def clear_history():
    global conversation_history
    conversation_history = []
    return jsonify({"success": True})

@app.route("/api/toggle_memory", methods=["POST"])
def toggle_memory():
    data = request.json or {}
    STATE["memory_enabled"] = data.get("enabled", False)
    if not STATE["memory_enabled"]:
        global conversation_history
        conversation_history = []
    return jsonify({"memory_enabled": STATE["memory_enabled"]})

@app.route("/api/quit", methods=["POST"])
def api_quit():
    def shutdown():
        time.sleep(0.5); os._exit(0)
    threading.Thread(target=shutdown, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/open-pdf", methods=["POST"])
def api_open_pdf():
    """Ouvre un PDF local via le serveur (contourne la restriction Firefox file://)."""
    data = request.json or {}
    path = data.get("path", "").strip()
    print(f"📄 open-pdf request: '{path}'")
    if not path or not os.path.isfile(path):
        print(f"❌ File not found: '{path}'")
        return jsonify({"ok": False, "error": "File not found"}), 404
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            import subprocess; subprocess.Popen(["open", path])
        else:
            import subprocess; subprocess.Popen(["xdg-open", path])
        print(f"✅ Opened: '{path}'")
        return jsonify({"ok": True})
    except Exception as e:
        print(f"❌ Error opening: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/chat", methods=["POST"])
def api_chat():
    global conversation_history
    if not STATE["ready"]:
        return jsonify({"error": "Système non initialisé."}), 503
    
    data = request.json or {}
    query = data.get("query", "").strip()
    web_search_requested = data.get("web_search", False)
    
    if not query:
        return jsonify({"error": "Question vide."}), 400

    db  = STATE["db"]
    llm = STATE["llm"]
    MAX_ARTICLES       = STATE["MAX_ARTICLES"]
    MAX_CHUNKS_ARTICLE = STATE["MAX_CHUNKS_ARTICLE"]
    K_INITIAL          = STATE["K_INITIAL"]

    def generate():
        global conversation_history
        t_start = time.time()
        try:
            mode            = detecter_mode(query)
            sources_info    = []
            articles_info   = []
            nb_web_total    = 0
            history_to_send = conversation_history[-(MAX_HISTORY*2):] if STATE["memory_enabled"] else []

            # ── Heartbeat : maintient la connexion SSE ouverte pendant le RAG ──
            # Firefox coupe après ~30s sans données → on envoie un ping toutes les 3s
            import threading, queue as _queue
            _result_queue = _queue.Queue()

            def _run_rag():
                try:
                    # -- Detection de theme --
                    themes_dispo  = lister_themes(db)
                    theme_detecte = detecter_theme_query(query, themes_dispo)

                    if mode == "auteur":
                        nom = extraire_nom_auteur(query)
                        if not nom:
                            _result_queue.put(("error", "Nom auteur non détecté")); return
                        articles_found, chunks = chercher_par_auteur(db, nom)
                        if not articles_found:
                            _result_queue.put(("error", f"No articles found for {nom}")); return
                        _result_queue.put(("ready", {
                            "articles_info": articles_found,
                            "docs": chunks,
                            "prompt_actif": build_prompt_with_history(PROMPT_RESUME, history_to_send),
                            "themes_dispo": themes_dispo,
                            "theme_detecte": theme_detecte,
                            "nb_web_total": 0,
                        }))
                    else:
                        if mode == "resume":
                            k_initial_val, max_articles, max_chunks_art, prompt_base = K_INITIAL, MAX_ARTICLES, MAX_CHUNKS_ARTICLE, PROMPT_RESUME
                        elif mode == "reference":
                            k_initial_val, max_articles, max_chunks_art, prompt_base = K_INITIAL+4, MAX_ARTICLES+2, max(3, MAX_CHUNKS_ARTICLE//2), PROMPT_REFERENCE
                        else:
                            k_initial_val, max_articles, max_chunks_art, prompt_base = K_INITIAL, MAX_ARTICLES, MAX_CHUNKS_ARTICLE, PROMPT_QUESTION

                        prompt_actif   = build_prompt_with_history(prompt_base, history_to_send)
                        query_enrichie = expand_query(query)

                        if web_search_requested == "only":
                            query_ss = extraire_mots_cles_llm(query, llm)
                            docs_web, nb_web_total_local = chercher_semantic_scholar(query_ss, limit=10)
                            if not docs_web:
                                _result_queue.put(("error", "No results on Semantic Scholar.")); return
                            docs = docs_web
                            ai_local = [{"filename": "🌐 WEB: "+d.metadata.get("titre",""), "auteur": d.metadata.get("auteur","Inconnu"), "annee": d.metadata.get("annee","n.d."), "titre": d.metadata.get("titre",""), "nb_chunks": 1, "score": 0} for d in docs_web]
                        else:
                            ma_eff  = max(2, max_articles // 2) if (web_search_requested and STATE["modele"] != "api") else max_articles
                            mca_eff = max(3, max_chunks_art // 2) if (web_search_requested and STATE["modele"] != "api") else max_chunks_art
                            bm25_w = float(cfg.get("bm25_weight", 0.3))
                            ai_local, docs = recuperer_articles_complets(
                                db, query_enrichie,
                                bm25_retriever=STATE["bm25"], bm25_weight=bm25_w,
                                k_initial=k_initial_val, max_articles=ma_eff,
                                max_chunks_par_article=mca_eff,
                                reranker=STATE["reranker"],
                            )
                            nb_web_total_local = 0
                            if web_search_requested:
                                query_ss = extraire_mots_cles_llm(query, llm)
                                docs_web, nb_web_total_local = chercher_semantic_scholar(query_ss)
                                if docs_web:
                                    docs.extend(docs_web)
                                    ai_local += [{"filename": "🌐 WEB: "+d.metadata.get("titre",""), "auteur": d.metadata.get("auteur","Inconnu"), "annee": d.metadata.get("annee","n.d."), "titre": d.metadata.get("titre",""), "nb_chunks": 1, "score": 0} for d in docs_web]
                            if not docs:
                                _result_queue.put(("error", "No excerpts found.")); return

                        _result_queue.put(("ready", {
                            "articles_info": ai_local,
                            "docs": docs,
                            "prompt_actif": prompt_actif,
                            "themes_dispo": themes_dispo,
                            "theme_detecte": theme_detecte,
                            "nb_web_total": nb_web_total_local,
                        }))
                except Exception as e:
                    _result_queue.put(("error", str(e)))

            # Lancer le RAG en arrière-plan
            rag_thread = threading.Thread(target=_run_rag, daemon=True)
            rag_thread.start()

            # Envoyer des heartbeats pendant que le RAG tourne
            while _result_queue.empty():
                yield ": keepalive\n\n"
                rag_thread.join(timeout=3)

            status, payload = _result_queue.get()
            if status == "error":
                yield f"data: {json.dumps({'error': payload})}\n\n"; return

            articles_info  = payload["articles_info"]
            docs           = payload["docs"]
            prompt_actif   = payload["prompt_actif"]
            themes_dispo   = payload["themes_dispo"]
            theme_detecte  = payload["theme_detecte"]
            nb_web_total   = payload["nb_web_total"]

            contexte  = format_docs(docs)
            rag_chain = (
                {"context": lambda _: contexte, "input": RunnablePassthrough()}
                | prompt_actif | llm | StrOutputParser()
            )

            full_response = ""
            for chunk in rag_chain.stream(query):
                if chunk:
                    full_response += chunk
                    yield f"data: {json.dumps({'token': chunk})}\n\n"
                    sys.stdout.flush()

            if STATE["memory_enabled"]:
                conversation_history.append({"role": "user",      "content": query})
                conversation_history.append({"role": "assistant", "content": full_response[:800]})
                if len(conversation_history) > MAX_HISTORY * 2:
                    conversation_history = conversation_history[-(MAX_HISTORY*2):]

            seen = set()
            for doc in docs:
                m   = doc.metadata
                key = (m.get("auteur","?"), m.get("annee","?"))
                if key not in seen:
                    seen.add(key)
                    pdf_path = m.get("source", "")
                    # Construire un lien file:// pour ouvrir le PDF directement dans Firefox
                    file_link = ""
                    if pdf_path and not pdf_path.startswith("http") and os.path.isfile(pdf_path):
                        import urllib.parse
                        normalized = pdf_path.replace("\\", "/")
                        if not normalized.startswith("/"):
                            normalized = "/" + normalized
                        file_link = "file://" + urllib.parse.quote(normalized, safe=":/")
                    sources_info.append({
                        "auteur":     m.get("auteur","?"),
                        "annee":      m.get("annee","?"),
                        "titre":      m.get("titre", m.get("source","?")),
                        "section":    m.get("section","?"),
                        "url":        pdf_path,
                        "doi":        m.get("doi", ""),
                        "zotero_link": file_link,
                    })

            articles_out = [{"auteur": a.get("auteur","?"), "annee": a.get("annee","?"),
                             "titre": (a.get("titre") or a.get("filename","?"))[:80],
                             "nb_chunks": a.get("nb_chunks",0), "score": a.get("score",0)}
                            for a in articles_info]

            chars_history = sum(len(m["content"]) for m in history_to_send)
            chars_in = len(contexte) + len(query) + chars_history
            tokens_in = int(chars_in / 3.5) + 150 
            tokens_out = int(len(full_response) / 3.5)

            yield f"data: {json.dumps({'done': True, 'mode': mode, 'theme_detected': theme_detecte or '', 'themes_available': themes_dispo, 'articles': articles_out, 'sources': sources_info, 'nom_llm': STATE['nom_llm'], 'elapsed': round(time.time()-t_start,1), 'history_count': len(conversation_history)//2, 'tokens_in': tokens_in, 'tokens_out': tokens_out, 'nb_web_total': nb_web_total})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no","Transfer-Encoding":"chunked"})


# ── HTML ──────────────────────────────────────────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RefChat — Document Research</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Figtree:wght@300;400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
<style>
  :root {
    --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d;
    --accent:#58a6ff; --accent2:#3fb950; --accent3:#d29922;
    --text:#e6edf3; --text2:#8b949e; --text3:#6e7681;
    --radius:12px;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:'Figtree',sans-serif; height:100vh; display:flex; flex-direction:column; overflow:hidden; }

  /* Header */
  header { display:flex; align-items:center; justify-content:space-between; padding:14px 24px; background:var(--bg2); border-bottom:1px solid var(--border); flex-shrink:0; gap:16px; }
  .header-left { display:flex; align-items:center; gap:12px; }
  .logo { font-family:'DM Serif Display',serif; font-size:1.4rem; color:var(--accent); letter-spacing:-0.5px; }
  .logo span { color:var(--text2); font-style:italic; font-size:1rem; }
  #status-dot { width:9px; height:9px; border-radius:50%; background:var(--text3); flex-shrink:0; transition:background 0.4s; }
  #status-dot.ready   { background:var(--accent2); box-shadow:0 0 8px var(--accent2); }
  #status-dot.error   { background:#f85149; }
  #status-dot.loading { background:var(--accent3); animation:pulse 1s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
  #status-text { font-size:0.78rem; color:var(--text2); }
  #memory-badge { display:inline-flex; align-items:center; gap:5px; background:var(--bg3); border:1px solid var(--border); color:var(--text3); font-size:0.72rem; padding:3px 10px; border-radius:20px; font-family:'DM Mono',monospace; transition:all 0.3s; cursor:pointer; }
  #memory-badge.active { border-color:#bc8cff44; color:#bc8cff; background:#2a1a3a; }
  #memory-badge .mem-dot { width:6px; height:6px; border-radius:50%; background:var(--text3); }
  #memory-badge.active .mem-dot { background:#bc8cff; box-shadow:0 0 6px #bc8cff; }
  .model-select { display:flex; align-items:center; gap:8px; }
  .model-select label { font-size:0.78rem; color:var(--text2); }
  select { background:var(--bg3); border:1px solid var(--border); color:var(--text); font-size:0.8rem; padding:5px 10px; border-radius:6px; cursor:pointer; font-family:'Figtree',sans-serif; }
  select:focus { outline:2px solid var(--accent); }
  #btn-init { background:var(--accent); color:#0d1117; border:none; padding:6px 14px; border-radius:6px; font-size:0.8rem; font-weight:600; cursor:pointer; font-family:'Figtree',sans-serif; transition:opacity 0.2s; }
  #btn-init:hover { opacity:0.85; } #btn-init:disabled { opacity:0.4; cursor:not-allowed; }

  /* Layout */
  .main { display:flex; flex:1; overflow:hidden; }
  #chat-area { flex:1; display:flex; flex-direction:column; overflow:hidden; }
  #messages { flex:1; overflow-y:auto; padding:24px 0; scroll-behavior:smooth; }
  #messages::-webkit-scrollbar { width:6px; }
  #messages::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }

  /* Messages */
  .message { display:flex; gap:14px; padding:16px 24px; animation:fadeIn 0.3s ease; max-width:900px; margin:0 auto; width:100%; }
  @keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
  .message.user { flex-direction:row-reverse; }
  .avatar { width:34px; height:34px; border-radius:50%; flex-shrink:0; display:flex; align-items:center; justify-content:center; font-size:0.9rem; }
  .avatar.user-av { background:#1f6feb; }
  .avatar.bot-av  { background:#21262d; border:1px solid var(--border); }
  .bubble { max-width:calc(100% - 50px); padding:14px 18px; border-radius:var(--radius); font-size:0.92rem; line-height:1.65; }
  .message.user .bubble { background:#1c2842; border:1px solid #1f6feb44; color:var(--text); }
  .message.bot .bubble  { background:var(--bg2); border:1px solid var(--border); color:var(--text); }

  /* Markdown */
  .bubble h1,.bubble h2,.bubble h3 { font-family:'DM Serif Display',serif; color:var(--accent); margin:1em 0 0.4em; }
  .bubble p { margin:0.5em 0; } .bubble ul,.bubble ol { padding-left:1.4em; margin:0.5em 0; }
  .bubble li { margin:0.25em 0; }
  .bubble code { background:var(--bg3); padding:1px 5px; border-radius:4px; font-family:'DM Mono',monospace; font-size:0.85em; color:#f0883e; }
  .bubble pre { background:var(--bg3); padding:12px; border-radius:8px; overflow-x:auto; margin:0.75em 0; border:1px solid var(--border); }
  .bubble pre code { background:none; color:var(--text); padding:0; }
  .bubble strong { color:#fff; font-weight:600; } .bubble em { color:var(--text2); }
  .bubble blockquote { border-left:3px solid var(--accent); padding-left:12px; color:var(--text2); margin:0.5em 0; font-style:italic; }
  .bubble table { border-collapse:collapse; width:100%; margin:0.75em 0; }
  .bubble th { background:var(--bg3); padding:6px 12px; text-align:left; font-size:0.85rem; }
  .bubble td { padding:5px 12px; border-top:1px solid var(--border); font-size:0.85rem; }

  .mode-badge { display:inline-block; font-size:0.7rem; font-weight:600; padding:2px 8px; border-radius:20px; margin-bottom:8px; font-family:'DM Mono',monospace; text-transform:uppercase; letter-spacing:0.5px; }
  .mode-question  { background:#1f3a5f; color:var(--accent); }
  .mode-resume    { background:#1a3a2a; color:var(--accent2); }
  .mode-reference { background:#3a2a1a; color:var(--accent3); }
  .mode-auteur    { background:#2a1a3a; color:#bc8cff; }
  .sources { margin-top:12px; padding-top:10px; border-top:1px solid var(--border); }
  .sources-title { font-size:0.72rem; color:var(--text3); margin-bottom:6px; font-family:'DM Mono',monospace; text-transform:uppercase; }
  .source-tag { display:inline-block; margin:2px; background:var(--bg3); border:1px solid var(--border); color:var(--text2); font-size:0.72rem; padding:2px 8px; border-radius:4px; font-family:'DM Mono',monospace; }
  
  .elapsed-time { margin-top:8px; font-size:0.7rem; color:var(--text3); font-family:'DM Mono',monospace; border-top:1px solid var(--border); padding-top:6px; display:flex; justify-content:space-between; align-items:center; gap:10px; }
  .elapsed-right { display:flex; align-items:center; gap:10px; }
  .tokens-badge { display:inline-block; padding:2px 6px; background:#1c2842; border:1px solid #1f6feb44; border-radius:4px; font-size:0.65rem; color:#58a6ff; font-family:'DM Mono',monospace; }
  .bubble-actions { display:flex; gap:6px; align-items:center; }
  .btn-copy, .btn-export { display:inline-flex; align-items:center; gap:4px; background:var(--bg3); border:1px solid var(--border); color:var(--text3); font-size:0.68rem; padding:3px 9px; border-radius:6px; cursor:pointer; font-family:'DM Mono',monospace; transition:all 0.2s; white-space:nowrap; }
  .btn-copy:hover { border-color:var(--accent); color:var(--accent); }
  .btn-export:hover { border-color:var(--accent2); color:var(--accent2); }
  .btn-copy.copied { border-color:var(--accent2); color:var(--accent2); }
  .source-tag.zotero-link { cursor:pointer; text-decoration:none; transition:all 0.2s; }
  .source-tag.zotero-link:hover { border-color:#e06c3388; color:#e06c33; background:#2a1a0a; }
  
  .articles-info { margin-top:8px; padding:8px 12px; background:var(--bg3); border-radius:8px; border:1px solid var(--border); }
  .articles-info-title { font-size:0.7rem; color:var(--text3); margin-bottom:4px; font-family:'DM Mono',monospace; text-transform:uppercase; }
  .article-row { font-size:0.75rem; color:var(--text2); padding:2px 0; display:flex; justify-content:space-between; border-bottom:1px solid var(--border); }
  .article-row:last-child { border-bottom:none; }
  .thinking { display:flex; align-items:center; gap:8px; color:var(--text3); font-size:0.85rem; padding:16px 24px; max-width:900px; margin:0 auto; width:100%; }
  .dots span { display:inline-block; width:6px; height:6px; background:var(--accent); border-radius:50%; animation:bounce 1.2s infinite; }
  .dots span:nth-child(2) { animation-delay:0.2s; } .dots span:nth-child(3) { animation-delay:0.4s; }
  @keyframes bounce { 0%,80%,100%{transform:translateY(0)} 40%{transform:translateY(-6px)} }
  .error-msg { background:#2d1515; border:1px solid #f8514933; color:#f85149; padding:12px 16px; border-radius:8px; font-size:0.88rem; }

  /* Input */
  #input-area { padding:16px 24px; background:var(--bg2); border-top:1px solid var(--border); flex-shrink:0; }
  .input-wrapper { display:flex; gap:10px; max-width:900px; margin:0 auto; align-items:flex-end; }
  #query-input { flex:1; background:var(--bg3); border:1px solid var(--border); color:var(--text); font-family:'Figtree',sans-serif; font-size:0.92rem; padding:12px 16px; border-radius:10px; resize:none; min-height:46px; max-height:160px; line-height:1.5; transition:border-color 0.2s; }
  #query-input:focus { outline:2px solid var(--accent); border-color:transparent; }
  #query-input::placeholder { color:var(--text3); }
  #btn-send { background:var(--accent); color:#0d1117; border:none; width:46px; height:46px; border-radius:10px; font-size:1.1rem; cursor:pointer; display:flex; align-items:center; justify-content:center; flex-shrink:0; transition:opacity 0.2s,transform 0.1s; }
  #btn-send:hover { opacity:0.85; } #btn-send:active { transform:scale(0.95); }
  #btn-send:disabled { opacity:0.3; cursor:not-allowed; }
  #btn-send.stop { background:#f85149; color:#fff; }
  .hint { font-size:0.72rem; color:var(--text3); text-align:center; margin-top:8px; max-width:900px; margin-left:auto; margin-right:auto; }

  /* Sidebar */
  #sidebar { width:240px; background:var(--bg2); border-left:1px solid var(--border); padding:20px 16px; display:flex; flex-direction:column; gap:16px; overflow-y:auto; flex-shrink:0; }
  .sidebar-section-title { font-size:0.68rem; color:var(--text3); font-family:'DM Mono',monospace; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:6px; }
  .example-btn { display:block; width:100%; background:var(--bg3); border:1px solid var(--border); color:var(--text2); padding:8px 10px; border-radius:8px; font-size:0.78rem; cursor:pointer; text-align:left; font-family:'Figtree',sans-serif; margin-bottom:6px; transition:border-color 0.2s,color 0.2s; }
  .example-btn:hover { border-color:var(--accent); color:var(--text); }
  .stats-row { display:flex; justify-content:space-between; align-items:center; margin-top:4px; }
  .stats-label { font-size:0.75rem; color:var(--text3); }
  .stats-val { font-size:0.78rem; color:var(--text2); font-family:'DM Mono',monospace; }
  .action-btn { background:transparent; border:1px solid var(--border); color:var(--text3); padding:6px 12px; border-radius:6px; font-size:0.78rem; cursor:pointer; width:100%; font-family:'Figtree',sans-serif; transition:border-color 0.2s,color 0.2s; margin-bottom:6px; text-align:left; }
  .action-btn:hover { border-color:var(--accent2); color:var(--accent2); }
  .action-btn.danger:hover { border-color:#f85149; color:#f85149; }

  @media (max-width:700px) { #sidebar { display:none; } }

  /* ═══ WIZARD PREMIER LANCEMENT ═══════════════════════════════════════════ */
  #wizard-overlay { position:fixed; inset:0; background:rgba(0,0,0,0.85); z-index:200; display:flex; align-items:center; justify-content:center; }
  #wizard-overlay.hidden { display:none; }
  .wizard-box { background:var(--bg2); border:1px solid var(--border); border-radius:16px; padding:36px 40px; width:560px; max-width:95vw; }
  .wizard-box h2 { font-family:'DM Serif Display',serif; font-size:1.6rem; color:var(--text); margin-bottom:6px; }
  .wizard-box .wizard-sub { color:var(--text2); font-size:0.9rem; margin-bottom:28px; }
  .wizard-step { display:none; }
  .wizard-step.active { display:block; }
  .wiz-label { font-size:0.8rem; color:var(--text2); font-family:'DM Mono',monospace; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px; display:block; }
  .wiz-input { width:100%; background:var(--bg3); border:1px solid var(--border); color:var(--text); font-size:0.9rem; padding:11px 14px; border-radius:8px; font-family:'Figtree',sans-serif; }
  .wiz-input:focus { outline:2px solid var(--accent); border-color:transparent; }
  .wiz-hint { font-size:0.73rem; color:var(--text3); margin-top:5px; }
  .wiz-actions { display:flex; gap:10px; margin-top:24px; }
  .wiz-btn-primary { background:var(--accent); color:#0d1117; border:none; padding:10px 22px; border-radius:8px; font-size:0.88rem; font-weight:600; cursor:pointer; font-family:'Figtree',sans-serif; }
  .wiz-btn-primary:hover { opacity:0.85; } .wiz-btn-primary:disabled { opacity:0.4; cursor:not-allowed; }
  .wiz-btn-secondary { background:transparent; border:1px solid var(--border); color:var(--text2); padding:10px 22px; border-radius:8px; font-size:0.88rem; cursor:pointer; font-family:'Figtree',sans-serif; }
  .wiz-btn-secondary:hover { border-color:var(--accent); color:var(--text); }
  .model-card { display:block; background:var(--bg3); border:1px solid var(--border); border-radius:8px; padding:10px 14px; cursor:pointer; transition:border-color 0.2s,background 0.2s; user-select:none; }
  .model-card:hover { border-color:var(--accent); }
  .model-card.selected { border-color:var(--accent2); background:#0d1f17; }
  .wiz-detected { background:var(--bg3); border:1px solid var(--accent2); border-radius:8px; padding:10px 14px; font-size:0.82rem; color:var(--accent2); margin-bottom:12px; font-family:'DM Mono',monospace; word-break:break-all; }

  /* ═══ LOG D'INGESTION (dans le chat) ═════════════════════════════════════ */
  .ingest-panel { background:var(--bg2); border:1px solid var(--border); border-radius:var(--radius); padding:16px 20px; max-width:900px; margin:0 auto 16px; width:calc(100% - 48px); }
  .ingest-header { display:flex; align-items:center; gap:10px; margin-bottom:12px; }
  .ingest-title { font-size:0.9rem; font-weight:600; color:var(--text); }
  .ingest-spinner { width:14px; height:14px; border:2px solid var(--border); border-top-color:var(--accent); border-radius:50%; animation:spin 0.8s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .ingest-log-box { background:var(--bg3); border:1px solid var(--border); border-radius:8px; padding:10px 14px; font-family:'DM Mono',monospace; font-size:0.73rem; color:var(--text2); height:220px; overflow-y:auto; white-space:pre-wrap; line-height:1.6; }
  .ingest-log-box::-webkit-scrollbar { width:4px; }
  .ingest-log-box::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }
  .ingest-progress { margin-top:10px; height:4px; background:var(--bg3); border-radius:2px; overflow:hidden; }
  .ingest-progress-bar { height:100%; background:var(--accent); width:0%; transition:width 0.3s; border-radius:2px; animation:indeterminate 1.5s infinite; }
  @keyframes indeterminate { 0%{transform:translateX(-100%)} 100%{transform:translateX(400%)} }
  .ingest-done { color:var(--accent2); } .ingest-error-txt { color:#f85149; }

  /* ═══ MODAL PARAMÈTRES ════════════════════════════════════════════════════ */
  #settings-modal { position:fixed; inset:0; background:rgba(0,0,0,0.75); z-index:150; display:none; align-items:center; justify-content:center; }
  #settings-modal.open { display:flex; }
  .settings-box { background:var(--bg2); border:1px solid var(--border); border-radius:16px; padding:28px 32px; width:500px; max-width:95vw; max-height:90vh; overflow-y:auto; }
  .settings-box h3 { font-family:'DM Serif Display',serif; font-size:1.2rem; margin-bottom:18px; }
  .sfield { margin-bottom:16px; }
  .sfield label { display:block; font-size:0.78rem; color:var(--text2); font-family:'DM Mono',monospace; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:5px; }
  .sfield input { width:100%; background:var(--bg3); border:1px solid var(--border); color:var(--text); font-size:0.88rem; padding:9px 12px; border-radius:8px; font-family:'Figtree',sans-serif; }
  .sfield input:focus { outline:2px solid var(--accent); border-color:transparent; }
  .sfield .shint { font-size:0.71rem; color:var(--text3); margin-top:3px; }
  .s-actions { display:flex; gap:10px; margin-top:20px; flex-wrap:wrap; }
  .s-btn { padding:9px 18px; border-radius:8px; font-size:0.84rem; cursor:pointer; font-family:'Figtree',sans-serif; border:none; }
  .s-btn.primary { background:var(--accent); color:#0d1117; font-weight:600; }
  .s-btn.primary:hover { opacity:0.85; }
  .s-btn.secondary { background:transparent; border:1px solid var(--border); color:var(--text2); }
  .s-btn.secondary:hover { border-color:var(--accent); color:var(--text); }
  .s-btn.success { background:#1a3a2a; border:1px solid var(--accent2); color:var(--accent2); }
  .s-sep { border:none; border-top:1px solid var(--border); margin:20px 0; }

  /* Notif toast */
  .toast { position:fixed; bottom:24px; right:24px; padding:12px 20px; border-radius:8px; font-size:0.84rem; z-index:999; font-family:'Figtree',sans-serif; box-shadow:0 4px 20px rgba(0,0,0,0.4); pointer-events:none; }
  .toast.ok  { background:#1a3a2a; border:1px solid #3fb95066; color:var(--accent2); }
  .toast.err { background:#2d1515; border:1px solid #f8514966; color:#f85149; }
</style>
</head>
<body>

<div id="wizard-overlay" class="hidden">
  <div class="wizard-box">
    <h2>📚 Welcome to RefChat</h2>
    <p class="wizard-sub">Initial setup — takes about 2 minutes</p>

    <div class="wizard-step active" id="wiz-step-0">
      <div id="wiz-ollama-checking" style="text-align:center;padding:20px 0;">
        <div style="font-size:2rem;margin-bottom:10px">⏳</div>
        <div style="color:var(--text2);font-size:0.9rem">Checking Ollama…</div>
      </div>
      <div id="wiz-ollama-ok" style="display:none">
        <div style="background:var(--bg3);border:1px solid var(--accent2);border-radius:8px;padding:10px 14px;margin-bottom:16px;">
          <div style="color:var(--accent2);font-size:0.85rem;font-weight:600;margin-bottom:6px">✅ Ollama detected and running</div>
          <div id="wiz-models-status" style="font-size:0.75rem;color:var(--text2);font-family:'DM Mono',monospace;line-height:1.6"></div>
        </div>

        <div id="wiz-model-selection" style="margin-bottom:16px">
          <label class="wiz-label">🤖 Choose models to install</label>
          <div style="display:flex;flex-direction:column;gap:8px;margin-top:4px">

            <label id="card-q4" class="model-card selected" onclick="toggleModel('mistral:7b-instruct-q4_0', this)">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                  <span style="font-weight:600;color:var(--text)">⚡ Mistral 7B Q4 — GPU</span>
                  <span id="check-q4" style="color:var(--accent2);margin-left:8px">✓</span>
                </div>
                <span style="font-family:'DM Mono',monospace;font-size:0.75rem;color:var(--accent3)">~4 GB</span>
              </div>
              <div style="font-size:0.75rem;color:var(--text2);margin-top:3px">
                Recommended · Q4 quantized · Fast on GPU · Ideal for most setups
              </div>
            </label>

            <label id="card-mistral" class="model-card" onclick="toggleModel('mistral', this)">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                  <span style="font-weight:600;color:var(--text)">🖥️ Mistral 7B — CPU</span>
                  <span id="check-mistral" style="color:var(--accent2);margin-left:8px;display:none">✓</span>
                </div>
                <span style="font-family:'DM Mono',monospace;font-size:0.75rem;color:var(--accent3)">~4.1 GB</span>
              </div>
              <div style="font-size:0.75rem;color:var(--text2);margin-top:3px">
                No GPU · Slower · Compatible with all machines · RAM: 8 GB minimum
              </div>
            </label>

            <label id="card-mixtral" class="model-card" onclick="toggleModel('mixtral', this)">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                  <span style="font-weight:600;color:var(--text)">🧠 Mixtral 8x7B — CPU</span>
                  <span id="check-mixtral" style="color:var(--accent2);margin-left:8px;display:none">✓</span>
                </div>
                <span style="font-family:'DM Mono',monospace;font-size:0.75rem;color:var(--accent3)">~26 GB</span>
              </div>
              <div style="font-size:0.75rem;color:var(--text2);margin-top:3px">
                Very powerful · Slow without GPU · RAM: 32 GB minimum · Long download
              </div>
            </label>

          </div>
          <div style="font-size:0.72rem;color:var(--text3);margin-top:8px">
            💡 You can always install more models later via <code>ollama pull</code>
          </div>
        </div>

        <div id="wiz-pull-progress" style="display:none;margin-bottom:16px">
          <div style="font-size:0.85rem;color:var(--text2);margin-bottom:8px">📥 Downloading…</div>
          <div id="wiz-pull-log" style="background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:10px 14px;font-family:'DM Mono',monospace;font-size:0.73rem;color:var(--text2);height:80px;overflow-y:auto;white-space:pre-wrap;line-height:1.6"></div>
          <div class="ingest-progress" style="margin-top:8px"><div class="ingest-progress-bar" id="wiz-pull-bar"></div></div>
        </div>

        <div class="wiz-actions">
          <button class="wiz-btn-primary" id="wiz-btn-install" onclick="wizInstallModels()">📥 Install selected models →</button>
          <button class="wiz-btn-secondary" onclick="wizShowHw()">Skip — install later</button>
        </div>
      </div>
      <div id="wiz-ollama-missing" style="display:none">
        <div style="background:#2d1515;border:1px solid #f8514933;border-radius:8px;padding:14px 16px;margin-bottom:16px;">
          <div style="color:#f85149;font-size:0.9rem;font-weight:600;margin-bottom:8px">⚠️ Ollama not detected</div>
          <div style="color:var(--text2);font-size:0.83rem;line-height:1.6;margin-bottom:10px">
            RefChat requires <strong>Ollama</strong> to run language models locally.<br>
            Install it, then relaunch RefChat via <code>RefChat.bat</code> — it will handle the rest automatically.
          </div>
          <a href="https://ollama.com/download" target="_blank"
             style="display:inline-block;background:var(--accent);color:#0d1117;padding:8px 16px;border-radius:6px;font-size:0.85rem;font-weight:600;text-decoration:none">
            ⬇️ Download Ollama
          </a>
        </div>
        <div class="wiz-hint" style="margin-bottom:12px">
          After installation, restart RefChat via <code>RefChat.bat</code> — models will be downloaded automatically.
        </div>
        <div class="wiz-actions">
          <button class="wiz-btn-secondary" onclick="wizCheckOllama()">🔄 Re-check</button>
          <button class="wiz-btn-secondary" onclick="wizShowHw()">Continue anyway →</button>
        </div>
      </div>
    </div>

    <!-- ── STEP 1 — Hardware profile ── -->
    <div class="wizard-step" id="wiz-step-hw">
      <p style="color:var(--text2);font-size:0.88rem;margin-bottom:18px;line-height:1.6">
        RefChat needs to know your hardware to set optimal parameters for local models.<br>
        <strong style="color:var(--text)">num_thread</strong> = CPU cores Ollama uses ·
        <strong style="color:var(--text)">num_gpu</strong> = GPU layers offloaded ·
        <strong style="color:var(--text)">num_batch</strong> = prompt batch size
      </p>

      <div id="wiz-hw-detect-status" style="text-align:center;padding:14px 0;color:var(--text3);font-size:0.85rem;font-family:'DM Mono',monospace">
        ⏳ Detecting your hardware…
      </div>

      <div id="wiz-hw-summary" style="display:none;background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:12px 16px;margin-bottom:16px;font-size:0.8rem;font-family:'DM Mono',monospace;color:var(--text2);line-height:2"></div>

      <div id="wiz-hw-params" style="display:none">
        <div style="background:rgba(88,166,255,0.07);border:1px solid #58a6ff44;border-radius:8px;padding:10px 14px;margin-bottom:14px;font-size:0.78rem;color:var(--text);line-height:1.6">
          💡 <strong style="color:var(--accent)">How it works:</strong>
          Set <code>num_gpu=99</code> if you have a GPU (offloads everything, fastest).
          Use <code>num_gpu=0</code> for CPU-only.
          <code>num_thread</code> = ~75% of your CPU cores.
          <code>num_batch</code> = larger → faster but more RAM needed.
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:10px">
          <div>
            <label style="display:block;font-size:0.72rem;color:var(--text3);font-family:'DM Mono',monospace;text-transform:uppercase;letter-spacing:.5px;margin-bottom:5px">num_thread</label>
            <input type="number" id="wiz-hw-thread" min="1" max="256" value="4"
                   style="width:100%;background:var(--bg);border:1px solid var(--border);color:var(--text);padding:8px;border-radius:6px;font-family:'DM Mono',monospace;font-size:0.9rem;text-align:center">
            <div style="font-size:0.68rem;color:var(--text3);margin-top:3px">CPU threads for Ollama</div>
          </div>
          <div>
            <label style="display:block;font-size:0.72rem;color:var(--text3);font-family:'DM Mono',monospace;text-transform:uppercase;letter-spacing:.5px;margin-bottom:5px">num_gpu</label>
            <input type="number" id="wiz-hw-gpu" min="0" max="99" value="99"
                   style="width:100%;background:var(--bg);border:1px solid var(--border);color:var(--text);padding:8px;border-radius:6px;font-family:'DM Mono',monospace;font-size:0.9rem;text-align:center">
            <div style="font-size:0.68rem;color:var(--text3);margin-top:3px">99 = all on GPU · 0 = CPU</div>
          </div>
          <div>
            <label style="display:block;font-size:0.72rem;color:var(--text3);font-family:'DM Mono',monospace;text-transform:uppercase;letter-spacing:.5px;margin-bottom:5px">num_batch</label>
            <input type="number" id="wiz-hw-batch" min="64" max="4096" step="64" value="512"
                   style="width:100%;background:var(--bg);border:1px solid var(--border);color:var(--text);padding:8px;border-radius:6px;font-family:'DM Mono',monospace;font-size:0.9rem;text-align:center">
            <div style="font-size:0.68rem;color:var(--text3);margin-top:3px">Prompt batch size</div>
          </div>
        </div>
        <div style="font-size:0.7rem;color:var(--text3)">
          Not sure? Keep the recommended values — you can always change them later in ⚙️ Settings.
        </div>
      </div>

      <div class="wiz-actions" style="margin-top:20px">
        <button class="wiz-btn-primary" id="wiz-hw-save-btn" onclick="wizHwSave()" style="display:none">Save & Continue →</button>
        <button class="wiz-btn-secondary" onclick="wizShowStep(1)">Skip — use defaults</button>
      </div>
    </div>


    <div class="wizard-step" id="wiz-step-1">
      
      <div style="background: rgba(88, 166, 255, 0.1); border: 1px solid var(--accent); border-radius: 8px; padding: 14px 16px; margin-bottom: 20px;">
        <div style="color: var(--accent); font-size: 0.95rem; font-weight: 600; margin-bottom: 8px;">🐳 Prerequisite: Docker Desktop</div>
        <div style="color: var(--text); font-size: 0.85rem; line-height: 1.5;">
          To intelligently parse your scientific PDFs, RefChat uses <strong>GROBID</strong>.<br>
          👉 Please <strong>open Docker Desktop</strong> before continuing — RefChat will start it automatically!
        </div>
      </div>

      <div id="wiz-auto-detect" class="wiz-detected" style="display:none"></div>
      <label class="wiz-label">📁 Zotero Storage Folder</label>
      <input class="wiz-input" type="text" id="wiz-zotero"
             placeholder="e.g. C:\Users\YourName\Zotero\storage" />
      <div class="wiz-hint">
        The folder containing subfolders like <code>ABCD1234/</code> with your PDFs.<br>
        Typical location: <code>C:\Users\[your name]\Zotero\storage</code>
      </div>
      <div class="wiz-actions">
        <button class="wiz-btn-primary" onclick="wizNext()">Next →</button>
      </div>
    </div>

    <div class="wizard-step" id="wiz-step-2">
      <label class="wiz-label">🔑 Mistral API Key <span style="color:var(--text3)">(optional)</span></label>
      <input class="wiz-input" type="password" id="wiz-apikey"
             placeholder="Leave empty to use only local models" />
      <div class="wiz-hint">
        For the fast cloud mode (☁️ Mistral API). Create a key on <strong>console.mistral.ai</strong>
      </div>
      <div class="wiz-actions">
        <button class="wiz-btn-primary" onclick="wizSaveAndIndex()">💾 Save & Index →</button>
        <button class="wiz-btn-secondary" onclick="wizSaveAndIndex(true)">Skip — index later</button>
      </div>
    </div>

    <div class="wizard-step" id="wiz-step-3">
      <div style="font-size:0.9rem;color:var(--text2);margin-bottom:16px">
        🔄 Indexing your Zotero library…<br>
        <span style="font-size:0.8rem;color:var(--text3)">This may take a few minutes depending on your library size.</span>
      </div>

      <div style="background: rgba(210, 153, 34, 0.1); border: 1px solid var(--accent3); border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; font-size: 0.82rem; color: var(--text); line-height: 1.5;">
        💡 <strong>Need to pause?</strong><br>
        Go to the terminal window (black screen) and press <code>Ctrl + C</code> to stop cleanly. Relaunch RefChat later — indexing will resume <strong>exactly</strong> where it left off!
      </div>

      <div class="ingest-log-box" id="wiz-log">Starting…</div>
      <div class="ingest-progress"><div class="ingest-progress-bar" id="wiz-bar"></div></div>
      <div id="wiz-done-msg" style="display:none;margin-top:14px;color:var(--accent2);font-size:0.88rem"></div>
    </div>

    <div class="wizard-step" id="wiz-step-4">
      <div style="text-align:center;padding:12px 0">
        <div style="font-size:3rem;margin-bottom:12px">✅</div>
        <div style="font-size:1.1rem;font-weight:600;color:var(--text);margin-bottom:6px">RefChat is ready!</div>
        <div id="wiz-final-count" style="color:var(--text2);font-size:0.88rem;margin-bottom:20px"></div>
      </div>
      <div class="wiz-actions" style="justify-content:center">
        <button class="wiz-btn-primary" onclick="wizFinish()">🚀 Get started</button>
      </div>
    </div>
  </div>
</div>

<div id="check-articles-modal" style="position:fixed;inset:0;background:rgba(0,0,0,0.8);z-index:160;display:none;align-items:center;justify-content:center;">
  <div style="background:var(--bg2);border:1px solid var(--border);border-radius:16px;padding:28px 32px;width:560px;max-width:95vw;max-height:80vh;display:flex;flex-direction:column;gap:0;">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
      <h3 style="font-family:'DM Serif Display',serif;font-size:1.2rem;color:var(--text)">☑ Check articles before indexing</h3>
      <button onclick="closeCheckArticlesModal()" style="background:transparent;border:none;color:var(--text3);font-size:1.2rem;cursor:pointer;padding:4px 8px;border-radius:4px;" onmouseover="this.style.color='var(--text)'" onmouseout="this.style.color='var(--text3)'">✕</button>
    </div>
    <p style="font-size:0.8rem;color:var(--text2);margin-bottom:16px">Check the articles you want to <strong style="color:#f85149">exclude from indexing</strong> (add to blacklist). Unchecked articles will be indexed normally.</p>

    <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
      <input type="text" id="check-articles-search" placeholder="🔍 Filter by name…" oninput="filterCheckArticles()"
             style="flex:1;background:var(--bg3);border:1px solid var(--border);color:var(--text);font-size:0.82rem;padding:7px 12px;border-radius:7px;font-family:'Figtree',sans-serif;">
      <button onclick="checkArticlesSelectAll(true)" style="background:transparent;border:1px solid var(--border);color:var(--text3);font-size:0.75rem;padding:6px 10px;border-radius:6px;cursor:pointer;font-family:'Figtree',sans-serif;white-space:nowrap" onmouseover="this.style.color='var(--text)'" onmouseout="this.style.color='var(--text3)'">All</button>
      <button onclick="checkArticlesSelectAll(false)" style="background:transparent;border:1px solid var(--border);color:var(--text3);font-size:0.75rem;padding:6px 10px;border-radius:6px;cursor:pointer;font-family:'Figtree',sans-serif;white-space:nowrap" onmouseover="this.style.color='var(--text)'" onmouseout="this.style.color='var(--text3)'">None</button>
    </div>

    <div id="check-articles-count" style="font-size:0.75rem;color:var(--text3);font-family:'DM Mono',monospace;margin-bottom:8px;"></div>

    <div id="check-articles-list" style="flex:1;overflow-y:auto;border:1px solid var(--border);border-radius:8px;background:var(--bg3);max-height:320px;">
      <div style="padding:20px;text-align:center;color:var(--text3);font-size:0.85rem">Loading…</div>
    </div>

    <div style="display:flex;gap:10px;margin-top:18px;justify-content:flex-end;">
      <button onclick="closeCheckArticlesModal()" style="background:transparent;border:1px solid var(--border);color:var(--text2);padding:9px 18px;border-radius:8px;font-size:0.84rem;cursor:pointer;font-family:'Figtree',sans-serif;" onmouseover="this.style.borderColor='var(--accent)'" onmouseout="this.style.borderColor='var(--border)'">Cancel</button>
      <button onclick="applyBlacklistAndIndex()" id="check-articles-apply" style="background:#2d1515;border:1px solid #f8514966;color:#f85149;padding:9px 18px;border-radius:8px;font-size:0.84rem;font-weight:600;cursor:pointer;font-family:'Figtree',sans-serif;" onmouseover="this.style.background='#3d1515'" onmouseout="this.style.background='#2d1515'">🚫 Blacklist selected & Index rest</button>
    </div>
  </div>
</div>

<div id="settings-modal">
  <div class="settings-box">
    <h3>⚙️ Settings</h3>
    <div class="sfield">
      <label>Zotero Storage Folder</label>
      <input type="text" id="s-zotero" placeholder="C:\Users\...\Zotero\storage" />
      <div class="shint">Subfolders ABCD1234/ containing PDFs</div>
    </div>
    <div class="sfield">
      <label>Mistral API Key</label>
      <input type="password" id="s-apikey" placeholder="Leave empty = local models only" />
      <div class="shint">console.mistral.ai → Your key</div>
    </div>
    <div class="sfield">
      <label>Semantic Scholar API Key <span style="color:var(--accent2);font-size:0.75em">free</span></label>
      <input type="password" id="s-ss-apikey" placeholder="Leave empty = public API (rate-limited)" />
      <div class="shint">
        Increases rate limits · Free at
        <a href="https://www.semanticscholar.org/product/api#api-key-form" target="_blank" style="color:var(--accent)">semanticscholar.org</a>
      </div>
    </div>
    <div class="sfield">
      <label>ChromaDB folder</label>
      <input type="text" id="s-dbpath" placeholder="path/to/chroma_db" />
      <div class="shint">Where the vector index is stored</div>
    </div>
    <hr class="s-sep">
    <h3 style="font-size:1rem;margin-bottom:6px">🖥️ Hardware / Performance</h3>
    <p style="font-size:0.78rem;color:var(--text2);margin-bottom:10px">
      Controls how Ollama uses your CPU and GPU for local models. Leave blank to use safe defaults.
      <button onclick="settingsDetectHw()" style="margin-left:8px;background:var(--bg3);border:1px solid var(--border);color:var(--accent);padding:3px 10px;border-radius:5px;font-size:0.75rem;cursor:pointer;font-family:'Figtree',sans-serif">🔍 Auto-detect</button>
    </p>
    <div id="s-hw-info" style="display:none;background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:10px 14px;margin-bottom:12px;font-size:0.78rem;color:var(--text2);font-family:'DM Mono',monospace;line-height:1.9"></div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px">
      <div class="sfield">
        <label>num_thread</label>
        <input type="number" id="s-num-thread" placeholder="auto" min="1" max="256" />
        <div class="shint">~75% of your cores</div>
      </div>
      <div class="sfield">
        <label>num_gpu</label>
        <input type="number" id="s-num-gpu" placeholder="99" min="0" max="99" />
        <div class="shint">99=GPU · 0=CPU only</div>
      </div>
      <div class="sfield">
        <label>num_batch</label>
        <input type="number" id="s-num-batch" placeholder="512" min="64" max="4096" step="64" />
        <div class="shint">Larger = faster, more RAM</div>
      </div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px">
      <div class="sfield">
        <label>num_ctx <span style="color:var(--accent2);font-size:0.7em">&#128187; LOCAL</span> <span style="color:var(--accent2);font-size:0.7em">auto-detected</span></label>
        <input type="number" id="s-num-ctx-local" placeholder="4096" min="512" max="32768" step="512" />
        <div class="shint">VRAM-limited: 4GB&#8594;4096 &middot; 8GB&#8594;8192</div>
      </div>
    </div>
    <hr class="s-sep">
    <h3 style="font-size:1rem;margin-bottom:6px">🔀 Hybrid Search BM25 + Dense</h3>
    <p style="font-size:0.78rem;color:var(--text2);margin-bottom:10px">
      Combines keyword matching (BM25) with semantic search (E5). Higher BM25 weight
      improves exact geographic/proper-noun recall. Requires <code>pip install rank-bm25</code>.
    </p>
    <div class="sfield">
      <label>BM25 weight <span style="color:var(--accent2);font-size:0.7em">0 = dense only · 1 = BM25 only</span></label>
      <input type="number" id="s-bm25-weight" placeholder="0.3" min="0" max="1" step="0.05" style="width:120px" />
      <div class="shint">Recommended: 0.3 (70% dense / 30% BM25) — takes effect on next query, no restart needed</div>
    </div>

    <div class="s-actions">
      <button class="s-btn primary" onclick="settingsSave()">💾 Save</button>
      <button class="s-btn secondary" onclick="settingsClose()">Cancel</button>
    </div>
    <hr class="s-sep">
    <h3 style="font-size:1rem;margin-bottom:12px">🔄 Re-index library</h3>
    
    <div style="background: rgba(88, 166, 255, 0.1); border: 1px solid var(--accent); border-radius: 8px; padding: 10px 14px; margin-bottom: 12px;">
      <div style="color: var(--accent); font-size: 0.85rem; font-weight: 600; margin-bottom: 4px;">🐳 Don't forget Docker</div>
      <div style="color: var(--text); font-size: 0.75rem; line-height: 1.4;">
        Open <strong>Docker Desktop</strong> before indexing to benefit from intelligent PDF parsing (GROBID).
      </div>
    </div>

    <p style="font-size:0.82rem;color:var(--text2);margin-bottom:12px">Indexes from the configured Zotero folder. Already-indexed articles are skipped.</p>
    <div class="s-actions">
      <button class="s-btn success" id="s-btn-ingest" onclick="settingsStartIngest()">▶ Start indexing</button>
      <button class="s-btn secondary" onclick="openManageDB()" style="margin-top:8px">🗄️ Manage indexed articles</button>
    </div>
    <hr class="s-sep">
    <h3 style="font-size:1rem;margin-bottom:12px">🤖 Ollama Models</h3>
    <p style="font-size:0.82rem;color:var(--text2);margin-bottom:14px">Install or verify available local models.</p>
    <div id="s-models-list" style="display:flex;flex-direction:column;gap:8px;margin-bottom:14px">
      <div style="color:var(--text3);font-size:0.8rem;font-family:'DM Mono',monospace">Loading…</div>
    </div>
    <div class="s-actions" style="flex-wrap:wrap">
      <button class="s-btn secondary" onclick="settingsRefreshModels()">🔄 Refresh</button>
    </div>
  </div>
</div>

<!-- ═══ MANAGE DB MODAL ═══════════════════════════════════════════════════ -->
<div id="manage-db-modal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.75);z-index:400;align-items:center;justify-content:center">
  <div style="background:var(--bg2);border:1px solid var(--border);border-radius:14px;width:min(720px,96vw);max-height:88vh;display:flex;flex-direction:column;padding:24px;gap:14px">
    <div style="display:flex;align-items:center;justify-content:space-between">
      <h2 style="font-family:'DM Serif Display',serif;font-size:1.2rem;color:var(--accent)">🗄️ Manage indexed articles</h2>
      <button onclick="closeManageDB()" style="background:none;border:none;color:var(--text2);cursor:pointer;font-size:1.3rem;line-height:1">✕</button>
    </div>
    <div style="display:flex;gap:10px;align-items:center">
      <input id="mdb-search" type="text" placeholder="Filter by filename, author, year…" oninput="filterManageDB()"
        style="flex:1;background:var(--bg3);border:1px solid var(--border);color:var(--text);padding:8px 12px;border-radius:8px;font-family:'Figtree',sans-serif;font-size:0.85rem">
      <span id="mdb-count" style="font-size:0.75rem;color:var(--text3);white-space:nowrap;font-family:'DM Mono',monospace"></span>
    </div>
    <div style="display:flex;gap:8px;align-items:center">
      <button onclick="mdbSelectAll()" class="s-btn secondary" style="font-size:0.73rem;padding:4px 10px">☑ All</button>
      <button onclick="mdbSelectNone()" class="s-btn secondary" style="font-size:0.73rem;padding:4px 10px">☐ None</button>
      <span id="mdb-selected-count" style="font-size:0.75rem;color:var(--text3);margin-left:auto;font-family:'DM Mono',monospace">0 selected</span>
    </div>
    <div id="mdb-list" style="flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:3px;min-height:0;max-height:420px;padding-right:4px">
      <div style="color:var(--text3);font-size:0.85rem;padding:20px;text-align:center">Loading…</div>
    </div>
    <div style="display:flex;gap:12px;align-items:center;border-top:1px solid var(--border);padding-top:14px;flex-wrap:wrap">
      <button onclick="mdbDeleteSelected()" id="mdb-btn-delete"
        style="background:#da3633;color:#fff;border:none;padding:8px 18px;border-radius:8px;font-size:0.85rem;font-weight:600;cursor:pointer;font-family:'Figtree',sans-serif;transition:opacity 0.2s"
        disabled>🗑️ Delete from DB &amp; blacklist</button>
      <span style="font-size:0.73rem;color:var(--text3);line-height:1.4">Deleted articles are added to the blacklist and won't be re-indexed.<br>Remove them from <code style="font-family:'DM Mono',monospace">refchat_ignore.txt</code> to re-enable them.</span>
    </div>
  </div>
</div>

<header>
  <div class="header-left">
    <div class="logo">RefChat <span>/ Document RAG</span></div>
    <div id="status-dot" class="loading"></div>
    <div id="status-text">Initializing…</div>
  </div>
  <div style="display:flex;align-items:center;gap:10px">
    <div id="memory-badge" onclick="toggleMemory()" title="Click to toggle memory">
      <div class="mem-dot"></div>
      <span id="mem-label">Memory OFF</span>
    </div>
    <div class="model-select">
      <label>Model:</label>
      <select id="model-select">
        <option value="">Loading…</option>
      </select>
      <button id="btn-init">Connect</button>
    </div>
  </div>
</header>

<div class="main">
  <div id="chat-area">
    <div id="messages">
      <div id="welcome" style="display:flex;flex-direction:column;align-items:center;justify-content:center;flex:1;gap:12px;padding:40px;">
        <h2 style="font-family:'DM Serif Display',serif;font-size:2rem;color:var(--text);text-align:center">Welcome to RefChat</h2>
        <p style="color:var(--text2);text-align:center;font-size:0.9rem;max-width:400px">Query your document library with natural language questions.</p>
        <div style="background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);padding:20px 24px;margin-top:8px;text-align:center;max-width:360px;width:100%;">
          <div style="color:var(--text2);font-size:0.88rem;margin-bottom:12px">
            Select a model and click <strong>Connect</strong> to start.
          </div>
        </div>
      </div>
    </div>

    <div id="input-area">
      <div class="input-wrapper">
        <textarea id="query-input" placeholder="Ask a question…" rows="1" disabled></textarea>
        <button id="btn-send" disabled>↑</button>
      </div>
      <div class="hint">Enter to send · Shift+Enter for new line</div>
    </div>
  </div>

  <div id="sidebar">
    <div>
      <div class="sidebar-section-title">Examples</div>
      <button class="example-btn" onclick="setQuery('Summarize the work on this topic')">📋 Thematic synthesis</button>
      <button class="example-btn" onclick="setQuery('Which articles deal with this topic?')">🔎 Search by topic</button>
      <button class="example-btn" onclick="setQuery('What are the main conclusions of these articles?')">🔬 Key conclusions</button>
      <button class="example-btn" onclick="setQuery('Articles by [Author Name]')">👤 Search by author</button>
      <button class="example-btn" onclick="setQuery('What methods are used in these studies?')">📋 Methods</button>
    </div>
    <div>
      <div class="sidebar-section-title">System</div>
      <div class="stats-row"><span class="stats-label">Model</span><span class="stats-val" id="stat-model">—</span></div>
      <div class="stats-row"><span class="stats-label">Chunks</span><span class="stats-val" id="stat-docs">—</span></div>
      <div class="stats-row"><span class="stats-label">Memory</span><span class="stats-val" id="stat-mem">0 / 3</span></div>
      <div class="stats-row"><span class="stats-label">Index</span><span class="stats-val" id="stat-index" title="">—</span></div>
    </div>
    <div>
      <div>
        <div class="sidebar-section-title" style="display:flex;align-items:center;justify-content:space-between;">
          <span>Themes</span>
          <span id="themes-count" style="font-family:'DM Mono',monospace;font-size:0.7rem;color:var(--text3)"></span>
        </div>
        <div id="themes-list" style="display:flex;flex-direction:column;gap:4px;margin-top:4px;max-height:160px;overflow-y:auto;">
          <div style="color:var(--text3);font-size:0.75rem;font-style:italic">Loading...</div>
        </div>
      </div>
      <div class="sidebar-section-title">Actions</div>
      <button class="action-btn" id="btn-toggle-mem" onclick="toggleMemory()">🧠 Enable memory</button>
      <button class="action-btn" onclick="clearMemory()">🧹 Clear memory</button>

      <div id="web-search-badge" onclick="toggleWebSearch()" title="Click to change mode" style="display:inline-flex; align-items:center; gap:5px; background:var(--bg3); border:1px solid var(--border); color:var(--text3); font-size:0.72rem; padding:6px 10px; border-radius:8px; font-family:'DM Mono',monospace; transition:all 0.3s; cursor:pointer; margin-bottom:6px; width:100%; justify-content:center;">
        <div class="web-dot" style="width:8px; height:8px; border-radius:50%; background:var(--text3); transition:all 0.3s;"></div>
        <span id="web-label" style="font-weight:600;">🗄️ Local only</span>
      </div>

      <div id="new-articles-badge" style="display:none;background:#1a2a3a;border:1px solid var(--accent);border-radius:8px;padding:10px 12px;margin-bottom:6px;">
        <div style="font-size:0.75rem;color:var(--accent);font-weight:600;margin-bottom:6px" id="new-articles-label">📥 0 new articles</div>
        <div id="new-articles-preview" style="font-size:0.68rem;color:var(--text3);font-family:'DM Mono',monospace;margin-bottom:8px;line-height:1.5"></div>
        <div style="display:flex;gap:6px;margin-top:2px">
          <button onclick="openIngestPanel()" style="flex:1;background:var(--accent);color:#0d1117;border:none;padding:6px 10px;border-radius:6px;font-size:0.78rem;font-weight:600;cursor:pointer;font-family:'Figtree',sans-serif">▶ Index now</button>
          <button onclick="openCheckArticlesModal()" title="Choose which articles to blacklist before indexing" style="background:var(--bg3);color:var(--text2);border:1px solid var(--border);padding:6px 10px;border-radius:6px;font-size:0.78rem;cursor:pointer;font-family:'Figtree',sans-serif;white-space:nowrap" onmouseover="this.style.borderColor='var(--accent2)';this.style.color='var(--accent2)'" onmouseout="this.style.borderColor='var(--border)';this.style.color='var(--text2)'">☑ Check</button>
        </div>
      </div>
      <button class="action-btn" onclick="openIngestPanel()">🔄 Index library</button>
      <button class="action-btn" onclick="openThemePanel()" style="border-color:var(--accent2);color:var(--accent2)">🏷️ Thématisation</button>
      <button class="action-btn danger" onclick="clearChat()">🗑 Clear conversation</button>
      <button class="action-btn" onclick="settingsOpen()" style="border-color:var(--accent3);color:var(--accent3)">⚙️ Settings</button>
      <button class="action-btn danger" onclick="quitApp()" style="margin-top:8px;border-color:#f85149;color:#f85149;font-weight:600">⏻ Quit RefChat</button>
    </div>
  </div>
</div>

<script>
marked.setOptions({ breaks:true, gfm:true });

const messagesEl  = document.getElementById('messages');
const welcomeEl   = document.getElementById('welcome');
const queryInput  = document.getElementById('query-input');
const btnSend     = document.getElementById('btn-send');
const btnInit     = document.getElementById('btn-init');
const modelSelect = document.getElementById('model-select');
const statusDot   = document.getElementById('status-dot');
const statusText  = document.getElementById('status-text');
const statModel   = document.getElementById('stat-model');
const statDocs    = document.getElementById('stat-docs');
const statMem     = document.getElementById('stat-mem');
const memBadge    = document.getElementById('memory-badge');

let isReady = false, isLoading = false, memoryEnabled = false;
let currentAbortController = null;

// 3 états : false=local seul | true=hybride | "only"=web seul
let webSearchEnabled = false;

function toggleWebSearch() {
  if      (webSearchEnabled === false) webSearchEnabled = true;
  else if (webSearchEnabled === true)  webSearchEnabled = "only";
  else                                 webSearchEnabled = false;

  const badge = document.getElementById('web-search-badge');
  const label = document.getElementById('web-label');
  const dot   = badge.querySelector('.web-dot');

  if (webSearchEnabled === false) {
    badge.style.borderColor = 'var(--border)';
    badge.style.color       = 'var(--text3)';
    badge.style.background  = 'var(--bg3)';
    dot.style.background    = 'var(--text3)';
    dot.style.boxShadow     = 'none';
    label.textContent = '🗄️ Local only';
    sysMsg('— Local mode: Zotero only —');

  } else if (webSearchEnabled === true) {
    badge.style.borderColor = 'var(--accent)';
    badge.style.color       = 'var(--accent)';
    badge.style.background  = '#1f3a5f';
    dot.style.background    = 'var(--accent)';
    dot.style.boxShadow     = '0 0 8px var(--accent)';
    label.textContent = '🔀 Local + Web';
    sysMsg('— Hybrid mode: Zotero + Semantic Scholar —');

  } else {
    badge.style.borderColor = 'var(--accent2)';
    badge.style.color       = 'var(--accent2)';
    badge.style.background  = '#0d1f17';
    dot.style.background    = 'var(--accent2)';
    dot.style.boxShadow     = '0 0 8px var(--accent2)';
    label.textContent = '🌐 Web only';
    sysMsg('— Web mode: Semantic Scholar only (10 results) —');
  }
}

async function wizInstallModels() {
  if (wizSelectedModels.size === 0) {
    wizShowStep(1);
    return;
  }
  document.getElementById('wiz-btn-install').disabled = true;
  document.getElementById('wiz-model-selection').style.display = 'none';
  document.getElementById('wiz-pull-progress').style.display = 'block';

  const logEl = document.getElementById('wiz-pull-log');
  const models = Array.from(wizSelectedModels);

  for (const model of models) {
    logEl.textContent += `\n⬇️  Pulling ${model}…`;
    logEl.scrollTop = logEl.scrollHeight;
    try {
      const r = await fetch('/api/ollama/pull', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ model })
      });
      const d = await r.json();
      if (d.success) {
        logEl.textContent += `\n✅ ${model} installed.`;
      } else {
        logEl.textContent += `\n⚠️  Failed: ${d.error}`;
      }
    } catch(e) {
      logEl.textContent += `\n❌ Error: ${e.message}`;
    }
    logEl.scrollTop = logEl.scrollHeight;
  }

  document.getElementById('wiz-pull-bar').style.animation = 'none';
  document.getElementById('wiz-pull-bar').style.width = '100%';
  logEl.textContent += '\n\n✅ Done! You can continue.';
  logEl.scrollTop = logEl.scrollHeight;

  document.querySelector('#wiz-ollama-ok .wiz-actions').innerHTML =
    '<button class="wiz-btn-primary" onclick="wizShowHw()">Next →</button>';
  document.getElementById('wiz-pull-progress').appendChild(
    document.querySelector('#wiz-ollama-ok .wiz-actions')
  );
}

async function wizCheckOllama() {
  document.getElementById('wiz-ollama-checking').style.display = 'block';
  document.getElementById('wiz-ollama-ok').style.display = 'none';
  document.getElementById('wiz-ollama-missing').style.display = 'none';
  try {
    const r = await fetch('/api/ollama/check');
    const d = await r.json();
    document.getElementById('wiz-ollama-checking').style.display = 'none';
    if (d.ollama_installed && d.ollama_running) {
      document.getElementById('wiz-ollama-ok').style.display = 'block';
      const modelsEl = document.getElementById('wiz-models-status');
      if (d.models_missing.length === 0) {
        modelsEl.textContent = '✅ All models available: ' + d.models_available.slice(0,4).join(', ');
      } else {
        modelsEl.textContent = '⚠️ Missing models: ' + d.models_missing.join(', ') + '\n→ Relaunch via RefChat.bat to download them automatically.';
      }
    } else {
      document.getElementById('wiz-ollama-missing').style.display = 'block';
    }
  } catch(e) {
    document.getElementById('wiz-ollama-checking').style.display = 'none';
    document.getElementById('wiz-ollama-missing').style.display = 'block';
  }
}

function wizShowStep(n) {
  document.querySelectorAll('.wizard-step').forEach(s => s.classList.remove('active'));
  const el = document.getElementById('wiz-step-' + n) || document.getElementById('wiz-step-hw');
  if (el) el.classList.add('active');
  wizCurrentStep = n;
}

function wizShowHw() {
  document.querySelectorAll('.wizard-step').forEach(s => s.classList.remove('active'));
  document.getElementById('wiz-step-hw').classList.add('active');
  wizHwDetect();
}

async function wizHwDetect() {
  document.getElementById('wiz-hw-detect-status').style.display = 'block';
  document.getElementById('wiz-hw-summary').style.display = 'none';
  document.getElementById('wiz-hw-params').style.display = 'none';
  document.getElementById('wiz-hw-save-btn').style.display = 'none';
  try {
    const r = await fetch('/api/hardware/detect');
    const d = await r.json();
    // Summary card
    const sumEl = document.getElementById('wiz-hw-summary');
    sumEl.innerHTML =
      `🖥️ <strong>${d.cpu_count} CPU cores</strong>&nbsp;&nbsp;` +
      `💾 <strong>${d.ram_gb > 0 ? d.ram_gb + ' GB RAM' : 'RAM unknown'}</strong>&nbsp;&nbsp;` +
      (d.has_gpu ? `🎮 <strong>${d.gpu_name} — ${d.gpu_vram_gb} GB VRAM</strong>` : `⚠️ No GPU detected`);
    // Pre-fill inputs with saved values or recommendations
    document.getElementById('wiz-hw-thread').value = (d.saved_num_thread !== '') ? d.saved_num_thread : d.rec_threads;
    document.getElementById('wiz-hw-gpu').value    = (d.saved_num_gpu    !== '') ? d.saved_num_gpu    : d.rec_gpu;
    document.getElementById('wiz-hw-batch').value  = (d.saved_num_batch  !== '') ? d.saved_num_batch  : d.rec_batch;
    document.getElementById('wiz-hw-detect-status').style.display = 'none';
    sumEl.style.display = 'block';
    document.getElementById('wiz-hw-params').style.display = 'block';
    document.getElementById('wiz-hw-save-btn').style.display = 'inline-block';
  } catch(e) {
    document.getElementById('wiz-hw-detect-status').textContent =
      '⚠️ Detection failed. Enter values manually or skip.';
    document.getElementById('wiz-hw-params').style.display = 'block';
    document.getElementById('wiz-hw-save-btn').style.display = 'inline-block';
  }
}

async function wizHwSave() {
  const t = parseInt(document.getElementById('wiz-hw-thread').value);
  const g = parseInt(document.getElementById('wiz-hw-gpu').value);
  const b = parseInt(document.getElementById('wiz-hw-batch').value);
  if (!isNaN(t) && !isNaN(g) && !isNaN(b)) {
    await fetch('/api/config', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ num_thread: t, num_gpu: g, num_batch: b })
    });
    showToast('✅ Hardware config saved');
  }
  wizShowStep(1);
}

async function settingsDetectHw() {
  const infoEl = document.getElementById('s-hw-info');
  infoEl.style.display = 'block';
  infoEl.textContent = '⏳ Detecting…';
  try {
    const r = await fetch('/api/hardware/detect');
    const d = await r.json();
    infoEl.innerHTML =
      `🖥️ <strong>${d.cpu_count} cores</strong> · ` +
      `💾 <strong>${d.ram_gb > 0 ? d.ram_gb + ' GB' : '?'} RAM</strong> · ` +
      (d.has_gpu ? `🎮 <strong>${d.gpu_name} (${d.gpu_vram_gb} GB VRAM)</strong>` : `⚠️ No GPU`) +
      `<br>Recommended: <code>thread=${d.rec_threads}</code> · <code>gpu=${d.rec_gpu}</code> · <code>batch=${d.rec_batch}</code> · <code>num_ctx=${d.rec_num_ctx_local}</code>`;
    document.getElementById('s-num-thread').value    = d.rec_threads;
    document.getElementById('s-num-gpu').value       = d.rec_gpu;
    document.getElementById('s-num-batch').value     = d.rec_batch;
    document.getElementById('s-num-ctx-local').value = d.rec_num_ctx_local;
  } catch(e) {
    infoEl.textContent = '⚠️ Detection failed: ' + e.message;
  }
}

async function wizNext() {
  const zotero = document.getElementById('wiz-zotero').value.trim();
  if (!zotero) { showToast('⚠️ Please enter the Zotero folder', true); return; }
  wizShowStep(2);
}

async function wizSaveAndIndex(skipIndex = false) {
  const zotero = document.getElementById('wiz-zotero').value.trim();
  const apikey = document.getElementById('wiz-apikey').value.trim();
  await fetch('/api/config', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ zotero_path: zotero, mistral_api_key: apikey })
  });
  if (skipIndex) {
    document.getElementById('wizard-overlay').classList.add('hidden');
    return;
  }
  wizShowStep(3);
  const r = await fetch('/api/ingest/start', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ zotero_path: zotero })
  });
  const d = await r.json();
  if (!d.success) {
    document.getElementById('wiz-log').textContent = '❌ ' + d.error;
    return;
  }
  ingestPoll = setInterval(async () => {
    const s = await (await fetch('/api/ingest/status')).json();
    const logEl = document.getElementById('wiz-log');
    if (s.log.length) logEl.textContent = s.log.join('\n');
    logEl.scrollTop = logEl.scrollHeight;
    if (s.done) {
      clearInterval(ingestPoll); ingestPoll = null;
      document.getElementById('wiz-bar').style.animation = 'none';
      document.getElementById('wiz-bar').style.width = '100%';
      if (s.error) {
        logEl.textContent += '\n❌ Error: ' + s.error;
      } else {
        const st = await (await fetch('/api/status')).json();
        document.getElementById('wiz-final-count').textContent =
          `${st.nb_extraits || '?'} excerpts indexed in your database.`;
        wizShowStep(4);
      }
    }
  }, 1500);
}

function wizFinish() {
  document.getElementById('wizard-overlay').classList.add('hidden');
  refreshStatus();
}

// ══ INGESTION PANEL ═══════════════════════════════════════════════════════════

function openIngestPanel() {
  const panel = document.createElement('div');
  panel.id = 'ingest-chat-panel';
  panel.className = 'ingest-panel';
  panel.innerHTML = `
    <div class="ingest-header">
      <div class="ingest-spinner" id="ingest-spinner"></div>
      <span class="ingest-title">🔄 Indexing in progress…</span>
      <span id="ingest-status-badge" style="margin-left:auto;font-size:0.72rem;color:var(--text3);font-family:'DM Mono',monospace"></span>
    </div>
    
    <div style="background: rgba(88, 166, 255, 0.1); border: 1px solid var(--accent); border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; font-size: 0.78rem; color: var(--text); line-height: 1.4;">
      🐳 <strong>Important:</strong> Make sure <strong>Docker Desktop</strong> is open so GROBID can parse your PDFs correctly.
    </div>
    
    <div style="background: rgba(210, 153, 34, 0.1); border: 1px solid var(--accent3); border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; font-size: 0.78rem; color: var(--text); line-height: 1.4;">
      💡 <strong>To pause:</strong> Go to the terminal (black screen) and press <code>Ctrl + C</code>. Relaunch the app later and click Index again — it will resume where it left off.
    </div>

    <div class="ingest-log-box" id="ingest-chat-log">Starting indexing…</div>
    <div class="ingest-progress"><div class="ingest-progress-bar" id="ingest-chat-bar"></div></div>
    
    <div style="background: rgba(88, 166, 255, 0.1); border: 1px solid var(--accent); border-radius: 8px; padding: 12px 16px; margin-top: 12px;">
      <div style="color: var(--accent); font-size: 0.88rem; font-weight: 600; margin-bottom: 6px;">
        🛡️ Post-indexing tip: Run the Audit
      </div>
      <div style="color: var(--text); font-size: 0.8rem; line-height: 1.5;">
        Once indexing is done, we strongly recommend running <code>Audit_data_base.py</code> in your terminal.<br>
        It will check that the AI found all <strong>Abstracts</strong> and <strong>Metadata</strong> (Authors/Years). If not, you can fix them manually without them being overwritten later!
      </div>
    </div>
  `;
  
  const old = document.getElementById('ingest-chat-panel');
  if (old) old.remove();
  messagesEl.appendChild(panel);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  fetch('/api/config').then(r => r.json()).then(async cfg => {
    const zotero = cfg.zotero_path;
    if (!zotero) { showToast('⚠️ Please configure the Zotero folder in ⚙️ Settings first', true); panel.remove(); return; }

    const r = await fetch('/api/ingest/start', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ zotero_path: zotero })
    });
    const d = await r.json();
    if (!d.success) {
      document.getElementById('ingest-chat-log').textContent = '❌ ' + d.error;
      document.getElementById('ingest-spinner').style.display = 'none';
      return;
    }

    if (ingestPoll) clearInterval(ingestPoll);
    ingestPoll = setInterval(async () => {
      const s = await (await fetch('/api/ingest/status')).json();
      const logEl = document.getElementById('ingest-chat-log');
      if (!logEl) { clearInterval(ingestPoll); return; }

      if (s.log.length) {
        logEl.textContent = s.log.join('\n');
        logEl.scrollTop = logEl.scrollHeight;
      }

      const total = s.log.filter(l => l.includes('PDFs trouvés')).join('');
      const done  = s.log.filter(l => l.includes('✓') || l.includes('✅')).length;
      const pctEl = document.getElementById('ingest-status-badge');
      if (pctEl) pctEl.textContent = `${done} articles processed`;

      if (s.done) {
        clearInterval(ingestPoll); ingestPoll = null;
        const bar = document.getElementById('ingest-chat-bar');
        if (bar) { bar.style.animation='none'; bar.style.width='100%'; bar.style.background = s.error ? '#f85149' : 'var(--accent2)'; }
        const spinner = document.getElementById('ingest-spinner');
        if (spinner) spinner.style.display = 'none';

        const titleEl = panel.querySelector('.ingest-title');
        if (s.error) {
          if (titleEl) titleEl.textContent = '❌ Indexing error';
          if (logEl) logEl.innerHTML += `\n<span class="ingest-error-txt">❌ ${s.error}</span>`;
          showToast('❌ Indexing failed', true);
        } else {
          if (titleEl) titleEl.textContent = '✅ Indexing complete';
          showToast('✅ Library indexed!');
          await refreshStatus();
          await scanNouveauxArticles();  
        }
      }
    }, 1500);
  });
}

// ══ SETTINGS MODAL ════════════════════════════════════════════════════════════

const MODEL_META = {
  'mistral:7b-instruct-q4_0': { key: 'mistral-light', label: '⚡ Mistral 7B Q4 — GPU (recommended)', size: '~4 GB' },
  'mistral':                   { key: 'mistral',       label: '🖥️ Mistral 7B — CPU only',               size: '~4.1 GB' },
  'mixtral':                   { key: 'mixtral',       label: '🧠 Mixtral 8x7B — CPU only',              size: '~26 GB' },
};
const MODEL_ALWAYS = [
  { key: 'api', label: '☁️ Mistral API (cloud)' }
];
const ALL_KNOWN_MODELS = [
  { ollama: 'mistral:7b-instruct-q4_0', label: '⚡ Mistral 7B Q4 — GPU', size: '~4 GB', desc: 'Recommended · Fast on GPU · 6 GB VRAM min' },
  { ollama: 'mistral',                   label: '🖥️ Mistral 7B — CPU only',             size: '~4.1 GB', desc: 'No GPU · All machines compatible · 8 GB RAM' },
  { ollama: 'mixtral',                   label: '🧠 Mixtral 8x7B — CPU only',           size: '~26 GB', desc: 'Very powerful · Slow without GPU · 32 GB RAM min' },
];

async function refreshModelSelect() {
  const select = document.getElementById('model-select');
  try {
    const r = await fetch('/api/ollama/check');
    const d = await r.json();
    const available = d.models_available || [];

    select.innerHTML = '';

    MODEL_ALWAYS.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.key; opt.textContent = m.label;
      select.appendChild(opt);
    });

    let hasLocal = false;
    for (const [ollama_id, meta] of Object.entries(MODEL_META)) {
      const installed = available.some(a => a.startsWith(ollama_id.split(':')[0]));
      if (installed) {
        const opt = document.createElement('option');
        opt.value = meta.key; opt.textContent = meta.label;
        select.appendChild(opt);
        hasLocal = true;
      }
    }

    if (hasLocal) {
      const lightOpt = Array.from(select.options).find(o => o.value === 'mistral-light');
      if (lightOpt) select.value = 'mistral-light';
    }

    if (select.options.length === 0) {
      const opt = document.createElement('option');
      opt.value = ''; opt.textContent = 'No models installed';
      select.appendChild(opt);
    }
  } catch(e) {
    select.innerHTML = '<option value="api">☁️ Mistral API (cloud)</option>';
  }
}

async function settingsRefreshModels() {
  const container = document.getElementById('s-models-list');
  container.innerHTML = '<div style="color:var(--text3);font-size:0.8rem;font-family:DM Mono,monospace">Vérification en cours…</div>';
  try {
    const r = await fetch('/api/ollama/check');
    const d = await r.json();
    const available = d.models_available || [];

    container.innerHTML = '';
    for (const m of ALL_KNOWN_MODELS) {
      const installed = available.some(a => a.startsWith(m.ollama.split(':')[0]));
      const card = document.createElement('div');
      card.style.cssText = 'background:var(--bg3);border:1px solid ' + (installed ? 'var(--accent2)' : 'var(--border)') + ';border-radius:8px;padding:10px 14px;';
      card.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center">
          <div>
            <span style="font-weight:600;color:var(--text);font-size:0.88rem">${m.label}</span>
            <span style="margin-left:8px;font-size:0.75rem;color:${installed ? 'var(--accent2)' : 'var(--text3)'}">${installed ? '✅ Installed' : '❌ Not installed'}</span>
          </div>
          <span style="font-family:'DM Mono',monospace;font-size:0.73rem;color:var(--accent3)">${m.size}</span>
        </div>
        <div style="font-size:0.73rem;color:var(--text2);margin-top:3px">${m.desc}</div>
        ${!installed ? `<button onclick="settingsInstallModel('${m.ollama}', this)" style="margin-top:8px;background:var(--accent);color:#0d1117;border:none;padding:5px 12px;border-radius:5px;font-size:0.78rem;font-weight:600;cursor:pointer;font-family:Figtree,sans-serif">⬇️ Installer (${m.size})</button>` : ''}
      `;
      container.appendChild(card);
    }
    await refreshModelSelect();
  } catch(e) {
    container.innerHTML = '<div style="color:#f85149;font-size:0.8rem">Impossible de contacter Ollama</div>';
  }
}

async function settingsInstallModel(model, btn) {
  btn.disabled = true;
  btn.textContent = '⏳ Downloading…';
  const card = btn.closest('div[style*="border-radius:8px"]');
  try {
    const r = await fetch('/api/ollama/pull', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ model })
    });
    const d = await r.json();
    if (d.success) {
      showToast('✅ ' + model + ' installed!');
      await settingsRefreshModels();
    } else {
      btn.disabled = false;
      btn.textContent = '⬇️ Retry';
      showToast('❌ Error: ' + d.error, true);
    }
  } catch(e) {
    btn.disabled = false;
    btn.textContent = '⬇️ Retry';
    showToast('❌ ' + e.message, true);
  }
}

async function settingsOpen() {
  const r = await fetch('/api/config');
  const d = await r.json();
  document.getElementById('s-zotero').value    = d.zotero_path || '';
  document.getElementById('s-apikey').value     = d.mistral_api_key || '';
  document.getElementById('s-dbpath').value     = d.db_path || '';
  document.getElementById('s-ss-apikey').value  = d.semantic_scholar_api_key || '';
  document.getElementById('s-num-thread').value = d.num_thread !== undefined ? d.num_thread : '';
  document.getElementById('s-num-gpu').value    = d.num_gpu    !== undefined ? d.num_gpu    : '';
  document.getElementById('s-num-batch').value  = d.num_batch  !== undefined ? d.num_batch  : '';
  document.getElementById('s-num-ctx-local').value = d.ollama_num_ctx_local !== undefined ? d.ollama_num_ctx_local : '';
  document.getElementById('s-bm25-weight').value   = d.bm25_weight !== undefined ? d.bm25_weight : 0.3;
  document.getElementById('settings-modal').classList.add('open');
  settingsRefreshModels();
}

function settingsClose() {
  document.getElementById('settings-modal').classList.remove('open');
}

async function settingsSave() {
  const zotero   = document.getElementById('s-zotero').value.trim();
  const apikey   = document.getElementById('s-apikey').value.trim();
  const dbpath   = document.getElementById('s-dbpath').value.trim();
  const ssApiKey = document.getElementById('s-ss-apikey').value.trim();
  const numThread = document.getElementById('s-num-thread').value.trim();
  const numGpu    = document.getElementById('s-num-gpu').value.trim();
  const numBatch  = document.getElementById('s-num-batch').value.trim();
  const numCtxLocal = document.getElementById('s-num-ctx-local').value.trim();
  const bm25Weight  = document.getElementById('s-bm25-weight').value.trim();
  const hwPayload = {};
  if (numThread) hwPayload.num_thread = parseInt(numThread);
  if (numGpu !== '') hwPayload.num_gpu = parseInt(numGpu);
  if (numBatch) hwPayload.num_batch = parseInt(numBatch);
  if (numCtxLocal) hwPayload.ollama_num_ctx_local = parseInt(numCtxLocal);
  if (bm25Weight !== '') hwPayload.bm25_weight = parseFloat(bm25Weight);
  const r = await fetch('/api/config', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ zotero_path: zotero, mistral_api_key: apikey,
                           db_path: dbpath, semantic_scholar_api_key: ssApiKey,
                           ...hwPayload })
  });
  const d = await r.json();
  showToast(d.success ? '✅ Settings saved' : '❌ ' + d.error, !d.success);
}

async function settingsStartIngest() {
  const zotero = document.getElementById('s-zotero').value.trim();
  if (!zotero) { showToast('⚠️ Please enter the Zotero folder', true); return; }
  await settingsSave();
  settingsClose();
  openIngestPanel();
}

document.getElementById('settings-modal').addEventListener('click', e => {
  if (e.target === document.getElementById('settings-modal')) settingsClose();
});

// ══ CHAT ══════════════════════════════════════════════════════════════════════

function setStopMode(active) {
  if (active) { btnSend.textContent='■'; btnSend.classList.add('stop'); btnSend.disabled=false; }
  else        { btnSend.textContent='↑'; btnSend.classList.remove('stop'); btnSend.disabled=false; }
}

function stopQuery() {
  if (currentAbortController) { currentAbortController.abort(); currentAbortController=null; }
}

async function sendQuery() {
  if (!isReady || isLoading) return;
  const query = queryInput.value.trim();
  if (!query) return;
  queryInput.value=''; queryInput.style.height='auto';
  isLoading=true; setStopMode(true);

  appendMessage('user', query);
  const thinkingEl = appendThinking();

  const botDiv = document.createElement('div');
  botDiv.className='message bot';
  const avatar = document.createElement('div');
  avatar.className='avatar bot-av'; avatar.textContent='🤖';
  const bubble = document.createElement('div');
  bubble.className='bubble';
  botDiv.appendChild(avatar); botDiv.appendChild(bubble);

  let fullText='', bubbleReady=false;
  currentAbortController = new AbortController();

  try {
    const response = await fetch('/api/chat', {
      method:'POST', 
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
          query: query,
          web_search: webSearchEnabled
      }), 
      signal:currentAbortController.signal
    });
    
    const reader=response.body.getReader(), decoder=new TextDecoder();
    let buffer='';
    while (true) {
      const {done,value}=await reader.read(); if (done) break;
      buffer+=decoder.decode(value,{stream:true});
      const lines=buffer.split('\n'); buffer=lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let payload; try { payload=JSON.parse(line.slice(6)); } catch { continue; }
        if (payload.error) { thinkingEl.remove(); appendMessage('bot',null,null,payload.error); break; }
        if (payload.token) {
          if (!bubbleReady) { thinkingEl.remove(); messagesEl.appendChild(botDiv); bubbleReady=true; }
          fullText+=payload.token;
          // ── OPTIMISATION RAM : texte brut pendant le stream, Markdown rendu UNE SEULE FOIS à la fin ──
          bubble.textContent=fullText;
          messagesEl.scrollTop=messagesEl.scrollHeight;
        }
        if (payload.done) {
          const meta=payload;
          const modeMap={question:'💬 Question',resume:'📋 Summary',reference:'🔎 References',auteur:'👤 Author'};
          let html=`<div class="mode-badge mode-${meta.mode||'question'}">${modeMap[meta.mode]||'💬 Question'}</div>`+marked.parse(fullText);

          // ── Articles analysés ──
          if (meta.articles&&meta.articles.length) {
            html+=`<div class="articles-info"><div class="articles-info-title">📂 ${meta.articles.length} article(s) analyzed</div>`;
            for (const a of meta.articles) {
              const isWeb = (a.nb_chunks === 0 || (a.titre && a.titre.startsWith('🌐')));
              html+=`<div class="article-row" style="${isWeb ? 'border-left:2px solid var(--accent);padding-left:6px' : ''}">
                       <span>${isWeb ? '🌐 ' : ''}${escHtml(a.auteur)} (${escHtml(a.annee)}) — ${escHtml(a.titre)}</span>
                       <span style="color:var(--text3);font-family:'DM Mono',monospace;font-size:0.72rem">${isWeb ? 'web' : a.nb_chunks+' chunks'}</span>
                     </div>`;
            }
            html+=`</div>`;
          }

          // ── Bouton Copy — entre le texte et les sources ──
          // Le texte est stocké via un id unique sur la bulle, pas en data-text (évite les bugs de backticks)
          const bubbleId = 'bubble-' + Date.now();
          bubble.id = bubbleId;
          html+=`<div style="margin:10px 0 4px 0;">
                   <button class="btn-copy" onclick="copyBubbleById('${bubbleId}')">📋 Copy text</button>
                 </div>`;

          // ── Sources citées avec liens Zotero / DOI ──
          if (meta.sources&&meta.sources.length) {
            html+=`<div class="sources"><div class="sources-title">📎 Cited sources</div>`;
            for (const s of meta.sources) {
              if (s.section === 'Web Search') {
                // Source web → DOI ou Google Scholar
                const href = s.url && s.url.startsWith('http')
                  ? s.url
                  : 'https://scholar.google.com/scholar?q=' + encodeURIComponent((s.titre||'') + ' ' + s.auteur + ' ' + s.annee);
                const isDoi = s.url && s.url.includes('doi.org');
                html+=`<a href="${escHtml(href)}" target="_blank" rel="noopener"
                          title="${isDoi ? 'Open via DOI' : 'Search on Google Scholar'}"
                          class="source-tag" style="color:var(--accent);text-decoration:none;border-color:#58a6ff44;">
                          🌐 ${escHtml(s.auteur)}, ${escHtml(s.annee)}
                          <span style="font-size:0.65rem;opacity:0.7;margin-left:3px">${isDoi ? '🔗 DOI' : '🔍'}</span>
                        </a>`;
              } else if (s.zotero_link) {
                // Source locale → ouvre via /api/open-pdf côté serveur
                const doiLink = s.doi
                  ? ` <a href="https://doi.org/${escHtml(s.doi)}" target="_blank" rel="noopener"
                         title="Open DOI" style="color:var(--accent);font-size:0.65rem;margin-left:4px;text-decoration:none;">🔗</a>`
                  : '';
                // Encodage base64 du chemin pour éviter tout conflit de guillemets dans onclick
                const b64Path = btoa(unescape(encodeURIComponent(s.url || '')));
                html+=`<span class="source-tag zotero-link" title="Open PDF"
                          onclick="openLocalPdf(atob('${b64Path}'))" style="cursor:pointer;">
                          📄 ${escHtml(s.auteur)}, ${escHtml(s.annee)}
                          <span style="font-size:0.62rem;opacity:0.6;margin-left:3px">↗ PDF</span>
                        </span>${doiLink}`;
              } else {
                // Fallback : tag simple sans lien
                html+=`<span class="source-tag">📄 ${escHtml(s.auteur)}, ${escHtml(s.annee)}</span>`;
              }
            }
            html+=`</div>`;
          }

          // ── Barre inférieure : stats uniquement ──
          if (meta.elapsed!==undefined) {
            const webInfo = meta.nb_web_total > 0
              ? ` <span style="color:var(--accent2);font-size:0.65rem;margin-left:6px">🌐 ${meta.nb_web_total.toLocaleString()} results on Semantic Scholar</span>`
              : '';
            html+=`<div class="elapsed-time">
                     <div class="elapsed-right">
                       <span>⏱️ ${meta.elapsed}s — ${escHtml(meta.nom_llm||'')}${webInfo}</span>
                       <span class="tokens-badge">📥 ~${meta.tokens_in} tok | 📤 ~${meta.tokens_out} tok</span>
                     </div>
                   </div>`;
          }

          bubble.innerHTML=html;
          // Stocker le texte brut proprement sur l'élément DOM (pas en attribut HTML)
          bubble._rawText = fullText;
          if (memoryEnabled) updateMemoryBadge(meta.history_count||0);
          messagesEl.scrollTop=messagesEl.scrollHeight;
        }
      }
    }
  } catch(e) {
    if (bubbleReady) bubble.innerHTML=marked.parse(fullText)+`<div class="elapsed-time" style="color:#f85149">⏹ Stopped</div>`;
    // Rendu Markdown final même si stream interrompu
    else { thinkingEl.remove(); if (e.name!=='AbortError') appendMessage('bot',null,null,'Network error: '+e.message); else appendMessage('bot',null,null,'⏹ Generation stopped.'); }
  }
  isLoading=false; currentAbortController=null; setStopMode(false); queryInput.focus();
}

queryInput.addEventListener('input',()=>{ queryInput.style.height='auto'; queryInput.style.height=Math.min(queryInput.scrollHeight,160)+'px'; });
queryInput.addEventListener('keydown',e=>{ if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendQuery();} });
btnSend.addEventListener('click',()=>{ if(isLoading) stopQuery(); else sendQuery(); });
btnInit.addEventListener('click',initSystem);

// ── Ouvrir un PDF local via le serveur Flask (contourne restriction Firefox) ──
function openLocalPdf(path) {
  fetch('/api/open-pdf', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({path: path})
  }).catch(e => console.warn('open-pdf error:', e));
}
function copyBubbleById(bubbleId) {
  const bubble = document.getElementById(bubbleId);
  const text = bubble && bubble._rawText ? bubble._rawText : '';
  const btn = bubble && bubble.querySelector('.btn-copy');
  navigator.clipboard.writeText(text).then(() => {
    if (btn) { btn.textContent='✅ Copied!'; btn.classList.add('copied'); setTimeout(()=>{ btn.textContent='📋 Copy text'; btn.classList.remove('copied'); },2000); }
  }).catch(() => {
    const ta=document.createElement('textarea'); ta.value=text; ta.style.position='fixed'; ta.style.opacity='0';
    document.body.appendChild(ta); ta.select(); document.execCommand('copy'); document.body.removeChild(ta);
    if (btn) { btn.textContent='✅ Copied!'; btn.classList.add('copied'); setTimeout(()=>{ btn.textContent='📋 Copy text'; btn.classList.remove('copied'); },2000); }
  });
}

async function initSystem() {
  const modele=modelSelect.value;
  btnInit.disabled=true; setStatus('loading','Loading…');
  try {
    const r=await fetch('/api/init',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({modele})});
    const data=await r.json();
    if (data.success) {
      isReady=true; setStatus('ready',data.nom_llm+' ready');
      queryInput.disabled=false; btnSend.disabled=false; queryInput.focus();
      await refreshStatus();
      if (welcomeEl) welcomeEl.remove();
    } else { setStatus('error','Error: '+data.error); }
  } catch(e) { setStatus('error','Connection failed'); }
  btnInit.disabled=false;
}

async function refreshStatus() {
  try {
    const r=await fetch('/api/status'); const d=await r.json();
    statModel.textContent=d.modele||'—'; statDocs.textContent=d.nb_extraits??'—';
  } catch {}
}

function updateMemoryBadge(count) {
  statMem.textContent=`${count} / 3`;
  const label=document.getElementById('mem-label');
  if (memoryEnabled) { label.textContent=`Memory: ${count}/3`; memBadge.classList.add('active'); }
  else { label.textContent='Memory OFF'; memBadge.classList.remove('active'); }
}

async function toggleMemory() {
  memoryEnabled=!memoryEnabled;
  await fetch('/api/toggle_memory',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:memoryEnabled})});
  const btn=document.getElementById('btn-toggle-mem');
  if (memoryEnabled) { btn.textContent='🧠 Disable memory'; btn.style.borderColor='#bc8cff'; btn.style.color='#bc8cff'; }
  else { btn.textContent='🧠 Enable memory'; btn.style.borderColor=''; btn.style.color=''; }
  updateMemoryBadge(0);
  sysMsg(memoryEnabled ? '— Memory enabled (3 exchanges) —' : '— Memory disabled —');
}

function setStatus(state,text) { statusDot.className=state; statusText.textContent=text; }

function appendMessage(role,text,data,error) {
  const div=document.createElement('div'); div.className='message '+(role==='user'?'user':'bot');
  const avatar=document.createElement('div'); avatar.className='avatar '+(role==='user'?'user-av':'bot-av'); avatar.textContent=role==='user'?'👤':'🤖';
  const bubble=document.createElement('div'); bubble.className='bubble';
  if (error) bubble.innerHTML=`<div class="error-msg">⚠️ ${escHtml(error)}</div>`;
  else if (role==='user') bubble.textContent=text;
  else {
    const modeMap={question:['question','💬 Question'],resume:['resume','📋 Summary'],reference:['reference','🔎 References'],auteur:['auteur','👤 Author']};
    const [mc,ml]=modeMap[data.mode]||['question','💬 Question'];
    let html=`<div class="mode-badge mode-${mc}">${ml}</div>`+marked.parse(text);
    if (data.articles&&data.articles.length) {
      html+=`<div class="articles-info"><div class="articles-info-title">📂 ${data.articles.length} article(s) analyzed</div>`;
      for (const a of data.articles) html+=`<div class="article-row"><span>${escHtml(a.auteur)} (${escHtml(a.annee)}) — ${escHtml(a.titre)}</span><span style="color:var(--text3);font-family:'DM Mono',monospace;font-size:0.72rem">${a.nb_chunks} chunks</span></div>`;
      html+=`</div>`;
    }
    if (data.sources&&data.sources.length) {
      html+=`<div class="sources"><div class="sources-title">Cited sources</div>`;
      for (const s of data.sources) html+=`<span class="source-tag">${escHtml(s.auteur)}, ${escHtml(s.annee)}</span>`;
      html+=`</div>`;
    }
    bubble.innerHTML=html;
  }
  div.appendChild(avatar); div.appendChild(bubble);
  messagesEl.appendChild(div); messagesEl.scrollTop=messagesEl.scrollHeight;
}

function appendThinking() {
  const div=document.createElement('div'); div.className='thinking';
  div.innerHTML=`<div class="dots"><span></span><span></span><span></span></div> The model is thinking…`;
  messagesEl.appendChild(div); messagesEl.scrollTop=messagesEl.scrollHeight; return div;
}

function sysMsg(text) {
  const div=document.createElement('div');
  div.style.cssText='text-align:center;color:var(--text3);font-size:0.75rem;padding:8px;font-family:DM Mono,monospace;';
  div.textContent=text; messagesEl.appendChild(div); messagesEl.scrollTop=messagesEl.scrollHeight;
}

async function clearMemory() {
  await fetch('/api/clear_history',{method:'POST'}); updateMemoryBadge(0); sysMsg('— Memory cleared —');
}

function clearChat() {
  messagesEl.querySelectorAll('.message,.thinking,.ingest-panel,div[style*="text-align:center"]').forEach(m=>m.remove());
  clearMemory();
}

function setQuery(text) { queryInput.value=text; queryInput.focus(); }
function escHtml(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function showToast(msg, isErr=false) {
  const div=document.createElement('div');
  div.className='toast '+(isErr?'err':'ok'); div.textContent=msg;
  document.body.appendChild(div); setTimeout(()=>div.remove(), 4000);
}

async function quitApp() {
  if (!confirm('Close RefChat and stop the server?')) return;
  await fetch('/api/quit',{method:'POST'});
  document.body.innerHTML='<div style="display:flex;align-items:center;justify-content:center;height:100vh;background:#0d1117;color:#8b949e;font-family:Figtree,sans-serif;font-size:1.1rem;">RefChat arrêté. Vous pouvez fermer cet onglet.</div>';
}

// ══ INIT AU CHARGEMENT ════════════════════════════════════════════════════════

async function scanNouveauxArticles() {
  try {
    const r = await fetch('/api/ingest/scan');
    const d = await r.json();
    const badge    = document.getElementById('new-articles-badge');
    const label    = document.getElementById('new-articles-label');
    const preview  = document.getElementById('new-articles-preview');

    const statIndex = document.getElementById('stat-index');
    if (d.nouveaux && d.nouveaux > 0) {
      badge.style.display  = 'block';
      label.textContent    = `📥 ${d.nouveaux} new article${d.nouveaux > 1 ? 's' : ''} detected`;
      if (d.liste && d.liste.length) {
        const noms = d.liste.map(n => '• ' + n.replace(/\.[^.]+$/, '').substring(0, 40)).join('\n');
        preview.textContent = noms + (d.nouveaux > 5 ? `\n… +${d.nouveaux - 5} autres` : '');
      }
      if (statIndex) {
        statIndex.textContent = `📥 ${d.nouveaux} new`;
        statIndex.style.color = 'var(--accent3)';
        statIndex.title = `${d.nouveaux} new article${d.nouveaux > 1 ? 's' : ''} not yet indexed (${d.total} total in Zotero)`;
      }
    } else {
      badge.style.display = 'none';
      if (statIndex) {
        statIndex.textContent = d.erreur ? '⚠️ —' : '✅ Up to date';
        statIndex.style.color = d.erreur ? 'var(--text3)' : 'var(--accent2)';
        statIndex.title = d.erreur ? d.erreur : `All ${d.total} article${d.total !== 1 ? 's' : ''} indexed`;
      }
    }
  } catch {
  }
}

// ══ CHECK ARTICLES MODAL ══════════════════════════════════════════════════════

let checkArticlesData = [];

async function openCheckArticlesModal() {
  document.getElementById('check-articles-modal').style.display = 'flex';
  document.getElementById('check-articles-search').value = '';
  const listEl = document.getElementById('check-articles-list');
  listEl.innerHTML = '<div style="padding:20px;text-align:center;color:var(--text3);font-size:0.85rem">Loading…</div>';

  try {
    const r = await fetch('/api/ingest/scan/full');
    const d = await r.json();
    checkArticlesData = d.nouveaux || [];
    renderCheckArticles(checkArticlesData);
  } catch(e) {
    listEl.innerHTML = '<div style="padding:20px;text-align:center;color:#f85149;font-size:0.85rem">Error: ' + e.message + '</div>';
  }
}

function closeCheckArticlesModal() {
  document.getElementById('check-articles-modal').style.display = 'none';
}

function renderCheckArticles(items) {
  const listEl = document.getElementById('check-articles-list');
  const countEl = document.getElementById('check-articles-count');

  if (!items.length) {
    listEl.innerHTML = '<div style="padding:20px;text-align:center;color:var(--accent2);font-size:0.85rem">✅ No new articles detected.</div>';
    countEl.textContent = '';
    return;
  }

  countEl.textContent = items.length + ' article' + (items.length > 1 ? 's' : '') + ' to index';

  listEl.innerHTML = items.map((item, i) => {
    const name = item.filename.replace(/\.pdf$/i, '');
    const short = name.length > 70 ? name.substring(0, 70) + '…' : name;
    return `<label style="display:flex;align-items:flex-start;gap:10px;padding:9px 14px;border-bottom:1px solid var(--border);cursor:pointer;transition:background 0.15s;" 
                   onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                   onmouseout="this.style.background='transparent'"
                   data-index="${i}" data-filename="${escHtml(item.filename)}">
      <input type="checkbox" data-index="${i}" style="margin-top:3px;accent-color:#f85149;flex-shrink:0;" onchange="updateCheckCount()">
      <span style="font-size:0.78rem;color:var(--text2);font-family:'DM Mono',monospace;line-height:1.5">${escHtml(short)}</span>
    </label>`;
  }).join('');
  updateCheckCount();
}

function filterCheckArticles() {
  const q = document.getElementById('check-articles-search').value.toLowerCase();
  const filtered = checkArticlesData.filter(item => item.filename.toLowerCase().includes(q));
  renderCheckArticles(filtered);
}

function checkArticlesSelectAll(checked) {
  document.querySelectorAll('#check-articles-list input[type=checkbox]').forEach(cb => cb.checked = checked);
  updateCheckCount();
}

function updateCheckCount() {
  const total = document.querySelectorAll('#check-articles-list input[type=checkbox]').length;
  const selected = document.querySelectorAll('#check-articles-list input[type=checkbox]:checked').length;
  const countEl = document.getElementById('check-articles-count');
  const applyBtn = document.getElementById('check-articles-apply');
  countEl.textContent = total + ' article' + (total > 1 ? 's' : '') + ' to index' + (selected > 0 ? ` — ${selected} selected to blacklist` : '');
  if (selected > 0) {
    applyBtn.textContent = `🚫 Blacklist ${selected} & Index rest`;
  } else {
    applyBtn.textContent = '▶ Index all';
    applyBtn.style.background = '#0d1f17';
    applyBtn.style.borderColor = 'var(--accent2)66';
    applyBtn.style.color = 'var(--accent2)';
    applyBtn.onmouseover = () => applyBtn.style.background = '#1a3a2a';
    applyBtn.onmouseout  = () => applyBtn.style.background = '#0d1f17';
  }
  if (selected > 0) {
    applyBtn.style.background = '#2d1515';
    applyBtn.style.borderColor = '#f8514966';
    applyBtn.style.color = '#f85149';
    applyBtn.onmouseover = () => applyBtn.style.background = '#3d1515';
    applyBtn.onmouseout  = () => applyBtn.style.background = '#2d1515';
  }
}

async function applyBlacklistAndIndex() {
  const checked = Array.from(document.querySelectorAll('#check-articles-list input[type=checkbox]:checked'));
  const toBlacklist = checked.map(cb => cb.closest('label').dataset.filename);

  if (toBlacklist.length > 0) {
    const r = await fetch('/api/blacklist/add', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ filenames: toBlacklist })
    });
    const d = await r.json();
    if (!d.success) { showToast('❌ Blacklist error: ' + d.error, true); return; }
    showToast(`🚫 ${d.added} article${d.added > 1 ? 's' : ''} added to blacklist`);
  }

  closeCheckArticlesModal();
  await scanNouveauxArticles();
  openIngestPanel();
}

document.getElementById('check-articles-modal').addEventListener('click', e => {
  if (e.target === document.getElementById('check-articles-modal')) closeCheckArticlesModal();
});


function openThemePanel() {
  const old = document.getElementById('theme-chat-panel');
  if (old) old.remove();
  const panel = document.createElement('div');
  panel.id = 'theme-chat-panel';
  panel.className = 'ingest-panel';
  panel.innerHTML = `
    <div class="ingest-header">
      <div class="ingest-spinner" id="theme-spinner"></div>
      <span class="ingest-title">🏷️ Thématisation en cours…</span>
    </div>
    <div style="background:rgba(63,185,80,0.08);border:1px solid var(--accent2);border-radius:8px;padding:10px 14px;margin-bottom:10px;font-size:0.78rem;color:var(--text);line-height:1.5">
      💡 <strong>What does this do?</strong><br>
      Analyses existing embeddings to automatically group your articles into themes.
      Duration: 2–5 min. No PDF re-reading. No Docker needed.<br><br>
      ⚠️ <strong>Recommended workflow:</strong><br>
      1. Run a <strong>dry-run first</strong> from the command line to check results:<br>
      <code style="background:var(--bg1);padding:2px 6px;border-radius:4px;font-size:0.73rem">python refchat_theme.py --dry-run --topics 60 --show</code><br>
      2. Edit <code style="background:var(--bg1);padding:2px 6px;border-radius:4px;font-size:0.73rem">refchat_stopwords.txt</code> to remove parasitic label words if needed.<br>
      3. Only then launch here to write themes to the database.
    </div>
    <div style="display:flex;gap:8px;margin-bottom:10px;align-items:center">
      <label style="font-size:0.75rem;color:var(--text2);font-family:'DM Mono',monospace;white-space:nowrap">Nb thèmes :</label>
      <input type="number" id="theme-n-topics" placeholder="auto" min="2" max="50"
        style="width:70px;background:var(--bg3);border:1px solid var(--border);color:var(--text);padding:5px 8px;border-radius:6px;font-family:'DM Mono',monospace;font-size:0.8rem">
      <span style="font-size:0.7rem;color:var(--text3)">Laisse vide = détection automatique</span>
    </div>
    <div class="ingest-log-box" id="theme-chat-log">Démarrage…</div>
    <div class="ingest-progress"><div class="ingest-progress-bar" id="theme-chat-bar"></div></div>
    <div id="theme-done-msg" style="display:none;margin-top:10px;font-size:0.82rem;color:var(--accent2)"></div>
  `;
  messagesEl.appendChild(panel);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  const nTopicsInput = document.getElementById('theme-n-topics');
  const nTopics = nTopicsInput && nTopicsInput.value ? parseInt(nTopicsInput.value) : null;

  fetch('/api/theme/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ n_topics: nTopics, min_docs: 2 })
  }).then(r => r.json()).then(d => {
    if (!d.success) {
      const logEl = document.getElementById('theme-chat-log');
      if (logEl) logEl.textContent = 'Erreur : ' + d.error;
      const sp = document.getElementById('theme-spinner');
      if (sp) sp.style.display = 'none';
      return;
    }
    const poll = setInterval(async () => {
      const s = await (await fetch('/api/theme/status')).json();
      const logEl  = document.getElementById('theme-chat-log');
      const barEl  = document.getElementById('theme-chat-bar');
      const doneEl = document.getElementById('theme-done-msg');
      const title  = panel.querySelector('.ingest-title');
      const sp     = document.getElementById('theme-spinner');
      if (!logEl) { clearInterval(poll); return; }
      if (s.log && s.log.length) { logEl.textContent = s.log.join('\n'); logEl.scrollTop = logEl.scrollHeight; }
      if (s.done) {
        clearInterval(poll);
        if (sp) sp.style.display = 'none';
        if (barEl) { barEl.style.animation = 'none'; barEl.style.width = '100%'; }
        if (s.error) {
          if (title) title.textContent = '❌ Erreur thématisation';
          if (barEl) barEl.style.background = '#f85149';
          showToast('❌ ' + s.error, true);
        } else {
          if (title) title.textContent = '✅ Thématisation terminée !';
          if (barEl) barEl.style.background = 'var(--accent2)';
          if (doneEl) { doneEl.style.display = 'block'; doneEl.innerHTML = '✅ Themes saved to database — sidebar updated.<br><span style="font-size:0.75rem;color:var(--text3)">💡 If results look off, edit <code>refchat_stopwords.txt</code> and re-run the dry-run before launching again.</span>'; }
          showToast('✅ Thématisation terminée !');
          await loadThemes();
        }
      }
    }, 1500);
  }).catch(e => {
    const logEl = document.getElementById('theme-chat-log');
    if (logEl) logEl.textContent = 'Erreur réseau : ' + e.message;
  });
}

async function loadThemes() {
  const listEl = document.getElementById('themes-list');
  const countEl = document.getElementById('themes-count');
  if (!listEl) return;
  try {
    const r = await fetch('/api/themes');
    const d = await r.json();
    if (!d.themes || d.themes.length === 0) {
      listEl.innerHTML = '<div style="color:var(--text3);font-size:0.75rem;font-style:italic">Aucun thème — clique sur 🏷️ Thématisation</div>';
      return;
    }
    countEl.textContent = d.themes.length + ' theme(s)';
    listEl.innerHTML = d.themes.map(t => {
      const short = t.length > 30 ? t.substring(0, 28) + '…' : t;
      return '<button class="example-btn" style="font-size:0.72rem;padding:5px 8px;margin-bottom:3px;" onclick="setQueryTheme(' + JSON.stringify(t) + ')" title="' + t + '">🏷 ' + short + '</button>';
    }).join('');
  } catch(e) {
    listEl.innerHTML = '<div style="color:var(--text3);font-size:0.72rem">Error loading themes</div>';
  }
}

function setQueryTheme(theme) {
  queryInput.value = 'Resume les articles du theme ' + theme;
  queryInput.focus();
}

// ═══ MANAGE DB ══════════════════════════════════════════════════════════════
let _mdbArticles = [];

async function openManageDB() {
  closeSettings();
  const modal = document.getElementById('manage-db-modal');
  modal.style.display = 'flex';
  await loadManageDBArticles();
}
function closeManageDB() {
  document.getElementById('manage-db-modal').style.display = 'none';
}
async function loadManageDBArticles() {
  const listEl  = document.getElementById('mdb-list');
  const countEl = document.getElementById('mdb-count');
  listEl.innerHTML = '<div style="color:var(--text3);font-size:0.85rem;padding:20px;text-align:center">Loading…</div>';
  try {
    const d = await (await fetch('/api/db/articles')).json();
    _mdbArticles = d.articles || [];
    countEl.textContent = `${_mdbArticles.length} articles`;
    renderManageDBList(_mdbArticles);
  } catch(e) {
    listEl.innerHTML = `<div style="color:#f85149;padding:12px">Error: ${e.message}</div>`;
  }
}
function renderManageDBList(articles) {
  const listEl = document.getElementById('mdb-list');
  if (!articles.length) {
    listEl.innerHTML = '<div style="color:var(--text3);font-size:0.85rem;padding:20px;text-align:center">No articles found</div>';
    mdbUpdateCount(); return;
  }
  listEl.innerHTML = articles.map(a => {
    const badge   = a.doc_type === 'thesis'
      ? '<span style="background:#2a1a3a;color:#bc8cff;font-size:0.62rem;padding:1px 6px;border-radius:4px;margin-left:4px;font-family:\'DM Mono\',monospace">THESIS</span>' : '';
    const label   = a.auteur ? `${a.auteur}${a.annee ? ' (' + a.annee + ')' : ''}` : a.filename;
    const subline = a.auteur ? a.filename : (a.date || '');
    return `<label style="display:flex;align-items:center;gap:10px;padding:7px 10px;border-radius:6px;cursor:pointer;border:1px solid transparent;transition:background 0.12s" onmouseover="this.style.background='var(--bg3)'" onmouseout="this.style.background=''">
      <input type="checkbox" data-filename="${a.filename}" onchange="mdbUpdateCount()" style="width:15px;height:15px;accent-color:var(--accent);flex-shrink:0">
      <div style="flex:1;min-width:0">
        <div style="font-size:0.82rem;color:var(--text);font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${label}${badge}</div>
        <div style="font-size:0.7rem;color:var(--text3);font-family:'DM Mono',monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${subline}</div>
      </div>
      <span style="font-size:0.68rem;color:var(--text3);white-space:nowrap;font-family:'DM Mono',monospace;flex-shrink:0">${a.nb_chunks} chunks</span>
    </label>`;
  }).join('');
  mdbUpdateCount();
}
function filterManageDB() {
  const q = document.getElementById('mdb-search').value.toLowerCase().trim();
  const filtered = q ? _mdbArticles.filter(a =>
    a.filename.toLowerCase().includes(q) ||
    (a.auteur||'').toLowerCase().includes(q) ||
    (a.annee||'').includes(q) ||
    (a.titre||'').toLowerCase().includes(q)
  ) : _mdbArticles;
  renderManageDBList(filtered);
  document.getElementById('mdb-count').textContent = q
    ? `${filtered.length} / ${_mdbArticles.length} articles`
    : `${_mdbArticles.length} articles`;
}
function mdbUpdateCount() {
  const checked = document.querySelectorAll('#mdb-list input[type=checkbox]:checked');
  document.getElementById('mdb-selected-count').textContent = `${checked.length} selected`;
  document.getElementById('mdb-btn-delete').disabled = checked.length === 0;
}
function mdbSelectAll()  { document.querySelectorAll('#mdb-list input[type=checkbox]').forEach(cb => cb.checked = true);  mdbUpdateCount(); }
function mdbSelectNone() { document.querySelectorAll('#mdb-list input[type=checkbox]').forEach(cb => cb.checked = false); mdbUpdateCount(); }

async function mdbDeleteSelected() {
  const checked   = document.querySelectorAll('#mdb-list input[type=checkbox]:checked');
  const filenames = Array.from(checked).map(cb => cb.dataset.filename);
  if (!filenames.length) return;
  if (!confirm(`Delete ${filenames.length} article(s) from the database and blacklist them?\nThis cannot be undone without re-indexing.`)) return;

  const btn = document.getElementById('mdb-btn-delete');
  btn.disabled = true; btn.textContent = '⏳ Deleting…';
  try {
    const d = await (await fetch('/api/db/delete', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({filenames})
    })).json();
    if (d.success || d.deleted > 0) {
      showToast(`🗑️ ${d.deleted} article(s) deleted and blacklisted`);
      await loadManageDBArticles();
    } else {
      showToast('❌ ' + (d.errors||[]).join(', '), true);
    }
  } catch(e) {
    showToast('❌ Network error: ' + e.message, true);
  } finally {
    btn.disabled = false; btn.textContent = '🗑️ Delete from DB & blacklist';
    mdbUpdateCount();
  }
}
document.getElementById('manage-db-modal').addEventListener('click', e => {
  if (e.target === document.getElementById('manage-db-modal')) closeManageDB();
});

window.addEventListener('load', async () => {
  try {
    const r=await fetch('/api/setup/check'); const d=await r.json();
    if (d.first_launch) {
      const overlay=document.getElementById('wizard-overlay');
      overlay.classList.remove('hidden');
      wizCheckOllama(); 
      if (d.zotero_path) {
        document.getElementById('wiz-zotero').value=d.zotero_path;
        if (d.auto_detected) {
          const det=document.getElementById('wiz-auto-detect');
          det.style.display='block';
          det.textContent='✅ Zotero auto-detected: '+d.zotero_path;
        }
      }
    }
  } catch {}

  refreshModelSelect();
  scanNouveauxArticles();
  // Re-scan every 60s to detect new Zotero articles without needing a page reload
  setInterval(scanNouveauxArticles, 60000);

  try {
    const r=await fetch('/api/status'); const d=await r.json();
    if (d.ready) {
      isReady=true; setStatus('ready',(d.modele||'LLM')+' ready');
      queryInput.disabled=false; btnSend.disabled=false;
      statModel.textContent=d.modele||'—'; statDocs.textContent=d.nb_extraits??'—';
      if (welcomeEl) welcomeEl.remove();
    } else { setStatus('loading','Waiting for connection'); }
  } catch { setStatus('loading','Waiting for connection'); }
});
</script>
</body>
</html>
"""

# ── Lancement ─────────────────────────────────────────────────────────────────
def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://localhost:5001")

if __name__ == "__main__":
    print("🌐 RefChat — démarrage sur http://localhost:5001")
    print("   Ouverture automatique du navigateur…")
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)