# ============================================================
# CENTRAL CONFIG — RefChat
# No hardcoded paths — everything is saved in refchat_config.json
# ============================================================
import os, json, pathlib

_DIR      = pathlib.Path(__file__).parent.resolve()
_CFG_FILE = _DIR / "refchat_config.json"

_DEFAULTS = {
    "zotero_path":              "",
    "db_path":                  str(_DIR / "chroma_db"),
    "llm_model":                "mistral-large-latest",
    "mistral_api_key":          "",
    "semantic_scholar_api_key": "",
}

def _load():
    if _CFG_FILE.exists():
        try:
            with open(_CFG_FILE, encoding="utf-8") as f:
                return {**_DEFAULTS, **json.load(f)}
        except Exception:
            pass
    return dict(_DEFAULTS)

def save(updates: dict):
    current = _load()
    current.update(updates)
    with open(_CFG_FILE, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2, ensure_ascii=False)
    return current

def get(key, default=None):
    return _load().get(key, default)

# Alias
sauvegarder_config = save

# ── Exposed values (backward-compatibility imports) ───────────────────────────
_cfg = _load()

EMBEDDING_MODEL   = "intfloat/multilingual-e5-large"
E5_QUERY_PREFIX   = "query: "
E5_PASSAGE_PREFIX = "passage: "

DB_PATH   = _cfg.get("db_path",   str(_DIR / "chroma_db"))
LLM_MODEL = _cfg.get("llm_model", "mistral-light")

OLLAMA_TEMPERATURE = 0.1   # valeur par défaut (rétrocompatibilité)
OLLAMA_NUM_CTX     = 8192  # valeur par défaut (rétrocompatibilité)

def get_ollama_temperature() -> float:
    return float(_load().get("ollama_temperature", 0.1))

def get_ollama_num_ctx_local() -> int:
    """Contexte pour modèles locaux Ollama — limité par la VRAM."""
    return int(_load().get("ollama_num_ctx_local", 4096))

def get_ollama_num_ctx_api() -> int:
    """Contexte pour l'API cloud Mistral — pas de contrainte hardware."""
    return int(_load().get("ollama_num_ctx_api", 32768))

# Alias rétrocompatibilité
def get_ollama_num_ctx() -> int:
    return get_ollama_num_ctx_local()

MISTRAL_API_KEY = _cfg.get("mistral_api_key", "") or os.environ.get("MISTRAL_API_KEY", "")
