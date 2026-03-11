## What is RefChat?

RefChat is a self-hosted, web-based **Retrieval-Augmented Generation (RAG)** application that lets you query your personal Zotero PDF library using natural language. It intelligently indexes your scientific articles with GROBID (section-aware parsing) and answers your questions by citing only the sources in your database.

**Key features:**
- 🧠 **Smart indexing** — uses [GROBID](https://github.com/kermitt2/grobid) (via Docker) to split PDFs by section (Abstract, Introduction, Methods, Results, etc.)
- 🔍 **Three-stage retrieval** — E5 dense embeddings + BM25 keyword search (Reciprocal Rank Fusion) + cross-encoder reranking for maximum precision
- 📖 **Thesis detection** — automatically classifies long documents as theses/dissertations and limits their context footprint to prevent monopolisation
- 💬 **Multi-mode chat** — question answering, thematic synthesis, reference lookup, author search
- 🌐 **Web search integration** — optional [Semantic Scholar](https://api.semanticscholar.org/) API
- 🤖 **Model flexibility** — Mistral API (cloud) or local models via Ollama (Mistral 7B Q4, Mixtral 8x7B)
- 🗃️ **Persistent memory** — optional conversation history across turns
- 🛠️ **Audit tool** — CLI tool to fix missing abstracts or corrupted metadata
- 🔁 **Incremental indexing** — re-run safely; already-processed files are skipped

---

## What's new in v1.2

This version focuses on **retrieval quality** — finding the right articles before asking the LLM to answer.

### Retrieval pipeline overhaul

**v1.0** used a single semantic search over `k=8` abstract chunks, stopping at the first 8 unique articles found — meaning article selection depended heavily on the order of results, not their actual relevance.

**v1.2** introduces a three-stage pipeline:

| Stage | What it does |
|---|---|
| **E5 dense (k=150)** | Searches Abstract *and* Full text chunks with a large candidate pool |
| **BM25 (k=32)** | Exact keyword matching — critical for proper nouns, geographic terms, rare species names |
| **Reciprocal Rank Fusion** | Merges both ranked lists into a single top-20 candidate set |
| **Cross-encoder reranking** | Re-reads each (query, abstract) pair in full to score true relevance — then selects the final top N |

The key difference: every article in your library is a candidate, globally ranked before any selection. A highly relevant article is no longer skipped because it happened to appear at position 9 in a list capped at 8.

### Why a cross-encoder?

Dense embeddings (E5) compress text into a vector — a 1024-number summary. Two texts can have similar vectors without the query truly being answered by the document. A cross-encoder reads the full query and abstract *together* and scores their actual relevance, not just their vector proximity. It is slower (~300 ms for 20 candidates) but dramatically more precise.

The model used is `nreimers/mmarco-mMiniLMv2-L12-H384-v1`, trained on MS MARCO in 13 languages including French and English. It loads once at startup (~120 MB) and runs entirely locally — no API, no internet after the first download.

### Indexing quality improvements

- **Document type detection** — each PDF is classified as `article` or `thesis` via three signals: filename keywords, page count (>80 pages), and in-text markers (`jury`, `acknowledgements`, `école doctorale`…). Theses are capped at 5 chunks and 1 slot in results.
- **Sentence-aware chunking** — separators now respect sentence boundaries before line breaks.
- **Author/year prefix** — each chunk begins with `[Author Year]` so the embedding captures article identity alongside content.
- **Parasite chunk filtering** — short chunks (<150 chars) dominated by DOIs or years are discarded.
- **Scientific symbol handling** — Greek letters, isotope notation (δ¹³C, ‰, α, β…) are excluded from the word-ratio quality check so geochemistry-rich papers are not incorrectly rejected.

### Configurable BM25 weight

The BM25/dense ratio is now exposed in ⚙️ Settings (`bm25_weight`, default 0.3). Changes take effect on the next query — no restart needed.

---

## Architecture

```
refchat_main.py       ← Launcher: sets up venv, Ollama, models, then starts Flask
refchat_web.py        ← Flask web app + complete UI (single-file HTML/JS)
refchat_llm.py        ← RAG logic: retrieval, BM25, cross-encoder, prompts, Semantic Scholar
refchat_ingest.py     ← PDF ingestion pipeline (GROBID + fallback + ChromaDB)
refchat_config.py     ← Centralised configuration (reads/writes refchat_config.json)
Audit_database.py     ← CLI audit tool: fix metadata & inject missing abstracts
RefChat.bat           ← Windows one-click launcher (auto-installs everything)
```

---

## Requirements

| Requirement | Notes |
|---|---|
| **Docker Desktop** | **Crucial:** Required for GROBID (smart PDF parsing by section). Install from [docker.com](https://www.docker.com/products/docker-desktop/). |
| **Ollama** | Required to run local models offline. Download from [ollama.com](https://ollama.com/download). *(The Windows `.bat` launcher can auto-install it; manual install required on Linux/Mac).* |
| **Python** 3.9 – 3.12 | Tested on 3.11 |
| **RAM** | 8 GB minimum (16 GB+ recommended for local models) |
| **Disk** | ~6 GB for models + ChromaDB + cross-encoder cache |
| **Internet** | Required on first run to download models, embeddings, and the cross-encoder |

---

## Quick Start (Windows)

1. **Clone or download** this repository
2. **Double-click `RefChat.bat`** — it will:
   - Find Python on your system
   - Create an isolated `.venv`
   - Install all Python dependencies
   - Install Ollama (via winget) if not present
   - Download the Mistral 7B Q4 model (~4 GB, one time)
   - Open the app at `http://localhost:5001`
3. On first launch, a **setup wizard** guides you through:
   - Pointing to your `Zotero/storage` folder
   - Optionally adding a Mistral API key
   - Indexing your library

> **Tip:** Open Docker Desktop before indexing to enable GROBID's intelligent section parsing.

---

## Quick Start (Linux / macOS)

```bash
# 1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install flask langchain langchain-community langchain-chroma \
    langchain-huggingface langchain-ollama langchain-mistralai \
    langchain-text-splitters chromadb pymupdf sentence-transformers \
    tqdm requests beautifulsoup4 lxml rank-bm25

pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install Ollama → https://ollama.com/download
ollama pull mistral:7b-instruct-q4_0

# 4. (Optional) Start GROBID via Docker
docker run -d --rm -p 8070:8070 lfoppiano/grobid:0.8.1

# 5. Launch RefChat
python refchat_web.py
# → Open http://localhost:5001
```

---

## Configuration

Settings are stored in `refchat_config.json` and editable via the **⚙️ Settings** panel in the UI:

| Key | Default | Description |
|---|---|---|
| `zotero_path` | — | Path to your `Zotero/storage` folder |
| `db_path` | `./chroma_db` | Path where the ChromaDB vector database is stored |
| `llm_model` | `mistral-large-latest` | Active LLM: `mistral-large-latest` (API), `mistral-light`, `mistral`, or `mixtral` |
| `mistral_api_key` | — | Your Mistral API key ([console.mistral.ai](https://console.mistral.ai)) |
| `semantic_scholar_api_key` | — | Optional free key for higher Semantic Scholar rate limits |
| `bm25_weight` | `0.3` | BM25 weight in hybrid search (0 = dense only · 1 = BM25 only). Changes take effect on the next query — no restart needed. |

---

## Models

| Model | Key | Mode | Size | Notes |
|---|---|---|---|---|
| Mistral Large (cloud) | `api` | Cloud | — | Requires API key, largest context (~50k tokens) |
| Mistral 7B Q4 | `mistral-light` | Local GPU | ~4 GB | **Recommended** for GPU users |
| Mistral 7B | `mistral` | Local CPU | ~4.1 GB | Works on any machine, slower |
| Mixtral 8x7B | `mixtral` | Local CPU | ~26 GB | Powerful, requires 32 GB RAM |

---

## Chat Modes

RefChat automatically detects the intent of your query:

| Mode | Trigger keywords | Description |
|---|---|---|
| **Question** | (default) | Direct question answered with citations |
| **Synthesis** | summary, synthesize, overview… | Thematic synthesis across multiple articles |
| **References** | references, cite, which articles… | Lists relevant articles for a topic |
| **Author** | articles by, work by… | Retrieves all work by a given author |

---

## Retrieval Pipeline

RefChat uses a three-stage pipeline to identify the most relevant articles before generating an answer.

### Stage 1 — Hybrid search (E5 + BM25)

Two complementary strategies are run in parallel and merged via **Reciprocal Rank Fusion**:

- **Dense (E5)** — `intfloat/multilingual-e5-large` encodes queries and chunks into 1024-dimensional vectors. Strong for conceptual queries, synonyms, and cross-language understanding. Searches both `Abstract` and `Full text` sections with `k=150` to ensure global coverage of your library — including documents indexed without GROBID.
- **BM25** — exact keyword matching weighted by inverse document frequency. Critical for proper nouns, geographic names, and rare scientific terms that get diluted in dense space (e.g. "Greenland" vs "Deccan" in an archean geology query).

The BM25 weight (default 0.3 = 70% dense / 30% BM25) is configurable in ⚙️ Settings. The BM25 index is built once from your ChromaDB library and cached to `bm25_index.pkl` — subsequent startups load it instantly (<1s). The cache is automatically regenerated after re-indexation.

> **Requires:** `pip install rank-bm25`. If not installed, RefChat silently falls back to dense-only search.

### Stage 2 — Cross-encoder reranking

After RRF produces ~20 candidates, a **cross-encoder** reads each `(query, abstract)` pair together — not as separate vectors — and assigns a fine-grained relevance score. This step catches false positives that share vocabulary with the query but do not actually answer it.

Model: `nreimers/mmarco-mMiniLMv2-L12-H384-v1` (~120 MB, 13 languages including French and English). Loaded once at startup, runs entirely locally. If the model is unavailable, RefChat falls back silently to the RRF ranking.

### Stage 3 — Thesis slot limiting

Theses and dissertations are capped at **5 chunks** in context and **1 slot** among the final N articles, regardless of their retrieval score. This prevents a single long document from monopolising the LLM context window.

---

## Web Search (Semantic Scholar)

Click the **🗄️ Local only** button in the sidebar to cycle between three modes:

- **Local only** — searches your Zotero database exclusively
- **🔀 Local + Web** — combines your database with Semantic Scholar results
- **🌐 Web only** — searches Semantic Scholar exclusively (top 10 results)

Web results include a link to the DOI or a Google Scholar search. A free Semantic Scholar API key can be added in ⚙️ Settings to increase rate limits.

---

## Indexing

### First run
The setup wizard handles everything. It will:
1. Detect your Zotero folder automatically (or let you browse to it)
2. Optionally start GROBID via Docker
3. Index all PDFs and show a live log

### Subsequent runs
Click **🔄 Index library** in the sidebar. Already-processed files are skipped automatically.

### What the indexer does
For each PDF, the pipeline:
1. **Detects document type** — classifies the document as `article` or `thesis` using three signals: filename keywords (`thèse`, `dissertation`, `PhD`…), page count (>80 pages), and in-text keywords (`jury`, `acknowledgements`, `école doctorale`…). Theses are limited to **5 chunks max** in LLM context and **1 slot** among retrieved articles to prevent monopolisation.
2. **Parses by section** — GROBID extracts Abstract, Introduction, Methods, Results, Conclusion separately for precise retrieval. Falls back to raw text if GROBID is unavailable.
3. **Prepends author/year prefix** — each chunk text starts with `[Author Year]` so the embedding captures article identity alongside content.
4. **Filters parasite chunks** — short chunks (<150 chars) dominated by DOIs or years (bibliography leftovers) are discarded after splitting.
5. **Stores in ChromaDB** — with full metadata: `auteur`, `annee`, `titre`, `journal`, `doi`, `doc_type`, `section`, `num_pages`.

### PDF quality filters
The indexer silently rejects PDFs that are:
- Scans with no extractable text
- Too short (under 100 chars/page)
- Encoded with garbage/binary data
- Containing incoherent text (under 50% real words — scientific Unicode symbols like δ, α, ‰ are excluded from this ratio)

Rejected files are logged to `refchat_ingest_log.txt`.

### Re-indexing after updates
A **full re-indexation** is required to benefit from these features on your existing library:
- Document type detection (`doc_type` field)
- Author/year prefix in embeddings
- Improved sentence-aware chunk splitting
- Parasite chunk filtering

To force re-indexation of all files, delete `refchat_index_db.json` and re-run the indexer. The BM25 cache (`bm25_index.pkl`) is automatically regenerated afterward.

---

## Audit Tool

After indexing, run the audit tool from a terminal to find and fix issues:

```bash
python Audit_database.py
```

It scans the entire database and lets you interactively:
- **Correct** metadata (author, year, title, filename)
- **Delete** an article from the database
- **Inject** a missing abstract manually

Files modified via the audit tool are registered in `audit_modifications.json` and will never be overwritten by future indexing runs.

---

## File Structure

```
RefChat/
├── refchat_main.py          # Standalone launcher
├── refchat_web.py           # Flask app + web UI
├── refchat_llm.py           # RAG logic, prompts, BM25 + cross-encoder retrieval
├── refchat_ingest.py        # PDF ingestion pipeline
├── refchat_config.py        # Config loader/saver
├── refchat_config.json      # User settings (auto-generated)
├── refchat_index_db.json    # Ingestion tracking (auto-generated)
├── refchat_ingest_log.txt   # Ingestion log (auto-generated)
├── refchat_ignore.txt       # Blacklist of PDFs to skip
├── audit_modifications.json # Audit registry (auto-generated)
├── bm25_index.pkl           # BM25 index cache (auto-generated)
├── Audit_database.py        # CLI audit tool
├── RefChat.bat              # Windows launcher
├── refchat_icon.png         # App icon
└── chroma_db/               # Vector database (auto-generated)
```

---

## Privacy

All processing happens locally on your machine. No data is sent to external servers unless you:
- Use the **Mistral API** mode (queries are sent to Mistral's cloud)
- Enable **Semantic Scholar** web search (queries are sent to their public API)

The cross-encoder, BM25 index, and E5 embeddings all run locally — internet is only needed on first launch to download model weights.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions, bug reports and feature requests are welcome! Please open an issue or submit a pull request.
