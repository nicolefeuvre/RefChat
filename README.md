## What is RefChat?

RefChat is a self-hosted, web-based **Retrieval-Augmented Generation (RAG)** application that lets you query your personal Zotero PDF library using natural language. It intelligently indexes your scientific articles with GROBID (section-aware parsing) and answers your questions by citing only the sources in your database.

**Key features:**
- 🧠 **Smart indexing** — uses [GROBID](https://github.com/kermitt2/grobid) (via Docker) to split PDFs by section (Abstract, Introduction, Methods, Results, etc.)
- 🔍 **Three-stage retrieval** — E5 dense embeddings + BM25 keyword search (Reciprocal Rank Fusion) + cross-encoder reranking for maximum precision
- 🔬 **HyDE retrieval** *(v1.5)* — optional Hypothetical Document Embeddings: the LLM generates a hypothetical scientific passage to dramatically improve retrieval precision (+10–20%)
- 📐 **Windowed retrieval** *(v1.5)* — chunks indexed with a sequence number; neighboring chunks are automatically included to avoid truncated context
- 🏷️ **Automatic thematization** — clusters your library into themes using BERTopic; each theme gets a semantic label and is used to silently filter RAG results
- 🔎 **OCR pipeline** — detects image-based PDFs (scans), queues them, and re-indexes them with EasyOCR (GPU-accelerated)
- 📄 **Article detail panel** *(v1.5)* — click the 🔍 button next to any cited article to see its full abstract, metadata, DOI link, and indexed sections in a slide-in panel
- 💾 **Conversation persistence** *(v1.5)* — conversations are auto-saved to disk and listed in the sidebar; reload any past session with one click
- 📤 **Export conversation** *(v1.5)* — download the current chat as a clean Markdown file (sources included)
- 📖 **Thesis detection** — automatically classifies long documents as theses/dissertations and limits their context footprint
- 💬 **Multi-mode chat** — question answering, thematic synthesis, reference lookup, author search
- 🌐 **Web search integration** — optional [Semantic Scholar](https://api.semanticscholar.org/) API
- 🤖 **Model flexibility** — Mistral API (cloud) or local models via Ollama (Mistral 7B Q4, Mixtral 8x7B)
- 🗃️ **Persistent memory** — optional conversation history across turns
- 🛠️ **Audit tool** — CLI tool to fix missing abstracts or corrupted metadata
- 🔁 **Incremental indexing** — re-run safely; already-processed files are skipped

---

## What's new in v1.5

### HyDE — Hypothetical Document Embeddings

Instead of embedding the raw user query, RefChat can ask the LLM to generate a short 2–3 sentence scientific passage that would hypothetically answer the question. This passage is then embedded and used for the vector search — its embedding is typically much closer to real document chunks than the raw query.

**Toggle:** click the **🔬 HyDE OFF** button in the sidebar to activate. When enabled it adds ~1–2 s of latency (one extra LLM call). Recommended for precision-critical questions in large libraries.

---

### Windowed retrieval

New articles (indexed with v1.5+) receive a `chunk_seq` integer in their metadata. During retrieval, RefChat automatically expands each selected chunk to include its immediately preceding and following chunks from the same article — preventing the common problem of relevant passages being split at a chunk boundary.

> **Note:** existing articles in your database are not affected. Re-index an article to benefit from windowed retrieval.

---

### Article detail panel

Click the **🔍** button next to any article name or cited source tag to open a slide-in panel showing:

- Full title, authors, year, journal
- DOI with a clickable link
- Assigned theme (if thematization has been run)
- Full abstract text
- List of all indexed sections (Abstract, Introduction, Results…)
- **Open PDF** button (opens the file via the local server)

---

### Conversation persistence

Every conversation is automatically saved to disk in `personal_data/refchat_conversations/` as a JSON file after each message. The **History** section at the bottom of the sidebar lists all past sessions (most recent first). Click any entry to restore the full conversation; click ✕ to delete it permanently.

**New conversation:** the **➕ New conversation** button in the sidebar clears the current chat and resets the session (memory is also cleared).

---

### Export conversation

The **💾 Export conversation** button (sidebar) downloads the current chat as a Markdown file (`refchat-YYYY-MM-DD.md`) containing all questions, answers, and cited sources. The Markdown file can be opened in any editor or converted to PDF via a browser print.

---

## What's new in v1.4

### OCR pipeline for image-based PDFs

Many scientific PDFs (older scans, publisher-locked files) contain no extractable text. RefChat now detects these automatically during indexing and provides a one-click OCR pipeline.

**How it works:**

1. **Detection** — during normal indexing, PDFs that fail the `SCAN/EMPTY` or `TOO_SHORT` quality checks are added to an OCR queue (`refchat_ocr_queue.json`) instead of being silently discarded.
2. **Sidebar badge** — a **🔍 X unindexed image PDFs** badge appears in the sidebar whenever the queue is non-empty.
3. **OCR & re-index** — clicking "🔎 OCR & index" runs the full pipeline:
   - Each page is rendered at 200 DPI with **PyMuPDF** → numpy array
   - **EasyOCR** (FR + EN, GPU if available) extracts text
   - Text goes through the standard fallback chunk pipeline (abstract detection + full text splitting)
   - Chunks are added to ChromaDB; file is marked as ingested and removed from the queue
4. **Live log** — progress is streamed in the same panel used for regular indexing

**Installation:** EasyOCR is auto-installed by `RefChat.bat` on first launch (step 3b). GPU support is automatic if CUDA is available.

---

## What's new in v1.3

### Interactive thematization system

RefChat can now automatically organise your library into semantic themes and use those themes to silently improve RAG precision.

**Workflow:**

1. Click **🏷️ Organise library** in the sidebar
2. **Dry-run preview** — BERTopic clusters your articles; results are shown in a validation modal with per-theme cards:
   - Article count and list
   - Quality warnings (too small, too large, parasitic content, generic label)
   - Suggested rename if the auto-label is weak
3. **Edit before committing** — rename themes, delete unwanted ones, apply suggested labels
4. **Validate** — writes themes to ChromaDB metadata + `refchat_themes.json`

**Theme-aware RAG (3-priority detection):**

| Priority | Mechanism | When used |
|---|---|---|
| 1 | Active sidebar filter | User explicitly selected a theme via the accordion |
| 2 | Semantic cosine similarity | Query embedding compared to theme name embeddings (threshold 0.52) |
| 3 | Keyword regex | Keyword match on theme names |

When a theme is detected, the ChromaDB search is restricted to that theme's articles — improving precision without changing the prompt.

**Sidebar accordion** — themes are displayed as a collapsible list in the sidebar. Clicking a theme sets a persistent filter badge visible above the input field; clicking again clears it.

---

## What's new in v1.2

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
refchat_web.py        ← Flask web app + complete UI (single-file HTML/JS)
refchat_llm.py        ← RAG logic: retrieval, BM25, cross-encoder, prompts, Semantic Scholar
refchat_ingest.py     ← PDF ingestion pipeline (GROBID + fallback + ChromaDB + OCR queue)
refchat_theme.py      ← BERTopic clustering, quality checks, dry-run preview, apply mapping
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
| **GPU (optional)** | CUDA-compatible GPU accelerates EasyOCR and embedding inference |

---

## Quick Start (Windows)

1. **Clone or download** this repository
2. **Double-click `RefChat.bat`** — it will:
   - Find Python on your system
   - Create an isolated `.venv`
   - Install all Python dependencies (including EasyOCR)
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
    tqdm requests beautifulsoup4 lxml rank-bm25 easyocr

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

### Stage 0 — Theme filtering (optional)

If a theme has been set (via sidebar selection or semantic/keyword detection), the ChromaDB search is restricted to articles belonging to that theme. This narrows the candidate pool before hybrid search, improving precision on libraries with many articles.

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

### PDF quality filters & OCR queue
The indexer applies quality checks to every PDF. Files that fail are logged to `refchat_ingest_log.txt`:

| Rejection reason | Cause | OCR queued? |
|---|---|---|
| `SCAN/EMPTY` | Fewer than 50 extractable characters | ✅ Yes |
| `TOO_SHORT` | Under 100 chars/page (likely a scan) | ✅ Yes |
| `PDF_GARBAGE` | Binary/encoding corruption | ❌ No |
| `CORRUPT_ENCODING` | Under 70% readable characters | ❌ No |
| `INCOHERENT_TEXT` | Under 50% real words | ❌ No |

Image-based PDFs (`SCAN/EMPTY` and `TOO_SHORT`) are added to `refchat_ocr_queue.json` for later OCR processing. A badge in the sidebar shows when PDFs are waiting.

### Re-indexing after updates
A **full re-indexation** is required to benefit from these features on your existing library:
- Document type detection (`doc_type` field)
- Author/year prefix in embeddings
- Improved sentence-aware chunk splitting
- Parasite chunk filtering

To force re-indexation of all files, delete `refchat_index_db.json` and re-run the indexer. The BM25 cache (`bm25_index.pkl`) is automatically regenerated afterward.

---

## Thematization

### Running the clustering

Click **🏷️ Organiser la bibliothèque** to open the thematization workflow:

1. **Preview** — BERTopic builds a topic model over your article embeddings. A validation modal shows all detected themes, their articles, quality warnings, and suggested renames.
2. **Edit** — rename or delete themes before committing.
3. **Validate** — themes are written to ChromaDB metadata. The sidebar accordion is updated immediately.

### Quality checks

Each theme is analysed for:
- **Too small** (< 5 articles) — may indicate a noise cluster
- **Too large** (> 25 articles) — may need splitting
- **Parasitic content** — topic label contains acknowledgements, bibliography keywords
- **Generic label** — label contains only very common words (water, rock, hydrogen…)

### Theme filter in chat

The theme filter works at three priority levels:
1. **Active filter** (sidebar accordion click) — hard filter, only that theme's articles are searched
2. **Semantic detection** — query embedding compared to theme name embeddings
3. **Keyword detection** — regex match on theme name tokens

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
├── refchat_web.py               # Flask app + web UI
├── refchat_llm.py               # RAG logic, prompts, BM25 + cross-encoder retrieval
├── refchat_ingest.py            # PDF ingestion pipeline + OCR queue
├── refchat_theme.py             # BERTopic clustering + thematization workflow
├── refchat_config.py            # Config loader/saver
├── Audit_database.py            # CLI audit tool
├── RefChat.bat                  # Windows launcher (auto-installs everything)
├── test_theme_detection.py      # Standalone test: semantic vs keyword theme detection
│
├── refchat_config.json          # User settings (auto-generated)
├── refchat_index_db.json        # Ingestion tracking (auto-generated)
├── refchat_ingest_log.txt       # Ingestion log — rejects & errors (auto-generated)
├── refchat_themes.json          # Theme → article mapping (auto-generated after thematization)
├── refchat_ocr_queue.json       # Image PDFs awaiting OCR (auto-generated)
├── audit_modifications.json     # Audit registry (auto-generated)
├── bm25_index.pkl               # BM25 index cache (auto-generated)
├── refchat_ignore.txt           # Blacklist of PDFs to skip
├── refchat_icon.png             # App icon
└── chroma_db/                   # Vector database (auto-generated)
```

---

## Privacy

All processing happens locally on your machine. No data is sent to external servers unless you:
- Use the **Mistral API** mode (queries are sent to Mistral's cloud)
- Enable **Semantic Scholar** web search (queries are sent to their public API)

The cross-encoder, BM25 index, E5 embeddings, BERTopic clustering, and EasyOCR all run locally — internet is only needed on first launch to download model weights.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions, bug reports and feature requests are welcome! Please open an issue or submit a pull request.
