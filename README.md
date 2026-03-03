## What is RefChat?

RefChat is a self-hosted, web-based **Retrieval-Augmented Generation (RAG)** application that lets you query your personal Zotero PDF library using natural language. It intelligently indexes your scientific articles with GROBID (section-aware parsing) and answers your questions by citing only the sources in your database.

**Key features:**
- 🧠 **Smart indexing** — uses [GROBID](https://github.com/kermitt2/grobid) (via Docker) to split PDFs by section (Abstract, Introduction, Methods, Results, etc.)
- 🔍 **Semantic search** — powered by `intfloat/multilingual-e5-large` embeddings and ChromaDB
- 💬 **Multi-mode chat** — question answering, thematic synthesis, reference lookup, author search
- 🌐 **Hybrid web search** — optional integration with [Semantic Scholar](https://api.semanticscholar.org/) API
- 🤖 **Model flexibility** — use Mistral API (cloud) or local models via Ollama (Mistral 7B Q4, Mixtral 8x7B)
- 🗃️ **Persistent memory** — optional conversation history across turns
- 🛠️ **Audit tool** — CLI tool to fix missing abstracts or corrupted metadata
- 🔁 **Incremental indexing** — re-run safely; already-processed files are skipped

---

## Architecture

```
refchat_main.py       ← Launcher: sets up venv, Ollama, models, then starts Flask
refchat_web.py        ← Flask web app + complete UI (single-file HTML/JS)
refchat_llm.py        ← RAG logic: retrieval, prompts, LLM loading, Semantic Scholar
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
| **Ollama** | Required to run local models offline. Download from [ollama.com](https://ollama.com/download). *(Note: The Windows `.bat` launcher can auto-install it, but manual install is required for Linux/Mac).* |
| **Python** 3.9 – 3.12 | Tested on 3.11 |
| **Ollama** | Auto-installed by the launcher (Windows); manual install on Linux/Mac |
| **Docker Desktop** | Required for GROBID (smart PDF parsing). Without it, raw-text fallback is used |
| **RAM** | 8 GB minimum (16 GB+ recommended for local models) |
| **Disk** | ~5 GB for models + ChromaDB |
| **Internet** | Required on first run to download models and the embedding model |

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
    tqdm requests beautifulsoup4 lxml
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

Settings are stored in `refchat_config.json` and can be edited via the **⚙️ Settings** panel in the UI:

| Key | Description |
|---|---|
| `zotero_path` | Path to your `Zotero/storage` folder |
| `db_path` | Path where the ChromaDB vector database is stored |
| `llm_model` | Active LLM: `mistral-large-latest` (API), `mistral-light`, `mistral`, or `mixtral` |
| `mistral_api_key` | Your Mistral API key (get one at [console.mistral.ai](https://console.mistral.ai)) |
| `semantic_scholar_api_key` | Optional free API key for higher Semantic Scholar rate limits |

---

## Models

| Model | Key | Mode | Size | Notes |
|---|---|---|---|---|
| Mistral Large (cloud) | `api` | Cloud | — | Requires API key, largest context |
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

## Web Search (Semantic Scholar)

Click the **🗄️ Local only** button in the sidebar to cycle between three modes:

- **Local only** — searches your Zotero database exclusively
- **🔀 Local + Web** — combines your database with Semantic Scholar results
- **🌐 Web only** — searches Semantic Scholar exclusively (top 10 results)

Web results include a link to the DOI or a Google Scholar search. A free Semantic Scholar API key can be added to increase rate limits.

---

## Indexing

### First run
The setup wizard handles everything. It will:
1. Detect your Zotero folder automatically (or let you browse to it)
2. Optionally start GROBID via Docker
3. Index all PDFs and show a live log

### Subsequent runs
Click **🔄 Index library** in the sidebar. Already-processed files are skipped automatically. Indexing can be interrupted at any time (`Ctrl+C`) and will resume exactly where it left off.

### PDF quality filters
The indexer silently rejects PDFs that are:
- Scans with no extractable text
- Too short (under 100 chars/page)
- Encoded with garbage/binary data
- Containing incoherent text (under 50% real words)

Rejected files are logged to `refchat_ingest_log.txt`.

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
├── refchat_main.py         # Standalone launcher
├── refchat_web.py          # Flask app + web UI
├── refchat_llm.py          # RAG logic and prompts
├── refchat_ingest.py       # PDF ingestion pipeline
├── refchat_config.py       # Config loader/saver
├── refchat_config.json     # User settings (auto-generated)
├── refchat_index_db.json   # Ingestion tracking (auto-generated)
├── refchat_ingest_log.txt  # Ingestion log (auto-generated)
├── refchat_ignore.txt      # Blacklist of PDFs to skip
├── audit_modifications.json# Audit registry (auto-generated)
├── Audit_database.py       # CLI audit tool
├── RefChat.bat             # Windows launcher
├── refchat_icon.png        # App icon
└── chroma_db/              # Vector database (auto-generated)
```

---

## Privacy

All processing happens locally on your machine. No data is sent to external servers unless you:
- Use the **Mistral API** mode (queries are sent to Mistral's cloud)
- Enable **Semantic Scholar** web search (queries are sent to their public API)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions, bug reports and feature requests are welcome! Please open an issue or submit a pull request.
