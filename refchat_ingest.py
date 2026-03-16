import os
import re
import json
import unicodedata
from datetime import datetime
import sys
import subprocess
import importlib.util
import requests
import time

def install_if_missing():
    """Checks and installs missing dependencies at startup."""
    dependencies = {
        "langchain": "langchain",
        "langchain_community": "langchain-community",
        "langchain_huggingface": "langchain-huggingface",
        "langchain_chroma": "langchain-chroma",
        "langchain_text_splitters": "langchain-text-splitters",
        "chromadb": "chromadb",
        "torch": "torch",
        "tqdm": "tqdm",
        "sentence_transformers": "sentence-transformers",
        "pypdf": "pypdf",
        "requests": "requests",
        "bs4": "beautifulsoup4",
        "lxml": "lxml"
    }

    missing = []
    for module_name, package_name in dependencies.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)

    if missing:
        print(f"📦 Missing libraries detected: {', '.join(missing)}")
        print("⏳ Auto-installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            print("✅ Installation complete. Restarting script...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            print(f"❌ Installation error: {e}")
            sys.exit(1)

install_if_missing()

# ── CPU OPTIMISATION ──
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
try:
    import torch
    torch.set_num_threads(8)
except ImportError:
    pass

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm
from bs4 import BeautifulSoup

# ============================================================
# CENTRAL CONFIG
# ============================================================
from refchat_config import EMBEDDING_MODEL, DB_PATH, PERSONAL_DATA, get as config_get
import pathlib as _pathlib

ZOTERO_PATH  = config_get("zotero_path", "")
_BASE_DIR    = _pathlib.Path(__file__).parent.resolve()

LOG_FILE       = str(PERSONAL_DATA / "refchat_ingest_log.txt")
TRACKING_FILE  = str(PERSONAL_DATA / "refchat_index_db.json")
IGNORE_FILE    = str(_pathlib.Path(__file__).parent / "refchat_ignore.txt")  # user-editable, stays with scripts
AUDIT_LOG_FILE = str(PERSONAL_DATA / "audit_modifications.json")
OCR_QUEUE_FILE = str(PERSONAL_DATA / "refchat_ocr_queue.json")

CHUNK_SIZE    = 2000
CHUNK_OVERLAP = 250
BATCH_SIZE    = 500

MIN_CHARS_TOTAL      = 50
MIN_RATIO_READABLE   = 0.70
MIN_CHARS_PER_PAGE   = 100
MIN_RATIO_REAL_WORDS = 0.50

GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

_T0_GLOBAL = None

def _log(msg):
    try:
        tqdm.write(str(msg))
    except Exception:
        print(str(msg), flush=True)

def _elapsed():
    if _T0_GLOBAL is None:
        return "0s"
    s = int(time.time() - _T0_GLOBAL)
    return f"{s//60}m{s%60:02d}s" if s >= 60 else f"{s}s"

def _bar(idx, total, w=22):
    if total == 0:
        return "[" + "░"*w + "]  0.0%"
    n   = int(w * idx / total)
    pct = idx / total * 100
    return f"[{'█'*n}{'░'*(w-n)}] {pct:5.1f}%"

# ============================================================
# GROBID AUTO-START
# ============================================================
def check_and_start_grobid():
    _log("  🔍 Checking GROBID server (port 8070)...")
    try:
        r = requests.get("http://localhost:8070/api/isalive", timeout=2)
        if r.status_code == 200:
            _log("  ✅ GROBID server is online and ready.")
            return True
    except Exception:
        pass

    _log("  ⚠️  GROBID unreachable. Attempting auto-launch via Docker...")
    try:
        subprocess.run([
            "docker", "run", "-d", "--rm",
            "-p", "8070:8070",
            "lfoppiano/grobid:0.8.1"
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        _log("  ⏳ Container starting… (server takes about 10-20s to boot)")

        for i in range(30):
            time.sleep(2)
            try:
                r = requests.get("http://localhost:8070/api/isalive", timeout=2)
                if r.status_code == 200:
                    _log(f"  ✅ GROBID is operational (started in ~{i*2}s)!")
                    return True
            except Exception:
                pass

        _log("  ❌ Failure: GROBID server took too long to respond.")
        return False

    except FileNotFoundError:
        _log("  ❌ Error: 'docker' command not found. Is Docker Desktop open?")
        return False
    except subprocess.CalledProcessError:
        _log("  ❌ Docker error: unable to start container (port 8070 may already be in use).")
        return False

# ============================================================
# QUALITY CHECK AND METADATA
# ============================================================
def extract_metadata_from_filename(filename):
    name = os.path.splitext(filename)[0]
    pattern_zotero = r'^(.+?)\s*-\s*(\d{4})\s*-\s*(.+)$'
    m = re.match(pattern_zotero, name)
    if m:
        raw_author   = m.group(1).strip()
        year         = m.group(2).strip()
        title        = m.group(3).strip()
        first_author = re.split(r'\s+et\s+|\s+and\s+|,', raw_author, maxsplit=1)[0].strip()
        first_author = re.sub(r'\s*et\s+al\.?.*$', '', first_author, flags=re.IGNORECASE).strip()
        return {"auteur": first_author, "auteurs": raw_author, "annee": year, "titre": title}
    return {"auteur": "", "auteurs": "", "annee": "", "titre": name}

_DOI_PATTERNS = [
    re.compile(r'(?:https?://(?:dx\.)?doi\.org/|DOI:\s*|doi:\s*)(10\.\d{4,9}/[^\s,;\]\"\'\<\>]+)', re.IGNORECASE),
    re.compile(r'\b(10\.\d{4,9}/[^\s,;\]\"\'\<\>]{5,})', re.IGNORECASE),
]

_JOURNAL_PATTERNS = [
    re.compile(r'(Journal\s+of\s+Hydrology)', re.IGNORECASE),
    re.compile(r'(Hydrogeology\s+Journal)', re.IGNORECASE),
    re.compile(r'(Hydrological\s+(?:Processes?|Sciences?))', re.IGNORECASE),
    re.compile(r'(Water\s+Resources?\s+Research)', re.IGNORECASE),
    re.compile(r'(Groundwater)', re.IGNORECASE),
    re.compile(r'(Earth(?:\s+and)?\s+Planetary\s+Science\s+Letters?)', re.IGNORECASE),
    re.compile(r'(Geochimica\s+et\s+Cosmochimica\s+Acta)', re.IGNORECASE),
    re.compile(r'(Chemical\s+Geology)', re.IGNORECASE),
    re.compile(r'(Ore\s+Geology\s+Reviews?)', re.IGNORECASE),
    re.compile(r'(Economic\s+Geology)', re.IGNORECASE),
    re.compile(r'(Mineralium\s+Deposita)', re.IGNORECASE),
    re.compile(r'(Tectonophysics)', re.IGNORECASE),
    re.compile(r'(Tectonics)', re.IGNORECASE),
    re.compile(r'(Geomorphology)', re.IGNORECASE),
    re.compile(r'(Sedimentary\s+Geology)', re.IGNORECASE),
    re.compile(r'(Palaeogeography,?\s+Palaeoclimatology,?\s+Palaeoecology)', re.IGNORECASE),
    re.compile(r'(Gondwana\s+Research)', re.IGNORECASE),
    re.compile(r'(Comptes?\s+Rendus?\s+Geoscience)', re.IGNORECASE),
    re.compile(r'(American\s+Mineralogist)', re.IGNORECASE),
    re.compile(r'(European\s+Journal\s+of\s+Mineralogy)', re.IGNORECASE),
    re.compile(r'(Contributions?\s+to\s+Mineralogy\s+and\s+Petrology)', re.IGNORECASE),
    re.compile(r'(Journal\s+of\s+Petrology)', re.IGNORECASE),
    re.compile(r'(Lithos)', re.IGNORECASE),
    re.compile(r'(Environmental\s+(?:Science|Geology|Earth\s+Sciences?))', re.IGNORECASE),
    re.compile(r'(Science\s+of\s+the\s+Total\s+Environment)', re.IGNORECASE),
    re.compile(r'(Applied\s+Geochemistry)', re.IGNORECASE),
]

_THESIS_FILENAME_RE = re.compile(
    r'\b(th[eè]se|thesis|dissertation|m[eé]moire|phd|doctorat)\b',
    re.IGNORECASE
)

_THESIS_TEXT_RE = re.compile(
    r'\b(jury|remerciements?|acknowledgements?|th[eè]se|dissertation|'
    r'doctorat|directeur\s+de\s+th[eè]se|thesis\s+supervisor|'
    r'école\s+doctorale|graduate\s+school|universit[eé])\b',
    re.IGNORECASE
)

def detect_doc_type(filename, num_pages, text_sample):
    """Classify a document as 'thesis' or 'article' using 3 combined signals:
    - filename contains thesis-related keywords (strong: weight 2)
    - number of pages > 80 (weight 1)
    - text sample contains >= 3 thesis-related keywords (weight 1)
    Returns 'thesis' if total weight >= 2, else 'article'.
    """
    signals = 0
    if _THESIS_FILENAME_RE.search(os.path.splitext(filename)[0]):
        signals += 2
    if num_pages > 80:
        signals += 1
    if len(_THESIS_TEXT_RE.findall(text_sample[:5000])) >= 3:
        signals += 1
    return "thesis" if signals >= 2 else "article"


def extract_journal_doi(first_page_text):
    zone = first_page_text[:3000]
    doi = ""
    for pattern in _DOI_PATTERNS:
        m = pattern.search(zone)
        if m:
            doi = re.sub(r'[\s\n]+.*$', '', m.group(1).rstrip('.'))
            break
    journal = ""
    for pattern in _JOURNAL_PATTERNS:
        m = pattern.search(zone)
        if m:
            journal = m.group(1).strip()
            break
    return journal, doi

def ratio_readable_chars(text):
    if not text: return 0.0
    n = sum(1 for c in text if unicodedata.category(c).startswith(('L', 'N', 'P', 'Z')) or c in (' ', '\n', '\t'))
    return n / len(text)

_SCIENTIFIC_SYMBOL_RE = re.compile(
    r'^[\u03b1-\u03c9\u0391-\u03a9\u0394\u03b4\u2030\u2080-\u2089'
    r'\u00b0\u00b1\u2192\u2260\u2248\u00b5\u2030\u2081-\u2089\u00b2\u00b3]+$'
)

def ratio_real_words(text):
    words = text.split()
    if not words: return 0.0
    def _is_real(w):
        return 2 <= len(w) <= 25 or bool(_SCIENTIFIC_SYMBOL_RE.match(w))
    return len([w for w in words if _is_real(w)]) / len(words)

def contains_pdf_garbage(text):
    patterns = [r'obj<<', r'endobj', r'stream\s', r'BT\s+/F', r'â€|Ã©|Ã |ï¬', r'ÿþ', r'\x00']
    return any(re.search(p, text) for p in patterns)

def evaluate_quality(text, num_pages):
    if len(text.strip()) < MIN_CHARS_TOTAL:
        return False, f"SCAN/EMPTY ({len(text.strip())} chars)"
    chars_per_page = len(text) / max(num_pages, 1)
    if chars_per_page < MIN_CHARS_PER_PAGE:
        return False, f"TOO_SHORT ({chars_per_page:.0f} chars/page)"
    if contains_pdf_garbage(text[:2000]):
        return False, "PDF_GARBAGE"
    if ratio_readable_chars(text) < MIN_RATIO_READABLE:
        return False, f"CORRUPT_ENCODING ({ratio_readable_chars(text):.0%} readable)"
    if ratio_real_words(text) < MIN_RATIO_REAL_WORDS:
        return False, f"INCOHERENT_TEXT ({ratio_real_words(text):.0%} plausible words)"
    return True, "OK"

# ============================================================
# CLEANING AND GROBID EXTRACTION (HYBRID APPROACH)
# ============================================================

_REFERENCE_LINE_PATTERNS = [
    re.compile(r'^\s*[\[\(]\d{1,3}[\]\)]\s+\S'),
    re.compile(r'^\s*\d{1,3}\.\s+[A-ZÀÁÂ]'),
    re.compile(r'^[A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ][a-zàáâãäåæçèéêëìíîïðñòóôõö]{1,20},?\s+[A-Z]{1,3}[.,].*\b(?:19|20)\d{2}\b'),
    re.compile(r'doi\s*:\s*10\.\d{4,9}/',    re.IGNORECASE),
    re.compile(r'https?://doi\.org/10\.\d{4,9}/', re.IGNORECASE),
    re.compile(r'\b(?:19|20)\d{2}\b.{5,100}(?:\d+\s*[\(:\,]\s*\d+\s*[\):\,–\-]\s*\d+)'),
]

def _is_reference_line(line):
    l = line.strip()
    if not l or len(l) < 10:
        return False
    for pat in _REFERENCE_LINE_PATTERNS:
        if pat.search(line):
            return True
    return False

def _trim_references_at_end(text):
    if not text or len(text) < 80:
        return text

    lines = text.split("\n")
    n = len(lines)
    if n < 3:
        return text

    start_zone  = max(0, int(n * 0.35)) if n > 20 else max(0, n - 15)
    block_start = None
    consec      = 0

    for i in range(start_zone, n):
        l = lines[i].strip()
        if not l: continue
        if _is_reference_line(lines[i]):
            if block_start is None:
                block_start = i
            consec += 1
            if consec >= 3:
                true_start = block_start
                while true_start > 0 and not lines[true_start - 1].strip():
                    true_start -= 1
                cut = "\n".join(lines[:true_start]).rstrip()
                if len(cut) >= 20:
                    return cut
        else:
            if len(l.split()) >= 4:
                consec      = 0
                block_start = None
    return text

def extract_sections_grobid(pdf_path):
    try:
        with open(pdf_path, 'rb') as f:
            response = requests.post(
                GROBID_URL,
                files={'input': f},
                data={'generateIDs': '1', 'consolidateHeader': '0'},
                timeout=300
            )

        if response.status_code != 200:
            return None, f"GROBID API error: {response.status_code}"

        soup = BeautifulSoup(response.content, 'xml')
        document = {"Abstract": "", "Sections": []}

        abstract_tag = soup.find('abstract')
        if abstract_tag:
            document["Abstract"] = abstract_tag.get_text(separator=" ", strip=True)

        zones = []
        if soup.find('body'): zones.append(soup.find('body'))
        if soup.find('back'): zones.append(soup.find('back'))

        for zone in zones:
            for ref in zone.find_all(['listBibl', 'biblStruct']):
                ref.decompose()

            current_section = {"title": "Introduction", "text": []}

            for tag in zone.find_all(['head', 'p']):
                if tag.name == 'head':
                    if current_section["text"]:
                        document["Sections"].append({
                            "titre": current_section["title"],
                            "texte": " ".join(current_section["text"])
                        })
                    current_section = {"title": tag.get_text(separator=" ", strip=True), "text": []}

                elif tag.name == 'p':
                    p_text = tag.get_text(separator=" ", strip=True)
                    if len(p_text) > 20:
                        current_section["text"].append(p_text)

            if current_section["text"]:
                document["Sections"].append({
                    "titre": current_section["title"],
                    "texte": " ".join(current_section["text"])
                })

        return document, "Success"

    except Exception as e:
        return None, f"GROBID exception: {e}"

# ============================================================
# CHUNKING AND INDEXING
# ============================================================

_PARASITE_RE = re.compile(r'10\.\d{4,9}/\S+|\b(?:19|20)\d{2}\b')

def _is_parasite_chunk(text, min_chars=150, max_ratio=0.5):
    """True if the chunk is short AND dominated by DOIs/years (bibliography leftovers)."""
    if len(text) >= min_chars:
        return False
    words = text.split()
    if not words:
        return True
    return len(_PARASITE_RE.findall(text)) / len(words) > max_ratio

def chunk_section(section_name, section_text, metadata_base, splitter):
    if len(section_text.strip()) < 100:
        return []
    auteur = metadata_base.get("auteur", "")
    annee  = metadata_base.get("annee", "")
    prefix = f"[{auteur} {annee}] " if auteur or annee else ""
    if "e5" in EMBEDDING_MODEL.lower():
        text = f"passage: {prefix}{section_text}"
    else:
        text = f"{prefix}{section_text}"
    docs = splitter.create_documents([text], metadatas=[{**metadata_base, "section": section_name}])
    return [d for d in docs if not _is_parasite_chunk(d.page_content)]

# Regex to detect abstract header + body in raw PDF text (used when GROBID fails)
_ABSTRACT_HEADER_RE = re.compile(
    r'(?:^|\n)\s*(?:ABSTRACT|Abstract|RÉSUMÉ|Résumé|résumé|RESUME|Summary|SUMMARY)\s*[:\-—]?\s*\n+'
    r'([\s\S]{100,2500}?)'
    r'(?=\n\s*(?:\d[\.\s]|Introduction|INTRODUCTION|Keywords|KEYWORDS|'
    r'Mots[- ]clés|Index[- ]terms|Background|BACKGROUND|\n\n\n))',
    re.MULTILINE
)
# Simpler fallback: "Abstract" on same line as text (no newline between header and body)
_ABSTRACT_INLINE_RE = re.compile(
    r'(?:^|\n)\s*(?:ABSTRACT|Abstract|RÉSUMÉ|Résumé|Summary)[:\s\-—]+([A-Z][^#\n]{100,2500}?)(?=\n\s*\n)',
    re.MULTILINE
)

def _extract_abstract_from_raw(text):
    """Regex fallback: tries to find the abstract in raw PDF text. Returns '' if not found."""
    sample = text[:6000]
    for pattern in (_ABSTRACT_HEADER_RE, _ABSTRACT_INLINE_RE):
        m = pattern.search(sample)
        if m:
            abstract = m.group(1).strip()
            if len(abstract) >= 100:
                return abstract
    return ""


def _first_page_proxy(text, min_chars=200, target_chars=1200):
    """Last-resort proxy: take the first meaningful content lines as a proxy abstract.
    Skips short header lines (title, authors, journal, DOI) and returns the first
    substantial paragraph-like content. Works even on poor OCR / scanned PDFs."""
    lines = text.split('\n')
    content_lines = []
    total = 0
    for line in lines:
        stripped = line.strip()
        # Skip short lines (headers, page numbers, DOI lines, etc.)
        if len(stripped) < 40:
            continue
        # Stop at obvious section headers
        if re.match(r'^(?:\d[\.\s]|Introduction|INTRODUCTION|Methods|METHODS|Keywords)', stripped):
            break
        content_lines.append(stripped)
        total += len(stripped)
        if total >= target_chars:
            break
    result = ' '.join(content_lines)
    return result[:target_chars] if len(result) >= min_chars else ""


def _get_abstract(doc_structure, raw_text):
    """3-level abstract extraction: GROBID → regex → first-page proxy.
    Returns (abstract_text, method_label)."""
    # Level 1: GROBID
    if doc_structure:
        abstract = doc_structure.get("Abstract", "")
        if abstract and len(abstract) >= 100:
            return abstract, "GROBID"

    # Level 2: regex on raw text
    abstract = _extract_abstract_from_raw(raw_text)
    if abstract:
        return abstract, "regex"

    # Level 3: first-page proxy (always works if text is readable)
    abstract = _first_page_proxy(raw_text)
    if abstract:
        return abstract, "proxy"

    return "", "none"


# ============================================================
# OCR QUEUE MANAGEMENT
# ============================================================

def _load_ocr_queue() -> dict:
    """Returns {filename: pdf_path} dict from OCR queue file."""
    if not os.path.exists(OCR_QUEUE_FILE):
        return {}
    try:
        with open(OCR_QUEUE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_ocr_queue(queue: dict):
    os.makedirs(os.path.dirname(OCR_QUEUE_FILE), exist_ok=True)
    with open(OCR_QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(queue, f, ensure_ascii=False, indent=2)

def _add_to_ocr_queue(filename: str, pdf_path: str):
    """Add a PDF to the OCR queue."""
    queue = _load_ocr_queue()
    queue[filename] = pdf_path
    _save_ocr_queue(queue)

def _remove_from_ocr_queue(filename: str):
    """Remove a PDF from the OCR queue after successful OCR ingestion."""
    queue = _load_ocr_queue()
    queue.pop(filename, None)
    _save_ocr_queue(queue)

# ============================================================
# OCR PIPELINE (EasyOCR + PyMuPDF)
# ============================================================

_easyocr_reader = None  # Module-level cache — loaded once per process

def _get_easyocr_reader(log_cb=None):
    """Return cached EasyOCR reader (FR+EN), initialising on first call."""
    global _easyocr_reader
    if _easyocr_reader is not None:
        return _easyocr_reader
    try:
        import easyocr
        import torch
        gpu = torch.cuda.is_available()
        if log_cb:
            log_cb(f"  🔧 Initialising EasyOCR (GPU={'yes' if gpu else 'no'}) …")
        _easyocr_reader = easyocr.Reader(["fr", "en"], gpu=gpu, verbose=False)
        return _easyocr_reader
    except ImportError:
        raise RuntimeError(
            "EasyOCR is not installed. Run: pip install easyocr"
        )

def ocr_pdf_easyocr(pdf_path: str, log_cb=None) -> str:
    """
    Render each page of a scanned PDF at 200 DPI with PyMuPDF and run
    EasyOCR on it.  Returns the concatenated text of all recognised pages.
    Processes at most MAX_OCR_PAGES pages to stay within reasonable limits.
    """
    MAX_OCR_PAGES = 40
    try:
        import fitz          # PyMuPDF
        import numpy as np
    except ImportError:
        raise RuntimeError(
            "PyMuPDF is not installed. Run: pip install pymupdf"
        )

    reader = _get_easyocr_reader(log_cb)
    doc    = fitz.open(pdf_path)
    pages  = min(len(doc), MAX_OCR_PAGES)
    texts  = []

    if log_cb:
        log_cb(f"  🔍 OCR — {pages} page(s) to process …")

    for i in range(pages):
        page = doc[i]
        # Render at 200 DPI (scale factor ≈ 200/72 ≈ 2.78)
        mat  = fitz.Matrix(200 / 72, 200 / 72)
        pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)

        results = reader.readtext(img, detail=0, paragraph=True)
        page_text = "\n".join(results)
        texts.append(page_text)

        if log_cb and (i + 1) % 5 == 0:
            log_cb(f"  📄 OCR progress: {i + 1}/{pages} pages")

    doc.close()
    full_text = "\n\n".join(t for t in texts if t.strip())
    if log_cb:
        log_cb(f"  ✅ OCR complete — {len(full_text)} chars extracted")
    return full_text


def run_ocr_ingest(pdf_path: str, db, log_cb=None) -> dict:
    """
    OCR-ingest a single scanned PDF:
      1. OCR the PDF with EasyOCR
      2. Build chunks via the fallback path (no GROBID)
      3. Add chunks to ChromaDB
      4. Mark file as ingested and remove from OCR queue
    Returns a result dict: {success, filename, n_chunks, error}
    """
    filename = os.path.basename(pdf_path)
    if log_cb:
        log_cb(f"📄 OCR ingest: {filename[:80]}")

    # ── OCR ──────────────────────────────────────────────────────────────────
    try:
        ocr_text = ocr_pdf_easyocr(pdf_path, log_cb)
    except Exception as e:
        if log_cb:
            log_cb(f"  ❌ OCR failed: {e}")
        return {"success": False, "filename": filename, "n_chunks": 0, "error": str(e)}

    if len(ocr_text.strip()) < MIN_CHARS_TOTAL:
        msg = f"OCR produced too little text ({len(ocr_text.strip())} chars)"
        if log_cb:
            log_cb(f"  ⛔ {msg}")
        return {"success": False, "filename": filename, "n_chunks": 0, "error": msg}

    # ── Metadata ─────────────────────────────────────────────────────────────
    meta     = extract_metadata_from_filename(filename)
    journal, doi = extract_journal_doi(ocr_text[:2000])
    doc_type = detect_doc_type(filename, 0, ocr_text)

    metadata_base = {
        "source":    pdf_path,
        "filename":  filename,
        "num_pages": 0,
        "auteur":    meta["auteur"],
        "auteurs":   meta["auteurs"],
        "annee":     meta["annee"],
        "titre":     meta["titre"],
        "journal":   journal,
        "doi":       doi,
        "doc_type":  doc_type,
        "ocr":       True,
    }

    # ── Chunk via fallback path ───────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    cleaned_text = _trim_references_at_end(ocr_text)

    abstract, method = _get_abstract(None, cleaned_text)
    if abstract:
        if log_cb:
            log_cb(f"  📋 Abstract via {method} ({len(abstract)} chars)")
        chunks = chunk_section("Abstract", abstract, metadata_base, splitter)
        chunks.extend(chunk_section("Full text", cleaned_text, metadata_base, splitter))
    else:
        if log_cb:
            log_cb("  ⚠️  No abstract detected — full text only")
        chunks = chunk_section("Full text", cleaned_text, metadata_base, splitter)

    if not chunks:
        msg = "No chunks generated from OCR text"
        if log_cb:
            log_cb(f"  ⛔ {msg}")
        return {"success": False, "filename": filename, "n_chunks": 0, "error": msg}

    # ── v1.5: tag each chunk with its sequence number (windowed retrieval) ──
    for _seq, _doc in enumerate(chunks):
        _doc.metadata["chunk_seq"] = _seq

    # ── Index into ChromaDB ───────────────────────────────────────────────────
    try:
        for i in range(0, len(chunks), BATCH_SIZE):
            db.add_documents(chunks[i:i + BATCH_SIZE])
        if log_cb:
            log_cb(f"  ⚡ {len(chunks)} chunks indexed")
    except Exception as e:
        if log_cb:
            log_cb(f"  ❌ ChromaDB error: {e}")
        return {"success": False, "filename": filename, "n_chunks": 0, "error": str(e)}

    # ── Tracking ─────────────────────────────────────────────────────────────
    already_ingested = load_already_ingested()
    mark_as_ingested(already_ingested, pdf_path, len(chunks))
    save_already_ingested(already_ingested)
    _remove_from_ocr_queue(filename)

    if log_cb:
        log_cb(f"  ✅ Done — {filename[:60]}")

    return {"success": True, "filename": filename, "n_chunks": len(chunks), "error": None}


def create_article_chunks(pdf_path, raw_text_fallback, metadata_base, splitter):
    doc_structure, status = extract_sections_grobid(pdf_path)
    chunks = []

    if doc_structure:
        found_sections = []

        # Get abstract via 3-level cascade
        abstract, method = _get_abstract(doc_structure, raw_text_fallback)
        if abstract:
            chunks.extend(chunk_section("Abstract", abstract, metadata_base, splitter))
            found_sections.append(f"Abstract[{method}]")

        for sec in doc_structure.get("Sections", []):
            title = sec["titre"]
            text  = sec["texte"]

            if re.search(r'acknowledgment|remerciement', title, re.IGNORECASE):
                continue

            text = _trim_references_at_end(text)

            c = chunk_section(title[:50], text, metadata_base, splitter)
            if c:
                chunks.extend(c)
                found_sections.append(title[:20])

        if chunks:
            _log(f"      🟢 [GROBID] {len(found_sections)} sections extracted.")
            return chunks, False, "Success"

    _log(f"      🟡 [FALLBACK] GROBID failed/empty ({status}). Using raw text.")
    cleaned_text = _trim_references_at_end(raw_text_fallback)

    abstract, method = _get_abstract(None, cleaned_text)
    if abstract:
        _log(f"      📋 [FALLBACK] Abstract via {method} ({len(abstract)} chars)")
        chunks = chunk_section("Abstract", abstract, metadata_base, splitter)
        chunks.extend(chunk_section("Full text", cleaned_text, metadata_base, splitter))
        return chunks, True, status

    _log(f"      ⚠️ [FALLBACK] No abstract found — Full text only")
    return chunk_section("Full text", cleaned_text, metadata_base, splitter), True, status

# ============================================================
# DATABASE MANAGEMENT AND MAIN
# ============================================================

def load_blacklist():
    if not os.path.exists(IGNORE_FILE):
        return set()
    with open(IGNORE_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def load_already_ingested():
    if not os.path.exists(TRACKING_FILE):
        return {}
    with open(TRACKING_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def load_audited_files():
    if not os.path.exists(AUDIT_LOG_FILE):
        return set()
    try:
        with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    except Exception:
        return set()

def save_already_ingested(already_ingested):
    with open(TRACKING_FILE, "w", encoding="utf-8") as f:
        json.dump(already_ingested, f, ensure_ascii=False, indent=2)

def mark_as_ingested(already_ingested, pdf_path, nb_chunks):
    already_ingested[pdf_path] = {
        "date":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "chunks":   nb_chunks,
        "filename": os.path.basename(pdf_path)
    }

def _detect_device_and_batch():
    """Auto-detect best compute device + safe batch size.
    Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU.
    CUDA batch size scaled to VRAM to avoid OOM on small GPUs.
    """
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
        return "mps", 128   # Apple Silicon
    return "cpu", 32        # CPU: small batch = less RAM pressure during ingest


def load_embedding():
    _log(f"⏳ Loading {EMBEDDING_MODEL} ...")
    device, batch_size = _detect_device_and_batch()
    _log(f"   ⚙️  Device: {device.upper()} — batch_size={batch_size}")

    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
        multi_process=False,
    )

    if "e5" in EMBEDDING_MODEL.lower():
        _log("   ℹ️  E5 model detected — 'passage: ' prefix added to chunks")
    _log(f"   ✅ Model loaded.")
    return embedding_function

def _scan_zotero_pdfs(zotero_path):
    pdfs = []
    for root, dirs, files in os.walk(zotero_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, file))
    return pdfs

def main():
    global _T0_GLOBAL, ZOTERO_PATH
    _T0_GLOBAL  = time.time()
    ZOTERO_PATH = config_get("zotero_path", ZOTERO_PATH)

    SEP = "─" * 56
    _log("╔══════════════════════════════════════════════════════╗")
    _log("║      RefChat — Library Indexing                      ║")
    _log("╚══════════════════════════════════════════════════════╝")
    _log(f"  📂 Folder    : {ZOTERO_PATH}")
    _log(f"  🧠 Embedding : {EMBEDDING_MODEL}")
    _log(f"  📅 Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _log("")

    grobid_ready = check_and_start_grobid()
    if not grobid_ready:
        _log("  ⚠️ WARNING: All PDFs will be processed using the Fallback method (raw text).")
    _log("")

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"--- LOG {datetime.now().strftime('%Y-%m-%d %H:%M')} ---\n\n")

    _log(SEP)
    _log("  STEP 1/3 — Loading embedding model")
    _log(SEP)
    t0 = time.time()
    embedding_function = load_embedding()
    _log(f"  ✅ Model ready in {time.time()-t0:.1f}s\n")

    db_exists = os.path.exists(DB_PATH)
    if db_exists:
        db       = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
        nb_before = db._collection.count()
        _log(f"  ✅ Existing database: {nb_before} chunks already indexed.")
    else:
        db        = None
        nb_before = 0
        _log("  ℹ️  No database found — creating from scratch.")

    already_ingested = load_already_ingested()
    _log(f"  📋 Tracking: {len(already_ingested)} articles already processed.\n")

    _log(SEP)
    _log("  STEP 2/3 — Scanning PDFs")
    _log(SEP)
    t0 = time.time()
    all_pdfs       = _scan_zotero_pdfs(ZOTERO_PATH)
    blacklist      = load_blacklist()
    audited_files  = load_audited_files()

    pdfs_to_process = []
    for p in all_pdfs:
        filename = os.path.basename(p)
        if p not in already_ingested and filename not in blacklist and filename not in audited_files:
            pdfs_to_process.append(p)

    total = len(pdfs_to_process)

    _log(f"  📄 PDFs found          : {len(all_pdfs)}")
    _log(f"  🚫 Ignored files       : {len(blacklist)} (blacklist)")
    _log(f"  🛡️ Protected (Audit)   : {len(audited_files)} (manually modified)")
    _log(f"  ✅ Already in database : {len(already_ingested)}")
    _log(f"  🆕 Remaining to index  : {total}  (scanned in {time.time()-t0:.1f}s)\n")

    if not pdfs_to_process:
        _log("  ✅ Database is already complete and up to date!")
        _log(f"     {nb_before} chunks indexed in total.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", ". ", "\n", " "]
    )

    valid_pdfs    = 0
    rejected_pdfs = 0
    rejection_reasons = {}

    _log(SEP)
    _log("  STEP 3/3 — Vectorisation and indexing")
    _log(SEP)
    _log("")

    for idx, pdf_path in enumerate(pdfs_to_process, start=1):
        file   = os.path.basename(pdf_path)
        t0_pdf = time.time()

        _log(f"{_bar(idx, total)}  [{idx}/{total}]  ⏱ {_elapsed()}")
        _log(f"  📄 {file[:80]}")

        try:
            loader    = PyPDFLoader(pdf_path)
            pages     = loader.load()
            raw_text  = "".join([p.page_content for p in pages]).strip()
            num_pages = len(pages)

            is_valid, reason = evaluate_quality(raw_text, num_pages)
            if not is_valid:
                duration = time.time() - t0_pdf
                rejected_pdfs += 1
                key = reason.split(" (")[0]
                rejection_reasons[key] = rejection_reasons.get(key, 0) + 1
                _log(f"  ⛔ REJECTED — {reason}  ({duration:.1f}s)")
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"REJECTED ({reason}): {pdf_path}\n")
                # Image-based PDFs (SCAN/EMPTY or TOO_SHORT) → add to OCR queue
                if key in ("SCAN/EMPTY", "TOO_SHORT"):
                    _add_to_ocr_queue(file, pdf_path)
                    _log(f"  🔎 Added to OCR queue for later processing")
                mark_as_ingested(already_ingested, pdf_path, 0)
                continue

            meta    = extract_metadata_from_filename(file)
            text_p1 = pages[0].page_content if pages else ""
            journal, doi = extract_journal_doi(text_p1)
            doc_type = detect_doc_type(file, num_pages, raw_text)

            meta_label = f"{meta['auteur']} ({meta['annee']})"
            if journal:
                meta_label += f"  —  {journal[:45]}"
            if doc_type == "thesis":
                meta_label += "  📖 [THESIS]"
            _log(f"  🏷️  {meta_label}")

            metadata_base = {
                "source":    pdf_path,
                "filename":  file,
                "num_pages": num_pages,
                "auteur":    meta["auteur"],
                "auteurs":   meta["auteurs"],
                "annee":     meta["annee"],
                "titre":     meta["titre"],
                "journal":   journal,
                "doi":       doi,
                "doc_type":  doc_type,
            }

            chunks, is_fallback, grobid_status = create_article_chunks(
                pdf_path, raw_text, metadata_base, splitter
            )

            # ── v1.5: tag each chunk with its sequence number (windowed retrieval) ──
            for _seq, _doc in enumerate(chunks):
                _doc.metadata["chunk_seq"] = _seq

            if is_fallback:
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"WARNING (Fallback GROBID -> {grobid_status}): {pdf_path}\n")

            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i + BATCH_SIZE]
                if db is None:
                    db = Chroma.from_documents(
                        documents=batch, embedding=embedding_function,
                        persist_directory=DB_PATH
                    )
                    db_exists = True
                else:
                    db.add_documents(batch)

            mark_as_ingested(already_ingested, pdf_path, len(chunks))
            valid_pdfs += 1

            duration = time.time() - t0_pdf
            speed    = len(raw_text) / max(duration, 0.1) / 1000
            _log(f"  ⚡ {len(chunks)} chunks  |  {duration:.1f}s  |  {speed:.1f} k-chars/s")

            avg_s    = (time.time() - _T0_GLOBAL) / idx
            remain_s = int(avg_s * (total - idx))
            if remain_s > 0:
                remain_str = f"{remain_s//60}m{remain_s%60:02d}s" if remain_s >= 60 else f"{remain_s}s"
                _log(f"  ⏳ Estimated remaining time: ~{remain_str}")
            _log("")

            if valid_pdfs % 10 == 0:
                save_already_ingested(already_ingested)

        except Exception as e:
            duration = time.time() - t0_pdf
            rejected_pdfs += 1
            rejection_reasons["CRITICAL ERROR"] = rejection_reasons.get("CRITICAL ERROR", 0) + 1
            _log(f"  ❌ ERROR ({duration:.1f}s): {str(e)[:120]}")
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"CRITICAL ERROR: {pdf_path} -> {str(e)}\n")
            _log("")

    save_already_ingested(already_ingested)

    nb_after  = db._collection.count() if db else 0
    total_dur = int(time.time() - _T0_GLOBAL)
    dur_str   = f"{total_dur//60}m{total_dur%60:02d}s" if total_dur >= 60 else f"{total_dur}s"
    avg_speed = valid_pdfs / max(total_dur, 1) * 60

    _log("╔══════════════════════════════════════════════════════╗")
    _log(f"║  🎉  DONE  —  total duration: {dur_str:<23}  ║")
    _log("╚══════════════════════════════════════════════════════╝")
    _log(f"  📄 Articles indexed    : {valid_pdfs}")
    _log(f"  ⛔ Articles rejected   : {rejected_pdfs}")
    _log(f"  🧩 Chunks before       : {nb_before}")
    _log(f"  🧩 Chunks after        : {nb_after}")
    _log(f"  🧩 Chunks added        : {nb_after - nb_before}")
    _log(f"  📊 Average speed       : {avg_speed:.1f} articles/min")
    _log(f"  📁 ChromaDB database   : {DB_PATH}")
    _log(f"  💾 Log (rejects/errors): {LOG_FILE}")

    if rejection_reasons:
        _log("")
        _log("  Rejection breakdown:")
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
            _log(f"    — {reason}: {count} file(s)")

if __name__ == "__main__":
    main()
