"""
RefChat — Standalone Launcher
Automatically handles: venv, pip dependencies, Ollama, models, then starts the Flask app.
Designed to be compiled into a .exe with PyInstaller (RefChat.spec).
"""
import os
import sys
import subprocess
import shutil
import urllib.request
import time
import pathlib

PROJECT_DIR = pathlib.Path(__file__).parent.resolve()
VENV_DIR    = PROJECT_DIR / ".venv"

# Python executable inside the venv once created
if sys.platform == "win32":
    VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
    VENV_PIP    = VENV_DIR / "Scripts" / "pip.exe"
else:
    VENV_PYTHON = VENV_DIR / "bin" / "python"
    VENV_PIP    = VENV_DIR / "bin" / "pip"

REQUIRED_PACKAGES = [
    "flask",
    "langchain",
    "langchain-community",
    "langchain-chroma",
    "langchain-huggingface",
    "langchain-ollama",
    "langchain-mistralai",
    "langchain-text-splitters",
    "chromadb",
    "pymupdf",
    "sentence-transformers",
    "tqdm",
]

CHECK_IMPORTS = [
    "flask",
    "langchain_chroma",
    "langchain_huggingface",
    "chromadb",
    "fitz",
    "sentence_transformers",
]

REQUIRED_MODELS = ["mistral:7b-instruct-q4_0", "mistral"]

# ── Helpers ────────────────────────────────────────────────────────────────────

def _print(msg: str):
    """Print with immediate flush (useful for PyInstaller console)."""
    print(msg, flush=True)

def _run(cmd, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, **kwargs)

# ── STEP 1: Venv ──────────────────────────────────────────────────────────────

def ensure_venv():
    """Creates the venv if it does not exist."""
    if VENV_PYTHON.exists():
        return
    _print("\n[2/5] Creating isolated Python environment (.venv)...")
    result = _run([sys.executable, "-m", "venv", str(VENV_DIR), "--upgrade-deps"])
    if result.returncode != 0:
        _print("  ERROR: unable to create the venv.")
        sys.exit(1)
    _print("  Venv created.")

# ── STEP 2: pip dependencies ──────────────────────────────────────────────────

def _all_imports_ok() -> bool:
    """Checks that all critical packages can be imported inside the venv."""
    check = "; ".join(f"import {m}" for m in CHECK_IMPORTS)
    result = _run(
        [str(VENV_PYTHON), "-c", check],
        capture_output=True
    )
    return result.returncode == 0

def ensure_dependencies():
    """Installs missing dependencies into the venv."""
    if _all_imports_ok():
        _print("[3/5] Dependencies OK.")
        return

    _print("[3/5] Installing dependencies (first time ~5 min)...")
    _print("      Do not close this window.\n")

    # Upgrade pip silently
    _run([str(VENV_PIP), "install", "--upgrade", "pip", "--quiet"],
         capture_output=True)

    # Install main packages
    result = _run([
        str(VENV_PIP), "install", *REQUIRED_PACKAGES,
        "--quiet", "--no-warn-script-location"
    ])
    if result.returncode != 0:
        _print("  Error while installing packages.")
        _print("  Check your internet connection and restart RefChat.")
        sys.exit(1)

    # PyTorch CPU (lighter, ~200 MB)
    check_torch = _run(
        [str(VENV_PYTHON), "-c", "import torch"],
        capture_output=True
    )
    if check_torch.returncode != 0:
        _print("  Installing PyTorch (CPU, ~200 MB)...")
        _run([
            str(VENV_PIP), "install", "torch",
            "--index-url", "https://download.pytorch.org/whl/cpu",
            "--quiet"
        ])

    # Final check
    if not _all_imports_ok():
        _print("\n  ERROR: some packages could not be installed.")
        _print("  Restart RefChat — if the error persists, check your connection.")
        sys.exit(1)

    _print("  ✅ All dependencies are installed.")

# ── STEP 3: Ollama ────────────────────────────────────────────────────────────

def ensure_ollama_installed():
    """Installs Ollama via winget if not found."""
    if shutil.which("ollama"):
        return True

    # Check default install path
    ollama_default = pathlib.Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    if ollama_default.exists():
        os.environ["PATH"] += os.pathsep + str(ollama_default.parent)
        return True

    _print("[4/5] Ollama not detected. Installing automatically...")

    if shutil.which("winget"):
        result = _run([
            "winget", "install", "Ollama.Ollama",
            "--accept-package-agreements", "--accept-source-agreements", "--silent"
        ])
        ollama_path = str(pathlib.Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama")
        os.environ["PATH"] += os.pathsep + ollama_path

        if shutil.which("ollama") or ollama_default.exists():
            _print("  ✅ Ollama installed.")
            return True
        else:
            _print("  ❌ Automatic installation failed.")
    else:
        _print("  winget not available.")

    _print("\n  Install Ollama manually: https://ollama.com/download")
    _print("  Then restart RefChat.")
    return False

def start_ollama_server():
    """Starts ollama serve in the background if not already running."""
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/", timeout=2)
        _print("  Ollama already running.")
        return
    except Exception:
        pass

    _print("  Starting Ollama server...")
    kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    subprocess.Popen(["ollama", "serve"], **kwargs)

    for _ in range(12):
        time.sleep(1)
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/", timeout=2)
            _print("  Ollama ready.")
            return
        except Exception:
            pass
    _print("  ⚠️  Ollama is slow to start — continuing anyway.")

def ensure_models():
    """Downloads Mistral models if not already available."""
    try:
        result = _run(["ollama", "list"], capture_output=True, text=True)
        available = result.stdout.lower()
    except Exception:
        _print("  ⚠️  Unable to list Ollama models.")
        return

    has_mistral = any(
        ("mistral" in line and "instruct" in line) or line.strip().startswith("mistral ")
        for line in available.splitlines()
    )

    if not has_mistral:
        _print("\n  📥 Downloading Mistral model (~4 GB, one-time only)...")
        _print("     This may take several minutes.")
        result = _run(["ollama", "pull", "mistral:7b-instruct-q4_0"])
        if result.returncode != 0:
            _print("  Fallback: downloading standard mistral model...")
            _run(["ollama", "pull", "mistral"])
        _print("  ✅ Model ready.")
    else:
        _print("  Mistral model already available.")

# ── ENTRY POINT ────────────────────────────────────────────────────────────────

def main():
    _print("\n" + "═" * 56)
    _print("  RefChat — Starting")
    _print("═" * 56)

    # If already running inside the venv, go directly to the app
    already_in_venv = str(VENV_DIR).lower() in sys.executable.lower()

    if not already_in_venv:
        # ── Step 1: Python OK (sys.executable is the system python / PyInstaller exe)
        _print(f"\n[1/5] Python: {sys.executable}")

        # ── Step 2: Venv
        ensure_venv()

        # ── Step 3: Dependencies
        ensure_dependencies()

        # ── Relaunch this same script inside the venv so packages are available
        _print("\n  Relaunching inside the isolated environment...")
        result = _run([str(VENV_PYTHON), str(__file__)] + sys.argv[1:])
        sys.exit(result.returncode)

    # ════════════════════════════════════════════════════════
    # We are now inside the venv — packages are available
    # ════════════════════════════════════════════════════════
    _print(f"\n[1/5] Python (venv): {sys.executable}")
    _print("[2/5] Isolated environment active.")
    _print("[3/5] Dependencies available.")

    # ── Step 4: Ollama
    _print("\n[4/5] Checking Ollama...")
    if not ensure_ollama_installed():
        sys.exit(1)
    start_ollama_server()
    ensure_models()

    # ── Step 5: Launch Flask app
    _print("\n[5/5] Launching RefChat...\n")
    _print("═" * 56)
    _print("  Web interface: http://localhost:5001")
    _print("  The browser will open automatically.")
    _print("  To stop: Ctrl+C or close this window.")
    _print("═" * 56 + "\n")

    import threading, webbrowser

    def _open_browser():
        time.sleep(2)
        webbrowser.open("http://localhost:5001")

    threading.Thread(target=_open_browser, daemon=True).start()

    # Add project folder to path for local imports
    sys.path.insert(0, str(PROJECT_DIR))

    from refchat_web import app
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)


if __name__ == "__main__":
    main()
