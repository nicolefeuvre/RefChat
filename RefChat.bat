@echo off
title RefChat

set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

set CUDA_VISIBLE_DEVICES=0
set OLLAMA_GPU_OVERHEAD=0
set OLLAMA_NUM_GPU=99

set VENV_DIR=%PROJECT_DIR%.venv
set VENV_PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe
set DEPS_STAMP=%PROJECT_DIR%.venv\.deps_ok

echo.
echo  ==========================================
echo   RefChat - LLM for your Zotero library
echo  ==========================================
echo.

:: -----------------------------------------------------------------
:: STEP 1 : Find Python
:: -----------------------------------------------------------------
echo [1/5] Looking for Python...

set PYTHON_EXE=

if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if "%PYTHON_EXE%"=="" if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if "%PYTHON_EXE%"=="" if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if "%PYTHON_EXE%"=="" if exist "%LOCALAPPDATA%\Programs\Python\Python39\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python39\python.exe"
if "%PYTHON_EXE%"=="" if exist "%PROGRAMFILES%\Python312\python.exe" set "PYTHON_EXE=%PROGRAMFILES%\Python312\python.exe"
if "%PYTHON_EXE%"=="" if exist "%PROGRAMFILES%\Python311\python.exe" set "PYTHON_EXE=%PROGRAMFILES%\Python311\python.exe"
if "%PYTHON_EXE%"=="" if exist "%PROGRAMFILES%\Python310\python.exe" set "PYTHON_EXE=%PROGRAMFILES%\Python310\python.exe"
if "%PYTHON_EXE%"=="" if exist "%USERPROFILE%\anaconda3\python.exe" set "PYTHON_EXE=%USERPROFILE%\anaconda3\python.exe"
if "%PYTHON_EXE%"=="" if exist "%USERPROFILE%\miniconda3\python.exe" set "PYTHON_EXE=%USERPROFILE%\miniconda3\python.exe"
if "%PYTHON_EXE%"=="" if exist "%LOCALAPPDATA%\anaconda3\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\anaconda3\python.exe"
if "%PYTHON_EXE%"=="" if exist "%LOCALAPPDATA%\miniconda3\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\miniconda3\python.exe"
if "%PYTHON_EXE%"=="" where python >nul 2>&1 && set "PYTHON_EXE=python"

if "%PYTHON_EXE%"=="" (
    echo    ERROR: Python not found.
    echo    Install Python 3.11 from https://python.org
    pause
    exit /b 1
)
echo    OK - %PYTHON_EXE%

:: -----------------------------------------------------------------
:: STEP 2 : Venv
:: -----------------------------------------------------------------
echo.
echo [2/5] Preparing the venv...

if exist "%VENV_PYTHON%" goto venv_ok

echo    Creating venv...
if exist "%VENV_DIR%" rmdir /s /q "%VENV_DIR%"
"%PYTHON_EXE%" -m venv "%VENV_DIR%"
if %ERRORLEVEL% NEQ 0 (
    echo    ERROR: venv creation failed.
    pause
    exit /b 1
)
if not exist "%VENV_PYTHON%" (
    echo    ERROR: python.exe missing in .venv\Scripts\
    pause
    exit /b 1
)
echo    Venv created.

:: Force dep reinstall since venv is new
if exist "%DEPS_STAMP%" del "%DEPS_STAMP%"
goto venv_done

:venv_ok
echo    OK - Existing venv.

:venv_done

:: -----------------------------------------------------------------
:: STEP 3 : Dependencies
:: Uses a stamp file (.deps_ok) to skip re-checking on every launch.
:: The stamp is deleted if the venv is recreated.
:: -----------------------------------------------------------------
echo.
echo [3/5] Checking dependencies...

if exist "%DEPS_STAMP%" (
    echo    OK - Already installed ^(stamp present^).
    goto deps_done
)

"%VENV_PYTHON%" -c "import flask, langchain_chroma, langchain_huggingface, chromadb, fitz, sentence_transformers" >nul 2>&1
if %ERRORLEVEL%==0 goto deps_mark_ok

:: -- Full installation (first time or after stamp deletion) --
echo    Installing ^(first time ~5 min^)...
echo    Do not close this window.

"%VENV_PYTHON%" -m pip install --upgrade pip --quiet

echo    Installing Flask + LangChain...
"%VENV_PYTHON%" -m pip install flask langchain langchain-community langchain-chroma langchain-huggingface langchain-ollama langchain-mistralai langchain-text-splitters pypdf --quiet --no-warn-script-location
if %ERRORLEVEL% NEQ 0 (
    echo    ERROR Flask/LangChain.
    pause
    exit /b 1
)

echo    Installing ChromaDB + PyMuPDF...
"%VENV_PYTHON%" -m pip install chromadb pymupdf --quiet --no-warn-script-location
if %ERRORLEVEL% NEQ 0 (
    echo    ERROR ChromaDB/PyMuPDF.
    pause
    exit /b 1
)

echo    Installing sentence-transformers + tqdm...
"%VENV_PYTHON%" -m pip install sentence-transformers tqdm --quiet --no-warn-script-location
if %ERRORLEVEL% NEQ 0 (
    echo    ERROR sentence-transformers.
    pause
    exit /b 1
)

"%VENV_PYTHON%" -c "import torch" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo    Installing PyTorch CPU...
    "%VENV_PYTHON%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
)

"%VENV_PYTHON%" -c "import flask, langchain_chroma, langchain_huggingface, chromadb, fitz, sentence_transformers" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo    ERROR: packages still missing after installation.
    pause
    exit /b 1
)
echo    Packages successfully installed.

:deps_mark_ok
echo %DATE% %TIME% > "%DEPS_STAMP%"
echo    OK

:deps_done

:: -----------------------------------------------------------------
:: STEP 4 : Ollama
:: -----------------------------------------------------------------
echo.
echo [4/5] Checking Ollama...

set OLLAMA_OK=0
where ollama >nul 2>&1
if %ERRORLEVEL%==0 set OLLAMA_OK=1

if %OLLAMA_OK%==0 if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Ollama"
    set OLLAMA_OK=1
)

if %OLLAMA_OK%==0 (
    echo    Ollama not found. Installing via winget...
    winget install Ollama.Ollama --accept-package-agreements --accept-source-agreements --silent
    set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Ollama"
    where ollama >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo    ERROR: Install manually from https://ollama.com/download
        pause
        exit /b 1
    )
    set OLLAMA_OK=1
    echo    Ollama installed.
) else (
    echo    OK - Ollama found.
)

curl -s http://127.0.0.1:11434/ >nul 2>&1
if %ERRORLEVEL%==0 goto ollama_ready

echo    Starting Ollama...
start /b "" ollama serve >nul 2>&1

set WAIT=0
:wait_ollama
timeout /t 1 /nobreak >nul
curl -s http://127.0.0.1:11434/ >nul 2>&1
if %ERRORLEVEL%==0 goto ollama_ready
set /a WAIT=%WAIT%+1
if %WAIT% LSS 12 goto wait_ollama
echo    Ollama slow to start, continuing anyway...

:ollama_ready
echo    OK - Ollama ready.

ollama list 2>nul | findstr /i "mistral" >nul 2>&1
if %ERRORLEVEL%==0 goto models_ok

echo    Downloading Mistral ^(~4 GB, one-time only^)...
ollama pull mistral:7b-instruct-q4_0
if %ERRORLEVEL% NEQ 0 ollama pull mistral
echo    Model ready.

goto models_done

:models_ok
echo    OK - Mistral model present.

:models_done

:: -----------------------------------------------------------------
:: STEP 5 : Launch
:: -----------------------------------------------------------------
echo.
echo [5/5] Launching RefChat...
echo.
echo  ------------------------------------------
echo   Interface: http://localhost:5001
echo   Browser opens automatically.
echo   To stop: Ctrl+C
echo  ------------------------------------------
echo.

"%VENV_PYTHON%" "%PROJECT_DIR%refchat_web.py"

echo.
echo RefChat stopped. Press any key to close.

echo.
echo If you encounter package errors on next launch,
echo delete the file .venv\.deps_ok to force reinstallation.
pause
