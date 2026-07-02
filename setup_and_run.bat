@echo off
REM ============================================================================
REM Redrob AI Hiring Intelligence Platform - Windows Setup & Run
REM ============================================================================
REM Usage: setup_and_run.bat [command]
REM Commands: setup | rank | dashboard | test
REM ============================================================================

echo.
echo ╔══════════════════════════════════════════════════════════════════════╗
echo ║           Redrob AI Hiring Intelligence Platform                     ║
echo ║              Windows Setup ^& Execution Guide                       ║
echo ╚══════════════════════════════════════════════════════════════════════╝
echo.

set VENV_DIR=.venv
set PYTHON_CMD=python

REM Check Python
%PYTHON_CMD% --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.11+ first.
    exit /b 1
)

echo [OK] Found Python version:
%PYTHON_CMD% --version

REM Create virtual environment
if not exist "%VENV_DIR%" (
    echo [STEP] Creating virtual environment...
    %PYTHON_CMD% -m venv %VENV_DIR%
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

REM Activate virtual environment
echo [STEP] Activating virtual environment...
call %VENV_DIR%\Scripts\activate.bat

echo [OK] Virtual environment activated
echo [OK] Python path: 
where python

REM Upgrade pip
echo [STEP] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

REM Install dependencies
echo [STEP] Installing dependencies (this may take 5-10 minutes)...
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
pip install -r requirements.txt
pip install -e .

echo [OK] All dependencies installed

REM Verify
echo [STEP] Verifying installation...
python -c "import redrob_ranker; print('Package OK')"
python -c "from sentence_transformers import SentenceTransformer; print('Embeddings OK')"

echo.
echo ╔══════════════════════════════════════════════════════════════════════╗
echo ║              SETUP COMPLETE                                          ║
echo ╚══════════════════════════════════════════════════════════════════════╝
echo.
echo Next steps:
echo   1. Place candidates.jsonl in data\
echo   2. Run ranking:    .venv\Scripts\python.exe scripts\rank.py data\candidates.jsonl data\job_description.txt output\submission.csv
echo   3. Launch dashboard: .venv\Scripts\streamlit.exe run dashboard\app.py
echo.

REM Handle commands
if "%~1"=="rank" goto run_ranking
if "%~1"=="dashboard" goto run_dashboard
if "%~1"=="test" goto run_tests
goto end

:run_ranking
echo [STEP] Running ranking pipeline...
python scripts\rank.py data\candidates.jsonl data\job_description.txt output\submission.csv --top-k 100
goto end

:run_dashboard
echo [STEP] Launching dashboard...
streamlit run dashboard\app.py --server.port 8501
goto end

:run_tests
echo [STEP] Running tests...
pytest tests\ -v
goto end

:end
echo.
echo Done.
pause
