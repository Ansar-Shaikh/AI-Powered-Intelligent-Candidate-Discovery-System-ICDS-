# 🚀 Complete Quickstart Guide: Redrob AI Hiring Intelligence Platform

This guide walks you through every step from a fresh machine to a running dashboard.

---

## 📋 Prerequisites

Before starting, ensure you have:

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Python | 3.10+ | `python3 --version` |
| pip | 21+ | `pip --version` |
| Git | (optional) | `git --version` |
| RAM | 8GB+ recommended | - |
| Disk Space | 2GB free | - |

**Install Python (if not installed):**
- **Ubuntu/Debian:** `sudo apt update && sudo apt install python3 python3-venv python3-pip`
- **macOS:** `brew install python3` or download from python.org
- **Windows:** Download from https://python.org/downloads/

---

## 📁 Step 1: Get the Project

### Option A: Download the ZIP

```bash
# Download the project zip
cd ~/Downloads
# Extract
unzip redrob_ai_platform.zip -d ~/
cd ~/redrob_ai_platform
```

### Option B: Clone from Git (if you pushed to repo)

```bash
git clone <your-repo-url>
cd redrob_ai_hiring
```

**Verify you're in the project root:**
```bash
ls
# Should see: src/  scripts/  dashboard/  requirements.txt  README.md
```

---

## 🏗️ Step 2: Create Virtual Environment

A virtual environment isolates project dependencies from your system Python.

### Linux/macOS

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Verify (should show path to .venv/bin/python)
which python
```

### Windows (Command Prompt)

```cmd
:: Create virtual environment
python -m venv .venv

:: Activate it
.venv\Scripts\activate.bat

:: Verify
where python
```

### Windows (PowerShell)

```powershell
# Create virtual environment
python -m venv .venv

# Activate it (may need Set-ExecutionPolicy -ExecutionPolicy RemoteSigned first)
.venv\Scripts\Activate.ps1

# Verify
Get-Command python
```

**You should see `(.venv)` in your prompt when activated.**

---

## 📦 Step 3: Install Dependencies

With the virtual environment activated:

```bash
# Upgrade pip, setuptools, wheel
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU version, ~200MB)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install sentence-transformers (downloads BAAI/bge-small-en-v1.5 model ~130MB)
pip install sentence-transformers

# Install all other dependencies
pip install -r requirements.txt

# Install the package itself in editable mode
pip install -e .
```

**Expected output:**
```
Successfully installed redrob-ai-hiring-intelligence-1.0.0
```

**Troubleshooting:**
- If `torch` install fails: `pip install torch` (without index-url, larger download)
- If memory error: `pip install --no-cache-dir torch`
- On Windows with VS Build Tools issues: use pre-built wheels

---

## ✅ Step 4: Verify Installation

Run these checks to confirm everything works:

```bash
# Test 1: Core package import
python -c "import redrob_ranker; print(f'Version: {redrob_ranker.__version__}')"
# Expected: Version: 1.0.0

# Test 2: Configuration
python -c "from redrob_ranker.config import Config; c = Config(); print(f'Model: {c.EMBEDDING_MODEL}')"
# Expected: Model: BAAI/bge-small-en-v1.5

# Test 3: Embedding model (first run downloads ~130MB model)
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('BAAI/bge-small-en-v1.5'); print('Model loaded successfully')"
# Expected: Model loaded successfully

# Test 4: Run unit tests
pytest tests/ -v
# Expected: All tests pass (4 test files, 15+ tests)
```

---

## 📂 Step 5: Prepare Your Data

### Required Files

You need two files to run the pipeline:

1. **candidates.jsonl** — Candidate profiles (one JSON per line)
2. **job_description.txt** — The job description text

### Place Files

```bash
# Create data directory
mkdir -p data output

# Copy your candidates file
cp /path/to/your/candidates.jsonl data/candidates.jsonl

# Copy or create job description
cp /path/to/your/job_description.txt data/job_description.txt
```

### Verify Data Format

```bash
# Check candidates file
head -1 data/candidates.jsonl | python -m json.tool
# Should show a valid candidate JSON with candidate_id, profile, skills, etc.

# Count candidates
wc -l data/candidates.jsonl
# Expected: ~100,000 lines

# Check job description
cat data/job_description.txt | head -20
```

**Sample candidates.jsonl format:**
```json
{"candidate_id": "CAND_0000001", "profile": {"current_title": "ML Engineer", ...}, "skills": [...], ...}
{"candidate_id": "CAND_0000002", "profile": {"current_title": "Data Scientist", ...}, "skills": [...], ...}
```

---

## 🚀 Step 6: Run the Ranking Pipeline

### Quick Run (Recommended)

```bash
# With virtual environment activated:
python scripts/rank.py data/candidates.jsonl data/job_description.txt output/submission.csv
```

**This will:**
1. Load the job description
2. Load all candidates (streaming, memory-efficient)
3. Compute semantic embeddings (first run downloads model)
4. Score each candidate across 8 dimensions
5. Rank and produce submission.csv

**Expected runtime:** 8-12 minutes for 100K candidates on CPU

### With Options

```bash
# Specify top-k, batch size, and device
python scripts/rank.py \
    data/candidates.jsonl \
    data/job_description.txt \
    output/submission.csv \
    --top-k 100 \
    --batch-size 512 \
    --device cpu

# Test with fewer candidates first
python scripts/rank.py \
    data/candidates.jsonl \
    data/job_description.txt \
    output/test_submission.csv \
    --top-k 50
```

### Using the Automated Script

```bash
# Linux/macOS - Full setup and run
bash setup_and_run.sh all

# Or step by step:
bash setup_and_run.sh setup    # Setup only
bash setup_and_run.sh rank     # Run ranking only
bash setup_and_run.sh dashboard # Launch dashboard only
```

**Windows:**
```cmd
setup_and_run.bat setup
setup_and_run.bat rank
setup_and_run.bat dashboard
```

---

## ✅ Step 7: Validate Submission

```bash
python scripts/validate_submission.py output/submission.csv
```

**Expected output:**
```
Validation Result: PASS
Total rows: 100
Unique candidates: 100
```

**If validation fails, it will show specific errors to fix.**

---

## 📊 Step 8: Launch the Dashboard

### Install Dashboard Dependencies

```bash
pip install streamlit plotly
```

### Launch

```bash
streamlit run dashboard/app.py
```

**Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**Open http://localhost:8501 in your browser.**

### Dashboard Features

The dashboard has 10 interactive pages:

| Page | Purpose |
|------|---------|
| 🏠 Home | Overview and workflow guide |
| 📁 Data Upload | Upload candidates.jsonl with statistics |
| 📝 Job Description | Upload or paste JD with skill detection |
| ⚙️ Configuration | Adjust scoring weights interactively |
| 🚀 Run Ranking | Execute pipeline with progress tracking |
| 📈 Results Dashboard | Top candidates, score distributions, reasoning |
| 🔍 Semantic Search | Natural language candidate search |
| 👤 Candidate Explorer | Full profile inspection with radar charts |
| 📊 Analytics | Dataset insights and correlations |
| 💾 Export | Download CSV/JSON with validation |

---

## 🌐 Step 9: Launch the API (Optional)

```bash
pip install fastapi uvicorn

uvicorn api.main:app --reload --port 8000
```

**Open http://localhost:8000/docs for interactive API documentation.**

**Test the API:**
```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy", "pipeline_ready": true}
```

---

## 📁 Complete Project Structure After Setup

```
redrob_ai_platform/
├── .venv/                          # Virtual environment
├── data/
│   ├── candidates.jsonl            # Your candidate data
│   └── job_description.txt         # Your job description
├── output/
│   ├── submission.csv              # Generated submission
│   ├── detailed_report.json        # Full score breakdown
│   └── analysis_summary.json       # Dataset analytics
├── src/
│   └── redrob_ranker/              # Core package (12 modules)
├── scripts/
│   ├── embed.py                    # Pre-compute embeddings
│   ├── rank.py                     # Run ranking pipeline
│   ├── validate_submission.py      # Validate CSV
│   └── run_pipeline.py             # End-to-end runner
├── tests/                          # Unit tests
├── dashboard/
│   └── app.py                      # Streamlit dashboard
├── api/
│   └── main.py                     # FastAPI server
├── notebooks/
│   └── analysis.py                 # Jupyter analysis
├── config/
│   └── default.yaml                # Default configuration
├── setup_and_run.sh                # Automated setup script (Linux/Mac)
├── setup_and_run.bat               # Automated setup script (Windows)
├── requirements.txt                # Dependencies
├── pyproject.toml                  # Package metadata
└── README.md                       # Full documentation
```

---

## 🔄 Common Workflows

### Workflow 1: First Time Setup

```bash
# 1. Extract project
cd ~/redrob_ai_platform

# 2. Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
pip install -r requirements.txt
pip install -e .

# 4. Place data files
mkdir -p data output
cp ~/Downloads/candidates.jsonl data/
cp ~/Downloads/job_description.txt data/

# 5. Run pipeline
python scripts/rank.py data/candidates.jsonl data/job_description.txt output/submission.csv

# 6. Validate
python scripts/validate_submission.py output/submission.csv

# 7. Launch dashboard
pip install streamlit plotly
streamlit run dashboard/app.py
```

### Workflow 2: Re-run with Different Weights

```bash
# Edit config or use dashboard
# Then just re-run ranking:
python scripts/rank.py data/candidates.jsonl data/job_description.txt output/submission_v2.csv
```

### Workflow 3: Test with Small Dataset

```bash
# Take first 100 candidates for quick testing
head -100 data/candidates.jsonl > data/candidates_test.jsonl
python scripts/rank.py data/candidates_test.jsonl data/job_description.txt output/test.csv --top-k 50
```

### Workflow 4: Pre-compute Embeddings (Faster Re-runs)

```bash
# One-time: pre-compute embeddings
python scripts/embed.py data/candidates.jsonl data/embeddings.npz

# Future runs will be faster (embeddings cached)
# Note: Current pipeline doesn't use cached embeddings yet
# This is for future optimization
```

---

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'redrob_ranker'`

**Cause:** Package not installed or wrong directory.

**Fix:**
```bash
# Ensure you're in project root
cd ~/redrob_ai_platform

# Ensure venv is activated
source .venv/bin/activate

# Reinstall package
pip install -e .
```

### Issue: `torch` installation fails

**Fix:**
```bash
# Try without index-url (larger download but more compatible)
pip install torch

# Or use conda
conda install pytorch cpuonly -c pytorch
```

### Issue: Out of memory during ranking

**Fix:**
```bash
# Reduce batch size
python scripts/rank.py ... --batch-size 128

# Or process fewer candidates for testing
head -1000 data/candidates.jsonl > data/small.jsonl
python scripts/rank.py data/small.jsonl ...
```

### Issue: Model download is slow

**Fix:** The BAAI/bge-small-en-v1.5 model (~130MB) downloads on first run. This is normal.
```bash
# Pre-download model explicitly
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
```

### Issue: Dashboard won't start

**Fix:**
```bash
# Install streamlit
pip install streamlit plotly

# Check if port 8501 is free
lsof -i :8501  # Linux/Mac
netstat -ano | findstr 8501  # Windows

# Use different port
streamlit run dashboard/app.py --server.port 8502
```

### Issue: `pytest` not found

**Fix:**
```bash
pip install pytest pytest-cov
pytest tests/ -v
```

---

## 📊 Performance Expectations

| Step | Time | Memory |
|------|------|--------|
| Environment setup | 5-10 min | ~2GB download |
| Model download (first run) | 2-3 min | ~130MB |
| Load 100K candidates | 10-20 sec | ~500MB |
| Embed 100K candidates | 5-8 min | ~500MB |
| Score 100K candidates | 2-3 min | ~200MB |
| **Total pipeline** | **8-12 min** | **~500MB peak** |

---

## 🎯 Next Steps After Setup

1. **Explore the Dashboard** — Interactive candidate analysis
2. **Adjust Weights** — Use Configuration page to tune scoring
3. **Semantic Search** — Find candidates with natural language queries
4. **Export Results** — Download validated submission CSV
5. **Run Tests** — `pytest tests/ -v` to ensure everything works
6. **Read Documentation** — `README.md` for architecture details

---

## 📞 Support

| Resource | Location |
|----------|----------|
| Full documentation | `README.md` |
| API reference | `API_DOCUMENTATION.md` |
| Quick commands | `QUICK_REFERENCE.md` |
| Source code | `src/redrob_ranker/` |
| Tests | `tests/` |
| Configuration | `config/default.yaml` |

---

**You're all set! Start with:**
```bash
bash setup_and_run.sh all
```

Or manually:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/rank.py data/candidates.jsonl data/job_description.txt output/submission.csv
streamlit run dashboard/app.py
```
