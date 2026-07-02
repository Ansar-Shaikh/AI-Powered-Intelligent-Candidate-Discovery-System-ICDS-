#!/bin/bash
# ============================================================================
# Redrob AI Hiring Intelligence Platform - Complete Setup & Run Script
# ============================================================================
# This script handles everything from environment creation to dashboard launch
# Usage: bash setup_and_run.sh [command]
# Commands: setup | rank | dashboard | api | test | all
# ============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="redrob_ai_hiring"
PYTHON_VERSION="3.11"
VENV_DIR=".venv"
SRC_DIR="src"
DATA_DIR="data"
OUTPUT_DIR="output"
DASHBOARD_PORT="8501"
API_PORT="8000"

# Default paths (modify these to point to your actual data)
CANDIDATES_FILE="${DATA_DIR}/candidates.jsonl"
JD_FILE="${DATA_DIR}/job_description.txt"

print_header() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║           🎯 Redrob AI Hiring Intelligence Platform                  ║"
    echo "║              Complete Setup & Execution Guide                      ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
}

print_step() {
    echo ""
    echo -e "${BLUE}▶ $1${NC}"
    echo "───────────────────────────────────────────────────────────────────────"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# ============================================================================
# STEP 1: Check Prerequisites
# ============================================================================
check_prerequisites() {
    print_step "Step 1: Checking Prerequisites"

    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python is not installed. Please install Python ${PYTHON_VERSION}+ first."
        exit 1
    fi

    PYTHON_VER=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    print_success "Found Python ${PYTHON_VER}"

    # Check pip
    if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
        print_error "pip is not installed. Please install pip first."
        exit 1
    fi
    print_success "pip is available"

    # Check git (optional)
    if command -v git &> /dev/null; then
        print_success "git is available"
    else
        print_warning "git not found (optional, needed only for cloning)"
    fi
}

# ============================================================================
# STEP 2: Create Virtual Environment
# ============================================================================
create_venv() {
    print_step "Step 2: Creating Virtual Environment"

    if [ -d "${VENV_DIR}" ]; then
        print_warning "Virtual environment already exists at ${VENV_DIR}"
        read -p "Recreate? (y/N): " recreate
        if [[ $recreate =~ ^[Yy]$ ]]; then
            rm -rf "${VENV_DIR}"
            $PYTHON_CMD -m venv "${VENV_DIR}"
            print_success "Virtual environment recreated"
        else
            print_success "Using existing virtual environment"
        fi
    else
        $PYTHON_CMD -m venv "${VENV_DIR}"
        print_success "Virtual environment created at ${VENV_DIR}"
    fi
}

# ============================================================================
# STEP 3: Activate Virtual Environment
# ============================================================================
activate_venv() {
    print_step "Step 3: Activating Virtual Environment"

    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        # Windows
        source "${VENV_DIR}/Scripts/activate"
    else
        # Linux/Mac
        source "${VENV_DIR}/bin/activate"
    fi

    print_success "Virtual environment activated"
    print_success "Python path: $(which python)"
    print_success "Python version: $(python --version)"
}

# ============================================================================
# STEP 4: Upgrade pip & Install Dependencies
# ============================================================================
install_dependencies() {
    print_step "Step 4: Installing Dependencies"

    python -m pip install --upgrade pip setuptools wheel
    print_success "pip upgraded"

    # Install core dependencies first (heavy ones)
    print_step "Installing heavy dependencies (this may take 5-10 minutes)..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    print_success "PyTorch installed"

    pip install sentence-transformers
    print_success "sentence-transformers installed"

    # Install remaining dependencies
    pip install -r requirements.txt
    print_success "All dependencies installed"

    # Install in development mode
    pip install -e .
    print_success "Package installed in editable mode"
}

# ============================================================================
# STEP 5: Verify Installation
# ============================================================================
verify_installation() {
    print_step "Step 5: Verifying Installation"

    python -c "import redrob_ranker; print(f'redrob_ranker version: {redrob_ranker.__version__}')"
    print_success "Core package imports successfully"

    python -c "from redrob_ranker.config import Config; c = Config(); print(f'Config loaded: {c.EMBEDDING_MODEL}')"
    print_success "Configuration system works"

    python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers OK')"
    print_success "Embedding model library available"

    python -c "import numpy, pandas, sklearn; print('Core ML libraries OK')"
    print_success "ML libraries available"
}

# ============================================================================
# STEP 6: Setup Data Directory
# ============================================================================
setup_data() {
    print_step "Step 6: Setting Up Data Directory"

    mkdir -p "${DATA_DIR}"
    mkdir -p "${OUTPUT_DIR}"

    print_success "Created directories: ${DATA_DIR}, ${OUTPUT_DIR}"

    # Check if candidates file exists
    if [ ! -f "${CANDIDATES_FILE}" ]; then
        print_warning "Candidates file not found at ${CANDIDATES_FILE}"
        echo ""
        echo "Please place your candidates.jsonl file in the ${DATA_DIR}/ directory."
        echo "Expected path: ${CANDIDATES_FILE}"
        echo ""
        echo "The file should contain one JSON object per line (JSONL format):"
        echo '  {"candidate_id": "CAND_0000001", "profile": {...}, ...}'
        echo ""
    else
        print_success "Found candidates file: ${CANDIDATES_FILE}"
        LINES=$(wc -l < "${CANDIDATES_FILE}")
        print_success "Candidates file has ${LINES} records"
    fi

    # Check if job description exists
    if [ ! -f "${JD_FILE}" ]; then
        print_warning "Job description not found at ${JD_FILE}"
        echo "A default job description has been created. You can edit it."
    else
        print_success "Found job description: ${JD_FILE}"
    fi
}

# ============================================================================
# STEP 7: Run Tests
# ============================================================================
run_tests() {
    print_step "Step 7: Running Tests"

    if command -v pytest &> /dev/null; then
        pytest tests/ -v --tb=short
        print_success "All tests passed"
    else
        print_warning "pytest not installed, skipping tests"
        print_warning "Install with: pip install pytest pytest-cov"
    fi
}

# ============================================================================
# STEP 8: Run Ranking Pipeline
# ============================================================================
run_ranking() {
    print_step "Step 8: Running Ranking Pipeline"

    if [ ! -f "${CANDIDATES_FILE}" ]; then
        print_error "Candidates file not found. Please place it at ${CANDIDATES_FILE}"
        exit 1
    fi

    if [ ! -f "${JD_FILE}" ]; then
        print_error "Job description not found. Please place it at ${JD_FILE}"
        exit 1
    fi

    echo ""
    echo "Configuration:"
    echo "  Candidates: ${CANDIDATES_FILE}"
    echo "  Job Description: ${JD_FILE}"
    echo "  Output: ${OUTPUT_DIR}/submission.csv"
    echo ""

    python scripts/rank.py         "${CANDIDATES_FILE}"         "${JD_FILE}"         "${OUTPUT_DIR}/submission.csv"         --top-k 100         --batch-size 512         --device cpu

    print_success "Ranking complete!"
    print_success "Submission saved to: ${OUTPUT_DIR}/submission.csv"

    # Validate submission
    echo ""
    print_step "Validating Submission"
    python scripts/validate_submission.py "${OUTPUT_DIR}/submission.csv"
}

# ============================================================================
# STEP 9: Launch Dashboard
# ============================================================================
launch_dashboard() {
    print_step "Step 9: Launching Streamlit Dashboard"

    if ! command -v streamlit &> /dev/null; then
        print_warning "streamlit not installed. Installing..."
        pip install streamlit plotly
    fi

    print_success "Starting dashboard on http://localhost:${DASHBOARD_PORT}"
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  🚀 Dashboard is starting...                                         ║${NC}"
    echo -e "${GREEN}║  Open your browser to: http://localhost:${DASHBOARD_PORT}                    ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    streamlit run dashboard/app.py         --server.port ${DASHBOARD_PORT}         --server.address localhost         --browser.serverAddress localhost
}

# ============================================================================
# STEP 10: Launch API Server
# ============================================================================
launch_api() {
    print_step "Step 10: Launching FastAPI Server"

    if ! command -v uvicorn &> /dev/null; then
        print_warning "uvicorn not installed. Installing..."
        pip install uvicorn fastapi
    fi

    print_success "Starting API server on http://localhost:${API_PORT}"
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  🚀 API server is starting...                                        ║${NC}"
    echo -e "${GREEN}║  Open your browser to: http://localhost:${API_PORT}/docs                   ║${NC}"
    echo -e "${GREEN}║  Interactive docs: http://localhost:${API_PORT}/docs                       ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    uvicorn api.main:app         --host localhost         --port ${API_PORT}         --reload
}

# ============================================================================
# FULL SETUP (Steps 1-7)
# ============================================================================
full_setup() {
    print_header
    check_prerequisites
    create_venv
    activate_venv
    install_dependencies
    verify_installation
    setup_data
    run_tests

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║              ✅ SETUP COMPLETE                                       ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Next steps:"
    echo "  1. Place your candidates.jsonl in: ${DATA_DIR}/"
    echo "  2. Edit job description if needed: ${JD_FILE}"
    echo "  3. Run ranking:    bash setup_and_run.sh rank"
    echo "  4. Launch dashboard: bash setup_and_run.sh dashboard"
    echo ""
}

# ============================================================================
# MAIN COMMAND HANDLER
# ============================================================================
case "${1:-setup}" in
    setup|install)
        full_setup
        ;;
    rank|pipeline)
        activate_venv
        run_ranking
        ;;
    dashboard|ui)
        activate_venv
        launch_dashboard
        ;;
    api|server)
        activate_venv
        launch_api
        ;;
    test)
        activate_venv
        run_tests
        ;;
    all|full)
        full_setup
        run_ranking
        launch_dashboard
        ;;
    help|--help|-h)
        print_header
        echo "Usage: bash setup_and_run.sh [command]"
        echo ""
        echo "Commands:"
        echo "  setup      - Full environment setup (default)"
        echo "  rank       - Run ranking pipeline only"
        echo "  dashboard  - Launch Streamlit dashboard"
        echo "  api        - Launch FastAPI server"
        echo "  test       - Run unit tests"
        echo "  all        - Setup + Rank + Dashboard"
        echo "  help       - Show this help"
        echo ""
        echo "Examples:"
        echo "  bash setup_and_run.sh setup      # First time setup"
        echo "  bash setup_and_run.sh rank       # Just run ranking"
        echo "  bash setup_and_run.sh dashboard  # Just launch UI"
        echo "  bash setup_and_run.sh all        # Everything"
        echo ""
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Run 'bash setup_and_run.sh help' for usage"
        exit 1
        ;;
esac
