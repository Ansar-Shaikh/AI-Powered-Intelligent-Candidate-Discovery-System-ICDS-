.PHONY: help install install-dev test lint format clean run-pipeline embed rank validate

PYTHON := python3
PIP := pip3

help:
	@echo "Redrob AI Hiring Intelligence Platform"
	@echo ""
	@echo "Available targets:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run unit tests"
	@echo "  lint         - Run linters (black, isort, flake8, mypy)"
	@echo "  format       - Auto-format code with black and isort"
	@echo "  clean        - Remove build artifacts"
	@echo "  embed        - Generate candidate embeddings"
	@echo "  rank         - Run ranking pipeline"
	@echo "  validate     - Validate submission CSV"
	@echo "  run-pipeline - Run complete end-to-end pipeline"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

lint:
	black --check src/ tests/ scripts/
	isort --check-only src/ tests/ scripts/
	flake8 src/ tests/ scripts/
	mypy src/

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

embed:
	$(PYTHON) scripts/embed.py data/candidates.jsonl data/embeddings.npz

rank:
	$(PYTHON) scripts/rank.py data/candidates.jsonl data/job_description.txt output/submission.csv

validate:
	$(PYTHON) scripts/validate_submission.py output/submission.csv

run-pipeline:
	$(PYTHON) scripts/run_pipeline.py \
		--candidates data/candidates.jsonl \
		--jd data/job_description.txt \
		--output output/submission.csv \
		--embeddings-cache data/embeddings.npz
