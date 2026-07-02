# Redrob AI Hiring Intelligence Platform

> **AI-Powered Intelligent Candidate Discovery System (ICDS)**
##  dataset link access from here 

https://drive.google.com/drive/folders/11zf1uhjD0dqjIXN7UTR0cxDdXSWyJ3Si?usp=sharing

> A production-grade candidate ranking engine that understands job descriptions
> semantically, not just through keyword matching.

## Overview

The Redrob AI Hiring Intelligence Platform is a comprehensive system for ranking
candidates against job descriptions using a multi-factor hybrid approach:

- **Semantic Understanding**: BAAI/bge-small-en-v1.5 embeddings capture the
  meaning of both job descriptions and candidate profiles
- **Structural Scoring**: Rule-based analysis of skills, experience, career
  history, education, and logistics against explicit JD requirements
- **Behavioral Analysis**: Platform engagement signals (response rates,
  recruiter saves, activity levels) as quality indicators
- **Integrity Engine**: Honeypot detection and contradiction audit to identify
  seeded trap profiles
- **Explainability**: Every recommendation includes human-readable reasoning

## Architecture

```
Job Description
       |
       v
[Requirement Understanding]
       |
       v
[Candidate Parsing]  <-- streaming JSONL (memory-efficient)
       |
       v
[Feature Engineering]
       |
       v
[Semantic Embeddings]  <-- BAAI/bge-small-en-v1.5 (384-dim, CPU)
       |
       v
[Hybrid Retrieval]
       |
       v
[Multi-factor Ranking]
  - Semantic Score (40%)
  - Structural Score (60%)
    - Title Domain (30%)
    - Career Evidence (30%)
    - Experience Band (15%)
    - Skills Trust (15%)
    - Education Tier (5%)
    - Logistics (5%)
  - Behavioral Multiplier
  - Integrity Multiplier
       |
       v
[Explainability Engine]
       |
       v
[Submission Generation]
```

## Project Structure

```
redrob_ai_platform/
├── src/
│   └── redrob_ranker/
│       ├── __init__.py           # Package init
│       ├── config.py             # Centralized configuration
│       ├── data_models.py        # Pydantic domain models
│       ├── loading.py            # Data ingestion utilities
│       ├── semantic_scorer.py    # Embedding-based similarity
│       ├── structural_scorer.py  # Rule-based profile scoring
│       ├── behavioral_scorer.py  # Platform signal analysis
│       ├── integrity_engine.py   # Honeypot detection
│       ├── reasoning_engine.py   # Explainability generation
│       ├── ranking_pipeline.py   # Main orchestration
│       ├── evaluation.py         # Metrics and analysis
│       ├── embedding_engine.py   # Lightweight TF-IDF fallback
│       └── utils.py              # Shared utilities
├── scripts/
│   ├── embed.py                  # Pre-compute embeddings
│   ├── rank.py                   # Run ranking pipeline
│   ├── validate_submission.py    # Validate output CSV
│   └── run_pipeline.py           # End-to-end runner
├── tests/
│   ├── test_structural.py        # Structural scorer tests
│   ├── test_behavioral.py        # Behavioral scorer tests
│   ├── test_integrity.py         # Integrity engine tests
│   └── test_pipeline.py          # Pipeline integration tests
├── config/
│   └── default.yaml              # Default configuration
├── data/                         # Input data (not in repo)
├── output/                       # Generated submissions
├── notebooks/                    # Analysis notebooks
├── api/                          # FastAPI endpoints
├── dashboard/                    # Streamlit dashboard
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Development dependencies
├── pyproject.toml                # Project metadata
├── Makefile                      # Build automation
└── README.md                     # This file
```

## Installation

### Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd redrob_ai_platform

# Install dependencies
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

### Using Makefile

```bash
make install        # Production dependencies
make install-dev    # Development dependencies
make test           # Run tests
make lint           # Run linters
make format         # Auto-format code
```

## Usage

### 1. Generate Embeddings (Optional but Recommended)

Pre-compute candidate embeddings to speed up subsequent ranking runs:

```bash
python scripts/embed.py data/candidates.jsonl data/embeddings.npz
```

### 2. Run Ranking Pipeline

```bash
python scripts/rank.py data/candidates.jsonl data/job_description.txt output/submission.csv
```

Options:
- `--top-k`: Number of top candidates (default: 100)
- `--batch-size`: Processing batch size (default: 512)
- `--device`: Device for model (default: cpu)
- `--model`: Embedding model name (default: BAAI/bge-small-en-v1.5)

### 3. Validate Submission

```bash
python scripts/validate_submission.py output/submission.csv
```

### 4. Run Complete Pipeline

```bash
python scripts/run_pipeline.py \
    --candidates data/candidates.jsonl \
    --jd data/job_description.txt \
    --output output/submission.csv \
    --embeddings-cache data/embeddings.npz
```

## Configuration

All tunable parameters are centralized in `src/redrob_ranker/config.py`. Key
settings include:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `W_SEMANTIC` | 0.40 | Weight for semantic similarity |
| `W_STRUCTURAL` | 0.60 | Weight for structural score |
| `EMBEDDING_MODEL` | BAAI/bge-small-en-v1.5 | Embedding model |
| `EMBED_BATCH_SIZE` | 256 | Batch size for embedding |
| `EXP_IDEAL_LO` | 5.0 | Ideal experience lower bound |
| `EXP_IDEAL_HI` | 9.0 | Ideal experience upper bound |
| `BEHAVIORAL_FLOOR` | 0.30 | Minimum behavioral multiplier |
| `BEHAVIORAL_CEILING` | 1.15 | Maximum behavioral multiplier |

Override via environment variables:

```bash
export REDRob_W_SEMANTIC=0.35
export REDRob_W_STRUCTURAL=0.65
export REDRob_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

## Key Design Decisions

### Why BAAI/bge-small-en-v1.5?

- **384 dimensions**: CPU-friendly, fast inference
- **Strong retrieval performance**: Top-tier on BEIR benchmarks
- **No API calls**: Runs entirely offline, no rate limits
- **Small footprint**: ~130MB model size

### Why 40/60 Semantic/Structural Split?

The JD explicitly warns about keyword-stuffing honeypots. A 60% structural
weight ensures that explicit rules (title domain, career evidence, experience
band) dominate over semantic similarity, which can be gamed by stuffing
relevant keywords into summaries.

### Why Anti-Keyword-Stuffing in Skill Trust?

Skills are weighted by:
- **Proficiency level**: Self-reported, least trustworthy
- **Duration months**: Harder to fake, indicates actual usage
- **Endorsements**: Social proof, harder to game
- **Platform assessments**: Independent verification

This makes it expensive for honeypot profiles to achieve high skill trust
scores without genuine experience.

### Why Streaming Processing?

The candidate pool is ~100K records (~465MB). Loading all into RAM at once
would consume ~2-3GB. Streaming processes candidates in batches of 512,
keeping peak memory under 500MB.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/redrob_ranker --cov-report=html

# Run specific test file
pytest tests/test_structural.py -v
```

## Scoring Pipeline

### Semantic Score

1. Build candidate document from headline, summary, career descriptions
2. Embed with BAAI/bge-small-en-v1.5
3. Compute cosine similarity with JD embedding
4. Sigmoid-normalize to [0, 1]

### Structural Score

1. **Title Domain**: Match current title against ML/IR domain lexicon
2. **Career Evidence**: Count retrieval/ML/production terms in career history
3. **Experience Band**: Score against 5-9 year ideal range
4. **Skills Trust**: Weighted by proficiency, duration, endorsements, assessments
5. **Education Tier**: Tier-1/2/3/4 scoring
6. **Logistics**: Location, notice period, work mode, salary compatibility

Then apply penalties:
- Consulting-only career
- Research-only without production
- Title-chaser (frequent short stints)
- CV/speech/robotics without NLP/IR
- LangChain-only without pre-LLM history
- Stale hands-on (18+ months in leadership)
- Keyword stuffer (non-tech title + AI skills)
- Abroad without relocation

### Behavioral Multiplier

- Inactivity decay (90-day half-life)
- Recruiter saves bonus
- Profile views bonus
- Applications submitted bonus
- Response rate adjustment
- Interview completion rate adjustment

### Integrity Multiplier

- **Fatal** (2+ hard flags): 0.02x
- **Hard** (1 hard flag): 0.10x
- **Soft** (soft flags): 0.90^n

Hard flags: impossible dates, duration mismatches, expert skills with zero
duration, YOE span mismatches.

Soft flags: skill stuffing, semantic/structural contradictions.

## Explainability

Every ranked candidate includes a reasoning string:

```
Strong match for the Senior AI Engineer role. current title 'Machine Learning
Engineer' is squarely in the JD's domain; career history describes
retrieval/search/ranking work; describes shipping to production, not just
prototyping. Behavioral signals: 5 days since last activity (decay factor
0.962); saved by 3 recruiter(s) in last 30 days (+0.030). Score breakdown:
semantic=0.823, structural=0.891, behavioral=1.012, integrity=1.000
```

## Performance

| Metric | Value |
|--------|-------|
| Candidates processed | ~100,000 |
| Processing time | ~8-12 minutes (CPU) |
| Peak memory | ~500 MB |
| Embedding model size | ~130 MB |
| Embedding dimension | 384 |
| Batch size | 512 |

## License

MIT License - See LICENSE file for details.

## Team

Redrob AI Engineering Team
