#!/usr/bin/env python3
"""
Standalone embedding generation script.

Pre-computes candidate embeddings and saves them as a NumPy archive
for fast retrieval during ranking. This avoids re-embedding on every run.

Usage:
    python scripts/embed.py data/candidates.jsonl data/embeddings.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from redrob_ranker.config import Config
from redrob_ranker.loading import iter_candidates
from redrob_ranker.semantic_scorer import SemanticScorer
from redrob_ranker.utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute candidate embeddings")
    parser.add_argument("candidates", help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("output", help="Output path for embeddings.npz")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="Embedding model name")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for embedding")
    parser.add_argument("--device", default="cpu", help="Device for model (cpu/cuda)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of candidates (0 = all)")
    args = parser.parse_args()

    setup_logging(level=logging.INFO)

    cfg = Config(EMBEDDING_MODEL=args.model, EMBED_BATCH_SIZE=args.batch_size)
    scorer = SemanticScorer(model_name=args.model, device=args.device)

    candidates_path = Path(args.candidates)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading candidates from {candidates_path}")

    candidates = []
    ids = []
    for i, candidate in enumerate(iter_candidates(candidates_path)):
        if args.limit > 0 and i >= args.limit:
            break
        candidates.append(candidate)
        ids.append(candidate.candidate_id)
        if (i + 1) % 1000 == 0:
            logger.info(f"Loaded {i + 1} candidates...")

    logger.info(f"Total candidates loaded: {len(candidates)}")
    logger.info(f"Generating embeddings with {args.model}...")

    embeddings = scorer.embed_candidates(candidates)

    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"Saving to {output_path}")

    np.savez(
        output_path,
        embeddings=embeddings,
        candidate_ids=np.array(ids),
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
