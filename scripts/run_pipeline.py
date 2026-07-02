#!/usr/bin/env python3
"""
End-to-end pipeline runner.

Runs the complete pipeline: embed -> rank -> validate in one command.

Usage:
    python scripts/run_pipeline.py \
        --candidates data/candidates.jsonl \
        --jd data/job_description.txt \
        --output output/submission.csv \
        --embeddings-cache output/embeddings.npz
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete Redrob AI pipeline")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--jd", required=True, help="Path to job_description.txt")
    parser.add_argument("--output", required=True, help="Output submission.csv path")
    parser.add_argument("--embeddings-cache", help="Path to cache embeddings")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding generation")
    parser.add_argument("--skip-validate", action="store_true", help="Skip validation")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    scripts_dir = Path(__file__).parent
    src_dir = scripts_dir.parent / "src"
    sys.path.insert(0, str(src_dir))

    # Step 1: Generate embeddings (if needed)
    if not args.skip_embed and args.embeddings_cache:
        cache_path = Path(args.embeddings_cache)
        if not cache_path.exists():
            logger.info("Step 1: Generating embeddings...")
            cmd = [
                sys.executable,
                str(scripts_dir / "embed.py"),
                args.candidates,
                str(cache_path),
                "--model", args.model,
                "--batch-size", str(args.batch_size),
                "--device", args.device,
            ]
            result = subprocess.run(cmd, check=True)
            if result.returncode != 0:
                logger.error("Embedding generation failed")
                sys.exit(1)
        else:
            logger.info(f"Using cached embeddings: {cache_path}")

    # Step 2: Rank candidates
    logger.info("Step 2: Ranking candidates...")
    cmd = [
        sys.executable,
        str(scripts_dir / "rank.py"),
        args.candidates,
        args.jd,
        args.output,
        "--top-k", str(args.top_k),
        "--batch-size", str(args.batch_size),
        "--device", args.device,
        "--model", args.model,
    ]
    subprocess.run(cmd, check=True)

    # Step 3: Validate submission
    if not args.skip_validate:
        logger.info("Step 3: Validating submission...")
        cmd = [
            sys.executable,
            str(scripts_dir / "validate_submission.py"),
            args.output,
        ]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logger.warning("Validation found issues in submission")

    logger.info(f"Pipeline complete. Submission: {args.output}")


if __name__ == "__main__":
    main()
