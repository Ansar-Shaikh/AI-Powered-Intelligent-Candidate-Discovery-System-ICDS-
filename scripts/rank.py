#!/usr/bin/env python3
"""
Standalone ranking script.

Reads candidates, scores them using the full pipeline, and writes
a submission CSV with ranked results and explanations.

Usage:
    python scripts/rank.py data/candidates.jsonl data/job_description.txt output/submission.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from redrob_ranker.config import Config
from redrob_ranker.ranking_pipeline import RankingPipeline, write_submission, write_detailed_report
from redrob_ranker.loading import iter_candidates
from redrob_ranker.utils import setup_logging, format_duration
import time

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidates for a job description")
    parser.add_argument("candidates", help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("jd", help="Path to job_description.txt")
    parser.add_argument("output", help="Output path for submission.csv")
    parser.add_argument("--top-k", type=int, default=100, help="Number of top candidates to return")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for processing")
    parser.add_argument("--device", default="cpu", help="Device for embedding model")
    parser.add_argument("--detailed-report", help="Path for detailed JSON report")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="Embedding model name")
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    start_time = time.time()

    cfg = Config(EMBEDDING_MODEL=args.model)
    pipeline = RankingPipeline(cfg=cfg, device=args.device)
    pipeline.set_job_description(args.jd)

    logger.info(f"Ranking candidates from {args.candidates}")
    logger.info(f"Job description: {args.jd}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Top-K: {args.top_k}")

    scores = pipeline.rank_candidates(
        candidates_path=args.candidates,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )

    # Write submission
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_submission(scores, output_path, include_reasoning=True)

    # Write detailed report if requested
    if args.detailed_report:
        # Load candidates for detailed report
        candidates = {c.candidate_id: c for c in iter_candidates(args.candidates)}
        write_detailed_report(scores, candidates, args.detailed_report)

    elapsed = time.time() - start_time
    logger.info(f"Ranking complete in {format_duration(elapsed)}")
    logger.info(f"Submission written to {output_path}")


if __name__ == "__main__":
    main()
