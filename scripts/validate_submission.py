#!/usr/bin/env python3
"""
Submission validation script.

Validates that a submission CSV meets all hackathon requirements:
- Correct columns (rank, candidate_id, score, reasoning)
- Exactly 100 rows
- Rank is 1-100
- Score is in [0, 1]
- No duplicate candidate_ids
- No duplicate ranks
- Candidate IDs match expected format

Usage:
    python scripts/validate_submission.py output/submission.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_submission(path: str | Path) -> dict:
    """Validate a submission CSV file.

    Returns a dict with validation results.
    """
    path = Path(path)
    errors = []
    warnings = []

    if not path.exists():
        return {"valid": False, "errors": [f"File not found: {path}"], "warnings": []}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Check required columns
        required = {"rank", "candidate_id", "score"}
        missing = required - set(fieldnames)
        if missing:
            errors.append(f"Missing required columns: {missing}")

        rows = list(reader)

    # Check row count
    if len(rows) != 100:
        errors.append(f"Expected exactly 100 rows, got {len(rows)}")

    # Validate each row
    seen_ids = set()
    seen_ranks = set()
    id_pattern = re.compile(r"^CAND_\d{7}$")

    for i, row in enumerate(rows, 1):
        row_num = i + 1  # +1 for header

        # Rank
        try:
            rank = int(row.get("rank", ""))
            if rank < 1 or rank > 100:
                errors.append(f"Row {row_num}: rank {rank} out of range [1, 100]")
            if rank in seen_ranks:
                errors.append(f"Row {row_num}: duplicate rank {rank}")
            seen_ranks.add(rank)
        except (ValueError, TypeError):
            errors.append(f"Row {row_num}: invalid rank '{row.get('rank')}'")

        # Candidate ID
        cid = row.get("candidate_id", "")
        if not id_pattern.match(cid):
            errors.append(f"Row {row_num}: invalid candidate_id format '{cid}'")
        if cid in seen_ids:
            errors.append(f"Row {row_num}: duplicate candidate_id '{cid}'")
        seen_ids.add(cid)

        # Score
        try:
            score = float(row.get("score", ""))
            if score < 0 or score > 1:
                errors.append(f"Row {row_num}: score {score} out of range [0, 1]")
        except (ValueError, TypeError):
            errors.append(f"Row {row_num}: invalid score '{row.get('score')}'")

    # Check rank sequence
    expected_ranks = set(range(1, 101))
    missing_ranks = expected_ranks - seen_ranks
    if missing_ranks:
        errors.append(f"Missing ranks: {sorted(missing_ranks)}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "total_rows": len(rows),
        "unique_candidates": len(seen_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate submission CSV")
    parser.add_argument("submission", help="Path to submission.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    result = validate_submission(args.submission)

    print(f"\nValidation Result: {'PASS' if result['valid'] else 'FAIL'}")
    print(f"Total rows: {result['total_rows']}")
    print(f"Unique candidates: {result['unique_candidates']}")

    if result["errors"]:
        print(f"\nErrors ({len(result['errors'])}):")
        for err in result["errors"]:
            print(f"  - {err}")

    if result["warnings"]:
        print(f"\nWarnings ({len(result['warnings'])}):")
        for warn in result["warnings"]:
            print(f"  - {warn}")

    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
