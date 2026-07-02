"""
Main ranking pipeline: orchestrates all scoring modules into a unified ranking.

Design decisions:
- Lazy loading of the embedding model (only loaded when semantic scoring is needed).
- Streaming processing for memory efficiency (process candidates in batches).
- Deterministic tie-breaking: candidate_id ascending on equal scores.
- Full audit trail: every score component is preserved for explainability.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from redrob_ranker.behavioral_scorer import compute_behavioral_score
from redrob_ranker.config import Config
from redrob_ranker.data_models import Candidate, FinalScore
from redrob_ranker.integrity_engine import compute_integrity_report
from redrob_ranker.loading import build_candidate_document, iter_candidates, load_job_description
from redrob_ranker.reasoning_engine import generate_reasoning
from redrob_ranker.semantic_scorer import SemanticScorer, compute_semantic_score
from redrob_ranker.structural_scorer import compute_structural_score

logger = logging.getLogger(__name__)


class RankingPipeline:
    """Production-quality ranking pipeline for the Redrob AI Hiring Intelligence Platform."""

    def __init__(self, cfg: Config | None = None, device: str = "cpu"):
        self.cfg = cfg or Config()
        self.device = device
        self._semantic_scorer: SemanticScorer | None = None
        self._jd_text: str = ""

    @property
    def semantic_scorer(self) -> SemanticScorer:
        if self._semantic_scorer is None:
            self._semantic_scorer = SemanticScorer(
                model_name=self.cfg.EMBEDDING_MODEL,
                device=self.device,
            )
        return self._semantic_scorer

    def set_job_description(self, jd_path: str | Path) -> None:
        """Load and cache the job description embedding."""
        jd = load_job_description(jd_path)
        self._jd_text = jd.get_embedding_text()
        self.semantic_scorer.set_jd(self._jd_text)
        logger.info(f"Job description loaded from {jd_path}")

    def set_jd_text(self, text: str) -> None:
        """Set job description from raw text."""
        self._jd_text = text
        self.semantic_scorer.set_jd(text)
        # Also propagate to the module-level scorer cache used by
        # compute_semantic_score() in single-candidate scoring path.
        from redrob_ranker.semantic_scorer import compute_semantic_score
        cache = compute_semantic_score.__defaults__[-1]  # _scorer_cache dict
        cache["scorer"] = self.semantic_scorer
        logger.info("Job description set from text.")

    def score_candidate(self, candidate: Candidate) -> FinalScore:
        """Score a single candidate through all scoring dimensions."""
        # Structural score (fast, no model)
        structural = compute_structural_score(candidate, self.cfg)

        # Semantic score — jd_embedding and model args accepted for
        # API compatibility but unused in TF-IDF mode.
        semantic = compute_semantic_score(
            candidate,
            jd_embedding=None,
            model=None,
            cfg=self.cfg,
        )

        # Behavioral score
        behavioral = compute_behavioral_score(candidate, self.cfg)

        # Integrity check
        integrity = compute_integrity_report(
            candidate,
            semantic.normalized_similarity,
            structural.score,
            self.cfg,
        )

        # Composite score
        base = self.cfg.W_SEMANTIC * semantic.normalized_similarity + self.cfg.W_STRUCTURAL * structural.score
        final = base * behavioral.multiplier * integrity.multiplier
        final = max(0.0, min(1.0, final))

        # Generate reasoning
        reasoning = generate_reasoning(
            candidate, semantic, structural, behavioral, integrity, final, self.cfg
        )

        return FinalScore(
            candidate_id=candidate.candidate_id,
            final_score=final,
            semantic=semantic.normalized_similarity,
            structural=structural.score,
            behavioral_multiplier=behavioral.multiplier,
            integrity_multiplier=integrity.multiplier,
            reasoning=reasoning,
        )

    def score_batch(self, candidates: List[Candidate]) -> List[FinalScore]:
        """Score a batch of candidates efficiently.

        Uses batch embedding for semantic scores, then applies structural,
        behavioral, and integrity scoring per candidate.
        """
        if not candidates:
            return []

        # Batch semantic scoring
        semantic_scores = self.semantic_scorer.score_candidates(candidates)

        results = []
        for i, candidate in enumerate(candidates):
            structural = compute_structural_score(candidate, self.cfg)
            behavioral = compute_behavioral_score(candidate, self.cfg)
            integrity = compute_integrity_report(
                candidate,
                semantic_scores[i].normalized_similarity,
                structural.score,
                self.cfg,
            )

            base = self.cfg.W_SEMANTIC * semantic_scores[i].normalized_similarity + self.cfg.W_STRUCTURAL * structural.score
            final = base * behavioral.multiplier * integrity.multiplier
            final = max(0.0, min(1.0, final))

            reasoning = generate_reasoning(
                candidate, semantic_scores[i], structural, behavioral, integrity, final, self.cfg
            )

            results.append(FinalScore(
                candidate_id=candidate.candidate_id,
                final_score=final,
                semantic=semantic_scores[i].normalized_similarity,
                structural=structural.score,
                behavioral_multiplier=behavioral.multiplier,
                integrity_multiplier=integrity.multiplier,
                reasoning=reasoning,
            ))

        return results

    def rank_candidates(
        self,
        candidates_path: str | Path,
        top_k: int = 100,
        batch_size: int = 512,
    ) -> List[FinalScore]:
        """Stream candidates from file, score in batches, and return top-K ranked.

        Memory-efficient: processes candidates in batches without loading all into RAM.
        """
        all_scores: List[FinalScore] = []
        batch: List[Candidate] = []
        total_processed = 0

        logger.info(f"Starting ranking pipeline on {candidates_path}")

        for candidate in iter_candidates(candidates_path):
            batch.append(candidate)
            if len(batch) >= batch_size:
                scores = self.score_batch(batch)
                all_scores.extend(scores)
                total_processed += len(batch)
                batch = []
                if total_processed % 5000 == 0:
                    logger.info(f"Processed {total_processed} candidates...")

        # Process remaining batch
        if batch:
            scores = self.score_batch(batch)
            all_scores.extend(scores)
            total_processed += len(batch)

        logger.info(f"Total candidates processed: {total_processed}")

        # Sort by score descending, then candidate_id ascending
        all_scores.sort(key=lambda s: s.sort_key())

        # Assign ranks
        for i, score in enumerate(all_scores[:top_k]):
            score.rank = i + 1

        return all_scores[:top_k]

    def rank_from_list(self, candidates: List[Candidate], top_k: int = 100) -> List[FinalScore]:
        """Rank from an in-memory list of candidates."""
        all_scores = self.score_batch(candidates)
        all_scores.sort(key=lambda s: s.sort_key())
        for i, score in enumerate(all_scores[:top_k]):
            score.rank = i + 1
        return all_scores[:top_k]


# ---------------------------------------------------------------------------
# Submission utilities
# ---------------------------------------------------------------------------

def write_submission(
    scores: List[FinalScore],
    output_path: str | Path,
    include_reasoning: bool = True,
) -> None:
    """Write the ranked list to a CSV submission file."""
    output_path = Path(output_path)

    fieldnames = ["rank", "candidate_id", "score"]
    if include_reasoning:
        fieldnames.append("reasoning")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in scores:
            row = {
                "rank": s.rank,
                "candidate_id": s.candidate_id,
                "score": f"{s.final_score:.6f}",
            }
            if include_reasoning:
                row["reasoning"] = s.reasoning
            writer.writerow(row)

    logger.info(f"Submission written to {output_path} ({len(scores)} candidates)")


def write_detailed_report(
    scores: List[FinalScore],
    candidates: dict[str, Candidate],
    output_path: str | Path,
) -> None:
    """Write a detailed JSON report with full score breakdowns for analysis."""
    output_path = Path(output_path)

    report = []
    for s in scores:
        entry = {
            "rank": s.rank,
            "candidate_id": s.candidate_id,
            "final_score": round(s.final_score, 6),
            "semantic": round(s.semantic, 6),
            "structural": round(s.structural, 6),
            "behavioral_multiplier": round(s.behavioral_multiplier, 4),
            "integrity_multiplier": round(s.integrity_multiplier, 4),
            "reasoning": s.reasoning,
        }
        report.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Detailed report written to {output_path}")
