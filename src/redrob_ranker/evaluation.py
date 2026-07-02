"""
Evaluation utilities for the ranking pipeline.

Includes:
- NDCG@K computation for ranking quality.
- Precision@K and Recall@K for retrieval quality.
- Contradiction analysis for keyword-stuffing detection.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from redrob_ranker.config import Config
from redrob_ranker.data_models import Candidate, FinalScore

logger = logging.getLogger(__name__)


def ndcg_at_k(relevances: np.ndarray, k: int = 10) -> float:
    """Compute NDCG@K given relevance scores in predicted order.

    Args:
        relevances: Array of relevance scores in the predicted ranking order.
        k: Cutoff rank.

    Returns:
        NDCG@K score in [0, 1].
    """
    if len(relevances) == 0:
        return 0.0

    relevances = np.asarray(relevances)[:k]

    # DCG
    dcg = relevances[0] + np.sum(relevances[1:] / np.log2(np.arange(2, len(relevances) + 1)))

    # Ideal DCG
    ideal = np.sort(relevances)[::-1]
    idcg = ideal[0] + np.sum(ideal[1:] / np.log2(np.arange(2, len(ideal) + 1)))

    if idcg == 0:
        return 0.0

    return float(dcg / idcg)


def precision_at_k(predicted: List[str], relevant: set[str], k: int = 10) -> float:
    """Compute Precision@K."""
    if k == 0:
        return 0.0
    predicted_k = predicted[:k]
    hits = sum(1 for p in predicted_k if p in relevant)
    return hits / k


def recall_at_k(predicted: List[str], relevant: set[str], k: int = 10) -> float:
    """Compute Recall@K."""
    if not relevant:
        return 0.0
    predicted_k = predicted[:k]
    hits = sum(1 for p in predicted_k if p in relevant)
    return hits / len(relevant)


def analyze_contradictions(
    scores: List[FinalScore],
    candidates: Dict[str, Candidate],
    cfg: Config | None = None,
) -> Dict[str, any]:
    """Analyze score contradictions to detect keyword stuffing patterns.

    Returns statistics on candidates where structural >> semantic.
    """
    cfg = cfg or Config()
    contradictions = []

    for s in scores:
        delta = abs(s.structural - s.semantic)
        if delta > cfg.CONTRADICTION_DELTA_THRESHOLD:
            contradictions.append({
                "candidate_id": s.candidate_id,
                "structural": s.structural,
                "semantic": s.semantic,
                "delta": delta,
                "type": "structural_high" if s.structural > s.semantic else "semantic_high",
            })

    return {
        "total_analyzed": len(scores),
        "contradictions_found": len(contradictions),
        "contradiction_rate": len(contradictions) / len(scores) if scores else 0,
        "details": contradictions[:20],  # Top 20 for inspection
    }


def compute_ranking_metrics(
    predicted_ids: List[str],
    ground_truth_relevance: Dict[str, float],
    k_values: List[int] = [10, 50, 100],
) -> Dict[str, float]:
    """Compute comprehensive ranking metrics."""
    metrics = {}

    # Extract relevances in predicted order
    relevances = np.array([ground_truth_relevance.get(cid, 0.0) for cid in predicted_ids])

    for k in k_values:
        metrics[f"ndcg@{k}"] = ndcg_at_k(relevances, k)
        relevant_set = {cid for cid, rel in ground_truth_relevance.items() if rel > 0.5}
        metrics[f"precision@{k}"] = precision_at_k(predicted_ids, relevant_set, k)
        metrics[f"recall@{k}"] = recall_at_k(predicted_ids, relevant_set, k)

    return metrics


def generate_evaluation_report(
    scores: List[FinalScore],
    candidates: Dict[str, Candidate],
    output_path: str | None = None,
) -> str:
    """Generate a human-readable evaluation report."""
    lines = [
        "=" * 60,
        "RANKING EVALUATION REPORT",
        "=" * 60,
        f"Total candidates ranked: {len(scores)}",
        "",
        "Score Distribution:",
    ]

    final_scores = [s.final_score for s in scores]
    lines.append(f"  Mean: {np.mean(final_scores):.4f}")
    lines.append(f"  Median: {np.median(final_scores):.4f}")
    lines.append(f"  Std: {np.std(final_scores):.4f}")
    lines.append(f"  Min: {np.min(final_scores):.4f}")
    lines.append(f"  Max: {np.max(final_scores):.4f}")

    lines.append("")
    lines.append("Component Score Averages:")
    lines.append(f"  Semantic: {np.mean([s.semantic for s in scores]):.4f}")
    lines.append(f"  Structural: {np.mean([s.structural for s in scores]):.4f}")
    lines.append(f"  Behavioral Mult: {np.mean([s.behavioral_multiplier for s in scores]):.4f}")
    lines.append(f"  Integrity Mult: {np.mean([s.integrity_multiplier for s in scores]):.4f}")

    lines.append("")
    lines.append("Top 10 Candidates:")
    for s in scores[:10]:
        lines.append(f"  #{s.rank} {s.candidate_id}: {s.final_score:.4f}")

    lines.append("")
    lines.append("Bottom 10 Candidates:")
    for s in scores[-10:]:
        lines.append(f"  #{s.rank} {s.candidate_id}: {s.final_score:.4f}")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

    return report
