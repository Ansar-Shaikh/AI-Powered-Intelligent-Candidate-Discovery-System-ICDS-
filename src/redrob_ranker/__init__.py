"""
Redrob AI Hiring Intelligence Platform.

A production-grade candidate discovery and ranking system that uses
semantic understanding, multi-factor scoring, and explainable AI
to identify the best candidates for a given job description.

Modules:
    config: Centralized configuration
    data_models: Pydantic domain models
    loading: Data ingestion utilities
    semantic_scorer: Embedding-based similarity scoring
    structural_scorer: Rule-based profile scoring
    behavioral_scorer: Platform signal analysis
    integrity_engine: Honeypot detection and contradiction audit
    reasoning_engine: Explainable recommendation generation
    ranking_pipeline: Main orchestration pipeline
    evaluation: Ranking quality metrics
    utils: Shared utilities
"""

__version__ = "1.0.0"
__author__ = "Redrob AI Engineering Team"

from redrob_ranker.config import Config
from redrob_ranker.data_models import (
    Candidate,
    JobDescription,
    FinalScore,
    SemanticScore,
    StructuralScore,
    BehavioralScore,
    IntegrityReport,
)
from redrob_ranker.ranking_pipeline import RankingPipeline

__all__ = [
    "Config",
    "Candidate",
    "JobDescription",
    "FinalScore",
    "SemanticScore",
    "StructuralScore",
    "BehavioralScore",
    "IntegrityReport",
    "RankingPipeline",
]
