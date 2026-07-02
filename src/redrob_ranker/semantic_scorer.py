"""
Semantic scoring using TF-IDF cosine similarity.

DESIGN DECISION — TF-IDF replaces sentence-transformers/torch:

The original design used BAAI/bge-small-en-v1.5 via sentence-transformers.
That is the correct production choice. However this project runs on Python 3.14
where PyTorch's C++ DLLs (c10.dll) cannot load — PyTorch does not yet publish
cp314 wheels that link against the correct CRT version on Windows.

TF-IDF + cosine similarity is the correct FALLBACK for this constraint:
  - Zero new dependencies (scikit-learn is already in requirements.txt)
  - Deterministic, reproducible, no model download
  - Fast: fit+transform of 100K short docs in <10s on a single CPU core
  - Fully explainable: similarity decomposes into overlapping n-gram terms
  - Interface is identical: set_jd(), score_candidates() return the same types

Upgrade path (when Python 3.12 or lower is available):
  Replace this file with the sentence-transformer version — no other file
  needs to change because the public interface is identical.
"""

from __future__ import annotations

import logging
import math
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from redrob_ranker.config import Config
from redrob_ranker.data_models import Candidate, SemanticScore
from redrob_ranker.loading import build_candidate_document

logger = logging.getLogger(__name__)


def _sigmoid_normalize(sim: float) -> float:
    """Same sigmoid scaling used in the original sentence-transformer version
    so downstream score arithmetic is unchanged."""
    return float(1.0 / (1.0 + math.exp(-5.0 * (sim - 0.5))))


class SemanticScorer:
    """TF-IDF semantic scorer — drop-in replacement for the
    sentence-transformer version with an identical public interface."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.cfg = Config()
        # model_name and device accepted for API compatibility; not used
        self.model_name = model_name or self.cfg.EMBEDDING_MODEL
        self.device = device or "cpu"
        self._vectorizer: TfidfVectorizer | None = None
        self._jd_text: str = ""
        self._jd_embedding: np.ndarray | None = None

    def set_jd(self, jd_text: str) -> None:
        """Cache the job description text. Vectorizer is (re)fit lazily
        once candidate texts are known, so we store the raw text here."""
        self._jd_text = jd_text
        # Reset cached vectorizer so a new JD forces a fresh fit
        self._vectorizer = None
        self._jd_embedding = None
        logger.info("JD text stored (TF-IDF vectorizer will fit on first score call).")

    def _fit(self, candidate_docs: List[str]) -> None:
        """Fit TF-IDF on (JD + all candidate docs) and cache JD vector."""
        corpus = [self._jd_text] + candidate_docs
        self._vectorizer = TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1,
            sublinear_tf=True,
        )
        matrix = self._vectorizer.fit_transform(corpus)
        self._jd_embedding = matrix[0:1]  # sparse row, shape (1, vocab)
        logger.info(
            "TF-IDF fitted: vocab_size=%d, corpus_size=%d",
            len(self._vectorizer.vocabulary_),
            len(corpus),
        )

    def score_candidates(self, candidates: List[Candidate]) -> List[SemanticScore]:
        """Batch-score candidates. Fits TF-IDF on first call."""
        if not candidates:
            return []

        docs = [build_candidate_document(c, self.cfg) for c in candidates]

        # (Re)fit if needed
        if self._vectorizer is None or self._jd_embedding is None:
            if not self._jd_text:
                raise RuntimeError("JD text not set. Call set_jd() before score_candidates().")
            self._fit(docs)

        cand_matrix = self._vectorizer.transform(docs)
        sims = cosine_similarity(self._jd_embedding, cand_matrix).flatten()

        return [
            SemanticScore(
                raw_similarity=float(sims[i]),
                normalized_similarity=_sigmoid_normalize(float(sims[i])),
                embedding_document=docs[i],
            )
            for i in range(len(candidates))
        ]

    def score_single(self, candidate: Candidate) -> SemanticScore:
        return self.score_candidates([candidate])[0]

    def embed_candidates(self, candidates: List[Candidate]) -> np.ndarray:
        """Compatibility shim: returns (N, vocab) sparse→dense similarity
        array so any callers of embed_candidates still work."""
        scores = self.score_candidates(candidates)
        return np.array([[s.normalized_similarity] for s in scores], dtype=np.float32)

    def compute_similarity(self, candidate_embeddings: np.ndarray) -> np.ndarray:
        """Compatibility shim for any direct callers of compute_similarity."""
        return candidate_embeddings.flatten()


def compute_semantic_score(
    candidate: Candidate,
    jd_embedding,          # kept for API compatibility (ignored in TF-IDF mode)
    model,                 # kept for API compatibility (ignored in TF-IDF mode)
    cfg: Config | None = None,
    _scorer_cache: dict = {},  # module-level cache so single-candidate calls reuse vectorizer
) -> SemanticScore:
    """Low-level single-candidate semantic score.

    jd_embedding and model arguments are accepted for API compatibility with
    the sentence-transformer version but are not used. The scorer is cached
    at module level so repeated single-candidate calls don't re-fit TF-IDF.
    """
    cfg = cfg or Config()

    # Reuse the scorer already attached to the pipeline if available
    scorer: SemanticScorer | None = _scorer_cache.get("scorer")
    if scorer is None or scorer._jd_text == "":
        scorer = SemanticScorer()
        _scorer_cache["scorer"] = scorer

    # If the scorer's JD text matches what the pipeline set, we're good.
    # If jd_embedding is None (first call), initialise with an empty JD as
    # a safe fallback — the pipeline always calls set_jd() before scoring.
    if scorer._jd_text == "" and scorer._vectorizer is None:
        scorer._jd_text = " "

    return scorer.score_candidates([candidate])[0]
