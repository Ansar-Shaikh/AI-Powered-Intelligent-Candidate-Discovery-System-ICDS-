"""
Lightweight embedding engine using TF-IDF + SVD as a fallback.

When sentence-transformers is unavailable or too heavy for the target
environment, this module provides a pure-scikit-learn alternative that
produces reasonable semantic similarity scores without neural models.

Design decisions:
- TF-IDF on n-grams captures domain vocabulary.
- TruncatedSVD reduces dimensionality while preserving semantic structure.
- No external model downloads needed.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from redrob_ranker.config import Config
from redrob_ranker.data_models import Candidate
from redrob_ranker.loading import build_candidate_document

logger = logging.getLogger(__name__)


class LightweightEmbeddingEngine:
    """CPU-only, no-download embedding engine using TF-IDF + SVD."""

    def __init__(self, n_components: int = 128, max_features: int = 20000):
        self.n_components = n_components
        self.max_features = max_features
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self._jd_embedding: np.ndarray | None = None
        self._is_fitted = False

    def fit(self, documents: List[str]) -> None:
        """Fit TF-IDF + SVD on a corpus of documents."""
        logger.info(f"Fitting lightweight embedding engine on {len(documents)} documents...")

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words="english",
        )
        tfidf = self.vectorizer.fit_transform(documents)

        n_comp = min(self.n_components, tfidf.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd.fit(tfidf)
        self._is_fitted = True
        logger.info(f"Lightweight engine fitted. Components: {n_comp}")

    def transform(self, documents: List[str]) -> np.ndarray:
        """Transform documents to embedding vectors."""
        if not self._is_fitted:
            raise RuntimeError("Engine not fitted. Call fit() first.")
        tfidf = self.vectorizer.transform(documents)
        return self.svd.transform(tfidf)

    def set_jd(self, jd_text: str) -> None:
        """Cache the JD embedding."""
        if not self._is_fitted:
            raise RuntimeError("Engine not fitted.")
        self._jd_embedding = self.transform([jd_text])[0]

    def score_candidates(self, candidates: List[Candidate], cfg: Config | None = None) -> np.ndarray:
        """Compute cosine similarity scores for candidates."""
        cfg = cfg or Config()
        docs = [build_candidate_document(c, cfg) for c in candidates]
        embeddings = self.transform(docs)

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        jd_norm = np.linalg.norm(self._jd_embedding)
        if jd_norm == 0:
            return np.zeros(len(candidates))

        jd_emb = self._jd_embedding / jd_norm
        similarities = embeddings @ jd_emb
        return similarities
