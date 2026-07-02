"""
FastAPI backend for the Redrob AI Hiring Intelligence Platform.

Provides REST endpoints for:
- Uploading job descriptions
- Uploading candidate datasets
- Running ranking pipelines
- Retrieving ranked results
- Getting candidate explanations
"""

from __future__ import annotations

import io
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from redrob_ranker.config import Config
from redrob_ranker.data_models import Candidate, FinalScore
from redrob_ranker.loading import load_candidates_blob, load_job_description
from redrob_ranker.ranking_pipeline import RankingPipeline, write_submission

logger = logging.getLogger(__name__)

# Global pipeline instance (singleton)
_pipeline: RankingPipeline | None = None


class RankRequest(BaseModel):
    top_k: int = 100
    batch_size: int = 512


class RankResponse(BaseModel):
    status: str
    total_candidates: int
    top_k: int
    results: List[dict]


class CandidateDetail(BaseModel):
    candidate_id: str
    final_score: float
    semantic: float
    structural: float
    behavioral_multiplier: float
    integrity_multiplier: float
    reasoning: str
    rank: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Redrob AI Hiring Intelligence API...")
    global _pipeline
    _pipeline = RankingPipeline(cfg=Config())
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Redrob AI Hiring Intelligence Platform",
    description="AI-powered candidate discovery and ranking API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/rank", response_model=RankResponse)
async def rank_candidates(
    candidates_file: UploadFile = File(...),
    jd_file: UploadFile = File(...),
    request: RankRequest = RankRequest(),
) -> RankResponse:
    """Rank candidates against a job description.

    Args:
        candidates_file: JSONL file containing candidate profiles
        jd_file: Text file containing the job description
        request: Ranking parameters

    Returns:
        Ranked list of top candidates with scores and explanations
    """
    if _pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    try:
        # Save uploaded files to temp
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".jsonl", delete=False) as tmp_candidates:
            content = await candidates_file.read()
            tmp_candidates.write(content)
            candidates_path = tmp_candidates.name

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as tmp_jd:
            content = await jd_file.read()
            tmp_jd.write(content)
            jd_path = tmp_jd.name

        # Load candidates
        candidates = load_candidates_blob(open(candidates_path, "rb").read())

        # Set JD
        _pipeline.set_job_description(jd_path)

        # Rank
        scores = _pipeline.rank_from_list(candidates, top_k=request.top_k)

        results = [
            {
                "rank": s.rank,
                "candidate_id": s.candidate_id,
                "score": round(s.final_score, 6),
                "semantic": round(s.semantic, 4),
                "structural": round(s.structural, 4),
                "behavioral_multiplier": round(s.behavioral_multiplier, 4),
                "integrity_multiplier": round(s.integrity_multiplier, 4),
                "reasoning": s.reasoning,
            }
            for s in scores
        ]

        return RankResponse(
            status="success",
            total_candidates=len(candidates),
            top_k=len(results),
            results=results,
        )

    except Exception as e:
        logger.error(f"Ranking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temp files
        Path(candidates_path).unlink(missing_ok=True)
        Path(jd_path).unlink(missing_ok=True)


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "pipeline_ready": _pipeline is not None}


@app.get("/config")
async def get_config() -> dict:
    """Get current configuration."""
    cfg = Config()
    return {
        "embedding_model": cfg.EMBEDDING_MODEL,
        "embedding_dim": cfg.EMBEDDING_DIM,
        "semantic_weight": cfg.W_SEMANTIC,
        "structural_weight": cfg.W_STRUCTURAL,
        "reference_date": str(cfg.REFERENCE_DATE),
    }
