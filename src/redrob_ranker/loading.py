"""
Data loading utilities for candidate and job description ingestion.

Supports streaming from JSONL/JSONL.GZ for memory-efficient processing
of large candidate pools (~100K records, ~465MB).
"""

from __future__ import annotations

import gzip
import io
import json
from datetime import date
from pathlib import Path
from typing import Iterator, List, Optional

from redrob_ranker.config import Config
from redrob_ranker.data_models import Candidate, JobDescription


def iter_candidates(path: str | Path) -> Iterator[Candidate]:
    """Stream candidate records from JSONL or JSONL.GZ.

    Memory-efficient: yields one record at a time, keeping peak
    memory flat (~a few hundred MB) instead of loading the full pool.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                raw = json.loads(line)
                yield Candidate.model_validate(raw)


def load_candidates_blob(raw: bytes) -> List[Candidate]:
    """Parse uploaded blob that may be JSON array, JSONL, or gzipped JSONL."""
    if raw[:2] == b"\x1f\x8b":  # gzip magic
        raw = gzip.decompress(raw)

    text = raw.decode("utf-8").strip()
    if text.startswith("["):
        data = json.loads(text)
        return [Candidate.model_validate(item) for item in data]

    return [
        Candidate.model_validate(json.loads(line))
        for line in text.splitlines()
        if line.strip()
    ]


def parse_date(value: str | None) -> date | None:
    """Parse ISO date string to date object."""
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except (ValueError, TypeError):
        return None


def months_between(start: date, end: date) -> float:
    """Calculate months between two dates as a float."""
    return (end.year - start.year) * 12 + (end.month - start.month) + (end.day - start.day) / 30.0


def career_span_years(candidate: Candidate, cfg: Config | None = None) -> float:
    """Observable career span: earliest start date to reference date."""
    cfg = cfg or Config()
    starts = [
        parse_date(job.start_date)
        for job in candidate.career_history
        if job.start_date
    ]
    starts = [s for s in starts if s]
    if not starts:
        return 0.0
    return max(0.0, months_between(min(starts), cfg.REFERENCE_DATE) / 12.0)


def build_candidate_document(candidate: Candidate, cfg: Config | None = None) -> str:
    """Flatten a candidate into the text that gets embedded.

    Deliberately favours narrative fields (summary, role descriptions) over
    the skills list: the skills list is exactly where keyword stuffers live,
    and the JD warns the dataset is seeded with them. Descriptions are where
    plain-language strong candidates describe what they actually built.
    """
    cfg = cfg or Config()
    parts: List[str] = []

    # Headline and summary carry dense self-description
    profile = candidate.profile
    if profile.headline:
        parts.append(profile.headline)
    if profile.summary:
        parts.append(profile.summary[:400])

    # Top career entries by recency
    sorted_jobs = sorted(
        candidate.career_history,
        key=lambda j: parse_date(j.start_date) or date.min,
        reverse=True,
    )
    for job in sorted_jobs[:cfg.EMBED_CAREER_ENTRIES]:
        desc = (job.description or "")[:cfg.EMBED_CAREER_CHARS]
        title = job.title or ""
        parts.append(f"{title}: {desc}")

    # Top skills by endorsements — anti-keyword-stuffing filter:
    # only include skills with >= 6 months of use
    top_skills = sorted(
        [s for s in candidate.skills if (s.duration_months or 0) >= 6],
        key=lambda s: s.endorsements,
        reverse=True,
    )[:8]
    skill_names = ", ".join(s.name for s in top_skills)
    if skill_names:
        parts.append(f"Skills: {skill_names}")

    return " | ".join(parts)


def load_job_description(path: str | Path) -> JobDescription:
    """Load and parse a job description from text file."""
    text = Path(path).read_text(encoding="utf-8")
    return JobDescription(
        title="Senior AI Engineer - Search & Retrieval",
        raw_text=text,
    )
