"""
Integrity engine: honeypot detection and contradiction audit.

The JD explicitly warns that the dataset contains seeded honeypot profiles.
This module detects those profiles and assigns integrity multipliers.

Design decisions:
- All checks are deterministic, no LLM calls (constraint compliance).
- Contradiction audit compares semantic and structural scores; a large gap
  suggests keyword stuffing (high structural from skills, low semantic from
  narrative).
- Honeypot detection checks for impossible dates, mismatched durations, and
  skill-level inconsistencies.
"""

from __future__ import annotations

import logging
from datetime import date

from redrob_ranker.config import Config
from redrob_ranker.data_models import Candidate, IntegrityReport
from redrob_ranker.loading import months_between, parse_date

logger = logging.getLogger(__name__)


def _check_impossible_dates(candidate: Candidate, flags: list[str]) -> None:
    """Check for career entries with impossible or future dates."""
    for job in candidate.career_history:
        start = parse_date(job.start_date)
        end = parse_date(job.end_date)
        if start and start > date.today():
            flags.append(f"future_start_date:{job.start_date}")
        if end and end > date.today():
            flags.append(f"future_end_date:{job.end_date}")
        if start and end and end < start:
            flags.append(f"end_before_start:{job.start_date}->{job.end_date}")


def _check_duration_mismatch(candidate: Candidate, flags: list[str], cfg: Config) -> None:
    """Check for career entries where duration_months doesn't match date range."""
    for job in candidate.career_history:
        start = parse_date(job.start_date)
        end = parse_date(job.end_date) or date.today()
        if start:
            expected = months_between(start, end)
            actual = job.duration_months or 0
            if abs(expected - actual) > cfg.HONEYPOT_DURATION_MISMATCH_MONTHS:
                flags.append(
                    f"duration_mismatch:{job.title}@{job.company} "
                    f"expected={expected:.1f}m actual={actual}m"
                )


def _check_expert_zero_duration(candidate: Candidate, flags: list[str], cfg: Config) -> None:
    """Skills claimed as 'expert' with zero duration months are suspicious."""
    expert_zero = [
        s.name for s in candidate.skills
        if (s.proficiency.value if hasattr(s.proficiency, "value") else s.proficiency) == "expert"
        and (s.duration_months or 0) == 0
    ]
    if len(expert_zero) >= cfg.HONEYPOT_EXPERT_ZERO_DURATION_MIN:
        flags.append(f"expert_zero_duration:{len(expert_zero)} skills")


def _check_yoe_span(candidate: Candidate, flags: list[str], cfg: Config) -> None:
    """Check if career span matches claimed years of experience."""
    starts = [parse_date(j.start_date) for j in candidate.career_history if j.start_date]
    starts = [s for s in starts if s]
    if not starts:
        return
    span = months_between(min(starts), date.today()) / 12.0
    yoe = candidate.profile.years_of_experience or 0.0
    if abs(span - yoe) > cfg.HONEYPOT_YOE_SPAN_SLACK_YEARS:
        flags.append(f"yoe_mismatch:career_span={span:.1f}y vs yoe={yoe:.1f}y")


def _check_skill_stuffing(candidate: Candidate, flags: list[str]) -> None:
    """Detect suspicious skill lists: many skills with no endorsements or duration."""
    total = len(candidate.skills)
    if total == 0:
        return
    no_evidence = sum(
        1 for s in candidate.skills
        if (s.duration_months or 0) == 0 and (s.endorsements or 0) == 0
    )
    if no_evidence / total > 0.8 and total > 10:
        flags.append(f"skill_stuffing:{no_evidence}/{total} skills have no duration or endorsements")


def _check_contradiction(
    candidate: Candidate,
    semantic_score: float,
    structural_score: float,
    flags: list[str],
    cfg: Config,
) -> None:
    """Audit: large gap between semantic and structural scores suggests
    keyword stuffing (high structural from skills, low semantic from narrative).
    """
    delta = abs(structural_score - semantic_score)
    if delta > cfg.CONTRADICTION_DELTA_THRESHOLD:
        if structural_score > semantic_score:
            flags.append(
                f"contradiction:structural({structural_score:.3f}) >> semantic({semantic_score:.3f}) "
                "possible keyword stuffing"
            )
        else:
            flags.append(
                f"contradiction:semantic({semantic_score:.3f}) >> structural({structural_score:.3f}) "
                "possible under-representation in structured data"
            )


def compute_integrity_report(
    candidate: Candidate,
    semantic_score: float = 0.0,
    structural_score: float = 0.0,
    cfg: Config | None = None,
) -> IntegrityReport:
    """Full integrity check for a candidate."""
    cfg = cfg or Config()
    hard_flags: list[str] = []
    soft_flags: list[str] = []

    _check_impossible_dates(candidate, hard_flags)
    _check_duration_mismatch(candidate, hard_flags, cfg)
    _check_expert_zero_duration(candidate, hard_flags, cfg)
    _check_yoe_span(candidate, hard_flags, cfg)
    _check_skill_stuffing(candidate, soft_flags)
    _check_contradiction(candidate, semantic_score, structural_score, soft_flags, cfg)

    if hard_flags:
        logger.debug(f"Honeypot flags for {candidate.candidate_id}: {hard_flags}")

    return IntegrityReport(hard_flags=hard_flags, soft_flags=soft_flags)
