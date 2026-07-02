"""
Reasoning engine: generate human-readable explanations for each ranking.

Design decisions:
- No LLM calls (constraint compliance). All reasoning is template-based.
- Explanations are deterministic and reproducible.
- Each explanation includes: fit summary, evidence, concerns, and behavioral notes.
- Template selection is hash-based for variety without randomness.
"""

from __future__ import annotations

import hashlib

from redrob_ranker.config import Config
from redrob_ranker.data_models import (
    BehavioralScore,
    Candidate,
    IntegrityReport,
    SemanticScore,
    StructuralScore,
)


# ---------------------------------------------------------------------------
# Template banks
# ---------------------------------------------------------------------------

_FIT_TEMPLATES = {
    "strong": (
        "Strong match for the Senior AI Engineer role. {evidence}.",
        "Excellent fit: {evidence}.",
        "Highly aligned with the JD's requirements. {evidence}.",
    ),
    "good": (
        "Good match with solid evidence. {evidence}.",
        "Well-qualified candidate. {evidence}.",
        "Strong profile with relevant experience. {evidence}.",
    ),
    "moderate": (
        "Moderate fit with some relevant background. {evidence}.",
        "Partial match: {evidence}.",
        "Some alignment with role requirements. {evidence}.",
    ),
    "weak": (
        "Weak match for this role. {evidence}.",
        "Limited relevance to the position. {evidence}.",
        "Profile does not strongly align with JD requirements. {evidence}.",
    ),
}

_CONCERN_TEMPLATES = (
    "Note: {concerns}.",
    "Caution: {concerns}.",
    "Watch out: {concerns}.",
    "Flag: {concerns}.",
)

_BEHAVIORAL_TEMPLATES = (
    "Behavioral signals: {notes}.",
    "Platform engagement: {notes}.",
    "Recruiter interaction: {notes}.",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_template(templates: tuple[str, ...], seed: str) -> str:
    idx = int(hashlib.md5(seed.encode()).hexdigest(), 16) % len(templates)
    return templates[idx]


def _classify_fit(final_score: float) -> str:
    if final_score >= 0.75:
        return "strong"
    if final_score >= 0.55:
        return "good"
    if final_score >= 0.35:
        return "moderate"
    return "weak"


def _format_evidence(structural: StructuralScore) -> str:
    evidence = structural.evidence[:3]
    if evidence:
        return "; ".join(evidence)
    matched = structural.matched_skills[:5]
    if matched:
        return f"matched skills: {', '.join(matched)}"
    return "profile reviewed"


def _format_concerns(structural: StructuralScore, integrity: IntegrityReport) -> str:
    concerns = structural.concerns[:3]
    if integrity.hard_flags:
        concerns = [f"integrity alert: {integrity.hard_flags[0]}"] + concerns
    if concerns:
        return "; ".join(concerns)
    return "no major concerns"


def _format_behavioral(behavioral: BehavioralScore) -> str:
    notes = behavioral.notes[:2]
    if notes:
        return "; ".join(notes)
    return "neutral engagement signals"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_reasoning(
    candidate: Candidate,
    semantic: SemanticScore,
    structural: StructuralScore,
    behavioral: BehavioralScore,
    integrity: IntegrityReport,
    final_score: float,
    cfg: Config | None = None,
) -> str:
    """Generate a concise, explainable reasoning string for a candidate ranking."""
    cfg = cfg or Config()
    seed = candidate.candidate_id
    fit_class = _classify_fit(final_score)

    # Fit summary
    evidence_str = _format_evidence(structural)
    concerns_str = _format_concerns(structural, integrity)

    fit_template = _pick_template(_FIT_TEMPLATES[fit_class], seed + "fit")
    fit_text = fit_template.format(evidence=evidence_str)

    parts = [fit_text]

    # Add concerns if any
    if structural.concerns or integrity.hard_flags:
        concern_template = _pick_template(_CONCERN_TEMPLATES, seed + "concern")
        parts.append(concern_template.format(concerns=concerns_str))

    # Add behavioral notes
    if behavioral.notes:
        behav_template = _pick_template(_BEHAVIORAL_TEMPLATES, seed + "behav")
        parts.append(behav_template.format(notes=_format_behavioral(behavioral)))

    # Add penalties summary
    if structural.penalties:
        parts.append(f"Penalties applied: {', '.join(structural.penalties)}")

    # Add score breakdown
    parts.append(
        f"Score breakdown: semantic={semantic.normalized_similarity:.3f}, "
        f"structural={structural.score:.3f}, "
        f"behavioral={behavioral.multiplier:.3f}, "
        f"integrity={integrity.multiplier:.3f}"
    )

    return " ".join(parts)
