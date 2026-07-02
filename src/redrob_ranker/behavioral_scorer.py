"""
Behavioral scoring using Redrob platform signals.

Design decisions:
- Uses exponential decay for inactivity (90-day half-life).
- Recruiter save count as social proof (capped to prevent outlier distortion).
- Response rate as direct proxy for recruiter engagement quality.
- All multipliers are bounded to prevent any single signal from dominating.
"""

from __future__ import annotations

import math
from datetime import date, datetime

from redrob_ranker.config import Config
from redrob_ranker.data_models import BehavioralScore, Candidate


def _days_inactive(candidate: Candidate, cfg: Config) -> int:
    """Days since last activity."""
    last_active = candidate.redrob_signals.last_active_date
    if not last_active:
        return 9999
    try:
        last = datetime.strptime(last_active[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return 9999
    return max(0, (cfg.REFERENCE_DATE - last).days)


def compute_behavioral_score(candidate: Candidate, cfg: Config | None = None) -> BehavioralScore:
    """Compute behavioral multiplier for a candidate.

    Returns a multiplier in [floor, ceiling] that scales the base score.
    """
    cfg = cfg or Config()
    signals = candidate.redrob_signals
    result = BehavioralScore()

    # 1. Inactivity decay
    inactive = _days_inactive(candidate, cfg)
    result.days_inactive = inactive
    if inactive > 0:
        decay = math.exp(-cfg.BEHAVIORAL_DECAY_LAMBDA * inactive)
        result.notes.append(f"{inactive} days since last activity (decay factor {decay:.3f})")
    else:
        decay = 1.0

    # 2. Recruiter saves (social proof)
    saves = signals.saved_by_recruiters_30d
    if saves > 0:
        save_bonus = min(saves, cfg.RECRUITER_SAVE_MAX) * cfg.RECRUITER_SAVE_BONUS
        result.notes.append(f"saved by {saves} recruiter(s) in last 30 days (+{save_bonus:.3f})")
        decay += save_bonus

    # 3. Profile views
    views = signals.profile_views_received_30d
    if views >= cfg.PROFILE_VIEWS_THRESHOLD:
        result.notes.append(f"{views} profile views in last 30 days (+{cfg.PROFILE_VIEWS_BONUS:.3f})")
        decay += cfg.PROFILE_VIEWS_BONUS

    # 4. Applications submitted
    apps = signals.applications_submitted_30d
    if apps > 0:
        result.notes.append(f"{apps} applications submitted in last 30 days (+{cfg.APP_SUBMITTED_BONUS:.3f})")
        decay += cfg.APP_SUBMITTED_BONUS

    # 5. Response rate
    response_rate = signals.recruiter_response_rate
    result.response_rate = response_rate
    if response_rate is not None:
        if response_rate < 0.30:
            result.concerns.append(f"low recruiter response rate ({response_rate:.1%})")
            decay -= 0.05
        elif response_rate > 0.80:
            result.notes.append(f"high recruiter response rate ({response_rate:.1%})")
            decay += 0.02

    # 6. Interview completion rate
    icr = signals.interview_completion_rate
    if icr is not None and icr < 0.40:
        result.concerns.append(f"low interview completion rate ({icr:.1%})")
        decay -= 0.03

    # Clamp
    result.multiplier = max(cfg.BEHAVIORAL_FLOOR, min(cfg.BEHAVIORAL_CEILING, decay))
    return result
