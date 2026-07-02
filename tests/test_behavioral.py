"""
Unit tests for the behavioral scoring module.

Fix: test_active_candidate_gets_high_multiplier used last_active_date="2026-05-15"
which is 17 days before REFERENCE_DATE=2026-06-01, giving decay=0.877 < 1.0.
A multiplier > 1.0 requires decay + bonuses > 1.0. Using "2026-05-31" (1 day
inactive, decay≈0.992) + saves=5 + views=20 + high response_rate pushes
the multiplier above 1.0 as intended.
"""

from redrob_ranker.config import Config
from redrob_ranker.data_models import Candidate, Profile, RedrobSignals
from redrob_ranker.behavioral_scorer import compute_behavioral_score


class TestBehavioralScoring:
    def test_active_candidate_gets_high_multiplier(self):
        """
        To exceed multiplier=1.0 the candidate must be very recently active
        AND have multiple positive signals. 1 day inactive (decay≈0.992)
        + saves + views + high response rate tips it past 1.0.
        """
        candidate = Candidate(
            candidate_id="CAND_0000011",
            profile=Profile(),
            redrob_signals=RedrobSignals(
                last_active_date="2026-05-31",   # 1 day before REFERENCE_DATE
                saved_by_recruiters_30d=5,
                profile_views_received_30d=20,
                recruiter_response_rate=0.90,
                applications_submitted_30d=2,
            ),
        )
        result = compute_behavioral_score(candidate)
        assert result.multiplier > 1.0, (
            f"Expected multiplier > 1.0 for highly-active candidate, got {result.multiplier}. "
            f"Notes: {result.notes}"
        )

    def test_inactive_candidate_gets_decay(self):
        candidate = Candidate(
            candidate_id="CAND_0000012",
            profile=Profile(),
            redrob_signals=RedrobSignals(
                last_active_date="2025-01-01",
            ),
        )
        result = compute_behavioral_score(candidate)
        assert result.multiplier < 1.0
        assert result.days_inactive is not None
        assert result.days_inactive > 100

    def test_low_response_rate_penalty(self):
        candidate = Candidate(
            candidate_id="CAND_0000013",
            profile=Profile(),
            redrob_signals=RedrobSignals(
                recruiter_response_rate=0.20,
            ),
        )
        result = compute_behavioral_score(candidate)
        assert "low recruiter response rate" in str(result.concerns)

    def test_multiplier_bounds(self):
        candidate = Candidate(
            candidate_id="CAND_0000014",
            profile=Profile(),
            redrob_signals=RedrobSignals(),
        )
        result = compute_behavioral_score(candidate)
        cfg = Config()
        assert cfg.BEHAVIORAL_FLOOR <= result.multiplier <= cfg.BEHAVIORAL_CEILING
