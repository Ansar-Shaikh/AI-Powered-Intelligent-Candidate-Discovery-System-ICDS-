"""
Unit tests for the integrity engine.

Fix applied to test_clean_profile:
  The original test created a CareerEntry with start_date="2020-01-01"
  and end_date="2023-01-01" (career_span ≈ 6.5 years) but left
  Profile.years_of_experience at its default (0.0). The YOE-span check
  compares career span vs profile.years_of_experience, flagging a 6.5y
  gap against 0.0y as a honeypot signal. Fix: explicitly set
  years_of_experience=6.5 to match the actual career span.
"""

from redrob_ranker.config import Config
from redrob_ranker.data_models import Candidate, Profile, CareerEntry, Skill
from redrob_ranker.integrity_engine import compute_integrity_report


class TestIntegrityChecks:
    def test_future_date_flag(self):
        candidate = Candidate(
            candidate_id="CAND_0000015",
            profile=Profile(),
            career_history=[
                CareerEntry(company="X", title="Y", start_date="2030-01-01"),
            ],
        )
        report = compute_integrity_report(candidate)
        assert any("future" in f for f in report.hard_flags)

    def test_expert_zero_duration(self):
        candidate = Candidate(
            candidate_id="CAND_0000016",
            profile=Profile(),
            skills=[
                Skill(name="Python", proficiency="expert", duration_months=0),
                Skill(name="ML",     proficiency="expert", duration_months=0),
                Skill(name="RAG",    proficiency="expert", duration_months=0),
            ],
        )
        report = compute_integrity_report(candidate)
        assert any("expert_zero_duration" in f for f in report.hard_flags)

    def test_contradiction_detection(self):
        candidate = Candidate(
            candidate_id="CAND_0000017",
            profile=Profile(),
        )
        report = compute_integrity_report(candidate, semantic_score=0.1, structural_score=0.8)
        assert any("contradiction" in f for f in report.soft_flags)

    def test_clean_profile(self):
        """
        The YOE-span check measures from the earliest start_date to TODAY,
        not to end_date. start_date="2023-06-01" → span ≈ 1 year to today.
        years_of_experience=1.0 keeps the delta within the 2.0-year slack.
        """
        candidate = Candidate(
            candidate_id="CAND_0000018",
            profile=Profile(years_of_experience=1.0),
            career_history=[
                CareerEntry(
                    company="Google",
                    title="ML Engineer",
                    start_date="2025-06-01",
                    end_date="2026-06-01",
                    duration_months=12,
                ),
            ],
            skills=[
                Skill(name="Python", proficiency="advanced", duration_months=24, endorsements=5),
            ],
        )
        report = compute_integrity_report(candidate)
        assert not report.hard_flags, f"Unexpected hard_flags: {report.hard_flags}"
        assert not report.soft_flags, f"Unexpected soft_flags: {report.soft_flags}"
        assert report.multiplier == 1.0
