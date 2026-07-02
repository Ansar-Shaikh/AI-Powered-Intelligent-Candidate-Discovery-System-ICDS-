"""
Unit tests for the ranking pipeline.
"""

from redrob_ranker.config import Config
from redrob_ranker.data_models import Candidate, Profile, CareerEntry, Skill, RedrobSignals
from redrob_ranker.ranking_pipeline import RankingPipeline


class TestRankingPipeline:
    def test_pipeline_ranks_candidates(self):
        cfg = Config()
        pipeline = RankingPipeline(cfg=cfg)
        pipeline.set_jd_text("Senior AI Engineer with experience in embeddings, retrieval, and ranking systems.")

        candidates = [
            Candidate(
                candidate_id="CAND_0000019",
                profile=Profile(
                    current_title="Machine Learning Engineer",
                    years_of_experience=7.0,
                    summary="Built semantic search systems using embeddings and BM25 at Google.",
                ),
                career_history=[
                    CareerEntry(
                        company="Google",
                        title="ML Engineer",
                        description="Built semantic search and ranking systems.",
                        is_current=True,
                    )
                ],
                skills=[
                    Skill(name="Python", proficiency="expert", duration_months=36, endorsements=10),
                    Skill(name="Embeddings", proficiency="expert", duration_months=24, endorsements=8),
                ],
                redrob_signals=RedrobSignals(
                    last_active_date="2026-05-20",
                    recruiter_response_rate=0.90,
                ),
            ),
            Candidate(
                candidate_id="CAND_0000020",
                profile=Profile(
                    current_title="Marketing Manager",
                    years_of_experience=5.0,
                ),
                career_history=[
                    CareerEntry(company="Acme", title="Marketing Manager"),
                ],
                skills=[
                    Skill(name="SEO", proficiency="expert", duration_months=36),
                ],
            ),
        ]

        scores = pipeline.rank_from_list(candidates, top_k=2)

        assert len(scores) == 2
        assert scores[0].candidate_id == "CAND_0000019"
        assert scores[0].final_score > scores[1].final_score
        assert scores[0].rank == 1
        assert scores[1].rank == 2
        assert scores[0].reasoning

    def test_tie_breaking(self):
        cfg = Config()
        pipeline = RankingPipeline(cfg=cfg)
        pipeline.set_jd_text("AI Engineer")

        candidates = [
            Candidate(
                candidate_id="CAND_0000021",
                profile=Profile(current_title="ML Engineer", years_of_experience=5.0),
            ),
            Candidate(
                candidate_id="CAND_0000022",
                profile=Profile(current_title="ML Engineer", years_of_experience=5.0),
            ),
        ]

        scores = pipeline.rank_from_list(candidates, top_k=2)
        # Tie broken by candidate_id ascending
        assert scores[0].candidate_id < scores[1].candidate_id
