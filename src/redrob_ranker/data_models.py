"""
Domain models for the Redrob AI Hiring Intelligence Platform.

All data structures are Pydantic models for runtime validation,
serialization safety, and IDE autocomplete support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ProficiencyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class EducationTier(str, Enum):
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4 = "tier_4"


class WorkMode(str, Enum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    FLEXIBLE = "flexible"


# ---------------------------------------------------------------------------
# Nested Models
# ---------------------------------------------------------------------------

class Skill(BaseModel):
    name: str
    proficiency: ProficiencyLevel
    endorsements: int = 0
    duration_months: int = 0


class CareerEntry(BaseModel):
    company: str
    title: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_months: Optional[int] = None
    is_current: bool = False
    industry: Optional[str] = None
    company_size: Optional[str] = None
    description: Optional[str] = None


class EducationEntry(BaseModel):
    institution: str
    degree: str
    field_of_study: str
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    grade: Optional[str] = None
    tier: EducationTier = EducationTier.TIER_4


class Certification(BaseModel):
    name: str
    issuer: str
    year: Optional[int] = None


class Language(BaseModel):
    language: str
    proficiency: str


class SalaryRange(BaseModel):
    min: float = 0.0
    max: float = 0.0

    @model_validator(mode="after")
    def check_range(self) -> SalaryRange:
        if self.min > self.max and self.max > 0:
            # Generator noise - swap silently
            self.min, self.max = self.max, self.min
        return self


class RedrobSignals(BaseModel):
    profile_completeness_score: Optional[float] = None
    signup_date: Optional[str] = None
    last_active_date: Optional[str] = None
    open_to_work_flag: bool = False
    profile_views_received_30d: int = 0
    applications_submitted_30d: int = 0
    recruiter_response_rate: Optional[float] = None
    avg_response_time_hours: Optional[float] = None
    skill_assessment_scores: Dict[str, float] = Field(default_factory=dict)
    connection_count: int = 0
    endorsements_received: int = 0
    notice_period_days: Optional[int] = None
    expected_salary_range_inr_lpa: SalaryRange = Field(default_factory=SalaryRange)
    preferred_work_mode: str = "flexible"
    willing_to_relocate: bool = False
    github_activity_score: Optional[float] = None
    search_appearance_30d: int = 0
    saved_by_recruiters_30d: int = 0
    interview_completion_rate: Optional[float] = None
    offer_acceptance_rate: Optional[float] = None
    verified_email: bool = False
    verified_phone: bool = False
    linkedin_connected: bool = False


class Profile(BaseModel):
    anonymized_name: Optional[str] = None
    headline: Optional[str] = None
    summary: Optional[str] = None
    location: Optional[str] = None
    country: Optional[str] = None
    years_of_experience: float = 0.0
    current_title: Optional[str] = None
    current_company: Optional[str] = None
    current_company_size: Optional[str] = None
    current_industry: Optional[str] = None


# ---------------------------------------------------------------------------
# Top-Level Candidate Model
# ---------------------------------------------------------------------------

class Candidate(BaseModel):
    candidate_id: str = Field(..., pattern=r"^CAND_\d{7}$")
    profile: Profile = Field(default_factory=Profile)
    career_history: List[CareerEntry] = Field(default_factory=list)
    education: List[EducationEntry] = Field(default_factory=list)
    skills: List[Skill] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)
    languages: List[Language] = Field(default_factory=list)
    redrob_signals: RedrobSignals = Field(default_factory=RedrobSignals)

    @field_validator("candidate_id")
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        if not v.startswith("CAND_") or len(v) != 12:
            raise ValueError(f"Invalid candidate_id format: {v}")
        return v

    def get_narrative_text(self) -> str:
        """Build the full narrative text for embedding and analysis."""
        parts = []
        p = self.profile
        if p.headline:
            parts.append(p.headline)
        if p.summary:
            parts.append(p.summary)
        for job in self.career_history:
            if job.description:
                parts.append(f"{job.title}: {job.description}")
        return " ".join(parts)

    def get_current_job(self) -> Optional[CareerEntry]:
        for job in self.career_history:
            if job.is_current:
                return job
        return None

    def get_most_recent_jobs(self, n: int = 3) -> List[CareerEntry]:
        """Return the n most recent jobs by start date."""
        sorted_jobs = sorted(
            self.career_history,
            key=lambda j: j.start_date or "",
            reverse=True,
        )
        return sorted_jobs[:n]


# ---------------------------------------------------------------------------
# Scoring Result Models
# ---------------------------------------------------------------------------

@dataclass
class SemanticScore:
    raw_similarity: float = 0.0
    normalized_similarity: float = 0.0
    embedding_document: str = ""


@dataclass
class StructuralScore:
    score: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    penalties: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    matched_skills: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)


@dataclass
class BehavioralScore:
    multiplier: float = 1.0
    notes: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    days_inactive: Optional[int] = None
    response_rate: Optional[float] = None


@dataclass
class IntegrityReport:
    hard_flags: List[str] = field(default_factory=list)
    soft_flags: List[str] = field(default_factory=list)

    @property
    def multiplier(self) -> float:
        from redrob_ranker.config import Config
        cfg = Config()
        if len(self.hard_flags) >= 2:
            return cfg.INTEGRITY_FATAL
        if len(self.hard_flags) == 1:
            return cfg.INTEGRITY_HARD
        return cfg.INTEGRITY_SOFT_DECAY ** len(self.soft_flags)

    @property
    def is_suspect(self) -> bool:
        return bool(self.hard_flags)


@dataclass
class FinalScore:
    candidate_id: str
    final_score: float
    semantic: float
    structural: float
    behavioral_multiplier: float
    integrity_multiplier: float
    rank: int = 0
    reasoning: str = ""

    def sort_key(self) -> tuple:
        """Tie-break: candidate_id ascending on equal scores."""
        return (-round(self.final_score, 4), self.candidate_id)


# ---------------------------------------------------------------------------
# Job Description Model
# ---------------------------------------------------------------------------

class JobDescription(BaseModel):
    title: str
    raw_text: str
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    experience_range_years: tuple[float, float] = (0.0, 100.0)
    location_preferences: List[str] = Field(default_factory=list)
    disqualifiers: List[str] = Field(default_factory=list)

    def get_embedding_text(self) -> str:
        """Text used for semantic similarity computation."""
        return self.raw_text
