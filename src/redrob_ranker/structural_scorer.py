"""
Structural scoring against the explicit rules of the job description.

The JD is unusually explicit: it names the profile it wants, the profiles it
will reject, and the traps planted in the dataset. This module encodes those
rules as deterministic, inspectable features. Every penalty corresponds to a
named section of the job description.

Design decisions:
- All keyword matching is lowercase substring with word-boundary guards for
  short tokens to prevent false positives.
- Module-level compiled regexes avoid recompilation per candidate (100K+).
- Evidence phrases are deterministic (hash-based rotation) for reproducibility.
"""

from __future__ import annotations

import hashlib
import math
import re

from redrob_ranker.config import Config
from redrob_ranker.data_models import Candidate, StructuralScore
from redrob_ranker.loading import parse_date

# ---------------------------------------------------------------------------
# Pre-compiled regexes (performance: avoid 100K recompilations)
# ---------------------------------------------------------------------------

# NLP/IR terms that are unambiguous enough to count as positive NLP evidence.
_NLP_POSITIVE_TERMS: tuple[str, ...] = (
    "natural language processing", "natural language", "information retrieval",
    "language model", "large language", "embedding", "embeddings",
    "retrieval", "semantic search", "vector search", "text classification",
    "sentiment analysis", "named entity", "machine translation",
    "question answering", "reading comprehension",
    "ranking", "recommendation", "recommender",
    "word2vec", "bert", "transformer", "lstm",
)

_NLP_POSITIVE_COMPILED: tuple[re.Pattern, ...] = tuple(
    re.compile(r"\b" + re.escape(t) + r"\b") for t in _NLP_POSITIVE_TERMS
)

# LangChain/wrapper vocabulary
_LANGCHAIN_TERMS: tuple[str, ...] = (
    "langchain", "openai api", "gpt-", "chatgpt", "llm api",
    "llamaindex", "llama index", "llama_index", "openai.chat",
)

# Pre-LLM ML production terms
_PRE_LLM_TERMS: tuple[str, ...] = (
    "word2vec", "fasttext", "glove embedding",
    "bert", "lstm", "seq2seq", "attention mechanism",
    "xgboost", "scikit-learn", "sklearn", "tensorflow", "pytorch", "keras",
    "a/b test", "ndcg", "mrr", "bm25",
    "elasticsearch",
    "feature engineering", "gradient boosting", "random forest",
    "collaborative filtering", "matrix factorization",
    "sentence-transformer", "sentence transformer",
)

# Research-flavoured titles
_RESEARCH_FLAVORED_TITLES: tuple[str, ...] = (
    "research engineer", "research scientist",
    "research analyst", "ai researcher", "ml researcher",
)


def _contains_any(text: str, terms: tuple[str, ...] | frozenset | set) -> bool:
    return any(t in text for t in terms)


def _count_hits(text: str, terms: tuple[str, ...]) -> int:
    return sum(1 for t in terms if t in text)


def _skill_matches_jd(name: str, cfg: Config) -> bool:
    """Check if skill name matches any JD skill token.

    Long tokens (>= 5 chars) use plain substring containment.
    Short tokens (< 5 chars: 'rag', 'e5', 'nlp', 'lora', 'bge') use
    word-boundary regex to prevent false positives.
    """
    name_lower = name.lower()
    # Long tokens: substring match
    for token in cfg.JD_SKILLS:
        if len(token) >= 5:
            if token in name_lower or name_lower in token:
                return True
    # Short tokens: word-boundary regex
    short_skills = [s for s in cfg.JD_SKILLS if len(s) < 5]
    if short_skills:
        pattern = re.compile(r"\b(" + "|".join(re.escape(s) for s in short_skills) + r")\b")
        if pattern.search(name_lower):
            return True
    return False


# ---------------------------------------------------------------------------
# Title Domain Score
# ---------------------------------------------------------------------------

_TITLE_EVIDENCE_TEMPLATES = (
    "current title '{title}' is squarely in the JD's domain",
    "role as '{title}' maps directly onto the JD's search-and-ranking mandate",
    "'{title}' title aligns with the retrieval/ML focus the JD requires",
    "current position as '{title}' fits the target ML/AI profile",
    "title '{title}' places this candidate in the JD's core domain",
)


def _title_domain_score(candidate: Candidate, result: StructuralScore, cfg: Config) -> float:
    profile = candidate.profile
    title = (profile.current_title or "").lower()
    headline = (profile.headline or "").lower()
    combined = f"{title} {headline}"

    if _contains_any(title, cfg.NON_TECH_TITLE_TERMS):
        return 0.05

    if _contains_any(combined, cfg.ML_TITLE_TERMS):
        # Deterministic rotation by hashing title
        idx = int(hashlib.md5(profile.current_title.encode()).hexdigest(), 16) % len(_TITLE_EVIDENCE_TEMPLATES)
        result.evidence.append(_TITLE_EVIDENCE_TEMPLATES[idx].format(title=profile.current_title))
        return 1.0

    if _contains_any(combined, cfg.ADJACENT_TITLE_TERMS):
        return 0.55

    if _contains_any(combined, cfg.ENGINEERING_TITLE_TERMS) and not _contains_any(
        title, cfg.NON_TECH_TITLE_TERMS
    ):
        return 0.35

    return 0.05


# ---------------------------------------------------------------------------
# Career Evidence Score
# ---------------------------------------------------------------------------

_TIER_1_REGEX: re.Pattern | None = None


def _get_tier1_regex(cfg: Config) -> re.Pattern:
    global _TIER_1_REGEX
    if _TIER_1_REGEX is None:
        _TIER_1_REGEX = re.compile(
            r"\b(" + "|".join(re.escape(t1) for t1 in cfg.TIER_1_COMPANIES) + r")\b"
        )
    return _TIER_1_REGEX


def _career_evidence_score(candidate: Candidate, result: StructuralScore, narrative: str, cfg: Config) -> float:
    """Evidence of having built retrieval/ranking/ML systems in production."""
    history = candidate.career_history
    profile = candidate.profile
    history_narrative = " ".join((j.description or "") for j in history).lower()

    retrieval_hits = _count_hits(history_narrative, cfg.RETRIEVAL_EVIDENCE_TERMS)
    production_hits = _count_hits(history_narrative, cfg.PRODUCTION_EVIDENCE_TERMS)
    ml_hits = _count_hits(history_narrative, cfg.ML_EVIDENCE_TERMS)

    # Product-company exposure
    product_roles = [
        j for j in history
        if not any(ind in (j.industry or "").lower() for ind in cfg.CONSULTING_INDUSTRIES)
    ]

    # Tier-1 company check
    tier1_regex = _get_tier1_regex(cfg)
    tier1_roles = [
        j for j in history
        if tier1_regex.search((j.company or "").lower())
    ]

    score = 0.0
    if retrieval_hits:
        score += min(0.55, 0.18 * retrieval_hits)
        result.evidence.append("career history describes retrieval/search/ranking work")
    if ml_hits:
        score += min(0.20, 0.07 * ml_hits)
    if production_hits:
        score += min(0.15, 0.05 * production_hits)

    # Product-company bonus only when ML/retrieval evidence exists
    if product_roles and (retrieval_hits or ml_hits):
        score += 0.10

    # Tier-1 prestige bonus only when ML/retrieval evidence exists
    if tier1_roles and (retrieval_hits or ml_hits):
        score += cfg.TIER_1_COMPANY_BONUS
        company_names = ", ".join(dict.fromkeys(j.company for j in tier1_roles))[:60]
        result.evidence.append(f"Tier-1 product company experience ({company_names})")

    score = min(1.0, score)

    if retrieval_hits and production_hits:
        result.evidence.append("describes shipping to production, not just prototyping")

    return score


# ---------------------------------------------------------------------------
# Experience Band Score
# ---------------------------------------------------------------------------

def _experience_band_score(candidate: Candidate, result: StructuralScore, cfg: Config) -> float:
    yoe = candidate.profile.years_of_experience or 0.0

    if cfg.EXP_IDEAL_LO <= yoe <= cfg.EXP_IDEAL_HI:
        return 1.0
    if yoe < cfg.EXP_HARD_FLOOR:
        result.concerns.append(f"only {yoe:.1f} years of experience for a senior founding-team role")
        return 0.05
    if yoe < cfg.EXP_IDEAL_LO:
        result.concerns.append(f"{yoe:.1f} years is below the JD's 5-9 band")
        return 0.55
    if yoe <= cfg.EXP_SOFT_HI:
        return 0.80
    result.concerns.append(f"{yoe:.1f} years may read as over-band for a hands-on IC role")
    return 0.55


# ---------------------------------------------------------------------------
# Skills Trust Score
# ---------------------------------------------------------------------------

def _skills_trust_score(candidate: Candidate, result: StructuralScore, cfg: Config) -> float:
    """JD-skill coverage, trust-weighted.

    A skill only counts in proportion to evidence it was actually used:
    proficiency alone is self-reported; endorsements and duration_months
    are harder to fake. This is the anti-keyword-stuffing weighting.
    """
    prof_weight = {"beginner": 0.25, "intermediate": 0.55, "advanced": 0.85, "expert": 1.0}
    assessments = candidate.redrob_signals.skill_assessment_scores or {}

    total = 0.0
    for skill in candidate.skills:
        name = (skill.name or "").lower()
        if not _skill_matches_jd(name, cfg):
            continue

        duration = skill.duration_months or 0
        endorsements = skill.endorsements or 0
        trust = min(1.0, duration / 24.0) * (
            0.5 + 0.5 * min(1.0, math.log1p(endorsements) / math.log1p(30))
        )
        weight = prof_weight.get(skill.proficiency.value if hasattr(skill.proficiency, "value") else skill.proficiency, 0.4)

        # Platform assessment as independent verification
        assessed = assessments.get(skill.name)
        if assessed is not None:
            trust *= 0.6 + 0.4 * (assessed / 100.0)

        contribution = weight * trust
        if contribution > 0.15:
            result.matched_skills.append(skill.name)
        total += contribution

    return min(1.0, total / 4.0)


# ---------------------------------------------------------------------------
# Education Tier Score
# ---------------------------------------------------------------------------

def _education_tier_score(candidate: Candidate, result: StructuralScore, cfg: Config) -> float:
    edu = candidate.education
    if not edu:
        return 0.5

    tier_map = {"tier_1": 1, "tier_2": 2, "tier_3": 3, "tier_4": 4}
    best_tier = min(
        (e.tier.value if hasattr(e.tier, "value") else e.tier for e in edu),
        key=lambda t: tier_map.get(t, 4),
        default="tier_4"
    )
    return {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.50, "tier_4": 0.30}.get(best_tier, 0.5)


# ---------------------------------------------------------------------------
# Logistics Score
# ---------------------------------------------------------------------------

def _logistics_score(candidate: Candidate, result: StructuralScore, cfg: Config) -> float:
    profile = candidate.profile
    signals = candidate.redrob_signals
    location = (profile.location or "").lower()
    country = (profile.country or "").lower()
    relocate = signals.willing_to_relocate

    # Location scoring
    if country and country != "india":
        loc = cfg.LOCATION_ABROAD
        result.concerns.append(f"based in {profile.location}, {profile.country} -- JD doesn't sponsor visas")
    elif _contains_any(location, cfg.LOCATION_PREFERRED):
        loc = 1.0
        result.evidence.append(f"based in {profile.location} (JD's preferred location)")
    elif _contains_any(location, cfg.LOCATION_WELCOME):
        loc = 0.90
    elif relocate:
        loc = cfg.LOCATION_INDIA_RELOCATE
    else:
        loc = cfg.LOCATION_INDIA_NO_RELOCATE
        result.concerns.append("outside the JD's listed cities and not flagged willing to relocate")

    # Notice period
    raw_notice = signals.notice_period_days
    notice = 90 if raw_notice is None else int(raw_notice)
    notice_score = cfg.NOTICE_LONG
    for max_days, value in cfg.NOTICE_STEPS:
        if notice <= max_days:
            notice_score = value
            break
    if notice > 60:
        result.concerns.append(f"{notice}-day notice period (JD prefers sub-30)")

    # Work mode
    mode = signals.preferred_work_mode or "flexible"
    mode_score = 1.0 if mode in ("hybrid", "flexible", "onsite") else 0.75

    # Salary compatibility
    sal = signals.expected_salary_range_inr_lpa
    sal_mid = (sal.min + sal.max) / 2
    if 25 <= sal_mid <= 75:
        salary_compat = 1.0
    elif sal_mid > 100:
        salary_compat = 0.85
    else:
        salary_compat = 1.0

    return (0.55 * loc + 0.30 * notice_score + 0.15 * mode_score) * salary_compat


# ---------------------------------------------------------------------------
# Penalties
# ---------------------------------------------------------------------------

def _apply_penalties(candidate: Candidate, base: float, result: StructuralScore, narrative: str, cfg: Config) -> float:
    profile = candidate.profile
    history = candidate.career_history
    title = (profile.current_title or "").lower()
    signals = candidate.redrob_signals

    # 1. Keyword stuffer: JD-perfect skill list attached to non-technical career
    non_tech_title = _contains_any(title, cfg.NON_TECH_TITLE_TERMS)
    if non_tech_title:
        jd_skill_count = sum(
            1 for s in candidate.skills
            if _skill_matches_jd((s.name or "").lower(), cfg)
        )
        if jd_skill_count >= cfg.KEYWORD_STUFFER_MIN_JD_SKILLS:
            no_real_evidence = (
                _count_hits(narrative, cfg.RETRIEVAL_EVIDENCE_TERMS) == 0
                and _count_hits(narrative, cfg.ML_EVIDENCE_TERMS) == 0
            )
            if no_real_evidence:
                result.penalties.append("keyword_stuffer")
                result.concerns.append(f"AI skill list doesn't match a '{profile.current_title}' career")
                return base * cfg.PENALTY_KEYWORD_STUFFER

    # 2. Consulting-only career
    if history:
        services = [
            j for j in history
            if any(ind in (j.industry or "").lower() for ind in cfg.CONSULTING_INDUSTRIES)
            or _contains_any((j.company or "").lower(), cfg.CONSULTING_FIRMS)
        ]
        if len(services) == len(history):
            result.penalties.append("consulting_only")
            result.concerns.append("entire career at IT-services/consulting firms (explicit JD disqualifier)")
            base *= cfg.PENALTY_CONSULTING_ONLY

    # 3. Research-only career, no production
    if history and all(
        _contains_any((j.title or "").lower(), cfg.RESEARCH_TITLE_TERMS)
        or (j.industry or "").lower() in cfg.RESEARCH_INDUSTRIES
        for j in history
    ) and _count_hits(narrative, cfg.PRODUCTION_EVIDENCE_TERMS) == 0:
        result.penalties.append("research_only")
        result.concerns.append("pure research background with no production deployment signal")
        base *= cfg.PENALTY_RESEARCH_ONLY

    # 4. Title-chaser: many short stints
    yoe = profile.years_of_experience or 0.0
    if len(history) >= cfg.TITLE_CHASER_MIN_ROLES and yoe >= 4:
        tenures = [j.duration_months or 0 for j in history]
        if tenures and sum(tenures) / len(tenures) < cfg.TITLE_CHASER_MAX_AVG_TENURE_MONTHS:
            result.penalties.append("title_chaser")
            result.concerns.append("frequent short stints -- JD wants a 3+ year commitment")
            base *= cfg.PENALTY_TITLE_CHASER

    # 5. CV/speech/robotics specialist without NLP/IR exposure
    skills_text = " ".join((s.name or "").lower() for s in candidate.skills)
    full_text = f"{narrative} {skills_text}"
    cv_hits = _count_hits(full_text, cfg.CV_SPEECH_ROBOTICS_TERMS)
    if cv_hits >= 3:
        nlp_hits = sum(1 for pat in _NLP_POSITIVE_COMPILED if pat.search(full_text))
        if nlp_hits == 0:
            result.penalties.append("cv_only")
            result.concerns.append("primary expertise in CV/speech/robotics with no NLP/IR exposure")
            base *= cfg.PENALTY_CV_ONLY

    # 6. LangChain-only disqualifier
    lc_in_narrative = _contains_any(narrative, _LANGCHAIN_TERMS)
    if lc_in_narrative:
        pre_llm_hits = _count_hits(narrative, _PRE_LLM_TERMS)
        if pre_llm_hits == 0:
            lc_skill_months = sum(
                s.duration_months or 0
                for s in candidate.skills
                if any(t in (s.name or "").lower() for t in ("langchain", "openai", "gpt", "chatgpt"))
            )
            non_lc_ml_months = sum(
                s.duration_months or 0
                for s in candidate.skills
                if _skill_matches_jd((s.name or "").lower(), cfg)
                and not any(t in (s.name or "").lower() for t in ("langchain", "openai", "gpt", "chatgpt"))
            )
            if lc_skill_months > 0 and non_lc_ml_months < 12:
                result.penalties.append("langchain_only")
                result.concerns.append(
                    "ML experience appears limited to recent LLM-wrapper work with no pre-LLM production history"
                )
                base *= cfg.PENALTY_LANGCHAIN_ONLY

    # 7. Outside India
    country = (profile.country or "").lower()
    if country and country != "india":
        result.penalties.append("abroad")
        base *= 0.80 if signals.willing_to_relocate else 0.55

    # 8. Stale hands-on: 18+ months in leadership with no hands-on verbs
    current = next((j for j in history if j.is_current), None)
    if current and _contains_any((current.title or "").lower(), cfg.LEADERSHIP_TITLE_TERMS):
        months = current.duration_months or 0
        desc = (current.description or "").lower()
        if months >= cfg.STALE_HANDS_ON_MONTHS and not _contains_any(desc, cfg.HANDS_ON_VERBS):
            result.penalties.append("stale_hands_on")
            result.concerns.append("18+ months in a non-coding leadership role -- JD says 'this role writes code'")
            base *= cfg.PENALTY_STALE_HANDS_ON

    # 9. Research-title without production proof
    from datetime import date
    recent_jobs = sorted(
        history,
        key=lambda j: parse_date(j.start_date) or date.min,
        reverse=True,
    )[:2]
    for job in recent_jobs:
        job_title = (job.title or "").lower()
        if _contains_any(job_title, _RESEARCH_FLAVORED_TITLES):
            prod_hits = _count_hits(narrative, cfg.PRODUCTION_EVIDENCE_TERMS)
            if prod_hits < cfg.RESEARCH_PROD_MIN_HITS:
                if "research_title_no_prod" not in result.penalties:
                    result.penalties.append("research_title_no_prod")
                    result.concerns.append(
                        "research-flavoured title without strong production-deployment evidence"
                    )
                    base *= cfg.PENALTY_RESEARCH_TITLE_NO_PROD
                break

    return base


# ---------------------------------------------------------------------------
# Public Entry Point
# ---------------------------------------------------------------------------

def compute_structural_score(candidate: Candidate, cfg: Config | None = None) -> StructuralScore:
    """Compute the full structural score for a candidate."""
    cfg = cfg or Config()
    result = StructuralScore()
    profile = candidate.profile
    history = candidate.career_history

    # Build narrative once
    narrative = " ".join(
        [profile.summary or "", profile.headline or ""]
        + [(j.description or "") for j in history]
    ).lower()

    # Compute component scores
    components = {
        "title_domain": _title_domain_score(candidate, result, cfg),
        "career_evidence": _career_evidence_score(candidate, result, narrative, cfg),
        "experience_band": _experience_band_score(candidate, result, cfg),
        "skills_trust": _skills_trust_score(candidate, result, cfg),
        "education_tier": _education_tier_score(candidate, result, cfg),
        "logistics": _logistics_score(candidate, result, cfg),
    }

    base = sum(cfg.STRUCT_WEIGHTS[k] * v for k, v in components.items())
    result.components = components

    # Check for wholly irrelevant careers
    has_tech_title = False
    for j in history:
        j_title = (j.title or "").lower()
        if _contains_any(j_title, cfg.ENGINEERING_TITLE_TERMS) and not _contains_any(j_title, cfg.NON_TECH_TITLE_TERMS):
            has_tech_title = True
            break

    title_lower = (profile.current_title or "").lower()
    is_non_tech = _contains_any(title_lower, cfg.NON_TECH_TITLE_TERMS)
    is_ml_title = _contains_any(title_lower, cfg.ML_TITLE_TERMS) or components["title_domain"] >= 1.0

    if (
        (components["title_domain"] <= 0.05 and components["career_evidence"] <= 0.15)
        or (is_non_tech and not has_tech_title)
        or (not is_ml_title and components["career_evidence"] <= 0.15 and components["skills_trust"] <= 0.05)
    ):
        result.penalties.append("irrelevant_career")
        result.concerns.append("no ML/retrieval background found in title or career history")
        base = min(base, 0.08)

    result.score = max(0.0, min(1.0, _apply_penalties(candidate, base, result, narrative, cfg)))
    return result
