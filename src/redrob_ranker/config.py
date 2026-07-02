"""
Central configuration for the Redrob AI Hiring Intelligence Platform.

All tunable parameters live here so scoring behavior is auditable in one place.
Configuration supports environment variable overrides for production flexibility.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Tuple


@dataclass(frozen=True)
class Config:
    """Immutable configuration. Create once, pass everywhere."""

    # -----------------------------------------------------------------------
    # Determinism
    # -----------------------------------------------------------------------
    REFERENCE_DATE: dt.date = dt.date(2026, 6, 1)

    # -----------------------------------------------------------------------
    # Embedding
    # -----------------------------------------------------------------------
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM: int = 384
    EMBED_BATCH_SIZE: int = 256
    EMBED_CAREER_ENTRIES: int = 3
    EMBED_CAREER_CHARS: int = 300

    # -----------------------------------------------------------------------
    # Score Composition
    # -----------------------------------------------------------------------
    W_SEMANTIC: float = 0.40
    W_STRUCTURAL: float = 0.60

    # -----------------------------------------------------------------------
    # Structural Sub-weights (must sum to 1.0)
    # -----------------------------------------------------------------------
    STRUCT_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "title_domain": 0.30,
        "career_evidence": 0.30,
        "experience_band": 0.15,
        "skills_trust": 0.15,
        "education_tier": 0.05,
        "logistics": 0.05,
    })

    # -----------------------------------------------------------------------
    # Penalties
    # -----------------------------------------------------------------------
    PENALTY_CONSULTING_ONLY: float = 0.25
    PENALTY_RESEARCH_ONLY: float = 0.30
    PENALTY_TITLE_CHASER: float = 0.60
    PENALTY_CV_ONLY: float = 0.50
    PENALTY_STALE_HANDS_ON: float = 0.65
    PENALTY_KEYWORD_STUFFER: float = 0.05
    PENALTY_RESEARCH_TITLE_NO_PROD: float = 0.75
    PENALTY_LANGCHAIN_ONLY: float = 0.35

    RESEARCH_PROD_MIN_HITS: int = 2
    TITLE_CHASER_MIN_ROLES: int = 3
    TITLE_CHASER_MAX_AVG_TENURE_MONTHS: int = 20
    STALE_HANDS_ON_MONTHS: int = 18
    KEYWORD_STUFFER_MIN_JD_SKILLS: int = 4

    # -----------------------------------------------------------------------
    # Experience Band
    # -----------------------------------------------------------------------
    EXP_IDEAL_LO: float = 5.0
    EXP_IDEAL_HI: float = 9.0
    EXP_SOFT_LO: float = 4.0
    EXP_SOFT_HI: float = 11.0
    EXP_HARD_FLOOR: float = 3.0

    # -----------------------------------------------------------------------
    # Behavioral
    # -----------------------------------------------------------------------
    BEHAVIORAL_FLOOR: float = 0.30
    BEHAVIORAL_CEILING: float = 1.15
    BEHAVIORAL_DECAY_LAMBDA: float = 0.0077  # 90-day half-life

    RECRUITER_SAVE_MAX: int = 5
    RECRUITER_SAVE_BONUS: float = 0.01
    PROFILE_VIEWS_THRESHOLD: int = 10
    PROFILE_VIEWS_BONUS: float = 0.01
    APP_SUBMITTED_BONUS: float = 0.02

    # -----------------------------------------------------------------------
    # Integrity / Honeypot
    # -----------------------------------------------------------------------
    INTEGRITY_FATAL: float = 0.02
    INTEGRITY_HARD: float = 0.10
    INTEGRITY_SOFT_DECAY: float = 0.90

    HONEYPOT_YOE_SPAN_SLACK_YEARS: float = 2.0
    HONEYPOT_DURATION_MISMATCH_MONTHS: int = 6
    HONEYPOT_EXPERT_ZERO_DURATION_MIN: int = 3

    # -----------------------------------------------------------------------
    # Contradiction Audit
    # -----------------------------------------------------------------------
    CONTRADICTION_DELTA_THRESHOLD: float = 0.45
    CONTRADICTION_PENALTY: float = 0.20

    # -----------------------------------------------------------------------
    # Logistics
    # -----------------------------------------------------------------------
    LOCATION_PREFERRED: Tuple[str, ...] = ("pune", "noida")
    LOCATION_WELCOME: Tuple[str, ...] = (
        "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram",
        "ghaziabad", "faridabad", "new delhi",
    )
    LOCATION_INDIA_RELOCATE: float = 0.75
    LOCATION_INDIA_NO_RELOCATE: float = 0.55
    LOCATION_ABROAD: float = 0.20

    NOTICE_STEPS: List[Tuple[int, float]] = field(default_factory=lambda: [
        (30, 1.00), (60, 0.85), (90, 0.70)
    ])
    NOTICE_LONG: float = 0.55

    # -----------------------------------------------------------------------
    # Tier-1 Companies
    # -----------------------------------------------------------------------
    TIER_1_COMPANIES: FrozenSet[str] = frozenset({
        "google", "meta", "microsoft", "apple", "amazon", "netflix", "openai",
        "deepmind", "anthropic", "twitter", "linkedin", "uber", "airbnb",
        "stripe", "salesforce", "nvidia",
        "zomato", "swiggy", "paytm", "phonepe", "razorpay", "cred", "meesho",
        "flipkart", "ola", "byju", "unacademy", "freshworks", "zepto", "blinkit",
        "sarvam", "yellow.ai", "observe.ai", "mad street den",
        "lenskart", "urban company", "policybazaar", "groww", "zerodha",
        "navi", "slice", "healthifyme", "browserstack", "cleartax", "chargebee",
        "postman", "hasura", "setu", "cashfree",
    })
    TIER_1_COMPANY_BONUS: float = 0.08

    # -----------------------------------------------------------------------
    # Lexicons
    # -----------------------------------------------------------------------
    JD_SKILLS: FrozenSet[str] = frozenset({
        "embedding", "embeddings", "sentence-transformers", "sentence transformers",
        "bge", "e5", "openai embeddings",
        "vector", "pinecone", "weaviate", "qdrant", "milvus", "faiss",
        "opensearch", "elasticsearch", "bm25",
        "retrieval", "information retrieval", "semantic search", "hybrid search",
        "ranking", "learning to rank", "ltr", "re-ranking", "reranking",
        "recommendation", "recommender", "recsys",
        "nlp", "natural language processing",
        "llm", "large language model", "fine-tuning", "fine-tuning llms",
        "lora", "qlora", "peft", "rag",
        "python", "pytorch", "transformers", "transformer",
        "ndcg", "mrr", "a/b testing", "ab testing", "xgboost",
    })

    RETRIEVAL_EVIDENCE_TERMS: Tuple[str, ...] = (
        "retrieval", "ranking", "search", "recommendation", "recommender",
        "embedding", "vector", "semantic", "relevance", "bm25", "elasticsearch",
        "opensearch", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
        "information retrieval", "learning to rank", "re-rank", "rerank",
        "two-tower", "ndcg", "personalization", "query understanding",
        "matching engine", "candidate generation",
    )

    PRODUCTION_EVIDENCE_TERMS: Tuple[str, ...] = (
        "production", "shipped", "deployed", "launched", "real users", "scale",
        "latency", "a/b", "monitoring", "served", "in prod", "rollout",
    )

    ML_EVIDENCE_TERMS: Tuple[str, ...] = (
        "machine learning", "ml model", "ml pipeline", "deep learning", "pytorch",
        "tensorflow", "fine-tun", "llm", "nlp", "feature engineering",
        "model training", "inference",
    )

    ENGINEERING_TITLE_TERMS: Tuple[str, ...] = (
        "engineer", "developer", "scientist", "ml ", " ml", "machine learning",
        "ai ", " ai", "data scientist", "sde", "swe", "programmer", "architect",
    )

    ML_TITLE_TERMS: Tuple[str, ...] = (
        "machine learning", "ml engineer", "ai engineer", "data scientist",
        "applied scientist", "nlp", "search", "relevance", "recommendation",
        "recommender", "information retrieval", "deep learning", "llm",
    )

    ADJACENT_TITLE_TERMS: Tuple[str, ...] = (
        "data engineer", "backend", "software engineer", "full stack",
        "platform engineer", "sde", "swe",
    )

    NON_TECH_TITLE_TERMS: Tuple[str, ...] = (
        "marketing", "sales", "hr ", "hr manager", "human resources", "recruiter",
        "accountant", "finance", "operations manager", "customer support",
        "business analyst", "project manager", "product manager",
        "graphic designer", "content writer", "civil engineer",
        "mechanical engineer", "teacher", "consultant - business", "legal",
        "administrative", "office manager",
    )

    LEADERSHIP_TITLE_TERMS: Tuple[str, ...] = (
        "head of", "director", "vp ", "vice president", "chief", "cto",
        "engineering manager", "delivery manager", "general manager",
        "solution architect", "enterprise architect", "principal architect",
    )

    HANDS_ON_VERBS: Tuple[str, ...] = (
        "implemented", "built", "wrote", "coded", "developed", "shipped",
        "debugged", "optimized", "refactored",
    )

    CONSULTING_INDUSTRIES: Tuple[str, ...] = ("it services", "consulting", "outsourcing", "bpo")
    CONSULTING_FIRMS: Tuple[str, ...] = (
        "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
        "capgemini", "hcl", "tech mahindra", "mindtree", "ltimindtree", "mphasis",
        "ibm global services", "dxc", "ntt data", "genpact",
    )

    RESEARCH_TITLE_TERMS: Tuple[str, ...] = ("research", "researcher", "postdoc", "phd ")
    RESEARCH_INDUSTRIES: Tuple[str, ...] = ("academic", "research", "education", "university")

    CV_SPEECH_ROBOTICS_TERMS: Tuple[str, ...] = (
        "computer vision", "image classification", "object detection", "opencv",
        "speech recognition", "tts", "asr", "robotics", "slam", "autonomous",
        "image segmentation", "video analytics", "face recognition",
    )

    NLP_IR_TERMS: Tuple[str, ...] = (
        "nlp", "natural language", "text", "retrieval", "search", "ranking",
        "recommendation", "llm", "language model", "embedding", "information retrieval",
    )

    # -----------------------------------------------------------------------
    # Factory method for env overrides
    # -----------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> Config:
        """Create config with environment variable overrides."""
        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            env_val = os.environ.get(f"REDRob_{field_name.upper()}")
            if env_val is not None:
                field_type = cls.__dataclass_fields__[field_name].type
                if field_type == float:
                    kwargs[field_name] = float(env_val)
                elif field_type == int:
                    kwargs[field_name] = int(env_val)
                elif field_type == str:
                    kwargs[field_name] = env_val
        return cls(**kwargs)
