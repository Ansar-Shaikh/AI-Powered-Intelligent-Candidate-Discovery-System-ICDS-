"""
Redrob AI Hiring Intelligence Platform - Exploratory Analysis Notebook

This notebook provides interactive analysis of the candidate dataset,
ranking results, and model performance.

To run as a Jupyter notebook, convert with:
    jupytext --to notebook notebooks/analysis.py

Or run directly as a Python script for data exploration.
"""

# %% [markdown]
# # 🔬 Redrob AI Hiring Intelligence - Exploratory Analysis
# 
# This notebook provides deep insights into:
# - Candidate dataset characteristics
# - Ranking model performance
# - Feature importance analysis
# - Score distribution analysis
# - Behavioral signal patterns

# %%
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from redrob_ranker.config import Config
from redrob_ranker.loading import iter_candidates
from redrob_ranker.data_models import Candidate

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# %% [markdown]
# ## 1. Dataset Overview

# %%
# Load candidates
DATA_PATH = Path("data/candidates.jsonl")
candidates = list(iter_candidates(DATA_PATH))
print(f"Total candidates: {len(candidates):,}")

# %% [markdown]
# ### 1.1 Basic Statistics

# %%
# Extract key metrics
metrics = {
    "total_candidates": len(candidates),
    "avg_skills": np.mean([len(c.skills) for c in candidates]),
    "avg_experience": np.mean([c.profile.years_of_experience for c in candidates]),
    "avg_career_entries": np.mean([len(c.career_history) for c in candidates]),
    "avg_education_entries": np.mean([len(c.education) for c in candidates]),
    "with_summary": sum(1 for c in candidates if c.profile.summary),
    "with_headline": sum(1 for c in candidates if c.profile.headline),
    "open_to_work": sum(1 for c in candidates if c.redrob_signals.open_to_work_flag),
}

for k, v in metrics.items():
    print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

# %% [markdown]
# ### 1.2 Experience Distribution

# %%
yoe_values = [c.profile.years_of_experience for c in candidates]
plt.figure(figsize=(10, 5))
plt.hist(yoe_values, bins=30, edgecolor="black", alpha=0.7)
plt.axvline(5, color="green", linestyle="--", label="Ideal Min (5y)")
plt.axvline(9, color="green", linestyle="--", label="Ideal Max (9y)")
plt.xlabel("Years of Experience")
plt.ylabel("Count")
plt.title("Distribution of Years of Experience")
plt.legend()
plt.show()

# %% [markdown]
# ### 1.3 Skills Analysis

# %%
all_skills = []
for c in candidates:
    all_skills.extend([s.name for s in c.skills])

skill_counts = pd.Series(all_skills).value_counts().head(20)
plt.figure(figsize=(12, 6))
skill_counts.plot(kind="barh")
plt.xlabel("Frequency")
plt.title("Top 20 Skills in Dataset")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 1.4 Location Distribution

# %%
locations = [c.profile.location for c in candidates if c.profile.location]
loc_counts = pd.Series(locations).value_counts().head(15)
plt.figure(figsize=(10, 6))
loc_counts.plot(kind="bar")
plt.xlabel("Location")
plt.ylabel("Count")
plt.title("Top 15 Candidate Locations")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Behavioral Signals Analysis

# %%
response_rates = [c.redrob_signals.recruiter_response_rate for c in candidates 
                  if c.redrob_signals.recruiter_response_rate is not None]

plt.figure(figsize=(10, 5))
plt.hist(response_rates, bins=20, edgecolor="black", alpha=0.7)
plt.xlabel("Recruiter Response Rate")
plt.ylabel("Count")
plt.title("Distribution of Recruiter Response Rates")
plt.show()

# %% [markdown]
# ### 2.1 Profile Completeness

# %%
completeness = [c.redrob_signals.profile_completeness_score for c in candidates 
                if c.redrob_signals.profile_completeness_score is not None]

plt.figure(figsize=(10, 5))
plt.hist(completeness, bins=20, edgecolor="black", alpha=0.7)
plt.xlabel("Profile Completeness Score")
plt.ylabel("Count")
plt.title("Profile Completeness Distribution")
plt.show()

# %% [markdown]
# ## 3. Ranking Results Analysis (if available)

# %%
# Load ranking results if they exist
import csv

SUBMISSION_PATH = Path("output/submission.csv")
if SUBMISSION_PATH.exists():
    with open(SUBMISSION_PATH) as f:
        reader = csv.DictReader(f)
        ranked = list(reader)

    scores = [float(r["score"]) for r in ranked]

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(scores) + 1), scores, marker="o", markersize=3)
    plt.xlabel("Rank")
    plt.ylabel("Score")
    plt.title("Score vs Rank (Top 100)")
    plt.grid(True)
    plt.show()

    print(f"Top score: {max(scores):.4f}")
    print(f"Bottom score: {min(scores):.4f}")
    print(f"Mean score: {np.mean(scores):.4f}")
    print(f"Score std: {np.std(scores):.4f}")
else:
    print("No submission file found. Run the ranking pipeline first.")

# %% [markdown]
# ## 4. Feature Correlation Analysis

# %%
# If we have detailed reports
REPORT_PATH = Path("output/detailed_report.json")
if REPORT_PATH.exists():
    with open(REPORT_PATH) as f:
        report = json.load(f)

    df = pd.DataFrame(report)

    # Correlation matrix
    corr_cols = ["final_score", "semantic", "structural", "behavioral_multiplier", "integrity_multiplier"]
    corr = df[corr_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".3f")
    plt.title("Score Component Correlations")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 5. Honeypot Detection Analysis

# %%
from redrob_ranker.integrity_engine import compute_integrity_report
from redrob_ranker.structural_scorer import compute_structural_score
from redrob_ranker.semantic_scorer import SemanticScorer

# Sample a subset for analysis
sample = candidates[:1000]

integrity_flags = []
for c in sample:
    structural = compute_structural_score(c)
    # Mock semantic score for analysis
    report = compute_integrity_report(c, semantic_score=0.5, structural_score=structural.score)
    integrity_flags.append(len(report.hard_flags) + len(report.soft_flags))

plt.figure(figsize=(10, 5))
plt.hist(integrity_flags, bins=range(0, max(integrity_flags) + 2), edgecolor="black", alpha=0.7)
plt.xlabel("Number of Integrity Flags")
plt.ylabel("Count")
plt.title("Integrity Flag Distribution (Sample of 1000)")
plt.show()

print(f"Candidates with flags: {sum(1 for f in integrity_flags if f > 0)} / {len(sample)}")
print(f"Max flags on single candidate: {max(integrity_flags)}")

# %% [markdown]
# ## 6. Export Analysis Report

# %%
# Generate summary statistics
summary = {
    "dataset": {
        "total_candidates": len(candidates),
        "avg_skills": float(np.mean([len(c.skills) for c in candidates])),
        "avg_experience": float(np.mean([c.profile.years_of_experience for c in candidates])),
        "top_locations": dict(pd.Series([c.profile.location for c in candidates if c.profile.location]).value_counts().head(10)),
        "top_skills": dict(pd.Series(all_skills).value_counts().head(20)),
    },
    "generated_at": pd.Timestamp.now().isoformat(),
}

with open("output/analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print("Analysis complete! Summary saved to output/analysis_summary.json")
