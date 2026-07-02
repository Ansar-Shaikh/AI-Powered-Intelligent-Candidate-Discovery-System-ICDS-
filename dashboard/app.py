
"""
Redrob AI Hiring Intelligence Platform - Streamlit Dashboard
"""
from __future__ import annotations
 
import json
import os
import sys
import tempfile
import time
from pathlib import Path
 
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
 
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
 
import streamlit as st
 
from redrob_ranker.config import Config
from redrob_ranker.data_models import Candidate, FinalScore
from redrob_ranker.loading import iter_candidates, load_job_description
from redrob_ranker.ranking_pipeline import RankingPipeline, write_submission
from redrob_ranker.semantic_scorer import SemanticScorer
from redrob_ranker.structural_scorer import compute_structural_score
from redrob_ranker.behavioral_scorer import compute_behavioral_score
from redrob_ranker.integrity_engine import compute_integrity_report
 
st.set_page_config(
    page_title="Redrob AI Hiring Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
st.markdown("""
<style>
.main-header { font-size:2.2rem; font-weight:700; color:#1f77b4; margin-bottom:0.5rem; }
.sub-header  { font-size:1.1rem; color:#666; margin-bottom:1.5rem; }
.score-high  { color:#28a745; font-weight:bold; }
.score-med   { color:#ffc107; font-weight:bold; }
.score-low   { color:#dc3545; font-weight:bold; }
</style>
""", unsafe_allow_html=True)
 
# ── Session State ──────────────────────────────────────────────────────────
def _init():
    defaults = {
        "candidates_loaded": False,
        "candidates": [],
        "candidate_dict": {},
        "jd_loaded": False,
        "jd_text": "",
        "ranking_complete": False,
        "ranked_scores": [],
        "pipeline": None,
        "config": Config(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
 
_init()
 
# ── Sidebar ────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("## 🎯 Redrob AI Platform")
        st.markdown("---")
        st.markdown("### 📊 System Status")
        c1, c2 = st.columns(2)
        with c1:
            (st.success if st.session_state.candidates_loaded else st.warning)("✅ Data" if st.session_state.candidates_loaded else "⏳ Data")
        with c2:
            (st.success if st.session_state.jd_loaded else st.warning)("✅ JD" if st.session_state.jd_loaded else "⏳ JD")
        if st.session_state.ranking_complete:
            st.success("✅ Ranking Done")
        else:
            st.info("⏳ Not Ranked")
        if st.session_state.candidates_loaded:
            st.markdown(f"**Candidates:** {len(st.session_state.candidates):,}")
        if st.session_state.ranking_complete:
            st.markdown(f"**Ranked:** {len(st.session_state.ranked_scores)}")
        st.markdown("---")
        st.markdown("### 🧭 Navigation")
        page = st.radio("", [
            "🏠 Home", "📁 Data Upload", "📝 Job Description",
            "⚙️ Configuration", "🚀 Run Ranking", "📈 Results Dashboard",
            "👤 Candidate Explorer", "📊 Analytics", "💾 Export",
        ], label_visibility="collapsed")
        st.markdown("---")
        st.caption("Version 1.0.0 | IndiaRuns Challenge 2026")
        return page
 
# ── Home ───────────────────────────────────────────────────────────────────
def page_home():
    st.markdown('<div class="main-header">🎯 Redrob AI Hiring Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Production-grade Candidate Ranking · Semantic Search · Explainable AI</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.info("**🧠 Semantic Scoring**\nTF-IDF semantic similarity between JD and candidate profiles — no keyword dependency.")
    c2.info("**📊 Multi-Signal Ranking**\nTitle taxonomy · Skills trust · Career evidence · Behavioral multiplier · Honeypot detection.")
    c3.info("**🔍 Explainable Results**\nEvery recommendation includes structured reasoning grounded in scored evidence.")
    st.markdown("---")
    st.markdown("## 🚀 Quick Start — follow these steps in order")
    steps = [
        ("📁", "Data Upload", "Loads `data/candidates.jsonl` directly from disk — click **Load Candidates** button."),
        ("📝", "Job Description", "Loads `data/job_description.txt` from disk — click **Load JD** button."),
        ("🚀", "Run Ranking", "Click **Start Ranking** — takes 2–5 minutes for 100K candidates."),
        ("📈", "Results Dashboard", "View top candidates, score breakdown, and reasoning text."),
        ("💾", "Export", "Download `submission.csv` for hackathon submission."),
    ]
    for icon, title, desc in steps:
        st.markdown(f"**{icon} {title}** — {desc}")
 
# ── Data Upload ────────────────────────────────────────────────────────────
def page_data_upload():
    st.markdown("## 📁 Data Upload")
    st.info("Candidates file is loaded directly from disk — no browser upload needed, no file size limit.")
 
    default = str(project_root / "data" / "candidates.jsonl")
    path = st.text_input("Path to candidates.jsonl", value=default)
 
    if st.button("📂 Load Candidates", type="primary", use_container_width=True):
        if not os.path.exists(path):
            st.error(f"❌ File not found: {path}")
            return
        with st.spinner(f"Loading candidates from {path} ..."):
            try:
                candidates = list(iter_candidates(path))
                st.session_state.candidates = candidates
                st.session_state.candidate_dict = {c.candidate_id: c for c in candidates}
                st.session_state.candidates_loaded = True
                st.success(f"✅ Loaded {len(candidates):,} candidates!")
 
                st.markdown("### 📊 Dataset Statistics")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Candidates", f"{len(candidates):,}")
                c2.metric("Avg Skills", f"{np.mean([len(c.skills) for c in candidates]):.1f}")
                c3.metric("Avg Experience", f"{np.mean([c.profile.years_of_experience for c in candidates]):.1f} yrs")
                c4.metric("Unique Locations", len({c.profile.location for c in candidates if c.profile.location}))
 
                with st.expander("Preview first candidate"):
                    s = candidates[0]
                    st.json({"id": s.candidate_id, "title": s.profile.current_title,
                             "yoe": s.profile.years_of_experience, "skills": len(s.skills)})
 
                all_skills = [s.name for c in candidates for s in c.skills]
                if all_skills:
                    top = pd.Series(all_skills).value_counts().head(20)
                    fig = px.bar(x=top.index, y=top.values, title="Top 20 Skills",
                                 labels={"x": "Skill", "y": "Count"})
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ {e}")
                st.exception(e)
 
    elif st.session_state.candidates_loaded:
        st.success(f"✅ {len(st.session_state.candidates):,} candidates already loaded.")
        if st.button("🗑️ Clear"):
            st.session_state.candidates = []
            st.session_state.candidate_dict = {}
            st.session_state.candidates_loaded = False
            st.rerun()
 
# ── Job Description ────────────────────────────────────────────────────────
def page_jd():
    st.markdown("## 📝 Job Description")
 
    default_jd = str(project_root / "data" / "job_description.txt")
    jd_path = st.text_input("Path to job_description.txt", value=default_jd)
 
    if st.button("📂 Load JD from file", type="primary"):
        if not os.path.exists(jd_path):
            st.error(f"❌ File not found: {jd_path}")
        else:
            text = Path(jd_path).read_text(encoding="utf-8", errors="ignore")
            st.session_state.jd_text = text
            st.session_state.jd_loaded = True
            st.success("✅ JD loaded from file!")
 
    st.markdown("**— or paste text directly —**")
    pasted = st.text_area("Paste job description", value=st.session_state.jd_text, height=300)
    if st.button("💾 Save pasted JD"):
        st.session_state.jd_text = pasted
        st.session_state.jd_loaded = True
        st.success("✅ JD saved!")
 
    if st.session_state.jd_loaded and st.session_state.jd_text:
        with st.expander("📋 Current JD (preview)", expanded=False):
            st.text(st.session_state.jd_text[:2000])
        cfg = st.session_state.config
        jd_lower = st.session_state.jd_text.lower()
        matched = [s for s in cfg.JD_SKILLS if s in jd_lower]
        c1, c2, c3 = st.columns(3)
        c1.metric("Skills detected in JD", len(matched))
        c2.metric("Characters", f"{len(st.session_state.jd_text):,}")
        c3.metric("Words", f"{len(st.session_state.jd_text.split()):,}")
        if matched:
            st.markdown("**Detected skills:** " + ", ".join(matched[:20]))
 
# ── Configuration ──────────────────────────────────────────────────────────
def page_config():
    st.markdown("## ⚙️ Configuration")
    cfg = st.session_state.config
    st.markdown("### Score Weights")
    c1, c2 = st.columns(2)
    with c1:
        w_sem = st.slider("Semantic Weight", 0.0, 1.0, cfg.W_SEMANTIC, 0.05)
        w_str = st.slider("Structural Weight", 0.0, 1.0, cfg.W_STRUCTURAL, 0.05)
    with c2:
        st.metric("Sum (should be 1.0)", f"{w_sem + w_str:.2f}")
    if abs(w_sem + w_str - 1.0) > 0.01:
        st.warning("⚠️ Weights don't sum to 1.0")
    if st.button("💾 Save"):
        st.session_state.config = Config(W_SEMANTIC=w_sem, W_STRUCTURAL=w_str)
        st.session_state.ranking_complete = False
        st.success("✅ Saved — re-run ranking to apply.")
    with st.expander("Current config"):
        st.json({"W_SEMANTIC": cfg.W_SEMANTIC, "W_STRUCTURAL": cfg.W_STRUCTURAL,
                 "EXP_IDEAL": [cfg.EXP_IDEAL_LO, cfg.EXP_IDEAL_HI]})
 
# ── Run Ranking ────────────────────────────────────────────────────────────
def page_run():
    st.markdown("## 🚀 Run Ranking Pipeline")
    if not st.session_state.candidates_loaded:
        st.error("❌ Load candidates first → 📁 Data Upload")
        return
    if not st.session_state.jd_loaded:
        st.error("❌ Load job description first → 📝 Job Description")
        return
    st.success(f"✅ Ready — {len(st.session_state.candidates):,} candidates loaded.")
    top_k = st.number_input("Top-K to output", 10, 1000, 100, 10)
    prog = st.progress(0)
    status = st.empty()
    if st.button("▶️ Start Ranking", type="primary", use_container_width=True):
        try:
            t0 = time.time()
            status.text("Initialising pipeline...")
            prog.progress(10)
            pipeline = RankingPipeline(cfg=st.session_state.config, device="cpu")
            status.text("Setting job description...")
            prog.progress(20)
            pipeline.set_jd_text(st.session_state.jd_text)
            status.text(f"Scoring {len(st.session_state.candidates):,} candidates...")
            prog.progress(30)
            scores = pipeline.rank_from_list(st.session_state.candidates, top_k=int(top_k))
            prog.progress(100)
            elapsed = time.time() - t0
            st.session_state.ranked_scores = scores
            st.session_state.ranking_complete = True
            st.session_state.pipeline = pipeline
            status.text(f"Done in {elapsed:.1f}s")
            st.success(f"✅ Ranked {len(st.session_state.candidates):,} candidates in {elapsed:.1f}s — top {len(scores)} selected.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Top Score",  f"{scores[0].final_score:.4f}")
            c2.metric("Last Score", f"{scores[-1].final_score:.4f}")
            c3.metric("Avg Score",  f"{np.mean([s.final_score for s in scores]):.4f}")
            c4.metric("Ranked",     len(scores))
        except Exception as e:
            st.error(f"❌ {e}")
            st.exception(e)
 
# ── Results Dashboard ──────────────────────────────────────────────────────
def page_results():
    st.markdown("## 📈 Results Dashboard")
    if not st.session_state.ranking_complete:
        st.warning("⏳ Run ranking first → 🚀 Run Ranking")
        return
    scores = st.session_state.ranked_scores
 
    st.markdown("### 🏆 Top 10 Candidates")
    df = pd.DataFrame([{
        "Rank": s.rank, "Candidate ID": s.candidate_id,
        "Final Score": round(s.final_score, 4),
        "Semantic": round(s.semantic, 4),
        "Structural": round(s.structural, 4),
        "Behavioral ×": round(s.behavioral_multiplier, 3),
        "Integrity ×": round(s.integrity_multiplier, 3),
    } for s in scores[:10]])
    st.dataframe(df, use_container_width=True, height=380)
 
    st.markdown("### 📊 Score Distribution")
    all_scores = [s.final_score for s in scores]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogram", "Box Plot"))
    fig.add_trace(go.Histogram(x=all_scores, nbinsx=30, marker_color="#1f77b4"), row=1, col=1)
    fig.add_trace(go.Box(y=all_scores, marker_color="#ff7f0e"), row=1, col=2)
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
 
    st.markdown("### 📊 Component Breakdown (Top 50)")
    comp_df = pd.DataFrame([{
        "Rank": s.rank, "Semantic": s.semantic, "Structural": s.structural,
    } for s in scores[:50]])
    fig2 = px.bar(comp_df, x="Rank", y=["Semantic", "Structural"],
                  title="Semantic vs Structural by Rank", barmode="group")
    st.plotly_chart(fig2, use_container_width=True)
 
    st.markdown("### 📝 Reasoning Explorer")
    options = [(s.rank, s.candidate_id) for s in scores[:10]]
    sel = st.selectbox("Select candidate:", options, format_func=lambda x: f"#{x[0]} — {x[1]}")
    if sel:
        score = next((s for s in scores if s.rank == sel[0]), None)
        if score:
            c1, c2, c3 = st.columns(3)
            c1.metric("Final Score", f"{score.final_score:.4f}")
            c2.metric("Semantic",    f"{score.semantic:.4f}")
            c3.metric("Structural",  f"{score.structural:.4f}")
            st.info(score.reasoning)
 
# ── Candidate Explorer ─────────────────────────────────────────────────────
def page_explorer():
    st.markdown("## 👤 Candidate Explorer")
    if not st.session_state.candidates_loaded:
        st.error("❌ Load candidates first")
        return
 
    cid = st.text_input("Enter Candidate ID", placeholder="CAND_0000001")
    if st.session_state.ranking_complete:
        top_ids = [s.candidate_id for s in st.session_state.ranked_scores[:20]]
        picked = st.selectbox("Or pick from top 20:", [""] + top_ids)
        if picked:
            cid = picked
 
    if not cid:
        return
 
    c = st.session_state.candidate_dict.get(cid)
    if not c:
        st.error(f"Candidate {cid} not found")
        return
 
    st.markdown(f"### {c.candidate_id} — {c.profile.current_title or 'N/A'}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Experience", f"{c.profile.years_of_experience:.1f} yrs")
    col2.metric("Skills", len(c.skills))
    col3.metric("Location", c.profile.location or "N/A")
 
    t1, t2, t3, t4 = st.tabs(["Profile", "Career", "Education", "Signals"])
    with t1:
        st.markdown(f"**Headline:** {c.profile.headline or 'N/A'}")
        st.markdown(f"**Summary:** {c.profile.summary or 'N/A'}")
        if c.skills:
            st.dataframe(pd.DataFrame([{
                "Skill": s.name, "Level": s.proficiency,
                "Months": s.duration_months, "Endorsements": s.endorsements
            } for s in c.skills]), use_container_width=True)
    with t2:
        for j in c.career_history:
            with st.container():
                st.markdown(f"**{j.title}** @ {j.company} ({j.duration_months or '?'} mo)")
                st.caption(j.description[:200] if j.description else "")
                st.markdown("---")
    with t3:
        for e in c.education:
            st.markdown(f"**{e.degree}** — {e.institution} (Tier: {e.tier})")
    with t4:
        sig = c.redrob_signals
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Response Rate", f"{sig.recruiter_response_rate or 0:.0%}")
        c2.metric("Profile Views", sig.profile_views_received_30d)
        c3.metric("Open to Work", "✅" if sig.open_to_work_flag else "❌")
        c4.metric("Relocate", "✅" if sig.willing_to_relocate else "❌")
 
    if st.session_state.ranking_complete:
        ranked = next((s for s in st.session_state.ranked_scores if s.candidate_id == cid), None)
        if ranked:
            st.markdown("### 🎯 Score Breakdown")
            fig = go.Figure(go.Scatterpolar(
                r=[ranked.semantic, ranked.structural,
                   ranked.behavioral_multiplier, ranked.integrity_multiplier],
                theta=["Semantic", "Structural", "Behavioral", "Integrity"],
                fill="toself",
            ))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])),
                              title="Score Radar", height=350)
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Final Score", f"{ranked.final_score:.4f}")
            st.info(ranked.reasoning)
 
# ── Analytics ──────────────────────────────────────────────────────────────
def page_analytics():
    st.markdown("## 📊 Analytics")
    if not st.session_state.candidates_loaded:
        st.error("❌ Load candidates first")
        return
    cands = st.session_state.candidates
 
    yoe = [c.profile.years_of_experience for c in cands]
    fig = px.histogram(x=yoe, nbins=30, title="Experience Distribution",
                       labels={"x": "Years", "y": "Count"})
    st.plotly_chart(fig, use_container_width=True)
 
    locs = [c.profile.location for c in cands if c.profile.location]
    if locs:
        top_locs = pd.Series(locs).value_counts().head(15)
        fig2 = px.pie(values=top_locs.values, names=top_locs.index, title="Top 15 Locations")
        st.plotly_chart(fig2, use_container_width=True)
 
    skills = [s.name for c in cands for s in c.skills]
    if skills:
        top_sk = pd.Series(skills).value_counts().head(30)
        fig3 = px.bar(x=top_sk.index, y=top_sk.values, title="Top 30 Skills",
                      labels={"x": "Skill", "y": "Count"})
        fig3.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig3, use_container_width=True)
 
    if st.session_state.ranking_complete:
        sc = st.session_state.ranked_scores
        corr_df = pd.DataFrame([{
            "Final": s.final_score, "Semantic": s.semantic,
            "Structural": s.structural, "Behavioral": s.behavioral_multiplier,
        } for s in sc])
        fig4 = px.imshow(corr_df.corr(), text_auto=True, title="Score Correlations")
        st.plotly_chart(fig4, use_container_width=True)
 
# ── Export ─────────────────────────────────────────────────────────────────
def page_export():
    st.markdown("## 💾 Export")
    if not st.session_state.ranking_complete:
        st.warning("⏳ Run ranking first")
        return
    scores = st.session_state.ranked_scores
 
    # CSV
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        write_submission(scores, tmp.name, include_reasoning=True)
        csv_text = Path(tmp.name).read_text(encoding="utf-8")
        os.unlink(tmp.name)
 
    st.download_button("⬇️ Download submission.csv", data=csv_text,
                       file_name="submission.csv", mime="text/csv",
                       use_container_width=True, type="primary")
 
    # Also save to output/ folder on disk
    out_path = project_root / "output" / "submission.csv"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(csv_text, encoding="utf-8")
    st.success(f"✅ Also saved to {out_path}")
 
    # JSON
    report = [{"rank": s.rank, "candidate_id": s.candidate_id,
                "final_score": round(s.final_score, 6),
                "semantic": round(s.semantic, 6), "structural": round(s.structural, 6),
                "reasoning": s.reasoning} for s in scores]
    st.download_button("⬇️ Download detailed_report.json",
                       data=json.dumps(report, indent=2, ensure_ascii=False),
                       file_name="detailed_report.json", mime="application/json",
                       use_container_width=True)
 
    st.markdown("### 👁️ Preview (top 10)")
    prev_df = pd.DataFrame([{"rank": s.rank, "candidate_id": s.candidate_id,
                               "score": f"{s.final_score:.4f}",
                               "reasoning": s.reasoning[:120] + "..."} for s in scores[:10]])
    st.dataframe(prev_df, use_container_width=True)
 
# ── Main ───────────────────────────────────────────────────────────────────
def main():
    page = sidebar()
    dispatch = {
        "🏠 Home":               page_home,
        "📁 Data Upload":        page_data_upload,
        "📝 Job Description":    page_jd,
        "⚙️ Configuration":      page_config,
        "🚀 Run Ranking":        page_run,
        "📈 Results Dashboard":  page_results,
        "👤 Candidate Explorer": page_explorer,
        "📊 Analytics":          page_analytics,
        "💾 Export":             page_export,
    }
    dispatch.get(page, page_home)()
 
if __name__ == "__main__":
    main()
 