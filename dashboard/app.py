"""Streamlit dashboard interface for the Resume Analyzer application."""

import io
import sys
import json
from pathlib import Path

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration — must be the FIRST Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_BASE = "http://127.0.0.1:8000"

# ---------------------------------------------------------------------------
# CSS — dark-mode polish
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp { background: #0f1117; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #1a1d27; }

    /* Card-style containers */
    .metric-card {
        background: linear-gradient(135deg, #1e2235, #252a3d);
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.4rem 0;
    }
    .metric-card h4 { color: #a0aec0; font-size: 0.8rem; margin: 0; text-transform: uppercase; letter-spacing: 0.1em; }
    .metric-card p  { color: #e2e8f0; font-size: 1.6rem; font-weight: 700; margin: 0.3rem 0 0; }

    /* Skill tags */
    .skill-tag {
        display: inline-block;
        background: #2d3250;
        color: #7c83f7;
        border: 1px solid #4a4f7a;
        border-radius: 6px;
        padding: 3px 10px;
        margin: 3px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .skill-tag.required  { background: #1e3a2f; color: #4ade80; border-color: #2d6a4f; }
    .skill-tag.preferred { background: #2a2200; color: #facc15; border-color: #6b5700; }
    .skill-tag.missing   { background: #3a1e1e; color: #f87171; border-color: #7a2d2d; }
    .skill-tag.matched   { background: #1e3a2f; color: #4ade80; border-color: #2d6a4f; }

    /* Score bar */
    .score-bar-bg {
        background: #1e2235; border-radius: 8px; height: 20px; overflow: hidden; margin: 0.5rem 0;
    }
    .score-bar-fill {
        height: 100%; border-radius: 8px;
        background: linear-gradient(90deg, #4f46e5, #7c83f7);
        transition: width 0.6s ease;
    }

    /* Section headers */
    .section-header {
        color: #7c83f7;
        font-size: 1.05rem;
        font-weight: 700;
        border-left: 3px solid #4f46e5;
        padding-left: 0.7rem;
        margin: 1.2rem 0 0.6rem;
    }
    h1, h2, h3 { color: #e2e8f0 !important; }
    p, li, label { color: #a0aec0; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tags(skills: list[str], css_class: str = "") -> str:
    cls = f"skill-tag {css_class}".strip()
    return "".join(f'<span class="{cls}">{s}</span>' for s in skills)


def _score_bar(score: float, label: str = "") -> None:
    pct = int(score * 100)
    color = "#4ade80" if pct >= 70 else "#facc15" if pct >= 40 else "#f87171"
    st.markdown(
        f"""
        <div style="margin-bottom:0.8rem;">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="color:#a0aec0;font-size:0.85rem;">{label}</span>
            <span style="color:{color};font-weight:700;">{pct}%</span>
          </div>
          <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{pct}%;background:linear-gradient(90deg,{color}88,{color});"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _api_ok() -> bool:
    """Return True if the FastAPI backend is reachable."""
    try:
        r = requests.get(f"{API_BASE}/", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Sidebar — navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📄 Resume Analyzer")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📤 Resume Upload", "📋 JD Analyzer", "🎯 Match & Gap"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # API status indicator
    alive = _api_ok()
    dot   = "🟢" if alive else "🔴"
    msg   = "API Online" if alive else "API Offline — start uvicorn"
    st.markdown(f"{dot} **{msg}**")
    st.caption(f"`{API_BASE}`")

    st.markdown("---")
    st.caption("Resume Analyzer v0.1.0 · Phase 1")


# ===========================================================================
# Pages
# ===========================================================================

# ── Home ────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("# 🚀 AI-Powered Resume Analyzer")
    st.markdown(
        "Upload a resume, paste a job description, and instantly see skill matches, "
        "gaps, and learning recommendations."
    )
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, title, desc in [
        (c1, "📤", "Upload Resume",   "PDF or DOCX"),
        (c2, "🧠", "Extract Skills",  "250+ skill taxonomy"),
        (c3, "🎯", "Match to JD",     "Skill overlap scoring"),
        (c4, "📈", "Gap Analysis",    "Learn what's missing"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card"><h4>{icon} {title}</h4><p style="font-size:0.9rem;color:#a0aec0;">{desc}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown(
        "1. Go to **📤 Resume Upload** and upload your resume.\n"
        "2. Go to **📋 JD Analyzer** and paste a job description.\n"
        "3. Go to **🎯 Match & Gap** to see your score and skill gaps."
    )

    if not alive:
        st.warning(
            "⚠️ The FastAPI backend is not running. Start it with:\n"
            "```powershell\n.venv\\Scripts\\Activate.ps1\nuvicorn app.main:app --reload\n```"
        )


# ── Resume Upload ────────────────────────────────────────────────────────────
elif page == "📤 Resume Upload":
    st.markdown("# 📤 Resume Upload & Parsing")
    st.markdown("Upload your resume (PDF or DOCX) to extract text and skills automatically.")

    uploaded = st.file_uploader("Choose your resume", type=["pdf", "docx"])

    if uploaded:
        with st.spinner("Parsing resume…"):
            try:
                resp = requests.post(
                    f"{API_BASE}/resume/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state["resume_data"] = data
                    st.session_state["resume_text"] = data.get("text_preview", "")
                    st.success(f"✅ Parsed **{uploaded.name}** — {data['char_count']:,} characters")

                    # Metrics row
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f'<div class="metric-card"><h4>🛠 Skills Found</h4><p>{len(data["skills"])}</p></div>', unsafe_allow_html=True)
                    with m2:
                        exp = data["experience_years"]
                        st.markdown(f'<div class="metric-card"><h4>📅 Experience</h4><p>{exp if exp else "—"} yrs</p></div>', unsafe_allow_html=True)
                    with m3:
                        edu = data["education"]
                        deg = edu[0]["degree"] if edu else "—"
                        st.markdown(f'<div class="metric-card"><h4>🎓 Highest Degree</h4><p style="font-size:1rem;">{deg}</p></div>', unsafe_allow_html=True)

                    # Skills
                    st.markdown('<div class="section-header">Extracted Skills</div>', unsafe_allow_html=True)
                    st.markdown(_tags(data["skills"]), unsafe_allow_html=True)

                    # Text preview
                    with st.expander("📝 Extracted text preview"):
                        st.text(data["text_preview"])
                else:
                    st.error(f"API error {resp.status_code}: {resp.json().get('detail', resp.text)}")
            except requests.ConnectionError:
                st.error("❌ Cannot reach the backend. Make sure `uvicorn app.main:app --reload` is running.")


# ── JD Analyzer ──────────────────────────────────────────────────────────────
elif page == "📋 JD Analyzer":
    st.markdown("# 📋 Job Description Analyzer")
    st.markdown("Paste a job description to extract required skills, preferred skills, experience, and education requirements.")

    job_title = st.text_input("Job Title (optional)", placeholder="e.g. Senior Python Developer")
    jd_text   = st.text_area("Job Description", height=280, placeholder="Paste the full job description here…")

    if st.button("🔍 Analyze JD", type="primary"):
        if not jd_text.strip():
            st.warning("Please paste a job description.")
        else:
            with st.spinner("Analyzing…"):
                try:
                    resp = requests.post(
                        f"{API_BASE}/job/analyze",
                        json={"job_title": job_title, "description": jd_text},
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        d = resp.json()
                        st.session_state["jd_data"]    = d
                        st.session_state["jd_text"]    = (job_title + "\n" + jd_text).strip()

                        # Summary row
                        c1, c2, c3, c4 = st.columns(4)
                        with c1: st.markdown(f'<div class="metric-card"><h4>🔴 Required Skills</h4><p>{len(d["required_skills"])}</p></div>', unsafe_allow_html=True)
                        with c2: st.markdown(f'<div class="metric-card"><h4>🟡 Preferred Skills</h4><p>{len(d["preferred_skills"])}</p></div>', unsafe_allow_html=True)
                        with c3: st.markdown(f'<div class="metric-card"><h4>📅 Min Experience</h4><p>{d["min_experience_years"] or "—"} yrs</p></div>', unsafe_allow_html=True)
                        with c4:
                            edu = d["education_requirement"]
                            deg = edu[0]["degree"] if edu else "—"
                            st.markdown(f'<div class="metric-card"><h4>🎓 Education</h4><p style="font-size:0.95rem;">{deg}</p></div>', unsafe_allow_html=True)

                        # Skills
                        col_r, col_p = st.columns(2)
                        with col_r:
                            st.markdown('<div class="section-header">Required Skills</div>', unsafe_allow_html=True)
                            st.markdown(_tags(d["required_skills"], "required"), unsafe_allow_html=True)
                        with col_p:
                            st.markdown('<div class="section-header">Preferred Skills</div>', unsafe_allow_html=True)
                            st.markdown(_tags(d["preferred_skills"], "preferred"), unsafe_allow_html=True)

                        st.markdown(f"**Detected category:** `{d['job_category'] or 'unknown'}`")
                    else:
                        st.error(f"API error {resp.status_code}: {resp.json().get('detail', resp.text)}")
                except requests.ConnectionError:
                    st.error("❌ Cannot reach the backend.")


# ── Match & Gap ──────────────────────────────────────────────────────────────
elif page == "🎯 Match & Gap":
    st.markdown("# 🎯 Resume ↔ Job Match & Skill Gap")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### Your Resume Text")
        resume_text = st.text_area(
            "Resume text",
            value=st.session_state.get("resume_text", ""),
            height=220,
            label_visibility="collapsed",
            placeholder="Paste resume text or upload via the Resume Upload page first…",
        )
    with col_r:
        st.markdown("#### Job Description")
        jd_text = st.text_area(
            "JD text",
            value=st.session_state.get("jd_text", ""),
            height=220,
            label_visibility="collapsed",
            placeholder="Paste job description or analyze via the JD Analyzer page first…",
        )

    if st.button("🎯 Run Match & Gap Analysis", type="primary"):
        if not resume_text.strip() or not jd_text.strip():
            st.warning("Both resume text and job description are required.")
        else:
            with st.spinner("Matching…"):
                try:
                    match_resp = requests.post(
                        f"{API_BASE}/match",
                        json={"resume_text": resume_text, "job_description": jd_text},
                        timeout=30,
                    )
                    gap_resp = requests.get(
                        f"{API_BASE}/skill-gap",
                        params={"resume_text": resume_text, "job_description": jd_text},
                        timeout=30,
                    )
                    rec_resp = requests.get(
                        f"{API_BASE}/recommendations",
                        params={"resume_text": resume_text, "job_description": jd_text},
                        timeout=30,
                    )

                    if match_resp.status_code == 200:
                        m = match_resp.json()
                        g = gap_resp.json()   if gap_resp.status_code   == 200 else {}
                        r = rec_resp.json()   if rec_resp.status_code   == 200 else {}

                        # Score
                        score = m["skill_overlap_score"]
                        st.markdown("---")
                        st.markdown("### Overall Match Score (Skill Overlap)")
                        _score_bar(score, f"Skill Score — {int(score*100)}%")

                        # Breakdown
                        c1, c2, c3 = st.columns(3)
                        with c1: st.markdown(f'<div class="metric-card"><h4>✅ Required Matched</h4><p>{len(m["required_matched"])} / {m["total_required"]}</p></div>', unsafe_allow_html=True)
                        with c2: st.markdown(f'<div class="metric-card"><h4>⭐ Preferred Matched</h4><p>{len(m["preferred_matched"])} / {m["total_preferred"]}</p></div>', unsafe_allow_html=True)
                        with c3: st.markdown(f'<div class="metric-card"><h4>❌ Total Gaps</h4><p>{g.get("gap_count", "—")}</p></div>', unsafe_allow_html=True)

                        # Skill columns
                        st.markdown("---")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.markdown('<div class="section-header">✅ Matched Required</div>', unsafe_allow_html=True)
                            st.markdown(_tags(m["required_matched"], "matched") or "<span style='color:#555'>None</span>", unsafe_allow_html=True)
                        with col_b:
                            st.markdown('<div class="section-header">❌ Missing Required</div>', unsafe_allow_html=True)
                            st.markdown(_tags(g.get("required_gap", []), "missing") or "<span style='color:#555'>None</span>", unsafe_allow_html=True)
                        with col_c:
                            st.markdown('<div class="section-header">📚 Recommended to Learn</div>', unsafe_allow_html=True)
                            st.markdown(_tags(r.get("recommended_to_learn", []), "missing") or "<span style='color:#555'>None</span>", unsafe_allow_html=True)
                    else:
                        st.error(f"Match API error {match_resp.status_code}: {match_resp.text}")
                except requests.ConnectionError:
                    st.error("❌ Cannot reach the backend. Make sure uvicorn is running.")
