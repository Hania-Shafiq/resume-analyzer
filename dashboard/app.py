"""Streamlit dashboard interface for the Resume Analyzer application."""

from pathlib import Path
import re
import requests
import streamlit as st


# ---------------------------------------------------------------------------
# HTML Rendering Helper — strips indentation & blank lines to prevent code blocks
# ---------------------------------------------------------------------------
def H(s: str) -> None:
    s = re.sub(r'^[ \t]+', '', s, flags=re.M)
    s = re.sub(r'\n\s*\n', '\n', s)
    st.markdown(s, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page configuration — must be the FIRST Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Resume Analyzer | ATS Resume Checker and Job Match Score",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_BASE = "http://127.0.0.1:8000"

# ---------------------------------------------------------------------------
# Inject CSS stylesheet
# ---------------------------------------------------------------------------
css_file = Path(__file__).parent / "style.css"
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        H(f"<style>{f.read()}</style>")


# ---------------------------------------------------------------------------
# Isometric SVG Icons (Monochrome Emerald, 44x44px)
# ---------------------------------------------------------------------------
ICON_RESUME = """<svg class="card-icon-isometric" width="44" height="44" viewBox="0 0 48 48" fill="none" aria-hidden="true"><polygon points="12,24 26,16 40,24 26,32" fill="#047857" stroke="rgba(255,255,255,0.15)" stroke-width="1"/><polygon points="12,24 26,32 26,35 12,27" fill="#065F46" /><polygon points="40,24 26,32 26,35 40,27" fill="#044E3A" /><polygon points="12,18 26,10 40,18 26,26" fill="#34D399" stroke="rgba(255,255,255,0.3)" stroke-width="1"/><polygon points="12,18 26,26 26,29 12,21" fill="#10B981" /><polygon points="40,18 26,26 26,29 40,21" fill="#059669" /><line x1="20" y1="16" x2="32" y2="23" stroke="#065F46" stroke-width="1.5" stroke-linecap="round"/><line x1="17" y1="19" x2="27" y2="25" stroke="#065F46" stroke-width="1.5" stroke-linecap="round"/></svg>"""

ICON_JD = """<svg class="card-icon-isometric" width="44" height="44" viewBox="0 0 48 48" fill="none" aria-hidden="true"><polygon points="8,22 24,13 40,22 24,31" fill="#34D399" stroke="rgba(255,255,255,0.3)" stroke-width="1"/><polygon points="8,22 24,31 24,35 8,26" fill="#10B981"/><polygon points="40,22 24,31 24,35 40,26" fill="#059669"/><ellipse cx="26" cy="18" rx="8" ry="5" fill="#059669" fill-opacity="0.3" stroke="#A7F3D0" stroke-width="1.8"/><line x1="32" y1="21" x2="39" y2="26" stroke="#047857" stroke-width="3" stroke-linecap="round"/><line x1="32" y1="21" x2="39" y2="26" stroke="#10B981" stroke-width="1.5" stroke-linecap="round"/></svg>"""

ICON_MATCH = """<svg class="card-icon-isometric" width="44" height="44" viewBox="0 0 48 48" fill="none" aria-hidden="true"><polygon points="24,8 40,17 24,26 8,17" fill="#34D399" stroke="rgba(255,255,255,0.3)" stroke-width="1"/><polygon points="8,17 24,26 24,40 8,31" fill="#10B981" stroke="rgba(255,255,255,0.15)" stroke-width="1"/><polygon points="40,17 24,26 24,40 40,31" fill="#059669" stroke="rgba(255,255,255,0.15)" stroke-width="1"/><ellipse cx="24" cy="17" rx="9" ry="5" fill="none" stroke="#065F46" stroke-width="2"/><path d="M 16 17 A 9 5 0 0 1 31 15" fill="none" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round"/><circle cx="24" cy="17" r="1.5" fill="#FFFFFF"/></svg>"""

ICON_KEYWORD = """<svg class="card-icon-isometric" width="44" height="44" viewBox="0 0 48 48" fill="none" aria-hidden="true"><polygon points="24,9 41,18 28,34 11,25" fill="#34D399" stroke="rgba(255,255,255,0.3)" stroke-width="1"/><polygon points="11,25 28,34 28,39 11,30" fill="#10B981"/><polygon points="41,18 28,34 28,39 41,23" fill="#059669"/><ellipse cx="21" cy="16" rx="3" ry="1.8" fill="#065F46" stroke="rgba(255,255,255,0.4)" stroke-width="1"/><line x1="25" y1="21" x2="35" y2="27" stroke="#047857" stroke-width="1.8" stroke-linecap="round"/></svg>"""

ICON_GAP = """<svg class="card-icon-isometric" width="44" height="44" viewBox="0 0 48 48" fill="none" aria-hidden="true"><polygon points="10,24 24,16 38,24 24,32" fill="#10B981" stroke="rgba(255,255,255,0.2)" stroke-width="1"/><polygon points="10,24 24,32 24,40 10,32" fill="#059669"/><polygon points="38,24 24,32 24,40 38,32" fill="#047857"/><polygon points="18,12 28,6 38,12 28,18" fill="#34D399" stroke="rgba(255,255,255,0.3)" stroke-width="1"/><polygon points="18,12 28,18 28,24 18,18" fill="#10B981"/><polygon points="38,12 28,18 28,24 38,18" fill="#059669"/></svg>"""

ICON_EXP = """<svg class="card-icon-isometric" width="44" height="44" viewBox="0 0 48 48" fill="none" aria-hidden="true"><polygon points="10,27 18,22 26,27 18,32" fill="#34D399" stroke="rgba(255,255,255,0.2)" stroke-width="0.8"/><polygon points="10,27 18,32 18,39 10,34" fill="#10B981"/><polygon points="26,27 18,32 18,39 26,34" fill="#059669"/><polygon points="18,20 26,15 34,20 26,25" fill="#34D399" stroke="rgba(255,255,255,0.25)" stroke-width="0.8"/><polygon points="18,20 26,25 26,34 18,29" fill="#10B981"/><polygon points="34,20 26,25 26,34 34,29" fill="#059669"/><polygon points="26,13 34,8 42,13 34,18" fill="#6EE7B7" stroke="rgba(255,255,255,0.3)" stroke-width="0.8"/><polygon points="26,13 34,18 34,30 26,25" fill="#10B981"/><polygon points="42,13 34,18 34,30 42,25" fill="#059669"/></svg>"""

ICON_RECS = """<svg class="card-icon-isometric" width="44" height="44" viewBox="0 0 48 48" fill="none" aria-hidden="true"><polygon points="10,25 25,17 40,25 25,33" fill="#059669" stroke="rgba(255,255,255,0.2)" stroke-width="1"/><polygon points="10,25 25,33 25,37 10,29" fill="#047857"/><polygon points="40,25 25,33 25,37 40,29" fill="#064E3B"/><polygon points="13,18 26,10 39,18 26,26" fill="#34D399" stroke="rgba(255,255,255,0.3)" stroke-width="1"/><polygon points="13,18 26,26 26,30 13,22" fill="#10B981"/><polygon points="39,18 26,26 26,30 39,22" fill="#059669"/><polygon points="23,24 26,26 29,24 29,29 26,27 23,29" fill="#A7F3D0"/></svg>"""


# ---------------------------------------------------------------------------
# Helpers — Modern SaaS Components
# ---------------------------------------------------------------------------
def _tags(skills: list[str], kind: str = "matched") -> str:
    """Render small 12px pill chips."""
    if not skills:
        return '<span style="color: var(--text-tertiary); font-size: 0.85rem; font-style: italic;">None identified</span>'
    tags_html = "".join(f'<span class="skill-pill {kind}">{s}</span>' for s in skills)
    return f'<div class="skill-pills-wrap">{tags_html}</div>'


def _api_ok() -> bool:
    """Return True if the FastAPI backend is reachable."""
    try:
        r = requests.get(f"{API_BASE}/", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _render_top_bar(alive: bool) -> None:
    """Render slim top bar with logo-mark and API indicator."""
    dot_cls = "online" if alive else "offline"
    dot_txt = "API Connected" if alive else "API Disconnected"
    H(f"""<header class="top-bar">
<div class="brand-wrapper">
<div class="brand-logo-mark" aria-hidden="true"></div>
<span class="brand-name">Resume Analyzer</span>
<span class="brand-badge">v0.1.0</span>
</div>
<div class="status-indicator">
<span class="status-dot-saas {dot_cls}" aria-hidden="true"></span>
<span style="color: var(--text-secondary); font-size: 12px; font-weight: 500;">{dot_txt}</span>
</div>
</header>""")


def _render_score_ring(score: float, matched_req: int, total_req: int, matched_pref: int, total_pref: int) -> None:
    """Render clean SVG circular progress ring with tabular numeral and verdict badge."""
    pct = max(0, min(100, int(round(score * 100))))
    circumference = 339.292
    offset = circumference * (1.0 - (pct / 100.0))

    if pct >= 70:
        badge_cls = "strong"
        verdict = "Strong match"
    elif pct >= 40:
        badge_cls = "partial"
        verdict = "Partial match"
    else:
        badge_cls = "weak"
        verdict = "Weak match"

    H(f"""
    <article class="bento-card score-card-hero fade-up-1" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
    <div>
    <span class="section-label">OVERALL ALIGNMENT</span>
    <h2 class="card-title">Role Match Score</h2>
    </div>
    <div class="score-ring-svg">
    <svg viewBox="0 0 120 120" width="150" height="150" style="transform: rotate(-90deg);" aria-label="Match score progress ring: {pct}%">
    <circle cx="60" cy="60" r="54" fill="none" stroke="rgba(255, 255, 255, 0.08)" stroke-width="6" />
    <circle cx="60" cy="60" r="54" fill="none" stroke="#10B981" stroke-width="6"
    stroke-dasharray="339.29" stroke-dashoffset="{offset:.2f}" stroke-linecap="round"
    class="ring-stroke-animate" style="transition: stroke-dashoffset 0.8s ease;" />
    </svg>
    <div class="score-ring-center">
    <div class="score-tabular-num">{pct}%</div>
    </div>
    </div>
    <div>
    <div class="score-verdict-badge {badge_cls}">{verdict}</div>
    <p style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.85rem; line-height: 1.5;">
    {matched_req} of {total_req} required criteria verified<br/>
    {matched_pref} of {total_pref} preferred criteria met
    </p>
    </div>
    </article>
    """)


def _render_product_preview_stage() -> None:
    """Render 3D product preview stage with pure CSS layered panels and subtle mouse parallax."""
    H("""<div class="stage-wrapper" aria-label="Illustration of a resume, a job description, and a match score">
<div class="stage-glow" aria-hidden="true"></div>
<div class="stage-grid-floor" aria-hidden="true"></div>
<div class="stage-iso-group">
<div class="stage-3d-scene" id="previewStageScene">
<div class="iso-panel panel-resume">
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
<div class="mock-avatar"></div>
<div style="flex: 1;">
<div class="mock-bar" style="width: 65%;"></div>
<div class="mock-bar" style="width: 40%;"></div>
</div>
</div>
<div class="mock-bar heading"></div>
<div class="mock-bar" style="width: 90%;"></div>
<div class="mock-bar" style="width: 82%;"></div>
<div class="mock-bar" style="width: 70%;"></div>
<div class="mock-bar heading"></div>
<div class="mock-bar" style="width: 85%;"></div>
<div class="mock-bar" style="width: 60%;"></div>
<div class="mock-bar" style="width: 75%;"></div>
</div>
<div class="stage-connector conn-1" aria-hidden="true">
<div class="stage-connector-dot" style="left: 0;"></div>
<div class="stage-connector-dot" style="right: 0;"></div>
</div>
<div class="iso-panel panel-jd">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
<span class="mock-chip" style="background: rgba(255,255,255,0.06); color: var(--text-secondary);">ROLE SPEC</span>
<span style="font-size: 10px; color: var(--text-tertiary); font-family: 'JetBrains Mono', monospace;">REQ</span>
</div>
<div class="mock-bar" style="width: 75%; height: 7px; background: rgba(255,255,255,0.25); margin-bottom: 12px;"></div>
<div class="mock-bullet-row">
<div class="mock-bullet-dot"></div>
<div class="mock-bar" style="width: 85%; margin: 0;"></div>
</div>
<div class="mock-bullet-row">
<div class="mock-bullet-dot"></div>
<div class="mock-bar" style="width: 75%; margin: 0;"></div>
</div>
<div class="mock-bullet-row">
<div class="mock-bullet-dot"></div>
<div class="mock-bar" style="width: 65%; margin: 0;"></div>
</div>
<div style="margin-top: 14px; display: flex; gap: 6px; flex-wrap: wrap;">
<span class="mock-chip matched">Python</span>
<span class="mock-chip matched">FastAPI</span>
<span class="mock-chip" style="background: rgba(255,255,255,0.06); color: var(--text-secondary);">Docker</span>
</div>
</div>
<div class="stage-connector conn-2" aria-hidden="true">
<div class="stage-connector-dot" style="left: 0;"></div>
<div class="stage-connector-dot" style="right: 0;"></div>
</div>
<div class="iso-panel panel-match">
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 14px;">
<svg viewBox="0 0 54 54" width="46" height="46" style="transform: rotate(-90deg); flex-shrink: 0;" aria-hidden="true">
<circle cx="27" cy="27" r="22" fill="none" stroke="rgba(255, 255, 255, 0.08)" stroke-width="4.5" />
<circle cx="27" cy="27" r="22" fill="none" stroke="#10B981" stroke-width="4.5" stroke-dasharray="138.2" stroke-dashoffset="19.3" stroke-linecap="round" />
<text x="27" y="32" text-anchor="middle" font-family="'JetBrains Mono', monospace" font-size="12" font-weight="700" fill="#F2F4F8" transform="rotate(90 27 27)">86%</text>
</svg>
<div>
<div style="font-size: 10px; font-weight: 700; color: #10B981; letter-spacing: 0.06em; text-transform: uppercase;">STRONG MATCH</div>
<div style="font-size: 11px; color: var(--text-secondary); margin-top: 2px;">Role Fit Verified</div>
</div>
</div>
<div style="height: 1px; background: var(--border-subtle); margin-bottom: 12px;"></div>
<div style="display: flex; flex-direction: column; gap: 6px;">
<span class="mock-chip matched" style="width: fit-content;">&bull; Python (Verified)</span>
<span class="mock-chip matched" style="width: fit-content;">&bull; FastAPI (Verified)</span>
<span class="mock-chip missing" style="width: fit-content;">&bull; Docker (Missing)</span>
</div>
</div>
</div>
</div>
</div>
<script>
(function() {
var stage = document.querySelector('.stage-wrapper');
var scene = document.getElementById('previewStageScene');
if (!stage || !scene) return;
if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
stage.addEventListener('mousemove', function(e) {
var rect = stage.getBoundingClientRect();
var x = (e.clientX - rect.left) / rect.width - 0.5;
var y = (e.clientY - rect.top) / rect.height - 0.5;
scene.style.transform = 'rotateX(' + (55 - y * 12) + 'deg) rotateZ(' + (-35 + x * 12) + 'deg)';
});
stage.addEventListener('mouseleave', function() {
scene.style.transform = 'rotateX(55deg) rotateZ(-35deg)';
});
})();
</script>""")


def _render_footer() -> None:
    """Render minimal footer."""
    H("""
    <footer class="site-footer">
    <div>Resume Analyzer v0.1.0</div>
    <div>Built for recruiters and job seekers</div>
    </footer>
    """)


# ---------------------------------------------------------------------------
# Sidebar — Clean Linear-style Navigation
# ---------------------------------------------------------------------------
alive = _api_ok()

NAV_PAGES = ["Overview", "Upload Resume", "Job Description", "Match & Analysis"]
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Overview"

with st.sidebar:
    H("""
    <div class="sidebar-brand">
    <div class="sidebar-brand-row">
    <div class="brand-logo-mark" aria-hidden="true"></div>
    <span class="sidebar-brand-name">Resume Analyzer</span>
    </div>
    <div class="sidebar-brand-sub">CANDIDATE INTELLIGENCE</div>
    </div>
    """)

    H('<div class="sidebar-nav-label">WORKSPACE</div>')

    curr_idx = NAV_PAGES.index(st.session_state["nav_page"]) if st.session_state["nav_page"] in NAV_PAGES else 0

    page = st.radio(
        "Navigate",
        NAV_PAGES,
        index=curr_idx,
        label_visibility="collapsed",
    )
    st.session_state["nav_page"] = page

    status_cls = "online" if alive else "offline"
    status_txt = "Service Active" if alive else "Service Offline"
    has_res = bool(st.session_state.get("resume_data") or st.session_state.get("resume_text"))
    has_jd = bool(st.session_state.get("jd_data") or st.session_state.get("jd_text"))
    res_cls = "ready" if has_res else "pending"
    jd_cls = "ready" if has_jd else "pending"
    res_txt = "Ready" if has_res else "Pending"
    jd_txt = "Ready" if has_jd else "Pending"

    H(f"""
    <div class="sidebar-cards">
    <div class="sidebar-card">
    <div class="sidebar-card-status-row">
    <span class="status-dot-saas {status_cls}" aria-hidden="true"></span>
    <span class="sidebar-card-status-txt">{status_txt}</span>
    </div>
    <div class="sidebar-card-url">{API_BASE}</div>
    </div>
    <div class="sidebar-card">
    <div class="sidebar-card-label">WORKBENCH STATE</div>
    <div class="sidebar-card-row">
    <span class="sidebar-card-key">Resume</span>
    <span class="sidebar-card-val {res_cls}">{res_txt}</span>
    </div>
    <div class="sidebar-card-row">
    <span class="sidebar-card-key">Job Description</span>
    <span class="sidebar-card-val {jd_cls}">{jd_txt}</span>
    </div>
    </div>
    </div>
    """)

    H("""
    <div class="sidebar-footer">Build 0.1.0 · Phase 1</div>
    """)


# ===========================================================================
# Top Bar
# ===========================================================================
_render_top_bar(alive)


# ===========================================================================
# Pages
# ===========================================================================

# ── Overview ─────────────────────────────────────────────────────────────────
if page == "Overview":
    # Split hero: text + CTA + trust (left) | 3D stage (right)
    hero_left, hero_right = st.columns([1, 0.9], gap="large")

    with hero_left:
        H("""<section class="hero-section hero-copy">
<span class="hero-label">AI RESUME SCREENING PLATFORM</span>
<div role="heading" aria-level="1" class="hero-title">Resume Analyzer<span class="hero-tagline">Match Resumes to Job Descriptions Instantly</span></div>
<p class="hero-subtext">Upload a PDF or DOCX resume, paste a job description, and get an ATS-style match score, skill gap analysis, and clear improvement recommendations in seconds.</p>
</section>""")
        if st.button("Analyze a Resume", type="primary", key="hero_cta_btn"):
            st.session_state["nav_page"] = "Upload Resume"
            st.rerun()
        H("""<div class="trust-strip">
<span class="trust-item">PDF and DOCX supported</span>
<span class="trust-divider" aria-hidden="true"></span>
<span class="trust-item">Skill and experience extraction</span>
<span class="trust-divider" aria-hidden="true"></span>
<span class="trust-item">Instant match scoring</span>
</div>""")

    with hero_right:
        _render_product_preview_stage()

    # 3 Step Flow with equal heights, connecting line & monochrome isometric icons (ONE contiguous call)
    H(f"""<div class="step-grid-wrapper">
<div class="step-flow-line" aria-hidden="true"></div>
<div class="steps-grid">
<article class="step-card fade-up-1">
<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.85rem;">
<div class="step-dot-badge">
<span class="step-dot" aria-hidden="true"></span>
<span class="section-label" style="margin: 0;">STEP 01</span>
</div>
{ICON_RESUME}
</div>
<h3 class="card-title">Resume Parsing</h3>
<p style="font-size: 0.88rem; color: var(--text-secondary); line-height: 1.55; margin: 0; flex: 1;">
Extract text from PDF and DOCX resumes and turn it into structured skills, tools, and work experience.
</p>
</article>
<article class="step-card fade-up-2">
<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.85rem;">
<div class="step-dot-badge">
<span class="step-dot" aria-hidden="true"></span>
<span class="section-label" style="margin: 0;">STEP 02</span>
</div>
{ICON_JD}
</div>
<h3 class="card-title">Job Description Analysis</h3>
<p style="font-size: 0.88rem; color: var(--text-secondary); line-height: 1.55; margin: 0; flex: 1;">
Identify required skills, preferred qualifications, and minimum experience from any job posting.
</p>
</article>
<article class="step-card fade-up-3">
<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.85rem;">
<div class="step-dot-badge">
<span class="step-dot" aria-hidden="true"></span>
<span class="section-label" style="margin: 0;">STEP 03</span>
</div>
{ICON_MATCH}
</div>
<h3 class="card-title">Match Score and Skill Gap Analysis</h3>
<p style="font-size: 0.88rem; color: var(--text-secondary); line-height: 1.55; margin: 0; flex: 1;">
Get a resume-to-job match score, see missing keywords, and receive targeted learning suggestions.
</p>
</article>
</div>
</div>""")

    # Section H2 & 2x2 Feature Grid (ONE contiguous call)
    H(f"""<h2 class="section-heading-h2">What you get with Resume Analyzer</h2>
<div class="features-grid-2x2">
<article class="feature-tile fade-up-1">
<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
<span class="section-label">ATS KEYWORD MATCH</span>
{ICON_KEYWORD}
</div>
<p style="font-size: 0.9rem; color: var(--text-primary); line-height: 1.55; margin: 0.35rem 0 0 0;">
See which job description keywords your resume already covers.
</p>
</article>
<article class="feature-tile fade-up-2">
<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
<span class="section-label">SKILL GAP DETECTION</span>
{ICON_GAP}
</div>
<p style="font-size: 0.9rem; color: var(--text-primary); line-height: 1.55; margin: 0.35rem 0 0 0;">
Find the missing skills that lower your match score.
</p>
</article>
<article class="feature-tile fade-up-3">
<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
<span class="section-label">EXPERIENCE THRESHOLD CHECK</span>
{ICON_EXP}
</div>
<p style="font-size: 0.9rem; color: var(--text-primary); line-height: 1.55; margin: 0.35rem 0 0 0;">
Compare your years of experience against the role requirements.
</p>
</article>
<article class="feature-tile fade-up-4">
<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
<span class="section-label">LEARNING RECOMMENDATIONS</span>
{ICON_RECS}
</div>
<p style="font-size: 0.9rem; color: var(--text-primary); line-height: 1.55; margin: 0.35rem 0 0 0;">
Get specific skills and topics to close each gap.
</p>
</article>
</div>""")

    # FAQ Section
    H('<h2 class="section-heading-h2">Frequently asked questions</h2>')

    with st.expander("What is an ATS resume checker?"):
        st.write("It compares your resume against a job description the way applicant tracking systems filter candidates, using keywords, skills, and experience.")

    with st.expander("Which resume formats are supported?"):
        st.write("PDF and DOCX files.")

    with st.expander("How is the match score calculated?"):
        st.write("The score compares skills, keywords, and experience found in your resume with those required in the job description.")

    with st.expander("How do I improve my resume match score?"):
        st.write("Add the missing skills and keywords from the gap analysis where they honestly apply, and quantify your experience.")

    if not alive:
        st.warning(
            "Backend server is offline at http://127.0.0.1:8000. Start it with `uvicorn app.main:app --reload` to enable analysis."
        )

    _render_footer()


# ── Upload Resume ───────────────────────────────────────────────────────────
elif page == "Upload Resume":
    H("""
    <section class="hero-section">
    <span class="hero-label">RESUME INTAKE</span>
    <h1 class="hero-headline">Upload Your Resume (PDF or DOCX)</h1>
    <p class="hero-subtext">Extract text from PDF and DOCX resumes and turn it into structured skills, tools, and work experience.</p>
    </section>
    """)

    uploaded = st.file_uploader(
        "Drop your resume here (PDF or DOCX)",
        type=["pdf", "docx"],
        label_visibility="collapsed",
    )

    if uploaded:
        with st.spinner("Processing document..."):
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
                    
                    st.success(f"Parsed {uploaded.name} — {data['char_count']:,} characters processed.")

                    # Summary cards
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        H(f"""
                        <article class="bento-card fade-up-1">
                        <span class="section-label">EXTRACTED SKILLS</span>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.85rem; font-weight: 600; color: var(--accent-emerald);">
                        {len(data["skills"])}
                        </div>
                        <div style="font-size: 0.8rem; color: var(--text-tertiary); margin-top: 4px;">Verified in profile</div>
                        </article>
                        """)
                    with m2:
                        exp = data["experience_years"]
                        exp_str = f"{exp} yrs" if exp else "Not indicated"
                        H(f"""
                        <article class="bento-card fade-up-2">
                        <span class="section-label">EXPERIENCE</span>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.85rem; font-weight: 600; color: var(--text-primary);">
                        {exp_str}
                        </div>
                        <div style="font-size: 0.8rem; color: var(--text-tertiary); margin-top: 4px;">Cumulative duration</div>
                        </article>
                        """)
                    with m3:
                        edu = data["education"]
                        deg = edu[0]["degree"] if edu else "None listed"
                        H(f"""
                        <article class="bento-card fade-up-3">
                        <span class="section-label">HIGHEST DEGREE</span>
                        <div style="font-size: 1.25rem; font-weight: 600; color: var(--text-primary); line-height: 1.4; margin-top: 4px;">
                        {deg}
                        </div>
                        <div style="font-size: 0.8rem; color: var(--text-tertiary); margin-top: 4px;">Academic baseline</div>
                        </article>
                        """)

                    # Skills Card (ONE contiguous call)
                    H(f"""
                    <article class="bento-card fade-up-2" style="margin-top: 0.5rem;">
                    <span class="section-label">CANDIDATE TAXONOMY</span>
                    <h2 class="card-title">Extracted Skills</h2>
                    {_tags(data["skills"], "neutral")}
                    </article>
                    """)

                    with st.expander("Document preview text"):
                        st.text(data["text_preview"])

                else:
                    err_msg = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
                    st.error(f"Upload failed ({resp.status_code}): {err_msg}")
            except requests.ConnectionError:
                st.error("Cannot connect to backend service. Please confirm uvicorn is running on http://127.0.0.1:8000.")

    _render_footer()


# ── Job Description ─────────────────────────────────────────────────────────
elif page == "Job Description":
    H("""
    <section class="hero-section">
    <span class="hero-label">ROLE BENCHMARK</span>
    <h1 class="hero-headline">Add the Job Description</h1>
    <p class="hero-subtext">Identify required skills, preferred qualifications, and minimum experience from any job posting.</p>
    </section>
    """)

    job_title = st.text_input("Job Title (optional)", placeholder="e.g. Senior Backend Engineer")
    jd_text   = st.text_area("Job Description", height=240, placeholder="Paste the complete role description and requirement specifications here...")

    if st.button("Analyze Job Description", type="primary"):
        if not jd_text.strip():
            st.warning("Please provide job description text before analyzing.")
        else:
            with st.spinner("Analyzing criteria..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/job/analyze",
                        json={"job_title": job_title, "description": jd_text},
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        d = resp.json()
                        st.session_state["jd_data"] = d
                        st.session_state["jd_text"] = (job_title + "\n" + jd_text).strip()

                        # Metrics row
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            H(f"""
                            <article class="bento-card fade-up-1">
                            <span class="section-label">REQUIRED SKILLS</span>
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.85rem; font-weight: 600; color: var(--semantic-rose);">
                            {len(d["required_skills"])}
                            </div>
                            </article>
                            """)
                        with c2:
                            H(f"""
                            <article class="bento-card fade-up-2">
                            <span class="section-label">PREFERRED SKILLS</span>
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.85rem; font-weight: 600; color: var(--semantic-amber);">
                            {len(d["preferred_skills"])}
                            </div>
                            </article>
                            """)
                        with c3:
                            min_exp = d["min_experience_years"]
                            min_exp_str = f"{min_exp} yrs" if min_exp else "None"
                            H(f"""
                            <article class="bento-card fade-up-3">
                            <span class="section-label">MIN EXPERIENCE</span>
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.85rem; font-weight: 600; color: var(--text-primary);">
                            {min_exp_str}
                            </div>
                            </article>
                            """)
                        with c4:
                            edu = d["education_requirement"]
                            deg = edu[0]["degree"] if edu else "None"
                            H(f"""
                            <article class="bento-card fade-up-4">
                            <span class="section-label">EDUCATION</span>
                            <div style="font-size: 1.15rem; font-weight: 600; color: var(--text-primary); margin-top: 6px;">
                            {deg}
                            </div>
                            </article>
                            """)

                        col_r, col_p = st.columns(2)
                        with col_r:
                            H(f"""
                            <article class="bento-card fade-up-2" style="min-height: 180px;">
                            <span class="section-label">MANDATORY</span>
                            <h2 class="card-title">Required Qualifications</h2>
                            {_tags(d["required_skills"], "missing")}
                            </article>
                            """)

                        with col_p:
                            H(f"""
                            <article class="bento-card fade-up-3" style="min-height: 180px;">
                            <span class="section-label">SECONDARY</span>
                            <h2 class="card-title">Preferred Criteria</h2>
                            {_tags(d["preferred_skills"], "preferred")}
                            </article>
                            """)

                        cat = d.get('job_category') or 'Unclassified'
                        H(f"""
                        <div style="font-size: 12px; color: var(--text-secondary); margin-top: 0.5rem;">
                        Category classification: <strong style="color: var(--text-primary);">{cat}</strong>
                        </div>
                        """)
                    else:
                        err_msg = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
                        st.error(f"Analysis error {resp.status_code}: {err_msg}")
                except requests.ConnectionError:
                    st.error("Cannot connect to backend service. Please confirm uvicorn is running on http://127.0.0.1:8000.")

    _render_footer()


# ── Match & Analysis ────────────────────────────────────────────────────────
elif page == "Match & Analysis":
    H("""
    <section class="hero-section">
    <span class="hero-label">BENCHMARK EVALUATION</span>
    <h1 class="hero-headline">Resume Match Score and Gap Analysis</h1>
    <p class="hero-subtext">Get a resume-to-job match score, see missing keywords, and receive targeted learning suggestions.</p>
    </section>
    """)

    col_l, col_r = st.columns(2)
    with col_l:
        H('<span class="section-label">CANDIDATE SOURCE</span>')
        resume_text = st.text_area(
            "Resume text",
            value=st.session_state.get("resume_text", ""),
            height=180,
            label_visibility="collapsed",
            placeholder="Paste candidate resume text or upload via Upload page...",
        )
    with col_r:
        H('<span class="section-label">ROLE BENCHMARK</span>')
        jd_text = st.text_area(
            "JD text",
            value=st.session_state.get("jd_text", ""),
            height=180,
            label_visibility="collapsed",
            placeholder="Paste job description or analyze via Job Description page...",
        )

    if st.button("Run Match & Gap Analysis", type="primary"):
        if not resume_text.strip() or not jd_text.strip():
            st.warning("Both candidate resume text and job specification are required.")
        else:
            with st.spinner("Calculating semantic alignment and identifying competency gaps..."):
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
                        g = gap_resp.json() if gap_resp.status_code == 200 else {}
                        r = rec_resp.json() if rec_resp.status_code == 200 else {}

                        score = m["skill_overlap_score"]
                        req_matched = m.get("required_matched", [])
                        total_req = m.get("total_required", len(req_matched))
                        pref_matched = m.get("preferred_matched", [])
                        total_pref = m.get("total_preferred", len(pref_matched))
                        req_missing = g.get("required_gap", m.get("required_missing", []))
                        pref_missing = g.get("preferred_gap", m.get("preferred_missing", []))
                        recs = r.get("recommended_to_learn", [])
                        total_gaps = g.get("gap_count", len(req_missing) + len(pref_missing))

                        H('<div style="height: 1.5rem;"></div>')

                        # Clean Bento Grid:
                        # Row 1: Score Card (left, tall) + Summary & Skills (right)
                        bento_left, bento_right = st.columns([4.2, 7.8], gap="large")

                        with bento_left:
                            _render_score_ring(score, len(req_matched), total_req, len(pref_matched), total_pref)

                        with bento_right:
                            H(f"""
                            <article class="bento-card fade-up-2" style="height: 100%;">
                            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                            <div>
                            <span class="section-label">OVERVIEW SUMMARY</span>
                            <h2 class="card-title">Evaluation Breakdown</h2>
                            </div>
                            <div style="font-size: 12px; color: var(--text-secondary);">
                            <span style="font-weight: 600; color: var(--accent-emerald);">{len(req_matched)}</span> / {total_req} criteria
                            </div>
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 1rem 0;">
                            <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 0.85rem;">
                            <span class="section-label">MANDATORY MET</span>
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; font-weight: 600; color: var(--accent-emerald);">
                            {len(req_matched)}
                            </div>
                            </div>
                            <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 0.85rem;">
                            <span class="section-label">PREFERRED MET</span>
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; font-weight: 600; color: var(--semantic-amber);">
                            {len(pref_matched)}
                            </div>
                            </div>
                            <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 0.85rem;">
                            <span class="section-label">TOTAL GAPS</span>
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; font-weight: 600; color: var(--semantic-rose);">
                            {total_gaps}
                            </div>
                            </div>
                            </div>
                            <span class="section-label" style="margin-top: 0.75rem;">VERIFIED CANDIDATE SKILLS</span>
                            {_tags(req_matched, "matched")}
                            </article>
                            """)

                        # Row 2: Strengths / Gaps / Suggestions as three equal cards below
                        H('<div style="height: 0.75rem;"></div>')
                        c_str, c_gap, c_sug = st.columns(3, gap="medium")

                        with c_str:
                            H(f"""
                            <article class="bento-card fade-up-3" style="min-height: 240px;">
                            <span class="section-label" style="color: var(--accent-emerald);">VERIFIED STRENGTHS</span>
                            <h3 class="card-title">Required Met ({len(req_matched)})</h3>
                            <p style="font-size: 0.82rem; color: var(--text-secondary); margin-bottom: 0.75rem;">
                            Mandatory competencies documented in candidate background:
                            </p>
                            {_tags(req_matched, "matched")}
                            </article>
                            """)

                        with c_gap:
                            H(f"""
                            <article class="bento-card fade-up-4" style="min-height: 240px;">
                            <span class="section-label" style="color: var(--semantic-rose);">CRITICAL GAPS</span>
                            <h3 class="card-title">Missing Required ({len(req_missing)})</h3>
                            <p style="font-size: 0.82rem; color: var(--text-secondary); margin-bottom: 0.75rem;">
                            Essential prerequisites absent from candidate experience:
                            </p>
                            {_tags(req_missing, "missing")}
                            </article>
                            """)

                        with c_sug:
                            H(f"""
                            <article class="bento-card fade-up-5" style="min-height: 240px;">
                            <span class="section-label" style="color: var(--semantic-amber);">RECOMMENDED SKILLS</span>
                            <h3 class="card-title">Learning Priorities ({len(recs)})</h3>
                            <p style="font-size: 0.82rem; color: var(--text-secondary); margin-bottom: 0.75rem;">
                            Strategic proficiencies to quickly close the qualification deficit:
                            </p>
                            {_tags(recs, "preferred")}
                            </article>
                            """)

                        # Preferred breakdown expander
                        if pref_matched or pref_missing:
                            with st.expander("Secondary & Preferred Criteria Breakdown"):
                                col_pm, col_pg = st.columns(2)
                                with col_pm:
                                    H(f"""
                                    <div style="font-size: 12px; font-weight: 600; color: var(--text-secondary); margin-bottom: 6px;">MATCHED PREFERRED ({len(pref_matched)})</div>
                                    {_tags(pref_matched, "matched")}
                                    """)
                                with col_pg:
                                    H(f"""
                                    <div style="font-size: 12px; font-weight: 600; color: var(--text-secondary); margin-bottom: 6px;">MISSING PREFERRED ({len(pref_missing)})</div>
                                    {_tags(pref_missing, "missing")}
                                    """)

                    else:
                        st.error(f"Match API returned error {match_resp.status_code}: {match_resp.text}")
                except requests.ConnectionError:
                    st.error("Cannot connect to backend service. Please confirm uvicorn is running on http://127.0.0.1:8000.")

    _render_footer()
