"""Streamlit dashboard interface for the Resume Analyzer application."""

import hashlib
import html as _html
import os
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
# Backend API Resolution (Supports Streamlit Secrets, Environment Vars & Sidebar Input)
# ---------------------------------------------------------------------------
def _resolve_api_base() -> str:
    # 1. Runtime override in session state
    if st.session_state.get("custom_api_base"):
        return st.session_state["custom_api_base"].strip().rstrip("/")
    # 2. Streamlit Cloud Secrets (st.secrets["API_BASE"] or st.secrets["BACKEND_URL"])
    try:
        if hasattr(st, "secrets"):
            if "API_BASE" in st.secrets:
                return str(st.secrets["API_BASE"]).strip().rstrip("/")
            if "BACKEND_URL" in st.secrets:
                return str(st.secrets["BACKEND_URL"]).strip().rstrip("/")
    except Exception:
        pass
    # 3. Environment variable (e.g. from Docker or host)
    env_url = os.getenv("API_BASE") or os.getenv("BACKEND_URL")
    if env_url:
        return env_url.strip().rstrip("/")
    # 4. Local development fallback
    return "http://127.0.0.1:8000"

API_BASE = _resolve_api_base()

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
        r = requests.get(f"{API_BASE}/", timeout=8)
        return r.status_code == 200
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Role recommendations — API call + card renderer
# ---------------------------------------------------------------------------
def _api_error_detail(resp: requests.Response) -> str:
    """Pull a readable message out of a FastAPI error response."""
    try:
        detail = resp.json().get("detail", resp.text)
    except (ValueError, AttributeError):
        return resp.text or f"HTTP {resp.status_code}"
    if isinstance(detail, list):  # pydantic validation errors come as a list of dicts
        detail = "; ".join(str(d.get("msg", d)) if isinstance(d, dict) else str(d) for d in detail)
    return str(detail)


def _fetch_role_recommendations(resume_text: str, top_k: int = 4) -> tuple[dict | None, str | None]:
    """Call POST /recommend-roles. Returns (data, error_message); exactly one is None.

    Successful results are cached per (text, top_k) in session_state, because Streamlit
    re-runs the whole script on every interaction and we don't want to re-predict each time.
    Failures are NOT cached, so the next rerun retries automatically.
    """
    cache = st.session_state.setdefault("role_cache", {})
    key = hashlib.sha1(f"{top_k}|{resume_text}".encode("utf-8")).hexdigest()
    if key in cache:
        return cache[key], None

    try:
        resp = requests.post(
            f"{API_BASE}/recommend-roles",
            json={"resume_text": resume_text, "top_k": top_k},
            timeout=60,  # first call can be slow if the embedding model has to load
        )
    except requests.ConnectionError:
        return None, f"Can't reach the backend at {API_BASE}. Start it with `uvicorn app.main:app --reload`."
    except requests.Timeout:
        return None, "Role prediction timed out. Please try again in a moment."
    except requests.RequestException as exc:
        return None, f"Role prediction failed: {exc}"

    if resp.status_code != 200:
        detail = _api_error_detail(resp)
        if resp.status_code == 503:
            return None, f"Role recommendations are not available yet. {detail}"
        return None, f"Role recommendations failed ({resp.status_code}): {detail}"

    try:
        data = resp.json()
    except ValueError:
        return None, "The backend returned an unexpected response for role recommendations."
    # NOTE: don't write a bare `data["recommendations"]` here. Streamlit "magic" would
    # display it on the page as raw JSON.
    if not isinstance(data, dict) or not isinstance(data.get("recommendations"), list):
        return None, "The backend returned an unexpected response for role recommendations."
        
    cache[key] = data
    return data, None


def _render_role_recommendations(resume_text: str, top_k: int = 4) -> None:
    """Fetch and show the top-k best-fit roles as ranked progress bars.

    Never raises: loading shows a spinner, and any failure becomes a warning so the rest
    of the page (skills, match score, ...) is unaffected.
    """
    if not resume_text or not resume_text.strip():
        return

    with st.spinner("Finding best-fit roles..."):
        data, error = _fetch_role_recommendations(resume_text, top_k)

    if error:
        st.warning(error, icon=":material/warning:")
        return

    roles = data.get("recommendations") or []
    if not roles:
        st.info("No role recommendations were returned for this resume.")
        return

    top_pct = float(roles[0]["probability_percent"])
    if top_pct >= 60:
        badge_cls, verdict = "strong", "Clear fit"
    elif top_pct >= 35:
        badge_cls, verdict = "partial", "Likely fit"
    else:
        badge_cls, verdict = "neutral", "Mixed profile"

    rows = []
    for rank, r in enumerate(roles, start=1):
        pct = max(0.0, min(100.0, float(r["probability_percent"])))
        name = _html.escape(str(r["role"]))
        rows.append(
            f'<div class="role-row{" top" if rank == 1 else ""}">'
            f'<span class="role-rank">{rank}</span>'
            f'<div class="role-main">'
            f'<div class="role-head"><span class="role-name">{name}</span>'
            f'<span class="role-pct">{pct:.1f}%</span></div>'
            f'<div class="role-bar" role="progressbar" aria-label="{name} match" '
            f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{pct:.1f}">'
            f'<div class="role-bar-fill" style="width: {pct:.1f}%;"></div></div>'
            f'</div></div>'
        )

    model_name = _html.escape(str(data.get("model") or "role classifier"))
    H(f"""
    <article class="bento-card fade-up-3" style="margin-top: 0.5rem;">
    <div class="role-card-head">
    <div>
    <span class="section-label">CAREER FIT</span>
    <h2 class="card-title">Recommended Roles</h2>
    </div>
    <div class="score-verdict-badge {badge_cls}">{verdict}</div>
    </div>
    <p class="role-caption">Share of the model's confidence across all roles it knows. Percentages are relative to each other, not an absolute grade.</p>
    <div class="role-list">{"".join(rows)}</div>
    <p class="role-foot">Predicted from resume text by {model_name}. Use as guidance, not a verdict.</p>
    </article>
    """)



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

NAV_PAGES = ["Overview", "Upload Resume", "Job Description", "Match & Analysis", "Bulk Upload & Rank"]
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Overview"

with st.sidebar:
    H("""
    <div class="sidebar-brand">
    <div class="sidebar-brand-row">
    <div class="brand-logo-mark" aria-hidden="true"></div>
    <span class="sidebar-brand-name">Resume Analyzer</span>
    </div>
    <div class="sidebar-brand-sub"></div>
    </div>
    """)

    H('<div class="sidebar-nav-label">WORKSPACE</div>')

    NAV_ICONS = {
        "Overview": ":material/dashboard:",
        "Upload Resume": ":material/upload_file:",
        "Job Description": ":material/description:",
        "Match & Analysis": ":material/analytics:",
        "Bulk Upload & Rank": ":material/group_work:",
    }
    for _p in NAV_PAGES:
        _active = st.session_state["nav_page"] == _p
        if st.button(
            _p,
            key=f"nav_{_p}",
            icon=NAV_ICONS[_p],
            use_container_width=True,
            type="primary" if _active else "secondary",
        ):
            st.session_state["nav_page"] = _p
            st.rerun()
    page = st.session_state["nav_page"]

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

    with st.expander("⚙️ Backend API Config", expanded=(not alive)):
        st.caption("Paste your deployed Render backend URL:")
        custom_url_input = st.text_input(
            "Backend URL",
            value=st.session_state.get("custom_api_base", API_BASE),
            placeholder="https://your-api.onrender.com",
            label_visibility="collapsed",
            key="custom_api_url_input",
        )
        if st.button("🔗 Connect Backend", use_container_width=True, key="save_api_url_btn"):
            if custom_url_input.strip():
                st.session_state["custom_api_base"] = custom_url_input.strip().rstrip("/")
                st.rerun()

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
            f"Backend server is offline at {API_BASE}. If using Render, please paste your backend URL in the sidebar under '⚙️ Backend API Config'."
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
                    full_text = data.get("extracted_text") or data.get("text_preview", "")
                    st.session_state["resume_data"] = data
                    st.session_state["full_resume_text"] = full_text
                    st.session_state["resume_text"] = full_text
                    
                    st.success(f"Parsed {uploaded.name} — {data['char_count']:,} characters processed.")

                    # Summary cards
                    m1, m2, m3, m4 = st.columns(4)
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
                    with m4:
                        single_email = data.get("email")
                        if single_email:
                            import urllib.parse
                            _s = urllib.parse.quote("Interview Opportunity / Job Application")
                            _m_url = f"mailto:{single_email}?subject={_s}"
                            email_box = f'<a href="{_m_url}" target="_blank" style="color: var(--accent-emerald); text-decoration: none; word-break: break-all; font-weight: 500;">✉️ {single_email}</a>'
                        else:
                            email_box = '<span style="color: var(--text-tertiary);">Not detected</span>'
                        H(f"""
                        <article class="bento-card fade-up-4">
                        <span class="section-label">CONTACT EMAIL</span>
                        <div style="font-size: 0.95rem; font-weight: 600; margin-top: 6px; line-height: 1.4;">
                        {email_box}
                        </div>
                        <div style="font-size: 0.8rem; color: var(--text-tertiary); margin-top: 4px;">Direct mailto link</div>
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

                    # Role recommendations (own spinner + graceful error handling)
                    _render_role_recommendations(full_text)

                    with st.expander("Document preview text"):
                        st.text(data.get("extracted_text") or data["text_preview"])

                else:
                    err_msg = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
                    st.error(f"Upload failed ({resp.status_code}): {err_msg}")
            except requests.ConnectionError:
                st.error(f"Cannot connect to backend service at {API_BASE}. If deployed on Render, please make sure the service is running and configure the URL in the sidebar.")

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
        initial_resume = (
            st.session_state.get("full_resume_text")
            or st.session_state.get("resume_data", {}).get("extracted_text")
            or st.session_state.get("resume_text", "")
        )
        resume_text = st.text_area(
            "Resume text",
            value=initial_resume,
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
        stored_full = (
            st.session_state.get("full_resume_text")
            or st.session_state.get("resume_data", {}).get("extracted_text")
            or ""
        )
        text_preview_val = st.session_state.get("resume_data", {}).get("text_preview", "")
        if stored_full and (
            resume_text == stored_full
            or (text_preview_val and resume_text == text_preview_val)
            or (resume_text.endswith("...") and stored_full.startswith(resume_text.rstrip(".").rstrip()))
        ):
            analysis_resume_text = stored_full
        else:
            analysis_resume_text = resume_text if resume_text.strip() else stored_full

        if not analysis_resume_text.strip() or not jd_text.strip():
            st.warning("Both candidate resume text and job specification are required.")
        else:
            with st.spinner("Calculating semantic alignment and identifying competency gaps..."):
                try:
                    match_resp = requests.post(
                        f"{API_BASE}/match",
                        json={"resume_text": analysis_resume_text, "job_description": jd_text},
                        timeout=30,
                    )
                    gap_resp = requests.get(
                        f"{API_BASE}/skill-gap",
                        params={"resume_text": analysis_resume_text, "job_description": jd_text},
                        timeout=30,
                    )
                    rec_resp = requests.get(
                        f"{API_BASE}/recommendations",
                        params={"resume_text": analysis_resume_text, "job_description": jd_text},
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

                        # Role recommendations for the same resume text
                        _render_role_recommendations(analysis_resume_text)

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


# ── Bulk Upload & Rank ───────────────────────────────────────────────────────
elif page == "Bulk Upload & Rank":
    H("""
    <section class="hero-section">
    <span class="hero-label">TALENT POOL INTELLIGENCE</span>
    <h1 class="hero-headline">Bulk Resume Screening & Candidate Ranking</h1>
    <p class="hero-subtext">Upload batches of resumes (PDF, DOCX, TXT) to evaluate against role requirements, detect skill gaps, and instantly shortlist top candidates.</p>
    </section>
    """)

    # --- Section 1: Job Description Input ---
    H('<span class="section-label">STEP 1: TARGET ROLE BENCHMARK</span>')
    bulk_jd_default = st.session_state.get("jd_text", "") or st.session_state.get("bulk_jd_text", "")
    bulk_jd = st.text_area(
        "Target Job Description",
        value=bulk_jd_default,
        height=160,
        placeholder="Paste role description with required and preferred qualifications here...",
        help="All resumes will be evaluated and ranked against these requirements.",
        key="bulk_jd_input",
    )
    st.session_state["bulk_jd_text"] = bulk_jd

    H('<div style="height: 1rem;"></div>')

    # --- Section 2: Bulk File Uploader ---
    H('<span class="section-label">STEP 2: RESUME BATCH UPLOAD (UP TO 200 FILES)</span>')
    bulk_files = st.file_uploader(
        "Upload multiple resumes (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="bulk_file_uploader",
    )

    if bulk_files:
        st.caption(f"📁 **{len(bulk_files)} files selected** for analysis.")

    # --- Section 3: Run Analysis Trigger ---
    col_run, _ = st.columns([1, 2])
    with col_run:
        run_bulk = st.button("🚀 Analyze & Rank Resumes", type="primary", use_container_width=True)

    if run_bulk:
        if not bulk_jd.strip():
            st.warning("⚠️ Please provide a target Job Description before running bulk screening.")
        elif not bulk_files:
            st.warning("⚠️ Please upload at least one resume (PDF, DOCX, or TXT).")
        else:
            with st.status("Screening candidate cohort...", expanded=True) as status:
                st.write(f"Preparing {len(bulk_files)} files...")
                files_payload = [
                    ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                    for f in bulk_files
                ]

                st.write("Submitting batch to FastAPI screening pipeline...")
                try:
                    resp = requests.post(
                        f"{API_BASE}/resume/bulk-analyze",
                        files=files_payload,
                        data={"job_description": bulk_jd},
                        timeout=180,
                    )

                    if resp.status_code == 200:
                        bulk_data = resp.json()
                        st.session_state["bulk_raw_results"] = bulk_data
                        status.update(label="Batch screening completed successfully!", state="complete", expanded=False)
                        st.success(f"Successfully processed {bulk_data['summary']['total_uploaded']} resumes.")
                    else:
                        err_detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
                        status.update(label=f"Analysis failed: {err_detail}", state="error")
                        st.error(f"Bulk analysis error ({resp.status_code}): {err_detail}")

                except requests.ConnectionError:
                    status.update(label="API connection failed", state="error")
                    st.error(f"Cannot connect to backend service at {API_BASE}. Please verify your Render service URL in the sidebar.")
                except Exception as exc:
                    status.update(label=f"Unexpected error: {exc}", state="error")
                    st.error(f"Error during batch screening: {exc}")

    # --- Section 4: Display Results & Summary ---
    if "bulk_raw_results" in st.session_state:
        raw_res = st.session_state["bulk_raw_results"]
        summary = raw_res["summary"]
        all_results = raw_res["results"]

        H('<div style="height: 1.5rem;"></div>')
        H('<span class="section-label">COHORT OVERVIEW</span>')

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            H(f"""
            <article class="bento-card fade-up-1">
            <span class="section-label">UPLOADED</span>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.85rem; font-weight: 600; color: var(--text-primary);">
            {summary['total_uploaded']}
            </div>
            <div style="font-size: 0.8rem; color: var(--text-tertiary); margin-top: 4px;">Total files received</div>
            </article>
            """)
        with s2:
            H(f"""
            <article class="bento-card fade-up-2">
            <span class="section-label">ANALYZED</span>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.85rem; font-weight: 600; color: var(--accent-emerald);">
            {summary['total_analyzed']}
            </div>
            <div style="font-size: 0.8rem; color: var(--text-tertiary); margin-top: 4px;">Successfully parsed</div>
            </article>
            """)
        with s3:
            fail_color = "var(--semantic-rose)" if summary['total_failed'] > 0 else "var(--text-tertiary)"
            H(f"""
            <article class="bento-card fade-up-3">
            <span class="section-label">FAILED / REJECTED</span>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.85rem; font-weight: 600; color: {fail_color};">
            {summary['total_failed']}
            </div>
            <div style="font-size: 0.8rem; color: var(--text-tertiary); margin-top: 4px;">Parse or format issues</div>
            </article>
            """)
        with s4:
            dup_color = "var(--semantic-amber)" if summary['total_duplicates'] > 0 else "var(--text-tertiary)"
            H(f"""
            <article class="bento-card fade-up-4">
            <span class="section-label">DUPLICATES</span>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.85rem; font-weight: 600; color: {dup_color};">
            {summary['total_duplicates']}
            </div>
            <div style="font-size: 0.8rem; color: var(--text-tertiary); margin-top: 4px;">Identical file hashes</div>
            </article>
            """)

        # Per-file processing status expander
        with st.expander("📋 File Intake Status Log (Details & Errors)"):
            for entry in all_results:
                fname = entry.get("filename", "unknown")
                status_val = entry.get("status", "unknown")
                if status_val == "analyzed":
                    st.markdown(f"✅ **{fname}** — Analyzed (Score: **{int(round(entry.get('match_score', 0) * 100))}%**)")
                elif status_val == "duplicate":
                    st.markdown(f"⚠️ **{fname}** — Duplicate ({entry.get('error', 'Already seen')})")
                else:
                    st.markdown(f"❌ **{fname}** — Failed: *{entry.get('error', 'Unknown error')}*")

        # --- Section 5: Top-N Shortlisting & Ranking ---
        analyzed_candidates = [r for r in all_results if r.get("status") == "analyzed"]
        if analyzed_candidates:
            H('<div style="height: 1.5rem;"></div>')
            H('<span class="section-label">SHORTLIST CONTROLS</span>')

            total_avail = len(analyzed_candidates)
            if "top_n_choice" not in st.session_state:
                st.session_state["top_n_choice"] = min(5, total_avail)

            col_preset, col_custom = st.columns([2, 1], gap="medium")
            with col_preset:
                st.write("**How many candidates to shortlist?**")
                p5, p10, p20, p_all = st.columns(4)
                if p5.button("Top 5", use_container_width=True, type="primary" if st.session_state["top_n_choice"] == 5 else "secondary"):
                    st.session_state["top_n_choice"] = 5
                    st.rerun()
                if p10.button("Top 10", use_container_width=True, type="primary" if st.session_state["top_n_choice"] == 10 else "secondary"):
                    st.session_state["top_n_choice"] = 10
                    st.rerun()
                if p20.button("Top 20", use_container_width=True, type="primary" if st.session_state["top_n_choice"] == 20 else "secondary"):
                    st.session_state["top_n_choice"] = 20
                    st.rerun()
                if p_all.button(f"All ({total_avail})", use_container_width=True, type="primary" if st.session_state["top_n_choice"] == total_avail else "secondary"):
                    st.session_state["top_n_choice"] = total_avail
                    st.rerun()

            with col_custom:
                custom_n = st.number_input(
                    "Custom N",
                    min_value=1,
                    max_value=max(total_avail, 1),
                    value=min(st.session_state["top_n_choice"], max(total_avail, 1)),
                    step=1,
                )
                if custom_n != st.session_state["top_n_choice"]:
                    st.session_state["top_n_choice"] = custom_n
                    st.rerun()

            chosen_n = st.session_state["top_n_choice"]

            # Instant re-ranking via API
            try:
                rank_resp = requests.post(
                    f"{API_BASE}/rank",
                    json={"results": all_results, "top_n": chosen_n},
                    timeout=10,
                )
                if rank_resp.status_code == 200:
                    rank_data = rank_resp.json()
                    ranked_list = rank_data.get("ranked", [])
                    notice = rank_data.get("notice")

                    if notice:
                        st.info(f"ℹ️ {notice}")

                    # Export & Action Bar
                    top_bar_l, top_bar_r = st.columns([2, 1])
                    with top_bar_l:
                        H(f"""
                        <h2 class="card-title" style="margin-top: 0.5rem;">
                        Top {len(ranked_list)} Shortlisted Candidates
                        </h2>
                        """)
                    with top_bar_r:
                        try:
                            csv_resp = requests.post(
                                f"{API_BASE}/export/csv",
                                json={"ranked": ranked_list},
                                timeout=10,
                            )
                            if csv_resp.status_code == 200:
                                st.download_button(
                                    label="📥 Export Shortlist (CSV)",
                                    data=csv_resp.content,
                                    file_name="candidate_shortlist.csv",
                                    mime="text/csv",
                                    use_container_width=True,
                                )
                        except Exception:
                            pass

                    # Candidate Cards
                    for cand in ranked_list:
                        rank_num = cand.get("rank", 1)
                        c_name = cand.get("candidate_name") or cand.get("filename", "")
                        score = cand.get("match_score", 0)
                        pct = int(round(score * 100))
                        exp = cand.get("experience_years")
                        exp_str = f"{exp} yrs" if exp else "Not indicated"

                        req_m = cand.get("required_matched", [])
                        req_gap = cand.get("required_missing", [])
                        reason = cand.get("rank_reason", "")
                        email_addr = cand.get("email")

                        score_badge_cls = "strong" if pct >= 70 else ("partial" if pct >= 40 else "weak")

                        prof_val = cand.get("professional_experience_str") or (f"{cand.get('experience_years')} yrs" if cand.get("experience_years") else "Not indicated")
                        free_val = cand.get("freelance_experience_str") or "None"
                        if not cand.get("professional_experience_str") and not cand.get("experience_years") and free_val == "None":
                            prof_val = "Not indicated"
                            free_val = "Not indicated"

                        intern_sub = f" <span style='font-size: 0.72rem; color: var(--text-tertiary);'>(incl. internship)</span>" if cand.get("internship_note") else ""
                        comb_sub = f" <span style='font-size: 0.72rem; color: var(--text-tertiary);'>(Total: {cand.get('combined_experience_str')})</span>" if cand.get("combined_experience_str") else ""

                        if email_addr:
                            import urllib.parse
                            subj = urllib.parse.quote("Interview Opportunity / Application Follow-up")
                            bdy = urllib.parse.quote(
                                f"Dear {c_name},\n\nWe reviewed your application ({cand.get('filename', '')}) and would like to invite you for an interview.\n\nPlease let us know your availability for a call.\n\nBest regards,\nRecruitment Team"
                            )
                            mailto_url = f"mailto:{email_addr}?subject={subj}&body={bdy}"
                            email_html = f"""
                            <a href="{mailto_url}" target="_blank" style="
                                display: inline-flex;
                                align-items: center;
                                gap: 6px;
                                background: rgba(16, 185, 129, 0.12);
                                color: #34D399;
                                border: 1px solid rgba(16, 185, 129, 0.35);
                                padding: 3px 10px;
                                border-radius: 6px;
                                text-decoration: none;
                                font-size: 0.78rem;
                                font-weight: 500;
                            ">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="16" x="2" y="4" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/></svg>
                                <span>{email_addr}</span>
                                <span style="font-size: 0.72rem; color: #A7F3D0; font-weight: 600; margin-left: 4px; padding-left: 6px; border-left: 1px solid rgba(16, 185, 129, 0.35);">✉️ Email Candidate</span>
                            </a>
                            """
                        else:
                            email_html = f"""
                            <span style="
                                display: inline-flex;
                                align-items: center;
                                gap: 5px;
                                color: var(--text-tertiary);
                                font-size: 0.76rem;
                                background: rgba(255, 255, 255, 0.03);
                                border: 1px solid rgba(255, 255, 255, 0.06);
                                padding: 3px 8px;
                                border-radius: 6px;
                            ">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="16" x="2" y="4" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/></svg>
                                No email in resume
                            </span>
                            """

                        H(f"""
                        <article class="bento-card fade-up-1" style="margin-bottom: 1rem; border-left: 4px solid var(--accent-emerald);">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                        <div style="display: flex; align-items: flex-start; gap: 14px;">
                        <div style="
                            font-family: 'JetBrains Mono', monospace;
                            font-size: 1.6rem;
                            font-weight: 700;
                            color: var(--accent-emerald);
                            background: rgba(16, 185, 129, 0.1);
                            padding: 6px 14px;
                            border-radius: 8px;
                            border: 1px solid rgba(16, 185, 129, 0.25);
                            line-height: 1.2;
                        ">
                        #{rank_num}
                        </div>
                        <div>
                        <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                        <h3 style="font-size: 1.15rem; font-weight: 600; color: var(--text-primary); margin: 0;">
                        {c_name}
                        </h3>
                        {email_html}
                        </div>
                        <div style="font-size: 0.8rem; color: var(--text-tertiary); margin-top: 5px;">
                        File: {cand.get('filename', '')} &bull; Professional Experience: <strong style="color: var(--text-secondary);">{prof_val}</strong>{intern_sub} &bull; Freelance Experience: <strong style="color: var(--text-secondary);">{free_val}</strong>{comb_sub}
                        </div>
                        </div>
                        </div>
                        <div style="display: flex; align-items: center; gap: 12px;">
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; color: var(--text-primary);">
                        {pct}%
                        </div>
                        <span class="score-verdict-badge {score_badge_cls}">{score_badge_cls.capitalize()}</span>
                        </div>
                        </div>
                        <div style="font-size: 0.82rem; color: var(--text-secondary); margin: 0.75rem 0 0.5rem 0; font-style: italic;">
                        {reason}
                        </div>
                        <div style="margin-top: 0.75rem;">
                        <div style="font-size: 0.75rem; font-weight: 600; color: var(--text-secondary); margin-bottom: 4px; text-transform: uppercase;">
                        Key Matched Competencies:
                        </div>
                        {_tags(req_m, 'matched')}
                        </div>
                        """)
                        if req_gap:
                            H(f"""
                            <div style="margin-top: 0.5rem;">
                            <div style="font-size: 0.75rem; font-weight: 600; color: var(--semantic-rose); margin-bottom: 4px; text-transform: uppercase;">
                            Missing Requirements:
                            </div>
                            {_tags(req_gap, 'missing')}
                            </div>
                            """)
                        H("</article>")

                else:
                    st.error(f"Ranking endpoint failed: {rank_resp.text}")
            except Exception as exc:
                st.error(f"Error fetching ranked list: {exc}")

    _render_footer()

