# Member 4 notes: features, UI and deployment

## Skill evidence classification
`app/evidence.py` splits the resume into lines, tracks Skills sections (headers and `Skills:` labels), and runs spaCy sentence splitting/lemmas on prose lines.
- `high_depth`: a sentence with a strong action verb and a metric.
- `demonstrated`: a strong verb or a metric (verb preferably governing the skill via the dependency subtree; otherwise same-sentence fallback; with 4+ skills in one sentence only governing verbs count).
- `listed`: only in a Skills section, comma/pipe/slash list, after a `Skills:` label, or a bare mention.
The strongest evidence across all occurrences wins. Years, "5 years of experience" and numbers inside skill names are not metrics. Skills never found in the text are omitted. Keys are canonical lower-case names.

## Multipliers and scoring integration
`EVIDENCE_MULTIPLIERS = {high_depth: 1.0, demonstrated: 0.85, listed: 0.5}`. `skill_multipliers(text, skills=None)` returns `{skill: float}` and has no imports from `app.main`/`app.scoring`, so scoring can import it lazily. `app/scoring.py` already does this in composite mode.

**Heads-up:** `tests/test_scoring.py::test_composite_evidence_unavailable_note_added` expects the `evidence_unavailable` note. Once `app/evidence.py` exists that test will fail and must be updated by Member 1.

## Overlay
| Category | Colour | Rule |
| :--- | :--- | :--- |
| unverified | #f43f5e | skill classified `listed` (highest precedence) |
| required | #10b981 | required JD skill, `high_depth`/`demonstrated` |
| preferred | #f59e0b | preferred JD skill, `high_depth`/`demonstrated` |
| semantic | #3b82f6 | sentence with normalized similarity >= 0.55, no highlighted skill |
Level is per skill (all occurrences share it). Unverified applies to every resume skill classified `listed`, not only JD skills. Max 5 pages, 200 boxes/page, no OCR.

## Endpoints and validation
- `POST /resume/evidence`: 400 empty text, 422 bad body. Response is the skill dict plus `counts` (top-level `"counts"` key).
- `POST /resume/overlay`: 400 for non-PDF name/content, empty `jd_text`, >5 MB; 422 missing fields. Semantic failure sets `semantic_unavailable`.
- `POST /resume/star-rewrite`: existing.
- `POST /resume/anonymize`, `POST /resume/dossier`: **not in the attached backend**; dashboard assumes the contract in the README and shows "Feature not available on this server" on 404.

## Dashboard additions
Evidence chips (star for high depth, tooltips with evidence), optional scoring badges/breakdown/recency tables (only when the API returns them), identity card, "Resume PDF overlay" subview (radio inside Match & Analysis), STAR panel with disclaimer, anonymized screening toggle, dossier downloads, cold-start retry, 60 s first-request timeouts. The STAR panel sits below the analysis block, so clicking it re-runs the page and the analysis cards must be re-run (existing Streamlit behaviour).

## Known limitations
Overlay iframe height assumes ~900 px width; English only; en_core_web_sm parsing errors can misclassify a sentence; overlay `search_for` can miss text split across lines; anonymized mode hides email too.

## Files
New: `app/evidence.py`, `app/overlay.py`, `app/routers_extra.py`, `tests/test_evidence.py`, `tests/test_overlay.py`, `tests/test_cors.py`, `requirements-api.txt`, `requirements-semantic.txt`, `docs/NOTES_features_ui.md`.
Updated: `app/main.py` (CORS line + router line), `dashboard/app.py`, `README.md`.

## Screenshot checklist (capture yourself; none were taken)
1. Match & Analysis with evidence chips. 2. PDF overlay tab with coloured boxes, legend, filters. 3. STAR rewrite panel. 4. Anonymized screening view. 5. Dossier download buttons.

## Testing guide (nothing here has been executed)
Python 3.10-3.12.
```bash
# macOS/Linux
python -m venv .venv && source .venv/bin/activate
# Windows PowerShell:  python -m venv .venv ; .venv\Scripts\Activate.ps1
pip install -r requirements-api.txt pytest httpx requests streamlit
python -m spacy download en_core_web_sm
pip install -r requirements-semantic.txt     # optional, for hybrid/composite
SCORING_MODE=skills_only uvicorn app.main:app --reload     # Windows: set SCORING_MODE=skills_only
API_BASE=http://127.0.0.1:8000 streamlit run dashboard/app.py
pytest tests/test_evidence.py tests/test_overlay.py tests/test_cors.py
pytest
```
Manual checks: `curl` the evidence and overlay examples from the README; in the dashboard verify chips/counts, overlay colours, legend, checkboxes and page selector, STAR expanders, anonymized toggle (expect "Feature not available on this server" until backend exists). Cold start: point `API_BASE` at a dead URL and watch the waiting message, then "API unavailable". CORS: `ALLOWED_ORIGINS=https://example.com` and send an `Origin` header. Missing `en_core_web_sm` is the most common error.
