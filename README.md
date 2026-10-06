# AI-Powered Resume Analysis, Job Matching & Skill Gap Detection System

> An intelligent NLP system that parses resumes, extracts technical skills, matches candidate profiles against job descriptions, and pinpoints missing skills for career growth.

---

## 📌 Overview

The **Resume Analyzer** (`resume-analyzer`) is an end-to-end NLP and machine learning platform designed to streamline hiring and career guidance. The system processes resumes in PDF and DOCX formats, extracts hard and soft skills, normalizes them against standard skill taxonomies, calculates semantic and skill-based match scores against job descriptions, and provides actionable recommendations to bridge skill gaps.

---

## ✨ Features

- **Resume Parsing**: Extracts clean text and table contents from PDF (via PyMuPDF) and DOCX (via python-docx) documents with robust error handling.
- **Skill Extraction**: Identifies technical and domain-specific skills using rule-based and NLP entity extraction patterns.
- **Job Description (JD) Analysis**: Extracts required skills, qualifications, and experience benchmarks from raw job postings.
- **Semantic & Keyword Matching**: Combines rule-based skill overlap with transformer embeddings for deep contextual candidate-job matching.
- **Skill Gap Detection**: Pinpoints missing skills required by a job and categorizes them by priority.
- **Job & Course Recommendations**: Suggests matching job roles and relevant learning paths based on detected candidate gaps.
- **Job Role Recommendation**: Predicts the best-fit job roles for a resume (top 4, with probability %) using a classifier trained on a labelled resume dataset (TF-IDF + Logistic Regression, compared against sentence embeddings + Logistic Regression).
- **Interactive Dashboard**: A user-friendly Streamlit web interface for candidates and recruiters to upload resumes, view match scores, and explore gap reports.

---

## 🛠 Tech Stack

- **Core & Runtime**: Python 3.10+
- **Backend API**: [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/)
- **Natural Language Processing**: [spaCy](https://spacy.io/) (`en_core_web_sm`), [NLTK](https://www.nltk.org/)
- **Machine Learning & Embeddings**: [scikit-learn](https://scikit-learn.org/), [sentence-transformers](https://www.sbert.net/)
- **Document Processing**: [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/), [python-docx](https://python-docx.readthedocs.io/)
- **Data & Storage**: [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), SQLite
- **Frontend / Dashboard**: [Streamlit](https://streamlit.io/)

---

## 📂 Project Structure

```text
resume-analyzer/
├── app/
│   ├── __init__.py           # Makes app a Python package
│   ├── main.py               # FastAPI application entry point and route definitions
│   ├── parser.py             # Document parser for PDF and DOCX extraction
│   ├── preprocess.py         # Text normalization, tokenization, and cleaning
│   ├── skills.py             # Skill extraction and taxonomy lookup engine
│   ├── matcher.py            # Composite resume-to-job matching algorithm
│   ├── recommender.py        # Job-role prediction: recommend_roles()
│   ├── data/                 # Skill taxonomy and datasets
│   │   ├── skills.json       # Canonical skill taxonomy
│   │   └── resume_dataset.csv  # Labelled resumes (Category, Resume_str) for role classifier
│   └── models/
│       └── job_classifier.joblib  # Trained role classifier (created by notebook 03)
├── dashboard/
│   ├── app.py                # Streamlit interactive UI application
│   └── style.css             # Dashboard theme
├── notebooks/                # Jupyter exploration and experimentation notebooks
│   ├── 01_nlp_basics.ipynb
│   ├── 02_embeddings_demo.ipynb
│   └── 03_job_classifier.ipynb   # Trains and compares the job-role classifiers
├── tests/
│   ├── __init__.py           # Package marker for test suite
│   ├── test_parser.py        # Unit tests and CLI runner for parser
│   ├── test_skills.py
│   ├── test_jd_analysis.py
│   ├── test_matcher.py
│   ├── test_upload_flow.py
│   ├── test_recommender.py          # recommend_roles() tests
│   └── test_recommend_endpoint.py   # POST /recommend-roles tests
├── .gitignore                # Git ignore rules for venv, cache, and temp files
├── requirements.txt          # Python project dependencies
└── README.md                 # Project documentation and guide
```

---

## ⚙️ Prerequisites

- **Python**: Version 3.10 to 3.12 installed on your system.
- **Git**: Installed for version control and cloning.

Check your Python version in terminal:
```bash
python --version
```

---

## 🚀 Setup Steps (Windows)

Follow these steps in your preferred terminal:

### Option A: PowerShell

```powershell
# 1. Clone repository (or open existing project folder)
# git clone <your-repo-url>
cd resume-analyzer

# 2. Fix execution policy if virtual environment activation is blocked
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. Create virtual environment
python -m venv .venv

# 4. Activate virtual environment
.venv\Scripts\Activate.ps1

# 5. Upgrade pip
python -m pip install --upgrade pip

# 6. Install all dependencies
pip install -r requirements.txt

# 7. Download spaCy English language model
python -m spacy download en_core_web_sm
```

### Option B: Command Prompt (`cmd.exe`)

```cmd
:: 1. Navigate to project folder
cd resume-analyzer

:: 2. Create virtual environment
python -m venv .venv

:: 3. Activate virtual environment
.venv\Scripts\activate.bat

:: 4. Upgrade pip
python -m pip install --upgrade pip

:: 5. Install all dependencies
pip install -r requirements.txt

:: 6. Download spaCy English language model
python -m spacy download en_core_web_sm
```

---

## 🎯 Job Role Recommendation: One-Time Model Setup

The **Recommended Roles** feature needs a trained model file (`app/models/job_classifier.joblib`). It is created once by running a notebook.

1. **Place the dataset** at `app/data/resume_dataset.csv` (two columns: `Category` = job role, `Resume_str` = resume text).
2. **Train the model.** With the virtual environment activated, open the notebook and run every cell (**Kernel → Restart & Run All**):
   ```bash
   pip install jupyter ipykernel
   jupyter notebook notebooks/03_job_classifier.ipynb
   ```
   The notebook cleans the text, trains **TF-IDF + Logistic Regression** and **sentence embeddings + Logistic Regression**, compares them, and saves the better one to `app/models/job_classifier.joblib`.
   > The first run downloads the `all-MiniLM-L6-v2` model (~90 MB), so an internet connection is needed. Allow 1–3 minutes on CPU.
3. **Restart the backend** after training. The API keeps the model in memory, so a running server won't see a newly trained model until it is restarted.

Once the model exists, the **Recommended Roles** card appears in the dashboard on the **Upload Resume** page (below *Extracted Skills*) and on the **Match & Analysis** page (below the Strengths / Gaps / Learning Priorities cards). Percentages show relative fit among the roles in the dataset, not an absolute score.

---

## 🏃 How to Run

Make sure your virtual environment (`.venv`) is activated before running any of the following commands:

### 1. Run the FastAPI Backend
```bash
uvicorn app.main:app --reload
```
- API Base URL: `http://127.0.0.1:8000`
- Interactive Swagger UI Docs: `http://127.0.0.1:8000/docs`
- Alternative Redoc Docs: `http://127.0.0.1:8000/redoc`

### 2. Run the Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
- Dashboard URL: `http://localhost:8501`

### 3. Run Automated Tests
```bash
# Run all tests using pytest
pytest

# Or run parser unit tests with Python's built-in test runner
python tests/test_parser.py

# Or parse a sample resume directly using the parser CLI runner
python tests/test_parser.py "path/to/sample_resume.pdf"

# Role-recommendation tests only
pytest tests/test_recommender.py tests/test_recommend_endpoint.py
```

> Model-dependent role tests are **skipped automatically** until `app/models/job_classifier.joblib` exists. Run notebook `03_job_classifier.ipynb` first to enable them.

---

## 📡 API Usage Examples

Once the FastAPI server is running (`http://127.0.0.1:8000`), you can test endpoints via Swagger Docs or `curl`:

### 1. Upload & Parse Resume
**`POST /resume/upload`**
```bash
curl -X POST "http://127.0.0.1:8000/resume/upload" \
  -F "file=@sample_resume.pdf"
```

### 2. Analyze Job Description
**`POST /job/analyze`**
```bash
curl -X POST "http://127.0.0.1:8000/job/analyze" \
  -H "Content-Type: application/json" \
  -d '{"job_title": "Python Developer", "description": "Looking for a Python backend engineer with FastAPI, Docker, and PostgreSQL experience."}'
```

### 3. Match Resume Against Job Description
**`POST /match`**
```bash
curl -X POST "http://127.0.0.1:8000/match" \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "Python developer with FastAPI and Docker experience.", "job_description": "Requirements\nPython, FastAPI, Docker, PostgreSQL"}'
```

### 4. Detect Skill Gaps
**`GET /skill-gap`**
```bash
curl -G "http://127.0.0.1:8000/skill-gap" \
  --data-urlencode "resume_text=Python developer with FastAPI experience" \
  --data-urlencode "job_description=Requirements: Python, FastAPI, Docker"
```

### 5. Get Job / Course Recommendations
**`GET /recommendations`**
```bash
curl -G "http://127.0.0.1:8000/recommendations" \
  --data-urlencode "resume_text=Python developer with FastAPI experience"
```

### 6. Recommend Job Roles
**`POST /recommend-roles`**
```bash
curl -X POST "http://127.0.0.1:8000/recommend-roles" \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "Data analyst with 4 years of experience building Power BI dashboards and SQL reports.", "top_k": 4}'
```

Example response (values are illustrative):
```json
{
  "recommendations": [
    {"role": "Data Analyst", "probability_percent": 76.4},
    {"role": "Business Analyst", "probability_percent": 9.1},
    {"role": "Data Science", "probability_percent": 6.2},
    {"role": "Database Administrator", "probability_percent": 3.0}
  ],
  "model": "TF-IDF + LR"
}
```

| Field | Description |
| :--- | :--- |
| `resume_text` | Full resume text (e.g. `extracted_text` returned by `/resume/upload`) |
| `top_k` | Optional, default `4`, allowed range 1–50 |

| Status | Meaning |
| :---: | :--- |
| `200` | Success |
| `400` | `resume_text` is empty |
| `422` | Invalid request body (e.g. `top_k` out of range) |
| `503` | Model not trained yet; run `notebooks/03_job_classifier.ipynb` |

> **Windows tip:** quoting JSON in `cmd.exe` is awkward. Use the Swagger UI at `http://127.0.0.1:8000/docs` instead.

---

## 📊 How the Match Score Works

The total candidate-to-job match score is calculated using a balanced multi-criteria formula:

$$\text{Final Score} = 0.40 \cdot S_{\text{skills}} + 0.40 \cdot S_{\text{semantic}} + 0.10 \cdot S_{\text{experience}} + 0.10 \cdot S_{\text{education}}$$

| Component | Weight | Description |
| :--- | :---: | :--- |
| **Skill Overlap ($S_{\text{skills}}$)** | **40%** | Exact and taxonomy-mapped overlap between candidate skills and required job skills (Jaccard / weighted coverage). |
| **Semantic Similarity ($S_{\text{semantic}}$)** | **40%** | Dense cosine similarity using `sentence-transformers` embeddings between the candidate profile summary and job description text. |
| **Experience Fit ($S_{\text{experience}}$)** | **10%** | Comparison of total years of professional experience against the minimum years required by the posting. |
| **Education Fit ($S_{\text{education}}$)** | **10%** | Degree level matching (e.g. Bachelor's, Master's, PhD) compared to required credentials. |

---

## 📈 Evaluation & Benchmarks

*Placeholder results to be populated during experimental evaluation:*

| Metric | Target Baseline | Achieved Score | Notes |
| :--- | :---: | :---: | :--- |
| **Skill Extraction F1 Score** | $\ge 0.85$ | *TBD* | Evaluated on annotated test resumes |
| **Skill Classifier Accuracy** | $\ge 88\%$ | *TBD* | Multi-class category classification |
| **Matching MAE / Correlation** | Spearman $r \ge 0.75$ | *TBD* | Correlated with human recruiter ratings |

---

## 📋 Development Progress Checklist

- [x] **Phase 0: Environment Setup & Project Architecture**
  - [x] Directory structure and module setup
  - [x] Virtual environment and dependency configuration (`requirements.txt`)
  - [x] spaCy model and environment validation
- [x] **Phase 1: Resume Ingestion & Parsing Engine**
  - [x] PDF text extraction using PyMuPDF
  - [x] DOCX text & table extraction using python-docx
  - [x] Text normalization and error handling
  - [x] Parser unit tests and CLI runner
- [ ] **Phase 2: Text Preprocessing & Skill Extraction**
  - [ ] Text cleaning, lowercasing, stopword removal, tokenization
  - [ ] Skill taxonomy curation (`app/data/`)
  - [ ] spaCy phrase matcher & entity extraction for skills
- [ ] **Phase 3: Job Description Parsing & Scoring Engine**
  - [ ] JD requirement parser
  - [ ] Sentence-transformer semantic embedding integration
  - [ ] Multi-factor match scoring algorithm (`app/matcher.py`)
- [ ] **Phase 4: Skill Gap Analysis & Recommendations**
  - [ ] Missing skill identification and prioritization
  - [ ] Recommended courses and related job roles (`app/recommender.py`)
- [ ] **Phase 5: FastAPI Backend & Endpoints**
  - [ ] Upload, analyze, match, and recommendation API routes
  - [ ] Pydantic validation schemas
- [ ] **Phase 6: Streamlit UI Dashboard**
  - [ ] File upload widget and interactive charts
  - [ ] Match score breakdown visualizer and gap report
- [ ] **Phase 7: Testing, Evaluation & Documentation**
  - [ ] Comprehensive test suite
  - [ ] Precision, Recall, and F1 evaluation
- [x] **Phase 8: Job Role Classifier**
  - [x] EDA, text cleaning, and stratified split (`notebooks/03_job_classifier.ipynb`)
  - [x] TF-IDF + Logistic Regression vs. sentence embeddings + Logistic Regression
  - [x] Best model saved with joblib (`app/models/job_classifier.joblib`)
  - [x] `recommend_roles()` in `app/recommender.py`
  - [x] `POST /recommend-roles` endpoint
  - [x] "Recommended Roles" card in the Streamlit dashboard

---

## 🔧 Troubleshooting

### 1. `Can't find model 'en_core_web_sm'`
- **Cause**: The spaCy model has not been downloaded in your active virtual environment.
- **Fix**: Run:
  ```bash
  python -m spacy download en_core_web_sm
  ```

### 2. PyMuPDF (`fitz`) Installation Issues
- **Cause**: Outdated `pip` or wheel build failure on Windows.
- **Fix**:
  ```bash
  python -m pip install --upgrade pip setuptools wheel
  pip install --upgrade pymupdf
  ```

### 3. PowerShell Script Execution Blocked (`Activate.ps1 cannot be loaded`)
- **Cause**: Windows PowerShell security policy restricts unsigned script execution by default.
- **Fix**: Run this in PowerShell before activating:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```

### 4. `Port 8000 / 8501 already in use`
- **Cause**: Another service or previous server instance is still occupying the port.
- **Fix**:
  - Run Uvicorn on an alternate port:
    ```bash
    uvicorn app.main:app --port 8080 --reload
    ```
  - Run Streamlit on an alternate port:
    ```bash
    streamlit run dashboard/app.py --server.port 8502
    ```

### 5. `Trained model not found ... Run notebooks/03_job_classifier.ipynb`
- **Cause**: `app/models/job_classifier.joblib` doesn't exist yet. The API returns `503` and the dashboard shows a yellow warning instead of the Recommended Roles card.
- **Fix**: Run all cells of `notebooks/03_job_classifier.ipynb` (see *Job Role Recommendation: One-Time Model Setup*), then restart `uvicorn`.

### 6. Recommended Roles card is missing in the dashboard
- **Cause**: The backend isn't running, the model hasn't been trained, or the browser is showing an old version.
- **Fix**: Start `uvicorn app.main:app --reload`, make sure the model file exists, restart Streamlit, and hard-refresh the page (`Ctrl+Shift+R`).

### 7. Roles didn't change after retraining the model
- **Cause**: The API caches the model in memory.
- **Fix**: Restart `uvicorn`.

### 8. `FileNotFoundError: ...resume_dataset.csv` in the notebook
- **Cause**: The dataset isn't at the expected path.
- **Fix**: Copy it to `app/data/resume_dataset.csv` (create the `app/data` folder if needed).

### 9. `InconsistentVersionWarning` or the model fails to load after upgrading scikit-learn
- **Cause**: A `.joblib` file only loads reliably with the scikit-learn version that created it.
- **Fix**: Re-run `notebooks/03_job_classifier.ipynb` in your current environment to regenerate the model.

### 10. Notebook says `No module named 'app'`
- **Cause**: Jupyter was started from the wrong folder.
- **Fix**: Launch Jupyter from the project root (`jupyter notebook`) and open the notebook from `notebooks/`.

---

## 🔮 Future Improvements

- OCR support for scanned image-based PDF resumes (Tesseract / EasyOCR).
- Resume ranking for bulk recruiter uploads (ranking 50+ candidates at once).
- Exportable PDF analysis reports with skill roadmap visualizations.
- Integration with external course APIs (Coursera, edX) for real-time course recommendations.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) (placeholder).