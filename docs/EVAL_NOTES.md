# Evaluation and Ablation Notes: Match Scoring & Recency (Feature 6)

## Skill extraction

| Metric | Value |
| :--- | ---: |
| Resumes evaluated | 20 |
| Total labeled skills | 129 |
| Micro-averaged precision | 0.8543 |
| Micro-averaged recall | 1.0000 |
| Micro-averaged F1 | 0.9214 |
| Macro-averaged precision | 0.8713 |
| Macro-averaged recall | 1.0000 |
| Macro-averaged F1 | 0.9262 |

---

## 1. Document Scope & Status

- **Feature**: Multi-Modal Composite Scoring Engine with Recency (Feature 6)
- **Role**: Member 1 — Scoring Lead
- **Related Dependency**: Member 3 — Evaluation & Ablation Lead
- **Status of Prompt 1.5**: **BLOCKED**
- **Reason**: The repository does not contain an empirical ground-truth benchmark (paired Job Descriptions and resumes with recruiter relevance ratings/rankings). Consequently, empirical ablation experiments have **not** been performed, and tuned weights cannot be computed without fabricating experimental results. Production defaults remain pinned to the verified baseline configuration.

---

## 2. Existing Baseline Configuration (Production Defaults)

The scoring architecture currently implements four presets with conservative, verified engineering defaults:

### Presets and Weight Distribution
| Preset | Skill Coverage ($w_2$) | Semantic Similarity ($w_1$) | Experience Fit ($w_3$) | Education / Recency ($w_4$) | Sum |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `skills_only` (default) | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| `hybrid` | 0.70 | 0.30 | 0.00 | 0.00 | 1.00 |
| `proposal` | 0.40 | 0.40 | 0.10 (Experience) | 0.10 (Education) | 1.00 |
| `composite` | 0.40 | 0.30 | 0.15 (Experience) | 0.15 (Recency) | 1.00 |

### Semantic Similarity Calibration Anchors
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Lower Anchor (`DEFAULT_SEMANTIC_LOW`)**: `0.15` (raw cosine $\le 0.15$ maps to normalized similarity $0.0$)
- **Upper Anchor (`DEFAULT_SEMANTIC_HIGH`)**: `0.65` (raw cosine $\ge 0.65$ maps to normalized similarity $1.0$)
- **Mapping Function**:
  $$\text{sim}_{\text{norm}} = \text{clip}\left(\frac{\cos(\mathbf{u}, \mathbf{v}) - 0.15}{0.65 - 0.15}, 0.0, 1.0\right)$$

### Skill Recency Hyperparameters
- **Half-Life (`half_life`)**: `3.0` years
- **Floor (`floor`)**: `0.05`
- **Neutral / Undated Weight (`RECENCY_UNDATED`)**: `0.50`
- **Decay Function**:
  $$\text{weight} = \max\left(0.05, 0.5^{(\text{age} / 3.0)}\right), \quad \text{age} = \max(0, \text{current\_year} - \text{last\_used\_year})$$

### Evidence Multipliers
- `high_depth`: `1.00`
- `demonstrated`: `0.85`
- `listed`: `0.50`

---

## 3. Verified System Invariants (Measured & Code-Verified)

The following properties have been empirically measured and validated across the automated test suite (`tests/test_scoring.py`, `tests/test_recency.py`, `tests/test_api_endpoints.py`):

1. **Exact Backward Compatibility**: In `skills_only` mode, the score output matches legacy scoring across 100% of stored test cases ($0.80/0.20$ required/preferred weighting, required-only, preferred-only, and empty edge cases).
2. **Weight Normalization**: Preset weights across all four modes sum to $1.0 \pm 10^{-5}$.
3. **Bounded Scores**: Every component score and the composite `final_score` are strictly constrained within $[0.0, 1.0]$.
4. **Recency Discrimination**: Two candidate profiles with identical skill lists produce distinct composite scores when one has recent experience (`2024–Present`) versus stale experience (`2018`).
5. **JD Embedding Single-Pass Caching**: During bulk evaluation (`POST /resume/bulk-analyze`), the job description is encoded exactly once across 50 resumes (measured: 1 JD encode call + 50 resume calls = 51 total calls).
6. **Defensive Model Fallback**: If `SentenceTransformer` fails to load, `compute_scores()` degrades gracefully to `skills_only` and returns HTTP 200.

> [!NOTE]
> The above invariants confirm system correctness, stability, and adherence to design specifications. They do **not** represent empirical proof that the baseline weights are optimal for recruiter ranking quality.

---

## 4. Theoretical Expectations (Hypotheses to Validate in Ablation)

The following hypotheses motivated the multi-modal composite formula and require empirical ablation by Member 3:

1. **Semantic Similarity Trade-off ($w_1 = 0.30$)**: Semantic similarity captures domain relevance that keyword matching misses, but setting $w_1 > 0.40$ may dilute hard skill requirements and reward keyword-stuffed resumes.
2. **Recency Half-Life ($t_{1/2} = 3.0$ years)**: Fast-evolving technologies (e.g., cloud platforms, frameworks) depreciate within 3–4 years. An empirical grid search over $t_{1/2} \in [1.5, 5.0]$ is needed to determine recruiter preference.
3. **Experience Saturation ($w_3 = 0.15$)**: Capping candidate experience at the required years prevents senior over-qualification from dominating the score.

---

## 5. Dataset Audit & Missing Inputs

A comprehensive audit of the repository confirmed the state of available evaluation data:

- **`app/data/resume_dataset.csv`**: Contains labeled resumes categorized by broad job role (e.g., "Data Science", "HR") for training the role classifier. It contains **no** job descriptions and **no** candidate-to-job relevance scores.
- **`tests/eval/resumes/`**: Contains 20 resume files (PDF, DOCX, TXT), but has no paired job postings.
- **`tests/eval/skill_eval_template.csv`**: Contains filenames for the 20 resumes, but the `true_skills` column is unpopulated.
- **Matching Benchmark Data**: **Absent**. There are no benchmark JDs with human-ranked candidate sets.

### Required Inputs from Member 3 or Team Lead
To unblock the empirical ablation study and complete Prompt 1.5:
1. **Benchmark Matching Dataset**:
   - 5–10 representative Job Descriptions across different engineering domains.
   - 10–20 candidate resumes per JD with ground-truth relevance annotations (e.g., 0 = Not relevant, 1 = Partially relevant, 2 = Strong fit, 3 = Perfect match).
2. **Evaluation Metrics**:
   - $\text{NDCG}@5$ and $\text{NDCG}@10$ for ranking quality.
   - Spearman rank correlation ($\rho$) against human recruiter rank order.
3. **Ablation Results**:
   - Optimal $(w_1, w_2, w_3, w_4)$ maximizing $\text{NDCG}$.
   - Optimal $(LOW, HIGH)$ anchors for `all-MiniLM-L6-v2`.
   - Optimal decay half-life $t_{1/2}$.

---

## 6. Implementation Template for Prompt 1.5

Once Member 3 provides the tuned values, Member 1 will apply them using this exact specification:

```python
# app/scoring.py
DEFAULT_SEMANTIC_LOW: float = <TUNED_LOW>   # Old: 0.15
DEFAULT_SEMANTIC_HIGH: float = <TUNED_HIGH> # Old: 0.65

SCORING_CONFIG["composite"] = {
    "skill": <TUNED_W_SKILL>,       # Old: 0.40
    "semantic": <TUNED_W_SEMANTIC>, # Old: 0.30
    "experience": <TUNED_W_EXP>,    # Old: 0.15
    "recency": <TUNED_W_RECENCY>,   # Old: 0.15
}

# app/recency.py
RECENCY_CONFIG["half_life"] = <TUNED_HALF_LIFE> # Old: 3.0
```
