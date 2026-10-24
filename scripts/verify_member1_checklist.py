import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.scoring import compute_scores, SCORING_CONFIG, DEFAULT_SEMANTIC_LOW, DEFAULT_SEMANTIC_HIGH
from app.skills import analyze_job_description, extract_skills

print("=" * 80)
print("  MEMBER 1 VERIFICATION: ALL 4 SCORING MODES & CRITERIA COMPARISON")
print("=" * 80)

# Standard Benchmark JD
test_jd_text = """
Senior Full-Stack Cloud Engineer
Requirements:
- 3+ years of professional backend engineering experience
- Bachelor's degree in Computer Science, Software Engineering or related
- Python, FastAPI, Docker, PostgreSQL

Preferred:
- Kubernetes, AWS
"""

# Benchmark Candidate A (Standard qualified candidate: 3 years full-time, Bachelor's degree)
candidate_std = """
Alex Mercer
Email: alex@example.com

EDUCATION
Bachelor of Science in Computer Science, Tech University, 2023

EXPERIENCE
Software Engineer | CloudScale Inc.
2023 - Present (3 years)
- Built high-throughput microservices using Python and FastAPI.
- Deployed containerized applications with Docker and managed PostgreSQL databases.
- Automated deployments to Kubernetes clusters.
"""

jd_analysis = analyze_job_description(test_jd_text)
skills = extract_skills(candidate_std)
profile_std = {
    "experience_years": 3.0,
    "professional_months": 36,
    "education": [{"degree": "Bachelor", "field": "Computer Science"}]
}

# ---------------------------------------------------------------------------
# 1. Compare all four modes using the EXACT same candidate and JD
# ---------------------------------------------------------------------------
print("\n[PART 1: 4-MODE COMPARISON ON IDENTICAL CANDIDATE & JOB DESCRIPTION]")
print(f"Candidate Skills Found: {skills}")
print(f"JD Required Skills:     {jd_analysis.required_skills}")
print(f"JD Preferred Skills:    {jd_analysis.preferred_skills}")
print(f"JD Required Experience: {jd_analysis.min_experience_years} years\n")

results_by_mode = {}
for mode_name in ["skills_only", "hybrid", "proposal", "composite"]:
    res = compute_scores(
        resume_text=candidate_std,
        resume_skills=skills,
        jd_analysis=jd_analysis,
        resume_profile=profile_std,
        mode=mode_name,
    )
    results_by_mode[mode_name] = res

# Formatted Table
header = f"{'Metric / Component':<24} | {'skills_only':<11} | {'hybrid':<11} | {'proposal':<11} | {'composite':<11}"
print(header)
print("-" * len(header))

metrics = [
    ("Final Score", lambda r: f"{r['final_score']*100:.1f}%"),
    ("Skill Overlap Score", lambda r: f"{r['skill_overlap_score']*100:.1f}%"),
    ("Skill Weight", lambda r: f"{r['breakdown']['skill']['weight']:.2f}"),
    ("Skill Contribution", lambda r: f"{r['breakdown']['skill']['weighted_contribution']:.3f}"),
    ("Semantic Similarity", lambda r: f"{r['semantic_similarity']*100:.1f}%" if r['semantic_similarity'] is not None else "N/A"),
    ("Semantic Weight", lambda r: f"{r['breakdown']['semantic']['weight']:.2f}"),
    ("Semantic Contribution", lambda r: f"{r['breakdown']['semantic']['weighted_contribution']:.3f}"),
    ("Experience Fit", lambda r: f"{r['breakdown']['experience']['value']:.2f}"),
    ("Experience Weight", lambda r: f"{r['breakdown']['experience']['weight']:.2f}"),
    ("Experience Contribution", lambda r: f"{r['breakdown']['experience']['weighted_contribution']:.3f}"),
    ("Education Fit", lambda r: f"{r['breakdown'].get('education', {}).get('value', 'N/A')}"),
    ("Education Weight", lambda r: f"{r['breakdown'].get('education', {}).get('weight', 0):.2f}"),
    ("Education Contribution", lambda r: f"{r['breakdown'].get('education', {}).get('weighted_contribution', 0):.3f}"),
    ("Recency Factor", lambda r: f"{r['breakdown'].get('recency', {}).get('value', 'N/A')}"),
    ("Recency Weight", lambda r: f"{r['breakdown'].get('recency', {}).get('weight', 0):.2f}"),
    ("Recency Contribution", lambda r: f"{r['breakdown'].get('recency', {}).get('weighted_contribution', 0):.3f}"),
]

for title, fn in metrics:
    row_vals = [f"{fn(results_by_mode[m]):<11}" for m in ["skills_only", "hybrid", "proposal", "composite"]]
    print(f"{title:<24} | {' | '.join(row_vals)}")

# ---------------------------------------------------------------------------
# 2. Experience Fit Test: 1 Year vs 3 Years (Identical Candidate otherwise)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[PART 2: EXPERIENCE FIT TEST (1 YEAR vs 3 YEARS)]")
print("Target: 3+ years required by JD. Testing candidate with 1 yr vs 3 yrs in 'proposal' & 'composite'.")

profile_1yr = {
    "experience_years": 1.0,
    "professional_months": 12,
    "education": [{"degree": "Bachelor", "field": "Computer Science"}]
}

for m in ["proposal", "composite"]:
    res_3yr = compute_scores(resume_text=candidate_std, resume_skills=skills, jd_analysis=jd_analysis, resume_profile=profile_std, mode=m)
    res_1yr = compute_scores(resume_text=candidate_std, resume_skills=skills, jd_analysis=jd_analysis, resume_profile=profile_1yr, mode=m)
    
    score_3yr = res_3yr["final_score"] * 100
    score_1yr = res_1yr["final_score"] * 100
    diff = score_3yr - score_1yr
    
    print(f"\nMode [{m.upper()}]:")
    print(f"  3 Years Candidate Score: {score_3yr:.1f}% (Exp Fit: {res_3yr['breakdown']['experience']['value']:.2f})")
    print(f"  1 Year Candidate Score:  {score_1yr:.1f}% (Exp Fit: {res_1yr['breakdown']['experience']['value']:.2f})")
    print(f"  Advantage for 3 Years:   +{diff:.1f} points")
    assert score_3yr > score_1yr, f"3-year candidate must score higher than 1-year candidate in {m} mode!"

print("=> PASS: 1-year candidate receives a strictly lower score than the 3-year candidate!")

# ---------------------------------------------------------------------------
# 3. Skill Recency Test: Recent Role (2025-Present) vs Legacy Role (2018)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[PART 3: SKILL RECENCY TEST (RECENT vs DATED ROLE - FEATURE 6)]")

resume_recent = """
Jane Doe
EXPERIENCE
Senior Backend Engineer | InnovateTech
2025 - Present
• Architected scalable microservices using Python, FastAPI, and Docker.
"""

resume_legacy = """
Jane Doe
EXPERIENCE
Backend Engineer | OldCorp
2016 - 2018
• Built backend scripts using Python, FastAPI, and Docker.
Operations Manager | BizCo
2019 - Present
• Managed operations.
"""

rec_skills = ["python", "fastapi", "docker"]
res_recent = compute_scores(resume_text=resume_recent, resume_skills=rec_skills, jd_analysis=jd_analysis, mode="composite")
res_legacy = compute_scores(resume_text=resume_legacy, resume_skills=rec_skills, jd_analysis=jd_analysis, mode="composite")

score_rec = res_recent["final_score"] * 100
score_leg = res_legacy["final_score"] * 100
rec_diff = score_rec - score_leg

print(f"Recent Role (2025-Present) Score: {score_rec:.1f}%")
print(f"Legacy Role (2018) Score:         {score_leg:.1f}%")
print(f"Recency Advantage:                +{rec_diff:.1f} points")
print(f"Recent Python Recency Weight:     {res_recent['skill_recency']['python']['weight']} ({res_recent['skill_recency']['python']['source']})")
print(f"Legacy Python Recency Weight:     {res_legacy['skill_recency']['python']['weight']} ({res_legacy['skill_recency']['python']['source']})")

assert score_rec > score_leg, "Recent candidate must score higher in composite mode!"
print("=> PASS: Recently used skills receive higher credit than older skills!")

print("\n" + "=" * 80)
print("  ALL CHECKLIST ITEMS VERIFIED (100% GREEN)")
print("=" * 80)
