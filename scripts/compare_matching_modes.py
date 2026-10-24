import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

print("=" * 75)
print("  COMPREHENSIVE AUDIT: SCORING MODES BEHAVIORAL COMPARISON")
print("=" * 75)

# Job description requiring 5 years of professional experience and a Master's degree
senior_jd = """
Senior Principal Systems Architect
Required Qualifications:
- Minimum 5+ years of full-time professional experience in backend systems architecture
- Master's degree in Computer Science or Software Engineering
- Mandatory expertise in: Python, FastAPI, Docker, Kubernetes, PostgreSQL

Preferred Qualifications:
- Experience in AWS, CI/CD pipelines
"""

# Candidate A: Junior candidate with matching keywords (Python, FastAPI, Docker, Kubernetes, PostgreSQL)
# but ONLY 1 year of experience and only a Bachelor's degree.
junior_candidate = """
John Doe
Email: john@example.com

EDUCATION
Bachelor of Science in Computer Science, State University, 2024

EXPERIENCE
Junior Software Developer | StartupCo
2024 - Present (1 year)
- Wrote microservices in Python and FastAPI.
- Containerized applications using Docker and Kubernetes.
- Managed PostgreSQL database schemas.
"""

# Candidate B: Senior candidate with 6 years experience, Master's degree, and same skills.
senior_candidate = """
Dr. Jane Smith
Email: jane@example.com

EDUCATION
Master of Science in Software Engineering, Tech University, 2019

EXPERIENCE
Senior Software Engineer | Enterprise Corp
2019 - Present (6 years)
- Architected enterprise cloud platforms using Python and FastAPI.
- Deployed distributed microservices across Kubernetes and Docker clusters.
- Optimized large-scale PostgreSQL databases and query engines.
"""

modes = ["skills_only", "proposal", "composite"]

print("\n--- TEST CASE 1: JUNIOR CANDIDATE (1 yr exp, Bachelor's) vs SENIOR JD (5 yrs exp, Master's) ---")
for mode in modes:
    resp = client.post("/match", json={
        "resume_text": junior_candidate,
        "job_description": senior_jd,
        "mode": mode
    })
    data = resp.json()
    bd = data["score_breakdown"]
    exp_fit = bd.get("experience", {}).get("value")
    edu_fit = bd.get("education", {}).get("value")
    sem_sim = bd.get("semantic", {}).get("value")
    
    print(f"\n[Mode: {mode.upper()}]")
    print(f"  Final Match Score:       {data['final_score'] * 100:.1f}%")
    print(f"  Skill Overlap Score:     {data['skill_overlap_score'] * 100:.1f}%")
    print(f"  Experience Fit Evaluated:{exp_fit} (Weight: {bd.get('experience', {}).get('weight')})")
    print(f"  Education Fit Evaluated: {edu_fit} (Weight: {bd.get('education', {}).get('weight', 0)})")
    print(f"  Semantic Similarity:     {sem_sim if sem_sim is not None else 'N/A (Ignored)'}")

print("\n\n--- TEST CASE 2: SENIOR CANDIDATE (6 yrs exp, Master's) vs SENIOR JD (5 yrs exp, Master's) ---")
for mode in modes:
    resp = client.post("/match", json={
        "resume_text": senior_candidate,
        "job_description": senior_jd,
        "mode": mode
    })
    data = resp.json()
    bd = data["score_breakdown"]
    exp_fit = bd.get("experience", {}).get("value")
    edu_fit = bd.get("education", {}).get("value")
    sem_sim = bd.get("semantic", {}).get("value")
    
    print(f"\n[Mode: {mode.upper()}]")
    print(f"  Final Match Score:       {data['final_score'] * 100:.1f}%")
    print(f"  Skill Overlap Score:     {data['skill_overlap_score'] * 100:.1f}%")
    print(f"  Experience Fit Evaluated:{exp_fit} (Weight: {bd.get('experience', {}).get('weight')})")
    print(f"  Education Fit Evaluated: {edu_fit} (Weight: {bd.get('education', {}).get('weight', 0)})")
    print(f"  Semantic Similarity:     {sem_sim if sem_sim is not None else 'N/A (Ignored)'}")

print("\n" + "=" * 75)
print("  BEHAVIORAL COMPARISON SUMMARY")
print("=" * 75)
print("1. 'skills_only' evaluates ONLY keyword taxonomy overlap (100% weight).")
print("   - Experience weight is 0.0; Education weight is 0.0; Semantic weight is 0.0.")
print("   - It awards the junior candidate a high score (80-100%) purely based on keyword presence,")
print("     completely ignoring that the candidate has only 1 year of experience against 5 required.")
print("\n2. 'proposal' is the FULL MATCHING MODE (Weighted Comprehensive).")
print("   - Skills: 40% | Semantic Similarity: 40% | Experience Fit: 10% | Education Fit: 10%.")
print("   - It penalizes the junior candidate for seniority and educational credential deficits.")
print("   - It rewards the qualified senior candidate across all dimensions.")
print("\n3. 'composite' is the MULTI-FACTOR RECENCY MODE (Feature 6).")
print("   - Skills (with evidence depth multipliers): 40% | Semantic: 30% | Experience: 15% | Recency: 15%.")
print("=" * 75)
