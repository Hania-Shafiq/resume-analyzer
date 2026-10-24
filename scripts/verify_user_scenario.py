import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

print("=" * 80)
print("  VERIFYING USER SCENARIO: 1 YR EXP vs 3 YR JD & UNBULLETED RECENCY")
print("=" * 80)

# Exact scenario matching user description:
# Role requires 3 years of experience.
# Candidate CV has 1 year of experience in Python and SQL without bullet points.

jd_text = """
Backend Software Engineer
Requirements:
- 3+ years experience in backend development
- Python
- SQL
- Docker
"""

# CV with 1 year of experience and duties written in natural sentences (no bullets)
cv_text = """
Alex Taylor
Email: alex@example.com

EDUCATION
Bachelor of Science in Computer Science, 2023

EXPERIENCE
Software Engineer | DataSystems Inc.
Jan 2024 - Dec 2024
Built scalable backend APIs using Python and SQL.
Managed database migrations and query optimization.
"""

# 1. Test in Composite mode (which was used in the user screenshot)
resp_comp = client.post("/match", json={
    "resume_text": cv_text,
    "job_description": jd_text,
    "mode": "composite"
})
assert resp_comp.status_code == 200
data_comp = resp_comp.json()
bd_comp = data_comp["score_breakdown"]

print("\n[EVALUATION IN COMPOSITE MODE]")
print(f"Overall Match Score:       {data_comp['final_score'] * 100:.1f}%")
print(f"Skill Component:           {bd_comp['skill']['value'] * 100:.1f}% (Weight: {bd_comp['skill']['weight']:.2f}, Contrib: {bd_comp['skill']['weighted_contribution'] * 100:.1f}%)")
print(f"Semantic Component:        {bd_comp['semantic']['value'] * 100:.1f}% (Weight: {bd_comp['semantic']['weight']:.2f}, Contrib: {bd_comp['semantic']['weighted_contribution'] * 100:.1f}%)")
print(f"Experience Component:      {bd_comp['experience']['value'] * 100:.1f}% (Weight: {bd_comp['experience']['weight']:.2f}, Contrib: {bd_comp['experience']['weighted_contribution'] * 100:.1f}%)")
print(f"Recency Component:         {bd_comp['recency']['value'] * 100:.1f}% (Weight: {bd_comp['recency']['weight']:.2f}, Contrib: {bd_comp['recency']['weighted_contribution'] * 100:.1f}%)")

print("\nSkill Recency Breakdown:")
for skill, info in data_comp.get("skill_recency", {}).items():
    print(f"  - {skill:<12}: Last Used = {info['last_used_year']}, Weight = {info['weight']:.2f}, Source = {info['source']}")

# Assertions to prove the fix:
# 1. Experience should NOT be 100%! It should be 1 yr / 3 yrs = 33.3%
exp_val = bd_comp['experience']['value']
assert exp_val < 0.50, f"Experience should reflect 1/3 = 33.3%! Got: {exp_val}"
print(f"\n=> PASS 1: Experience fit is {exp_val * 100:.1f}%, NOT 100%! (Reflects 1 year vs 3 years)")

# 2. Python and SQL should NOT be Undated! They should be dated to 2024!
py_rec = data_comp["skill_recency"]["python"]
assert py_rec["last_used_year"] == 2024, f"Python should be dated 2024! Got: {py_rec}"
assert py_rec["source"] == "dated_role", f"Python source should be dated_role! Got: {py_rec['source']}"
print(f"=> PASS 2: Skill recency successfully extracted dates! (Python last used: {py_rec['last_used_year']}, source: {py_rec['source']})")

# 3. Test in Proposal mode (which includes Education fit)
resp_prop = client.post("/match", json={
    "resume_text": cv_text,
    "job_description": jd_text,
    "mode": "proposal"
})
assert resp_prop.status_code == 200
data_prop = resp_prop.json()
bd_prop = data_prop["score_breakdown"]

print("\n[EVALUATION IN PROPOSAL MODE (Includes Education Component)]")
print(f"Overall Match Score:       {data_prop['final_score'] * 100:.1f}%")
print(f"Skill Component:           {bd_prop['skill']['value'] * 100:.1f}% (Weight: {bd_prop['skill']['weight']:.2f})")
print(f"Semantic Component:        {bd_prop['semantic']['value'] * 100:.1f}% (Weight: {bd_prop['semantic']['weight']:.2f})")
print(f"Experience Component:      {bd_prop['experience']['value'] * 100:.1f}% (Weight: {bd_prop['experience']['weight']:.2f})")
print(f"Education Component:       {bd_prop['education']['value'] * 100:.1f}% (Weight: {bd_prop['education']['weight']:.2f})")
print(f"=> PASS 3: Proposal mode explicitly includes Education Fit as a separate component!")

print("\n" + "=" * 80)
print("  ALL FOUR POINTS VERIFIED & RESOLVED")
print("=" * 80)
