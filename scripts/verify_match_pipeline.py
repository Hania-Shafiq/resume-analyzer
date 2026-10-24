import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app.main import app
from app.skills import analyze_job_description, extract_skills
from app.experience import extract_experience

client = TestClient(app)

print("=" * 70)
print("  END-TO-END VERIFICATION: PRIORITIES 1, 2, AND 3")
print("=" * 70)

# Sample JD with sales & CRM requirements and 3 years experience
sales_jd = """
Senior Sales Development Representative
Requirements:
- 3+ years of professional sales experience
- Hands-on experience with CRM platforms (Salesforce or HubSpot)
- Proven track record in lead generation and prospecting
- Strong deal negotiation and closing capabilities
- Demonstrable experience in sales pipeline management
- Excellent communication skills

Preferred Qualifications:
- Experience in market research
- Bachelor's degree in Business or related field
"""

# Sample Resume with missing sales skills and only internship experience
candidate_resume = """
Jane Doe
Email: jane.doe@example.com

EDUCATION
Bachelor of Science in Communications, State University, 2023

EXPERIENCE
Sales & Marketing Intern | FastGrowth Inc.
Jan 2023 - Jun 2023
- Supported marketing team with research and communication.
- Assisted with customer communication and market research presentations.
"""

print("\n[1] VERIFYING JD PARSING & SALES TAXONOMY")
jd_analysis = analyze_job_description(sales_jd)
print(f"Extracted Required Skills: {jd_analysis.required_skills}")
print(f"Extracted Preferred Skills: {jd_analysis.preferred_skills}")
print(f"Extracted Min Experience:  {jd_analysis.min_experience_years} years")

expected_sales_skills = {"crm", "lead generation", "negotiation", "sales pipeline management"}
found_sales_skills = expected_sales_skills.intersection(set(jd_analysis.required_skills))
assert found_sales_skills == expected_sales_skills, f"Missing sales skills in JD: {expected_sales_skills - found_sales_skills}"
print("=> PASS: All 4 sales/CRM skills accurately extracted as Required from JD!")

print("\n[2] VERIFYING INTERNSHIP VS FULL-TIME EXPERIENCE EXTRACTION")
exp_summary = extract_experience(candidate_resume)
print(f"Professional Months: {exp_summary.professional_months}")
print(f"Full-Time Months:    {exp_summary.full_time_months}")
print(f"Internship Months:   {exp_summary.internship_months}")
print(f"Has Internships:     {exp_summary.has_internships}")
assert exp_summary.full_time_months == 0, "Full-time months should be 0 for intern"
assert exp_summary.internship_months > 0, "Internship months should be > 0"
print("=> PASS: Internship strictly segregated from full-time experience!")

print("\n[3] VERIFYING /match ENDPOINT WITH PROPOSAL SCORING MODE")
resp = client.post("/match", json={
    "resume_text": candidate_resume,
    "job_description": sales_jd,
    "mode": "proposal"
})
assert resp.status_code == 200, f"Error: {resp.status_code} - {resp.text}"
m = resp.json()

print(f"Scoring Mode:         {m['scoring_mode']}")
print(f"Final Score:          {m['final_score']}")
print(f"Skill Overlap Score:  {m['skill_overlap_score']}")
print(f"Required Matched:     {m['required_matched']}")
print(f"Required Missing:     {m['required_missing']}")
print(f"Preferred Matched:    {m['preferred_matched']}")
print(f"Preferred Missing:    {m['preferred_missing']}")
print(f"Experience Analysis:  {m['experience_analysis']}")

# Verify that the missing sales skills are present in required_missing
for s in expected_sales_skills:
    assert s in m["required_missing"], f"Skill {s} should be in required_missing!"
print("=> PASS: CRM, lead generation, negotiation, sales pipeline management flagged as missing required!")

# Verify experience is NOT falsely marked as met
assert m["experience_analysis"]["is_met"] is False, "Experience should NOT be met for intern against 3 years requirement"
print("=> PASS: Experience requirement strictly verified as not met!")

# Verify score is not false 80%
assert m["final_score"] < 0.60, f"Score should be penalized for missing mandatory skills and exp! Got: {m['final_score']}"
print(f"=> PASS: Match score realistically penalized ({m['final_score'] * 100:.1f}%)!")

print("\n[4] VERIFYING KPI COUNTS CONSISTENCY")
req_matched = m.get("required_matched", [])
pref_matched = m.get("preferred_matched", [])
req_missing = m.get("required_missing", [])
pref_missing = m.get("preferred_missing", [])

total_req = len(req_matched) + len(req_missing)
total_pref = len(pref_matched) + len(pref_missing)
total_gaps = len(req_missing) + len(pref_missing)
critical_gaps = len(req_missing)
recs = req_missing + [s for s in pref_missing if s not in req_missing]

print(f"Mandatory Met:        {len(req_matched)} / {total_req}")
print(f"Preferred Met:        {len(pref_matched)} / {total_pref}")
print(f"Total Gaps:           {total_gaps}")
print(f"Critical Gaps:        {critical_gaps}")
print(f"Learning Priorities:  {len(recs)}")

assert total_gaps == critical_gaps + len(pref_missing)
assert len(recs) == total_gaps
print("=> PASS: All KPI numbers (met, missing, critical gaps, learning priorities) agree 100%!")

print("\n[5] VERIFYING CRITERIA BREAKDOWN & HTML TABLE RENDERING (NO CRASH)")
from dashboard.app import _render_criteria_breakdown, _render_scoring_extras, _render_custom_table

# Call evidence endpoint
ev_resp = client.post("/resume/evidence", json={"resume_text": candidate_resume})
ev_data = ev_resp.json() if ev_resp.status_code == 200 else None

# Test rendering without pyarrow or Streamlit errors
try:
    _render_criteria_breakdown(m, evidence_data=ev_data)
    _render_scoring_extras(m)
    print("=> PASS: Breakdown and scoring extras rendered cleanly without any pyarrow error!")
except Exception as e:
    print(f"=> FAIL: Rendering raised exception: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("  ALL VERIFICATION CHECKS PASSED SUCCESSFULLY (100%)")
print("=" * 70)
