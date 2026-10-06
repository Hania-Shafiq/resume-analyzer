"""Automate browser interaction with the live Streamlit dashboard and capture screenshots."""

import time
from pathlib import Path
from playwright.sync_api import sync_playwright

ARTIFACT_DIR = Path(r"C:\Users\HP\.gemini\antigravity\brain\710a38c7-0563-46f9-9c55-cedb1463b4ea")
FIXTURES_DIR = Path(r"C:\Users\HP\OneDrive\Desktop\resume-analyzer\tests\fixtures\resumes")

JD_TEXT = """
Role: Senior Cloud Backend Engineer
Requirements:
- Strong experience in Python, FastAPI, Docker, and PostgreSQL
- Proven expertise with Git and CI/CD pipelines
Preferred:
- Kubernetes, AWS, Redis, and Machine Learning
- 5+ years of software engineering experience
"""

def main():
    # Pick 15 realistic resumes: mix of PDF, DOCX, TXT
    sample_files = [
        str(FIXTURES_DIR / "resume_001_aisha_patel.pdf"),
        str(FIXTURES_DIR / "resume_002_james_chen.pdf"),
        str(FIXTURES_DIR / "resume_003_maria_garcia.pdf"),
        str(FIXTURES_DIR / "resume_004_david_kim.pdf"),
        str(FIXTURES_DIR / "resume_005_sarah_johnson.pdf"),
        str(FIXTURES_DIR / "resume_006_omar_hassan.pdf"),
        str(FIXTURES_DIR / "resume_007_emma_wilson.pdf"),
        str(FIXTURES_DIR / "resume_009_lisa_zhang.docx"),
        str(FIXTURES_DIR / "resume_010_carlos_rodriguez.docx"),
        str(FIXTURES_DIR / "resume_011_nina_petrov.docx"),
        str(FIXTURES_DIR / "resume_012_ahmed_ali.docx"),
        str(FIXTURES_DIR / "resume_013_yuki_tanaka.docx"),
        str(FIXTURES_DIR / "resume_017_alex_murphy.txt"),
        str(FIXTURES_DIR / "resume_018_sofia_costa.txt"),
        str(FIXTURES_DIR / "resume_019_kenji_watanabe.txt"),
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 1000})
        page = context.new_page()

        print("Navigating to http://localhost:8501...")
        page.goto("http://localhost:8501", timeout=60000)
        page.wait_for_load_state("networkidle")
        time.sleep(3)

        # 1. Click on "Bulk Upload & Rank" in the sidebar
        print("Clicking 'Bulk Upload & Rank' in sidebar...")
        bulk_nav_btn = page.get_by_role("button", name="Bulk Upload & Rank")
        bulk_nav_btn.click()
        time.sleep(3)

        # 2. Fill in the Job Description textarea
        print("Entering target Job Description...")
        jd_input = page.locator("textarea").first
        jd_input.fill(JD_TEXT.strip())
        time.sleep(1)

        # 3. Upload files via the file input
        print(f"Uploading {len(sample_files)} sample resumes...")
        file_input = page.locator('input[type="file"]')
        file_input.set_input_files(sample_files)
        time.sleep(3)

        # 4. Click "Analyze & Rank Resumes" button
        print("Clicking 'Analyze & Rank Resumes'...")
        analyze_btn = page.get_by_role("button", name="Analyze & Rank Resumes")
        analyze_btn.click()

        # Wait for processing to complete
        print("Waiting for batch analysis to finish...")
        page.wait_for_selector("text=Top 5 Shortlisted Candidates", timeout=120000)
        time.sleep(2)

        # 5. Capture Top 5 screenshot
        print("Capturing Top 5 Shortlist screenshot...")
        page.screenshot(path=str(ARTIFACT_DIR / "ui_screenshot_top5.png"), full_page=True)
        print("✓ Saved ui_screenshot_top5.png")

        # 6. Click "Top 10" button
        print("Clicking 'Top 10' shortlist button...")
        top10_btn = page.get_by_role("button", name="Top 10")
        top10_btn.click()
        time.sleep(2)
        page.wait_for_selector("text=Top 10 Shortlisted Candidates", timeout=10000)

        # 7. Capture Top 10 screenshot
        print("Capturing Top 10 Shortlist screenshot...")
        page.screenshot(path=str(ARTIFACT_DIR / "ui_screenshot_top10.png"), full_page=True)
        print("✓ Saved ui_screenshot_top10.png")

        browser.close()
        print("All UI browser checks completed successfully!")

if __name__ == "__main__":
    main()
