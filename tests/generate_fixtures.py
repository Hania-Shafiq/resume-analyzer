"""Generate realistic dummy resume fixtures for testing the bulk upload and ranking features."""

import io
import os
import random
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pymupdf as fitz
except ImportError:
    import fitz
import docx

FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "resumes"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Resume Data ─────────────────────────────────────────────────────────────

CANDIDATES = [
    {
        "name": "Aisha Patel", "skills": "Python, FastAPI, Docker, PostgreSQL, Git, CI/CD",
        "exp": 7, "edu": "Master of Science in Computer Science", "title": "Senior Backend Engineer",
        "summary": "Experienced backend engineer with 7 years of experience building scalable APIs and microservices. Specialized in Python web frameworks and cloud-native architectures.",
        "jobs": [
            ("Senior Backend Engineer", "TechCorp Inc.", "2020-Present", "Built high-throughput REST APIs serving 10M+ requests daily using FastAPI and PostgreSQL. Led Docker containerization initiative."),
            ("Software Engineer", "DataFlow Ltd.", "2017-2020", "Developed Python microservices, implemented CI/CD pipelines with Jenkins and GitHub Actions."),
        ],
    },
    {
        "name": "James Chen", "skills": "Python, PyTorch, TensorFlow, Machine Learning, Docker, AWS",
        "exp": 5, "edu": "PhD in Machine Learning", "title": "ML Engineer",
        "summary": "Machine learning engineer with 5 years of experience in deep learning research and production ML systems. Published 4 papers in top-tier conferences.",
        "jobs": [
            ("ML Engineer", "AI Solutions Co.", "2021-Present", "Designed and deployed PyTorch models on AWS SageMaker. Reduced inference latency by 40%."),
            ("Research Scientist", "University Lab", "2019-2021", "Conducted deep learning research using TensorFlow and PyTorch. Published in NeurIPS and ICML."),
        ],
    },
    {
        "name": "Maria Garcia", "skills": "React, TypeScript, Node.js, MongoDB, Git",
        "exp": 3, "edu": "Bachelor of Science in Computer Science", "title": "Frontend Developer",
        "summary": "Frontend developer with 3 years of experience building modern web applications. Passionate about user experience and responsive design.",
        "jobs": [
            ("Frontend Developer", "WebDesign Studio", "2021-Present", "Built responsive React applications with TypeScript. Integrated REST APIs with Node.js backend."),
        ],
    },
    {
        "name": "David Kim", "skills": "Java, Python, Kubernetes, Docker, AWS, Redis, PostgreSQL, CI/CD",
        "exp": 10, "edu": "Master of Science in Software Engineering", "title": "Principal Engineer",
        "summary": "Principal engineer with over 10 years of experience in distributed systems and cloud infrastructure. Expert in containerized deployments at scale.",
        "jobs": [
            ("Principal Engineer", "CloudScale Inc.", "2019-Present", "Architected Kubernetes-based platform serving 50+ microservices on AWS. Managed Redis caching layer and PostgreSQL databases."),
            ("Senior Software Engineer", "Enterprise Systems", "2015-2019", "Built Java and Python services, implemented CI/CD with Jenkins and Docker."),
            ("Software Engineer", "StartupX", "2014-2015", "Full-stack development with Python and React."),
        ],
    },
    {
        "name": "Sarah Johnson", "skills": "Python, Django, Flask, PostgreSQL, Redis, Elasticsearch, Docker",
        "exp": 8, "edu": "Bachelor of Science in Computer Science", "title": "Senior Python Developer",
        "summary": "Senior Python developer with 8 years of experience in web application development. Deep expertise in Django and Flask frameworks.",
        "jobs": [
            ("Senior Python Developer", "WebPlatform Co.", "2019-Present", "Led Python backend team, built Django and Flask APIs. Managed PostgreSQL and Redis infrastructure."),
            ("Python Developer", "DataSearch Inc.", "2016-2019", "Built Elasticsearch-powered search features. Containerized applications with Docker."),
        ],
    },
    {
        "name": "Omar Hassan", "skills": "AWS, Kubernetes, Docker, CI/CD, Python, Elasticsearch",
        "exp": 6, "edu": "Bachelor of Science in Information Technology", "title": "DevOps Engineer",
        "summary": "DevOps engineer with 6 years of experience automating infrastructure and managing cloud deployments. Strong focus on reliability and observability.",
        "jobs": [
            ("DevOps Engineer", "InfraOps Ltd.", "2020-Present", "Managed Kubernetes clusters on AWS. Implemented CI/CD pipelines and Elasticsearch monitoring."),
            ("Systems Engineer", "HostingCo", "2018-2020", "Automated server provisioning with Python and Docker. Managed AWS EC2 and S3 infrastructure."),
        ],
    },
    {
        "name": "Emma Wilson", "skills": "Python, FastAPI, Machine Learning, PostgreSQL, Docker",
        "exp": 4, "edu": "Master of Science in Data Science", "title": "Data Engineer",
        "summary": "Data engineer with 4 years of experience building data pipelines and ML-powered features. Proficient in Python and modern API frameworks.",
        "jobs": [
            ("Data Engineer", "AnalyticsPro", "2022-Present", "Built FastAPI data services, integrated machine learning models into production pipelines with PostgreSQL."),
            ("Junior Data Engineer", "DataFirst", "2020-2022", "Developed Python ETL scripts, managed Docker containers for data processing."),
        ],
    },
    {
        "name": "Raj Patel", "skills": "React, Node.js, TypeScript, MongoDB, Redis, Docker",
        "exp": 2, "edu": "Bachelor of Science in Computer Science", "title": "Full-Stack Developer",
        "summary": "Full-stack developer with 2 years of experience in modern JavaScript frameworks. Focused on building real-time web applications.",
        "jobs": [
            ("Full-Stack Developer", "AppBuilder Inc.", "2022-Present", "Built React and Node.js applications with TypeScript. Implemented Redis caching and MongoDB data layer."),
        ],
    },
    {
        "name": "Lisa Zhang", "skills": "Python, TensorFlow, PyTorch, Machine Learning, AWS, Docker",
        "exp": 9, "edu": "PhD in Artificial Intelligence", "title": "Senior ML Scientist",
        "summary": "Senior ML scientist with 9 years of experience in deep learning and NLP. Led teams building production AI systems at scale.",
        "jobs": [
            ("Senior ML Scientist", "DeepTech AI", "2020-Present", "Led team building TensorFlow and PyTorch models on AWS. Deployed production ML pipelines serving real-time predictions."),
            ("ML Engineer", "AI Research Lab", "2017-2020", "Developed deep learning models for NLP tasks. Published in ACL and EMNLP."),
            ("Research Assistant", "University", "2015-2017", "Conducted machine learning research using Python and TensorFlow."),
        ],
    },
    {
        "name": "Carlos Rodriguez", "skills": "Python, FastAPI, Docker, Kubernetes, PostgreSQL, AWS, Git, CI/CD",
        "exp": 12, "edu": "Master of Science in Computer Science", "title": "Staff Engineer",
        "summary": "Staff engineer with over 12 years of extensive experience in backend systems, cloud infrastructure, and team leadership. Expert in Python and DevOps practices.",
        "jobs": [
            ("Staff Engineer", "MegaCorp", "2020-Present", "Led architecture of microservices platform using FastAPI, Docker, and Kubernetes on AWS. Managed PostgreSQL clusters."),
            ("Senior Engineer", "CloudNative Inc.", "2016-2020", "Built Python backend services, implemented CI/CD with Git and GitHub Actions."),
            ("Software Engineer", "TechStart", "2012-2016", "Full-stack development with Python and JavaScript."),
        ],
    },
    {
        "name": "Nina Petrov", "skills": "React, TypeScript, Node.js, Python, MongoDB, Docker, Git",
        "exp": 5, "edu": "Bachelor of Science in Computer Science", "title": "Full-Stack Developer",
        "summary": "Full-stack developer with 5 years of experience in both frontend and backend technologies. Strong in React and Node.js ecosystem.",
        "jobs": [
            ("Full-Stack Developer", "ProductLab", "2021-Present", "Built React/TypeScript frontends and Node.js APIs. Used Docker for deployment and MongoDB for data."),
            ("Frontend Developer", "DesignWorks", "2019-2021", "Developed responsive web apps with React and TypeScript. Collaborated with Python backend team."),
        ],
    },
    {
        "name": "Ahmed Ali", "skills": "Python, Django, FastAPI, PostgreSQL, Redis, Docker, AWS",
        "exp": 6, "edu": "Bachelor of Science in Software Engineering", "title": "Backend Developer",
        "summary": "Backend developer with 6 years of experience building scalable web applications. Proficient in both Django and FastAPI frameworks.",
        "jobs": [
            ("Backend Developer", "WebServe Co.", "2021-Present", "Built FastAPI microservices and Django REST APIs. Managed PostgreSQL and Redis on AWS."),
            ("Junior Developer", "CodeFactory", "2018-2021", "Developed Python web applications with Django. Used Docker for local development."),
        ],
    },
    {
        "name": "Yuki Tanaka", "skills": "Java, Python, Kubernetes, Docker, AWS, CI/CD, Redis, MongoDB, Elasticsearch",
        "exp": 15, "edu": "Master of Science in Computer Science", "title": "Distinguished Engineer",
        "summary": "Distinguished engineer with 15 years of experience across multiple technology stacks. Expert in large-scale distributed systems and cloud architecture.",
        "jobs": [
            ("Distinguished Engineer", "GlobalTech", "2018-Present", "Led platform architecture with Kubernetes on AWS. Managed Redis, MongoDB, and Elasticsearch clusters."),
            ("Principal Engineer", "EnterpriseSoft", "2013-2018", "Built Java and Python services. Implemented CI/CD and Docker containerization."),
            ("Senior Developer", "TechFirm", "2009-2013", "Full-stack development with Java, Python, and relational databases."),
        ],
    },
    {
        "name": "Priya Sharma", "skills": "Python, Machine Learning, TensorFlow, FastAPI, PostgreSQL",
        "exp": 3, "edu": "Master of Science in Data Science", "title": "ML Engineer",
        "summary": "ML engineer with 3 years of experience in building and deploying machine learning models. Strong foundation in Python and data engineering.",
        "jobs": [
            ("ML Engineer", "DataAI Corp.", "2021-Present", "Built TensorFlow models served via FastAPI endpoints. Managed data pipelines with PostgreSQL."),
        ],
    },
    {
        "name": "Michael Brown", "skills": "React, Node.js, TypeScript, MongoDB, Python, Docker",
        "exp": 1, "edu": "Associate of Science in Information Technology", "title": "Junior Developer",
        "summary": "Junior developer with 1 year of experience in web development. Eager learner with foundational skills in React and Node.js.",
        "jobs": [
            ("Junior Developer", "WebStartup", "2023-Present", "Built React components and Node.js APIs. Used TypeScript and MongoDB for a SaaS product."),
        ],
    },
    {
        "name": "Fatima Khan", "skills": "Python, Flask, PostgreSQL, Git",
        "exp": 2, "edu": "Bachelor of Science in Computer Science", "title": "Software Developer",
        "summary": "Software developer with 2 years of experience in Python web development. Focused on clean code and test-driven development.",
        "jobs": [
            ("Software Developer", "SmallTech LLC", "2022-Present", "Developed Flask web applications with PostgreSQL backend. Managed code with Git and GitHub."),
        ],
    },
    {
        "name": "Alex Murphy", "skills": "Python, FastAPI, Docker, Kubernetes, PostgreSQL, AWS, Git, CI/CD, Redis",
        "exp": 8, "edu": "Bachelor of Science in Computer Science", "title": "Senior Backend Engineer",
        "summary": "Senior backend engineer with 8 years of experience in cloud-native applications. Expert in Python, containerization, and orchestration.",
        "jobs": [
            ("Senior Backend Engineer", "CloudApp Inc.", "2020-Present", "Architected FastAPI services deployed on Kubernetes/AWS. Managed PostgreSQL and Redis."),
            ("Backend Engineer", "ServerSide Ltd.", "2016-2020", "Built Python APIs, implemented CI/CD with Docker and Git."),
        ],
    },
    {
        "name": "Sofia Costa", "skills": "Python, Machine Learning, PyTorch, TensorFlow, AWS, Docker, FastAPI",
        "exp": 7, "edu": "PhD in Computational Linguistics", "title": "NLP Engineer",
        "summary": "NLP engineer with 7 years of experience in natural language processing and deep learning. Published researcher in computational linguistics.",
        "jobs": [
            ("NLP Engineer", "LanguageAI", "2021-Present", "Built PyTorch NLP models served via FastAPI on AWS. Deployed with Docker containers."),
            ("Research Scientist", "NLP Lab", "2017-2021", "Conducted NLP research using TensorFlow and PyTorch. Published in ACL and NAACL."),
        ],
    },
    {
        "name": "Kenji Watanabe", "skills": "React, TypeScript, Node.js, Python, MongoDB, Redis, Docker, Kubernetes",
        "exp": 4, "edu": "Bachelor of Science in Computer Science", "title": "Full-Stack Engineer",
        "summary": "Full-stack engineer with 4 years of experience in modern web technologies. Strong in both frontend and backend development.",
        "jobs": [
            ("Full-Stack Engineer", "AppWorks", "2022-Present", "Built React/TypeScript frontends and Python/Node.js backends. Deployed with Docker and Kubernetes."),
            ("Junior Developer", "WebCo", "2020-2022", "Developed MongoDB-backed APIs with Redis caching."),
        ],
    },
    {
        "name": "Grace Okonkwo", "skills": "Python, Django, PostgreSQL, Docker",
        "exp": 0, "edu": "Bachelor of Science in Computer Science", "title": "Graduate Developer",
        "summary": "Fresh graduate developer with strong academic background in computer science. Completed internship projects in Python web development.",
        "jobs": [
            ("Intern Developer", "TechInterns", "2023-2023", "Built a Django web application with PostgreSQL as a capstone project. Used Docker for deployment."),
        ],
    },
]


def _build_resume_text(c):
    """Build a plain-text resume from candidate data."""
    lines = [
        f"{c['name']}",
        f"{c['title']}",
        "",
        "PROFESSIONAL SUMMARY",
        c["summary"],
        "",
        f"Over {c['exp']} years of professional experience." if c["exp"] > 0 else "",
        "",
        "TECHNICAL SKILLS",
        c["skills"],
        "",
        "WORK EXPERIENCE",
    ]
    for title, company, dates, desc in c["jobs"]:
        lines.extend([
            f"{title} | {company} | {dates}",
            desc,
            "",
        ])
    lines.extend([
        "EDUCATION",
        c["edu"],
        "",
    ])
    return "\n".join(lines)


def create_pdf(text, filepath):
    """Create a PDF from text using PyMuPDF."""
    doc = fitz.open()
    # Split into pages of ~40 lines each
    all_lines = text.split("\n")
    page_size = 40
    for i in range(0, len(all_lines), page_size):
        page = doc.new_page()
        chunk = all_lines[i:i + page_size]
        y = 50
        for line in chunk:
            page.insert_text((50, y), line, fontsize=10)
            y += 16
    doc.save(str(filepath))
    doc.close()


def create_docx(text, filepath):
    """Create a DOCX from text using python-docx."""
    doc = docx.Document()
    lines = text.split("\n")
    # First line is the name as heading
    if lines:
        doc.add_heading(lines[0], level=1)
    for line in lines[1:]:
        if line.strip():
            doc.add_paragraph(line.strip())
    doc.save(str(filepath))


def create_txt(text, filepath):
    """Write plain text file."""
    filepath.write_text(text, encoding="utf-8")


def main():
    print(f"Generating {len(CANDIDATES)} resumes in {FIXTURES_DIR}...")

    # PDF: first 8, DOCX: next 8, TXT: last 4
    for i, c in enumerate(CANDIDATES):
        idx = i + 1
        name_slug = c["name"].lower().replace(" ", "_")
        text = _build_resume_text(c)

        if i < 8:
            fp = FIXTURES_DIR / f"resume_{idx:03d}_{name_slug}.pdf"
            create_pdf(text, fp)
        elif i < 16:
            fp = FIXTURES_DIR / f"resume_{idx:03d}_{name_slug}.docx"
            create_docx(text, fp)
        else:
            fp = FIXTURES_DIR / f"resume_{idx:03d}_{name_slug}.txt"
            create_txt(text, fp)

        print(f"  ✓ {fp.name}")

    # ── Edge Cases ───────────────────────────────────────────────────────────

    # 1. Empty PDF — valid PDF with no text
    print("  Creating edge cases...")
    doc = fitz.open()
    doc.new_page()  # blank page
    doc.save(str(FIXTURES_DIR / "empty.pdf"))
    doc.close()
    print("  ✓ empty.pdf")

    # 2. Corrupted PDF — random bytes
    (FIXTURES_DIR / "corrupted.pdf").write_bytes(os.urandom(1024))
    print("  ✓ corrupted.pdf")

    # 3. Wrong type — text file with .jpg extension
    (FIXTURES_DIR / "wrong_type.jpg").write_text("This is not an image", encoding="utf-8")
    print("  ✓ wrong_type.jpg")

    # 4. Duplicate of resume #1
    src = FIXTURES_DIR / "resume_001_aisha_patel.pdf"
    if src.exists():
        (FIXTURES_DIR / "duplicate_of_001.pdf").write_bytes(src.read_bytes())
        print("  ✓ duplicate_of_001.pdf")

    # 5. Very large file (~2MB)
    large_text = _build_resume_text(CANDIDATES[0]) * 200
    (FIXTURES_DIR / "very_large.txt").write_text(large_text, encoding="utf-8")
    print(f"  ✓ very_large.txt ({len(large_text) / (1024*1024):.1f} MB)")

    # 6. Non-English resume (Spanish)
    spanish_doc = docx.Document()
    spanish_doc.add_heading("Juan Pérez - Desarrollador de Software", level=1)
    spanish_doc.add_paragraph("Desarrollador de software con 5 años de experiencia en Python, Django y PostgreSQL.")
    spanish_doc.add_paragraph("Habilidades: Python, Django, PostgreSQL, Docker, Git")
    spanish_doc.add_paragraph("Experiencia: 5 años de experiencia profesional")
    spanish_doc.add_paragraph("Educación: Licenciatura en Ingeniería de Sistemas")
    spanish_doc.save(str(FIXTURES_DIR / "non_english.docx"))
    print("  ✓ non_english.docx")

    # 7. Minimal resume
    (FIXTURES_DIR / "minimal.txt").write_text("John Smith\nPython", encoding="utf-8")
    print("  ✓ minimal.txt")

    total = len(list(FIXTURES_DIR.glob("*")))
    print(f"\nDone! {total} files created in {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
