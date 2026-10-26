"""Tests for app/overlay.py and POST /resume/overlay (require en_core_web_sm + PyMuPDF)."""

import sys
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from app.main import app
from app.overlay import SEMANTIC_THRESHOLD, build_overlay
from app.skills import analyze_job_description

client = TestClient(app)

JD = "Senior Engineer\nRequirements:\n- Docker\n- Python\nPreferred:\n- Redis\n"
LINES = [
    "Skills: Python, Docker, Redis",
    "Deployed Docker containers, reducing build time by 30%.",
]


def make_pdf(lines):
    doc = fitz.open()
    page = doc.new_page()
    y = 60
    for line in lines:
        page.insert_text((50, y), line, fontsize=11)
        y += 30
    data = doc.write()
    doc.close()
    return data


def make_image_only_pdf():
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 64, 64), False)
    pix.clear_with(180)
    page.insert_image(fitz.Rect(50, 50, 250, 250), pixmap=pix)
    data = doc.write()
    doc.close()
    return data


def jd():
    return analyze_job_description(JD).to_dict() | {"raw_text": JD}


def post(pdf, jd_text=JD, name="resume.pdf", **extra):
    return client.post(
        "/resume/overlay",
        files={"file": (name, pdf, "application/pdf")},
        data={"jd_text": jd_text, **extra},
    )


def test_docker_highlighted_in_experience_sentence():
    out = build_overlay(make_pdf(LINES), jd())
    docker = [b for b in out["pages"][0]["boxes"] if b["skill"] == "docker"]
    assert docker and all(b["category"] == "required" for b in docker)
    assert max(b["bbox"][1] for b in docker) > min(b["bbox"][1] for b in docker)  # a lower (sentence) hit


def test_list_only_skill_is_unverified():
    out = build_overlay(make_pdf(LINES), jd())
    cats = {b["skill"]: b["category"] for b in out["pages"][0]["boxes"]}
    assert cats["redis"] == "unverified"  # preferred in JD, but only listed
    assert cats["python"] == "unverified"  # required in JD, but only listed


def test_boxes_are_normalized():
    out = build_overlay(make_pdf(LINES), jd())
    boxes = out["pages"][0]["boxes"]
    assert boxes
    for b in boxes:
        x0, y0, x1, y1 = b["bbox"]
        assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
        assert {"category", "skill", "text", "tooltip"} <= set(b)


def test_counts_match_boxes_and_legend():
    out = build_overlay(make_pdf(LINES), jd())
    total = {c: 0 for c in out["counts"]}
    for p in out["pages"]:
        for b in p["boxes"]:
            total[b["category"]] += 1
    assert total == out["counts"]
    assert {i["category"] for i in out["legend"]} == {"required", "preferred", "semantic", "unverified"}
    page = out["pages"][0]
    assert page["page"] == 1 and page["width"] > 0 and page["height"] > 0 and page["image_base64"]


def test_image_only_pdf_has_no_text_layer():
    out = build_overlay(make_image_only_pdf(), jd())
    assert out["has_text_layer"] is False
    assert out["pages"] and out["pages"][0]["image_base64"]
    assert out["pages"][0]["boxes"] == []
    assert sum(out["counts"].values()) == 0


def test_max_pages_limits_rendering():
    doc = fitz.open()
    for i in range(4):
        doc.new_page().insert_text((50, 60), f"Deployed Docker containers {i}, reducing time by 30%.")
    data = doc.write()
    doc.close()
    out = build_overlay(data, jd(), max_pages=2)
    assert len(out["pages"]) == 2 and out["total_pages"] == 4


def test_semantic_skipped_when_model_loader_fails(monkeypatch):
    fake = types.ModuleType("app.matcher")

    def boom(*a, **k):
        raise RuntimeError("model unavailable")

    fake.split_sentences = lambda t: [x for x in t.splitlines() if x.strip()]
    fake.compute_similarity_matrix = boom
    fake.normalize_similarity = lambda c, *a, **k: 1.0
    monkeypatch.setitem(sys.modules, "app.matcher", fake)
    out = build_overlay(make_pdf(LINES), jd(), include_semantic=True)
    assert out["semantic_unavailable"] is True
    assert out["counts"]["semantic"] == 0


def test_semantic_boxes_when_model_available(monkeypatch):
    fake = types.ModuleType("app.matcher")
    fake.split_sentences = lambda t: [x for x in t.splitlines() if x.strip()]
    fake.compute_similarity_matrix = lambda a, b: np.full((len(a), len(b)), 0.9)
    fake.normalize_similarity = lambda c, *a, **k: 1.0
    monkeypatch.setitem(sys.modules, "app.matcher", fake)
    pdf = make_pdf(LINES + ["Led weekly planning meetings with stakeholders."])
    out = build_overlay(pdf, jd(), include_semantic=True)
    assert out["semantic_unavailable"] is False
    sem = [b for b in out["pages"][0]["boxes"] if b["category"] == "semantic"]
    assert sem and all(b["skill"] is None for b in sem)
    assert SEMANTIC_THRESHOLD == 0.55


def test_invalid_pdf_bytes_raise_value_error():
    with pytest.raises(ValueError):
        build_overlay(b"%PDF-1.4 garbage", jd())


# ----------------------------- endpoint ------------------------------------
def test_endpoint_success():
    resp = post(make_pdf(LINES))
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_text_layer"] is True and data["pages"]
    assert data["counts"]["required"] >= 1


def test_endpoint_non_pdf_400():
    resp = client.post("/resume/overlay", files={"file": ("r.txt", b"hello", "text/plain")}, data={"jd_text": JD})
    assert resp.status_code == 400


def test_endpoint_fake_pdf_content_400():
    assert post(b"this is not a pdf", name="fake.pdf").status_code == 400


def test_endpoint_empty_jd_400():
    assert post(make_pdf(LINES), jd_text="   ").status_code == 400


def test_endpoint_oversize_400():
    big = b"%PDF-" + b"0" * (5 * 1024 * 1024 + 1)
    assert post(big).status_code == 400


def test_endpoint_missing_file_422():
    assert client.post("/resume/overlay", data={"jd_text": JD}).status_code == 422


def test_endpoint_semantic_failure_does_not_crash(monkeypatch):
    fake = types.ModuleType("app.matcher")
    fake.split_sentences = lambda t: [t]

    def boom(*a, **k):
        raise OSError("no model")

    fake.compute_similarity_matrix = boom
    fake.normalize_similarity = lambda c, *a, **k: 1.0
    monkeypatch.setitem(sys.modules, "app.matcher", fake)
    resp = post(make_pdf(LINES), include_semantic="true")
    assert resp.status_code == 200
    assert resp.json()["semantic_unavailable"] is True
