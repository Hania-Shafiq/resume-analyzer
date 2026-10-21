"""CORS configuration tests (ALLOWED_ORIGINS)."""

import importlib

import pytest
from fastapi.testclient import TestClient

import app.main as main_module


@pytest.fixture
def load_app(monkeypatch):
    def _load(value):
        if value is None:
            monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
        else:
            monkeypatch.setenv("ALLOWED_ORIGINS", value)
        return TestClient(importlib.reload(main_module).app)

    yield _load
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    importlib.reload(main_module)


def _get(client, origin):
    return client.get("/", headers={"Origin": origin})


def test_default_is_wildcard(load_app):
    client = load_app(None)
    assert _get(client, "https://anything.example").headers.get("access-control-allow-origin") == "*"


def test_blank_value_falls_back_to_wildcard(load_app):
    client = load_app("  ")
    assert _get(client, "https://x.example").headers.get("access-control-allow-origin") == "*"


def test_explicit_origins_are_trimmed_and_allowed(load_app):
    client = load_app(" https://example.com , https://dashboard.example.com ")
    for origin in ("https://example.com", "https://dashboard.example.com"):
        assert _get(client, origin).headers.get("access-control-allow-origin") == origin


def test_unlisted_origin_is_not_allowed(load_app):
    client = load_app("https://example.com")
    assert "access-control-allow-origin" not in _get(client, "https://evil.example").headers


def test_preflight_for_explicit_origin(load_app):
    client = load_app("https://example.com")
    resp = client.options("/resume/upload", headers={
        "Origin": "https://example.com",
        "Access-Control-Request-Method": "POST",
    })
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "https://example.com"
