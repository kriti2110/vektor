"""Smoke tests for the FastAPI app — make sure routes wire up and respond."""

from __future__ import annotations

import pytest


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from vektor.api.main import app

    with TestClient(app) as c:
        yield c


def test_healthz_responds(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_metrics_responds(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "vektor_request_duration_seconds" in r.text


def test_search_without_index_returns_503(client):
    # No index loaded at startup → /search should 503, not crash.
    r = client.post("/search", json={"query": "hello", "k": 5})
    assert r.status_code == 503
