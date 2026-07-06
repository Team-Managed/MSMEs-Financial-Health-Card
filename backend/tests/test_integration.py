"""
Integration smoke test — requires GOOGLE_API_KEY set in environment.
Skipped if key is not present (safe for CI without credentials).
"""
import os
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set — skipping integration test"
)


def test_full_pipeline_healthy_persona():
    from backend.app.main import app
    client = TestClient(app)
    response = client.post("/api/msme/healthy/analyze")
    assert response.status_code == 200
    data = response.json()
    assert data["cfcr_baseline"] > 0
    assert 0 <= data["baseline_score"] <= 100
    assert len(data["cfcr_by_scenario"]) == 5   # baseline + 4 scenarios
    assert isinstance(data["narrative"], str)
    assert len(data["narrative"]) > 50


def test_full_pipeline_buyer_concentrated_cfcr_drops():
    from backend.app.main import app
    client = TestClient(app)
    response = client.post("/api/msme/buyer_concentrated/analyze")
    assert response.status_code == 200
    data = response.json()
    baseline_cfcr = data["cfcr_baseline"]
    buyer_loss = next(r for r in data["cfcr_by_scenario"] if r["scenario"] == "buyer_loss")
    assert buyer_loss["cfcr"] < baseline_cfcr, "buyer_loss must reduce CFCR"
