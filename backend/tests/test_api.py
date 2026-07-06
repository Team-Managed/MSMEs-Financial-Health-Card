import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_get_personas():
    response = client.get("/api/personas")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 4
    ids = [p["id"] for p in data]
    assert "healthy" in ids
    assert "buyer_concentrated" in ids


@patch("backend.app.main.run_pipeline")
def test_analyze_returns_analysis_response(mock_run):
    from backend.app.data.personas import PERSONAS
    from backend.app.schemas.models import WeightVector
    from backend.app.graph.risk_engine import compute_risk

    profile = PERSONAS["healthy"]
    weights = WeightVector(gst=0.30, upi=0.30, aa=0.25, epfo=0.15)
    risk = compute_risk(profile, weights)

    mock_run.return_value = {
        "profile": profile,
        "weights": weights,
        "weight_rationale": [],
        "risk_output": risk,
        "narrative": "Test narrative.",
        "grounding_trace": [],
    }

    response = client.post("/api/msme/healthy/analyze")
    assert response.status_code == 200
    data = response.json()
    assert "cfcr_baseline" in data
    assert "baseline_score" in data
    assert "narrative" in data
    assert "grounding_trace" in data


def test_analyze_unknown_persona_returns_404():
    response = client.post("/api/msme/nonexistent/analyze")
    assert response.status_code == 404
