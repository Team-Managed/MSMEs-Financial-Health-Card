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
    assert 0 <= data["tail_risk"]["probability_cfcr_below_one"] <= 1
    assert data["tail_risk"]["simulations"] == 5000
    assert any(
        "not calibrated default probabilities" in assumption
        for assumption in data["tail_risk"]["assumptions"]
    )


def test_analyze_unknown_persona_returns_404():
    response = client.post("/api/msme/nonexistent/analyze")
    assert response.status_code == 404


@patch("backend.app.main.run_pipeline_with_profile")
def test_custom_analysis_uses_declared_financials_and_facility(mock_run):
    from backend.app.graph.risk_engine import compute_risk
    from backend.app.schemas.models import WeightVector

    weights = WeightVector(gst=0.30, upi=0.30, aa=0.25, epfo=0.15)

    def pipeline_state(profile, retriever):
        return {
            "profile": profile,
            "weights": weights,
            "weight_rationale": [],
            "risk_output": compute_risk(profile, weights),
            "narrative": "Test narrative.",
            "grounding_trace": [],
        }

    mock_run.side_effect = pipeline_state
    payload = {
        "sector": "services",
        "years_operating": 6,
        "profile_type": "healthy",
        "requested_amount_lakh": 20,
        "annual_interest_rate_pct": 12,
        "expected_utilization_pct": 75,
        "annual_turnover_lakh": 60,
        "avg_monthly_inflow_lakh": 5,
        "avg_monthly_operating_outflow_lakh": 3,
        "avg_bank_balance_lakh": 4,
        "existing_monthly_emi_lakh": 1,
        "top_buyer_share_pct": 35,
        "bounced_payments_12mo": 1,
        "gst_filing_consistency_pct": 90,
        "yoy_growth_pct": 8,
    }

    response = client.post("/api/analyze", json=payload)

    assert response.status_code == 200
    profile = mock_run.call_args.args[0]
    assert profile.aa_bank_data.avg_account_balance == 400_000
    assert profile.aa_bank_data.existing_loan_emi_total == 115_000
    assert profile.upi.top_counterparty_share == 0.35
    assert profile.aa_bank_data.bounced_payment_count_12mo == 1
    summary = response.json()["profile_summary"]
    assert summary["requested_amount_lakh"] == 20
    assert summary["proposed_monthly_interest"] == 15_000
