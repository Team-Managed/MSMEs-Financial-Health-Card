import pytest
from unittest.mock import patch, MagicMock
from backend.app.data.personas import PERSONAS
from backend.app.schemas.models import WeightVector
from backend.app.graph.risk_engine import compute_risk
from backend.app.graph.nodes import node_explainer, node_grounding_validator


def _make_state():
    profile = PERSONAS["buyer_concentrated"]
    weights = WeightVector(gst=0.30, upi=0.30, aa=0.25, epfo=0.15)
    risk_output = compute_risk(profile, weights)
    return {
        "profile": profile,
        "weights": weights,
        "risk_output": risk_output,
        "retrieved_chunks": [
            {"chunk_id": "c001", "text": "RBI guidance on MSME credit risk.", "source": "rbi.pdf", "section": "s1"}
        ],
    }


@patch("backend.app.graph.nodes.genai")
def test_explainer_returns_narrative(mock_genai):
    mock_response = MagicMock()
    mock_response.text = (
        "The business shows a CFCR of 1.20 under baseline conditions. "
        "The buyer_loss scenario reduces CFCR significantly. "
        "Per RBI guidance [c001], concentrated counterparty exposure is a key risk factor."
    )
    mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
    state = _make_state()
    result = node_explainer(state)
    assert "narrative" in result
    assert isinstance(result["narrative"], str)
    assert len(result["narrative"]) > 20


def test_grounding_validator_catches_fabricated_number():
    state = _make_state()
    state["narrative"] = "The CFCR is 99.99 in the baseline scenario."
    state["retrieved_chunks"] = []
    result = node_grounding_validator(state)
    assert "grounding_trace" in result
    checks = result["grounding_trace"]
    # The number 99.99 is not in risk_output — should be flagged
    fails = [c for c in checks if c.status == "fail"]
    assert len(fails) >= 1


def test_grounding_validator_passes_real_number():
    state = _make_state()
    cfcr = state["risk_output"]["cfcr_baseline"]
    state["narrative"] = f"The CFCR is {cfcr} under baseline conditions."
    state["retrieved_chunks"] = []
    result = node_grounding_validator(state)
    checks = result["grounding_trace"]
    fails = [c for c in checks if c.status == "fail"]
    assert len(fails) == 0
