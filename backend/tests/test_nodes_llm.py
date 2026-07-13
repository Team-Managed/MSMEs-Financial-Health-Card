import json
import pytest
from unittest.mock import patch, MagicMock
from backend.app.data.personas import PERSONAS
from backend.app.schemas.models import WeightVector
from backend.app.graph.risk_engine import compute_risk
from backend.app.graph.nodes import (
    _build_safe_summary_from_risk,
    _parse_explainer_output,
    node_explainer,
    node_grounding_validator,
)


def _make_state():
    profile = PERSONAS["buyer_concentrated"]
    weights = WeightVector(gst=0.30, upi=0.30, aa=0.25, epfo=0.15)
    risk_output = compute_risk(profile, weights)
    return {
        "profile": profile,
        "weights": weights,
        "risk_output": risk_output,
        "explainer_chunks": [
            {"chunk_id": "c001", "text": "RBI guidance on MSME credit risk.", "source": "rbi.pdf", "section": "s1"}
        ],
    }


@patch("backend.app.graph.nodes.genai")
def test_explainer_returns_narrative(mock_genai):
    mock_response = MagicMock()
    mock_response.text = '''```json
    {"narrative": "The business shows a CFCR under baseline conditions.", "claims": [
      {"source_field": "cfcr_baseline", "value": 1.2, "text": "CFCR is 1.2.", "type": "numeric"},
    {"source_field": "__explainer_chunks__", "value": "c001", "text": "Guidance supports the risk.", "type": "citation"}
    ]}
    ```'''
    mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
    state = _make_state()
    result = node_explainer(state)
    assert "narrative" in result
    assert isinstance(result["narrative"], str)
    assert len(result["narrative"]) > 20
    assert len(result["claims"]) == 2


def test_explainer_parser_accepts_fenced_strict_json():
    narrative, claims = _parse_explainer_output('''```json
    {"narrative": "Grounded narrative.", "claims": [
      {"source_field": "cfcr_baseline", "value": 1.25, "text": "CFCR is 1.25.", "type": "numeric"}
    ]}
    ```''')

    assert narrative == "Grounded narrative."
    assert claims == [{
        "source_field": "cfcr_baseline",
        "value": 1.25,
        "text": "CFCR is 1.25.",
        "type": "numeric",
    }]


@pytest.mark.parametrize("raw", [
    "not json",
    '{"narrative": "Missing claims"}',
    '{"narrative": "Bad claim", "claims": [{"type": "numeric"}]}',
])
def test_explainer_parser_rejects_invalid_json_or_claims(raw):
    with pytest.raises((json.JSONDecodeError, ValueError)):
        _parse_explainer_output(raw)


def test_grounding_validator_catches_fabricated_number():
    state = _make_state()
    state["narrative"] = "The CFCR is 99.99 in the baseline scenario."
    state["claims"] = [{
        "source_field": "cfcr_baseline",
        "value": 99.99,
        "text": "The CFCR is 99.99 in the baseline scenario.",
        "type": "numeric",
    }]
    state["explainer_chunks"] = []
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
    state["claims"] = [{
        "source_field": "cfcr_baseline",
        "value": cfcr,
        "text": state["narrative"],
        "type": "numeric",
    }]
    state["explainer_chunks"] = []
    result = node_grounding_validator(state)
    checks = result["grounding_trace"]
    fails = [c for c in checks if c.status == "fail"]
    assert len(fails) == 0
    assert result["narrative"] == state["narrative"]


def test_grounding_validator_rejects_numeric_claim_with_wrong_source_value():
    state = _make_state()
    original = "The baseline CFCR is 99.99."
    state.update({
        "narrative": original,
        "claims": [{
            "source_field": "cfcr_baseline",
            "value": 99.99,
            "text": original,
            "type": "numeric",
        }],
    })

    result = node_grounding_validator(state)

    assert any(check.status == "fail" and check.source == "cfcr_baseline" for check in result["grounding_trace"])
    assert result["narrative"] != original
    assert "99.99" not in result["narrative"]


def test_grounding_validator_rejects_invalid_citation_and_replaces_narrative():
    state = _make_state()
    original = "Guidance supports this risk [invented-999]."
    state.update({
        "narrative": original,
        "claims": [{
            "source_field": "__explainer_chunks__",
            "value": "invented-999",
            "text": original,
            "type": "citation",
        }],
    })

    result = node_grounding_validator(state)

    assert any(check.status == "fail" and check.source == "invented-999" for check in result["grounding_trace"])
    assert result["narrative"] != original
    assert "invented-999" not in result["narrative"]


def test_grounding_validator_preserves_narrative_when_all_claims_pass():
    state = _make_state()
    cfcr = state["risk_output"]["cfcr_baseline"]
    original = f"Baseline CFCR is {cfcr}. Guidance supports review [c001]."
    state.update({
        "narrative": original,
        "claims": [
            {"source_field": "cfcr_baseline", "value": cfcr, "text": f"Baseline CFCR is {cfcr}.", "type": "numeric"},
            {"source_field": "__explainer_chunks__", "value": "c001", "text": "Guidance supports review.", "type": "citation"},
        ],
    })

    result = node_grounding_validator(state)

    assert result["narrative"] == original
    assert all(check.status == "pass" for check in result["grounding_trace"])


def test_safe_summary_contains_only_deterministic_risk_values():
    state = _make_state()
    risk_output = state["risk_output"]
    narrative, claims = _build_safe_summary_from_risk(risk_output)

    assert "Limited-confidence deterministic summary" in narrative
    assert "[" not in narrative
    validation = node_grounding_validator({
        **state,
        "narrative": narrative,
        "claims": claims,
        "explainer_chunks": [],
    })
    assert validation["narrative"] == narrative
    assert all(check.status == "pass" for check in validation["grounding_trace"])
