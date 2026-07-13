import json
import os
import pytest
from unittest.mock import patch, MagicMock
from backend.app.data.personas import PERSONAS
from backend.app.schemas.models import WeightVector
from backend.app.graph.nodes import (
    node_aggregator,
    node_sector_retriever,
    node_weight_setter,
    node_stress_generator,
    node_risk_engine,
    node_explainer_retriever,
    node_explainer,
    node_grounding_validator,
    _DEFAULT_WEIGHTS,
    _DEFAULT_RATIONALE,
)


def test_aggregator_sets_profile():
    state = {"persona_id": "healthy"}
    result = node_aggregator(state)
    assert "profile" in result
    assert result["profile"].msme_id == "p001"


def test_sector_retriever_sets_chunks_list(tmp_path):
    from backend.app.rag.retriever import Retriever
    retriever = Retriever(chroma_dir=str(tmp_path / "empty"))
    state = {"profile": PERSONAS["healthy"], "retriever": retriever}
    result = node_sector_retriever(state)
    assert "retrieved_chunks" in result
    assert isinstance(result["retrieved_chunks"], list)


def test_stress_generator_returns_four_scenarios():
    state = {"profile": PERSONAS["healthy"]}
    result = node_stress_generator(state)
    assert "scenarios" in result
    assert len(result["scenarios"]) == 4


def test_risk_engine_node_produces_cfcr():
    state = {
        "profile": PERSONAS["healthy"],
        "weights": WeightVector(gst=0.30, upi=0.30, aa=0.25, epfo=0.15),
    }
    result = node_risk_engine(state)
    assert "risk_output" in result
    assert result["risk_output"]["cfcr_baseline"] > 0


def test_explainer_retriever_builds_risk_specific_second_query():
    retriever = MagicMock()
    retriever.query.return_value = [{"chunk_id": "explainer-1", "text": "Material risk"}]
    risk_output = node_risk_engine({"profile": PERSONAS["buyer_concentrated"]})["risk_output"]

    result = node_explainer_retriever({
        "profile": PERSONAS["buyer_concentrated"],
        "risk_output": risk_output,
        "retriever": retriever,
    })

    query = retriever.query.call_args.args[0]
    assert "buyer concentration" in query.lower()
    assert "worst" in query.lower()
    assert result["explainer_chunks"] == retriever.query.return_value


@patch("backend.app.graph.nodes._get_gemini_model")
def test_explainer_and_validator_use_only_explainer_chunks(mock_get_model):
    response = MagicMock()
    response.text = json.dumps({
        "narrative": "Guidance applies [explainer-1].",
        "claims": [{
            "source_field": "__explainer_chunks__",
            "value": "explainer-1",
            "text": "Guidance applies.",
            "type": "citation",
        }],
    })
    mock_get_model.return_value.generate_content.return_value = response
    risk_output = node_risk_engine({"profile": PERSONAS["healthy"]})["risk_output"]
    state = {
        "profile": PERSONAS["healthy"],
        "risk_output": risk_output,
        "retrieved_chunks": [{"chunk_id": "weight-1", "source": "weights.pdf", "text": "Weights"}],
        "explainer_chunks": [{"chunk_id": "explainer-1", "source": "risk.pdf", "text": "Risks"}],
    }

    explanation = node_explainer(state)
    validation = node_grounding_validator({**state, **explanation})
    prompt = mock_get_model.return_value.generate_content.call_args.args[0]

    assert "explainer-1" in prompt
    assert "weight-1" not in prompt
    citation_checks = [item for item in validation["grounding_trace"] if item.type == "citation"]
    assert citation_checks[0].source == "explainer-1"
    assert citation_checks[0].status == "pass"


@patch("backend.app.graph.nodes.genai")
def test_weight_setter_returns_weight_vector(mock_genai):
    """LLM path is exercised when chunks are provided; mock must be called."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "weights": {"gst": 0.30, "upi": 0.30, "aa": 0.25, "epfo": 0.15},
        "rationale": [
            {"dimension": "gst", "reasoning": "Sector guidance.", "cited_chunk_id": "chunk-001"},
            {"dimension": "upi", "reasoning": "UPI data.", "cited_chunk_id": "chunk-001"},
            {"dimension": "aa", "reasoning": "Repayment record.", "cited_chunk_id": "chunk-001"},
            {"dimension": "epfo", "reasoning": "Payroll stability.", "cited_chunk_id": "chunk-001"},
        ],
    })
    mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
    state = {
        "profile": PERSONAS["healthy"],
        "retrieved_chunks": _FAKE_CHUNKS,
    }
    result = node_weight_setter(state)
    assert "weights" in result
    assert isinstance(result["weights"], WeightVector)
    mock_genai.GenerativeModel.return_value.generate_content.assert_called_once()


# ── Weight/rationale integrity tests ─────────────────────────────────────────

_FAKE_CHUNKS = [
    {
        "chunk_id": "chunk-001",
        "source": "RBI-2024",
        "section": "credit-weights",
        "text": "Sector-based weight guidance for MSME credit assessment.",
    }
]


@patch("backend.app.graph.nodes.genai")
def test_weight_setter_rejects_duplicate_dimensions(mock_genai):
    """Duplicate dimension in rationale → full fallback to defaults."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "weights": {"gst": 0.30, "upi": 0.30, "aa": 0.25, "epfo": 0.15},
        "rationale": [
            {"dimension": "gst", "reasoning": "GST strong.", "cited_chunk_id": "chunk-001"},
            {"dimension": "gst", "reasoning": "GST again.", "cited_chunk_id": "chunk-001"},  # duplicate
            {"dimension": "aa", "reasoning": "no retrieved guidance — default used", "cited_chunk_id": ""},
            {"dimension": "epfo", "reasoning": "no retrieved guidance — default used", "cited_chunk_id": ""},
        ],
    })
    mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
    state = {"profile": PERSONAS["healthy"], "retrieved_chunks": _FAKE_CHUNKS}
    result = node_weight_setter(state)
    assert result["weights"] == _DEFAULT_WEIGHTS
    assert result["weight_rationale"] == _DEFAULT_RATIONALE


@patch("backend.app.graph.nodes.genai")
def test_weight_setter_rejects_missing_dimension(mock_genai):
    """Missing dimension in rationale → full fallback to defaults."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "weights": {"gst": 0.30, "upi": 0.30, "aa": 0.25, "epfo": 0.15},
        "rationale": [
            {"dimension": "gst", "reasoning": "GST strong.", "cited_chunk_id": "chunk-001"},
            {"dimension": "upi", "reasoning": "UPI reliable.", "cited_chunk_id": "chunk-001"},
            {"dimension": "aa", "reasoning": "no retrieved guidance — default used", "cited_chunk_id": ""},
            # epfo missing
        ],
    })
    mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
    state = {"profile": PERSONAS["healthy"], "retrieved_chunks": _FAKE_CHUNKS}
    result = node_weight_setter(state)
    assert result["weights"] == _DEFAULT_WEIGHTS
    assert result["weight_rationale"] == _DEFAULT_RATIONALE


@patch("backend.app.graph.nodes.genai")
def test_weight_setter_rejects_fabricated_chunk_id(mock_genai):
    """cited_chunk_id not present in retrieved_chunks → full fallback to defaults."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "weights": {"gst": 0.30, "upi": 0.30, "aa": 0.25, "epfo": 0.15},
        "rationale": [
            {"dimension": "gst", "reasoning": "Guidance says 30%.", "cited_chunk_id": "INVENTED-999"},
            {"dimension": "upi", "reasoning": "UPI weight.", "cited_chunk_id": "chunk-001"},
            {"dimension": "aa", "reasoning": "no retrieved guidance — default used", "cited_chunk_id": ""},
            {"dimension": "epfo", "reasoning": "no retrieved guidance — default used", "cited_chunk_id": ""},
        ],
    })
    mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
    state = {"profile": PERSONAS["healthy"], "retrieved_chunks": _FAKE_CHUNKS}
    result = node_weight_setter(state)
    assert result["weights"] == _DEFAULT_WEIGHTS
    assert result["weight_rationale"] == _DEFAULT_RATIONALE


@patch("backend.app.graph.nodes.genai")
def test_weight_setter_rejects_empty_citation_without_default_phrase(mock_genai):
    """Empty cited_chunk_id without explicit default/no-guidance phrase → full fallback to prevent rationale/weight mismatch."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "weights": {"gst": 0.30, "upi": 0.30, "aa": 0.25, "epfo": 0.15},
        "rationale": [
            {"dimension": "gst", "reasoning": "Strong GST data available.", "cited_chunk_id": ""},
            {"dimension": "upi", "reasoning": "UPI patterns suggest stability.", "cited_chunk_id": ""},
            {"dimension": "aa", "reasoning": "Good repayment record.", "cited_chunk_id": ""},
            {"dimension": "epfo", "reasoning": "Consistent payroll data.", "cited_chunk_id": ""},
        ],
    })
    mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
    state = {"profile": PERSONAS["healthy"], "retrieved_chunks": _FAKE_CHUNKS}
    result = node_weight_setter(state)
    # Entire LLM result rejected; must return documented defaults
    assert result["weights"] == _DEFAULT_WEIGHTS
    assert result["weight_rationale"] == _DEFAULT_RATIONALE


@patch("backend.app.graph.nodes.genai")
def test_weight_setter_falls_back_on_bad_sum_weights(mock_genai):
    """LLM weights that don't sum to 1.0 → WeightVector validation raises → full fallback."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "weights": {"gst": 0.40, "upi": 0.40, "aa": 0.10, "epfo": 0.20},  # sum=1.10
        "rationale": [
            {"dimension": "gst", "reasoning": "Strong GST.", "cited_chunk_id": "chunk-001"},
            {"dimension": "upi", "reasoning": "UPI reliable.", "cited_chunk_id": "chunk-001"},
            {"dimension": "aa", "reasoning": "Repayment ok.", "cited_chunk_id": "chunk-001"},
            {"dimension": "epfo", "reasoning": "Payroll stable.", "cited_chunk_id": "chunk-001"},
        ],
    })
    mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
    state = {"profile": PERSONAS["healthy"], "retrieved_chunks": _FAKE_CHUNKS}
    result = node_weight_setter(state)
    assert result["weights"] == _DEFAULT_WEIGHTS
    assert result["weight_rationale"] == _DEFAULT_RATIONALE


@patch("backend.app.graph.nodes.genai")
def test_weight_setter_falls_back_on_out_of_range_weight(mock_genai):
    """LLM weight outside [0, 1] → WeightVector validation raises → full fallback."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "weights": {"gst": -0.10, "upi": 0.50, "aa": 0.35, "epfo": 0.25},
        "rationale": [
            {"dimension": "gst", "reasoning": "Negative GST.", "cited_chunk_id": "chunk-001"},
            {"dimension": "upi", "reasoning": "UPI ok.", "cited_chunk_id": "chunk-001"},
            {"dimension": "aa", "reasoning": "Repayment ok.", "cited_chunk_id": "chunk-001"},
            {"dimension": "epfo", "reasoning": "Payroll stable.", "cited_chunk_id": "chunk-001"},
        ],
    })
    mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
    state = {"profile": PERSONAS["healthy"], "retrieved_chunks": _FAKE_CHUNKS}
    result = node_weight_setter(state)
    assert result["weights"] == _DEFAULT_WEIGHTS
    assert result["weight_rationale"] == _DEFAULT_RATIONALE


@patch("backend.app.graph.nodes.genai")
def test_weight_setter_rejects_no_guidance_with_hardcoded_pct(mock_genai):
    """No-guidance rationale item that also contains a hard-coded percentage → full fallback.

    The wording "no retrieved guidance — default used, 30%" is misleading when
    the LLM chose a different weight, so the entire result must be rejected.
    """
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "weights": {"gst": 0.35, "upi": 0.30, "aa": 0.20, "epfo": 0.15},
        "rationale": [
            {"dimension": "gst", "reasoning": "Sector guidance at 35%.", "cited_chunk_id": "chunk-001"},
            {"dimension": "upi", "reasoning": "no retrieved guidance — default used, 30%.", "cited_chunk_id": ""},
            {"dimension": "aa", "reasoning": "no retrieved guidance — default used", "cited_chunk_id": ""},
            {"dimension": "epfo", "reasoning": "no retrieved guidance — default used", "cited_chunk_id": ""},
        ],
    })
    mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
    state = {"profile": PERSONAS["healthy"], "retrieved_chunks": _FAKE_CHUNKS}
    result = node_weight_setter(state)
    assert result["weights"] == _DEFAULT_WEIGHTS
    assert result["weight_rationale"] == _DEFAULT_RATIONALE


@patch("backend.app.graph.nodes.genai")
def test_weight_setter_accepts_valid_retrieved_citation(mock_genai):
    """LLM result with valid chunk IDs and explicit default phrases → accepted as-is."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "weights": {"gst": 0.35, "upi": 0.30, "aa": 0.20, "epfo": 0.15},
        "rationale": [
            {"dimension": "gst", "reasoning": "Sector guidance recommends 35%.", "cited_chunk_id": "chunk-001"},
            {"dimension": "upi", "reasoning": "UPI weight per RBI guidance.", "cited_chunk_id": "chunk-001"},
            {"dimension": "aa", "reasoning": "no retrieved guidance — default used", "cited_chunk_id": ""},
            {"dimension": "epfo", "reasoning": "no retrieved guidance — default used", "cited_chunk_id": ""},
        ],
    })
    mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
    state = {"profile": PERSONAS["healthy"], "retrieved_chunks": _FAKE_CHUNKS}
    result = node_weight_setter(state)
    assert result["weights"] == WeightVector(gst=0.35, upi=0.30, aa=0.20, epfo=0.15)
    gst_item = next(r for r in result["weight_rationale"] if r.dimension == "gst")
    assert gst_item.cited_chunk_id == "chunk-001"

