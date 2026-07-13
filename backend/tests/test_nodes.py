import os
import json
import pytest
from unittest.mock import patch, MagicMock
from backend.app.data.personas import PERSONAS
from backend.app.schemas.models import WeightVector
from backend.app.graph.nodes import (
    _DEFAULT_RATIONALE,
    _DEFAULT_WEIGHTS,
    _llm_model,
    node_aggregator,
    node_sector_retriever,
    node_weight_setter,
    node_stress_generator,
    node_risk_engine,
)


def test_llm_model_defaults_to_gemini_flash_lite(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert _llm_model() == "gemini-3.1-flash-lite"


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


def test_weight_setter_returns_weight_vector():
    """With no RAG chunks the node returns default weights without calling the LLM."""
    state = {
        "profile": PERSONAS["healthy"],
        "retrieved_chunks": [],
    }
    result = node_weight_setter(state)
    assert "weights" in result
    assert isinstance(result["weights"], WeightVector)


def _weight_setter_response(payload):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps(payload)
    response.usage = None
    return response


@patch("backend.app.graph.nodes._llm_model", return_value="test-model")
@patch("backend.app.graph.nodes._llm_client")
def test_weight_setter_falls_back_for_unbalanced_weights(mock_client_fn, mock_model_fn):
    mock_client_fn.return_value.chat.completions.create.return_value = _weight_setter_response({
        "weights": {"gst": 0.5, "upi": 0.5, "aa": 0.5, "epfo": 0.5},
        "rationale": [],
    })

    result = node_weight_setter({
        "profile": PERSONAS["healthy"],
        "retrieved_chunks": [{"chunk_id": "c001", "text": "Guidance", "source": "test", "section": "s1"}],
    })

    assert result["weights"] == _DEFAULT_WEIGHTS
    assert result["weight_rationale"] == _DEFAULT_RATIONALE


@patch("backend.app.graph.nodes._llm_model", return_value="test-model")
@patch("backend.app.graph.nodes._llm_client")
def test_weight_setter_falls_back_for_incomplete_rationale(mock_client_fn, mock_model_fn):
    mock_client_fn.return_value.chat.completions.create.return_value = _weight_setter_response({
        "weights": {"gst": 0.3, "upi": 0.3, "aa": 0.25, "epfo": 0.15},
        "rationale": [
            {"dimension": "gst", "reasoning": "Guidance", "cited_chunk_id": "c001"},
            {"dimension": "gst", "reasoning": "Guidance", "cited_chunk_id": "c001"},
        ],
    })

    result = node_weight_setter({
        "profile": PERSONAS["healthy"],
        "retrieved_chunks": [{"chunk_id": "c001", "text": "Guidance", "source": "test", "section": "s1"}],
    })

    assert result["weights"] == _DEFAULT_WEIGHTS
    assert result["weight_rationale"] == _DEFAULT_RATIONALE
