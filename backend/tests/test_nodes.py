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


def test_weight_setter_returns_weight_vector():
    """With no RAG chunks the node returns default weights without calling the LLM."""
    state = {
        "profile": PERSONAS["healthy"],
        "retrieved_chunks": [],
    }
    result = node_weight_setter(state)
    assert "weights" in result
    assert isinstance(result["weights"], WeightVector)
