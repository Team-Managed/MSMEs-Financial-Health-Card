"""
Layer 5: Agent Trajectory Evaluation.

Verifies that the LangGraph pipeline:
  - Executes all 7 nodes, producing the expected final state keys (no node skipped)
  - Executes nodes in the canonical order defined in golden_dataset.json
  - Calls the retriever exactly once per pipeline run

Source of truth for expected order and call counts: golden_dataset.json.
In production, the same properties can be verified against LangSmith trace logs;
these tests provide the same guarantee without requiring a live LangSmith connection.

All tests in this module run without GOOGLE_API_KEY (LLM calls are mocked).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

GOLDEN = json.loads(
    (Path(__file__).parent / "golden_dataset.json").read_text()
)
EXPECTED_ORDER = GOLDEN["expected_node_order"]
PERSONA_IDS = list(GOLDEN["personas"].keys())


# ── Shared fixtures ───────────────────────────────────────────────────────────

class _RecordingRetriever:
    """Fake Retriever that records query calls and returns empty chunks."""

    def __init__(self):
        self.call_count = 0

    def query(self, text: str, n_results: int = 5) -> list[dict]:
        self.call_count += 1
        return []


@pytest.fixture(autouse=True)
def _reset_compiled_graph():
    """Force pipeline recompilation before and after each test."""
    import backend.app.graph.pipeline as _pm
    _pm._COMPILED_GRAPH = None
    yield
    _pm._COMPILED_GRAPH = None


def _mock_agent_explain():
    """Return a patcher + mock response for the explainer's chat.completions.create call."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "CFCR baseline pass."
    mock_response.usage = None
    patcher = patch("backend.app.graph.nodes._gemini")
    return patcher, mock_response


# ── Layer 5a: State completeness (all nodes produced their keys) ──────────────

@pytest.mark.parametrize("persona_id", PERSONA_IDS)
def test_all_nodes_produce_expected_state_keys(persona_id):
    """Final state must contain keys from every node — proves none was skipped."""
    from backend.app.graph.pipeline import run_pipeline

    patcher, mock_response = _mock_agent_explain()
    with patcher as mock_client_fn:
        mock_client_fn.return_value.chat.completions.create.return_value = mock_response
        result = run_pipeline(persona_id, retriever=_RecordingRetriever())

    expected_keys = {
        "profile",           # aggregator
        "retrieved_chunks",  # sector_retriever
        "weights",           # weight_setter  (empty corpus → defaults, still sets key)
        "scenarios",         # stress_generator
        "risk_output",       # risk_engine
        "narrative",         # explainer
        "grounding_trace",   # grounding_validator
    }
    missing = expected_keys - result.keys()
    assert not missing, f"[{persona_id}] Missing state keys: {missing}"


# ── Layer 5b: Execution order ─────────────────────────────────────────────────

@pytest.mark.parametrize("persona_id", PERSONA_IDS)
def test_node_execution_order(persona_id):
    """
    Nodes must execute in the canonical order from golden_dataset.json.
    Strategy: patch each node function in the pipeline module's namespace with a
    call-recording wrapper, reset _COMPILED_GRAPH so the pipeline rebuilds using
    the patched references, then assert the recorded order.
    """
    import backend.app.graph.pipeline as _pm
    from backend.app.graph.pipeline import run_pipeline

    call_log: list[str] = []

    # Capture original node functions from pipeline module's namespace
    original_nodes = {
        name: getattr(_pm, f"node_{name}")
        for name in EXPECTED_ORDER
    }

    def _make_recorder(node_name: str, original_fn):
        def _wrapper(state: dict) -> dict:
            call_log.append(node_name)
            return original_fn(state)
        return _wrapper

    patched = {
        f"node_{name}": _make_recorder(name, fn)
        for name, fn in original_nodes.items()
    }

    patcher, mock_response = _mock_agent_explain()
    with patcher as mock_client_fn:
        mock_client_fn.return_value.chat.completions.create.return_value = mock_response
        with patch.multiple("backend.app.graph.pipeline", **patched):
            _pm._COMPILED_GRAPH = None  # force recompile with wrapped functions
            run_pipeline(persona_id, retriever=_RecordingRetriever())

    assert call_log == EXPECTED_ORDER, (
        f"[{persona_id}] Unexpected node execution order.\n"
        f"  Expected: {EXPECTED_ORDER}\n"
        f"  Got:      {call_log}"
    )


# ── Layer 5c: Retriever call count ────────────────────────────────────────────

@pytest.mark.parametrize("persona_id", PERSONA_IDS)
def test_retriever_called_exactly_once_per_run(persona_id):
    """sector_retriever must call retriever.query exactly once per pipeline run."""
    from backend.app.graph.pipeline import run_pipeline

    retriever = _RecordingRetriever()
    expected_calls = GOLDEN["expected_retriever_calls_per_profile"]

    patcher, mock_response = _mock_agent_explain()
    with patcher as mock_client_fn:
        mock_client_fn.return_value.chat.completions.create.return_value = mock_response
        run_pipeline(persona_id, retriever=retriever)

    assert retriever.call_count == expected_calls, (
        f"[{persona_id}] Expected retriever.query called {expected_calls}x, "
        f"got {retriever.call_count}x"
    )
