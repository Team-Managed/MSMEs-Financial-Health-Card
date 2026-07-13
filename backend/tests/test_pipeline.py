import pytest
import json
from unittest.mock import patch, MagicMock
from backend.app.graph.pipeline import run_pipeline


@patch("backend.app.graph.nodes.genai")
@patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"})
def test_pipeline_end_to_end(mock_genai, tmp_path):
    """Full pipeline run with mocked LLM calls."""
    weight_resp = MagicMock()
    weight_resp.text = '{"weights": {"gst": 0.30, "upi": 0.30, "aa": 0.25, "epfo": 0.15}, "rationale": [{"dimension": "gst", "reasoning": "ok", "cited_chunk_id": ""}, {"dimension": "upi", "reasoning": "ok", "cited_chunk_id": ""}, {"dimension": "aa", "reasoning": "ok", "cited_chunk_id": ""}, {"dimension": "epfo", "reasoning": "ok", "cited_chunk_id": ""}]}'
    explain_resp = MagicMock()
    explain_resp.text = "CFCR baseline 1.20 is solid. Buyer_loss scenario cuts CFCR. Score 72.00."
    mock_genai.GenerativeModel.return_value.generate_content.side_effect = [
        weight_resp, explain_resp
    ]

    from backend.app.rag.retriever import Retriever
    retriever = Retriever(chroma_dir=str(tmp_path / "empty"))

    result = run_pipeline("healthy", retriever=retriever)

    assert "cfcr_baseline" in result
    assert result["cfcr_baseline"] > 0
    assert "narrative" in result
    assert "grounding_trace" in result
    assert "weight_rationale" in result


@patch("backend.app.graph.nodes._get_gemini_model")
def test_pipeline_makes_distinct_weight_and_explainer_retrieval_calls(mock_get_model):
    class RecordingRetriever:
        def __init__(self):
            self.queries = []

        def query(self, text, n_results=5):
            self.queries.append(text)
            prefix = "weight" if len(self.queries) == 1 else "explainer"
            return [{
                "chunk_id": f"{prefix}-chunk",
                "source": f"{prefix}.pdf",
                "section": "Guidance",
                "text": f"{prefix} evidence",
            }]

    weight_response = MagicMock()
    weight_response.text = json.dumps({
        "weights": {"gst": 0.30, "upi": 0.30, "aa": 0.25, "epfo": 0.15},
        "rationale": [
            {"dimension": dim, "reasoning": "Grounded.", "cited_chunk_id": "weight-chunk"}
            for dim in ("gst", "upi", "aa", "epfo")
        ],
    })
    explainer_response = MagicMock()
    explainer_response.text = "Risk guidance [explainer-chunk]."
    mock_get_model.return_value.generate_content.side_effect = [weight_response, explainer_response]
    retriever = RecordingRetriever()

    result = run_pipeline("buyer_concentrated", retriever=retriever)

    assert len(retriever.queries) == 2
    assert retriever.queries[0] != retriever.queries[1]
    assert result["retrieved_chunks"][0]["chunk_id"] == "weight-chunk"
    assert result["explainer_chunks"][0]["chunk_id"] == "explainer-chunk"
    citation_checks = [item for item in result["grounding_trace"] if item.type == "citation"]
    assert citation_checks[0].source == "explainer-chunk"
    assert citation_checks[0].status == "pass"
