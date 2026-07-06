import pytest
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
