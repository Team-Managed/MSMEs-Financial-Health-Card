import pytest
from unittest.mock import patch, MagicMock
from backend.app.graph.pipeline import run_pipeline


@patch("backend.app.graph.nodes._llm_model", return_value="test-model")
@patch("backend.app.graph.nodes._llm_client")
@patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"})
def test_pipeline_end_to_end(mock_client_fn, mock_model_fn, tmp_path):
    """Full pipeline run with mocked OpenAI-compatible LLM calls."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "CFCR baseline 1.20 is solid. Buyer_loss scenario cuts CFCR. Score 72.00."
    mock_response.usage = None
    mock_client_fn.return_value.chat.completions.create.return_value = mock_response

    from backend.app.rag.retriever import Retriever
    retriever = Retriever(chroma_dir=str(tmp_path / "empty"))

    result = run_pipeline("healthy", retriever=retriever)

    assert "cfcr_baseline" in result
    assert result["cfcr_baseline"] > 0
    assert "narrative" in result
    assert "grounding_trace" in result
    assert "weight_rationale" in result
