import os
import tempfile
import pytest
from unittest.mock import patch
from backend.app.rag.retriever import Retriever


def test_retriever_returns_empty_list_when_no_corpus(tmp_path):
    """Retriever must degrade gracefully if index not built."""
    r = Retriever(chroma_dir=str(tmp_path / "empty_chroma"))
    results = r.query("MSME credit risk")
    assert results == []


def test_retriever_does_not_load_embeddings_without_an_index(tmp_path):
    with patch("backend.app.rag.retriever.SentenceTransformerEmbeddingFunction") as embedding_factory:
        retriever = Retriever(chroma_dir=str(tmp_path / "missing_chroma"))

    assert retriever.query("MSME credit risk") == []
    embedding_factory.assert_not_called()


def test_retriever_roundtrip(tmp_path):
    """Insert a document, retrieve it."""
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    chroma_dir = str(tmp_path / "chroma")
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=chroma_dir)
    col = client.get_or_create_collection("rag_corpus", embedding_function=ef)
    col.add(
        ids=["chunk_001"],
        documents=["RBI guidelines on MSME credit assessment require alternate data."],
        metadatas=[{"source": "rbi_circular.pdf", "section": "Section 3"}],
    )

    r = Retriever(chroma_dir=chroma_dir)
    results = r.query("MSME alternate data credit", n_results=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk_001"
    assert "source" in results[0]
    assert "text" in results[0]
