from backend.app.rag.retriever import Retriever


def test_retriever_returns_empty_list_when_no_corpus(tmp_path):
    """Retriever must degrade gracefully if index not built."""
    r = Retriever(chroma_dir=str(tmp_path / "empty_chroma"))
    results = r.query("MSME credit risk")
    assert results == []


def test_retriever_roundtrip(tmp_path):
    """Insert a document, retrieve it."""
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    chroma_dir = str(tmp_path / "chroma")
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=chroma_dir)
    col = client.get_or_create_collection(
        "rag_corpus_v2", embedding_function=ef, metadata={"hnsw:space": "cosine"}
    )
    col.add(
        ids=["chunk_001"],
        documents=["RBI guidelines on MSME credit assessment require alternate data."],
        metadatas=[{
            "source": "rbi_circular.pdf",
            "section": "Section 3",
            "page": 4,
            "page_number": 4,
            "chunk_index": 0,
            "source_checksum": "abc123",
            "source_version": "sha256:abc123",
        }],
    )

    r = Retriever(chroma_dir=chroma_dir)
    results = r.query("MSME alternate data credit", n_results=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk_001"
    assert "source" in results[0]
    assert "text" in results[0]
    assert results[0]["page"] == 4
    assert results[0]["page_number"] == 4
    assert results[0]["section"] == "Section 3"
    assert isinstance(results[0]["distance"], float)


def test_retriever_filters_by_distance_and_returns_metadata():
    class FakeCollection:
        metadata = {"hnsw:space": "cosine"}

        def count(self):
            return 2

        def query(self, **kwargs):
            assert kwargs["include"] == ["documents", "metadatas", "distances"]
            return {
                "ids": [["near", "far"]],
                "documents": [["Relevant guidance", "Unrelated material"]],
                "metadatas": [[
                    {"source": "guide.pdf", "page_number": 2, "section": "Signals"},
                    {"source": "guide.pdf", "page_number": 9, "section": "Other"},
                ]],
                "distances": [[0.25, 0.91]],
            }

    retriever = Retriever(collection=FakeCollection(), max_distance=0.8)

    assert retriever.query("alternate data", n_results=2) == [{
        "chunk_id": "near",
        "text": "Relevant guidance",
        "source": "guide.pdf",
        "section": "Signals",
        "page": 2,
        "page_number": 2,
        "distance": 0.25,
    }]


def test_retriever_validates_n_results():
    retriever = Retriever(collection=None, embedding_function=object())

    for invalid in (0, -1, True, 1.5):
        try:
            retriever.query("query", n_results=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for n_results={invalid!r}")
