from __future__ import annotations

import hashlib

from backend.app.rag import build_index


def test_chunk_pages_preserves_page_boundaries_and_sentence_overlap():
    first_sentence = " ".join(["Opening"] + ["context"] * 179) + "."
    overlap_sentence = " ".join(["Shared"] + ["guidance"] * 59) + "."
    final_sentence = " ".join(["Closing"] + ["detail"] * 179) + "."
    pages = [f"Credit Assessment\n\n{first_sentence} {overlap_sentence} {final_sentence}", "Second page only."]

    chunks = build_index._chunk_pages(pages, target_words=350, max_words=400, overlap_words=50)

    page_one = [chunk for chunk in chunks if chunk.page_number == 1]
    page_two = [chunk for chunk in chunks if chunk.page_number == 2]
    assert len(page_one) == 2
    assert len(page_two) == 1
    assert overlap_sentence in page_one[0].text
    assert overlap_sentence in page_one[1].text
    assert all(len(chunk.text.split()) <= 400 for chunk in chunks)
    assert all("Second page only" not in chunk.text for chunk in page_one)
    assert page_one[0].section == "Credit Assessment"
    assert page_two[0].section == "Page 2"


def test_chunk_pages_safely_splits_one_overlong_sentence_and_skips_blanks():
    overlong = " ".join(f"word{i}" for i in range(925)) + "."

    chunks = build_index._chunk_pages(["  \n", overlong], target_words=350, max_words=400, overlap_words=50)

    assert chunks
    assert {chunk.page_number for chunk in chunks} == {2}
    assert all(0 < len(chunk.text.split()) <= 400 for chunk in chunks)
    assert "word0" in chunks[0].text
    assert "word924" in chunks[-1].text


def test_chunk_id_and_metadata_are_deterministic_and_versioned():
    checksum = hashlib.sha256(b"pdf-content").hexdigest()
    chunk = build_index.PageChunk(
        text="MSME lending guidance.",
        page_number=3,
        section="Credit Risk",
        chunk_index=1,
    )

    first = build_index._chunk_id(checksum, chunk)
    second = build_index._chunk_id(checksum, chunk)
    metadata = build_index._chunk_metadata("source.pdf", checksum, chunk)

    assert first == second
    assert first != build_index._chunk_id("different-checksum", chunk)
    assert metadata == {
        "source": "source.pdf",
        "page": 3,
        "page_number": 3,
        "section": "Credit Risk",
        "chunk_index": 1,
        "source_checksum": checksum,
        "source_version": f"sha256:{checksum}",
    }


def test_build_index_recreates_collection_to_remove_stale_records(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    pdf = corpus / "guidance.pdf"
    pdf.write_bytes(b"version-one")

    class FakeCollection:
        def __init__(self):
            self.records = {}

        def upsert(self, ids, documents, metadatas):
            for record_id, document, metadata in zip(ids, documents, metadatas):
                self.records[record_id] = (document, metadata)

        def count(self):
            return len(self.records)

    class FakeClient:
        def __init__(self):
            self.collection = None
            self.deleted = []

        def delete_collection(self, name):
            self.deleted.append(name)
            if self.collection is None:
                raise ValueError("missing")
            self.collection = None

        def create_collection(self, name, embedding_function, metadata):
            assert metadata["hnsw:space"] == "cosine"
            self.collection = FakeCollection()
            return self.collection

    fake_client = FakeClient()
    monkeypatch.setattr(build_index.chromadb, "PersistentClient", lambda path: fake_client)
    monkeypatch.setattr(build_index, "_extract_pages_from_pdf", lambda path: ["First version sentence."])

    build_index.build_index(corpus, tmp_path / "chroma", embedding_function=object())
    first_ids = set(fake_client.collection.records)
    assert len(first_ids) == 1

    pdf.write_bytes(b"version-two-shorter")
    monkeypatch.setattr(build_index, "_extract_pages_from_pdf", lambda path: ["Replacement sentence."])
    build_index.build_index(corpus, tmp_path / "chroma", embedding_function=object())

    assert fake_client.deleted == [build_index.COLLECTION_NAME, build_index.COLLECTION_NAME]
    assert len(fake_client.collection.records) == 1
    assert not first_ids.intersection(fake_client.collection.records)