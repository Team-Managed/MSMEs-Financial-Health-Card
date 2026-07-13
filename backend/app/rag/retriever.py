"""
RAG retriever — wraps a persisted ChromaDB collection.
Gracefully returns [] if the index does not exist.
"""
from __future__ import annotations
import logging
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "rag_corpus"
_EMBED_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_CHROMA_DIR = str(Path(__file__).parent / "chroma_store")


class Retriever:
    def __init__(self, chroma_dir: str = _DEFAULT_CHROMA_DIR):
        self._col = None
        if not (Path(chroma_dir) / "chroma.sqlite3").exists():
            logger.warning("RAG index is not built — retriever will return []")
            return
        try:
            self._ef = SentenceTransformerEmbeddingFunction(model_name=_EMBED_MODEL)
            self._client = chromadb.PersistentClient(path=chroma_dir)
            self._col = self._client.get_or_create_collection(
                _COLLECTION_NAME, embedding_function=self._ef
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ChromaDB init failed (%s) — retriever will return []", exc)

    def query(self, text: str, n_results: int = 5) -> list[dict]:
        if self._col is None:
            return []
        try:
            count = self._col.count()
        except Exception:
            return []
        if count == 0:
            logger.warning("RAG corpus is empty — run build_index.py first")
            return []
        actual_n = min(n_results, count)
        res = self._col.query(query_texts=[text], n_results=actual_n)
        chunks = []
        for i, doc_id in enumerate(res["ids"][0]):
            meta = res["metadatas"][0][i]
            chunks.append({
                "chunk_id": doc_id,
                "text": res["documents"][0][i],
                "source": meta.get("source", "unknown"),
                "section": meta.get("section", ""),
            })
        return chunks
