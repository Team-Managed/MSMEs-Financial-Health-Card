"""
RAG retriever — wraps a persisted ChromaDB collection.
Gracefully returns [] if the index does not exist.
"""
from __future__ import annotations
import logging
import math
import os
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "rag_corpus_v2"
_EMBED_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_CHROMA_DIR = str(Path(__file__).parent / "chroma_store")
_DEFAULT_MAX_DISTANCE = 0.8


def _validate_max_distance(value: float | str, source: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be a finite number in [0, 2]") from exc

    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 2.0:
        raise ValueError(f"{source} must be a finite number in [0, 2]")
    return parsed


class Retriever:
    def __init__(
        self,
        chroma_dir: str = _DEFAULT_CHROMA_DIR,
        collection_name: str = _COLLECTION_NAME,
        max_distance: float | None = None,
        collection=None,
        embedding_function=None,
    ):
        if max_distance is None:
            raw = os.environ.get("RAG_MAX_DISTANCE", str(_DEFAULT_MAX_DISTANCE))
            self._max_distance = _validate_max_distance(raw, "RAG_MAX_DISTANCE")
        else:
            self._max_distance = _validate_max_distance(max_distance, "max_distance")
        self._collection_name = collection_name
        self._ef = embedding_function or SentenceTransformerEmbeddingFunction(model_name=_EMBED_MODEL)
        self._col = collection

        if self._col is not None:
            return

        try:
            self._client = chromadb.PersistentClient(path=chroma_dir)
            self._col = self._client.get_collection(
                name=self._collection_name,
                embedding_function=self._ef,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ChromaDB collection unavailable (%s) — retriever will return []", exc)
            self._col = None

    def query(self, text: str, n_results: int = 5) -> list[dict]:
        if isinstance(n_results, bool) or not isinstance(n_results, int) or n_results < 1:
            raise ValueError("n_results must be an integer >= 1")
        if self._col is None:
            return []

        metadata = getattr(self._col, "metadata", None) or {}
        if metadata.get("hnsw:space") not in (None, "cosine"):
            logger.warning(
                "Collection %s uses incompatible distance space %r",
                self._collection_name,
                metadata.get("hnsw:space"),
            )
            return []

        try:
            count = self._col.count()
        except Exception:  # noqa: BLE001
            return []
        if count == 0:
            logger.warning("RAG corpus is empty — run build_index.py first")
            return []
        actual_n = min(n_results, count)
        try:
            res = self._col.query(
                query_texts=[text],
                n_results=actual_n,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAG query failed (%s)", exc)
            return []

        chunks = []
        for i, doc_id in enumerate(res["ids"][0]):
            meta = res["metadatas"][0][i]
            distance = float(res.get("distances", [[0.0]])[0][i])
            if distance > self._max_distance:
                continue

            page_number = meta.get("page_number")
            page = meta.get("page", page_number)
            chunks.append({
                "chunk_id": doc_id,
                "text": res["documents"][0][i],
                "source": meta.get("source", "unknown"),
                "section": meta.get("section", ""),
                "page": page,
                "page_number": page_number if page_number is not None else page,
                "distance": distance,
            })
        return chunks
