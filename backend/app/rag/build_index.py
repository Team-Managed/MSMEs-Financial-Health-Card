"""
One-time script: chunk PDFs in corpus/ → embed → persist to Chroma.
Run from the backend/ directory:
    uv run python -m app.rag.build_index

Requires PDFs to be placed in backend/app/rag/corpus/ manually (see AGENT.md §4a).
"""
import os
import hashlib
import logging
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CORPUS_DIR = Path(__file__).parent / "corpus"
CHROMA_DIR = Path(__file__).parent / "chroma_store"
CHUNK_SIZE = 500          # tokens (approximate — we split by ~500 words)
CHUNK_OVERLAP = 50        # words
COLLECTION_NAME = "rag_corpus"
EMBED_MODEL = "all-MiniLM-L6-v2"


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def _extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        raise RuntimeError(
            "pypdf not installed — add pypdf>=4.0.0 to pyproject.toml dependencies and run uv sync"
        )


def build_index(corpus_dir: Path = CORPUS_DIR, chroma_dir: Path = CHROMA_DIR) -> None:
    pdfs = list(corpus_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning("No PDFs found in %s — index will be empty", corpus_dir)
        return

    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    col = client.get_or_create_collection(COLLECTION_NAME, embedding_function=ef)

    for pdf_path in pdfs:
        logger.info("Processing %s", pdf_path.name)
        text = _extract_text_from_pdf(pdf_path)
        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{pdf_path.name}:{i}".encode()).hexdigest()[:16]
            col.upsert(
                ids=[chunk_id],
                documents=[chunk],
                metadatas=[{"source": pdf_path.name, "section": f"chunk_{i}"}],
            )
        logger.info("  → %d chunks indexed from %s", len(chunks), pdf_path.name)

    logger.info("Index built: %d total chunks in %s", col.count(), chroma_dir)


if __name__ == "__main__":
    build_index()
