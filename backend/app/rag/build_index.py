"""Build a page-aware Chroma index for the RAG corpus.

Run from backend/:
    uv run python -m app.rag.build_index
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
import re

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CORPUS_DIR = Path(__file__).parent / "corpus"
CHROMA_DIR = Path(__file__).parent / "chroma_store"
COLLECTION_NAME = "rag_corpus_v2"
EMBED_MODEL = "all-MiniLM-L6-v2"
TARGET_WORDS = 420
MAX_WORDS = 500
OVERLAP_WORDS = 50


@dataclass(frozen=True)
class PageChunk:
    text: str
    page_number: int
    section: str
    chunk_index: int


def _word_count(text: str) -> int:
    return len(text.split())


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_paragraph_into_units(paragraph: str) -> list[str]:
    sentence_like = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph.strip()) if s.strip()]
    return sentence_like or [paragraph.strip()]


def _split_overlong_unit(unit: str, max_words: int, overlap_words: int) -> list[str]:
    words = unit.split()
    if len(words) <= max_words:
        return [unit]

    step = max(max_words - overlap_words, 1)
    slices: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        slices.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return slices


def _section_label(page_text: str, page_number: int) -> str:
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    if not lines:
        return f"Page {page_number}"

    candidate = lines[0]
    words = candidate.split()
    if 1 <= len(words) <= 12 and not re.search(r"[.!?;:]$", candidate):
        return candidate
    return f"Page {page_number}"


def _chunk_page_text(
    page_text: str,
    page_number: int,
    section: str,
    target_words: int,
    max_words: int,
    overlap_words: int,
) -> list[PageChunk]:
    paragraphs = [
        _normalize_space(part)
        for part in re.split(r"\n\s*\n+", page_text)
        if _normalize_space(part)
    ]
    if not paragraphs:
        return []

    units: list[str] = []
    for paragraph in paragraphs:
        for sentence in _split_paragraph_into_units(paragraph):
            normalized = _normalize_space(sentence)
            if not normalized:
                continue
            units.extend(_split_overlong_unit(normalized, max_words=max_words, overlap_words=overlap_words))

    chunks: list[PageChunk] = []
    i = 0
    chunk_index = 0

    while i < len(units):
        j = i
        current: list[str] = []
        current_words = 0

        while j < len(units):
            candidate_words = _word_count(units[j])
            if current and current_words + candidate_words > max_words:
                break
            current.append(units[j])
            current_words += candidate_words
            j += 1
            if current_words >= target_words:
                break

        if not current:
            j = i + 1
            current = [units[i]]

        chunk_text = _normalize_space(" ".join(current))
        if chunk_text:
            chunks.append(PageChunk(
                text=chunk_text,
                page_number=page_number,
                section=section,
                chunk_index=chunk_index,
            ))
            chunk_index += 1

        if j >= len(units):
            break

        overlap_acc = 0
        next_i = j
        for k in range(j - 1, i - 1, -1):
            overlap_acc += _word_count(units[k])
            if overlap_acc >= overlap_words:
                next_i = k
                break
        if next_i <= i:
            next_i = i + 1
        i = next_i

    return chunks


def _chunk_pages(
    pages: list[str],
    target_words: int = TARGET_WORDS,
    max_words: int = MAX_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> list[PageChunk]:
    chunks: list[PageChunk] = []
    for page_idx, raw_page in enumerate(pages, start=1):
        text = (raw_page or "").strip()
        if not text:
            continue
        section = _section_label(text, page_idx)
        page_fallback = f"Page {page_idx}"
        if section != page_fallback:
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if line.strip() == section:
                    body = "\n".join(lines[i + 1 :]).strip()
                    text = body or text
                    break
        chunks.extend(_chunk_page_text(
            text,
            page_number=page_idx,
            section=section,
            target_words=target_words,
            max_words=max_words,
            overlap_words=overlap_words,
        ))
    return chunks


def _extract_pages_from_pdf(pdf_path: Path) -> list[str]:
    try:
        import pypdf
    except ImportError as exc:
        raise RuntimeError(
            "pypdf not installed — add pypdf>=4.0.0 and sync environment"
        ) from exc

    reader = pypdf.PdfReader(str(pdf_path))
    return [(page.extract_text() or "") for page in reader.pages]


def _file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _chunk_id(source_checksum: str, chunk: PageChunk) -> str:
    text_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:16]
    raw = f"{source_checksum}:{chunk.page_number}:{chunk.chunk_index}:{text_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _chunk_metadata(source_name: str, source_checksum: str, chunk: PageChunk) -> dict:
    return {
        "source": source_name,
        "page": chunk.page_number,
        "page_number": chunk.page_number,
        "section": chunk.section,
        "chunk_index": chunk.chunk_index,
        "source_checksum": source_checksum,
        "source_version": f"sha256:{source_checksum}",
    }


def _create_v2_collection(client, embedding_function):
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted prior collection %s", COLLECTION_NAME)
    except Exception:
        logger.info("No existing collection %s to delete", COLLECTION_NAME)

    return client.create_collection(
        COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )


def build_index(
    corpus_dir: Path = CORPUS_DIR,
    chroma_dir: Path = CHROMA_DIR,
    embedding_function=None,
) -> None:
    pdfs = sorted(corpus_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning("No PDFs found in %s — index build skipped", corpus_dir)
        return

    ef = embedding_function or SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    col = _create_v2_collection(client, ef)

    total_pages = 0
    blank_pages = 0
    total_chunks = 0
    skipped_chunks = 0

    for pdf_path in pdfs:
        checksum = _file_checksum(pdf_path)
        pages = _extract_pages_from_pdf(pdf_path)
        total_pages += len(pages)
        blank_pages += sum(1 for page in pages if not (page or "").strip())

        chunks = _chunk_pages(pages)
        if not chunks:
            logger.warning("Skipping %s: no extractable non-blank chunks", pdf_path.name)
            continue

        for chunk in chunks:
            if not chunk.text.strip():
                skipped_chunks += 1
                continue

            col.upsert(
                ids=[_chunk_id(checksum, chunk)],
                documents=[chunk.text],
                metadatas=[_chunk_metadata(pdf_path.name, checksum, chunk)],
            )
            total_chunks += 1

        logger.info(
            "Indexed %s: pages=%d chunks=%d checksum=%s",
            pdf_path.name,
            len(pages),
            len(chunks),
            checksum[:12],
        )

    logger.info(
        "Index build complete collection=%s total_pages=%d blank_pages=%d total_chunks=%d skipped_chunks=%d stored=%d",
        COLLECTION_NAME,
        total_pages,
        blank_pages,
        total_chunks,
        skipped_chunks,
        col.count(),
    )


if __name__ == "__main__":
    build_index()
