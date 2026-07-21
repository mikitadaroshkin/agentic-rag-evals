"""Corpus ingestion: load -> token-aware chunk -> embed -> persist to Chroma.

Idempotent: re-running rebuilds the collection from scratch so the index always matches
the corpus on disk. Run via ``rag-ingest`` (console script) or ``python -m agentic_rag.ingest``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
import tiktoken

from .config import get_settings
from .embeddings import get_embedder
from .schemas import Chunk

COLLECTION = "corpus"
_ENCODER = tiktoken.get_encoding("cl100k_base")


def _title_of(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def chunk_text(text: str, *, max_tokens: int = 256, overlap: int = 40) -> list[str]:
    """Token-aware sliding-window chunking on the cl100k tokenizer."""
    tokens = _ENCODER.encode(text)
    if not tokens:
        return []
    step = max(1, max_tokens - overlap)
    chunks = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + max_tokens]
        chunks.append(_ENCODER.decode(window).strip())
        if start + max_tokens >= len(tokens):
            break
    return [c for c in chunks if c]


def load_corpus(corpus_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        doc_id = path.stem
        title = _title_of(raw, doc_id)
        for i, body in enumerate(chunk_text(raw)):
            chunks.append(Chunk(id=f"{doc_id}::{i}", doc_id=doc_id, title=title, text=body))
    return chunks


def build_index(corpus_dir: Path | None = None, index_dir: Path | None = None) -> int:
    settings = get_settings()
    corpus_dir = corpus_dir or settings.corpus_dir
    index_dir = index_dir or settings.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_corpus(corpus_dir)
    if not chunks:
        raise SystemExit(f"No .md documents found in {corpus_dir}")

    embedder = get_embedder()
    vectors = embedder.embed([c.text for c in chunks])

    client = chromadb.PersistentClient(path=str(index_dir))
    # Rebuild cleanly so the index can never drift from the corpus.
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[c.id for c in chunks],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        metadatas=[{"doc_id": c.doc_id, "title": c.title} for c in chunks],
    )
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the corpus into the vector store.")
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--index", type=Path, default=None)
    args = parser.parse_args()
    n = build_index(args.corpus, args.index)
    settings = get_settings()
    print(f"Indexed {n} chunks -> {args.index or settings.index_dir}")


if __name__ == "__main__":
    main()
