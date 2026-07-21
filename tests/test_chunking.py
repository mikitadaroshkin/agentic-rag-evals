import tiktoken

from agentic_rag.ingest import chunk_text

_ENC = tiktoken.get_encoding("cl100k_base")


def test_chunk_text_respects_max_tokens():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_text(text, max_tokens=100, overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        assert len(_ENC.encode(c)) <= 100


def test_chunk_text_overlaps():
    text = " ".join(f"tok{i}" for i in range(300))
    chunks = chunk_text(text, max_tokens=100, overlap=30)
    # Consecutive chunks should share some trailing/leading tokens.
    first_tail = set(chunks[0].split()[-10:])
    second_head = set(chunks[1].split()[:30])
    assert first_tail & second_head


def test_chunk_text_empty():
    assert chunk_text("") == []
