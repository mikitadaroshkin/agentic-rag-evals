import pytest

from agentic_rag.retrieval import get_retriever


@pytest.mark.parametrize(
    "query,expected_doc",
    [
        ("What is BM25 ranking?", "bm25"),
        ("cosine similarity between vectors", "cosine-similarity"),
        ("retrieval augmented generation for LLMs", "retrieval-augmented-generation"),
    ],
)
def test_hybrid_retrieval_finds_expected_doc(query, expected_doc):
    chunks, confidence = get_retriever().retrieve(query, top_k=5)
    assert chunks, "retrieval returned nothing"
    assert 0.0 <= confidence <= 1.0
    assert expected_doc in {c.doc_id for c in chunks}


def test_retrieval_is_deterministic():
    r = get_retriever()
    first = [c.id for c in r.retrieve("vector space model", top_k=5)[0]]
    second = [c.id for c in r.retrieve("vector space model", top_k=5)[0]]
    assert first == second
