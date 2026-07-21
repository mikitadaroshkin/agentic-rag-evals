from evals.metrics import (
    hit_at_k,
    mean,
    ranked_unique_docs,
    recall_at_k,
    reciprocal_rank,
)


def test_ranked_unique_docs_preserves_first_seen_order():
    assert ranked_unique_docs(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_hit_at_k():
    assert hit_at_k(["a", "b"], ["b"]) == 1.0
    assert hit_at_k(["a", "b"], ["z"]) == 0.0


def test_recall_at_k_partial_and_full():
    assert recall_at_k(["a", "b", "c"], ["a", "b"]) == 1.0
    assert recall_at_k(["a", "x"], ["a", "b"]) == 0.5
    assert recall_at_k(["x"], []) == 0.0  # undefined -> 0, never divide by zero


def test_reciprocal_rank_uses_first_hit():
    assert reciprocal_rank(["x", "a", "b"], ["a"]) == 0.5
    assert reciprocal_rank(["a", "x"], ["a"]) == 1.0
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_mean_handles_empty():
    assert mean([]) == 0.0
    assert mean([1.0, 0.0]) == 0.5
