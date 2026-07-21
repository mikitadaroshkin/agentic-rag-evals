"""Deterministic retrieval metrics.

These need no LLM: they compare the ranked list of document ids the pipeline retrieved
against the gold supporting documents for each question. Computed at the document level
(chunks deduplicated to their parent document, order preserved).
"""

from __future__ import annotations

import math


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors (returns 0 for a zero vector)."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def ranked_unique_docs(chunk_doc_ids: list[str]) -> list[str]:
    """Deduplicate a ranked list of chunk doc-ids, keeping first-seen order."""
    return list(dict.fromkeys(chunk_doc_ids))


def hit_at_k(retrieved_docs: list[str], supporting: list[str]) -> float:
    return 1.0 if set(retrieved_docs) & set(supporting) else 0.0


def recall_at_k(retrieved_docs: list[str], supporting: list[str]) -> float:
    if not supporting:
        return 0.0
    found = len(set(retrieved_docs) & set(supporting))
    return found / len(supporting)


def reciprocal_rank(retrieved_docs: list[str], supporting: list[str]) -> float:
    support = set(supporting)
    for i, doc in enumerate(retrieved_docs, start=1):
        if doc in support:
            return 1.0 / i
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
