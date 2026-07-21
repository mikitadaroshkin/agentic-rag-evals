"""Hybrid retrieval: dense (Chroma) + lexical (BM25), fused with Reciprocal Rank Fusion.

RRF is a rank-based fusion that needs no score calibration between the two retrievers —
each contributes ``1 / (k + rank)`` per document, which is robust when cosine similarities
and BM25 scores live on incomparable scales. The top dense cosine similarity is surfaced as
a retrieval-confidence signal the graph uses to decide whether to reformulate and retry.
"""

from __future__ import annotations

import re
from functools import lru_cache

import chromadb

from .config import get_settings
from .embeddings import get_embedder
from .ingest import COLLECTION
from .schemas import Chunk

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _rrf(ranklists: list[list[str]], k: int) -> dict[str, float]:
    """Reciprocal Rank Fusion over several ordered id lists."""
    scores: dict[str, float] = {}
    for ranklist in ranklists:
        for rank, doc_id in enumerate(ranklist):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


class HybridRetriever:
    def __init__(self):
        from rank_bm25 import BM25Okapi

        settings = get_settings()
        client = chromadb.PersistentClient(path=str(settings.index_dir))
        self.col = client.get_collection(COLLECTION)

        data = self.col.get(include=["documents", "metadatas"])
        self.ids: list[str] = data["ids"]
        self.docs: list[str] = data["documents"]
        self.metas: list[dict] = data["metadatas"]
        self._by_id = {
            cid: Chunk(id=cid, doc_id=m["doc_id"], title=m["title"], text=doc)
            for cid, doc, m in zip(self.ids, self.docs, self.metas, strict=True)
        }
        self.bm25 = BM25Okapi([_tokenize(d) for d in self.docs])
        self.embedder = get_embedder()

    def _dense(self, query: str, n: int) -> tuple[list[str], dict[str, float]]:
        qv = self.embedder.embed([query])[0]
        res = self.col.query(query_embeddings=[qv], n_results=min(n, len(self.ids)))
        ids = res["ids"][0]
        # Chroma cosine distance -> similarity.
        sims = {i: 1.0 - d for i, d in zip(ids, res["distances"][0], strict=True)}
        return ids, sims

    def _lexical(self, query: str, n: int) -> list[str]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.ids[i] for i in ranked[:n] if scores[i] > 0]

    def retrieve(self, query: str, top_k: int | None = None) -> tuple[list[Chunk], float]:
        settings = get_settings()
        top_k = top_k or settings.rag_top_k
        pool = max(top_k * 4, 20)

        dense_ids, sims = self._dense(query, pool)
        lexical_ids = self._lexical(query, pool)
        fused = _rrf([dense_ids, lexical_ids], settings.rag_rrf_k)

        # Sort by fused score desc, breaking ties on chunk id so the order is stable
        # under tiny float perturbations (reproducible cassettes).
        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        chunks = []
        for cid, score in ordered:
            c = self._by_id[cid].model_copy(update={"score": round(score, 5)})
            chunks.append(c)

        # Confidence = best dense cosine similarity in [0, 1] (0 when nothing dense-matched).
        confidence = max((sims.get(c.id, 0.0) for c in chunks), default=0.0)
        confidence = max(0.0, min(1.0, confidence))
        return chunks, confidence


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()
