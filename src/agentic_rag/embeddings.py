"""Embedding backends.

Default is a local ``sentence-transformers`` model - real vectors, no key, no network at
query time once the model is cached. The hosted backend (OpenAI-compatible) is a drop-in swap
for teams that prefer a managed embedding endpoint.
"""

from __future__ import annotations

from functools import lru_cache

from .config import get_settings


class LocalEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or (
            self.model.get_sentence_embedding_dimension
        )
        self.dim = get_dim()

    def embed(self, texts: list[str]) -> list[list[float]]:
        import torch

        # Single-threaded encoding keeps float reductions deterministic across machines, so
        # retrieval order (and the recorded eval cassettes) reproduce in CI. Set per-call
        # because a co-resident local generation model may raise the global thread count.
        torch.set_num_threads(1)
        vecs = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return vecs.tolist()


class OpenAIEmbedder:
    def __init__(self, model_name: str):
        from openai import OpenAI

        settings = get_settings()
        self.client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        self.model_name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model_name, input=texts)
        return [d.embedding for d in resp.data]


@lru_cache(maxsize=1)
def get_embedder():
    settings = get_settings()
    if settings.rag_embedding_backend == "openai":
        return OpenAIEmbedder(settings.rag_embedding_model)
    return LocalEmbedder(settings.rag_embedding_model)
