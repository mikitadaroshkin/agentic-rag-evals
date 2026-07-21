"""Shared fixtures.

Tests run fully offline: no API key, deterministic FakeLLM, local embeddings. The vector
index is built once per session if it is not already present.
"""

from __future__ import annotations

import os

import pytest

# Force the offline path regardless of the developer's local .env.
os.environ["RAG_TRACING"] = "none"
os.environ["RAG_CASSETTE"] = "off"
os.environ["OPENAI_API_KEY"] = ""


@pytest.fixture(scope="session", autouse=True)
def _index() -> None:
    from agentic_rag.config import get_settings
    from agentic_rag.ingest import build_index

    settings = get_settings()
    if not (settings.index_dir / "chroma.sqlite3").exists():
        build_index()
