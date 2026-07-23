"""FastAPI surface.

``POST /query`` runs the agentic graph for one question and returns the grounded answer,
its citations, the groundedness verdict, and the path the graph actually took (useful for
debugging retries/abstentions). ``GET /healthz`` reports index readiness.
"""

from __future__ import annotations

from fastapi import FastAPI

from .config import get_settings
from .graph import run_query
from .schemas import QueryRequest, QueryResponse

app = FastAPI(
    title="agentic-rag-evals",
    version="0.1.0",
    description="Evaluation-driven agentic RAG - reference implementation.",
)


@app.get("/healthz")
def healthz() -> dict:
    settings = get_settings()
    index_ready = (settings.index_dir / "chroma.sqlite3").exists()
    return {
        "status": "ok" if index_ready else "no-index",
        "index_ready": index_ready,
        "embedding_backend": settings.rag_embedding_backend,
        "live_llm": settings.has_live_llm,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    return run_query(req.question, top_k=req.top_k)
