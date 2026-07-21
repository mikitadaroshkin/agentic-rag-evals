"""Typed data structures shared across the pipeline.

The graph state is a plain ``TypedDict`` (LangGraph merges partial updates into it),
while API / retrieval payloads are pydantic models so FastAPI can validate them.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A retrievable unit of the corpus."""

    id: str
    doc_id: str
    title: str
    text: str
    # Populated at retrieval time.
    score: float = 0.0


class Verdict(BaseModel):
    """Output of the groundedness guardrail."""

    grounded: bool
    reason: str = ""


def _keep_last(_existing: str, new: str) -> str:
    """Reducer: later node writes win for scalar string fields."""
    return new


class RAGState(TypedDict, total=False):
    """State threaded through the LangGraph graph.

    ``total=False`` because nodes contribute partial updates. The annotated
    reducers make the merge semantics explicit for the fields that several nodes
    may touch.
    """

    question: str
    # Query actually used for retrieval (may be a reformulation).
    query: Annotated[str, _keep_last]
    reformulations: int
    retrieved: list[Chunk]
    retrieval_confidence: float
    context_sufficient: bool
    answer: str
    citations: list[str]
    verdict: Verdict
    abstained: bool
    trace: list[str]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What is BM25 used for in retrieval?"])
    top_k: int | None = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    title: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    abstained: bool
    grounded: bool
    citations: list[Citation]
    reformulations: int
    retrieval_confidence: float
    path: list[str]


JudgeLabel = Literal["correct", "partially_correct", "incorrect"]


class JudgeResult(BaseModel):
    """Structured LLM-as-judge output for a single answer."""

    correctness: JudgeLabel
    faithful: bool
    reasoning: str = ""

    @property
    def correctness_score(self) -> float:
        return {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}[self.correctness]
