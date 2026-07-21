"""LangGraph assembly.

    retrieve -> reason --(sufficient)----> answer -> verify --(grounded)--> END
        ^          |  (insufficient,                   |  (ungrounded)
        |          |   retries left)                   v
        +----- reformulate                           abstain -> END
                   |  (insufficient, retries exhausted) ^
                   +------------------------------------+

The loop back through ``reformulate`` is what makes this agentic rather than a linear
pipeline: from a real retrieval-confidence signal (and an LLM sufficiency check in the
ambiguous band), the graph decides whether to answer now, rewrite the query and retry,
or abstain. The ``verify`` guardrail can also route a would-be answer to ``abstain``.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from .config import get_settings
from .nodes import (
    abstain_node,
    answer_node,
    reason_node,
    reformulate_node,
    retrieve_node,
    verify_node,
)
from .schemas import Citation, QueryResponse, RAGState


def _route_after_reason(state: dict) -> str:
    settings = get_settings()
    if state.get("context_sufficient"):
        return "answer"
    if state.get("reformulations", 0) < settings.rag_max_reformulations:
        return "reformulate"
    # Retries exhausted and context still insufficient: abstain rather than guess.
    return "abstain"


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("reason", reason_node)
    g.add_node("reformulate", reformulate_node)
    g.add_node("answer", answer_node)
    g.add_node("verify", verify_node)
    g.add_node("abstain", abstain_node)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "reason")
    g.add_conditional_edges(
        "reason",
        _route_after_reason,
        {"answer": "answer", "reformulate": "reformulate", "abstain": "abstain"},
    )
    g.add_edge("reformulate", "retrieve")
    g.add_edge("answer", "verify")
    g.add_edge("verify", END)
    g.add_edge("abstain", END)
    return g.compile()


def run_graph_state(question: str, top_k: int | None = None) -> dict:
    """Execute the graph and return the raw final state (used by the eval harness)."""
    initial: dict = {"question": question, "reformulations": 0, "trace": []}
    if top_k is not None:
        initial["_top_k"] = top_k
    return build_graph().invoke(initial)


def run_query(question: str, top_k: int | None = None) -> QueryResponse:
    """Execute the graph for one question and shape the result for the API."""
    final = run_graph_state(question, top_k)

    by_id = {c.id: c for c in final.get("retrieved", [])}
    citations = [
        Citation(chunk_id=cid, doc_id=by_id[cid].doc_id, title=by_id[cid].title)
        for cid in final.get("citations", [])
        if cid in by_id
    ]
    verdict = final.get("verdict")
    return QueryResponse(
        question=question,
        answer=final.get("answer", ""),
        abstained=bool(final.get("abstained", False)),
        grounded=bool(verdict.grounded) if verdict else True,
        citations=citations,
        reformulations=final.get("reformulations", 0),
        retrieval_confidence=final.get("retrieval_confidence", 0.0),
        path=final.get("trace", []),
    )
