"""Graph node implementations.

Each node is a pure ``RAGState -> partial RAGState`` function wrapped in a trace span.
Node functions never raise on model quirks; they degrade to an honest abstention instead.
"""

from __future__ import annotations

import re

from . import prompts
from .config import get_settings
from .llm import chat
from .retrieval import get_retriever
from .schemas import Chunk, Verdict
from .tracing import span
from .util import loads_json

_CHUNK_ID_RE = re.compile(r"\[([a-zA-Z0-9_:\-]+)\]")
ABSTAIN_MESSAGE = "I don't have enough grounded information in the corpus to answer that."


def retrieve_node(state: dict) -> dict:
    query = state.get("query") or state["question"]
    top_k = state.get("_top_k")
    with span("node.retrieve", query=query):
        chunks, confidence = get_retriever().retrieve(query, top_k)
    trace = state.get("trace", []) + ["retrieve"]
    return {
        "query": query,
        "retrieved": chunks,
        "retrieval_confidence": round(confidence, 4),
        "trace": trace,
    }


def reason_node(state: dict) -> dict:
    settings = get_settings()
    chunks: list[Chunk] = state.get("retrieved", [])
    confidence = state.get("retrieval_confidence", 0.0)
    trace = state.get("trace", []) + ["reason"]

    # Confidence-banded routing (see Settings.rag_confidence_floor/trust).
    if not chunks or confidence < settings.rag_confidence_floor:
        # Retrieval clearly failed -> insufficient without spending an LLM call.
        return {"context_sufficient": False, "_missing": "stronger passages", "trace": trace}
    if confidence >= settings.rag_confidence_trust:
        # Retrieval clearly succeeded -> trust it and answer directly.
        return {"context_sufficient": True, "_missing": "", "trace": trace}

    # Ambiguous band: let the model decide whether the context is enough.
    with span("node.reason", confidence=confidence):
        raw = chat(
            prompts.SUFFICIENCY_SYSTEM,
            prompts.sufficiency_prompt(state["question"], chunks),
            model=settings.rag_llm_model,
            json_mode=True,
        )
    parsed = loads_json(raw)
    sufficient = bool(parsed.get("sufficient", True))
    return {
        "context_sufficient": sufficient,
        "_missing": str(parsed.get("missing", "")),
        "trace": trace,
    }


def reformulate_node(state: dict) -> dict:
    settings = get_settings()
    trace = state.get("trace", []) + ["reformulate"]
    with span("node.reformulate"):
        new_query = chat(
            prompts.REFORMULATE_SYSTEM,
            prompts.reformulate_prompt(state["question"], state.get("_missing", "")),
            model=settings.rag_llm_model,
        )
    new_query = new_query.strip().splitlines()[0] if new_query.strip() else state["question"]
    return {
        "query": new_query,
        "reformulations": state.get("reformulations", 0) + 1,
        "trace": trace,
    }


def answer_node(state: dict) -> dict:
    settings = get_settings()
    chunks: list[Chunk] = state.get("retrieved", [])
    trace = state.get("trace", []) + ["answer"]

    if not chunks:
        return {"answer": ABSTAIN_MESSAGE, "citations": [], "abstained": True, "trace": trace}

    with span("node.answer"):
        raw = chat(
            prompts.ANSWER_SYSTEM,
            prompts.answer_prompt(state["question"], chunks),
            model=settings.rag_llm_model,
        )

    if raw.strip().upper().startswith("INSUFFICIENT_CONTEXT"):
        return {"answer": ABSTAIN_MESSAGE, "citations": [], "abstained": True, "trace": trace}

    valid_ids = {c.id for c in chunks}
    citations = [cid for cid in dict.fromkeys(_CHUNK_ID_RE.findall(raw)) if cid in valid_ids]
    return {"answer": raw.strip(), "citations": citations, "abstained": False, "trace": trace}


def abstain_node(state: dict) -> dict:
    # Reached when retrieval confidence stays below the floor after bounded reformulation:
    # the corpus almost certainly does not contain the answer, so decline instead of guessing.
    trace = state.get("trace", []) + ["abstain"]
    return {
        "answer": ABSTAIN_MESSAGE,
        "citations": [],
        "abstained": True,
        "verdict": Verdict(grounded=True, reason="insufficient retrieval confidence"),
        "trace": trace,
    }


def verify_node(state: dict) -> dict:
    settings = get_settings()
    trace = state.get("trace", []) + ["verify"]

    if state.get("abstained"):
        # Abstaining is already the safe outcome; nothing to verify.
        return {"verdict": Verdict(grounded=True, reason="abstained"), "trace": trace}

    chunks: list[Chunk] = state.get("retrieved", [])
    with span("node.verify"):
        raw = chat(
            prompts.VERIFY_SYSTEM,
            prompts.verify_prompt(state["answer"], chunks),
            model=settings.rag_llm_model,
            json_mode=True,
        )
    parsed = loads_json(raw)
    grounded = bool(parsed.get("grounded", True))
    verdict = Verdict(grounded=grounded, reason=str(parsed.get("reason", "")))

    if not grounded:
        # Guardrail trip: replace an ungrounded answer with an honest abstention.
        return {
            "answer": ABSTAIN_MESSAGE,
            "citations": [],
            "abstained": True,
            "verdict": verdict,
            "trace": trace,
        }
    return {"verdict": verdict, "trace": trace}
