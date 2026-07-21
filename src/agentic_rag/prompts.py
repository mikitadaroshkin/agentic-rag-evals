"""Prompt templates.

Kept in one module so a prompt change is a reviewable diff — and, because the eval
cassettes are keyed on the exact prompt text, any change here forces a re-record of
the fixtures (which is exactly when the eval numbers should be re-examined).
"""

from __future__ import annotations

from .schemas import Chunk


def format_context(chunks: list[Chunk]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(f"[{c.id}] (source: {c.title})\n{c.text}")
    return "\n\n".join(blocks)


# --- reason: is the retrieved context enough to answer? ---
SUFFICIENCY_SYSTEM = (
    "You are the routing step of a RAG agent. Given a question and retrieved context, "
    "decide whether the context is sufficient to answer the question accurately. "
    'Reply with strict JSON: {"sufficient": true|false, "missing": "<what is missing, if any>"}.'
)


def sufficiency_prompt(question: str, chunks: list[Chunk]) -> str:
    return (
        f"Question: {question}\n\n"
        f"Retrieved context:\n{format_context(chunks)}\n\n"
        "Is this context sufficient to answer the question? Respond with JSON only."
    )


# --- reformulate: rewrite the query for a better second retrieval pass ---
REFORMULATE_SYSTEM = (
    "You rewrite search queries for a dense+lexical retriever. Given the original "
    "question and what was missing from the first retrieval, produce ONE improved "
    "search query. Reply with the query text only, no preamble."
)


def reformulate_prompt(question: str, missing: str) -> str:
    return (
        f"Original question: {question}\n"
        f"Missing from first retrieval: {missing or 'relevant passages'}\n"
        "Improved search query:"
    )


# --- answer: grounded generation with citations ---
ANSWER_SYSTEM = (
    "You are a careful assistant answering strictly from the provided context. "
    "Cite the chunk ids you used inline as [chunk_id]. If the context does not "
    "contain the answer, say exactly: INSUFFICIENT_CONTEXT. Do not use outside knowledge."
)


def answer_prompt(question: str, chunks: list[Chunk]) -> str:
    return (
        f"Context:\n{format_context(chunks)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, citing chunk ids inline as [chunk_id]."
    )


# --- verify: groundedness guardrail ---
VERIFY_SYSTEM = (
    "You are a groundedness checker. Given the context and a candidate answer, decide "
    "whether every factual claim in the answer is supported by the context. "
    'Reply with strict JSON: {"grounded": true|false, "reason": "<short reason>"}.'
)


def verify_prompt(answer: str, chunks: list[Chunk]) -> str:
    return (
        f"Context:\n{format_context(chunks)}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        "Is the answer fully supported by the context? Respond with JSON only."
    )


# --- LLM-as-judge (offline eval) ---
JUDGE_SYSTEM = (
    "You are grading a RAG system's answer against a reference answer. Judge correctness "
    "(does it convey the same key facts as the reference?) and faithfulness (is it free of "
    "claims that contradict the reference?). Be strict but fair to paraphrase. "
    'Reply with strict JSON: '
    '{"correctness": "correct|partially_correct|incorrect", "faithful": true|false, '
    '"reasoning": "<one sentence>"}.'
)


def judge_prompt(question: str, reference: str, answer: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Reference answer: {reference}\n\n"
        f"System answer: {answer}\n\n"
        "Grade the system answer. Respond with JSON only."
    )
