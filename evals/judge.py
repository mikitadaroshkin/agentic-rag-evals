"""LLM-as-judge scoring for answer quality.

The judge grades a system answer against the golden reference on two axes: correctness
(does it convey the same key facts?) and faithfulness (is it free of contradicting claims?).
It runs through the same cassette layer as the rest of the pipeline, so judged scores are
reproducible and free in replay mode.
"""

from __future__ import annotations

from agentic_rag import prompts
from agentic_rag.config import get_settings
from agentic_rag.llm import chat
from agentic_rag.schemas import JudgeResult
from agentic_rag.util import loads_json

_VALID = ("correct", "partially_correct", "incorrect")


def _coerce_correctness(parsed: dict, raw: str) -> str:
    """Read the correctness label, tolerating small-model output that isn't clean JSON."""
    label = str(parsed.get("correctness", "")).lower().strip()
    if label in _VALID:
        return label
    text = raw.lower()
    if "partial" in text:
        return "partially_correct"
    if "incorrect" in text or "wrong" in text or '"false"' in text:
        return "incorrect"
    if "correct" in text:
        return "correct"
    return "incorrect"


def judge_answer(question: str, reference: str, answer: str) -> JudgeResult:
    settings = get_settings()
    raw = chat(
        prompts.JUDGE_SYSTEM,
        prompts.judge_prompt(question, reference, answer),
        model=settings.rag_judge_model,
        json_mode=True,
    )
    parsed = loads_json(raw)
    faithful = parsed.get("faithful")
    if faithful is None:
        faithful = "unfaithful" not in raw.lower() and "not faithful" not in raw.lower()
    return JudgeResult(
        correctness=_coerce_correctness(parsed, raw),
        faithful=bool(faithful),
        reasoning=str(parsed.get("reasoning", ""))[:200],
    )
