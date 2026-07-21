"""LLM provider abstraction.

Three execution modes, selected by config so the same code path serves production,
free/deterministic CI, and key-less local development:

* **live** — a real OpenAI-compatible chat completion (any base_url: OpenAI, Ollama, vLLM).
* **cassette** — record real responses to disk, then replay them deterministically. This is
  what lets the eval harness gate every PR for free and reproduce headline numbers exactly.
* **FakeLLM** — a deterministic, dependency-free stand-in used when no key is present, so the
  full graph and tests run offline. Its outputs are heuristic, never network-backed.

Everything funnels through :func:`chat`, keyed on the exact (model, system, user) text.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache

from . import prompts
from .config import get_settings
from .tracing import span

# ---------------------------------------------------------------------------
# Cassette store (record / replay)
# ---------------------------------------------------------------------------


def _cassette_key(
    model: str, system: str, user: str, temperature: float, json_mode: bool = False
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "system": system.strip(),
            "user": user.strip(),
            "t": temperature,
            "json": json_mode,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


@lru_cache(maxsize=1)
def _load_cassettes() -> dict[str, dict]:
    settings = get_settings()
    if settings.cassette_path.exists():
        return json.loads(settings.cassette_path.read_text())
    return {}


def _save_cassette(key: str, model: str, user: str, response: str) -> None:
    settings = get_settings()
    store = _load_cassettes()
    # Store a short prompt preview alongside the response so the fixtures are reviewable.
    store[key] = {"model": model, "prompt": user[:280], "response": response}
    settings.cassette_path.parent.mkdir(parents=True, exist_ok=True)
    settings.cassette_path.write_text(json.dumps(store, indent=2, sort_keys=True) + "\n")
    _load_cassettes.cache_clear()


class CassetteMiss(RuntimeError):
    """Raised in replay mode when no recorded response matches the prompt."""


# ---------------------------------------------------------------------------
# Live provider
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI

    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


def _live_chat(model: str, system: str, user: str, temperature: float, json_mode: bool) -> str:
    client = _openai_client()
    kwargs = {
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    if json_mode:
        # Force well-formed JSON for the routing/verify/judge steps.
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Local transformers provider (no key, runs on CPU)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _hf_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = get_settings().rag_local_model
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    model.eval()
    return tokenizer, model


def _local_hf_chat(system: str, user: str, json_mode: bool) -> str:
    import os

    import torch

    tokenizer, model = _hf_model()
    # Generation can use every core; embedding calls reset threads to 1 for their determinism.
    torch.set_num_threads(os.cpu_count() or 1)

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            # Routing/verify/judge emit short JSON; answers need more room.
            max_new_tokens=128 if json_mode else 224,
            do_sample=False,  # greedy -> deterministic for a one-time cassette recording
            pad_token_id=tokenizer.eos_token_id,
        )
    completion = generated[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(completion, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Deterministic offline fallback
# ---------------------------------------------------------------------------

_BLOCK_RE = re.compile(r"\[([a-zA-Z0-9_:\-]+)\] \(source:.*?\)\n(.*?)(?=\n\n\[|\Z)", re.DOTALL)
_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2}


def _fake_chat(system: str, user: str) -> str:
    """Heuristic, deterministic responses. Good enough to exercise every graph branch."""
    if system == prompts.SUFFICIENCY_SYSTEM:
        has_context = "source:" in user
        return json.dumps({"sufficient": has_context, "missing": "" if has_context else "passages"})

    if system == prompts.REFORMULATE_SYSTEM:
        m = re.search(r"Original question:\s*(.+)", user)
        return (m.group(1).strip() if m else user).splitlines()[0]

    if system == prompts.ANSWER_SYSTEM:
        # Extractive: quote the first informative sentence of the top block, cite its id.
        for chunk_id, body in _BLOCK_RE.findall(user):
            # Skip markdown heading lines to reach real prose.
            lines = [ln for ln in body.splitlines() if ln.strip() and not ln.startswith("#")]
            prose = " ".join(lines)
            sentence = prose.strip().split(". ")[0].strip().rstrip(".")
            if len(sentence) > 20:
                return f"{sentence} [{chunk_id}]."
        return "INSUFFICIENT_CONTEXT"

    if system == prompts.VERIFY_SYSTEM:
        return json.dumps({"grounded": True, "reason": "extracted from context"})

    if system == prompts.JUDGE_SYSTEM:
        ref = _extract_field(user, "Reference answer:")
        ans = _extract_field(user, "System answer:")
        rt, at = _tokens(ref), _tokens(ans)
        overlap = len(rt & at) / max(len(rt), 1)
        if overlap >= 0.5:
            label = "correct"
        elif overlap >= 0.25:
            label = "partially_correct"
        else:
            label = "incorrect"
        return json.dumps({"correctness": label, "faithful": True, "reasoning": "token-overlap"})

    return "INSUFFICIENT_CONTEXT"


def _extract_field(text: str, label: str) -> str:
    idx = text.find(label)
    if idx < 0:
        return ""
    rest = text[idx + len(label) :].strip()
    return rest.split("\n\n", 1)[0].strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def chat(
    system: str, user: str, *, model: str, temperature: float = 0.0, json_mode: bool = False
) -> str:
    """Return an assistant completion for (system, user), honouring the cassette mode."""
    settings = get_settings()
    key = _cassette_key(model, system, user, temperature, json_mode)

    with span("llm.chat", **{"llm.model": model, "llm.mode": settings.rag_cassette}):
        if settings.rag_cassette == "replay":
            hit = _load_cassettes().get(key)
            if hit is None:
                raise CassetteMiss(
                    f"No cassette for key={key} (model={model}). "
                    "Prompts changed? Re-record with RAG_CASSETTE=record."
                )
            return hit["response"]

        backend = settings.resolved_llm_backend
        if backend == "openai":
            response = _live_chat(model, system, user, temperature, json_mode)
        elif backend == "local_hf":
            response = _local_hf_chat(system, user, json_mode)
        else:
            # Deterministic offline path (tests, key-less local dev).
            return _fake_chat(system, user)

        if settings.rag_cassette == "record":
            _save_cassette(key, model, user, response)
        return response
