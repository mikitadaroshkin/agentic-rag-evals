import json

import pytest

from agentic_rag import llm, prompts
from agentic_rag.config import Settings


def _use_settings(monkeypatch, **overrides):
    settings = Settings(openai_api_key="", **overrides)
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    llm._load_cassettes.cache_clear()
    return settings


def test_fake_llm_returns_valid_sufficiency_json(monkeypatch):
    _use_settings(monkeypatch, rag_cassette="off")
    out = llm.chat(prompts.SUFFICIENCY_SYSTEM, "Question: x\n\nsource: doc", model="fake")
    parsed = json.loads(out)
    assert set(parsed) >= {"sufficient", "missing"}


def test_fake_llm_judge_scores_by_overlap(monkeypatch):
    _use_settings(monkeypatch, rag_cassette="off")
    ref = "cosine similarity measures the angle between vectors"
    user = prompts.judge_prompt("q", ref, ref)
    parsed = json.loads(llm.chat(prompts.JUDGE_SYSTEM, user, model="fake"))
    assert parsed["correctness"] == "correct"


def test_cassette_record_then_replay_roundtrip(tmp_path, monkeypatch):
    path = tmp_path / "cassettes.json"
    # Seed a cassette entry by hand, then confirm replay serves it.
    _use_settings(monkeypatch, rag_cassette="replay", cassette_path=path)
    key = llm._cassette_key("m", "sys", "user", 0.0)
    path.write_text(json.dumps({key: {"model": "m", "prompt": "user", "response": "RECORDED"}}))
    llm._load_cassettes.cache_clear()
    assert llm.chat("sys", "user", model="m") == "RECORDED"


def test_cassette_replay_miss_raises(tmp_path, monkeypatch):
    path = tmp_path / "empty.json"
    path.write_text("{}")
    _use_settings(monkeypatch, rag_cassette="replay", cassette_path=path)
    llm._load_cassettes.cache_clear()
    with pytest.raises(llm.CassetteMiss):
        llm.chat("sys", "unseen prompt", model="m")
