from agentic_rag.graph import run_query


def test_answerable_question_runs_full_path():
    resp = run_query("What is BM25 and why is it used in retrieval?")
    assert resp.path[:2] == ["retrieve", "reason"]
    assert "answer" in resp.path and "verify" in resp.path
    assert resp.answer
    # Any citation returned must correspond to a real retrieved chunk id.
    assert all("::" in c.chunk_id for c in resp.citations)


def test_confidence_in_unit_range():
    resp = run_query("What is a word embedding?")
    assert 0.0 <= resp.retrieval_confidence <= 1.0


def test_reformulation_is_bounded():
    # Even a vague query must terminate within the configured retry budget.
    resp = run_query("tell me about the thing with the vectors")
    assert resp.reformulations <= 2


def test_out_of_corpus_question_abstains():
    # A question the corpus cannot answer should be declined, not guessed.
    resp = run_query("Who won the 2018 FIFA World Cup?")
    assert resp.abstained is True
    assert "abstain" in resp.path
    assert resp.citations == []
