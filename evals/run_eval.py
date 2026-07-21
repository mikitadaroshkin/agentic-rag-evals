"""Offline evaluation harness — the project's centrepiece.

Runs every golden question through the agentic graph, then reports:

* **Retrieval quality** (deterministic): hit@k, recall@k, MRR vs gold supporting docs.
* **Answer quality** (LLM-as-judge): correctness + faithfulness vs gold reference answers.
* **Abstention behaviour**: does the guardrail abstain on out-of-corpus questions, and how
  often does it wrongly abstain on answerable ones?

The result is checked against ``evals/thresholds.json`` and the process exits non-zero on any
regression, which is exactly what the CI workflow gates every PR on. Run in ``replay`` cassette
mode for a free, deterministic result; run with a live key in ``record`` mode to (re)baseline.

    python -m evals.run_eval                 # uses current .env / cassette mode
    python -m evals.run_eval --limit 5       # quick subset
    python -m evals.run_eval --no-judge      # retrieval metrics only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag.config import ROOT, get_settings
from agentic_rag.embeddings import get_embedder
from agentic_rag.graph import run_graph_state
from agentic_rag.ingest import build_index
from evals.judge import judge_answer
from evals.metrics import cosine, hit_at_k, mean, ranked_unique_docs, recall_at_k, reciprocal_rank

GOLDEN_PATH = ROOT / "evals" / "golden.jsonl"
THRESHOLDS_PATH = ROOT / "evals" / "thresholds.json"
DEFAULT_REPORT = ROOT / "evals" / "reports" / "latest.json"


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ensure_index() -> None:
    settings = get_settings()
    if not (settings.index_dir / "chroma.sqlite3").exists():
        n = build_index()
        print(f"[eval] built index ({n} chunks)")


def evaluate(limit: int | None = None, use_judge: bool = True) -> dict:
    settings = get_settings()
    ensure_index()
    golden = load_golden()
    if limit:
        golden = golden[:limit]

    items = []
    for row in golden:
        state = run_graph_state(row["question"])
        retrieved_docs = ranked_unique_docs([c.doc_id for c in state.get("retrieved", [])])
        abstained = bool(state.get("abstained", False))
        answer = state.get("answer", "")
        expected_abstain = bool(row.get("expected_abstain", False))

        item = {
            "id": row["id"],
            "question": row["question"],
            "expected_abstain": expected_abstain,
            "abstained": abstained,
            "reformulations": state.get("reformulations", 0),
            "retrieved_docs": retrieved_docs,
            "answer": answer,
        }

        if expected_abstain:
            item["abstain_correct"] = abstained
        else:
            support = row["supporting_docs"]
            item["hit"] = hit_at_k(retrieved_docs, support)
            item["recall"] = recall_at_k(retrieved_docs, support)
            item["rr"] = reciprocal_rank(retrieved_docs, support)
            item["reference"] = row["reference"]
            if use_judge:
                jr = judge_answer(row["question"], row["reference"], answer)
                item["correctness"] = jr.correctness
                item["correctness_score"] = jr.correctness_score
                item["faithful"] = jr.faithful
                item["judge_reasoning"] = jr.reasoning
        items.append(item)

    _attach_answer_similarity(items)
    return _aggregate(items, settings, use_judge)


def _attach_answer_similarity(items: list[dict]) -> None:
    """Deterministic answer-quality signal: cosine(answer, reference) in embedding space.

    Complements the LLM-as-judge with a reproducible number that needs no model calls.
    """
    scored = [it for it in items if not it["expected_abstain"]]
    if not scored:
        return
    embedder = get_embedder()
    answer_vecs = embedder.embed([it["answer"] for it in scored])
    ref_vecs = embedder.embed([it["reference"] for it in scored])
    for it, av, rv in zip(scored, answer_vecs, ref_vecs, strict=True):
        it["answer_similarity"] = round(cosine(av, rv), 4)


def _aggregate(items: list[dict], settings, use_judge: bool) -> dict:
    answerable = [it for it in items if not it["expected_abstain"]]
    controls = [it for it in items if it["expected_abstain"]]

    retrieval = {
        "k": settings.rag_top_k,
        "hit_at_k": round(mean([it["hit"] for it in answerable]), 4),
        "recall_at_k": round(mean([it["recall"] for it in answerable]), 4),
        "mrr": round(mean([it["rr"] for it in answerable]), 4),
    }
    answer = {}
    if answerable:
        # Deterministic, always available.
        answer["answer_similarity"] = round(
            mean([it.get("answer_similarity", 0.0) for it in answerable]), 4
        )
    if use_judge and answerable:
        correct = [float(it["correctness"] == "correct") for it in answerable]
        scores = [it["correctness_score"] for it in answerable]
        answer["answer_correctness"] = round(mean(scores), 4)
        answer["pct_correct"] = round(mean(correct), 4)
        answer["faithfulness"] = round(mean([float(it["faithful"]) for it in answerable]), 4)
    abstain_acc = (
        round(mean([float(it["abstain_correct"]) for it in controls]), 4) if controls else None
    )
    abstention = {
        "abstain_accuracy": abstain_acc,
        "false_abstain_rate": round(mean([float(it["abstained"]) for it in answerable]), 4),
    }

    backend = settings.resolved_llm_backend
    gen_model = settings.rag_local_model if backend == "local_hf" else settings.rag_llm_model
    judge_model = settings.rag_local_model if backend == "local_hf" else settings.rag_judge_model
    return {
        "config": {
            "llm_backend": backend,
            "llm_model": gen_model,
            "judge_model": judge_model,
            "embedding_backend": settings.rag_embedding_backend,
            "embedding_model": settings.rag_embedding_model,
            "top_k": settings.rag_top_k,
            "cassette_mode": settings.rag_cassette,
            "judged": use_judge,
        },
        "counts": {
            "answerable": len(answerable),
            "abstain_controls": len(controls),
            "total": len(items),
        },
        "retrieval": retrieval,
        "answer": answer,
        "abstention": abstention,
        "mean_reformulations": round(mean([float(it["reformulations"]) for it in items]), 3),
        "items": items,
    }


def check_thresholds(report: dict, thresholds: dict) -> tuple[bool, list[str]]:
    flat = {**report["retrieval"], **report.get("answer", {})}
    if report["abstention"].get("abstain_accuracy") is not None:
        flat["abstain_accuracy"] = report["abstention"]["abstain_accuracy"]

    failures = []
    for metric, floor in thresholds.items():
        if metric.startswith("_"):
            continue
        value = flat.get(metric)
        if value is None:
            continue  # metric not produced in this run mode (e.g. --no-judge)
        if value < floor:
            failures.append(f"{metric}={value:.3f} < floor {floor:.3f}")
    return (len(failures) == 0), failures


def _print_summary(report: dict, passed: bool, failures: list[str]) -> None:
    r, a, ab = report["retrieval"], report.get("answer", {}), report["abstention"]
    c = report["config"]
    print("\n" + "=" * 60)
    print(f"  agentic-rag-evals — {report['counts']['answerable']} answerable + "
          f"{report['counts']['abstain_controls']} controls")
    print(f"  llm={c['llm_model']}  judge={c['judge_model']}  emb={c['embedding_backend']}  "
          f"mode={c['cassette_mode']}")
    print("-" * 60)
    print(f"  Retrieval   hit@{r['k']}={r['hit_at_k']:.3f}  "
          f"recall@{r['k']}={r['recall_at_k']:.3f}  MRR={r['mrr']:.3f}")
    if "answer_similarity" in a:
        line = f"  Answer      similarity={a['answer_similarity']:.3f}"
        if "answer_correctness" in a:
            line += (f"  correctness={a['answer_correctness']:.3f}"
                     f"  pct_correct={a['pct_correct']:.3f}"
                     f"  faithfulness={a['faithfulness']:.3f}")
        print(line)
    print(f"  Abstention  accuracy={ab['abstain_accuracy']}  "
          f"false_abstain={ab['false_abstain_rate']:.3f}")
    print(f"  Reformulations (mean) = {report['mean_reformulations']}")
    print("-" * 60)
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    for f in failures:
        print(f"    - {f}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N items")
    parser.add_argument(
        "--no-judge", action="store_true", help="skip LLM-as-judge (retrieval metrics only)"
    )
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_REPORT, help="where to write the JSON report"
    )
    parser.add_argument(
        "--no-gate", action="store_true", help="do not exit non-zero on threshold failure"
    )
    args = parser.parse_args()

    report = evaluate(limit=args.limit, use_judge=not args.no_judge)

    thresholds = json.loads(THRESHOLDS_PATH.read_text())
    passed, failures = check_thresholds(report, thresholds)
    report["thresholds"] = {k: v for k, v in thresholds.items() if not k.startswith("_")}
    report["passed"] = passed
    report["failures"] = failures

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    _print_summary(report, passed, failures)
    print(f"[eval] report -> {args.report}")

    if not passed and not args.no_gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
