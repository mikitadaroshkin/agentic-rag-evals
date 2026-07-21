"""Fetch the demo corpus from Wikipedia (CC BY-SA 4.0) and write it to ``data/corpus/``.

The corpus is a coherent set of information-retrieval / ML / RAG concept articles, so
retrieval and the golden Q/A are substantive. The fetched snapshot is committed to the
repo for reproducibility; this script exists for provenance and refresh. Text is licensed
CC BY-SA 4.0; per-file attribution and the source URL are written into each document.

Usage:
    python data/fetch_corpus.py
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx

API = "https://en.wikipedia.org/w/api.php"

# (Wikipedia article title, output slug)
ARTICLES = [
    ("Information retrieval", "information-retrieval"),
    ("Okapi BM25", "bm25"),
    ("Vector space model", "vector-space-model"),
    ("Tf–idf", "tf-idf"),
    ("Word embedding", "word-embedding"),
    ("Sentence embedding", "sentence-embedding"),
    ("Cosine similarity", "cosine-similarity"),
    ("Precision and recall", "precision-and-recall"),
    ("Evaluation measures (information retrieval)", "ir-evaluation-measures"),
    ("Nearest neighbor search", "nearest-neighbor-search"),
    ("Learning to rank", "learning-to-rank"),
    ("Retrieval-augmented generation", "retrieval-augmented-generation"),
    ("Transformer (deep learning architecture)", "transformer"),
    ("Large language model", "large-language-model"),
    ("Question answering", "question-answering"),
]

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
MAX_CHARS = 16000  # keep each document modest; enough for several chunks


def fetch_extract(client: httpx.Client, title: str) -> tuple[str, str]:
    resp = client.get(
        API,
        params={
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "exsectionformat": "plain",
            "redirects": "1",
            "format": "json",
            "titles": title,
        },
    )
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    return page["title"], page.get("extract", "")


def clean(text: str) -> str:
    # Drop boilerplate tail sections; collapse excess blank lines.
    tail_markers = ("\nSee also", "\nReferences", "\nExternal links", "\nFurther reading")
    for marker in tail_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:MAX_CHARS].rsplit("\n", 1)[0]


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    ua = "agentic-rag-evals/0.1 (+https://github.com/mikitadaroshkin/agentic-rag-evals) httpx"
    with httpx.Client(timeout=30, headers={"User-Agent": ua}) as client:
        for title, slug in ARTICLES:
            resolved, extract = fetch_extract(client, title)
            body = clean(extract)
            url = "https://en.wikipedia.org/wiki/" + resolved.replace(" ", "_")
            doc = (
                f"# {resolved}\n\n{body}\n\n"
                f"---\nSource: [{resolved}]({url}) — Wikipedia, "
                f"licensed under CC BY-SA 4.0.\n"
            )
            (CORPUS_DIR / f"{slug}.md").write_text(doc, encoding="utf-8")
            print(f"wrote {slug}.md  ({len(body)} chars)  <- {resolved}")


if __name__ == "__main__":
    main()
