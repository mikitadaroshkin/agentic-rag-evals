# Corpus

A small, coherent corpus of **information-retrieval / ML / RAG concept articles** — chosen so
that retrieval is meaningful and the golden Q/A are substantive (a RAG system answering questions
about retrieval).

## Provenance & license

The documents in `corpus/` are plain-text extracts of English Wikipedia articles, fetched with
[`fetch_corpus.py`](fetch_corpus.py). Each file ends with an attribution line linking back to its
source article.

- **Source:** English Wikipedia (see per-file attribution footer for the exact article + URL).
- **License:** text is licensed under **CC BY-SA 4.0**. Reuse must keep attribution and preserve
  the same license.
- **Snapshot:** the extracts are committed so the repo is self-contained and CI is deterministic.
  Re-fetch a fresh snapshot with `python data/fetch_corpus.py` (requires network).

No client, proprietary, or personal data is present anywhere in this repository.

## Articles

| slug | article |
|------|---------|
| `information-retrieval` | Information retrieval |
| `bm25` | Okapi BM25 |
| `vector-space-model` | Vector space model |
| `tf-idf` | Tf–idf |
| `word-embedding` | Word embedding |
| `sentence-embedding` | Sentence embedding |
| `cosine-similarity` | Cosine similarity |
| `precision-and-recall` | Precision and recall |
| `ir-evaluation-measures` | Evaluation measures (information retrieval) |
| `nearest-neighbor-search` | Nearest neighbor search |
| `learning-to-rank` | Learning to rank |
| `retrieval-augmented-generation` | Retrieval-augmented generation |
| `transformer` | Transformer (deep learning architecture) |
| `large-language-model` | Large language model |
| `question-answering` | Question answering |
