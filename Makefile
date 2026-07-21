.DEFAULT_GOAL := help
PY ?= python

.PHONY: help setup ingest serve test lint eval eval-replay eval-record fetch-corpus docker-build docker-up

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv and install (dev extras)
	uv venv .venv
	uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
	uv pip install --python .venv/bin/python -e ".[dev]"

fetch-corpus: ## Re-fetch the Wikipedia corpus snapshot (needs network)
	$(PY) data/fetch_corpus.py

ingest: ## Build the vector index from the corpus
	$(PY) -m agentic_rag.ingest

serve: ## Run the FastAPI service on :8000
	$(PY) -m uvicorn agentic_rag.api:app --reload --port 8000

test: ## Run the test suite (offline, no key)
	$(PY) -m pytest

lint: ## Lint with ruff
	ruff check .

eval: ## Run the eval harness with the current .env settings
	$(PY) -m evals.run_eval

eval-replay: ## Run the eval offline from recorded cassettes (what CI gates on)
	RAG_CASSETTE=replay $(PY) -m evals.run_eval

eval-record: ## Re-baseline: run against a live model and record cassettes + report (needs OPENAI_API_KEY)
	RAG_CASSETTE=record $(PY) -m evals.run_eval

docker-build: ## Build the container image
	docker build -t agentic-rag-evals:local .

docker-up: ## Build and run via docker compose
	docker compose up --build
