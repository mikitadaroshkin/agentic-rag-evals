# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    RAG_TRACING=none

WORKDIR /app

# libgomp1 is required by onnxruntime (a chromadb dependency).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better layer caching. CPU-only torch keeps the image lean.
# Editable install so `agentic_rag` resolves its data/index paths relative to /app.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -e .

COPY data ./data
COPY evals ./evals

# Bake the embedding model + vector index into the image so the container starts ready.
RUN python -m agentic_rag.ingest

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "agentic_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
