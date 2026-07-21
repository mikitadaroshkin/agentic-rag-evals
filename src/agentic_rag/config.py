"""Central configuration, sourced from environment / ``.env``.

One ``Settings`` instance is shared process-wide. Every knob has a safe default so
the pipeline imports and runs with no environment at all (local embeddings + FakeLLM).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    """Locate the project root (which holds ``data/`` and ``evals/``).

    Editable installs resolve it from the package location; for a non-editable/wheel install
    run from the project directory, fall back to the current working directory. Either way,
    ``CORPUS_DIR`` / ``INDEX_DIR`` env vars can override the derived paths outright.
    """
    pkg_root = Path(__file__).resolve().parents[2]
    if (pkg_root / "data" / "corpus").exists():
        return pkg_root
    cwd = Path.cwd()
    if (cwd / "data" / "corpus").exists():
        return cwd
    return pkg_root


ROOT = _project_root()

# Load .env once, before Settings reads the environment.
load_dotenv(ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # --- provider ---
    # Backend for generation + judge: "auto" (openai if a key is set, else the offline
    # FakeLLM), "openai", "local_hf" (a local transformers model), or "fake".
    rag_llm_backend: str = "auto"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    rag_llm_model: str = "gpt-4o-mini"
    rag_judge_model: str = "gpt-4o-mini"
    # Model id used when rag_llm_backend == "local_hf".
    rag_local_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # --- embeddings ---
    rag_embedding_backend: str = "local"  # "local" | "openai"
    rag_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- cassettes ---
    rag_cassette: str = "off"  # "off" | "record" | "replay"

    # --- retrieval / graph ---
    rag_top_k: int = 5
    rag_max_reformulations: int = 2
    rag_rrf_k: int = 60  # Reciprocal Rank Fusion damping constant.
    # `reason` routes on retrieval confidence, consulting the LLM only in the ambiguous band:
    #   confidence < floor          -> insufficient (reformulate, then abstain)
    #   floor <= confidence < trust -> ask the LLM whether the context is sufficient
    #   confidence >= trust         -> sufficient (strong retrieval; answer directly)
    # Floor sits between in-corpus and out-of-corpus confidence; trust marks clearly-good hits.
    rag_confidence_floor: float = 0.25
    rag_confidence_trust: float = 0.45

    # --- paths ---
    corpus_dir: Path = ROOT / "data" / "corpus"
    index_dir: Path = ROOT / "data" / "index"
    cassette_path: Path = ROOT / "evals" / "fixtures" / "cassettes.json"

    # --- tracing ---
    rag_tracing: str = "console"  # "console" | "none" | "otlp"
    otel_exporter_otlp_endpoint: str = ""

    @property
    def resolved_llm_backend(self) -> str:
        """Concrete backend after resolving 'auto'."""
        if self.rag_llm_backend != "auto":
            return self.rag_llm_backend
        return "openai" if self.openai_api_key else "fake"

    @property
    def has_live_llm(self) -> bool:
        """True when a real (non-fake, non-replay) model call would be made."""
        return self.rag_cassette != "replay" and self.resolved_llm_backend in ("openai", "local_hf")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
