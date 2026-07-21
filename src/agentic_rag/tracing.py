"""OpenTelemetry setup.

Spans wrap each graph node, retrieval, and LLM call so a run is inspectable end to
end. Defaults to the console exporter (no collector needed); set ``RAG_TRACING=otlp``
plus ``OTEL_EXPORTER_OTLP_ENDPOINT`` to ship to a real collector.
"""

from __future__ import annotations

from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from .config import get_settings

_CONFIGURED = False


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings = get_settings()
    provider = TracerProvider(resource=Resource.create({"service.name": "agentic-rag-evals"}))

    if settings.rag_tracing == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif settings.rag_tracing == "otlp" and settings.otel_exporter_otlp_endpoint:
        # Imported lazily so the OTLP exporter is an optional dependency at runtime.
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint))
        )
    # "none" -> provider with no processors (spans are created but dropped).

    trace.set_tracer_provider(provider)
    _CONFIGURED = True


def get_tracer() -> trace.Tracer:
    _configure()
    return trace.get_tracer("agentic_rag")


@contextmanager
def span(name: str, **attributes):
    """Convenience context manager that starts a span and sets attributes."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as s:
        for key, value in attributes.items():
            s.set_attribute(key, value)
        yield s
