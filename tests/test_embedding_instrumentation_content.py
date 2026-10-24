from __future__ import annotations

import json

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pydantic_ai import Embedder
from pydantic_ai.embeddings import TestEmbeddingModel
from pydantic_ai.models.instrumented import InstrumentationSettings


def _instrumented_span(*, include_content: bool):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    embedder = Embedder(
        TestEmbeddingModel(dimensions=3),
        instrument=InstrumentationSettings(
            tracer_provider=provider,
            include_content=include_content,
        ),
    )
    result = embedder.embed_query_sync('hello')
    span, = exporter.get_finished_spans()
    provider.shutdown()
    return result, span


def test_instrumented_embeddings_include_vectors_when_content_enabled():
    result, span = _instrumented_span(include_content=True)

    assert span.attributes is not None
    assert span.attributes['embeddings'] == json.dumps(result.embeddings)


def test_instrumented_embeddings_omit_vectors_when_content_disabled():
    _, span = _instrumented_span(include_content=False)

    assert span.attributes is not None
    assert 'embeddings' not in span.attributes
