from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

import pytest
from opentelemetry.trace import NoOpTracerProvider

from pydantic_ai import Agent, CachePoint, ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.usage import RequestUsage

try:
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
except ImportError:  # pragma: lax no cover
    otel_sdk_installed = False
else:
    otel_sdk_installed = True

pytestmark = pytest.mark.skipif(not otel_sdk_installed, reason='opentelemetry-sdk not installed')


@dataclass(frozen=True)
class CacheUsage:
    read: int = 0
    write: int = 0
    input_tokens: int = 2000
    provider_name: str | None = 'test'
    model_name: str = 'cache-model'


class ResponseNameFunctionModel(FunctionModel):
    def set_response_model_name(self, model_name: str) -> None:
        self._model_name = model_name


def cache_spans(
    usages: Sequence[CacheUsage],
    *,
    retention: timedelta | None = None,
    prompt: str | list[str | CachePoint] = 'prompt',
    tracer_provider: TracerProvider | NoOpTracerProvider | None = None,
) -> tuple[list[ReadableSpan], InMemorySpanExporter | None]:
    exporter = None
    if tracer_provider is None:
        exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    call_index = 0

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_index
        usage = usages[call_index]
        call_index += 1
        model.set_response_model_name(usage.model_name)
        parts = (
            [TextPart('done')]
            if call_index == len(usages)
            else [ToolCallPart('continue_run', {}, tool_call_id=f'call-{call_index}')]
        )
        return ModelResponse(
            parts=parts,
            usage=RequestUsage(
                input_tokens=usage.input_tokens,
                cache_read_tokens=usage.read,
                cache_write_tokens=usage.write,
            ),
            provider_name=usage.provider_name,
        )

    profile = ModelProfile(prompt_cache_retention=retention) if retention is not None else None
    model = ResponseNameFunctionModel(model_function, model_name='cache-model', profile=profile)
    agent = Agent(
        model,
        capabilities=[
            Instrumentation(settings=InstrumentationSettings(tracer_provider=tracer_provider, include_content=False))
        ],
    )

    @agent.tool_plain
    def continue_run() -> str:
        return 'continue'

    agent.run_sync(prompt)
    return (
        [span for span in exporter.get_finished_spans() if span.name.startswith('chat ')] if exporter else []
    ), exporter


def cache_attributes(span: ReadableSpan) -> dict[str, object]:
    assert span.attributes is not None
    return {key: value for key, value in span.attributes.items() if key.startswith('pydantic_ai.cache.')}


def test_stable_cache_health() -> None:
    spans, _ = cache_spans(
        [CacheUsage(write=1500), CacheUsage(read=1500), CacheUsage(read=1600)], retention=timedelta(hours=1)
    )

    assert [cache_attributes(span) for span in spans] == [
        {'pydantic_ai.cache.established_tokens': 1500},
        {'pydantic_ai.cache.hit_ratio': 0.75, 'pydantic_ai.cache.established_tokens': 1500},
        {'pydantic_ai.cache.hit_ratio': 0.8, 'pydantic_ai.cache.established_tokens': 1600},
    ]
    assert all(not span.events for span in spans)


@pytest.mark.parametrize(
    ('retention', 'prompt', 'reason', 'has_event'),
    [
        (timedelta(hours=1), 'prompt', 'unexpected', True),
        (timedelta(0), 'prompt', 'ttl-expired', False),
        (None, 'prompt', 'unknown', False),
        (timedelta(0), ['context', CachePoint(ttl='1h')], 'unexpected', True),
    ],
)
def test_cache_collapse_classification(
    retention: timedelta | None, prompt: str | list[str | CachePoint], reason: str, has_event: bool
) -> None:
    spans, _ = cache_spans([CacheUsage(write=1400), CacheUsage(read=100)], retention=retention, prompt=prompt)

    assert cache_attributes(spans[-1]) == {
        'pydantic_ai.cache.hit_ratio': 0.05,
        'pydantic_ai.cache.established_tokens': 100,
        'pydantic_ai.cache.collapsed': True,
        'pydantic_ai.cache.wasted_tokens': 1300,
        'pydantic_ai.cache.collapse_reason': reason,
    }
    assert [event.name for event in spans[-1].events] == (['pydantic_ai.cache.collapse'] if has_event else [])
    if has_event:
        assert dict(spans[-1].events[0].attributes or {}) == {
            'established_tokens': 1400,
            'cache_read_tokens': 100,
            'wasted_tokens': 1300,
            'provider_name': 'test',
            'model_name': 'cache-model',
        }


def test_collapse_event_without_provider_name() -> None:
    spans, _ = cache_spans(
        [CacheUsage(write=1400, provider_name=None), CacheUsage(read=100, provider_name=None)],
        retention=timedelta(hours=1),
    )

    (event,) = spans[-1].events
    assert event.name == 'pydantic_ai.cache.collapse'
    assert 'provider_name' not in (event.attributes or {})


def test_model_switch_and_switch_back() -> None:
    spans, _ = cache_spans(
        [
            CacheUsage(write=1400, model_name='first'),
            CacheUsage(model_name='second'),
            CacheUsage(read=100, model_name='first'),
        ],
        retention=timedelta(hours=1),
    )

    assert cache_attributes(spans[1]) == {}
    assert cache_attributes(spans[2])['pydantic_ai.cache.collapse_reason'] == 'unexpected'


def test_sub_threshold_collapse_and_rebaseline() -> None:
    spans, _ = cache_spans(
        [
            CacheUsage(write=1000),
            CacheUsage(read=100),
            CacheUsage(write=1400),
            CacheUsage(read=100),
            CacheUsage(read=100),
        ],
        retention=timedelta(hours=1),
    )

    assert 'pydantic_ai.cache.collapsed' not in cache_attributes(spans[1])
    assert cache_attributes(spans[3])['pydantic_ai.cache.collapsed'] is True
    assert 'pydantic_ai.cache.collapsed' not in cache_attributes(spans[4])
    assert len([event for span in spans for event in span.events if event.name == 'pydantic_ai.cache.collapse']) == 1


def test_no_cache_attributes_and_run_aggregate() -> None:
    no_cache_spans, exporter = cache_spans([CacheUsage()])
    assert exporter is not None
    assert cache_attributes(no_cache_spans[0]) == {}
    run_span = next(span for span in exporter.get_finished_spans() if span.name.startswith('invoke_agent '))
    assert 'pydantic_ai.cache.hit_ratio' not in (run_span.attributes or {})

    _, exporter = cache_spans([CacheUsage(read=500)])
    assert exporter is not None
    run_span = next(span for span in exporter.get_finished_spans() if span.name.startswith('invoke_agent '))
    assert (run_span.attributes or {})['pydantic_ai.cache.hit_ratio'] == 0.25


def test_cache_marks_update_without_recording() -> None:
    spans, exporter = cache_spans(
        [CacheUsage(write=1400), CacheUsage(read=100), CacheUsage(read=100)],
        retention=timedelta(hours=1),
        tracer_provider=NoOpTracerProvider(),
    )
    assert spans == []
    assert exporter is None
