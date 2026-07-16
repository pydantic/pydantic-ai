from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest
from pytest_mock import MockerFixture

from pydantic_ai import Agent, CachePoint, ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.usage import RequestUsage

from .conftest import try_import

with try_import() as otel_sdk_imports_successful:
    from opentelemetry.context import Context
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.trace.sampling import Decision, Sampler, SamplingResult

    class DropFirstChatSpanSampler(Sampler):
        """Drop the first `chat` span and record everything else, to exercise non-recording spans."""

        def __init__(self) -> None:
            self._chat_spans_seen = 0

        def should_sample(
            self, parent_context: Context | None, trace_id: int, name: str, *args: object, **kwargs: object
        ) -> SamplingResult:
            if name.startswith('chat '):
                self._chat_spans_seen += 1
                if self._chat_spans_seen == 1:
                    return SamplingResult(Decision.DROP)
            return SamplingResult(Decision.RECORD_AND_SAMPLE)

        def get_description(self) -> str:  # pragma: no cover - required by the ABC, never called here
            return 'DropFirstChatSpanSampler'


pytestmark = pytest.mark.skipif(not otel_sdk_imports_successful(), reason='opentelemetry-sdk not installed')

# These are unit tests rather than VCR tests: they pin the *derived* span attributes, span events,
# and sampling/recording behavior of the `Instrumentation` capability for preset usage sequences,
# which requires exact control over `cache_read/write_tokens` per step — recorded provider traffic
# can't produce deterministic cache-token sequences (cache state on the provider side is not
# reproducible at playback time).


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
    sampler: Sampler | None = None,
) -> tuple[list[ReadableSpan], InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider(sampler=sampler) if sampler is not None else TracerProvider()
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
    return [span for span in exporter.get_finished_spans() if span.name.startswith('chat ')], exporter


def cache_attributes(span: ReadableSpan) -> dict[str, object]:
    assert span.attributes is not None
    return {key: value for key, value in span.attributes.items() if key.startswith('pydantic_ai.cache.')}


def test_stable_cache_health() -> None:
    """Growing cache reads across a run produce hit-ratio/established attributes and no collapse."""
    spans, _ = cache_spans(
        [CacheUsage(write=1500), CacheUsage(read=1500), CacheUsage(read=1600)], retention=timedelta(hours=1)
    )

    assert [cache_attributes(span) for span in spans] == [
        # The establishing request reads nothing back, so its cold-start hit ratio is honestly 0.0.
        {'pydantic_ai.cache.hit_ratio': 0.0, 'pydantic_ai.cache.established_tokens': 1500},
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
    """A collapse is classified by retention (incl. `CachePoint` extension); only `unexpected` emits the event."""
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
    """Event attributes must skip `None` values (OTel attributes cannot be None)."""
    spans, _ = cache_spans(
        [CacheUsage(write=1400, provider_name=None), CacheUsage(read=100, provider_name=None)],
        retention=timedelta(hours=1),
    )

    (event,) = spans[-1].events
    assert event.name == 'pydantic_ai.cache.collapse'
    assert 'provider_name' not in (event.attributes or {})


def test_model_switch_and_switch_back() -> None:
    """A model switch is never a collapse (fresh per-model mark); switching back is judged against the old mark."""
    spans, _ = cache_spans(
        [
            CacheUsage(write=1400, model_name='first'),
            CacheUsage(write=1500, model_name='second'),
            CacheUsage(read=100, model_name='first'),
        ],
        retention=timedelta(hours=1),
    )

    # The switched-to model writes its own prefix: judged against a fresh mark, never `first`'s.
    assert cache_attributes(spans[1]) == {
        'pydantic_ai.cache.hit_ratio': 0.0,
        'pydantic_ai.cache.established_tokens': 1500,
    }
    assert cache_attributes(spans[2])['pydantic_ai.cache.collapse_reason'] == 'unexpected'


def test_sub_threshold_collapse_and_rebaseline() -> None:
    """Prefixes below the minimum are never judged, and a collapse re-baselines the mark so it warns once."""
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
    """Non-caching runs get zero cache attributes; runs with cache reads get a run-span aggregate ratio."""
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
    """The mark must be established during a sampled-out (non-recording) span, so a collapse is
    still detected on the next, recorded span."""
    spans, _ = cache_spans(
        [CacheUsage(write=1400), CacheUsage(read=100)],
        retention=timedelta(hours=1),
        sampler=DropFirstChatSpanSampler(),
    )

    (span,) = spans
    attributes = cache_attributes(span)
    assert attributes['pydantic_ai.cache.collapsed'] is True
    assert attributes['pydantic_ai.cache.collapse_reason'] == 'unexpected'
    assert [event.name for event in span.events] == ['pydantic_ai.cache.collapse']


def test_unreported_cache_usage_reports_waste_without_alerting() -> None:
    """A `0/0` response after an established prefix re-sends it uncached, so the waste is reported —
    but the cause is ambiguous (caching disabled on a write-reporting provider vs. a full miss on a
    read-only-reporting one), so it is classified `unreported` and never alerts."""
    spans, _ = cache_spans(
        [CacheUsage(write=1400), CacheUsage(), CacheUsage(read=1400)],
        retention=timedelta(hours=1),
    )

    assert cache_attributes(spans[1]) == {
        'pydantic_ai.cache.hit_ratio': 0.0,
        'pydantic_ai.cache.established_tokens': 1400,
        'pydantic_ai.cache.collapsed': True,
        'pydantic_ai.cache.wasted_tokens': 1400,
        'pydantic_ai.cache.collapse_reason': 'unreported',
    }
    assert not spans[1].events
    # The mark survives untouched: the provider may still hold the prefix, so a later hit is not a
    # fresh establish and is judged against the original mark.
    assert cache_attributes(spans[2]) == {
        'pydantic_ai.cache.hit_ratio': 0.7,
        'pydantic_ai.cache.established_tokens': 1400,
    }
    assert not spans[2].events


def test_unreported_request_does_not_refresh_idle_clock(mocker: MockerFixture) -> None:
    """The `0/0` request must not update `last_seen`: with the clock pinned, the later collapse is
    classified against the *first* request's timestamp (`ttl-expired`), which a clock-refreshing
    implementation would misreport as `unexpected`."""
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    mocker.patch(
        'pydantic_ai._utils.now_utc',
        side_effect=[t0, t0 + timedelta(minutes=100), t0 + timedelta(minutes=101)],
    )
    spans, _ = cache_spans(
        [CacheUsage(write=1400), CacheUsage(), CacheUsage(read=100)],
        retention=timedelta(minutes=15),
    )

    assert cache_attributes(spans[2])['pydantic_ai.cache.collapse_reason'] == 'ttl-expired'
    assert not spans[2].events


def test_unreported_usage_without_established_prefix_is_ignored() -> None:
    """Before anything is cached there is no waste to report, so `0/0` responses stay silent."""
    spans, _ = cache_spans([CacheUsage(), CacheUsage(read=1400)], retention=timedelta(hours=1))

    assert cache_attributes(spans[0]) == {}
    assert cache_attributes(spans[1]) == {
        'pydantic_ai.cache.hit_ratio': 0.7,
        'pydantic_ai.cache.established_tokens': 1400,
    }
