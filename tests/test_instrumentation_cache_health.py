from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pytest_mock import MockerFixture

from pydantic_ai import (
    Agent,
    CachePoint,
    CompactionPart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelResponse,
    ModelResponsePart,
    ModelRetry,
    NativeToolCallPart,
    NativeToolReturnPart,
    RunContext,
    TextPart,
    ToolCallPart,
)
from pydantic_ai._cache_health import CacheHealthDetector, CacheMark, ConversationCacheMarkStore
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings
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

        # Required by the ABC, never called here.
        def get_description(self) -> str:  # pragma: no cover
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
    input_tokens: int = 20000
    provider_name: str | None = 'test'
    model_name: str = 'cache-model'
    provider_url: str | None = None
    compacts: bool = False
    """Whether the response carries a `CompactionPart`, as when the provider compacted the history."""
    suspends: bool = False
    """Whether the response pauses the turn (Anthropic `pause_turn`), so the next usage continues the same request."""
    native_tool: bool = False
    """Whether the response ran a native tool (web search), whose cache usage may be summed over several passes."""
    details: Mapping[str, int] = field(default_factory=dict[str, int])


class ResponseNameFunctionModel(FunctionModel):
    def set_response_model_name(self, model_name: str) -> None:
        self._model_name = model_name


def cache_spans(
    usages: Sequence[CacheUsage],
    *,
    retention: timedelta | None = None,
    prompt: str | list[str | CachePoint] = 'prompt',
    sampler: Sampler | None = None,
    use_fallback: bool = False,
    capabilities: Sequence[AbstractCapability[Any]] = (),
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
        if usage.suspends:
            parts: list[ModelResponsePart] = [TextPart('working')]
        elif call_index == len(usages):
            parts = [TextPart('done')]
        else:
            parts = [ToolCallPart('continue_run', {}, tool_call_id=f'call-{call_index}')]
        if usage.native_tool:
            parts = [
                NativeToolCallPart(
                    'web_search', {'query': 'x'}, tool_call_id=f'srv-{call_index}', provider_name='test'
                ),
                NativeToolReturnPart('web_search', 'results', tool_call_id=f'srv-{call_index}', provider_name='test'),
                *parts,
            ]
        return ModelResponse(
            parts=parts,
            usage=RequestUsage(
                input_tokens=usage.input_tokens,
                cache_read_tokens=usage.read,
                cache_write_tokens=usage.write,
                details=dict(usage.details),
            ),
            provider_name=usage.provider_name,
            state='suspended' if usage.suspends else 'complete',
        )

    profile = ModelProfile(default_cache_retention=retention) if retention is not None else None
    model = ResponseNameFunctionModel(model_function, model_name='cache-model', profile=profile)
    agent = Agent(
        FallbackModel(model) if use_fallback else model,
        capabilities=[
            Instrumentation(settings=InstrumentationSettings(tracer_provider=tracer_provider, include_content=False)),
            *capabilities,
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
        [CacheUsage(write=15000), CacheUsage(read=15000), CacheUsage(read=16000)], retention=timedelta(hours=1)
    )

    assert [cache_attributes(span) for span in spans] == [
        # The establishing request reads nothing back, so its cold-start hit ratio is honestly 0.0.
        {'pydantic_ai.cache.hit_ratio': 0.0, 'pydantic_ai.cache.established_tokens': 15000},
        {'pydantic_ai.cache.hit_ratio': 0.75, 'pydantic_ai.cache.established_tokens': 15000},
        {'pydantic_ai.cache.hit_ratio': 0.8, 'pydantic_ai.cache.established_tokens': 16000},
    ]
    assert all(not span.events for span in spans)


def test_continued_request_is_judged_by_its_final_segment() -> None:
    """A `pause_turn` continuation merges into one response whose usage sums both segments' requests.

    Each segment re-reads the ~14k prefix, so the merged response reports ~29k cached tokens. The mark
    comes from the final segment instead, whose prompt carries the whole prefix, so the next request
    reading back that prefix is healthy rather than a ~14k-token collapse.
    """
    spans, _ = cache_spans(
        [
            CacheUsage(write=14000),
            CacheUsage(read=14000, write=500, suspends=True),
            CacheUsage(read=14500, write=200),
            CacheUsage(read=14700),
        ],
        retention=timedelta(hours=1),
    )

    assert [cache_attributes(span) for span in spans] == [
        {'pydantic_ai.cache.hit_ratio': 0.0, 'pydantic_ai.cache.established_tokens': 14000},
        {'pydantic_ai.cache.hit_ratio': 0.725, 'pydantic_ai.cache.established_tokens': 14700},
        {'pydantic_ai.cache.hit_ratio': 0.735, 'pydantic_ai.cache.established_tokens': 14700},
    ]
    assert all(not span.events for span in spans)


def test_native_tool_response_does_not_raise_the_mark() -> None:
    """A native tool's cache reads may be summed over its passes, so they're judged but don't set the mark.

    Four passes over the ~8k prefix report ~33k cached tokens; raising the mark to that would make the
    next, healthy request look like a collapse.
    """
    spans, _ = cache_spans(
        [
            CacheUsage(write=8000),
            CacheUsage(read=32800, write=600, input_tokens=40000, native_tool=True),
            CacheUsage(read=8400),
        ],
        retention=timedelta(hours=1),
    )

    assert [cache_attributes(span) for span in spans] == [
        {'pydantic_ai.cache.hit_ratio': 0.0, 'pydantic_ai.cache.established_tokens': 8000},
        {'pydantic_ai.cache.hit_ratio': 0.82, 'pydantic_ai.cache.established_tokens': 8000},
        {'pydantic_ai.cache.hit_ratio': 0.42, 'pydantic_ai.cache.established_tokens': 8400},
    ]
    assert all(not span.events for span in spans)


def test_native_tool_response_with_compaction_pass_does_not_raise_the_mark() -> None:
    """A reported single main-model pass plus a compaction pass is still summed usage."""
    spans, _ = cache_spans(
        [
            CacheUsage(write=8000),
            CacheUsage(read=20000, native_tool=True, details={'message_iterations': 1, 'compaction_iterations': 1}),
            CacheUsage(read=8000),
        ],
        retention=timedelta(hours=1),
    )

    assert [cache_attributes(span)['pydantic_ai.cache.established_tokens'] for span in spans] == [8000, 8000, 8000]
    assert all('pydantic_ai.cache.collapsed' not in cache_attributes(span) for span in spans)


def test_native_tool_response_with_separate_tool_use_prompt_count_sets_the_mark() -> None:
    """Gemini counts tool-use prompt tokens apart from cache reads, so its cache reads are an ordinary prefix."""
    spans, _ = cache_spans(
        [
            CacheUsage(write=8000),
            CacheUsage(read=12000, native_tool=True, details={'tool_use_prompt_tokens': 24000}),
            CacheUsage(read=5000),
        ],
        retention=timedelta(hours=1),
    )

    assert cache_attributes(spans[1])['pydantic_ai.cache.established_tokens'] == 12000
    assert cache_attributes(spans[2]) == {
        'pydantic_ai.cache.hit_ratio': 0.25,
        'pydantic_ai.cache.established_tokens': 5000,
        'pydantic_ai.cache.collapsed': True,
        'pydantic_ai.cache.missed_tokens': 7000,
        'pydantic_ai.cache.collapse_reason': 'unexpected',
    }


def test_response_rejected_by_a_later_hook_still_counts() -> None:
    """A response an `after_model_request` hook rejects with `ModelRetry` was still served, so it sets the mark."""

    @dataclass
    class RejectFirstResponse(AbstractCapability[None]):
        rejected: bool = False

        async def after_model_request(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            if not self.rejected:
                self.rejected = True
                raise ModelRetry('Try again.')
            return response

    spans, _ = cache_spans(
        [CacheUsage(write=15000), CacheUsage(read=15000)],
        retention=timedelta(hours=1),
        capabilities=[RejectFirstResponse()],
    )

    assert [cache_attributes(span) for span in spans] == [
        {'pydantic_ai.cache.hit_ratio': 0.0, 'pydantic_ai.cache.established_tokens': 15000},
        {'pydantic_ai.cache.hit_ratio': 0.75, 'pydantic_ai.cache.established_tokens': 15000},
    ]


def test_nested_agent_request_is_not_taken_for_a_continuation_segment() -> None:
    """A request made by a nested agent inside an instrumented request is its own, not one of that request's segments."""

    def nested_model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart('nested')],
            usage=RequestUsage(input_tokens=20000, cache_read_tokens=100),
            provider_name='test',
        )

    nested_agent = Agent(FunctionModel(nested_model_function, model_name='cache-model'))

    @dataclass
    class RunNestedAgent(AbstractCapability[None]):
        async def after_model_request(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            await nested_agent.run('nested')
            return response

    spans, _ = cache_spans(
        [CacheUsage(write=15000), CacheUsage(read=15000)],
        retention=timedelta(hours=1),
        capabilities=[RunNestedAgent()],
    )

    assert [cache_attributes(span) for span in spans] == [
        {'pydantic_ai.cache.hit_ratio': 0.0, 'pydantic_ai.cache.established_tokens': 15000},
        {'pydantic_ai.cache.hit_ratio': 0.75, 'pydantic_ai.cache.established_tokens': 15000},
    ]


@pytest.mark.parametrize(
    ('retention', 'prompt', 'reason', 'has_event'),
    [
        (timedelta(hours=1), 'prompt', 'unexpected', True),
        (timedelta(0), 'prompt', 'ttl_expired', False),
        (None, 'prompt', 'unknown', False),
        (timedelta(0), ['context', CachePoint(ttl='1h')], 'unexpected', True),
    ],
)
def test_cache_collapse_classification(
    retention: timedelta | None, prompt: str | list[str | CachePoint], reason: str, has_event: bool
) -> None:
    """A collapse is classified by retention (incl. `CachePoint` extension); only `unexpected` emits the event."""
    spans, _ = cache_spans([CacheUsage(write=14000), CacheUsage(read=1000)], retention=retention, prompt=prompt)

    assert cache_attributes(spans[-1]) == {
        'pydantic_ai.cache.hit_ratio': 0.05,
        'pydantic_ai.cache.established_tokens': 1000,
        'pydantic_ai.cache.collapsed': True,
        'pydantic_ai.cache.missed_tokens': 13000,
        'pydantic_ai.cache.collapse_reason': reason,
    }
    assert [event.name for event in spans[-1].events] == (['pydantic_ai.cache.collapse'] if has_event else [])
    if has_event:
        assert dict(spans[-1].events[0].attributes or {}) == {
            'established_tokens': 14000,
            'cache_read_tokens': 1000,
            'missed_tokens': 13000,
            'provider_name': 'test',
            'model_name': 'cache-model',
        }


def test_collapse_event_without_provider_name() -> None:
    """Event attributes must skip `None` values (OTel attributes cannot be None)."""
    spans, _ = cache_spans(
        [CacheUsage(write=14000, provider_name=None), CacheUsage(read=1000, provider_name=None)],
        retention=timedelta(hours=1),
    )

    (event,) = spans[-1].events
    assert event.name == 'pydantic_ai.cache.collapse'
    assert 'provider_name' not in (event.attributes or {})


def test_model_switch_and_switch_back() -> None:
    """A model switch is never a collapse (fresh per-model mark); switching back is judged against the old mark."""
    spans, _ = cache_spans(
        [
            CacheUsage(write=14000, model_name='first'),
            CacheUsage(write=15000, model_name='second'),
            CacheUsage(read=1000, model_name='first'),
        ],
        retention=timedelta(hours=1),
    )

    # The switched-to model writes its own prefix: judged against a fresh mark, never `first`'s.
    assert cache_attributes(spans[1]) == {
        'pydantic_ai.cache.hit_ratio': 0.0,
        'pydantic_ai.cache.established_tokens': 15000,
    }
    assert cache_attributes(spans[2])['pydantic_ai.cache.collapse_reason'] == 'unexpected'


def test_miss_thresholds_and_rebaseline() -> None:
    """A shortfall must be more than 5% of the established prefix and at least 2,000 tokens to collapse,
    and a collapse re-baselines the mark so it is reported once."""
    spans, _ = cache_spans(
        [
            CacheUsage(write=1900),
            # 1,800 tokens short: most of the prefix, but under the 2,000-token floor.
            CacheUsage(read=100),
            CacheUsage(write=100000),
            # 4,000 tokens short: over the floor, but only 4% of the prefix.
            CacheUsage(read=96000),
            # 95,000 tokens short.
            CacheUsage(read=5000),
            CacheUsage(read=5000),
        ],
        retention=timedelta(hours=1),
    )

    assert [cache_attributes(span).get('pydantic_ai.cache.collapsed') for span in spans] == [
        None,
        None,
        None,
        None,
        True,
        None,
    ]
    assert cache_attributes(spans[4])['pydantic_ai.cache.missed_tokens'] == 95000
    assert len([event for span in spans for event in span.events if event.name == 'pydantic_ai.cache.collapse']) == 1


def test_partial_prefix_move_collapses() -> None:
    """A change deep in the history moves only the tail of the prefix: reading back 90% of it is still a
    collapse, which a "fell below half" rule would miss."""
    spans, _ = cache_spans([CacheUsage(write=100000), CacheUsage(read=90000)], retention=timedelta(hours=1))

    assert cache_attributes(spans[-1])['pydantic_ai.cache.collapse_reason'] == 'unexpected'
    assert cache_attributes(spans[-1])['pydantic_ai.cache.missed_tokens'] == 10000


def test_sustained_collapse_emits_the_event_once() -> None:
    """A prefix that moves on every request collapses on every request, and each span records the
    waste, but the event fires once per collapse: a healthy read-back re-arms it."""
    spans, _ = cache_spans(
        [
            CacheUsage(write=14000),
            CacheUsage(write=14000),
            CacheUsage(write=14000),
            CacheUsage(read=14000),
            CacheUsage(write=14000),
        ],
        retention=timedelta(hours=1),
    )

    assert [cache_attributes(span).get('pydantic_ai.cache.collapse_reason') for span in spans] == [
        None,
        'unexpected',
        'unexpected',
        None,
        'unexpected',
    ]
    assert [[event.name for event in span.events] for span in spans] == [
        [],
        ['pydantic_ai.cache.collapse'],
        [],
        [],
        ['pydantic_ai.cache.collapse'],
    ]


def test_collapse_without_an_event_does_not_latch(mocker: MockerFixture) -> None:
    """Only a collapse that emitted the event holds back the next one: an unexpected collapse right after
    a `ttl_expired` one (the prefix kept moving once the cache was re-written) still emits it."""
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    mocker.patch(
        'pydantic_ai._utils.now_utc',
        side_effect=[t0, t0 + timedelta(hours=2), t0 + timedelta(hours=2, minutes=1)],
    )
    spans, _ = cache_spans(
        [CacheUsage(write=14000), CacheUsage(write=14000), CacheUsage(write=14000)],
        retention=timedelta(hours=1),
    )

    assert [cache_attributes(span).get('pydantic_ai.cache.collapse_reason') for span in spans] == [
        None,
        'ttl_expired',
        'unexpected',
    ]
    assert [[event.name for event in span.events] for span in spans] == [[], [], ['pydantic_ai.cache.collapse']]


def test_no_cache_attributes_without_caching_nor_on_the_run_span() -> None:
    """Non-caching runs get no cache attributes, and the agent-run span never carries a hit ratio: a ratio
    aggregated across models isn't interpretable (OTel GenAI semconv dropped cache attributes from
    `invoke_agent` spans), and backends can compute one from the `gen_ai.aggregated_usage.*` counts."""
    no_cache_spans, _ = cache_spans([CacheUsage()])
    assert cache_attributes(no_cache_spans[0]) == {}

    spans, exporter = cache_spans([CacheUsage(read=5000)])
    assert cache_attributes(spans[0])['pydantic_ai.cache.hit_ratio'] == 0.25
    run_span = next(span for span in exporter.get_finished_spans() if span.name.startswith('invoke_agent '))
    assert not [key for key in (run_span.attributes or {}) if key.startswith('pydantic_ai.cache.')]
    assert (run_span.attributes or {})['gen_ai.aggregated_usage.cache_read.input_tokens'] == 5000


def test_cache_marks_update_without_recording() -> None:
    """The mark must be established during a sampled-out (non-recording) span, so a collapse is
    still detected on the next, recorded span."""
    spans, _ = cache_spans(
        [CacheUsage(write=14000), CacheUsage(read=1000)],
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
        [CacheUsage(write=14000), CacheUsage(), CacheUsage(read=14000)],
        retention=timedelta(hours=1),
    )

    assert cache_attributes(spans[1]) == {
        'pydantic_ai.cache.hit_ratio': 0.0,
        'pydantic_ai.cache.established_tokens': 14000,
        'pydantic_ai.cache.collapsed': True,
        'pydantic_ai.cache.missed_tokens': 14000,
        'pydantic_ai.cache.collapse_reason': 'unreported',
    }
    assert not spans[1].events
    # The mark survives untouched: the provider may still hold the prefix, so a later hit is not a
    # fresh establish and is judged against the original mark.
    assert cache_attributes(spans[2]) == {
        'pydantic_ai.cache.hit_ratio': 0.7,
        'pydantic_ai.cache.established_tokens': 14000,
    }
    assert not spans[2].events


def test_unreported_request_does_not_refresh_idle_clock(mocker: MockerFixture) -> None:
    """The `0/0` request must not update `last_seen`: with the clock pinned, the later collapse is
    classified against the *first* request's timestamp (`ttl_expired`), which a clock-refreshing
    implementation would misreport as `unexpected`."""
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    mocker.patch(
        'pydantic_ai._utils.now_utc',
        side_effect=[t0, t0 + timedelta(minutes=100), t0 + timedelta(minutes=101)],
    )
    spans, _ = cache_spans(
        [CacheUsage(write=14000), CacheUsage(), CacheUsage(read=1000)],
        retention=timedelta(minutes=15),
    )

    assert cache_attributes(spans[2])['pydantic_ai.cache.collapse_reason'] == 'ttl_expired'
    assert not spans[2].events


def test_unreported_usage_without_established_prefix_is_ignored() -> None:
    """Before anything is cached there is no waste to report, so `0/0` responses stay silent."""
    spans, _ = cache_spans([CacheUsage(), CacheUsage(read=14000)], retention=timedelta(hours=1))

    assert cache_attributes(spans[0]) == {}
    assert cache_attributes(spans[1]) == {
        'pydantic_ai.cache.hit_ratio': 0.7,
        'pydantic_ai.cache.established_tokens': 14000,
    }


def test_fallback_model_collapse_is_classified_not_raised() -> None:
    """`FallbackModel` has no profile of its own, and the model that actually served the request isn't
    reachable from here, so a collapse under it is classified `unknown` rather than raising
    `NotImplementedError` and failing an otherwise successful run."""
    spans, _ = cache_spans(
        [CacheUsage(write=14000), CacheUsage(read=1000)],
        retention=timedelta(hours=1),
        use_fallback=True,
    )

    assert cache_attributes(spans[-1]) == {
        'pydantic_ai.cache.hit_ratio': 0.05,
        'pydantic_ai.cache.established_tokens': 1000,
        'pydantic_ai.cache.collapsed': True,
        'pydantic_ai.cache.missed_tokens': 13000,
        'pydantic_ai.cache.collapse_reason': 'unknown',
    }
    assert not spans[-1].events


# ---- Cache marks across the runs of a conversation ---------------------------------------


class ConversationModel(FunctionModel):
    """Serves a queue of cache usages, one per request, across any number of runs."""

    def __init__(self, *, retention: timedelta | None = timedelta(hours=1), requested: timedelta | None = None):
        super().__init__(
            self._respond, model_name='cache-model', profile=ModelProfile(default_cache_retention=retention)
        )
        self.usages: list[CacheUsage] = []
        self.requested = requested
        self.resolved_settings: list[ModelSettings | None] = []

    def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        usage = self.usages.pop(0)
        return ModelResponse(
            parts=[CompactionPart(content='summary'), TextPart('done')] if usage.compacts else [TextPart('done')],
            usage=RequestUsage(
                input_tokens=usage.input_tokens, cache_read_tokens=usage.read, cache_write_tokens=usage.write
            ),
            provider_name=usage.provider_name,
            provider_url=usage.provider_url,
        )

    def resolve_cache_retention(self, model_settings: ModelSettings | None) -> timedelta | None:
        self.resolved_settings.append(model_settings)
        return self.requested


def conversation_agent(model: ConversationModel) -> tuple[Agent, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    agent = Agent(
        model,
        capabilities=[
            Instrumentation(settings=InstrumentationSettings(tracer_provider=tracer_provider, include_content=False))
        ],
    )
    return agent, exporter


def chat_cache_attributes(exporter: InMemorySpanExporter) -> list[dict[str, object]]:
    return [cache_attributes(span) for span in exporter.get_finished_spans() if span.name.startswith('chat ')]


COLLAPSED_ON_CONTINUATION = {
    'pydantic_ai.cache.hit_ratio': 0.05,
    'pydantic_ai.cache.established_tokens': 1000,
    'pydantic_ai.cache.collapsed': True,
    'pydantic_ai.cache.missed_tokens': 13000,
    'pydantic_ai.cache.collapse_reason': 'unexpected',
}


def test_collapse_on_first_request_of_continued_conversation() -> None:
    """The next turn re-sends what the previous one cached, so its first request is judged against the
    previous run's mark: a run keeping marks to itself would see a fresh establish here (issue #7900)."""
    model = ConversationModel()
    agent, exporter = conversation_agent(model)

    model.usages = [CacheUsage(write=14000)]
    first = agent.run_sync('first turn')
    model.usages = [CacheUsage(read=1000)]
    second = agent.run_sync('second turn', message_history=first.all_messages())

    assert second.conversation_id == first.conversation_id
    assert chat_cache_attributes(exporter)[-1] == COLLAPSED_ON_CONTINUATION
    chat_span = [span for span in exporter.get_finished_spans() if span.name.startswith('chat ')][-1]
    assert [event.name for event in chat_span.events] == ['pydantic_ai.cache.collapse']


def test_collapse_on_continuation_from_serialized_history() -> None:
    """History serialized and loaded back carries its conversation id, so it continues the same marks."""
    model = ConversationModel()
    agent, exporter = conversation_agent(model)

    model.usages = [CacheUsage(write=14000)]
    first = agent.run_sync('first turn')
    history = ModelMessagesTypeAdapter.validate_json(first.all_messages_json())
    model.usages = [CacheUsage(read=1000)]
    agent.run_sync('second turn', message_history=history)

    assert chat_cache_attributes(exporter)[-1] == COLLAPSED_ON_CONTINUATION


def test_new_conversation_starts_from_a_clean_mark() -> None:
    model = ConversationModel()
    agent, exporter = conversation_agent(model)

    model.usages = [CacheUsage(write=14000)]
    agent.run_sync('first conversation')
    model.usages = [CacheUsage(read=1000)]
    agent.run_sync('second conversation')

    assert chat_cache_attributes(exporter)[-1] == {
        'pydantic_ai.cache.hit_ratio': 0.05,
        'pydantic_ai.cache.established_tokens': 1000,
    }


def test_marks_are_shared_by_injected_instrumentation() -> None:
    """`Agent(instrument=...)` builds its `Instrumentation` afresh for every run, so the marks can't live on it."""
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    model = ConversationModel()
    agent = Agent(model)
    agent.instrument = InstrumentationSettings(tracer_provider=tracer_provider, include_content=False)

    model.usages = [CacheUsage(write=14000)]
    first = agent.run_sync('first turn')
    model.usages = [CacheUsage(read=1000)]
    agent.run_sync('second turn', message_history=first.all_messages())

    assert chat_cache_attributes(exporter)[-1] == COLLAPSED_ON_CONTINUATION


def test_endpoint_switch_is_not_a_collapse() -> None:
    """Two endpoints serving the same provider and model name keep separate caches, so switching is a fresh mark."""
    model = ConversationModel()
    agent, exporter = conversation_agent(model)

    model.usages = [CacheUsage(write=14000, provider_url='https://eu.example.com')]
    first = agent.run_sync('first turn')
    model.usages = [CacheUsage(write=14000, provider_url='https://us.example.com')]
    agent.run_sync('second turn', message_history=first.all_messages())

    assert chat_cache_attributes(exporter)[-1] == {
        'pydantic_ai.cache.hit_ratio': 0.0,
        'pydantic_ai.cache.established_tokens': 14000,
    }


def test_compaction_in_the_response_is_not_unexpected() -> None:
    """Provider-native compaction (Anthropic's, OpenAI's server-side) returns a `CompactionPart` from the
    request that compacted, and the provider drops the history before it: the prefix shrinks by design, so
    the collapse is recorded as `compacted` and emits no event. The next request is judged against the
    compacted prefix."""
    model = ConversationModel()
    agent, exporter = conversation_agent(model)

    model.usages = [CacheUsage(write=14000)]
    first = agent.run_sync('first turn')
    model.usages = [CacheUsage(write=3000, compacts=True)]
    second = agent.run_sync('second turn', message_history=first.all_messages())
    model.usages = [CacheUsage(read=3000)]
    agent.run_sync('third turn', message_history=second.all_messages())

    assert chat_cache_attributes(exporter)[1:] == [
        {
            'pydantic_ai.cache.hit_ratio': 0.0,
            'pydantic_ai.cache.established_tokens': 3000,
            'pydantic_ai.cache.collapsed': True,
            'pydantic_ai.cache.missed_tokens': 14000,
            'pydantic_ai.cache.collapse_reason': 'compacted',
        },
        {'pydantic_ai.cache.hit_ratio': 0.15, 'pydantic_ai.cache.established_tokens': 3000},
    ]
    assert not [event for span in exporter.get_finished_spans() for event in span.events]


def test_compaction_in_the_history_is_not_unexpected() -> None:
    """OpenAI's stateless compaction adds the `CompactionPart` to the history before the request it shrinks."""
    model = ConversationModel()
    agent, exporter = conversation_agent(model)

    model.usages = [CacheUsage(write=14000)]
    first = agent.run_sync('first turn')
    compacted = ModelResponse(parts=[CompactionPart(content='summary')], conversation_id=first.conversation_id)
    model.usages = [CacheUsage(write=3000)]
    agent.run_sync('second turn', message_history=[*first.all_messages(), compacted])

    assert chat_cache_attributes(exporter)[-1]['pydantic_ai.cache.collapse_reason'] == 'compacted'


def test_continuation_after_cache_expiry_is_ttl_expired(mocker: MockerFixture) -> None:
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    mocker.patch('pydantic_ai._utils.now_utc', side_effect=[t0, t0 + timedelta(minutes=30)])
    model = ConversationModel(retention=timedelta(minutes=5))
    agent, exporter = conversation_agent(model)

    model.usages = [CacheUsage(write=14000)]
    first = agent.run_sync('first turn')
    model.usages = [CacheUsage(read=1000)]
    agent.run_sync('second turn', message_history=first.all_messages())

    assert chat_cache_attributes(exporter)[-1]['pydantic_ai.cache.collapse_reason'] == 'ttl_expired'


@pytest.mark.parametrize(
    ('retention', 'requested', 'reason'),
    [
        # Settings that request nothing leave the provider's default in place.
        (timedelta(minutes=5), None, 'ttl_expired'),
        # Retention requested by the settings replaces the default, both ways.
        (timedelta(minutes=5), timedelta(hours=1), 'unexpected'),
        (timedelta(hours=1), timedelta(minutes=5), 'ttl_expired'),
        (None, timedelta(hours=1), 'unexpected'),
        (None, None, 'unknown'),
    ],
)
def test_collapse_classified_with_resolved_retention(
    mocker: MockerFixture, retention: timedelta | None, requested: timedelta | None, reason: str
) -> None:
    """Classification uses `Model.resolve_cache_retention()` for the request's settings, falling back to
    the profile's `default_cache_retention`."""
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    mocker.patch('pydantic_ai._utils.now_utc', side_effect=[t0, t0 + timedelta(minutes=30)])
    model = ConversationModel(retention=retention, requested=requested)
    agent, exporter = conversation_agent(model)

    model.usages = [CacheUsage(write=14000)]
    first = agent.run_sync('first turn')
    model.usages = [CacheUsage(read=1000)]
    agent.run_sync('second turn', message_history=first.all_messages(), model_settings={'temperature': 0.5})

    assert chat_cache_attributes(exporter)[-1]['pydantic_ai.cache.collapse_reason'] == reason
    # The resolver sees the settings the collapsing request was made with.
    assert model.resolved_settings[-1] == {'temperature': 0.5}


def test_idle_conversations_are_forgotten(mocker: MockerFixture) -> None:
    """A conversation idle past the longest documented cache retention is dropped, bounding memory."""
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    mocker.patch(
        'pydantic_ai._utils.now_utc',
        side_effect=[t0, t0 + timedelta(hours=25), t0 + timedelta(hours=25)],
    )
    model = ConversationModel()
    agent, exporter = conversation_agent(model)

    model.usages = [CacheUsage(write=14000)]
    first = agent.run_sync('first turn')
    # Any later update sweeps the conversations that went idle before it.
    model.usages = [CacheUsage(write=14000)]
    agent.run_sync('another conversation')
    model.usages = [CacheUsage(read=1000)]
    agent.run_sync('second turn', message_history=first.all_messages())

    assert 'pydantic_ai.cache.collapsed' not in chat_cache_attributes(exporter)[-1]


def test_least_recently_updated_conversations_are_forgotten(mocker: MockerFixture) -> None:
    mocker.patch.object(ConversationCacheMarkStore, 'max_conversations', 1)
    model = ConversationModel()
    agent, exporter = conversation_agent(model)

    model.usages = [CacheUsage(write=14000)]
    first = agent.run_sync('first turn')
    model.usages = [CacheUsage(write=14000)]
    agent.run_sync('another conversation')
    model.usages = [CacheUsage(read=1000)]
    agent.run_sync('second turn', message_history=first.all_messages())

    assert 'pydantic_ai.cache.collapsed' not in chat_cache_attributes(exporter)[-1]


def test_recently_updated_conversation_outlives_older_ones(mocker: MockerFixture) -> None:
    """Forgetting goes by the last update, not by when a conversation was first seen."""
    mocker.patch.object(ConversationCacheMarkStore, 'max_conversations', 2)
    model = ConversationModel()
    agent, exporter = conversation_agent(model)

    model.usages = [CacheUsage(write=14000)]
    first = agent.run_sync('first turn')
    model.usages = [CacheUsage(write=14000)]
    agent.run_sync('another conversation')
    model.usages = [CacheUsage(read=14000)]
    second = agent.run_sync('second turn', message_history=first.all_messages())
    model.usages = [CacheUsage(write=14000)]
    agent.run_sync('yet another conversation')
    model.usages = [CacheUsage(read=1000)]
    agent.run_sync('third turn', message_history=second.all_messages())

    assert chat_cache_attributes(exporter)[-1] == COLLAPSED_ON_CONTINUATION


def test_marks_forgotten_mid_run_are_kept_by_the_run(mocker: MockerFixture) -> None:
    """A run keeps judging against its conversation's marks even if the store forgot them meanwhile,
    and puts them back, so the next run of the conversation still sees them."""
    mocker.patch.object(ConversationCacheMarkStore, 'max_conversations', 1)
    model = ConversationModel()
    agent, exporter = conversation_agent(model)
    other_model = ConversationModel()
    other = Agent(other_model, capabilities=[Instrumentation(settings=InstrumentationSettings())])

    @agent.tool_plain
    async def run_another_conversation() -> str:
        other_model.usages = [CacheUsage(write=14000)]
        await other.run('another conversation')
        return 'done'

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        response = ConversationModel._respond(model, messages, info)  # pyright: ignore[reportPrivateUsage]
        if len(messages) == 1:
            response.parts = [ToolCallPart('run_another_conversation', {}, tool_call_id='call-1')]
        return response

    model.function = respond
    # The collapse re-baselines the mark to what the collapsing request established.
    model.usages = [CacheUsage(write=14000), CacheUsage(read=1000, write=13000)]
    first = agent.run_sync('first turn')
    assert chat_cache_attributes(exporter)[-1] == {
        'pydantic_ai.cache.hit_ratio': 0.05,
        'pydantic_ai.cache.established_tokens': 14000,
        'pydantic_ai.cache.collapsed': True,
        'pydantic_ai.cache.missed_tokens': 13000,
        'pydantic_ai.cache.collapse_reason': 'unexpected',
    }

    model.usages = [CacheUsage(read=1000)]
    agent.run_sync('second turn', message_history=first.all_messages())
    assert chat_cache_attributes(exporter)[-1] == COLLAPSED_ON_CONTINUATION


def test_marks_without_a_conversation_are_not_stored() -> None:
    """A run without a conversation id has nothing to share its marks with, so they stay private to it."""
    store = ConversationCacheMarkStore()
    marks = store.get(None)
    marks[('test', None, 'cache-model')] = CacheMark(established_tokens=14000, last_seen=datetime.now(UTC), run_id=None)
    store.update(None, marks, datetime.now(UTC))

    assert store.get(None) == {}
    assert not store._conversations  # pyright: ignore[reportPrivateUsage]


def test_concurrent_runs_of_a_new_conversation_keep_each_others_marks() -> None:
    """Runs of a new conversation that start concurrently each get their own marks before either stores
    any; the second to update merges into what the first stored, so a later run sees both."""
    store = ConversationCacheMarkStore()
    request_context = ModelRequestContext(
        model=ConversationModel(), messages=[], model_settings=None, model_request_parameters=ModelRequestParameters()
    )
    first = CacheHealthDetector(store, 'conversation', 'run-1', alert_on={'unexpected'})
    second = CacheHealthDetector(store, 'conversation', 'run-2', alert_on={'unexpected'})
    for detector, provider_name in ((first, 'provider-a'), (second, 'provider-b')):
        detector.observe(
            request_context,
            ModelResponse(
                parts=[TextPart('done')],
                usage=RequestUsage(input_tokens=20000, cache_write_tokens=14000),
                model_name='cache-model',
                provider_name=provider_name,
            ),
        )

    later = CacheHealthDetector(store, 'conversation', 'run-3', alert_on={'unexpected'})
    assert set(later.marks) == {('provider-a', None, 'cache-model'), ('provider-b', None, 'cache-model')}
    assert second.marks is later.marks
