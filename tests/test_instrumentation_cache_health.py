from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest
from pytest_mock import MockerFixture

from pydantic_ai import Agent, CachePoint, ModelMessage, ModelMessagesTypeAdapter, ModelResponse, TextPart, ToolCallPart
from pydantic_ai._cache_health import CacheMark, ConversationCacheMarkStore
from pydantic_ai.capabilities.instrumentation import Instrumentation
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
    input_tokens: int = 2000
    provider_name: str | None = 'test'
    model_name: str = 'cache-model'
    provider_url: str | None = None


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

    profile = ModelProfile(default_cache_retention=retention) if retention is not None else None
    model = ResponseNameFunctionModel(model_function, model_name='cache-model', profile=profile)
    agent = Agent(
        FallbackModel(model) if use_fallback else model,
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


def test_sustained_collapse_emits_the_event_once() -> None:
    """A prefix that moves on every request collapses on every request, and each span records the
    waste, but the event fires once per collapse: a healthy read-back re-arms it."""
    spans, _ = cache_spans(
        [
            CacheUsage(write=1400),
            CacheUsage(write=1400),
            CacheUsage(write=1400),
            CacheUsage(read=1400),
            CacheUsage(write=1400),
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
    a `ttl-expired` one (the prefix kept moving once the cache was re-written) still emits it."""
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    mocker.patch(
        'pydantic_ai._utils.now_utc',
        side_effect=[t0, t0 + timedelta(hours=2), t0 + timedelta(hours=2, minutes=1)],
    )
    spans, _ = cache_spans(
        [CacheUsage(write=1400), CacheUsage(write=1400), CacheUsage(write=1400)],
        retention=timedelta(hours=1),
    )

    assert [cache_attributes(span).get('pydantic_ai.cache.collapse_reason') for span in spans] == [
        None,
        'ttl-expired',
        'unexpected',
    ]
    assert [[event.name for event in span.events] for span in spans] == [[], [], ['pydantic_ai.cache.collapse']]


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


def test_fallback_model_collapse_is_classified_not_raised() -> None:
    """`FallbackModel` has no profile of its own, and the model that actually served the request isn't
    reachable from here, so a collapse under it is classified `unknown` rather than raising
    `NotImplementedError` and failing an otherwise successful run."""
    spans, _ = cache_spans(
        [CacheUsage(write=1400), CacheUsage(read=100)],
        retention=timedelta(hours=1),
        use_fallback=True,
    )

    assert cache_attributes(spans[-1]) == {
        'pydantic_ai.cache.hit_ratio': 0.05,
        'pydantic_ai.cache.established_tokens': 100,
        'pydantic_ai.cache.collapsed': True,
        'pydantic_ai.cache.wasted_tokens': 1300,
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
            parts=[TextPart('done')],
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
    'pydantic_ai.cache.established_tokens': 100,
    'pydantic_ai.cache.collapsed': True,
    'pydantic_ai.cache.wasted_tokens': 1300,
    'pydantic_ai.cache.collapse_reason': 'unexpected',
}


def test_collapse_on_first_request_of_continued_conversation() -> None:
    """The next turn re-sends what the previous one cached, so its first request is judged against the
    previous run's mark: a run keeping marks to itself would see a fresh establish here (issue #7900)."""
    model = ConversationModel()
    agent, exporter = conversation_agent(model)

    model.usages = [CacheUsage(write=1400)]
    first = agent.run_sync('first turn')
    model.usages = [CacheUsage(read=100)]
    second = agent.run_sync('second turn', message_history=first.all_messages())

    assert second.conversation_id == first.conversation_id
    assert chat_cache_attributes(exporter)[-1] == COLLAPSED_ON_CONTINUATION
    chat_span = [span for span in exporter.get_finished_spans() if span.name.startswith('chat ')][-1]
    assert [event.name for event in chat_span.events] == ['pydantic_ai.cache.collapse']


def test_collapse_on_continuation_from_serialized_history() -> None:
    """History serialized and loaded back carries its conversation id, so it continues the same marks."""
    model = ConversationModel()
    agent, exporter = conversation_agent(model)

    model.usages = [CacheUsage(write=1400)]
    first = agent.run_sync('first turn')
    history = ModelMessagesTypeAdapter.validate_json(first.all_messages_json())
    model.usages = [CacheUsage(read=100)]
    agent.run_sync('second turn', message_history=history)

    assert chat_cache_attributes(exporter)[-1] == COLLAPSED_ON_CONTINUATION


def test_new_conversation_starts_from_a_clean_mark() -> None:
    model = ConversationModel()
    agent, exporter = conversation_agent(model)

    model.usages = [CacheUsage(write=1400)]
    agent.run_sync('first conversation')
    model.usages = [CacheUsage(read=100)]
    agent.run_sync('second conversation')

    assert chat_cache_attributes(exporter)[-1] == {
        'pydantic_ai.cache.hit_ratio': 0.05,
        'pydantic_ai.cache.established_tokens': 100,
    }


def test_marks_are_shared_by_injected_instrumentation() -> None:
    """`Agent(instrument=...)` builds its `Instrumentation` afresh for every run, so the marks can't live on it."""
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    model = ConversationModel()
    agent = Agent(model)
    agent.instrument = InstrumentationSettings(tracer_provider=tracer_provider, include_content=False)

    model.usages = [CacheUsage(write=1400)]
    first = agent.run_sync('first turn')
    model.usages = [CacheUsage(read=100)]
    agent.run_sync('second turn', message_history=first.all_messages())

    assert chat_cache_attributes(exporter)[-1] == COLLAPSED_ON_CONTINUATION


def test_endpoint_switch_is_not_a_collapse() -> None:
    """Two endpoints serving the same provider and model name keep separate caches, so switching is a fresh mark."""
    model = ConversationModel()
    agent, exporter = conversation_agent(model)

    model.usages = [CacheUsage(write=1400, provider_url='https://eu.example.com')]
    first = agent.run_sync('first turn')
    model.usages = [CacheUsage(write=1400, provider_url='https://us.example.com')]
    agent.run_sync('second turn', message_history=first.all_messages())

    assert chat_cache_attributes(exporter)[-1] == {
        'pydantic_ai.cache.hit_ratio': 0.0,
        'pydantic_ai.cache.established_tokens': 1400,
    }


def test_continuation_after_cache_expiry_is_ttl_expired(mocker: MockerFixture) -> None:
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    mocker.patch('pydantic_ai._utils.now_utc', side_effect=[t0, t0 + timedelta(minutes=30)])
    model = ConversationModel(retention=timedelta(minutes=5))
    agent, exporter = conversation_agent(model)

    model.usages = [CacheUsage(write=1400)]
    first = agent.run_sync('first turn')
    model.usages = [CacheUsage(read=100)]
    agent.run_sync('second turn', message_history=first.all_messages())

    assert chat_cache_attributes(exporter)[-1]['pydantic_ai.cache.collapse_reason'] == 'ttl-expired'


@pytest.mark.parametrize(
    ('retention', 'requested', 'reason'),
    [
        # Settings that request nothing leave the provider's default in place.
        (timedelta(minutes=5), None, 'ttl-expired'),
        # Retention requested by the settings replaces the default, both ways.
        (timedelta(minutes=5), timedelta(hours=1), 'unexpected'),
        (timedelta(hours=1), timedelta(minutes=5), 'ttl-expired'),
        (None, timedelta(hours=1), 'unexpected'),
        (None, None, 'unknown'),
    ],
)
def test_collapse_classified_with_resolved_retention(
    mocker: MockerFixture, retention: timedelta | None, requested: timedelta | None, reason: str
) -> None:
    """Classification uses `Model.resolve_cache_retention()` for the request's settings, falling back to
    the profile's `default_cache_retention`."""
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    mocker.patch('pydantic_ai._utils.now_utc', side_effect=[t0, t0 + timedelta(minutes=30)])
    model = ConversationModel(retention=retention, requested=requested)
    agent, exporter = conversation_agent(model)

    model.usages = [CacheUsage(write=1400)]
    first = agent.run_sync('first turn')
    model.usages = [CacheUsage(read=100)]
    agent.run_sync('second turn', message_history=first.all_messages(), model_settings={'temperature': 0.5})

    assert chat_cache_attributes(exporter)[-1]['pydantic_ai.cache.collapse_reason'] == reason
    # The resolver sees the settings the collapsing request was made with.
    assert model.resolved_settings[-1] == {'temperature': 0.5}


def test_idle_conversations_are_forgotten(mocker: MockerFixture) -> None:
    """A conversation idle past the longest documented cache retention is dropped, bounding memory."""
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    mocker.patch(
        'pydantic_ai._utils.now_utc',
        side_effect=[t0, t0 + timedelta(hours=25), t0 + timedelta(hours=25)],
    )
    model = ConversationModel()
    agent, exporter = conversation_agent(model)

    model.usages = [CacheUsage(write=1400)]
    first = agent.run_sync('first turn')
    # Any later update sweeps the conversations that went idle before it.
    model.usages = [CacheUsage(write=1400)]
    agent.run_sync('another conversation')
    model.usages = [CacheUsage(read=100)]
    agent.run_sync('second turn', message_history=first.all_messages())

    assert 'pydantic_ai.cache.collapsed' not in chat_cache_attributes(exporter)[-1]


def test_least_recently_updated_conversations_are_forgotten(mocker: MockerFixture) -> None:
    mocker.patch.object(ConversationCacheMarkStore, 'max_conversations', 1)
    model = ConversationModel()
    agent, exporter = conversation_agent(model)

    model.usages = [CacheUsage(write=1400)]
    first = agent.run_sync('first turn')
    model.usages = [CacheUsage(write=1400)]
    agent.run_sync('another conversation')
    model.usages = [CacheUsage(read=100)]
    agent.run_sync('second turn', message_history=first.all_messages())

    assert 'pydantic_ai.cache.collapsed' not in chat_cache_attributes(exporter)[-1]


def test_recently_updated_conversation_outlives_older_ones(mocker: MockerFixture) -> None:
    """Forgetting goes by the last update, not by when a conversation was first seen."""
    mocker.patch.object(ConversationCacheMarkStore, 'max_conversations', 2)
    model = ConversationModel()
    agent, exporter = conversation_agent(model)

    model.usages = [CacheUsage(write=1400)]
    first = agent.run_sync('first turn')
    model.usages = [CacheUsage(write=1400)]
    agent.run_sync('another conversation')
    model.usages = [CacheUsage(read=1400)]
    second = agent.run_sync('second turn', message_history=first.all_messages())
    model.usages = [CacheUsage(write=1400)]
    agent.run_sync('yet another conversation')
    model.usages = [CacheUsage(read=100)]
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
        other_model.usages = [CacheUsage(write=1400)]
        await other.run('another conversation')
        return 'done'

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        response = ConversationModel._respond(model, messages, info)  # pyright: ignore[reportPrivateUsage]
        if len(messages) == 1:
            response.parts = [ToolCallPart('run_another_conversation', {}, tool_call_id='call-1')]
        return response

    model.function = respond
    # The collapse re-baselines the mark to what the collapsing request established.
    model.usages = [CacheUsage(write=1400), CacheUsage(read=100, write=1300)]
    first = agent.run_sync('first turn')
    assert chat_cache_attributes(exporter)[-1] == {
        'pydantic_ai.cache.hit_ratio': 0.05,
        'pydantic_ai.cache.established_tokens': 1400,
        'pydantic_ai.cache.collapsed': True,
        'pydantic_ai.cache.wasted_tokens': 1300,
        'pydantic_ai.cache.collapse_reason': 'unexpected',
    }

    model.usages = [CacheUsage(read=100)]
    agent.run_sync('second turn', message_history=first.all_messages())
    assert chat_cache_attributes(exporter)[-1] == COLLAPSED_ON_CONTINUATION


def test_marks_without_a_conversation_are_not_stored() -> None:
    """A run without a conversation id has nothing to share its marks with, so they stay private to it."""
    store = ConversationCacheMarkStore()
    marks = store.get(None)
    marks[('test', None, 'cache-model')] = CacheMark(
        established_tokens=1400, last_seen=datetime.now(timezone.utc), run_id=None
    )
    store.update(None, marks, datetime.now(timezone.utc))

    assert store.get(None) == {}
    assert not store._conversations  # pyright: ignore[reportPrivateUsage]
