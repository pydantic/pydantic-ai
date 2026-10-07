"""Tests for the observational `WarnOnCacheBusts` capability.

The public behavior is driven through `Agent(..., capabilities=[...])` with a
`FunctionModel` that returns preset `RequestUsage` per step, so each response
carries the `cache_read_tokens` / `cache_write_tokens` the monitor reads. The
repo runs pytest with `filterwarnings=['error']`, so an unexpected
`CacheBustWarning` fails a test on its own; runs that should stay silent assert
that explicitly.

A `FunctionModel` publishes no cache retention window unless a test gives it a
profile with `default_cache_retention`, so collapses under the plain helpers are
classified `unknown`; tests about retention pass `retention=`.
"""

from __future__ import annotations

import copy
import pickle
import warnings
from datetime import UTC, datetime, timedelta

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import (
    CompactionPart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage, RunUsage
from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.warn_on_cache_busts import (
    CacheBustWarning,
    WarnOnCacheBusts,
)

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def _usage(*, read: int = 0, write: int = 0, passes: int | None = None) -> RequestUsage:
    details = {} if passes is None else {'message_iterations': passes}
    return RequestUsage(
        input_tokens=10, output_tokens=5, cache_read_tokens=read, cache_write_tokens=write, details=details
    )


def _profile(retention: timedelta | None) -> ModelProfile | None:
    return ModelProfile(default_cache_retention=retention) if retention is not None else None


def _agent_for_runs(
    runs: list[list[RequestUsage]],
    monitor: WarnOnCacheBusts[None],
    *,
    retention: timedelta | None = None,
    instrumentation: Instrumentation | None = None,
) -> Agent[None, str]:
    """Agent whose model serves one preset-usage sequence per `Agent.run`, in order.

    Within a run, every response but the last returns a tool call so the run keeps
    stepping; the last returns text so the run finishes and the next `Agent.run` moves
    on to the next sequence. Each step's `after_model_request` sees the matching usage.
    `retention` is the model's documented cache retention; without it collapses are `unknown`.
    """
    queue = [list(run) for run in runs]

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        usages = queue[0]
        usage = usages.pop(0)
        if usages:
            return ModelResponse(parts=[ToolCallPart('noop', {})], usage=usage)
        queue.pop(0)
        return ModelResponse(parts=[TextPart('done')], usage=usage)

    def noop() -> str:
        return 'ok'

    return Agent(
        FunctionModel(fn, profile=_profile(retention)),
        deps_type=type(None),
        capabilities=[monitor] if instrumentation is None else [monitor, instrumentation],
        tools=[noop],
    )


def _agent(
    usages: list[RequestUsage], monitor: WarnOnCacheBusts[None], *, retention: timedelta | None = None
) -> Agent[None, str]:
    """Agent whose model emits one preset-usage response per step of a single run."""
    return _agent_for_runs([usages], monitor, retention=retention)


def _agent_from_responses(responses: list[ModelResponse], monitor: WarnOnCacheBusts[None]) -> Agent[None, str]:
    """Agent whose model replays preset `ModelResponse`s, one per step.

    Lets a test control `provider_name` per response (a mid-run model switch) -- a field
    `FunctionModel` leaves untouched -- which the simpler `_agent` helper can't. Every response
    but the last must carry a tool call so the run keeps stepping.
    """
    state = {'i': 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        i = state['i']
        state['i'] += 1
        return responses[i]

    def noop() -> str:
        return 'ok'

    return Agent(FunctionModel(fn), deps_type=type(None), capabilities=[monitor], tools=[noop])


class _Clock:
    """The cache-health clock, pinned so a test sets the gap between requests instead of sleeping."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.now = T0
        monkeypatch.setattr('pydantic_ai._utils.now_utc', lambda: self.now)


def _busts(record: pytest.WarningsRecorder) -> list[CacheBustWarning]:
    return [w.message for w in record if isinstance(w.message, CacheBustWarning)]


def _run_context(*, run_id: str, conversation_id: str | None) -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id=run_id, conversation_id=conversation_id)


def _request_context() -> ModelRequestContext:
    return ModelRequestContext(
        model=TestModel(), messages=[], model_settings=None, model_request_parameters=ModelRequestParameters()
    )


async def test_collapse_warns() -> None:
    """A large drop in cache_read below the established prefix warns."""
    usages = [_usage(read=0, write=8000), _usage(read=8000, write=200), _usage(read=500)]
    agent = _agent(usages, WarnOnCacheBusts())
    with pytest.warns(CacheBustWarning, match='request 3'):
        result = await agent.run('hi')
    assert result.output == 'done'


async def test_stable_prefix_is_silent() -> None:
    """An append-only run whose reads keep pace with the prefix never warns."""
    usages = [_usage(read=0, write=8000), _usage(read=8000, write=200), _usage(read=8200)]
    agent = _agent(usages, WarnOnCacheBusts())
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        result = await agent.run('hi')
    assert result.output == 'done'


async def test_small_shortfall_never_warns() -> None:
    """A miss has to be more than 5% of the established prefix and at least 2,000 tokens.

    1,800 tokens short of a 1,900-token prefix is under the absolute floor; 4,000 tokens short of a
    100,000-token prefix is under the relative one.
    """
    usages = [_usage(read=0, write=1900), _usage(read=100), _usage(read=0, write=100000), _usage(read=96000)]
    agent = _agent(usages, WarnOnCacheBusts())
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        await agent.run('hi')


async def test_partial_prefix_move_warns() -> None:
    """A change deep in the history moves only the tail of the prefix, so 90% still reads back: that warns."""
    agent = _agent([_usage(read=0, write=100000), _usage(read=90000)], WarnOnCacheBusts())
    with pytest.warns(CacheBustWarning) as record:
        await agent.run('hi')
    assert [bust.missed_tokens for bust in _busts(record)] == [10000]


async def test_tunable_thresholds_catch_smaller_regression() -> None:
    """Lowering both floors flags a regression the defaults ignore."""
    usages = [_usage(read=0, write=200), _usage(read=150)]
    monitor = WarnOnCacheBusts[None](min_missed_ratio=0.0, min_missed_tokens=10)
    agent = _agent(usages, monitor)
    with pytest.warns(CacheBustWarning):
        await agent.run('hi')


async def test_error_filter_escalates_to_exception() -> None:
    """`filterwarnings('error', ...)` turns a bust into a raised exception (dev/CI enforcement)."""
    usages = [_usage(read=0, write=8000), _usage(read=100)]
    agent = _agent(usages, WarnOnCacheBusts())
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        with pytest.raises(CacheBustWarning):
            await agent.run('hi')


async def test_new_conversation_starts_from_a_clean_mark() -> None:
    """Reusing one monitor across unrelated runs judges each conversation alone (no leaked mark)."""
    monitor = WarnOnCacheBusts[None]()

    busting = _agent([_usage(read=0, write=8000), _usage(read=100)], monitor)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', CacheBustWarning)
        await busting.run('first')

    # A second run without history is a new conversation: it must not inherit the 8000-token prefix.
    silent = _agent([_usage(read=0, write=0), _usage(read=0, write=0)], monitor)
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        await silent.run('second')


async def test_continuing_a_conversation_across_runs_warns() -> None:
    """The next turn's first request is judged against the prefix the previous run established.

    A per-run reset would give the second run no mark to compare against, so a prefix that
    moved between turns -- the most common place for one to move -- would never warn.
    """
    agent = _agent_for_runs([[_usage(read=0, write=8000), _usage(read=8000)], [_usage(read=100)]], WarnOnCacheBusts())
    first = await agent.run('first')
    with pytest.warns(CacheBustWarning, match='request 1') as record:
        await agent.run('second', message_history=first.all_messages())
    assert 'an earlier run of this conversation established ~8000' in str(record[0].message)


async def test_continuing_from_serialized_history_warns() -> None:
    """History that round-trips through JSON keeps its conversation id, so the mark still applies."""
    agent = _agent_for_runs([[_usage(read=0, write=8000), _usage(read=8000)], [_usage(read=100)]], WarnOnCacheBusts())
    first = await agent.run('first')
    history = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(first.all_messages()))
    with pytest.warns(CacheBustWarning, match='an earlier run of this conversation'):
        await agent.run('second', message_history=history)


async def test_healthy_continuation_is_silent() -> None:
    """A next turn that reads back the previous run's prefix is the stable case and stays silent."""
    agent = _agent_for_runs(
        [[_usage(read=0, write=8000), _usage(read=8000)], [_usage(read=8000, write=300), _usage(read=8300)]],
        WarnOnCacheBusts(),
    )
    first = await agent.run('first')
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        await agent.run('second', message_history=first.all_messages())


async def test_within_run_collapse_names_a_prior_request_not_an_earlier_run() -> None:
    """A collapse against a mark this same run established is worded as such."""
    agent = _agent_for_runs([[_usage(read=0, write=8000), _usage(read=8000)], [_usage(read=100)]], WarnOnCacheBusts())
    first = await agent.run('first')
    with pytest.warns(CacheBustWarning) as record:
        await agent.run('second', message_history=first.all_messages())
    assert 'a prior request' not in str(record[0].message)

    agent = _agent([_usage(read=0, write=8000), _usage(read=100)], WarnOnCacheBusts())
    with pytest.warns(CacheBustWarning, match='a prior request established ~8000'):
        await agent.run('hi')


async def test_forked_conversation_starts_from_a_clean_mark() -> None:
    """`conversation_id='new'` forks the history into a new conversation, which is judged alone."""
    agent = _agent_for_runs([[_usage(read=0, write=8000), _usage(read=8000)], [_usage(read=100)]], WarnOnCacheBusts())
    first = await agent.run('first')
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        await agent.run('second', message_history=first.all_messages(), conversation_id='new')


async def test_conversations_are_judged_apart() -> None:
    """Interleaved runs of two conversations keep separate marks.

    B's fresh mark must not be compared against A's prefix (a shared mark would warn on B's
    first request), and B's low read-back must not disturb A's mark (A's continuation still
    warns against its own 8000).
    """
    monitor = WarnOnCacheBusts[None]()
    agent = _agent_for_runs(
        [
            [_usage(read=0, write=8000), _usage(read=8000)],  # A, turn 1
            [_usage(read=100)],  # B, turn 1: a new conversation
            [_usage(read=8000)],  # A, turn 2: healthy
            [_usage(read=100)],  # A, turn 3: collapse
        ],
        monitor,
    )
    a1 = await agent.run('a1')
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        await agent.run('b1')
        a2 = await agent.run('a2', message_history=a1.all_messages())
    with pytest.warns(CacheBustWarning, match='an earlier run of this conversation established ~8000'):
        await agent.run('a3', message_history=a2.all_messages())


async def test_collapse_within_retention_warns_as_unexpected(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a known retention window, a collapse inside it is a moved prefix: it warns, classified `unexpected`."""
    clock = _Clock(monkeypatch)
    agent = _agent_for_runs(
        [[_usage(read=0, write=8000)], [_usage(read=100)]], WarnOnCacheBusts(), retention=timedelta(minutes=5)
    )
    first = await agent.run('first')
    clock.now = T0 + timedelta(minutes=4)
    with pytest.warns(CacheBustWarning) as record:
        await agent.run('second', message_history=first.all_messages())
    (bust,) = _busts(record)
    assert (bust.reason, bust.established_tokens, bust.cache_read_tokens, bust.missed_tokens) == (
        'unexpected',
        8000,
        100,
        7900,
    )
    assert 'was ~240s earlier, within its ~300s cache retention window, so the cacheable prefix moved' in str(bust)


async def test_collapse_after_retention_elapsed_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A collapse once the retention window has elapsed is the provider's cache expiring, not a bust.

    This is what replaced the old `cache_ttl_seconds` guess: the window comes from the model, and
    an expiry it explains (`ttl_expired`) doesn't warn, so a user coming back to a conversation
    after a break isn't told their prefix moved.
    """
    clock = _Clock(monkeypatch)
    agent = _agent_for_runs(
        [[_usage(read=0, write=8000)], [_usage(read=100)]], WarnOnCacheBusts(), retention=timedelta(minutes=5)
    )
    first = await agent.run('first')
    clock.now = T0 + timedelta(minutes=6)
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        await agent.run('second', message_history=first.all_messages())


async def test_retention_requested_by_settings_extends_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """The window is the one the request's settings ask for, not only the profile default.

    `FunctionModel` requests no retention of its own, so this subclass stands in for a provider
    setting like `anthropic_cache='1h'`: the 6-minute gap is past the profile's 5 minutes but
    inside the requested hour, so the collapse is `unexpected` and warns.
    """

    class OneHourCacheModel(FunctionModel):
        def resolve_cache_retention(self, model_settings: object) -> timedelta | None:
            return timedelta(hours=1)

    usages = [_usage(read=0, write=8000), _usage(read=100)]

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')], usage=usages.pop(0))

    clock = _Clock(monkeypatch)
    agent = Agent(
        OneHourCacheModel(fn, profile=_profile(timedelta(minutes=5))),
        deps_type=type(None),
        capabilities=[WarnOnCacheBusts()],
    )
    first = await agent.run('first')
    clock.now = T0 + timedelta(minutes=6)
    with pytest.warns(CacheBustWarning) as record:
        await agent.run('second', message_history=first.all_messages())
    assert [bust.reason for bust in _busts(record)] == ['unexpected']


async def test_run_without_conversation_id_is_judged_alone() -> None:
    """A run context that carries no conversation id gets private marks, so the next such run starts clean."""
    monitor = WarnOnCacheBusts[None]()
    first = await monitor.for_run(_run_context(run_id='run-1', conversation_id=None))
    await first.after_model_request(
        _run_context(run_id='run-1', conversation_id=None),
        request_context=_request_context(),
        response=ModelResponse(parts=[TextPart('done')], usage=_usage(read=0, write=8000)),
    )
    second = await monitor.for_run(_run_context(run_id='run-2', conversation_id=None))
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        await second.after_model_request(
            _run_context(run_id='run-2', conversation_id=None),
            request_context=_request_context(),
            response=ModelResponse(parts=[TextPart('done')], usage=_usage(read=100)),
        )


async def test_hook_on_an_unbound_instance_uses_private_marks() -> None:
    """A hook called on the instance the agent was built with, rather than a `for_run` copy, still judges."""
    monitor = WarnOnCacheBusts[None]()
    ctx = _run_context(run_id='run-1', conversation_id='conversation')
    await monitor.after_model_request(
        ctx,
        request_context=_request_context(),
        response=ModelResponse(parts=[TextPart('done')], usage=_usage(read=0, write=8000)),
    )
    with pytest.warns(CacheBustWarning, match='request 2: .* a prior request established ~8000'):
        await monitor.after_model_request(
            ctx,
            request_context=_request_context(),
            response=ModelResponse(parts=[TextPart('done')], usage=_usage(read=100)),
        )


async def test_model_failover_does_not_warn() -> None:
    """A mid-run `FallbackModel` failover reads an empty cache on the new model, which must not warn."""
    a_calls = {'n': 0}

    def model_a(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        a_calls['n'] += 1
        if a_calls['n'] == 1:
            # Establish a large cached prefix on model A, then keep the run stepping.
            return ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=0, write=8000))
        raise ModelAPIError('model-a', 'model A is down')

    def model_b(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # B's cache is empty: it reads back nothing of A's prefix.
        return ModelResponse(parts=[TextPart('done')], usage=_usage(read=0))

    def noop() -> str:
        return 'ok'

    fallback = FallbackModel(
        FunctionModel(model_a, model_name='model-a'),
        FunctionModel(model_b, model_name='model-b'),
    )
    agent = Agent(fallback, deps_type=type(None), capabilities=[WarnOnCacheBusts()], tools=[noop])
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        result = await agent.run('hi')
    assert result.output == 'done'


async def test_switch_back_uses_preserved_mark() -> None:
    """Marks are kept per model, so a collapse after switching back to an earlier model still warns.

    A reset-on-switch design would have discarded model A's mark at the switch to B, so the return
    to A would compare against nothing and stay silent. The warning proves the mark survived.
    """
    responses = [
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=0, write=8000), provider_name='anthropic'),
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=0, write=8000), provider_name='openai'),
        ModelResponse(parts=[TextPart('done')], usage=_usage(read=100), provider_name='anthropic'),
    ]
    agent = _agent_from_responses(responses, WarnOnCacheBusts())
    with pytest.warns(CacheBustWarning, match='request 3'):
        result = await agent.run('hi')
    assert result.output == 'done'


async def test_unknown_retention_warns_and_names_both_causes() -> None:
    """Without a retention window a collapse can't be attributed: it warns as `unknown`, naming both causes."""
    usages = [_usage(read=0, write=8000), _usage(read=100)]
    agent = _agent(usages, WarnOnCacheBusts())
    with pytest.warns(CacheBustWarning) as record:
        await agent.run('hi')
    (bust,) = _busts(record)
    assert (bust.reason, bust.established_tokens, bust.cache_read_tokens, bust.missed_tokens) == (
        'unknown',
        8000,
        100,
        7900,
    )
    message = str(bust)
    assert 'publishes no cache retention window' in message
    assert "or the provider's cache expired" in message


async def test_retention_is_timed_per_key_after_switch_away_and_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """After switching away and back, the retention window is timed from the same model's last request.

    Anthropic at 0, OpenAI at 4 minutes, Anthropic again at 6 minutes, with a 5-minute window.
    Timed from whatever ran in between (2 minutes) the collapse would look `unexpected` and warn;
    timed from Anthropic's own previous request (6 minutes) it is an expiry and stays silent.
    """
    clock = _Clock(monkeypatch)
    responses = [
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=0, write=8000), provider_name='anthropic'),
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=0, write=8000), provider_name='openai'),
        ModelResponse(parts=[TextPart('done')], usage=_usage(read=100), provider_name='anthropic'),
    ]
    times = [T0, T0 + timedelta(minutes=4), T0 + timedelta(minutes=6)]

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        clock.now = times.pop(0)
        return responses.pop(0)

    def noop() -> str:
        return 'ok'

    agent = Agent(
        FunctionModel(fn, profile=_profile(timedelta(minutes=5))),
        deps_type=type(None),
        capabilities=[WarnOnCacheBusts()],
        tools=[noop],
    )
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        await agent.run('hi')


async def test_compaction_does_not_warn() -> None:
    """Provider-native compaction replaces the history before its `CompactionPart` with a summary, so the
    request that compacted reads back far less than the old prefix by design: classified `compacted`, silent.
    """
    responses = [
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=0, write=8000)),
        ModelResponse(parts=[CompactionPart(content='summary'), TextPart('done')], usage=_usage(read=0, write=1500)),
    ]
    agent = _agent_from_responses(responses, WarnOnCacheBusts())
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        await agent.run('hi')


async def test_unreported_cache_usage_does_not_warn() -> None:
    """A `0/0` response after an established prefix doesn't warn.

    It looks the same whether caching was off for that request (on providers that report cache
    writes) or the cache fully missed (on providers that only report reads), so it isn't evidence
    that the prefix moved. The mark stays put, so a later healthy read-back is judged against it.
    """
    usages = [_usage(read=0, write=8000), _usage(read=0, write=0), _usage(read=0, write=0), _usage(read=8000)]
    agent = _agent(usages, WarnOnCacheBusts())
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        await agent.run('hi')


async def test_sustained_collapse_with_cache_writes_warns_once() -> None:
    """A run that keeps writing an unread cache (read stays low, write stays high) warns once.

    Each step reports read==0, write==2000: the prefix moves every request, so the provider
    re-writes a cache nothing reads back. After the first collapse the mark re-baselines to the
    2000 tokens that request wrote, and every later request collapses against that too; the
    alert latch is what holds it to a single warning until a healthy read-back re-arms it.
    """
    usages = [
        _usage(read=0, write=8000),
        _usage(read=0, write=2000),
        _usage(read=0, write=2000),
        _usage(read=0, write=2000),
    ]
    agent = _agent(usages, WarnOnCacheBusts())
    with pytest.warns(CacheBustWarning) as record:
        await agent.run('hi')
    busts = [w for w in record if issubclass(w.category, CacheBustWarning)]
    assert len(busts) == 1
    assert 'request 2' in str(busts[0].message)


async def test_recollapse_after_restabilize_warns_again() -> None:
    """The latch re-arms: a healthy read-back between two collapses lets the second one warn."""
    usages = [
        _usage(read=0, write=8000),  # establish 8000
        _usage(read=100),  # collapse -> warn (request 2)
        _usage(read=8000, write=200),  # healthy read-back re-stabilizes, clearing the latch
        _usage(read=100),  # collapse again -> warn (request 4)
    ]
    agent = _agent(usages, WarnOnCacheBusts())
    with pytest.warns(CacheBustWarning) as record:
        await agent.run('hi')
    busts = [str(w.message) for w in record if issubclass(w.category, CacheBustWarning)]
    assert len(busts) == 2
    assert 'request 2' in busts[0]
    assert 'request 4' in busts[1]


async def test_collapse_latch_carries_across_runs() -> None:
    """A collapse that spans a turn boundary still warns once, not again on the next run's first request.

    Each request writes a fresh 2000-token prefix nothing reads back: the second run's first request
    collapses against the mark the first run re-baselined to, but the cache never re-stabilized in
    between, so it stays quiet.
    """
    agent = _agent_for_runs(
        [[_usage(read=0, write=8000), _usage(read=0, write=2000)], [_usage(read=0, write=2000)]],
        WarnOnCacheBusts(),
    )
    with pytest.warns(CacheBustWarning, match='request 2'):
        first = await agent.run('first')
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        await agent.run('second', message_history=first.all_messages())


def _native_tool_response(usage: RequestUsage, *, steps: bool = True) -> ModelResponse:
    """A response containing a native tool call.

    The trailing `ToolCallPart` keeps the run stepping; pass `steps=False` for a final text answer.
    """
    parts = [
        NativeToolCallPart('web_search', {'query': 'x'}, tool_call_id='srv-1', provider_name='test'),
        NativeToolReturnPart('web_search', 'results', tool_call_id='srv-1', provider_name='test'),
        ToolCallPart('noop', {}) if steps else TextPart('done'),
    ]
    return ModelResponse(parts=parts, usage=usage)


async def test_native_tool_response_does_not_raise_the_mark() -> None:
    """A response with native tool calls reports usage summed over its sampling passes.

    Three web searches inside one request read the ~8k prefix on each of four passes, so the
    response says ~33k cached tokens. That is not a prefix the next request can read back; raising
    the mark to it made the next, perfectly healthy request look like a collapse.
    """
    responses = [
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=0, write=8000)),
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=8000, write=200)),
        _native_tool_response(_usage(read=32800, write=600)),
        ModelResponse(parts=[TextPart('done')], usage=_usage(read=8400, write=300)),
    ]
    agent = _agent_from_responses(responses, WarnOnCacheBusts())
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        result = await agent.run('hi')
    assert result.output == 'done'


async def test_collapse_after_native_tool_response_still_warns() -> None:
    """The mark established before a native-tool response still judges the requests after it."""
    responses = [
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=0, write=8000)),
        _native_tool_response(_usage(read=32000, write=600)),
        ModelResponse(parts=[TextPart('done')], usage=_usage(read=500)),
    ]
    agent = _agent_from_responses(responses, WarnOnCacheBusts())
    with pytest.warns(CacheBustWarning, match='request 3.*established ~8000'):
        await agent.run('hi')


async def test_native_tool_response_can_prove_a_collapse() -> None:
    """A summed read that is still below the threshold means the first pass read little: warn.

    Pins that a native-tool response is judged, not skipped: every pass read at least what the
    first did, so a low total is a real collapse even though a high one proves nothing.
    """
    responses = [
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=0, write=8000)),
        _native_tool_response(_usage(read=1000, write=9000), steps=False),
    ]
    agent = _agent_from_responses(responses, WarnOnCacheBusts())
    with pytest.warns(CacheBustWarning, match='request 2'):
        await agent.run('hi')


async def test_native_tool_response_does_not_clear_the_collapse_latch() -> None:
    """A healthy-looking summed read cannot prove the cache re-stabilized, so a sustained collapse warns once."""
    responses = [
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=0, write=8000)),
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=100)),  # collapse -> warn
        _native_tool_response(_usage(read=16000, write=8000)),  # two passes, each re-reading a rewritten cache
        ModelResponse(parts=[TextPart('done')], usage=_usage(read=100)),  # still collapsed: latched
    ]
    agent = _agent_from_responses(responses, WarnOnCacheBusts())
    with pytest.warns(CacheBustWarning) as record:
        await agent.run('hi')
    busts = [str(w.message) for w in record if issubclass(w.category, CacheBustWarning)]
    assert len(busts) == 1
    assert 'request 2' in busts[0]


async def test_tool_use_prompt_accounting_updates_the_mark_and_rearms_the_latch() -> None:
    """Google's separate tool-use prompt count leaves cache reads as an ordinary prefix count."""
    responses = [
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=0, write=8000)),
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=100)),  # collapse -> warn
        _native_tool_response(
            RequestUsage(
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=12000,
                cache_write_tokens=0,
                details={'tool_use_prompt_tokens': 24000},
            )
        ),  # healthy cache read -> raise the mark and re-arm
        ModelResponse(parts=[TextPart('done')], usage=_usage(read=5000)),  # collapse against 12000
    ]
    agent = _agent_from_responses(responses, WarnOnCacheBusts())
    with pytest.warns(CacheBustWarning) as record:
        await agent.run('hi')

    busts = [str(w.message) for w in record if issubclass(w.category, CacheBustWarning)]
    assert len(busts) == 2
    assert 'request 2' in busts[0]
    assert 'request 4' in busts[1]
    assert 'established ~12000' in busts[1]


async def test_reported_single_pass_with_a_native_tool_establishes_the_mark() -> None:
    """The reported pass count overrides the native-tool part heuristic."""
    responses = [
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=0, write=4000)),
        _native_tool_response(_usage(read=4000, write=4000, passes=1)),  # raises the mark to 8000
        ModelResponse(parts=[TextPart('done')], usage=_usage(read=3000)),  # below half of 8000
    ]
    agent = _agent_from_responses(responses, WarnOnCacheBusts())
    with pytest.warns(CacheBustWarning, match='request 3.*established ~8000'):
        await agent.run('hi')


async def test_reported_single_pass_with_compaction_and_a_native_tool_does_not_raise_the_mark() -> None:
    responses: list[ModelResponse] = [
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=0, write=8000)),
        _native_tool_response(
            RequestUsage(
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=20000,
                cache_write_tokens=0,
                details={'message_iterations': 1, 'compaction_iterations': 1},
            )
        ),
        ModelResponse(parts=[TextPart('done')], usage=_usage(read=8000)),
    ]
    agent = _agent_from_responses(responses, WarnOnCacheBusts())
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        result = await agent.run('hi')
    assert result.output == 'done'


async def test_reported_multiple_passes_without_a_native_tool_part_keeps_the_mark() -> None:
    """A reported pass count above one is multi-pass even when no native tool part survived."""
    responses = [
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=0, write=8000)),
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=24000, write=600, passes=3)),
        ModelResponse(parts=[TextPart('done')], usage=_usage(read=8200)),
    ]
    agent = _agent_from_responses(responses, WarnOnCacheBusts())
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        result = await agent.run('hi')
    assert result.output == 'done'


async def test_native_tool_response_as_first_request_establishes_no_mark() -> None:
    """A conversation whose first response ran a native tool starts without a mark, not from its sum."""
    responses = [
        _native_tool_response(_usage(read=0, write=8000)),  # one pass wrote 8000; the sum cannot say which
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=100)),  # no mark yet: silent
        ModelResponse(parts=[TextPart('done')], usage=_usage(read=100)),
    ]
    agent = _agent_from_responses(responses, WarnOnCacheBusts())
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        result = await agent.run('hi')
    assert result.output == 'done'


async def test_mark_kept_across_a_native_tool_response_still_names_the_earlier_run() -> None:
    """The kept mark keeps its origin: a collapse after a next-turn native-tool response names the earlier run."""
    responses = [
        ModelResponse(parts=[TextPart('first')], usage=_usage(read=0, write=8000)),
        _native_tool_response(_usage(read=24000, write=600)),  # second run opens with three passes
        ModelResponse(parts=[TextPart('done')], usage=_usage(read=100)),
    ]
    agent = _agent_from_responses(responses, WarnOnCacheBusts())
    first = await agent.run('first')
    with pytest.warns(CacheBustWarning, match='request 2.*an earlier run of this conversation established ~8000'):
        await agent.run('second', message_history=first.all_messages())


def test_invalid_config_rejected() -> None:
    """Out-of-range thresholds fail fast at construction rather than distorting detection."""
    with pytest.raises(ValueError, match='min_missed_ratio'):
        WarnOnCacheBusts[None](min_missed_ratio=-0.1)
    with pytest.raises(ValueError, match='min_missed_tokens'):
        WarnOnCacheBusts[None](min_missed_tokens=-1)
    with pytest.raises(ValueError, match='collapse_ratio'):
        WarnOnCacheBusts[None](collapse_ratio=1.5)
    with pytest.raises(ValueError, match='min_prefix_tokens'):
        WarnOnCacheBusts[None](min_prefix_tokens=-1)


def test_config_boundaries() -> None:
    """`min_missed_ratio=1.0` (a request can't miss more than the whole prefix) is rejected; `0.0` is accepted."""
    with pytest.raises(ValueError, match='min_missed_ratio'):
        WarnOnCacheBusts[None](min_missed_ratio=1.0)
    # The lower bound is inclusive: 0.0 leaves only `min_missed_tokens` to decide.
    WarnOnCacheBusts[None](min_missed_ratio=0.0)


async def test_collapse_ratio_is_deprecated_and_converted() -> None:
    """`collapse_ratio` still works, as its `min_missed_ratio` inverse, and warns once at construction.

    With `collapse_ratio=0.5`, reading back 5,000 of 8,000 tokens (37.5% missed) stays quiet where the
    5% default would warn, and reading back 3,000 (62.5% missed) warns. A positional first argument is
    still `collapse_ratio`.
    """
    with pytest.warns(HarnessDeprecationWarning, match=r'pass `min_missed_ratio=0.5`'):
        monitor = WarnOnCacheBusts[None](collapse_ratio=0.5)
    assert monitor.min_missed_ratio == 0.5
    agent = _agent([_usage(read=0, write=8000), _usage(read=5000), _usage(read=3000)], monitor)
    with pytest.warns(CacheBustWarning) as record:
        await agent.run('hi')
    assert [type(w.message) for w in record] == [CacheBustWarning]
    assert 'request 3' in str(record[0].message)

    with pytest.warns(HarnessDeprecationWarning, match='collapse_ratio'):
        assert WarnOnCacheBusts[None](0.25).min_missed_ratio == 0.75
    with pytest.raises(TypeError, match='`collapse_ratio` is its deprecated inverse'):
        WarnOnCacheBusts[None](collapse_ratio=0.5, min_missed_ratio=0.1)


async def test_min_prefix_tokens_is_deprecated_and_still_honored() -> None:
    """`min_prefix_tokens` warns once at construction and still keeps smaller prefixes from being judged."""
    with pytest.warns(HarnessDeprecationWarning, match='min_missed_tokens'):
        monitor = WarnOnCacheBusts[None](min_prefix_tokens=50000)
    agent = _agent([_usage(read=0, write=40000), _usage(read=100)], monitor)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        await agent.run('hi')


async def test_cache_ttl_seconds_is_deprecated_and_ignored() -> None:
    """`cache_ttl_seconds` warns once, at construction, and no longer affects detection.

    The collapse below comes well within the old 1-second TTL, which used to only change the
    message; it still warns, and running the agent doesn't repeat the deprecation (the `for_run`
    copy must not re-trigger it), which the suite's `filterwarnings=error` would turn into a failure.
    """
    with pytest.warns(HarnessDeprecationWarning, match=r'`WarnOnCacheBusts\(cache_ttl_seconds=...\)` is deprecated'):
        monitor = WarnOnCacheBusts[None](cache_ttl_seconds=1.0)
    agent = _agent([_usage(read=0, write=8000), _usage(read=100)], monitor)
    with pytest.warns(CacheBustWarning) as record:
        await agent.run('hi')
    assert [type(w.message) for w in record] == [CacheBustWarning]


def test_bust_warning_constructed_directly_carries_its_fields() -> None:
    """A `CacheBustWarning` built outside the monitor has its fields, and keeps them through `copy` and `pickle`."""
    warning = CacheBustWarning(
        'collapsed', reason='unexpected', established_tokens=8000, cache_read_tokens=100, missed_tokens=7900
    )
    for restored in (warning, copy.copy(warning), pickle.loads(pickle.dumps(warning))):
        assert restored.args == ('collapsed',)
        assert (restored.reason, restored.established_tokens, restored.cache_read_tokens, restored.missed_tokens) == (
            'unexpected',
            8000,
            100,
            7900,
        )


def test_observation_state_is_not_constructor_surface() -> None:
    """Marks live in non-init state, so they can't be seeded through the constructor."""
    with pytest.raises(TypeError):
        WarnOnCacheBusts[None](_state=None)  # pyright: ignore[reportCallIssue]
    with pytest.raises(TypeError):
        WarnOnCacheBusts[None](_store=None)  # pyright: ignore[reportCallIssue]


# ---- Alongside Pydantic AI's instrumentation -----------------------------------------------------


def _instrumentation() -> tuple[Instrumentation, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return Instrumentation(
        settings=InstrumentationSettings(tracer_provider=tracer_provider, include_content=False)
    ), exporter


@pytest.mark.parametrize(
    ('retention', 'gap', 'second', 'reason'),
    [
        (timedelta(minutes=5), timedelta(minutes=4), _usage(read=100), 'unexpected'),
        (timedelta(minutes=5), timedelta(minutes=6), _usage(read=100), 'ttl_expired'),
        (None, timedelta(minutes=4), _usage(read=100), 'unknown'),
        (timedelta(minutes=5), timedelta(minutes=4), _usage(read=0, write=0), 'unreported'),
    ],
)
async def test_classifies_like_instrumentation_with_both_on_one_agent(
    monkeypatch: pytest.MonkeyPatch, retention: timedelta | None, gap: timedelta, second: RequestUsage, reason: str
) -> None:
    """With instrumentation on the same agent, both judge the continuation's collapse, and agree on why.

    Each keeps its own marks: if they shared one set, whichever observed the response first would
    re-baseline the mark to the collapsing request's 100 tokens, and the other would see no collapse.
    The span records every classification; the event fires for `unexpected` only, and the warning
    for `unexpected` and `unknown`.
    """
    clock = _Clock(monkeypatch)
    instrumentation, exporter = _instrumentation()
    agent = _agent_for_runs(
        [[_usage(read=0, write=8000)], [second]],
        WarnOnCacheBusts(),
        retention=retention,
        instrumentation=instrumentation,
    )
    first = await agent.run('first')
    clock.now = T0 + gap
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always', CacheBustWarning)
        await agent.run('second', message_history=first.all_messages())

    span = [span for span in exporter.get_finished_spans() if span.name.startswith('chat ')][-1]
    assert (span.attributes or {})['pydantic_ai.cache.collapse_reason'] == reason
    assert [event.name for event in span.events] == (['pydantic_ai.cache.collapse'] if reason == 'unexpected' else [])
    busts = [w.message for w in record if isinstance(w.message, CacheBustWarning)]
    assert [bust.reason for bust in busts] == ([reason] if reason in ('unexpected', 'unknown') else [])


async def test_both_on_one_agent_latch_and_rearm_together() -> None:
    """A sustained collapse surfaces once through each output, and both re-arm on the same healthy read-back."""
    instrumentation, exporter = _instrumentation()
    usages = [
        _usage(read=0, write=8000),
        _usage(read=0, write=2000),
        _usage(read=0, write=2000),
        _usage(read=2000),
        _usage(read=0, write=2000),
    ]
    agent = _agent_for_runs([usages], WarnOnCacheBusts(), retention=timedelta(hours=1), instrumentation=instrumentation)
    with pytest.warns(CacheBustWarning) as record:
        await agent.run('hi')

    spans = [span for span in exporter.get_finished_spans() if span.name.startswith('chat ')]
    assert [bool(span.events) for span in spans] == [False, True, False, False, True]
    assert [str(bust).split(':')[0] for bust in _busts(record)] == [
        'Cache hit collapsed at model request 2',
        'Cache hit collapsed at model request 5',
    ]
