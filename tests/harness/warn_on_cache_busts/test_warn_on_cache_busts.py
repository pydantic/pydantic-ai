"""Tests for the observational `WarnOnCacheBusts` capability.

The public behavior is driven through `Agent(..., capabilities=[...])` with a
`FunctionModel` that returns preset `RequestUsage` per step, so each response
carries the `cache_read_tokens` / `cache_write_tokens` the monitor reads. The
repo runs pytest with `filterwarnings=['error']`, so an unexpected
`CacheBustWarning` fails a test on its own; runs that should stay silent assert
that explicitly.
"""

from __future__ import annotations

import warnings

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage, RunUsage

from pydantic_ai_harness.warn_on_cache_busts import (
    CacheBustWarning,
    WarnOnCacheBusts,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _usage(*, read: int = 0, write: int = 0) -> RequestUsage:
    return RequestUsage(input_tokens=10, output_tokens=5, cache_read_tokens=read, cache_write_tokens=write)


def _agent_for_runs(runs: list[list[RequestUsage]], monitor: WarnOnCacheBusts[None]) -> Agent[None, str]:
    """Agent whose model serves one preset-usage sequence per `Agent.run`, in order.

    Within a run, every response but the last returns a tool call so the run keeps
    stepping; the last returns text so the run finishes and the next `Agent.run` moves
    on to the next sequence. Each step's `after_model_request` sees the matching usage.
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

    return Agent(FunctionModel(fn), deps_type=type(None), capabilities=[monitor], tools=[noop])


def _agent(usages: list[RequestUsage], monitor: WarnOnCacheBusts[None]) -> Agent[None, str]:
    """Agent whose model emits one preset-usage response per step of a single run."""
    return _agent_for_runs([usages], monitor)


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


def _install_clock(monkeypatch: pytest.MonkeyPatch, times: list[float]) -> None:
    """Drive the monitor's monotonic clock with a preset sequence.

    The monitor reads its `_now` seam once when a run starts (`for_run`, to time out idle
    conversations) and once per model response, so `times` needs one entry per run start
    followed by one per step of that run. This controls the inter-request gap deterministically
    instead of relying on wall-clock timing.
    """
    seq = iter(times)
    monkeypatch.setattr('pydantic_ai_harness.warn_on_cache_busts._capability._now', lambda: next(seq))


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


async def test_below_min_prefix_never_warns() -> None:
    """A prefix under `min_prefix_tokens` is too small to judge, so a drop is ignored."""
    usages = [_usage(read=0, write=500), _usage(read=500), _usage(read=10)]
    agent = _agent(usages, WarnOnCacheBusts())
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        await agent.run('hi')


async def test_tunable_thresholds_catch_smaller_regression() -> None:
    """Lowering the floor and raising the ratio flags a regression the defaults ignore."""
    usages = [_usage(read=0, write=200), _usage(read=150)]
    monitor = WarnOnCacheBusts[None](collapse_ratio=1.0, min_prefix_tokens=100)
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


async def test_idle_conversation_is_forgotten_after_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """A conversation idle for longer than the cache TTL starts its next run from a clean mark.

    The provider cache is gone by then, so a low read-back is an expiry, not a bust: warning
    about it would be noise on every conversation a user comes back to after a break. The
    forgotten conversation is also dropped from memory, which is what bounds a long-lived
    agent's footprint.
    """
    _install_clock(monkeypatch, [0.0, 0.0, 0.0, 400.0, 400.0])
    monitor = WarnOnCacheBusts[None]()
    agent = _agent_for_runs([[_usage(read=0, write=8000), _usage(read=8000)], [_usage(read=100)]], monitor)
    first = await agent.run('first')
    assert len(monitor._conversations) == 1  # pyright: ignore[reportPrivateUsage]
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        second = await agent.run('second', message_history=first.all_messages())
    conversation_id = second.all_messages()[-1].conversation_id
    assert conversation_id == first.all_messages()[-1].conversation_id
    assert set(monitor._conversations) == {conversation_id}  # pyright: ignore[reportPrivateUsage]


async def test_other_idle_conversations_are_forgotten_when_a_run_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Starting any run sweeps every expired conversation, not just its own."""
    _install_clock(monkeypatch, [0.0, 0.0, 400.0, 400.0])
    monitor = WarnOnCacheBusts[None]()
    agent = _agent_for_runs([[_usage(read=0, write=8000)], [_usage(read=0, write=8000)]], monitor)
    first = await agent.run('first')
    second = await agent.run('second')
    first_id, second_id = first.all_messages()[-1].conversation_id, second.all_messages()[-1].conversation_id
    assert first_id != second_id
    assert set(monitor._conversations) == {second_id}  # pyright: ignore[reportPrivateUsage]


async def test_sweep_is_amortized_but_own_conversation_is_judged_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Other conversations are swept about once per TTL; a run's own conversation is checked on every start.

    A at 0 and 100, B at 301 (sweep due: A was last seen 201s ago, so it stays), A again at 450:
    no sweep is due (149s since the last one), but A's own gap of 350s exceeds the TTL, so A must
    still start from a clean mark rather than compare against an expired cache.
    """
    _install_clock(monkeypatch, [0.0, 0.0, 100.0, 100.0, 301.0, 301.0, 450.0, 450.0])
    monitor = WarnOnCacheBusts[None]()
    agent = _agent_for_runs(
        [[_usage(read=0, write=8000)], [_usage(read=8000)], [_usage(read=0, write=8000)], [_usage(read=100)]],
        monitor,
    )
    a1 = await agent.run('a1')
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        a2 = await agent.run('a2', message_history=a1.all_messages())
        b1 = await agent.run('b1')
        await agent.run('a3', message_history=a2.all_messages())
    a_id, b_id = a1.all_messages()[-1].conversation_id, b1.all_messages()[-1].conversation_id
    assert set(monitor._conversations) == {a_id, b_id}  # pyright: ignore[reportPrivateUsage]


async def test_marks_established_after_a_sweep_still_reach_the_next_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run swept out mid-flight re-registers its conversation, so what it establishes afterwards is kept.

    Run A of conversation X sits in a tool call long enough for a run of another conversation to
    sweep X out (400s idle). A then re-establishes a prefix; the next run of X must be judged
    against it, not start from a clean mark because A was writing into an orphaned state.
    """
    # A starts (0), A step 1 (0), B starts inside A's tool (400), B step 1 (400), A step 2 (401), C starts (402), C step 1 (402).
    _install_clock(monkeypatch, [0.0, 0.0, 400.0, 400.0, 401.0, 402.0, 402.0])
    responses = [
        ModelResponse(parts=[ToolCallPart('nested', {})], usage=_usage(read=0, write=8000)),  # A step 1
        ModelResponse(parts=[TextPart('done')], usage=_usage(read=0, write=8000)),  # B step 1, a new conversation
        ModelResponse(parts=[TextPart('done')], usage=_usage(read=8000, write=200)),  # A step 2, after the sweep
        ModelResponse(parts=[TextPart('done')], usage=_usage(read=100)),  # C step 1, continuing A's conversation
    ]
    state = {'i': 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        i = state['i']
        state['i'] += 1
        return responses[i]

    agent: Agent[None, str] = Agent(FunctionModel(fn), deps_type=type(None), capabilities=[WarnOnCacheBusts()])

    @agent.tool_plain
    async def nested() -> str:
        await agent.run('b')
        return 'ok'

    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        a = await agent.run('a')
    with pytest.warns(CacheBustWarning, match='an earlier run of this conversation established ~8200'):
        await agent.run('c', message_history=a.all_messages())


async def test_conversation_within_ttl_is_remembered(monkeypatch: pytest.MonkeyPatch) -> None:
    """A conversation resumed inside the TTL keeps its mark, and the gap is short enough to hedge generically."""
    _install_clock(monkeypatch, [0.0, 0.0, 0.0, 200.0, 200.0])
    agent = _agent_for_runs([[_usage(read=0, write=8000), _usage(read=8000)], [_usage(read=100)]], WarnOnCacheBusts())
    first = await agent.run('first')
    with pytest.warns(CacheBustWarning) as record:
        await agent.run('second', message_history=first.all_messages())
    message = str(record[0].message)
    assert 'e.g. a gap longer than the cache TTL' in message
    assert 'past the assumed' not in message


async def test_run_without_conversation_id_is_judged_alone() -> None:
    """A run context that carries no conversation id gets private marks and leaves no trace behind."""
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
    assert monitor._conversations == {}  # pyright: ignore[reportPrivateUsage]


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


async def test_switch_back_within_ttl_uses_preserved_mark() -> None:
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


async def test_expiry_gap_named_when_beyond_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """A collapse after a gap longer than the assumed TTL names the gap, avoiding mis-attribution."""
    _install_clock(monkeypatch, [0.0, 0.0, 400.0])
    usages = [_usage(read=0, write=8000), _usage(read=100)]
    agent = _agent(usages, WarnOnCacheBusts())
    with pytest.warns(CacheBustWarning, match='past the assumed') as record:
        await agent.run('hi')
    assert '400s earlier' in str(record[0].message)


async def test_small_gap_keeps_generic_expiry_hedge(monkeypatch: pytest.MonkeyPatch) -> None:
    """A collapse with a short inter-request gap keeps the generic TTL hedge, not a concrete gap."""
    _install_clock(monkeypatch, [0.0, 0.0, 5.0])
    usages = [_usage(read=0, write=8000), _usage(read=100)]
    agent = _agent(usages, WarnOnCacheBusts())
    with pytest.warns(CacheBustWarning) as record:
        await agent.run('hi')
    message = str(record[0].message)
    assert 'e.g. a gap longer than the cache TTL' in message
    assert 'past the assumed' not in message


async def test_expiry_gap_measured_per_key_after_switch_away_and_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """After switching away and back, the expiry gap is measured against the same model's last request.

    A global last-observation clock would time the gap from whatever ran in between (model B at
    250s), report ~150s, and withhold the expiry hedge exactly when expiry is the likely cause.
    Keying the clock per model measures A's own gap (400s) and names it.
    """
    _install_clock(monkeypatch, [0.0, 0.0, 250.0, 400.0])
    responses = [
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=0, write=8000), provider_name='anthropic'),
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=0, write=8000), provider_name='openai'),
        ModelResponse(parts=[TextPart('done')], usage=_usage(read=100), provider_name='anthropic'),
    ]
    agent = _agent_from_responses(responses, WarnOnCacheBusts())
    with pytest.warns(CacheBustWarning, match='past the assumed') as record:
        await agent.run('hi')
    assert '400s earlier' in str(record[0].message)


async def test_caching_off_mid_run_warns_once_not_per_step() -> None:
    """A `0/0` response after an established prefix warns once, not on every remaining request.

    Caching toggled off mid-run reports read==0, write==0. Against a mark that only grew, that
    tripped the collapse check on every subsequent step. The collapse latch surfaces it once and
    then stays quiet until a healthy read-back re-arms it.
    """
    usages = [_usage(read=0, write=8000), _usage(read=0, write=0), _usage(read=0, write=0)]
    agent = _agent(usages, WarnOnCacheBusts())
    with pytest.warns(CacheBustWarning) as record:
        await agent.run('hi')
    busts = [w for w in record if issubclass(w.category, CacheBustWarning)]
    assert len(busts) == 1
    assert 'request 2' in str(busts[0].message)


async def test_sustained_collapse_with_cache_writes_warns_once() -> None:
    """A run that keeps writing an unread cache (read stays low, write stays high) warns once.

    Each step reports read==0, write==2000: the prefix moves every request, so the provider
    re-writes a cache nothing reads back. Re-baselining the mark to `read + write` would hold it
    at 2000 and re-warn; re-baselining to `read` would still let the intervening `max()` re-grow
    it and warn every other step. The collapse latch is what holds it to a single warning.
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
    """A collapse that spans a turn boundary still warns once, not again on the next run's first request."""
    agent = _agent_for_runs(
        [[_usage(read=0, write=8000), _usage(read=100)], [_usage(read=100)]],
        WarnOnCacheBusts(),
    )
    with pytest.warns(CacheBustWarning, match='request 2'):
        first = await agent.run('first')
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        await agent.run('second', message_history=first.all_messages())


def test_invalid_config_rejected() -> None:
    """Out-of-range thresholds fail fast at construction rather than distorting detection."""
    with pytest.raises(ValueError, match='collapse_ratio'):
        WarnOnCacheBusts[None](collapse_ratio=1.5)
    with pytest.raises(ValueError, match='collapse_ratio'):
        WarnOnCacheBusts[None](collapse_ratio=-0.1)
    with pytest.raises(ValueError, match='min_prefix_tokens'):
        WarnOnCacheBusts[None](min_prefix_tokens=-1)
    with pytest.raises(ValueError, match='cache_ttl_seconds'):
        WarnOnCacheBusts[None](cache_ttl_seconds=-1.0)
    # `nan` compares false against everything and `inf` never elapses: either would keep every
    # conversation forever, so both are rejected rather than silently disabling eviction.
    with pytest.raises(ValueError, match='cache_ttl_seconds'):
        WarnOnCacheBusts[None](cache_ttl_seconds=float('nan'))
    with pytest.raises(ValueError, match='cache_ttl_seconds'):
        WarnOnCacheBusts[None](cache_ttl_seconds=float('inf'))


def test_config_boundaries() -> None:
    """`collapse_ratio=0.0` (never warns) and `cache_ttl_seconds=0.0` are rejected; `1.0` is accepted."""
    with pytest.raises(ValueError, match='collapse_ratio'):
        WarnOnCacheBusts[None](collapse_ratio=0.0)
    with pytest.raises(ValueError, match='cache_ttl_seconds'):
        WarnOnCacheBusts[None](cache_ttl_seconds=0.0)
    # The upper bound is inclusive: 1.0 warns on any regression at all.
    WarnOnCacheBusts[None](collapse_ratio=1.0)


def test_observation_state_is_not_constructor_surface() -> None:
    """Marks and timing live in non-init state, so they can't be seeded through the constructor."""
    with pytest.raises(TypeError):
        WarnOnCacheBusts[None](_state=None)  # pyright: ignore[reportCallIssue]
    with pytest.raises(TypeError):
        WarnOnCacheBusts[None](_conversations={})  # pyright: ignore[reportCallIssue]
