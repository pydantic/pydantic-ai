"""Tests for the observational `CacheStabilityMonitor` capability.

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
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from pydantic_ai_harness.cache_stability import (
    CacheBustWarning,
    CacheStabilityMonitor,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _usage(*, read: int = 0, write: int = 0) -> RequestUsage:
    return RequestUsage(input_tokens=10, output_tokens=5, cache_read_tokens=read, cache_write_tokens=write)


def _agent(usages: list[RequestUsage], monitor: CacheStabilityMonitor[None]) -> Agent[None, str]:
    """Agent whose model emits one preset-usage response per step.

    Every response but the last returns a tool call so the run keeps stepping;
    the last returns text so the run finishes. Each step's `after_model_request`
    sees the matching usage.
    """
    state = {'i': 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        i = state['i']
        state['i'] += 1
        usage = usages[i]
        if i == len(usages) - 1:
            return ModelResponse(parts=[TextPart('done')], usage=usage)
        return ModelResponse(parts=[ToolCallPart('noop', {})], usage=usage)

    def noop() -> str:
        return 'ok'

    return Agent(FunctionModel(fn), deps_type=type(None), capabilities=[monitor], tools=[noop])


def _agent_from_responses(responses: list[ModelResponse], monitor: CacheStabilityMonitor[None]) -> Agent[None, str]:
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
    """Drive the monitor's monotonic clock with a preset sequence, one value per model request.

    The monitor calls its `_now` seam exactly once per response, so `times` must have one entry
    per step. This controls the inter-request gap deterministically instead of relying on
    wall-clock timing.
    """
    seq = iter(times)
    monkeypatch.setattr('pydantic_ai_harness.cache_stability._capability._now', lambda: next(seq))


async def test_collapse_warns() -> None:
    """A large drop in cache_read below the established prefix warns."""
    usages = [_usage(read=0, write=8000), _usage(read=8000, write=200), _usage(read=500)]
    agent = _agent(usages, CacheStabilityMonitor())
    with pytest.warns(CacheBustWarning, match='request 3'):
        result = await agent.run('hi')
    assert result.output == 'done'


async def test_stable_prefix_is_silent() -> None:
    """An append-only run whose reads keep pace with the prefix never warns."""
    usages = [_usage(read=0, write=8000), _usage(read=8000, write=200), _usage(read=8200)]
    agent = _agent(usages, CacheStabilityMonitor())
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        result = await agent.run('hi')
    assert result.output == 'done'


async def test_below_min_prefix_never_warns() -> None:
    """A prefix under `min_prefix_tokens` is too small to judge, so a drop is ignored."""
    usages = [_usage(read=0, write=500), _usage(read=500), _usage(read=10)]
    agent = _agent(usages, CacheStabilityMonitor())
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        await agent.run('hi')


async def test_tunable_thresholds_catch_smaller_regression() -> None:
    """Lowering the floor and raising the ratio flags a regression the defaults ignore."""
    usages = [_usage(read=0, write=200), _usage(read=150)]
    monitor = CacheStabilityMonitor[None](collapse_ratio=1.0, min_prefix_tokens=100)
    agent = _agent(usages, monitor)
    with pytest.warns(CacheBustWarning):
        await agent.run('hi')


async def test_error_filter_escalates_to_exception() -> None:
    """`filterwarnings('error', ...)` turns a bust into a raised exception (dev/CI enforcement)."""
    usages = [_usage(read=0, write=8000), _usage(read=100)]
    agent = _agent(usages, CacheStabilityMonitor())
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        with pytest.raises(CacheBustWarning):
            await agent.run('hi')


async def test_for_run_resets_between_runs() -> None:
    """Reusing one monitor across runs judges each run independently (no leaked high-water mark)."""
    monitor = CacheStabilityMonitor[None]()

    busting = _agent([_usage(read=0, write=8000), _usage(read=100)], monitor)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', CacheBustWarning)
        await busting.run('first')

    # A second run with no caching must not inherit the first run's 8000-token prefix.
    silent = _agent([_usage(read=0, write=0), _usage(read=0, write=0)], monitor)
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        await silent.run('second')


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
    agent = Agent(fallback, deps_type=type(None), capabilities=[CacheStabilityMonitor()], tools=[noop])
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
    agent = _agent_from_responses(responses, CacheStabilityMonitor())
    with pytest.warns(CacheBustWarning, match='request 3'):
        result = await agent.run('hi')
    assert result.output == 'done'


async def test_expiry_gap_named_when_beyond_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """A collapse after a gap longer than the assumed TTL names the gap, avoiding mis-attribution."""
    _install_clock(monkeypatch, [0.0, 400.0])
    usages = [_usage(read=0, write=8000), _usage(read=100)]
    agent = _agent(usages, CacheStabilityMonitor())
    with pytest.warns(CacheBustWarning, match='past the assumed') as record:
        await agent.run('hi')
    assert '400s earlier' in str(record[0].message)


async def test_small_gap_keeps_generic_expiry_hedge(monkeypatch: pytest.MonkeyPatch) -> None:
    """A collapse with a short inter-request gap keeps the generic TTL hedge, not a concrete gap."""
    _install_clock(monkeypatch, [0.0, 5.0])
    usages = [_usage(read=0, write=8000), _usage(read=100)]
    agent = _agent(usages, CacheStabilityMonitor())
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
    _install_clock(monkeypatch, [0.0, 250.0, 400.0])
    responses = [
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=0, write=8000), provider_name='anthropic'),
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=0, write=8000), provider_name='openai'),
        ModelResponse(parts=[TextPart('done')], usage=_usage(read=100), provider_name='anthropic'),
    ]
    agent = _agent_from_responses(responses, CacheStabilityMonitor())
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
    agent = _agent(usages, CacheStabilityMonitor())
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
    agent = _agent(usages, CacheStabilityMonitor())
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
    agent = _agent(usages, CacheStabilityMonitor())
    with pytest.warns(CacheBustWarning) as record:
        await agent.run('hi')
    busts = [str(w.message) for w in record if issubclass(w.category, CacheBustWarning)]
    assert len(busts) == 2
    assert 'request 2' in busts[0]
    assert 'request 4' in busts[1]


def test_invalid_config_rejected() -> None:
    """Out-of-range thresholds fail fast at construction rather than distorting detection."""
    with pytest.raises(ValueError, match='collapse_ratio'):
        CacheStabilityMonitor[None](collapse_ratio=1.5)
    with pytest.raises(ValueError, match='collapse_ratio'):
        CacheStabilityMonitor[None](collapse_ratio=-0.1)
    with pytest.raises(ValueError, match='min_prefix_tokens'):
        CacheStabilityMonitor[None](min_prefix_tokens=-1)
    with pytest.raises(ValueError, match='cache_ttl_seconds'):
        CacheStabilityMonitor[None](cache_ttl_seconds=-1.0)


def test_config_boundaries() -> None:
    """`collapse_ratio=0.0` (never warns) and `cache_ttl_seconds=0.0` are rejected; `1.0` is accepted."""
    with pytest.raises(ValueError, match='collapse_ratio'):
        CacheStabilityMonitor[None](collapse_ratio=0.0)
    with pytest.raises(ValueError, match='cache_ttl_seconds'):
        CacheStabilityMonitor[None](cache_ttl_seconds=0.0)
    # The upper bound is inclusive: 1.0 warns on any regression at all.
    CacheStabilityMonitor[None](collapse_ratio=1.0)


def test_per_run_state_is_not_constructor_surface() -> None:
    """Per-run marks/timing live in non-init state, so they can't be seeded through the constructor."""
    with pytest.raises(TypeError):
        CacheStabilityMonitor[None](_state=None)  # pyright: ignore[reportCallIssue]
