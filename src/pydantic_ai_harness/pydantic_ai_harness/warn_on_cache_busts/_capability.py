"""WarnOnCacheBusts: an observational cache-collapse warning.

This is the runtime `observe` arm of the cache-prefix-stability work. It does not
inspect the structured request (that signal false-positives on internal metadata
serialization strips, and is blind to serialization-level busts). Instead it reads
the provider's own ground-truth verdict -- `response.usage.cache_read_tokens` -- and
warns when a cache hit that was previously established collapses. That verdict is
cross-provider for free: pyai normalizes every provider into the `cache_read_tokens`
/ `cache_write_tokens` fields on `RequestUsage` via genai-prices.

The detection and classification are Pydantic AI core's (`pydantic_ai._cache_health`), the
same detector behind the `pydantic_ai.cache.*` span attributes and the
`pydantic_ai.cache.collapse` span event that instrumentation records. This capability is a
second output over it: a Python warning instead of telemetry.

The deterministic, always-on structural catch lives at the wire level in `tests/` (VCR
cassette prefix assertion), not here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Literal

from pydantic_ai._cache_health import (
    COLLAPSE_RATIO,
    MIN_PREFIX_TOKENS,
    CacheCollapse,
    CacheHealthDetector,
    CollapseReason,
    ConversationCacheMarkStore,
)
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness._warn import warn_argument_ignored

_WARN_ON: frozenset[CollapseReason] = frozenset({'unexpected', 'unknown'})
"""The collapses worth a warning: those the provider's retention window doesn't explain.

A `ttl-expired` collapse is the provider's cache expiring under an unchanged prefix, and an
`unreported` one can't be told apart from caching being off for that request, so neither is
evidence that the prefix moved.
"""

_SILENCE_HINT = (
    '    import warnings\n'
    '    from pydantic_ai_harness.warn_on_cache_busts import CacheBustWarning\n'
    "    warnings.filterwarnings('ignore', category=CacheBustWarning)  # silence\n"
    "    warnings.filterwarnings('error', category=CacheBustWarning)   # escalate in dev/CI"
)


@dataclass
class _RunState:
    """Per-run observation state: this run's step counter over its conversation's detector."""

    detector: CacheHealthDetector
    step: int = 0


class CacheBustWarning(UserWarning):
    """Warned when a previously-established prompt cache hit collapses on a later request.

    Emitted by `WarnOnCacheBusts` when a request read back far fewer cached tokens for the same
    provider, endpoint, and model than a prior request in the same conversation established --
    whether that prior request was earlier in this run or in an earlier run continued via
    `message_history` -- and the provider's cache retention window doesn't explain it. The
    likely cause is a moved cacheable prefix: reordered tools, injected timestamps, a
    serialization-level block hop, or history rewritten between turns.

    `reason` says how sure that is. `'unexpected'` means the retention window should still have
    been active, so the prefix moved. `'unknown'` means the provider publishes no retention
    window, so a provider-side cache expiry can't be ruled out.

    Silence it, or escalate it to an error in dev/CI, with the stdlib `warnings` machinery
    (no bespoke API):

        import warnings
        from pydantic_ai_harness.warn_on_cache_busts import CacheBustWarning

        # Silence the whole category:
        warnings.filterwarnings('ignore', category=CacheBustWarning)

        # Silence one intentional bust, scoped to the operation that causes it:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', CacheBustWarning)
            result = agent.run_sync('...')  # e.g. a step that switches models or adds a file

        # Treat every bust as an error (dev/CI enforcement):
        warnings.filterwarnings('error', category=CacheBustWarning)

    In tests, assert an intentional bust with `pytest.warns(CacheBustWarning)`, or silence
    a legitimately-busting test with
    `@pytest.mark.filterwarnings('ignore::pydantic_ai_harness.warn_on_cache_busts.CacheBustWarning')`.
    """

    reason: Literal['unexpected', 'unknown']
    """How the collapse was classified: `'unexpected'` when it happened within the provider's cache
    retention window, `'unknown'` when the provider publishes no retention window.

    Collapses classified `'ttl-expired'` (the retention window elapsed) or `'unreported'` (the
    response reported no cache usage) don't warn.
    """
    established_tokens: int
    """The cached-prefix size the request was judged against."""
    cache_read_tokens: int
    """The cached tokens the request read back."""
    wasted_tokens: int
    """Previously established tokens that were not read back."""


@dataclass
class WarnOnCacheBusts(AbstractCapability[AgentDepsT]):
    """Warn when a conversation's prompt cache hit collapses between requests.

    Attach it to any agent whose model uses prompt caching. On each response the monitor
    reads `usage.cache_read_tokens` and tracks the largest cacheable prefix the conversation
    has established (`cache_read_tokens + cache_write_tokens`, a high-water mark), keyed by the
    response's `(provider_name, provider_url, model_name)`. When a later request for the same
    key reads back fewer than `collapse_ratio` of that established prefix, the collapse is
    classified against the provider's cache retention window, and a `CacheBustWarning` is
    emitted unless the window explains it. The mark then re-baselines to what the collapsing
    request established, and the warning stays quiet about that collapse until a healthy
    read-back re-stabilizes the cache, so a sustained collapse warns once rather than on every
    subsequent request.

    The retention window is the one the request's settings ask for (such as
    `anthropic_cache='1h'`), as resolved by `Model.resolve_cache_retention()`, or else the
    provider's documented `ModelProfile.default_cache_retention`, extended by any `CachePoint`
    TTLs. A collapse after the window elapsed is a cache expiry (`ttl-expired`), not a moved
    prefix, and doesn't warn. When the provider publishes no retention window the collapse
    can't be attributed, so it warns with `reason='unknown'`. A response reporting no cache
    usage at all doesn't warn: it looks the same whether caching was off for that request or
    the cache fully missed.

    This is the same detector, with the same classification, that Pydantic AI's instrumentation
    uses for its `pydantic_ai.cache.*` span attributes. Its `pydantic_ai.cache.collapse` span
    event fires only for `unexpected` collapses; this warning also fires for `unknown` ones, so
    that it still catches moved prefixes on providers that don't publish a retention window.

    Marks are kept per conversation (`RunContext.conversation_id`), not per run, so a run
    that continues an earlier one via `message_history` -- including history that was
    serialized and loaded back, which carries the conversation id with it -- is judged against
    the prefix the earlier run established. That is where a moved prefix most often hides:
    the first request of the next turn re-sends what the previous turn cached. A run that
    starts a new conversation (no history, or `conversation_id='new'`) starts from a clean
    mark.

    Keying per provider, endpoint, and model means a mid-run model switch does not warn: a
    `FallbackModel` failover or a per-step model change uses a different cache key, so it
    starts a fresh mark for that key instead of comparing against the previous model's. Marks
    are kept per key rather than reset, so switching back to an earlier model still compares
    against that model's established prefix, and its retention window is timed from that same
    model's previous request, not whatever ran in between.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.warn_on_cache_busts import WarnOnCacheBusts

    agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[WarnOnCacheBusts()])
    result = await agent.run('...')  # a CacheBustWarning fires if a cached prefix collapses mid-run
    # ...and on the next turn, if the prefix the first turn cached no longer reads back:
    await agent.run('...', message_history=result.all_messages())
    ```

    The monitor is silent when caching is off or unreported (`cache_read_tokens` stays 0), so
    it never fires spuriously in tests that don't exercise caching. Silencing and dev/CI
    escalation both go through the stdlib `warnings` filters -- see `CacheBustWarning`.
    """

    collapse_ratio: float = COLLAPSE_RATIO
    """Warn when a request reads back less than this fraction of the established prefix.

    Conservative by default (0.5): only a drop below half the previously-cached prefix counts
    as a collapse, so ordinary provider rounding or a partial cache miss does not fire. Raise
    it toward 1.0 to warn on smaller regressions. Must be greater than 0.0 (a ratio of 0.0
    could never warn, so it is rejected rather than treated as a silent disable switch).
    """

    min_prefix_tokens: int = MIN_PREFIX_TOKENS
    """Only judge collapse once the established prefix reaches this many tokens.

    Below a provider's minimum cacheable size (Anthropic's is 1024) `cache_read_tokens` is
    noisy or zero, so small prefixes are ignored to avoid false positives.
    """

    cache_ttl_seconds: float | None = None
    """Deprecated and ignored: the cache retention window now comes from the model.

    It is the retention the request's settings ask for, else the provider's documented
    `ModelProfile.default_cache_retention`, extended by any `CachePoint` TTLs. A collapse after
    that window elapsed is classified as a cache expiry and doesn't warn.
    """

    _store: ConversationCacheMarkStore = field(
        init=False, default_factory=ConversationCacheMarkStore, compare=False, repr=False
    )
    # Private marks, in case a hook runs on this instance rather than on the copy `for_run` binds.
    _state: _RunState = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.collapse_ratio <= 1.0:
            raise ValueError('collapse_ratio must be greater than 0.0 and at most 1.0')
        if self.min_prefix_tokens < 0:
            raise ValueError('min_prefix_tokens must be non-negative')
        if self.cache_ttl_seconds is not None:
            warn_argument_ignored(
                'WarnOnCacheBusts',
                'cache_ttl_seconds',
                "the cache retention window now comes from the model's settings and profile "
                '(`Model.resolve_cache_retention()`, else `ModelProfile.default_cache_retention`), '
                'extended by any `CachePoint` TTLs. Remove the argument.',
            )
        self._state = _RunState(self._detector(conversation_id=None, run_id=None))

    def _detector(self, *, conversation_id: str | None, run_id: str | None) -> CacheHealthDetector:
        return CacheHealthDetector(
            self._store,
            conversation_id,
            run_id,
            alert_on=_WARN_ON,
            collapse_ratio=self.collapse_ratio,
            min_prefix_tokens=self.min_prefix_tokens,
        )

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        """Bind this run to its conversation's marks.

        The marks live on the instance the agent was built with, so every run of a conversation
        that goes through it -- in this process -- shares them. A run without a conversation id
        gets private marks and is judged alone. Marks are forgotten once a conversation has been
        idle for longer than any provider keeps a cache, or when more than 4,096 conversations
        have been active more recently.
        """
        # The deprecated argument already warned when this instance was built; the copy mustn't warn again.
        run = replace(self, cache_ttl_seconds=None)
        run._store = self._store
        run._state = _RunState(self._detector(conversation_id=ctx.conversation_id, run_id=ctx.run_id))
        return run

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Judge this response's cache read against the established prefix for its model, then update it."""
        state = self._state
        state.step += 1
        health = state.detector.observe(request_context, response)
        if health is not None and (collapse := health.collapse) is not None and collapse.alert:
            warnings.warn(_bust_warning(collapse, step=state.step, run_id=state.detector.run_id), stacklevel=2)
        return response


def _bust_warning(collapse: CacheCollapse, *, step: int, run_id: str | None) -> CacheBustWarning:
    previous = collapse.previous
    origin = 'a prior request' if previous.run_id == run_id else 'an earlier run of this conversation'
    idle = collapse.idle.total_seconds()
    reason = collapse.reason
    if reason == 'unknown':
        cause = (
            f'The provider publishes no cache retention window, so either the cacheable prefix moved between '
            f"requests or the provider's cache expired (the previous request for this model was ~{idle:.0f}s earlier)."
        )
    else:
        # `_WARN_ON` alerts on nothing else, and an `unexpected` collapse was judged against a known window.
        assert reason == 'unexpected' and collapse.retention is not None
        cause = (
            f'The previous request for this model was ~{idle:.0f}s earlier, within its '
            f'~{collapse.retention.total_seconds():.0f}s cache retention window, so the cacheable prefix moved '
            'between requests.'
        )
    warning = CacheBustWarning(
        f'Cache hit collapsed at model request {step}: read {collapse.cache_read_tokens} cached tokens but '
        f'{origin} established ~{previous.established_tokens} (~{collapse.wasted_tokens} tokens re-sent uncached). '
        f'{cause}\n\nTo silence or escalate:\n\n{_SILENCE_HINT}\n'
    )
    warning.reason = reason
    warning.established_tokens = previous.established_tokens
    warning.cache_read_tokens = collapse.cache_read_tokens
    warning.wasted_tokens = collapse.wasted_tokens
    return warning
