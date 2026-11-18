"""`WarnOnCacheBusts`: warn when a conversation's prompt cache hit collapses.

It reads the provider-reported `usage.cache_read_tokens` rather than inspecting the
structured request, which false-positives on internal metadata and misses
serialization-level busts. Detection and classification are core's
(`pydantic_ai._cache_health`), the detector behind instrumentation's `pydantic_ai.cache.*`
span attributes.
"""

from __future__ import annotations

import warnings
from dataclasses import KW_ONLY, dataclass, field, replace
from functools import partial
from typing import Any, Literal

from pydantic_ai._cache_health import (
    MIN_MISSED_RATIO,
    MIN_MISSED_TOKENS,
    CacheCollapse,
    CacheHealthDetector,
    CollapseReason,
    ConversationCacheMarkStore,
)
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import AgentDepsT, RunContext

_WARN_ON: frozenset[CollapseReason] = frozenset({'unexpected', 'unknown'})
"""The collapses the provider's retention window doesn't explain (see `CacheBustWarning.reason`)."""

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

    Collapses classified `'ttl_expired'` (the retention window elapsed), `'compacted'` (a
    `CompactionPart` replaced the history), or `'unreported'` (the response reported no cache
    usage) don't warn.
    """
    established_tokens: int
    """The cached-prefix size the request was judged against."""
    cache_read_tokens: int
    """The cached tokens the request read back."""
    missed_tokens: int
    """Previously established tokens that were not read back."""

    def __init__(
        self,
        message: str,
        *,
        reason: Literal['unexpected', 'unknown'],
        established_tokens: int,
        cache_read_tokens: int,
        missed_tokens: int,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.established_tokens = established_tokens
        self.cache_read_tokens = cache_read_tokens
        self.missed_tokens = missed_tokens

    def __reduce__(self) -> tuple[Any, ...]:
        # `BaseException.__reduce__` rebuilds from `args` alone, which can't supply the keyword-only fields
        # that `copy` and `pickle` need to call `__init__` again.
        rebuild = partial(
            type(self),
            reason=self.reason,
            established_tokens=self.established_tokens,
            cache_read_tokens=self.cache_read_tokens,
            missed_tokens=self.missed_tokens,
        )
        return rebuild, self.args


class CacheNotEnabledWarning(UserWarning):
    """Warned when a request long enough to cache went to a model whose prompt caching wasn't configured.

    Emitted by `WarnOnCacheBusts` once per conversation and model, when the model caches nothing unless the
    request configures it (Anthropic, Bedrock Claude and Nova, and OpenRouter's Anthropic routes; OpenAI and
    OpenRouter's Gemini routes cache implicitly, so they're never reported), but neither the unified `cache`
    setting nor a provider-specific cache setting was set, the history has no `CachePoint`, and the provider
    reported no cache usage. Setting `cache=False` (or a provider-specific cache setting) says caching was
    considered, so it doesn't warn.

    Silence or escalate it with the stdlib `warnings` filters, like `CacheBustWarning`.
    """


@dataclass
class WarnOnCacheBusts(AbstractCapability[AgentDepsT]):
    """Warn when a conversation's prompt cache hit collapses between requests.

    Attach it to any agent whose model uses prompt caching. On each response the monitor
    reads `usage.cache_read_tokens` and tracks the cacheable prefix the conversation has
    established (`cache_read_tokens + cache_write_tokens`; it grows with the prefix and
    re-baselines after a collapse), keyed by the response's `(provider_name, provider_url, model_name)`. When a later request for the same
    key falls short of that established prefix by more than `min_missed_ratio` of it and by at
    least `min_missed_tokens`, the collapse is classified against the provider's cache retention
    window, and a `CacheBustWarning` is emitted unless the window explains it. The mark then
    re-baselines to what the collapsing request established, and the warning stays quiet about
    that collapse until a healthy read-back re-stabilizes the cache, so a sustained collapse
    warns once rather than on every subsequent request.

    The retention window is the one the request's settings ask for (such as
    `anthropic_cache='1h'`), as resolved by `Model.resolve_cache_retention()`, or else the
    provider's documented `ModelProfile.default_cache_retention`, extended by any `CachePoint`
    TTLs. A collapse after the window elapsed is a cache expiry (`ttl_expired`), not a moved
    prefix, and doesn't warn; nor does one explained by provider-native compaction
    (`compacted`), which replaces the history before a `CompactionPart` with its summary. When
    the provider publishes no retention window the collapse can't be attributed, so it warns
    with `reason='unknown'`. A response reporting no cache usage at all doesn't warn: it looks
    the same whether caching was off for that request or the cache fully missed.

    Marks are kept per conversation (`RunContext.conversation_id`), not per run, so a run
    that continues an earlier one via `message_history` is judged against the prefix the
    earlier run established. Keying per provider, endpoint, and model means a mid-run model
    switch starts a fresh mark for the new key instead of warning.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.warn_on_cache_busts import WarnOnCacheBusts

    agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[WarnOnCacheBusts()])
    result = await agent.run('...')  # a CacheBustWarning fires if a cached prefix collapses mid-run
    # ...and on the next turn, if the prefix the first turn cached no longer reads back:
    await agent.run('...', message_history=result.all_messages())
    ```

    The collapse warning is silent when caching is off or unreported (`cache_read_tokens` stays 0),
    so it never fires spuriously in tests that don't exercise caching. Separately, when a model that needs
    prompt caching configured on the request has none configured, a request long enough to cache emits a
    `CacheNotEnabledWarning`, once per conversation and model.
    Silencing and dev/CI escalation both go through the stdlib `warnings` filters -- see `CacheBustWarning`.
    """

    _: KW_ONLY

    min_missed_ratio: float = MIN_MISSED_RATIO
    """Only warn when a request falls short of the established prefix by more than this fraction of it.

    Message history is append-only, so any real shortfall means the prefix moved or the cache
    expired; the default (`0.05`) only keeps provider rounding out. Must be at least `0.0` and
    less than `1.0` (a request can't miss more than the whole prefix, so `1.0` could never warn).
    """

    min_missed_tokens: int = MIN_MISSED_TOKENS
    """Only warn when a request falls short of the established prefix by at least this many tokens.

    Keeps small partial misses on small prefixes out: together with the default `min_missed_ratio`
    this is the rule Claude Code uses for a prompt-cache miss.
    """

    _store: ConversationCacheMarkStore = field(
        init=False, default_factory=ConversationCacheMarkStore, compare=False, repr=False
    )
    # Private marks, in case a hook runs on this instance rather than on the copy `for_run` binds.
    _state: _RunState = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_missed_ratio < 1.0:
            raise ValueError('min_missed_ratio must be at least 0.0 and less than 1.0')
        if self.min_missed_tokens < 0:
            raise ValueError('min_missed_tokens must be non-negative')
        self._state = _RunState(self._detector(conversation_id=None, run_id=None))

    def _detector(self, *, conversation_id: str | None, run_id: str | None) -> CacheHealthDetector:
        return CacheHealthDetector(
            self._store,
            conversation_id,
            run_id,
            alert_on=_WARN_ON,
            min_missed_ratio=self.min_missed_ratio,
            min_missed_tokens=self.min_missed_tokens,
        )

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        """Bind this run to its conversation's marks.

        The marks live on the instance the agent was built with, so every run of a conversation
        that goes through it -- in this process -- shares them. A run without a conversation id
        gets private marks and is judged alone. Marks are forgotten once a conversation has been
        idle for longer than any provider keeps a cache, or when more than 4,096 conversations
        have been active more recently.
        """
        run = replace(self)
        run._store = self._store
        run._state = _RunState(run._detector(conversation_id=ctx.conversation_id, run_id=ctx.run_id))
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
        if health is not None and health.not_enabled:
            warnings.warn(_not_enabled_warning(response, step=state.step), stacklevel=2)
        elif health is not None and (collapse := health.collapse) is not None and collapse.alert:
            warnings.warn(_bust_warning(collapse, step=state.step, run_id=state.detector.run_id), stacklevel=2)
        return response


def _not_enabled_warning(response: ModelResponse, *, step: int) -> CacheNotEnabledWarning:
    return CacheNotEnabledWarning(
        f'Prompt caching is not enabled at model request {step}: {response.usage.input_tokens} input tokens were sent '
        f'to {response.model_name!r} uncached. Enable it with `model_settings={{"cache": True}}` or the `Caching()` '
        'capability, or set `cache=False` to leave it off on purpose.'
    )


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
    return CacheBustWarning(
        f'Cache hit collapsed at model request {step}: read {collapse.cache_read_tokens} cached tokens but '
        f'{origin} established ~{previous.established_tokens} (~{collapse.missed_tokens} tokens re-sent uncached). '
        f'{cause}\n\nTo silence or escalate:\n\n{_SILENCE_HINT}\n',
        reason=reason,
        established_tokens=previous.established_tokens,
        cache_read_tokens=collapse.cache_read_tokens,
        missed_tokens=collapse.missed_tokens,
    )
