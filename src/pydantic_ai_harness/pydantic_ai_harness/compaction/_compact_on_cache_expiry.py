"""`CompactOnCacheExpiry` -- request compaction once the provider's prompt cache has expired."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from pydantic_ai._utils import replace_no_init
from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    CompactionRequest,
    CompactionResult,
    durable_operation,
)
from pydantic_ai.messages import CompactionPart, ModelMessage, ModelRequest, ModelResponse
from pydantic_ai.profiles import ModelProfile, PromptCacheOutlook, prompt_cache_outlook
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness.compaction._clear_tool_results import ClearToolResults
from pydantic_ai_harness.compaction._context_window import DEFAULT_CONTEXT_WINDOW
from pydantic_ai_harness.compaction._pinning import reinject_pinned
from pydantic_ai_harness.compaction._shared import (
    CompactionStrategy,
    compact_with_span,
    estimate_context_tokens,
    get_compaction_reclaim,
    resolve_token_trigger,
    validate_token_trigger,
)

if TYPE_CHECKING:
    from pydantic_ai.models import Model, ModelRequestContext
    from pydantic_ai.settings import ModelSettings


DEFAULT_MAX_FRACTION = 0.2
"""Default size, as a fraction of the context window, below which an expired history is left alone."""


class _DefaultFraction(float):
    """The default `max_fraction`, told apart from an explicit `0.2` so one beside `max_tokens` is still rejected."""


# Type checkers see the plain default; at runtime the sentinel tells an omitted `max_fraction` from an explicit one,
# so the public field type stays what callers can pass.
if TYPE_CHECKING:
    _DEFAULT_FRACTION: float = DEFAULT_MAX_FRACTION
else:
    _DEFAULT_FRACTION = _DefaultFraction(DEFAULT_MAX_FRACTION)

CLEARED_TOOL_RESULT_HINT = '[Tool result cleared to save context. Call the tool again if you need it.]'
"""Placeholder the default fallback leaves in place of a cleared tool result.

It tells the model the result can be fetched again: with a bare placeholder, models answered from
what they guessed the cleared result said rather than calling the tool again."""

FALLBACK_NOTE = 'Earlier tool results in this conversation were cleared to save context.'
"""The `CompactionPart` content the fallback leaves when its strategy didn't write a summary of its own."""

_FALLBACK_METADATA_KEY = 'pydantic_ai_harness.compact_on_cache_expiry'
"""`CompactionPart.metadata` key marking the part the fallback left, so the next one replaces it."""


def _default_fallback() -> CompactionStrategy[Any]:
    # `max_tokens=1` only satisfies `ClearToolResults`' required trigger, which a fallback bypasses.
    return ClearToolResults(max_tokens=1, keep_pairs=5, placeholder=CLEARED_TOOL_RESULT_HINT)


@dataclass
class CompactOnCacheExpiry(AbstractCapability[AgentDepsT]):
    """Request compaction on the first request after the provider's prompt cache has expired.

    Rewriting history moves the cached prompt prefix, so the next request pays a cache write for
    everything after the edit. Once the conversation has been idle for longer than the provider
    keeps the prefix, the next request pays that write anyway, so compacting then costs no extra
    cache spend.

    It doesn't decide how to compact: it calls
    [`ctx.request_compaction(reason='cache_expiry')`][pydantic_ai.tools.RunContext.request_compaction],
    and the agent's capability with a
    [`compact_history`][pydantic_ai.capabilities.AbstractCapability.compact_history] hook serves it,
    like [`NativeCompaction`][pydantic_ai.capabilities.NativeCompaction]. When none does, `fallback`
    runs, as this capability's own `compact_history` hook, which is asked after every other capability's.

    The decision uses [`prompt_cache_outlook`][pydantic_ai.profiles.prompt_cache_outlook]: the gap
    from the last model response to the request being sent, against the retention the request's
    settings ask for or else the provider's documented default. When the outlook is `'unknown'`
    (no documented retention, a `FallbackModel`, or a history without a response) nothing is
    compacted, unless `fallback_retention` is set.

    By default only a history above a fifth of the model's context window is compacted, so a short
    conversation keeps its tool results.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.capabilities import Caching, NativeCompaction
        from pydantic_ai_harness import CompactOnCacheExpiry

        agent = Agent(
            'anthropic:claude-sonnet-5',
            capabilities=[Caching(), NativeCompaction(), CompactOnCacheExpiry()],
        )
        ```
    """

    fallback: CompactionStrategy[AgentDepsT] | None = field(default_factory=_default_fallback)
    """The compaction to run when no other capability served the request. `None` leaves the history alone.

    Its own trigger is ignored. Defaults to clearing all but the 5 most recent tool results, with a
    placeholder that tells the model it can call the tool again."""

    max_tokens: int | None = field(default=None, kw_only=True)
    """Only compact an expired history whose estimated context tokens exceed this value.

    Replaces the default `max_fraction`."""

    max_fraction: float | None = field(default=_DEFAULT_FRACTION, kw_only=True)
    """Only compact an expired history above this fraction of the request model's window.

    Defaults to `0.2`, so compacting reclaims enough to be worth the lost content. `None` compacts
    every expired history. Set `max_tokens` instead for an absolute size."""

    context_window: int | None = field(default=None, kw_only=True)
    """Window override in tokens, consulted only alongside `max_fraction`."""

    fallback_context_window: int = field(default=DEFAULT_CONTEXT_WINDOW, kw_only=True)
    """Window assumed when the request model cannot be resolved, alongside `max_fraction`."""

    fallback_retention: timedelta | None = field(default=None, kw_only=True)
    """Retention assumed when neither the request's settings nor the provider document one. `None` disables.

    Set it for a provider that caches without a documented retention, such as Gemini or OpenAI
    models before GPT-5.6. Choose a value at or above the provider's real retention: one that is
    too short compacts while the cache is still warm and costs the cache hit."""

    tokenizer: Callable[[str], int] | None = field(default=None, kw_only=True)
    """Optional token counter for `max_tokens`/`max_fraction` and the fallback's span; defaults to the characters-per-token heuristic."""

    id: str | None = field(default='compact_on_cache_expiry', kw_only=True)
    """One-off: an agent has one expiry policy, so the id is fixed by default.

    Two of them resolve to one, merged field by field, except that a `fallback` or size floor one of them changed
    from the default is kept as a whole, `None` included; see `combine`. The fallback runs as this capability's `compact` durable
    operation, which needs a stable id."""

    def __post_init__(self) -> None:
        if self.max_fraction is _DEFAULT_FRACTION:
            # Left at its default, it gives way to `max_tokens`; stated explicitly beside it, it's a conflict.
            self.max_fraction = None if self.max_tokens is not None else DEFAULT_MAX_FRACTION
        validate_token_trigger(self.max_tokens, self.max_fraction, self.fallback_context_window, self.context_window)
        if self.fallback_retention is not None and self.fallback_retention <= timedelta(0):
            raise ValueError('fallback_retention must be positive.')

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Merge the configuration, with the later stated `fallback` and size floor winning as a whole.

        The field-by-field merge reads `None` as "not stated", but `fallback=None` and `max_fraction=None` are
        opt-outs here, and `max_tokens` and `max_fraction` are one choice of floor. So each of those is taken from
        the last capability that changed it from the default, and a default one beside it doesn't undo it.
        """
        expiries = [capability for capability in capabilities if isinstance(capability, CompactOnCacheExpiry)]
        assert len(expiries) == len(capabilities)
        merged = super().combine(capabilities)
        assert isinstance(merged, cls)
        default_fallback = _default_fallback()
        fallback = next((c.fallback for c in reversed(expiries) if c.fallback != default_fallback), default_fallback)
        max_tokens, max_fraction = next(
            (
                (c.max_tokens, c.max_fraction)
                for c in reversed(expiries)
                if (c.max_tokens, c.max_fraction) != (None, DEFAULT_MAX_FRACTION)
            ),
            (None, DEFAULT_MAX_FRACTION),
        )
        merged = replace_no_init(merged, fallback=fallback, max_tokens=max_tokens, max_fraction=max_fraction)
        merged.__post_init__()
        return merged

    def get_ordering(self) -> CapabilityOrdering:
        # Innermost, so its `compact_history` is the last asked: the fallback only serves a request that no other
        # capability did, wherever this one is listed.
        return CapabilityOrdering(position='innermost')

    async def prepare_model_request(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Request compaction when the cache has expired and the history is above `max_tokens`/`max_fraction`.

        Decided for each attempt, for the model that will serve it.
        """
        messages = request_context.messages
        if isinstance(messages[-1], ModelResponse):
            # Resuming a response the provider paused: it must get the exact history back, however long the pause.
            # Core doesn't compact it either, but a request made now would wait for the next step, when the cache
            # is warm again.
            return request_context
        if get_compaction_reclaim(ctx) > 0:
            # A compaction's own trigger already compacted the history for this request.
            return request_context
        model = request_context.model
        if self._outlook(messages, model, request_context.model_settings, _request_time(messages)) != 'cold':
            return request_context
        floor = resolve_token_trigger(
            self.max_tokens, self.max_fraction, model, self.fallback_context_window, self.context_window
        )
        if floor is not None and (
            estimate_context_tokens(
                messages, self.tokenizer, model_request_parameters=request_context.model_request_parameters
            )
            <= floor
        ):
            return request_context
        ctx.request_compaction(reason='cache_expiry')
        return request_context

    async def compact_history(self, ctx: RunContext[AgentDepsT], request: CompactionRequest) -> CompactionResult | None:
        """Run `fallback` for a `'cache_expiry'` request no other capability served."""
        if request.reason != 'cache_expiry' or self.fallback is None:
            return None
        messages = request.messages
        request_ctx = ctx if request.model is ctx.model else replace(ctx, model=request.model)
        compacted = await compact_with_span(
            request_ctx,
            strategy='CompactOnCacheExpiry',
            messages=messages,
            compact=lambda: self._compact_fallback(request_ctx, messages),
            tokenizer=self.tokenizer,
            attributes={'compaction.idle_seconds': _idle_seconds(messages)},
        )
        if compacted is messages:
            return None
        return _fallback_result(messages, compacted)

    @durable_operation('compact')
    async def _compact_fallback(self, ctx: RunContext[AgentDepsT], messages: list[ModelMessage]) -> list[ModelMessage]:
        """The fallback's compaction of *messages*, as a durable operation of this capability.

        The fallback isn't registered on the agent, so a durable operation of its own (the summary call of
        `SummarizingCompaction`) doesn't bind; this one does, so a replay reuses the recorded history.
        """
        assert self.fallback is not None
        return reinject_pinned(messages, await self.fallback.compact(messages, ctx))

    def _outlook(
        self, messages: Sequence[ModelMessage], model: Model, model_settings: ModelSettings | None, now: datetime
    ) -> PromptCacheOutlook:
        profile: ModelProfile | None
        try:
            profile = model.profile
        except NotImplementedError:
            # `FallbackModel` has no profile of its own; the model that will serve the request isn't known here.
            profile = None
        retention = model.resolve_cache_retention(model_settings)
        if retention is None and (profile is None or profile.get('default_cache_retention') is None):
            retention = self.fallback_retention
        return prompt_cache_outlook(messages, profile=profile, retention=retention, now=now)


def _fallback_result(original: Sequence[ModelMessage], compacted: list[ModelMessage]) -> CompactionResult:
    """The fallback's compacted history as a `CompactionResult`, whose `part` marks where it starts.

    A strategy that summarizes leaves a `CompactionPart` of its own. One that only edits or drops messages, like
    the default clearing, gets one saying what happened, at the start of the post-compaction window so the tools
    and capabilities that window shows stay as they were. The part a previous fallback left there is replaced.
    """
    latest = _latest_compaction_part(compacted)
    original_parts = [part for message in original for part in message.parts if isinstance(part, CompactionPart)]
    if latest is not None and latest[1] not in original_parts:
        return CompactionResult(messages=compacted, part=latest[1], handler='compact_on_cache_expiry')
    part = CompactionPart(content=FALLBACK_NOTE, metadata={_FALLBACK_METADATA_KEY: True})
    if latest is None:
        messages = [ModelRequest(parts=[part]), *compacted]
    else:
        index, previous = latest
        holder = compacted[index]
        if isinstance(holder, ModelRequest) and _is_fallback_part(previous):
            renewed = replace(holder, parts=[part if p is previous else p for p in holder.parts])
            messages = [*compacted[:index], renewed, *compacted[index + 1 :]]
        else:
            messages = [*compacted[: index + 1], ModelRequest(parts=[part]), *compacted[index + 1 :]]
    return CompactionResult(messages=messages, part=part, handler='compact_on_cache_expiry')


def _latest_compaction_part(messages: Sequence[ModelMessage]) -> tuple[int, CompactionPart] | None:
    return next(
        (
            (index, part)
            for index in reversed(range(len(messages)))
            for part in reversed(messages[index].parts)
            if isinstance(part, CompactionPart)
        ),
        None,
    )


def _is_fallback_part(part: CompactionPart) -> bool:
    return part.metadata is not None and part.metadata.get(_FALLBACK_METADATA_KEY) is True


def _request_time(messages: Sequence[ModelMessage]) -> datetime:
    """When the request about to be sent was stamped, so the decision can be read back from the history.

    The run stamps that request before the hooks run. A history built by hand may leave it unstamped.
    """
    last = messages[-1] if messages else None
    if isinstance(last, ModelRequest) and last.timestamp is not None:
        return _aware(last.timestamp)
    return datetime.now(UTC)


def _idle_seconds(messages: Sequence[ModelMessage]) -> float:
    """Seconds from the last model response to the request being sent, which a `'cold'` outlook needed."""
    last = next((message for message in reversed(messages) if isinstance(message, ModelResponse)), None)
    if last is None:
        return 0.0
    return (_request_time(messages) - _aware(last.timestamp)).total_seconds()


def _aware(timestamp: datetime) -> datetime:
    # Historical messages may carry naive timestamps; read them as UTC, as `prompt_cache_outlook` does.
    return timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=UTC)
