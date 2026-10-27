"""`CompactOnCacheExpiry` -- ask for compaction when the provider's prompt cache has expired."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability, WrapModelRequestHandler, request_compaction
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse
from pydantic_ai.profiles import ModelProfile, PromptCacheOutlook, prompt_cache_outlook
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness.compaction._clear_tool_results import ClearToolResults
from pydantic_ai_harness.compaction._context_window import DEFAULT_CONTEXT_WINDOW
from pydantic_ai_harness.compaction._pinning import reinject_pinned
from pydantic_ai_harness.compaction._shared import (
    CompactionStrategy,
    compact_with_span,
    context_for_request,
    estimate_request_tokens,
    estimate_token_count,
    get_compaction_reclaim,
    record_compaction_reclaim,
    resolve_token_trigger,
    validate_token_trigger,
)

if TYPE_CHECKING:
    from pydantic_ai.models import Model, ModelRequestContext
    from pydantic_ai.settings import ModelSettings


DEFAULT_MIN_FRACTION = 0.2
"""Default size, as a fraction of the context window, below which an expired history is left alone."""

CLEARED_TOOL_RESULT_HINT = '[Tool result cleared to save context. Call the tool again if you need it.]'
"""Placeholder the default fallback leaves in place of a cleared tool result.

It tells the model the result can be fetched again: with a bare placeholder, models answered from
what they guessed the cleared result said rather than calling the tool again."""


_requested_for: ContextVar[tuple[str | None, int] | None] = ContextVar('_cache_expiry_requested_for', default=None)
"""The run and step whose expired request a `CompactOnCacheExpiry` already acted on, so a second one doesn't."""


def _default_fallback() -> CompactionStrategy[AgentDepsT]:
    # `max_tokens=1` only satisfies `ClearToolResults`' required trigger, which a fallback bypasses.
    return ClearToolResults(max_tokens=1, keep_pairs=5, placeholder=CLEARED_TOOL_RESULT_HINT)


@dataclass
class CompactOnCacheExpiry(AbstractCapability[AgentDepsT]):
    """Ask for compaction on the first request after the provider's prompt cache has expired.

    Rewriting history moves the cached prompt prefix, so the next request pays a cache write for
    everything after the edit. Once the conversation has been idle for longer than the provider
    keeps the prefix, the next request pays that write anyway, so compacting then costs no extra
    cache spend.

    It doesn't compact itself: it emits a
    [`CompactionRequestedEvent`][pydantic_ai.capabilities.CompactionRequestedEvent], and
    the agent's compaction capabilities -- `SummarizingCompaction`, `TieredCompaction`,
    `ClearToolResults`, `SlidingWindowCompaction`, `FallbackCompaction`, and `OpenAICompaction` --
    compact the history their own way. The first of them in the capabilities list to change the
    history handles the request. When none does, `fallback` runs.

    The decision uses [`prompt_cache_outlook`][pydantic_ai.profiles.prompt_cache_outlook]: the gap
    since the last model response, against the retention the request's settings ask for or else the
    provider's documented default. When the outlook is `'unknown'` (no documented retention, a
    `FallbackModel`, or a history without a response) nothing is compacted, unless
    `fallback_retention` is set.

    By default only a history above a fifth of the model's context window is compacted, so a short
    conversation keeps its tool results.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.capabilities import Caching
        from pydantic_ai_harness.compaction import CompactOnCacheExpiry, SummarizingCompaction

        agent = Agent(
            'anthropic:claude-sonnet-5',
            capabilities=[Caching(), SummarizingCompaction(max_fraction=0.8), CompactOnCacheExpiry()],
        )
        ```
    """

    fallback: CompactionStrategy[AgentDepsT] | None = field(default_factory=_default_fallback)
    """The compaction to run when no compaction capability handled the request. `None` leaves the history alone.

    Its own trigger is bypassed: its `compact` method is called directly. Defaults to clearing all but
    the 5 most recent tool results, which is what Claude Code does after an idle gap longer than its
    cache retention, with a placeholder that tells the model it can call the tool again."""

    max_tokens: int | None = field(default=None, kw_only=True)
    """Only compact an expired history whose estimated context tokens exceed this value.

    Replaces the default `max_fraction`."""

    max_fraction: float | None = field(default=DEFAULT_MIN_FRACTION, kw_only=True)
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
    """Optional token counter for `max_tokens`/`max_fraction` and the fallback; defaults to the characters-per-token heuristic."""

    def __post_init__(self) -> None:
        if self.max_tokens is not None and self.max_fraction == DEFAULT_MIN_FRACTION:
            self.max_fraction = None
        validate_token_trigger(self.max_tokens, self.max_fraction, self.fallback_context_window, self.context_window)
        if self.fallback_retention is not None and self.fallback_retention <= timedelta(0):
            raise ValueError('fallback_retention must be positive.')

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Request compaction when the cache has expired and the history is above `max_tokens`/`max_fraction`.

        Runs here rather than in `before_model_request` so the request comes before every compaction
        capability's own `before_model_request`, wherever this capability is listed.
        """
        return await handler(await self._compact_if_expired(ctx, request_context))

    async def _compact_if_expired(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        model = request_context.model
        messages: list[ModelMessage] = list(request_context.messages)
        if messages and isinstance(messages[-1], ModelResponse):
            # Resuming a response the provider paused: it must get the exact history back, however long the pause.
            return request_context
        now = _request_time(messages)
        if self._outlook(messages, model, request_context.model_settings, now) != 'cold':
            return request_context
        step = (ctx.run_id, ctx.run_step)
        if _requested_for.get() == step:
            return request_context
        request_ctx = context_for_request(ctx, request_context)
        floor = resolve_token_trigger(
            self.max_tokens, self.max_fraction, model, self.fallback_context_window, self.context_window
        )
        if floor is not None and (
            estimate_request_tokens(
                messages,
                self.tokenizer,
                model_request_parameters=request_context.model_request_parameters,
                reclaimed=get_compaction_reclaim(ctx),
            )
            <= floor
        ):
            return request_context
        _requested_for.set(step)
        request_context = replace(request_context, messages=messages)
        event = await request_compaction(request_ctx, request_context, reason='cache_expired')
        if event.compacted_by is not None or self.fallback is None:
            return request_context
        fallback = self.fallback
        compacted = await compact_with_span(
            request_ctx,
            strategy='CompactOnCacheExpiry',
            messages=messages,
            compact=lambda: _compact_pinned(fallback, messages, request_ctx),
            tokenizer=self.tokenizer,
            attributes={'compaction.idle_seconds': _idle_seconds(messages, now), 'compaction.reason': event.reason},
        )
        if compacted is messages:
            return request_context
        ctx.messages[:] = compacted
        record_compaction_reclaim(
            ctx,
            estimate_token_count(messages, self.tokenizer),
            estimate_token_count(compacted, self.tokenizer),
        )
        return replace(request_context, messages=compacted)

    def _outlook(
        self, messages: list[ModelMessage], model: Model, model_settings: ModelSettings | None, now: datetime
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


async def _compact_pinned(
    strategy: CompactionStrategy[AgentDepsT], messages: list[ModelMessage], ctx: RunContext[AgentDepsT]
) -> list[ModelMessage]:
    return reinject_pinned(messages, await strategy.compact(messages, ctx))


def _request_time(messages: list[ModelMessage]) -> datetime:
    """When the request about to be sent was stamped, so the decision can be read back from the history.

    The run stamps that request before the hooks run. A history built by hand may leave it unstamped.
    """
    last = messages[-1] if messages else None
    if isinstance(last, ModelRequest) and last.timestamp is not None:
        return last.timestamp if last.timestamp.tzinfo is not None else last.timestamp.replace(tzinfo=UTC)
    return datetime.now(UTC)


def _idle_seconds(messages: list[ModelMessage], now: datetime) -> float:
    """Seconds since the last model response, which a `'cold'` outlook guarantees exists."""
    last = next(message for message in reversed(messages) if isinstance(message, ModelResponse))
    timestamp = last.timestamp if last.timestamp.tzinfo is not None else last.timestamp.replace(tzinfo=UTC)
    return (now - timestamp).total_seconds()
