"""`CompactOnCacheExpiry` -- compact history when the provider's prompt cache has expired."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse
from pydantic_ai.profiles import ModelProfile, PromptCacheOutlook, prompt_cache_outlook
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness.compaction._clear_tool_results import ClearToolResults
from pydantic_ai_harness.compaction._context_window import DEFAULT_CONTEXT_WINDOW
from pydantic_ai_harness.compaction._pinning import reinject_pinned
from pydantic_ai_harness.compaction._shared import (
    CompactionStrategy,
    SupportsFocus,
    compact_with_span,
    context_for_request,
    estimate_context_tokens,
    estimate_token_count,
    get_compaction_reclaim,
    has_context_usage_anchor,
    record_compaction_reclaim,
    resolve_token_trigger,
    validate_token_trigger,
)

if TYPE_CHECKING:
    from pydantic_ai.models import Model, ModelRequestContext
    from pydantic_ai.settings import ModelSettings


DEFAULT_MIN_FRACTION = 0.2
"""Default size, as a fraction of the context window, below which an expired history is left alone."""


def _default_strategy() -> CompactionStrategy[AgentDepsT]:
    # `max_tokens=1` only satisfies `ClearToolResults`' required trigger, which this capability bypasses.
    return ClearToolResults(max_tokens=1, keep_pairs=5)


@dataclass
class CompactOnCacheExpiry(AbstractCapability[AgentDepsT]):
    """Compact the history on the first request after the provider's prompt cache has expired.

    Rewriting history moves the cached prompt prefix, so the next request pays a cache write for
    everything after the edit. Once the conversation has been idle for longer than the provider
    keeps the prefix, the next request pays that write anyway, so compacting then costs no extra
    cache spend.

    The decision uses [`prompt_cache_outlook`][pydantic_ai.profiles.prompt_cache_outlook]: the gap
    since the last model response, against the retention the request's settings ask for or else the
    provider's documented default. When the outlook is `'unknown'` (no documented retention, a
    `FallbackModel`, or a history without a response) nothing is compacted, unless
    `fallback_retention` is set.

    The default strategy clears all but the 5 most recent tool results, which is what Claude Code
    does after an idle gap longer than its cache retention. By default only a history above a fifth
    of the model's context window is compacted, so a short conversation keeps its tool results.

    Register it after size-based strategies such as `TieredCompaction`: they measure the history
    with the last response's reported usage, which doesn't reflect an edit made earlier in the same
    request.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.capabilities import Caching
        from pydantic_ai_harness.compaction import CompactOnCacheExpiry

        agent = Agent('anthropic:claude-sonnet-5', capabilities=[Caching(), CompactOnCacheExpiry()])
        ```
    """

    strategy: CompactionStrategy[AgentDepsT] = field(default_factory=_default_strategy)
    """The compaction to run once the cache has expired.

    Its own trigger is bypassed: this capability calls its `compact` method directly. Defaults to
    `ClearToolResults(max_tokens=1, keep_pairs=5)`."""

    max_tokens: int | None = field(default=None, kw_only=True)
    """Only compact an expired history whose estimated context tokens exceed this value.

    Replaces the default `max_fraction`."""

    max_fraction: float | None = field(default=DEFAULT_MIN_FRACTION, kw_only=True)
    """Only compact an expired history above this fraction of the request model's window.

    Defaults to `0.2`, so clearing reclaims enough to be worth the lost content. `None` compacts
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
    """Optional token counter for `max_tokens`/`max_fraction`; defaults to the characters-per-token heuristic."""

    def __post_init__(self) -> None:
        if self.max_tokens is not None and self.max_fraction == DEFAULT_MIN_FRACTION:
            self.max_fraction = None
        validate_token_trigger(self.max_tokens, self.max_fraction, self.fallback_context_window, self.context_window)
        if self.fallback_retention is not None and self.fallback_retention <= timedelta(0):
            raise ValueError('fallback_retention must be positive.')

    def with_focus(self, focus: str) -> CompactOnCacheExpiry[AgentDepsT]:
        """Return a copy whose strategy prioritizes `focus`, when it supports one."""
        if not isinstance(self.strategy, SupportsFocus):
            return self
        return replace(self, strategy=self.strategy.with_focus(focus))

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
    ) -> list[ModelMessage]:
        """Run the strategy unconditionally, for manual compaction and composition."""
        return await self.strategy.compact(messages, ctx)

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Compact when the cache has expired and the history is above `max_tokens`/`max_fraction`."""
        model = request_context.model
        messages: list[ModelMessage] = list(ctx.messages)
        if messages and isinstance(messages[-1], ModelResponse):
            # Resuming a response the provider paused: it must get the exact history back, however long the pause.
            return request_context
        now = _request_time(messages)
        if self._outlook(messages, model, request_context.model_settings, now) != 'cold':
            return request_context
        request_ctx = context_for_request(ctx, request_context)
        floor = resolve_token_trigger(
            self.max_tokens, self.max_fraction, model, self.fallback_context_window, self.context_window
        )
        if floor is not None and self._size(ctx, messages, request_context) <= floor:
            return request_context
        compacted = await compact_with_span(
            request_ctx,
            strategy='CompactOnCacheExpiry',
            messages=messages,
            compact=lambda: self._compact_pinned(messages, request_ctx),
            tokenizer=self.tokenizer,
            attributes={'compaction.idle_seconds': _idle_seconds(messages, now)},
        )
        ctx.messages[:] = compacted
        record_compaction_reclaim(
            ctx,
            estimate_token_count(messages, self.tokenizer),
            estimate_token_count(compacted, self.tokenizer),
        )
        return replace(request_context, messages=compacted)

    def _size(
        self, ctx: RunContext[AgentDepsT], messages: list[ModelMessage], request_context: ModelRequestContext
    ) -> int:
        size = estimate_context_tokens(
            messages, self.tokenizer, model_request_parameters=request_context.model_request_parameters
        )
        if has_context_usage_anchor(messages):
            # The anchor is the last response's reported usage, which predates edits an earlier strategy made to
            # this request; subtract what they reclaimed, as `ReportContextUsage` does.
            size = max(size - get_compaction_reclaim(ctx), 0)
        return size

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

    async def _compact_pinned(self, messages: list[ModelMessage], ctx: RunContext[AgentDepsT]) -> list[ModelMessage]:
        return reinject_pinned(messages, await self.compact(messages, ctx))


def _request_time(messages: list[ModelMessage]) -> datetime:
    """When the request about to be sent was stamped, so the decision can be read back from the history.

    The run stamps that request just before the hooks run. A history built by hand may leave it unstamped.
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
