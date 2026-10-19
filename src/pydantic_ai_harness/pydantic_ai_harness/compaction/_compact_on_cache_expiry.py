"""`CompactOnCacheExpiry` -- compact history when the provider's prompt cache has expired."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse
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
    record_compaction_reclaim,
    resolve_token_trigger,
    validate_token_trigger,
)

if TYPE_CHECKING:
    from pydantic_ai.models import Model, ModelRequestContext
    from pydantic_ai.settings import ModelSettings


def _default_strategy() -> CompactionStrategy[AgentDepsT]:
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
    does after an idle gap longer than its cache retention.

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
    """Only compact an expired history whose estimated context tokens exceed this value. `None` disables."""

    max_fraction: float | None = field(default=None, kw_only=True)
    """Only compact above this fraction of the request model's window, exclusive with `max_tokens`."""

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
        """Compact when the cache has expired and the history is above the optional size floor."""
        model = request_context.model
        messages: list[ModelMessage] = list(ctx.messages)
        if messages and isinstance(messages[-1], ModelResponse):
            # Resuming a response the provider paused: it must get the exact history back, however long the pause.
            return request_context
        if self._outlook(messages, model, request_context.model_settings) != 'cold':
            return request_context
        request_ctx = context_for_request(ctx, request_context)
        floor = resolve_token_trigger(
            self.max_tokens, self.max_fraction, model, self.fallback_context_window, self.context_window
        )
        if (
            floor is not None
            and estimate_context_tokens(
                messages, self.tokenizer, model_request_parameters=request_context.model_request_parameters
            )
            <= floor
        ):
            return request_context
        compacted = await compact_with_span(
            request_ctx,
            strategy='CompactOnCacheExpiry',
            messages=messages,
            compact=lambda: self._compact_pinned(messages, request_ctx),
            tokenizer=self.tokenizer,
        )
        ctx.messages[:] = compacted
        record_compaction_reclaim(
            ctx,
            estimate_token_count(messages, self.tokenizer),
            estimate_token_count(compacted, self.tokenizer),
        )
        return replace(request_context, messages=compacted)

    def _outlook(
        self, messages: list[ModelMessage], model: Model, model_settings: ModelSettings | None
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
        return prompt_cache_outlook(messages, profile=profile, retention=retention)

    async def _compact_pinned(self, messages: list[ModelMessage], ctx: RunContext[AgentDepsT]) -> list[ModelMessage]:
        return reinject_pinned(messages, await self.compact(messages, ctx))
