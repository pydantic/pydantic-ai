"""`ReportContextUsage` -- report how full the context is, for a host application."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness.compaction._context_window import DEFAULT_CONTEXT_WINDOW, resolve_context_window
from pydantic_ai_harness.compaction._events import ContextUsageMeasuredEvent
from pydantic_ai_harness.compaction._shared import (
    estimate_context_tokens,
    get_compaction_reclaim,
    has_context_usage_anchor,
    reset_compaction_reclaim,
)

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext


@dataclass(frozen=True)
class ContextUsage:
    """A single reading of how full the context is."""

    used_tokens: int
    """Estimated tokens in the message history about to be sent.

    Counted by `estimate_context_tokens`: the provider-reported usage of the most recent model
    response (tool schemas included) plus an estimate of messages and newly revealed tool schemas
    added since. With no reported usage, message text falls back to the character heuristic;
    schemas named by availability deltas are still estimated from the pending request
    parameters, but other tool schemas remain outside that heuristic.
    """

    window_tokens: int
    """Context window the reading is measured against."""

    resolved: bool
    """Whether `window_tokens` is the model's real window or the fallback.

    A gauge can render an unresolved window differently -- the percentage is a guess when
    the model is not in the pricing registry.
    """

    @property
    def fraction(self) -> float:
        """`used_tokens` as a fraction of the window."""
        return self.used_tokens / self.window_tokens


@dataclass
class ReportContextUsage(AbstractCapability[AgentDepsT]):
    """Report context usage to the application before each model request.

    A compaction strategy knows when to act but says nothing about how close the run is to
    the limit, so an application that wants to show "context: 73%" has to re-count the
    history itself and guess the denominator. This capability does neither: it reuses the
    same estimator the strategies use and the model's real context window.

    It only observes -- it never edits the history.

    Order matters: register it *after* a compaction capability to see the compacted history,
    or before it to see what triggered the compaction. After a preceding compactor rewrites
    anchored history, the reading subtracts that compactor's heuristic reclaim while retaining
    the anchor's fixed provider overhead.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_harness.compaction import ReportContextUsage, SummarizingCompaction

        agent = Agent(
            'anthropic:claude-sonnet-4-6',
            capabilities=[
                SummarizingCompaction(max_fraction=0.9, keep_messages=20),
                ReportContextUsage(),
            ],
        )
        ```
    """

    # Keyword-only since the positional `on_usage` was removed, so an old positional call fails instead of
    # dropping its window override.
    _: KW_ONLY

    context_window: int | None = None
    """Window override in tokens. `None` resolves it from the request's model."""

    fallback_context_window: int = DEFAULT_CONTEXT_WINDOW
    """Window assumed when the request's model is not in the pricing registry."""

    tokenizer: Callable[[str], int] | None = None
    """Optional tokenizer, matching the one your compaction strategy uses."""

    def __post_init__(self) -> None:
        if self.context_window is not None and self.context_window < 1:
            raise ValueError('context_window must be positive.')
        if self.fallback_context_window < 1:
            raise ValueError('fallback_context_window must be positive.')

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Reset the request correction even when the caller reuses a run ID."""
        reset_compaction_reclaim()

    def _measure(self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext) -> ContextUsage:
        """Build a reading for the request as it stands."""
        messages: list[ModelMessage] = list(request_context.messages)
        used = estimate_context_tokens(
            messages,
            self.tokenizer,
            model_request_parameters=request_context.model_request_parameters,
        )
        if has_context_usage_anchor(messages):
            used = max(used - get_compaction_reclaim(ctx), 0)
        if self.context_window is not None:
            return ContextUsage(used_tokens=used, window_tokens=self.context_window, resolved=True)
        # Resolved from the request's model rather than the run's: a capability may replace
        # `ModelRequestContext.model`, and the gauge should track where the request goes.
        window = resolve_context_window(request_context.model)
        return ContextUsage(
            used_tokens=used,
            window_tokens=window if window is not None else self.fallback_context_window,
            resolved=window is not None,
        )

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Measure the pending history and emit it."""
        reading = self._measure(ctx, request_context)
        await ctx.emit(
            ContextUsageMeasuredEvent(
                used_tokens=reading.used_tokens,
                window_tokens=reading.window_tokens,
                resolved=reading.resolved,
                fraction=reading.fraction,
            )
        )
        return request_context
