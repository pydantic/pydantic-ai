"""`TieredCompaction` -- escalation orchestrator over a sequence of strategies."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.compaction._context_window import DEFAULT_CONTEXT_WINDOW
from pydantic_ai_harness.compaction._pinning import reinject_pinned
from pydantic_ai_harness.compaction._shared import (
    CompactionStrategy,
    SupportsFocus,
    compact_with_span,
    context_for_request,
    estimate_token_count,
    resolve_token_trigger,
    validate_token_trigger,
)

if TYPE_CHECKING:
    from pydantic_ai.models import Model, ModelRequestContext


@dataclass
class TieredCompaction(AbstractCapability[AgentDepsT]):
    """Escalation orchestrator over a sequence of compaction strategies.

    Runs each tier in order, re-measuring the token count after each, and stops as soon as
    the conversation fits ``target_tokens``.  Order tiers cheap-to-expensive (e.g. clear
    tool results, deduplicate reads, then summarize) so the expensive summarization tier is
    only reached when the cheap passes cannot reclaim enough.

    Each tier's own trigger is bypassed -- `TieredCompaction` drives the tiers directly via
    their ``compact`` method and decides when to stop.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_harness.compaction import (
            ClearToolResults,
            SummarizingCompaction,
            TieredCompaction,
        )

        agent = Agent(
            'openai:gpt-4o',
            capabilities=[TieredCompaction(
                tiers=[
                    ClearToolResults(max_tokens=1),
                    SummarizingCompaction(model='openai:gpt-4o-mini', max_messages=1),
                ],
                target_tokens=100_000,
            )],
        )
        ```
    """

    tiers: Sequence[CompactionStrategy[AgentDepsT]]
    """Strategies to apply in order, cheap-to-expensive.  The last is typically a summarizer."""

    target_tokens: int | None = None
    """Stop escalating once the estimated token count is at or below this value.

    Mutually exclusive with `target_fraction`; exactly one of the two must be set.
    """

    target_fraction: float | None = field(default=None, kw_only=True)
    """Target expressed as a fraction of the model's context window, resolved per request.

    Use this instead of `target_tokens` when the same agent runs on models with
    different windows. Mutually exclusive with `target_tokens`.
    """

    context_window: int | None = field(default=None, kw_only=True)
    """Window override in tokens. `None` resolves it from the request's model.

    Unlike `fallback_context_window`, this applies whether or not resolution succeeds. Reach
    for it when the registry is confidently wrong: a beta- or tier-gated window it records as
    the maximum, or a self-hosted endpoint whose model id describes someone else's
    deployment. Only consulted alongside `target_fraction`."""

    fallback_context_window: int = field(default=DEFAULT_CONTEXT_WINDOW, kw_only=True)
    """Window assumed when the model is not in the pricing registry.

    Only consulted alongside `target_fraction`. Supply the real number for a deployment the
    registry cannot resolve."""

    tokenizer: Callable[[str], int] | None = None
    """Optional tokenizer for accurate token counting.

    A callable that returns the token count for a given string.
    When ``None``, uses a ~4 characters-per-token heuristic.
    """

    def __post_init__(self) -> None:
        if not self.tiers:
            raise ValueError('tiers must not be empty.')
        if self.target_tokens is None and self.target_fraction is None:
            raise ValueError('One of target_tokens or target_fraction must be set.')
        validate_token_trigger(
            self.target_tokens,
            self.target_fraction,
            self.fallback_context_window,
            self.context_window,
            tokens_name='target_tokens',
            fraction_name='target_fraction',
        )

    def with_focus(self, focus: str) -> TieredCompaction[AgentDepsT]:
        """Return a copy whose focus-capable tiers prioritize `focus`.

        A tiered strategy is focusable when any of its tiers is: the summarizing tier writes
        the prose, so the hint has to reach it rather than stopping at this wrapper. Tiers that
        cannot honour a focus are passed through unchanged.
        """
        return replace(
            self,
            tiers=[tier.with_focus(focus) if isinstance(tier, SupportsFocus) else tier for tier in self.tiers],
        )

    def _target(self, model: Model | str) -> int:
        """Absolute token target, resolved against *model* when expressed as a fraction."""
        target = resolve_token_trigger(
            self.target_tokens, self.target_fraction, model, self.fallback_context_window, self.context_window
        )
        if target is None:  # pragma: no cover -- __post_init__ rejects both being unset
            raise ValueError('One of target_tokens or target_fraction must be set.')
        return target

    async def _escalate(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
        target: int,
    ) -> list[ModelMessage]:
        """Apply tiers in order until the history fits *target* or tiers run out."""
        original = messages
        for tier in self.tiers:
            if estimate_token_count(messages, self.tokenizer) <= target:
                break
            messages = await tier.compact(messages, ctx)
            # Before the next stop decision, so escalation measures the history it would return.
            messages = reinject_pinned(original, messages)
        return messages

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
    ) -> list[ModelMessage]:
        """Apply tiers in order until the history fits the target or tiers run out."""
        return await self._escalate(messages, ctx, self._target(ctx.model))

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Escalate through the tiers when the conversation exceeds the target."""
        messages: list[ModelMessage] = list(request_context.messages)
        # The tiers get the request's context, not the run's: a tier that resolves a model --
        # a summarizing one, or a `TieredCompaction` nested inside this one -- has to reach the
        # same conclusion this gate did.
        request_ctx = context_for_request(ctx, request_context)
        # Resolved once, so the gate and the escalation loop cannot disagree about the target.
        target = self._target(request_ctx.model)
        if estimate_token_count(messages, self.tokenizer) <= target:
            return request_context
        request_context.messages = await compact_with_span(
            request_ctx,
            strategy='TieredCompaction',
            messages=messages,
            compact=lambda: self._escalate(messages, request_ctx, target),
            tokenizer=self.tokenizer,
        )
        return request_context
