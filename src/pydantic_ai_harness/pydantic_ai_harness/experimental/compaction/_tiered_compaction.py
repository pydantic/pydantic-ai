"""`TieredCompaction` -- escalation orchestrator over a sequence of strategies."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.experimental.compaction._shared import (
    CompactionStrategy,
    compact_with_span,
    estimate_token_count,
)

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext


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
        from pydantic_ai_harness.experimental.compaction import (
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

    target_tokens: int
    """Stop escalating once the estimated token count is at or below this value."""

    tokenizer: Callable[[str], int] | None = None
    """Optional tokenizer for accurate token counting.

    A callable that returns the token count for a given string.
    When ``None``, uses a ~4 characters-per-token heuristic.
    """

    def __post_init__(self) -> None:
        if not self.tiers:
            raise ValueError('tiers must not be empty.')
        if self.target_tokens < 1:
            raise ValueError('target_tokens must be positive.')

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
    ) -> list[ModelMessage]:
        """Apply tiers in order until the history fits ``target_tokens`` or tiers run out."""
        for tier in self.tiers:
            if estimate_token_count(messages, self.tokenizer) <= self.target_tokens:
                break
            messages = await tier.compact(messages, ctx)
        return messages

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Escalate through the tiers when the conversation exceeds ``target_tokens``."""
        messages: list[ModelMessage] = list(request_context.messages)
        if estimate_token_count(messages, self.tokenizer) <= self.target_tokens:
            return request_context
        request_context.messages = await compact_with_span(
            ctx,
            strategy='TieredCompaction',
            messages=messages,
            compact=lambda: self.compact(messages, ctx),
            tokenizer=self.tokenizer,
        )
        return request_context
