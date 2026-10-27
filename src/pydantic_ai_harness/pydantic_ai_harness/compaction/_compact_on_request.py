"""`CompactOnRequest` -- let the model ask for a compaction at a natural break in its task."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from pydantic_ai import ToolKind
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, durable_operation
from pydantic_ai.messages import ModelMessage, ToolCallPart
from pydantic_ai.tools import AgentDepsT, RunContext, Tool, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.compaction._clear_tool_results import ClearToolResults
from pydantic_ai_harness.compaction._deduplicate_file_reads import DeduplicateFileReads
from pydantic_ai_harness.compaction._fallback_compaction import FallbackCompaction
from pydantic_ai_harness.compaction._pinning import reinject_pinned
from pydantic_ai_harness.compaction._relevance_compaction import RelevanceCompaction
from pydantic_ai_harness.compaction._shared import (
    CompactionStrategy,
    SupportsFocus,
    compact_with_span,
    context_for_request,
    estimate_context_tokens,
    estimate_token_count,
    get_compaction_reclaim,
    record_compaction_reclaim,
)
from pydantic_ai_harness.compaction._sliding_window_compaction import SlidingWindowCompaction
from pydantic_ai_harness.compaction._summarizing_compaction import SummarizingCompaction
from pydantic_ai_harness.compaction._tiered_compaction import TieredCompaction

if TYPE_CHECKING:
    from pydantic_ai.capabilities import ValidatedToolArgs
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.toolsets import AgentToolset

_COMPACT_CONTEXT_KIND = ToolKind('compaction', 'compact_context')
"""The kind of the `compact_context` tool, so the capability recognizes it under any name."""


async def _compact_context(focus: str | None = None) -> str:
    """Compact the conversation history before your next turn, to free context for the work ahead.

    Call this when you have finished a phase of work, such as exploring a codebase, and the detail
    gathered so far is no longer needed verbatim. Older messages are condensed or dropped; the most
    recent ones are kept. Do not call it in the middle of a sub-task whose details you still need.

    Args:
        focus: What must survive the compaction, for example the plan and the paths you will edit next.
    """
    del focus
    return 'Compaction scheduled: older history will be compacted before your next turn.'


@dataclass(frozen=True)
class _Request:
    focus: str | None


@dataclass
class CompactOnRequest(AbstractCapability[AgentDepsT]):
    """Give the model a `compact_context` tool to ask for a compaction when its task reaches a break.

    Size-based triggers do not know where the task is: they can fire in the middle of a sub-task,
    or not fire right after a phase whose output is no longer needed. The model often knows that
    moment. This capability lets it say so, and runs `strategy` before the next model request of
    the same run.

    The tool compacts nothing itself. It records the request, and the compaction runs in the next
    `before_model_request`, so the tool call and its return stay paired in the kept tail and the
    model can see that it asked. A `focus` the model passes reaches a strategy that implements
    `SupportsFocus`, and is ignored by one that does not.

    It runs after the automatic compaction capabilities on the same agent, wherever it is listed.
    When one of those has already compacted the history for the request, the model's request is
    dropped rather than compacting twice in one step, and does not count against `max_per_run`.

    Under durable execution the strategy runs as this capability's `compact` operation. Under
    Temporal that is an activity, whose context omits `tracer` and `capabilities`, so a strategy
    that reads them (`RelevanceCompaction`, or `receipts=True` on a summarizing or sliding-window
    strategy) fails there.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_harness.compaction import CompactOnRequest, SummarizingCompaction, TieredCompaction

        agent = Agent(
            'openai:gpt-5.6-luna',
            capabilities=[
                TieredCompaction(tiers=[SummarizingCompaction(max_messages=1)], target_fraction=0.8),
                # `max_messages` is required by the strategy, but `CompactOnRequest` bypasses it.
                CompactOnRequest(
                    SummarizingCompaction(max_messages=1, keep_messages=6),
                    min_tokens=30_000,
                    max_per_run=2,
                ),
            ],
        )
        ```
    """

    strategy: CompactionStrategy[AgentDepsT]
    """The strategy to run when the model asks. Its `compact` is called directly, so its own trigger is bypassed."""

    min_tokens: int | None = field(default=None, kw_only=True)
    """Refuse a request while the history is estimated at fewer tokens than this. `None` disables the check."""

    max_per_run: int | None = field(default=None, kw_only=True)
    """Refuse requests once this many have been accepted in the run. `None` disables the check."""

    tokenizer: Callable[[str], int] | None = field(default=None, kw_only=True)
    """Optional token counter for `min_tokens` and the span; defaults to the characters-per-token heuristic."""

    id: str | None = field(default='compact_on_request', kw_only=True)

    _pending: _Request | None = field(default=None, init=False, repr=False, compare=False)
    _accepted: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.min_tokens is not None and self.min_tokens < 1:
            raise ValueError('min_tokens must be positive.')
        if self.max_per_run is not None and self.max_per_run < 1:
            raise ValueError('max_per_run must be positive.')

    def get_ordering(self) -> CapabilityOrdering:
        # Inside the automatic compactions, so it sees whether one of them already compacted this
        # request. Run first, it would leave them a history whose latest usage still measures the
        # pre-compaction request, and they would compact a second time.
        return CapabilityOrdering(
            wrapped_by=[
                ClearToolResults,
                DeduplicateFileReads,
                SlidingWindowCompaction,
                SummarizingCompaction,
                RelevanceCompaction,
                TieredCompaction,
                FallbackCompaction,
            ]
        )

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> CompactOnRequest[AgentDepsT]:
        """Return a copy with no pending request and no accepted requests counted."""
        return replace(self)

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Register the `compact_context` tool."""
        tool = Tool(_compact_context, name='compact_context', tool_kind=_COMPACT_CONTEXT_KIND)
        return FunctionToolset([tool], id=self.id or 'compact_on_request')

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: Any,
    ) -> Any:
        """Accept or refuse a `compact_context` call, answering with the reason when refused.

        The decision is made here rather than in the tool function: under durable execution the tool
        can run in an activity that sees neither the message history nor this run's state, while
        hooks run with the run itself.
        """
        if tool_def.tool_kind != _COMPACT_CONTEXT_KIND.id:
            return result
        if self.max_per_run is not None and self._accepted >= self.max_per_run:
            return f'Not compacted: the limit of {self.max_per_run} compaction requests for this run has been reached.'
        if self._pending is not None:
            return 'Not compacted again: a compaction is already scheduled before your next turn.'
        if self.min_tokens is not None:
            tokens = estimate_context_tokens(ctx.messages, self.tokenizer)
            if tokens < self.min_tokens:
                return (
                    f'Not compacted: the history is about {tokens} tokens, below the minimum of '
                    f'{self.min_tokens}, so compacting would save little.'
                )
        self._pending = _Request(args.get('focus') or None)
        self._accepted += 1
        return result

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Run the strategy when the model asked for a compaction in the previous step."""
        request = self._pending
        if request is None:
            return request_context
        self._pending = None
        if get_compaction_reclaim(ctx) > 0:
            # A capability earlier in the chain already compacted the history for this request. The
            # request did not run, so it does not count against `max_per_run`.
            self._accepted -= 1
            return request_context
        messages: list[ModelMessage] = list(ctx.messages)
        request_ctx = context_for_request(ctx, request_context)
        compacted = await compact_with_span(
            request_ctx,
            strategy=type(self.strategy).__name__,
            messages=messages,
            compact=lambda: self._compact(request_ctx, messages, request.focus),
            tokenizer=self.tokenizer,
            model_requested=True,
        )
        ctx.messages[:] = compacted
        record_compaction_reclaim(
            ctx,
            estimate_token_count(messages, self.tokenizer),
            estimate_token_count(compacted, self.tokenizer),
        )
        return replace(request_context, messages=compacted)

    @durable_operation('compact')
    async def _compact(
        self, ctx: RunContext[AgentDepsT], messages: list[ModelMessage], focus: str | None
    ) -> list[ModelMessage]:
        """The strategy's compaction of *messages*, as a durable operation of this capability.

        The strategy is not registered on the agent, so a durable operation of its own (the summary
        call of `SummarizingCompaction`) does not bind; this one does, and the strategy runs inline
        within it, so a replay reuses the recorded history instead of calling a model again.
        """
        strategy = self.strategy
        if focus is not None and isinstance(strategy, SupportsFocus):
            strategy = strategy.with_focus(focus)
        return reinject_pinned(messages, await strategy.compact(messages, ctx))
