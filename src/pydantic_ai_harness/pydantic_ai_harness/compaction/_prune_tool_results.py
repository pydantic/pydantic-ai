"""`PruneToolResults` -- let the model replace a tool result it just received with its own summary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import islice
from typing import TYPE_CHECKING, Any

from pydantic_ai import ToolKind
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.tools import AgentDepsT, RunContext, Tool, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.compaction._clamp_oversized_messages import ClampOversizedMessages
from pydantic_ai_harness.compaction._clear_tool_results import ClearToolResults
from pydantic_ai_harness.compaction._compact_on_request import CompactOnRequest
from pydantic_ai_harness.compaction._deduplicate_file_reads import DeduplicateFileReads
from pydantic_ai_harness.compaction._fallback_compaction import FallbackCompaction
from pydantic_ai_harness.compaction._relevance_compaction import RelevanceCompaction
from pydantic_ai_harness.compaction._shared import (
    cleared_return,
    compact_with_span,
    context_for_request,
)
from pydantic_ai_harness.compaction._sliding_window_compaction import SlidingWindowCompaction
from pydantic_ai_harness.compaction._summarizing_compaction import SummarizingCompaction
from pydantic_ai_harness.compaction._tiered_compaction import TieredCompaction

if TYPE_CHECKING:
    from pydantic_ai.capabilities import ValidatedToolArgs
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.toolsets import AgentToolset

_PRUNE_TOOL_RESULT_KIND = ToolKind('compaction', 'prune_tool_result')
"""The kind of the `prune_tool_result` tool, so the capability recognizes it under any name."""


async def _prune_tool_result(tool_call_id: str, summary: str) -> str:
    """Replace a tool result from your previous step with a short summary, so later requests do not resend it.

    Call this when a tool returned output you do not need, or far more than you need, such as a search
    that matched build artifacts or a log where one line mattered. The original is not shown to you
    again, so put anything you still need from it in `summary`. Only results from your previous step
    can be pruned.

    Args:
        tool_call_id: The ID of the tool call whose result to replace.
        summary: What the result contained and anything from it you still need, in a line or two.
    """
    del tool_call_id, summary
    return 'Pruned: the result will be replaced with your summary before your next turn.'


@dataclass
class PruneToolResults(AbstractCapability[AgentDepsT]):
    """Give the model a `prune_tool_result` tool to replace a tool result it just received with a summary.

    Rule-based clearing keeps the newest results, which is where an unexpectedly large or irrelevant
    result sits right after it arrives. Whether a result is useless is a judgement the model can make
    once it has read it. This capability lets it make that judgement, and replaces the result's content
    with `placeholder` followed by the model's summary before the next model request of the same run.

    Only results from the step the model just answered can be pruned, so the edit lands at the end of
    the history and the cached prefix before it is kept. The tool call stays paired with its result,
    and the replacement persists in the run's message history, as `ClearToolResults` does.

    It runs before the automatic compaction capabilities on the same agent, wherever it is listed, so
    a compaction in the same request works on the pruned history. Their token triggers anchor on the
    usage of the latest response, which still counts the pruned result, so a prune lowers those
    estimates from the following request on.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_harness.compaction import ClearToolResults, PruneToolResults

        agent = Agent(
            'openai:gpt-5.6-luna',
            capabilities=[ClearToolResults(max_fraction=0.7), PruneToolResults()],
        )
        ```
    """

    placeholder: str = '[tool result pruned]'
    """Prefix of the content that replaces a pruned result; the model's summary follows it."""

    id: str | None = field(default='prune_tool_results', kw_only=True)

    _pending: dict[str, str] = field(default_factory=dict[str, str], init=False, repr=False, compare=False)

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(
            wraps=[
                ClampOversizedMessages,
                ClearToolResults,
                DeduplicateFileReads,
                SlidingWindowCompaction,
                SummarizingCompaction,
                RelevanceCompaction,
                TieredCompaction,
                FallbackCompaction,
                CompactOnRequest,
            ]
        )

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> PruneToolResults[AgentDepsT]:
        """Return a copy with no pending prunes."""
        return replace(self)

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Register the `prune_tool_result` tool."""
        tool = Tool(_prune_tool_result, name='prune_tool_result', tool_kind=_PRUNE_TOOL_RESULT_KIND)
        return FunctionToolset([tool], id=self.id or 'prune_tool_results')

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: Any,
    ) -> Any:
        """Accept or refuse a `prune_tool_result` call, answering with the reason when refused.

        The decision is made here rather than in the tool function: under durable execution the tool
        can run in an activity that sees neither the message history nor this run's state.
        """
        if tool_def.tool_kind != _PRUNE_TOOL_RESULT_KIND.id:
            return result
        tool_call_id: str = args['tool_call_id']
        content = f'{self.placeholder} {args["summary"]}'
        target = next((part for part in _latest_returns(ctx.messages) if part.tool_call_id == tool_call_id), None)
        if target is None or cleared_return(target, content) is None:
            return f'Not pruned: {tool_call_id!r} is not a result from your previous step that can be pruned.'
        if len(content) >= len(str(target.content)):
            return 'Not pruned: the summary is not shorter than the result.'
        self._pending[tool_call_id] = content
        return result

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Replace the results the model pruned in the previous step with its summaries."""
        pending, self._pending = self._pending, {}
        if not pending:
            return request_context
        messages: list[ModelMessage] = list(ctx.messages)

        async def prune() -> list[ModelMessage]:
            return [_pruned(message, pending) if isinstance(message, ModelRequest) else message for message in messages]

        pruned = await compact_with_span(
            context_for_request(ctx, request_context),
            strategy='PruneToolResults',
            messages=messages,
            compact=prune,
            model_requested=True,
        )
        ctx.messages[:] = pruned
        return replace(request_context, messages=pruned)


def _latest_returns(messages: Sequence[ModelMessage]) -> list[ToolReturnPart]:
    """The tool results the latest model response, the last message, answered."""
    returns: list[ToolReturnPart] = []
    for message in islice(reversed(messages), 1, None):
        if isinstance(message, ModelResponse):
            break
        returns.extend(part for part in message.parts if isinstance(part, ToolReturnPart))
    return returns


def _pruned(message: ModelRequest, contents: Mapping[str, str]) -> ModelRequest:
    """*message* with the results in *contents* replaced, or *message* itself when it holds none of them."""
    if not any(isinstance(part, ToolReturnPart) and part.tool_call_id in contents for part in message.parts):
        return message
    return replace(
        message,
        parts=[
            (cleared_return(part, contents[part.tool_call_id]) or part)
            if isinstance(part, ToolReturnPart) and part.tool_call_id in contents
            else part
            for part in message.parts
        ],
    )
