"""`PruneToolResults` -- let the model replace a tool result it just received with its own summary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai import ToolKind
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.tools import AgentDepsT, RunContext, Tool, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.compaction._shared import cleared_return, compact_with_span, replace_returns

if TYPE_CHECKING:
    from pydantic_ai.capabilities import ValidatedToolArgs
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
    return 'Pruned: the result was replaced with your summary.'


@dataclass
class PruneToolResults(AbstractCapability[AgentDepsT]):
    """Give the model a `prune_tool_result` tool to replace a tool result it just received with a summary.

    Rule-based clearing keeps the newest results, which is where an unexpectedly large or irrelevant
    result sits right after it arrives. Whether a result is useless is a judgement the model can make
    once it has read it. This capability lets it make that judgement, and replaces the result's content
    with `placeholder` followed by the model's summary as soon as the call is accepted.

    Only results from the step the model just answered can be pruned, so the edit lands at the end of
    the history and the cached prefix before it is kept. The tool call stays paired with its result,
    and the replacement persists in the run's message history, as `ClearToolResults` does.

    Token-based compaction triggers and `ReportContextUsage` anchor on the usage of the latest response,
    which still counts the pruned result, so a prune lowers their estimates from the following response on.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_harness.compaction import ClearToolResults, PruneToolResults

        agent = Agent(
            'openai:gpt-6.1-sol',
            capabilities=[ClearToolResults(max_fraction=0.7), PruneToolResults()],
        )
        ```
    """

    strategy_id = 'prune_tool_results'

    placeholder: str = '[tool result pruned]'
    """Prefix of the content that replaces a pruned result; the model's summary follows it."""

    id: str | None = field(default='prune_tool_results', kw_only=True)

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
        """Replace the result a `prune_tool_result` call names, or answer with the reason it was refused.

        This runs in a hook rather than in the tool function: under durable execution the tool can run
        in an activity that sees neither the message history nor this run's state.
        """
        if tool_def.tool_kind != _PRUNE_TOOL_RESULT_KIND.id:
            return result
        tool_call_id: str = args['tool_call_id']
        content = f'{self.placeholder} {args["summary"]}'
        messages: list[ModelMessage] = list(ctx.messages)
        start, end = _latest_step(messages)
        target = next(
            (
                part
                for message in messages[start:end]
                for part in message.parts
                if isinstance(part, ToolReturnPart) and part.tool_call_id == tool_call_id
            ),
            None,
        )
        if target is None:
            return f'Not pruned: {tool_call_id!r} is not a result from your previous step.'
        if cleared_return(target, content) is None:
            return f'Not pruned: the result of {tool_call_id!r} is structured data that cannot be replaced.'
        if len(content) >= len(str(target.content)):
            return 'Not pruned: the summary is not shorter than the result.'

        async def prune() -> list[ModelMessage]:
            step = replace_returns(messages[start:end], {tool_call_id: content})
            return [*messages[:start], *step, *messages[end:]]

        ctx.messages[:] = await compact_with_span(
            ctx, strategy=self.strategy_id, messages=messages, compact=prune, model_requested=True
        )
        return result


def _latest_step(messages: Sequence[ModelMessage]) -> tuple[int, int]:
    """The bounds of the requests the latest model response answered.

    Only these are searched and rewritten, so a reused tool call ID can't reach an older result. On a
    resume with deferred tool results, the returns already executed in the latest step follow that
    response, so it is found by type rather than by position.
    """
    responses = [index for index, message in enumerate(messages) if isinstance(message, ModelResponse)]
    start = responses[-2] + 1 if len(responses) > 1 else 0
    return start, responses[-1]
