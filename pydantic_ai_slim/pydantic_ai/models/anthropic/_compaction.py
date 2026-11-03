"""On-demand compaction: settling a request for the `compaction` parameter, and its summarization prompt."""

from __future__ import annotations as _annotations

from dataclasses import replace

from anthropic.types.beta import (
    BetaOutputConfigParam,
    BetaTokenTaskBudgetParam,
    BetaToolChoiceAutoParam,
    BetaToolChoiceParam,
)

from ...messages import CompactionPart, ModelMessage, ModelResponse, ToolCallPart

ON_DEMAND_COMPACTION_BETA = 'compact-2026-09-04'
"""The beta for the `compaction` request parameter, needed on every later request that carries its signed block too."""

_SUMMARIZATION_PROMPT = """\
Write a summary of the conversation so far. It replaces the conversation: whoever continues it sees only your \
summary, so anything you leave out is lost.

Keep:
- the user's goals, requests and constraints, and how they changed;
- the decisions made and why, with exact names, identifiers, values, file paths and commands;
- the tool results that later work depends on;
- what is done, what is in progress and what comes next, including any request still waiting for an answer.

Write it as notes for whoever continues the conversation. Don't answer the latest request, and don't add anything \
the conversation doesn't contain.

The summary should focus on the following. Where it conflicts with the guidance above, follow it:
<focus>
{focus}
</focus>"""


def compaction_instructions(focus: str) -> str:
    """The full summarization prompt to send for `focus`, which would otherwise replace Anthropic's prompt entirely."""
    return _SUMMARIZATION_PROMPT.format(focus=focus)


def is_signed_compaction(part: CompactionPart) -> bool:
    """Whether `part` is a block from an on-demand compaction request, which carries a signature."""
    return part.provider_details is not None and part.provider_details.get('signature') is not None


def without_unanswered_tool_calls(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Drop the tool calls of a trailing response, which a compaction request rejects without their results."""
    last = messages[-1] if messages else None
    if not isinstance(last, ModelResponse) or not any(isinstance(part, ToolCallPart) for part in last.parts):
        return messages
    parts = [part for part in last.parts if not isinstance(part, ToolCallPart)]
    return [*messages[:-1], replace(last, parts=parts)] if parts else messages[:-1]


def settle_tool_choice(tool_choice: BetaToolChoiceParam | None) -> BetaToolChoiceParam | None:
    """Relax a forcing `tool_choice` to `auto`: a compaction request samples no reply, so it can't call a tool."""
    if tool_choice is None or tool_choice['type'] not in ('any', 'tool'):
        return tool_choice
    return BetaToolChoiceAutoParam(type='auto')


def settle_output_config(output_config: BetaOutputConfigParam | None) -> BetaOutputConfigParam | None:
    """Drop the structured output format and the remaining task budget, both of which a compaction request rejects."""
    if output_config is None:
        return None
    settled = output_config.copy()
    settled.pop('format', None)
    if (task_budget := settled.get('task_budget')) is not None and 'remaining' in task_budget:
        settled['task_budget'] = BetaTokenTaskBudgetParam(type=task_budget['type'], total=task_budget['total'])
    return settled or None
