from __future__ import annotations as _annotations

from collections.abc import Sequence
from typing import Any, TypeAlias

from .messages import ModelResponse, ToolCallPart
from .run import AgentRunResult

ToolCallSource: TypeAlias = AgentRunResult[Any] | ModelResponse

_UNSET = object()


def tool_calls(source: ToolCallSource, *, across_history: bool = False) -> list[ToolCallPart]:
    """Return function tool calls from a model response or agent run result.

    By default this reads the final response, matching `result.response.tool_calls`.
    Set `across_history=True` to collect calls from every model response in an
    agent run's message history.
    """
    if isinstance(source, ModelResponse):
        return source.tool_calls

    if not across_history:
        return source.response.tool_calls

    return [
        call for message in source.all_messages() if isinstance(message, ModelResponse) for call in message.tool_calls
    ]


def assert_tool_call(
    source: ToolCallSource,
    tool_name: str,
    *,
    args: Any = _UNSET,
    across_history: bool = False,
) -> ToolCallPart:
    """Assert that a tool call with `tool_name` and optional `args` exists."""
    calls = tool_calls(source, across_history=across_history)
    for call in calls:
        if call.tool_name != tool_name:
            continue
        if args is _UNSET or call.args == args:
            return call

    if not calls:
        raise AssertionError(f'No tool calls found; expected {tool_name!r}.')

    available = _format_tool_calls(calls)
    if args is _UNSET:
        raise AssertionError(f'Expected tool call {tool_name!r}; available tool calls: {available}.')
    raise AssertionError(f'Expected tool call {tool_name!r} with args {args!r}; available tool calls: {available}.')


def assert_tool_call_sequence(
    source: ToolCallSource,
    expected_tool_names: Sequence[str],
    *,
    across_history: bool = False,
) -> list[ToolCallPart]:
    """Assert that tool calls were emitted in exactly the expected order."""
    calls = tool_calls(source, across_history=across_history)
    actual_tool_names = [call.tool_name for call in calls]
    expected_tool_names = list(expected_tool_names)

    if actual_tool_names == expected_tool_names:
        return calls

    if not calls:
        raise AssertionError(f'No tool calls found; expected sequence {expected_tool_names!r}.')
    raise AssertionError(f'Expected tool call sequence {expected_tool_names!r}; got {actual_tool_names!r}.')


def _format_tool_calls(calls: Sequence[ToolCallPart]) -> str:
    return ', '.join(f'{call.tool_name}({call.args!r})' for call in calls)


__all__ = ('ToolCallSource', 'assert_tool_call', 'assert_tool_call_sequence', 'tool_calls')
