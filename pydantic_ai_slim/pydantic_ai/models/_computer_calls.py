"""Keep a response's `computer` calls from running, for a provider that says not to run them."""

from __future__ import annotations as _annotations

from collections.abc import Sequence
from dataclasses import replace

from ..computer_use import COMPUTER_KIND
from ..messages import ModelResponsePart, ToolCallPart
from . import ModelRequestParameters

UNFINISHED_COMPUTER_ARGS = '{"actions": ['
"""Arguments for a `computer` call that must not run: invalid JSON, like a call cut off by the token limit."""


def block_computer_calls(
    parts: Sequence[ModelResponsePart], model_request_parameters: ModelRequestParameters
) -> list[ModelResponsePart]:
    """Give every `computer` call in `parts` arguments that aren't valid JSON, so none of them runs.

    A blocked call keeps its place. Calls to a deferred `computer` tool are blocked too, because tool search can reveal
    it in the same response.

    The agent treats a blocked call like any call whose arguments don't parse: it asks the model again, and raises once
    the tool's retries are used up (`IncompleteToolCall` when the call is the last part of a response cut off by the
    token limit).
    """
    tool_names = {tool.name for tool in model_request_parameters.function_tools if tool.tool_kind == COMPUTER_KIND.id}
    return [
        replace(part, args=UNFINISHED_COMPUTER_ARGS)
        if isinstance(part, ToolCallPart) and part.tool_name in tool_names
        else part
        for part in parts
    ]
