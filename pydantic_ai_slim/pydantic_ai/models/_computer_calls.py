"""Keep a response's `computer` calls from running, for a provider that says not to run them."""

from __future__ import annotations as _annotations

from collections.abc import Sequence
from dataclasses import replace

from ..computer_use import COMPUTER_KIND
from ..messages import ModelResponsePart, ToolCallPart
from . import ModelRequestParameters

UNFINISHED_COMPUTER_ARGS = '{"actions": ['
"""The arguments given to a `computer` call that must not run.

They aren't valid JSON, like the arguments of a function call cut off by the token limit, so the call doesn't run.
"""


def block_computer_calls(
    parts: Sequence[ModelResponsePart], model_request_parameters: ModelRequestParameters
) -> list[ModelResponsePart]:
    """Give every `computer` call in `parts` arguments that aren't valid JSON, so none of them runs.

    A blocked call keeps its place. Native and function-tool `computer` calls are blocked alike, and so are calls to a
    deferred `computer` tool, because tool search can reveal it in the same response. Other function calls are left as
    they came.

    This doesn't end the run. The agent treats a blocked call like any function call whose arguments don't parse: it
    asks the model for the call again, and raises once the tool's retries are used up. When the response hit the token
    limit and the call is its last part, that error is `IncompleteToolCall`.
    """
    tool_names = {tool.name for tool in model_request_parameters.function_tools if tool.tool_kind == COMPUTER_KIND.id}
    return [
        replace(part, args=UNFINISHED_COMPUTER_ARGS)
        if isinstance(part, ToolCallPart) and part.tool_name in tool_names
        else part
        for part in parts
    ]
