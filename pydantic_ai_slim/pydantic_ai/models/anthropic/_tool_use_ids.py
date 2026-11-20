"""The IDs Anthropic is sent for function tool calls and their results."""

from __future__ import annotations as _annotations

import re

from ..._utils import guard_tool_call_id
from ...messages import ToolCallPart, ToolReturnPart
from .._tool_call_ids import hashed_tool_call_id

_TOOL_USE_ID_PATTERN = re.compile(r'[a-zA-Z0-9_-]+')
"""The `tool_use.id` / `tool_result.tool_use_id` values Anthropic accepts."""


def wire_tool_use_id(part: ToolCallPart | ToolReturnPart) -> str:
    """The ID to send for a function tool call or its result.

    A history from another provider can carry IDs Anthropic rejects with
    `tool_use.id: String should match pattern '^[a-zA-Z0-9_-]+$'`, like the Gemini IDs an
    OpenAI-compatible gateway returns with a base64 thought signature in them. Such an ID is
    sent as a hash of itself, so a call and its result still pair up, and the same history
    maps to the same request every time.
    """
    tool_call_id = guard_tool_call_id(t=part)
    if _TOOL_USE_ID_PATTERN.fullmatch(tool_call_id):
        return tool_call_id
    return f'toolu_{hashed_tool_call_id(tool_call_id, 24)}'
