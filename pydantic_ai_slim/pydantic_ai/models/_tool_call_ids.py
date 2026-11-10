"""Shared tool call ID helpers used by the base `Model` and the provider model classes."""

from __future__ import annotations as _annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime

from ..messages import ModelMessage, ModelResponse, ToolCallPart, ToolReturnPart

_BASE62_ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


def hashed_tool_call_id(tool_call_id: str, length: int) -> str:
    """The low-order `length` base62 digits of the SHA-256 of `tool_call_id`."""
    number = int.from_bytes(hashlib.sha256(tool_call_id.encode()).digest())
    digits: list[str] = []
    for _ in range(length):
        number, remainder = divmod(number, 62)
        digits.append(_BASE62_ALPHABET[remainder])
    return ''.join(digits)


def tool_call_id_scope(provider_response_id: str | None, timestamp: datetime) -> str:
    """The value that qualifies the tool call IDs of a response whose IDs are only unique within it.

    See `ModelProfile.tool_call_ids_are_response_scoped`. It's the response's `provider_response_id`, or its timestamp
    when it has none. A stored or durably replayed response keeps both, so restoring its IDs finds the same value.
    """
    if provider_response_id:
        return provider_response_id
    return (timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=UTC)).astimezone(UTC).isoformat()


def qualify_tool_call_id(tool_call_id: str, scope: str) -> str:
    """Make a response-scoped tool call ID unique across the history with a hash of its response's `scope`.

    The underscore and eight base62 characters keep an ID made of `[a-zA-Z0-9_-]` within that set, which Anthropic
    and Bedrock require of the IDs in a history from another provider.
    """
    return f'{tool_call_id}_{hashed_tool_call_id(scope, 8)}'


def stored_tool_call_id(
    tool_call_id: str, *, response_scoped: bool, provider_response_id: str | None, timestamp: datetime
) -> str:
    """The ID to store for a tool call ID a provider sent in the response with `provider_response_id` and `timestamp`.

    It's qualified when the profile sets `tool_call_ids_are_response_scoped`, passed as `response_scoped`. An empty
    ID is kept, so the caller still generates one.
    """
    if not (tool_call_id and response_scoped):
        return tool_call_id
    return qualify_tool_call_id(tool_call_id, tool_call_id_scope(provider_response_id, timestamp))


def unqualify_tool_call_id(tool_call_id: str, scope: str) -> str:
    """Restore the ID the provider sent from one `qualify_tool_call_id` qualified with `scope`."""
    restored = tool_call_id.removesuffix(f'_{hashed_tool_call_id(scope, 8)}')
    if restored != tool_call_id:
        return restored
    # Before the flag moved to `ModelProfile`, `OpenAIResponsesModel` qualified IDs as `{response_id}:{tool_call_id}`.
    return tool_call_id.removeprefix(f'{scope}:')


def with_unqualified_tool_call_ids(message: ModelMessage, scope: str) -> ModelMessage:
    """`message` with the IDs of its tool calls, or of its tool results, that were qualified with `scope` restored."""
    if isinstance(message, ModelResponse):
        response_parts = [
            replace(part, tool_call_id=unqualify_tool_call_id(part.tool_call_id, scope))
            if isinstance(part, ToolCallPart)
            else part
            for part in message.parts
        ]
        return replace(message, parts=response_parts)
    request_parts = [
        replace(part, tool_call_id=unqualify_tool_call_id(part.tool_call_id, scope))
        if isinstance(part, ToolReturnPart)
        else part
        for part in message.parts
    ]
    return replace(message, parts=request_parts)
