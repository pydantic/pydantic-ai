"""Digests of the history a response was created from, for models that continue server-side state.

A model that continues a conversation from a stored response (OpenAI's `previous_response_id`, Anthropic's message
threads) records [`ModelResponse.history_digest`][pydantic_ai.messages.ModelResponse.history_digest] on it, and a
later request continues from that response only while the digest still matches. Otherwise the server would carry on
from its own copy, missing any edit or redaction made to the history since.
"""

from __future__ import annotations as _annotations

import hashlib
import hmac
from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol, TypeVar

from pydantic import TypeAdapter

from ..messages import (
    BaseToolReturnPart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelRequestPart,
    ModelResponsePart,
    TextContent,
    ToolCallPart,
    UserPromptPart,
)

__all__ = (
    'Hasher',
    'call_as_sent',
    'continuation_digest',
    'history_hasher',
    'part_without_application_metadata',
    'without_application_metadata',
)


class Hasher(Protocol):
    def update(self, data: bytes, /) -> None: ...
    def copy(self) -> Hasher: ...
    def hexdigest(self) -> str: ...


_RESPONSE_PART_ADAPTER: TypeAdapter[ModelResponsePart] = TypeAdapter(ModelResponsePart)


def history_hasher(history: Sequence[ModelMessage], key: bytes | None, *, with_instructions: bool = False) -> Hasher:
    """A SHA-256, or an HMAC with `key`, fed the history a response is created from.

    A request's `instructions` are left out unless `with_instructions`: the server doesn't keep the top-level field
    with a response, so each request sends its own, and how an earlier request's instructions are rendered can differ
    between requests. Digests recorded before they were left out include them, as do those of requests that sent them
    as an input message.
    """
    # `exclude_none` keeps the digest stable when a release adds a field, which loads as `None` from older histories.
    # `exclude_defaults` would go further, but it drops the part kinds too, so a role change would go unnoticed.
    serialized = ModelMessagesTypeAdapter.dump_json(
        [without_application_metadata(message, keep_instructions=with_instructions) for message in history],
        exclude_none=True,
    )
    hasher: Hasher = hashlib.sha256() if key is None else hmac.new(key, digestmod=hashlib.sha256)
    hasher.update(serialized)
    return hasher


def continuation_digest(history: Hasher, parts: Sequence[ModelResponsePart]) -> str:
    """The digest of a response's parts after the history it was created from, both of which the server holds."""
    hasher = history.copy()
    for part in parts:
        hasher.update(b'\n')
        hasher.update(
            _RESPONSE_PART_ADAPTER.dump_json(
                part_without_application_metadata(call_as_sent(part) if isinstance(part, ToolCallPart) else part),
                exclude_none=True,
            )
        )
    return hasher.hexdigest()


def call_as_sent(part: ToolCallPart) -> ToolCallPart:
    """`part` as the server holds it: a base call, without the `tool_kind` the agent graph stamps on after `request()`.

    The digest is taken in `request()`, and the graph then promotes the call to its kind's typed subclass. Neither
    the kind nor the subclass is sent, so leaving them in would make a response compared later never match.
    """
    args = part.args if part.args is None or isinstance(part.args, str) else {**part.args}
    return ToolCallPart(
        tool_name=part.tool_name,
        args=args,
        tool_call_id=part.tool_call_id,
        id=part.id,
        provider_name=part.provider_name,
        provider_details=part.provider_details,
    )


def without_application_metadata(message: ModelMessage, *, keep_instructions: bool) -> ModelMessage:
    """`message` without the application-only `metadata` fields that are never sent to the model.

    They can hold values that don't serialize to JSON, and changing them doesn't change what the server holds.
    They're cleared by type, and a tool's return value is left whole: it's sent as is, `metadata` keys included.
    Without `keep_instructions`, a request's `instructions` are cleared too.
    """
    # Split by type so `part_without_application_metadata` sees a single part type.
    if isinstance(message, ModelRequest):
        parts = [part_without_application_metadata(p) for p in message.parts]
        if keep_instructions:
            return replace(message, parts=parts, metadata=None)
        return replace(message, parts=parts, metadata=None, instructions=None)
    return replace(message, parts=[part_without_application_metadata(p) for p in message.parts], metadata=None)


_PartT = TypeVar('_PartT', ModelRequestPart, ModelResponsePart)


def part_without_application_metadata(part: _PartT) -> _PartT:
    if isinstance(part, BaseToolReturnPart):
        return replace(part, metadata=None)
    if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
        content = [replace(item, metadata=None) if isinstance(item, TextContent) else item for item in part.content]
        return replace(part, content=content)
    return part
