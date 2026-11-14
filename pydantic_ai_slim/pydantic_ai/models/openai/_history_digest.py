"""The digest of the history a stored response was created from, which decides whether a request continues from it."""

from __future__ import annotations as _annotations

import hashlib
import hmac
from collections.abc import Sequence
from dataclasses import replace
from typing import TypeVar

from pydantic import TypeAdapter
from typing_extensions import Protocol

from ...computer_use import _LABEL_KEYS  # pyright: ignore[reportPrivateUsage]
from ...messages import (
    BaseToolReturnPart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    TextContent,
    ToolCallPart,
    UserPromptPart,
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
    """`part` as the server holds it: a base call, without what the agent graph adds after `request()`.

    The digest is taken in `request()`. The graph then promotes the call to its typed subclass and labels a
    `computer` call with its model's grid and screenshot limits. None of that is sent, so leaving it in would stop a
    later response from matching.
    """
    args = part.args if part.args is None or isinstance(part.args, str) else {**part.args}
    details = {key: value for key, value in (part.provider_details or {}).items() if key not in _LABEL_KEYS}
    return ToolCallPart(
        tool_name=part.tool_name,
        args=args,
        tool_call_id=part.tool_call_id,
        id=part.id,
        provider_name=part.provider_name,
        provider_details=details or None,
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


def with_history_digest(response: ModelResponse, history: Hasher | None) -> ModelResponse:
    """`response` with its digest recorded, when `history` says a later request may continue from it."""
    if history is None:
        return response
    return replace(response, history_digest=continuation_digest(history, response.parts))
