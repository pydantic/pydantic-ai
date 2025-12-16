"""Type definitions of OpenTelemetry GenAI spec message parts.

Based on the OpenTelemetry semantic conventions for GenAI:
https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-input-messages.json
https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-output-messages.json
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import JsonValue
from typing_extensions import NotRequired, TypedDict


class TextPart(TypedDict):
    """A text part in a GenAI message."""

    type: Literal['text']
    content: NotRequired[str]


class ToolCallPart(TypedDict):
    """A tool call part in a GenAI message."""

    type: Literal['tool_call']
    id: str
    name: str
    arguments: NotRequired[JsonValue]
    builtin: NotRequired[bool]  # Not (currently?) part of the spec, used by Logfire


class ToolCallResponsePart(TypedDict):
    """A tool call response part in a GenAI message."""

    type: Literal['tool_call_response']
    id: str
    name: str
    # TODO: This should be `response` not `result`
    result: NotRequired[JsonValue]
    builtin: NotRequired[bool]  # Not (currently?) part of the spec, used by Logfire


class UriPart(TypedDict):
    """A URI part in a GenAI message (for images, audio, video, documents).

    Per the semantic conventions, uses 'uri' type with modality field.
    """

    type: Literal['uri']
    uri: NotRequired[str]
    modality: NotRequired[str]


class BlobPart(TypedDict):
    """A blob (binary data) part in a GenAI message.

    Per the semantic conventions, uses 'blob' type with modality field.
    """

    type: Literal['blob']
    blob: NotRequired[str]
    modality: NotRequired[str]


class ReasoningPart(TypedDict):
    """A reasoning/thinking part in a GenAI message.

    Per the semantic conventions, uses 'reasoning' type.
    """

    type: Literal['reasoning']
    content: NotRequired[str]


MessagePart: TypeAlias = 'TextPart | ToolCallPart | ToolCallResponsePart | UriPart | BlobPart | ReasoningPart'


Role = Literal['system', 'user', 'assistant']


class ChatMessage(TypedDict):
    """A chat message in the GenAI format."""

    role: Role
    parts: list[MessagePart]


InputMessages: TypeAlias = list[ChatMessage]


class OutputMessage(ChatMessage):
    """An output message with optional finish reason."""

    finish_reason: NotRequired[str]


OutputMessages: TypeAlias = list[OutputMessage]
