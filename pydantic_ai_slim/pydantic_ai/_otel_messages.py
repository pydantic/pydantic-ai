"""Type definitions of OpenTelemetry GenAI spec message parts.

Based on https://github.com/lmolkova/semantic-conventions/blob/eccd1f806e426a32c98271c3ce77585492d26de2/docs/gen-ai/non-normative/models.ipynb
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import JsonValue
from typing_extensions import NotRequired, TypedDict


class TextPart(TypedDict):
    type: Literal['text']
    content: NotRequired[str]


class ToolCallPart(TypedDict):
    type: Literal['tool_call']
    id: str
    name: str
    arguments: NotRequired[JsonValue]
    builtin: NotRequired[bool]  # Not (currently?) part of the spec, used by Logfire


class ToolCallResponsePart(TypedDict):
    type: Literal['tool_call_response']
    id: str
    name: str
    result: NotRequired[JsonValue]
    builtin: NotRequired[bool]  # Not (currently?) part of the spec, used by Logfire


class MediaUrlPart(TypedDict):
    type: Literal['image-url', 'audio-url', 'video-url', 'document-url']
    url: NotRequired[str]


class BinaryDataPart(TypedDict):
    type: Literal['binary']
    media_type: str
    content: NotRequired[str]


class ThinkingPart(TypedDict):
    type: Literal['thinking']
    content: NotRequired[str]


MessagePart: TypeAlias = 'TextPart | ToolCallPart | ToolCallResponsePart | MediaUrlPart | BinaryDataPart | ThinkingPart'


Role = Literal['system', 'user', 'assistant']


class ChatMessage(TypedDict):
    role: Role
    parts: list[MessagePart]


InputMessages: TypeAlias = list[ChatMessage]


class OutputMessage(ChatMessage):
    finish_reason: NotRequired[str]


OutputMessages: TypeAlias = list[OutputMessage]
