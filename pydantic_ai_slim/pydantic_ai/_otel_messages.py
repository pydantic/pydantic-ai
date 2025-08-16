from __future__ import annotations

from typing import Literal

from pydantic import JsonValue
from typing_extensions import NotRequired, TypeAlias, TypedDict


class TextPart(TypedDict):
    type: Literal['text']
    content: NotRequired[str]


class ToolCallPart(TypedDict):
    type: Literal['tool_call']
    id: str
    name: str
    arguments: NotRequired[JsonValue]


class ToolCallResponsePart(TypedDict):
    type: Literal['tool_call_response']
    id: str
    name: str
    result: NotRequired[JsonValue]


class MediaUrlPart(TypedDict):
    type: Literal['image-url', 'audio-url', 'video-url', 'document-url']
    url: NotRequired[str]


class BinaryDataPart(TypedDict):
    type: Literal['binary']
    media_type: str
    binary_content: NotRequired[str]


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
