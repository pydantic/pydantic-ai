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


class UriPart(TypedDict):
    """Part type for URIs following OpenTelemetry GenAI semantic conventions.

    Used in instrumentation version 4+ to align with the GenAI spec:
    https://opentelemetry.io/docs/specs/semconv/gen-ai/non-normative/examples-llm-calls/#multimodal-inputs-example

    The modality field is present for supported types (ImageUrl, AudioUrl, VideoUrl) but omitted for
    unsupported types (DocumentUrl) since the spec only defines image, audio, and video modalities.
    """

    type: Literal['uri']
    modality: NotRequired[Literal['image', 'audio', 'video']]
    uri: NotRequired[str]
    mime_type: NotRequired[str]


class BinaryDataPart(TypedDict):
    type: Literal['binary']
    media_type: str
    content: NotRequired[str]


class BlobPart(TypedDict):
    """Part type for inline binary data following OpenTelemetry GenAI semantic conventions.

    Used in instrumentation version 4+ to align with the GenAI spec:
    https://opentelemetry.io/docs/specs/semconv/gen-ai/non-normative/examples-llm-calls/#multimodal-inputs-example

    The modality field is optional since it's inferred from media_type, which may fail for unknown MIME types.
    Only image, audio, and video modalities are included per the spec.
    """

    type: Literal['blob']
    modality: NotRequired[Literal['image', 'audio', 'video']]
    mime_type: NotRequired[str]
    content: NotRequired[str]


class ThinkingPart(TypedDict):
    type: Literal['thinking']
    content: NotRequired[str]


MessagePart: TypeAlias = (
    'TextPart | ToolCallPart | ToolCallResponsePart | MediaUrlPart | UriPart | BinaryDataPart | BlobPart | ThinkingPart'
)


Role = Literal['system', 'user', 'assistant']


class ChatMessage(TypedDict):
    role: Role
    parts: list[MessagePart]


InputMessages: TypeAlias = list[ChatMessage]


class OutputMessage(ChatMessage):
    finish_reason: NotRequired[str]


OutputMessages: TypeAlias = list[OutputMessage]
