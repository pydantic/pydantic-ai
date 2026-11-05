"""Replay complete steps while honoring edits to normalized message parts."""

from __future__ import annotations

import base64
import hashlib

from google.genai import interactions as gi
from pydantic import JsonValue, TypeAdapter

from ...exceptions import UnexpectedModelBehavior
from ...messages import (
    FilePart,
    ModelResponsePart,
    NativeToolCallPart,
    NativeToolReturnPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)
from ._response import NATIVE_TOOL_NAMES, media_kind

_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
_OBJECT_LIST = TypeAdapter(list[dict[str, JsonValue]])


def step_index(part: ModelResponsePart) -> int | None:
    value = (part.provider_details or {}).get('google_interactions_step_index')
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def replay_step(original: dict[str, JsonValue], parts: list[ModelResponsePart]) -> dict[str, JsonValue]:
    """Keep opaque fields and boundaries, updating only the represented parts the caller changed."""
    raw = _JSON_OBJECT.validate_python(original)
    part = parts[0]
    if isinstance(part, ThinkingPart):
        replay_thought(raw, part)
    elif isinstance(part, ToolCallPart):
        raw.update(
            id=part.tool_call_id, name=part.tool_name, arguments=_JSON_OBJECT.validate_python(part.args_as_dict())
        )
    elif isinstance(part, (TextPart, FilePart)):
        raw['content'] = replay_content(raw, parts)
    elif isinstance(part, NativeToolCallPart):
        validate_native_step(raw, part)
        raw.update(id=part.tool_call_id, arguments=_JSON_OBJECT.validate_python(part.args_as_dict()))
    elif isinstance(part, NativeToolReturnPart):
        validate_native_step(raw, part)
        raw.update(call_id=part.tool_call_id)
        raw['result'] = TypeAdapter(JsonValue).validate_python(part.content)
    else:
        raise UnexpectedModelBehavior('Persisted Google Interactions step does not match its message part.')
    TypeAdapter(gi.Step).validate_python(raw)
    return raw


def validate_native_step(raw: dict[str, JsonValue], part: NativeToolCallPart | NativeToolReturnPart) -> None:
    step_type = raw.get('type')
    name = NATIVE_TOOL_NAMES.get(str(step_type).removesuffix('_call').removesuffix('_result'))
    if name != part.tool_name and not (
        part.tool_name == 'mcp_server' and step_type in ('function_call', 'function_result')
    ):
        raise UnexpectedModelBehavior(
            f'Missing or unsupported Google Interactions replay step for native tool {part.tool_name!r}.'
        )


def replay_thought(raw: dict[str, JsonValue], part: ThinkingPart) -> None:
    if part.signature is None:
        raw.pop('signature', None)
    else:
        raw['signature'] = part.signature
    summary = _OBJECT_LIST.validate_python(raw.get('summary', []))
    text = ''.join(str(item.get('text', '')) for item in summary)
    if text != part.content:
        raw['summary'] = [{'type': 'text', 'text': part.content}] if part.content else []


def replay_content(raw: dict[str, JsonValue], parts: list[ModelResponsePart]) -> list[JsonValue]:
    original = _OBJECT_LIST.validate_python(raw.get('content', []))
    content: list[JsonValue] = []
    for part in parts:
        index = (part.provider_details or {}).get('google_interactions_content_index')
        if not isinstance(index, int) or not 0 <= index < len(original):
            raise UnexpectedModelBehavior('Invalid content index in persisted Google Interactions history.')
        value = dict(original[index])
        if isinstance(part, TextPart):
            if value.get('text') != part.content:
                value['text'] = part.content
                value.pop('annotations', None)
        elif isinstance(part, FilePart):
            fingerprint = (part.provider_details or {}).get('google_interactions_content_sha256')
            if 'data' in value or fingerprint != hashlib.sha256(part.content.data).hexdigest():
                value.pop('uri', None)
                value['data'] = base64.b64encode(part.content.data).decode()
                value['mime_type'] = part.content.media_type
                value['type'] = media_kind(part.content.media_type)
        else:
            raise UnexpectedModelBehavior('Invalid content part in persisted Google Interactions history.')
        content.append(value)
    return content
