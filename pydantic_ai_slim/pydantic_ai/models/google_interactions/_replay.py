"""Replay complete steps while honoring edits to normalized message parts."""

from __future__ import annotations

import base64
import hashlib

from google.genai import interactions as gi
from pydantic import JsonValue, TypeAdapter

from ...exceptions import UnexpectedModelBehavior
from ...messages import (
    BinaryContent,
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
_JSON_VALUE = TypeAdapter(JsonValue)
_CONTENT_LIST = TypeAdapter(list[object])
_STEP = TypeAdapter(gi.Step)


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
        raw['result'] = replay_native_result(part)
    else:
        raise UnexpectedModelBehavior('Persisted Google Interactions step does not match its message part.')
    _STEP.validate_python(raw)
    return raw


def replay_native_result(part: NativeToolReturnPart) -> JsonValue:
    result: object = part.content
    if part.tool_name.partition(':')[0] != 'mcp_server' or not isinstance(result, list):
        return _JSON_VALUE.validate_python(result)
    content: list[JsonValue] = []
    for item in _CONTENT_LIST.validate_python(result):
        if isinstance(item, BinaryContent):
            value = _JSON_OBJECT.validate_python(
                (item.vendor_metadata or {}).get('google_interactions_mcp_content', {})
            )
            value.setdefault('type', 'image')
            value['data'] = item.base64
            if item.media_type != value.get('mime_type', 'application/octet-stream'):
                value['mime_type'] = item.media_type
            content.append(value)
        else:
            content.append(_JSON_VALUE.validate_python(item))
    return content


def validate_native_step(raw: dict[str, JsonValue], part: NativeToolCallPart | NativeToolReturnPart) -> None:
    step_type = raw.get('type')
    name = NATIVE_TOOL_NAMES.get(str(step_type).removesuffix('_call').removesuffix('_result'))
    tool_name = part.tool_name.partition(':')[0] if part.tool_name.startswith('mcp_server:') else part.tool_name
    if name != tool_name and not (tool_name == 'mcp_server' and step_type in ('function_call', 'function_result')):
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
            replay_text_content(value, part)
        elif isinstance(part, FilePart):
            replay_file_content(value, part)
        else:
            raise UnexpectedModelBehavior('Invalid content part in persisted Google Interactions history.')
        content.append(value)
    return content


def replay_text_content(value: dict[str, JsonValue], part: TextPart) -> None:
    unchanged = value.get('text') == part.content or (
        'text' not in value
        and (part.provider_details or {}).get('google_interactions_text_sha256')
        == hashlib.sha256(part.content.encode()).hexdigest()
    )
    value['text'] = part.content
    if not unchanged:
        value.pop('annotations', None)


def replay_file_content(value: dict[str, JsonValue], part: FilePart) -> None:
    fingerprint = (part.provider_details or {}).get('google_interactions_content_sha256')
    original_media_type = (part.provider_details or {}).get(
        'google_interactions_content_media_type', value.get('mime_type', part.content.media_type)
    )
    send_data = 'data' in value or 'uri' not in value or fingerprint != hashlib.sha256(part.content.data).hexdigest()
    if send_data:
        value.pop('uri', None)
        value['data'] = base64.b64encode(part.content.data).decode()
    if send_data or original_media_type != part.content.media_type:
        value['mime_type'] = part.content.media_type
        value['type'] = media_kind(part.content.media_type)
