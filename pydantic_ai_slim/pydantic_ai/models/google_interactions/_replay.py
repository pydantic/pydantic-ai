"""Replay complete steps while honoring edits to normalized message parts."""

from __future__ import annotations

import base64
import hashlib
import json

from google.genai import interactions as gi
from pydantic import JsonValue, TypeAdapter

from ... import _utils
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
    step_type = raw.get('type')
    if isinstance(part, ThinkingPart) and step_type == 'thought':
        replay_thought(raw, part)
    elif isinstance(part, ToolCallPart) and step_type == 'function_call':
        raw.update(
            id=part.tool_call_id, name=part.tool_name, arguments=_JSON_OBJECT.validate_python(part.args_as_dict())
        )
    elif isinstance(part, (TextPart, FilePart)) and step_type == 'model_output':
        raw['content'] = replay_content(raw, parts)
    elif isinstance(part, NativeToolCallPart):
        validate_native_step(raw, part)
        args = part.args_as_dict()
        if part.tool_name.partition(':')[0] == 'mcp_server':
            args = args.get('tool_args', {})
        raw.update(id=part.tool_call_id, arguments=_JSON_OBJECT.validate_python(args))
    elif isinstance(part, NativeToolReturnPart):
        validate_native_step(raw, part)
        raw.update(call_id=part.tool_call_id)
        raw['result'] = replay_native_result(part)
    else:
        raise UnexpectedModelBehavior('Persisted Google Interactions step does not match its message part.')
    # Persisted steps come from history a client may control. A file URI would make Google read that
    # object with the application's credentials, bypassing `sanitize_messages`, so media replays inline.
    for key in ('content', 'summary', 'result'):
        items = raw.get(key)
        if isinstance(items, list) and any(isinstance(item, dict) and 'uri' in item for item in items):
            raise UnexpectedModelBehavior('Persisted Google Interactions history cannot replay a file URI.')
    _STEP.validate_python(raw)
    return raw


def replay_native_result(part: NativeToolReturnPart) -> JsonValue:
    if part.tool_name.partition(':')[0] != 'mcp_server':
        return _JSON_VALUE.validate_python(part.content)
    result = replay_mcp_media(part.content)
    if isinstance(result, dict) and isinstance(result.get('content'), list):
        # An MCP `CallToolResult`. Google rejects its own `$ref` copies of it on replay
        # (https://github.com/googleapis/python-genai/issues/3313), so only the result itself is sent.
        return {'call_tool_result_json': json.dumps(result)}
    return result


def replay_mcp_media(value: object) -> JsonValue:
    if isinstance(value, BinaryContent):
        return _replay_mcp_binary(value)
    if isinstance(value, list):
        return [replay_mcp_media(item) for item in _CONTENT_LIST.validate_python(value)]
    if _utils.is_str_dict(value):
        return {name: replay_mcp_media(item) for name, item in value.items()}
    return _JSON_VALUE.validate_python(value)


def _replay_mcp_binary(value: BinaryContent) -> dict[str, JsonValue]:
    metadata = _JSON_OBJECT.validate_python((value.vendor_metadata or {}).get('google_interactions_mcp_content', {}))
    if 'uri' in metadata and 'type' not in metadata:
        metadata['blob'] = value.base64
    else:
        metadata.setdefault('type', 'image')
        metadata['data'] = value.base64
    mime_key = 'mimeType' if 'mimeType' in metadata else 'mime_type'
    if value.media_type != metadata.get(mime_key, 'application/octet-stream'):
        metadata[mime_key] = value.media_type
    return metadata


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
    # Persisted thought steps carry no summary: `ThinkingPart.content` owns it.
    if part.content:
        raw['summary'] = [{'type': 'text', 'text': part.content}]
    else:
        raw.pop('summary', None)


def replay_content(raw: dict[str, JsonValue], parts: list[ModelResponsePart]) -> list[JsonValue]:
    original = _OBJECT_LIST.validate_python(raw.get('content', []))
    content: list[JsonValue] = []
    for part in parts:
        index = (part.provider_details or {}).get('google_interactions_content_index')
        if not isinstance(index, int) or not 0 <= index < len(original):
            raise UnexpectedModelBehavior('Invalid content index in persisted Google Interactions history.')
        value = dict(original[index])
        if isinstance(part, TextPart) and value.get('type') == 'text':
            replay_text_content(value, part)
        elif isinstance(part, FilePart) and value.get('type') in ('image', 'audio', 'video', 'document'):
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
    value.pop('uri', None)
    value['data'] = base64.b64encode(part.content.data).decode()
    if value.get('mime_type', 'application/octet-stream') != part.content.media_type:
        value['mime_type'] = part.content.media_type
        value['type'] = media_kind(part.content.media_type)
