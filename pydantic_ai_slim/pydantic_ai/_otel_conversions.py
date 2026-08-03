"""Conversion of OTel-format messages back to pydantic-ai ModelMessages.

This is the reverse of the instrumentation layer's forward conversion
(`InstrumentationSettings.messages_to_otel_messages`), enabling recorded conversations
to be replayed as `message_history`.
"""

from __future__ import annotations

import base64
import itertools
import json
from collections.abc import Sequence
from typing import Any, cast, get_args

from . import _utils
from .messages import (
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    FilePart,
    FinishReason,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    NativeToolCallPart,
    NativeToolReturnPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
    VideoUrl,
)

_FINISH_REASONS: frozenset[str] = frozenset(get_args(FinishReason))


def otel_messages_to_model_messages(
    otel_messages: str | Sequence[dict[str, Any]],
) -> list[ModelMessage]:
    """Convert OTEL format messages to pydantic-ai ModelMessages.

    This is the inverse of
    [`InstrumentationSettings.messages_to_otel_messages`][pydantic_ai.models.instrumented.InstrumentationSettings.messages_to_otel_messages].

    Supports the ChatMessage format used by pydantic-ai's OTEL instrumentation:

    - `pydantic_ai.all_messages` attribute on agent run spans
    - `gen_ai.input.messages` / `gen_ai.output.messages` attributes on model request spans

    Also supports the legacy v1 events format (with `event.name` keys).

    Multi-modal content is handled across instrumentation versions: the v2/v3 media parts
    (`image-url`/`audio-url`/`video-url`/`document-url`/`binary`) and the v4+ OTEL GenAI parts
    (`uri`/`blob`/`file`). Provider-hosted file references can't be rebuilt because the OTel format
    does not identify their provider, so they are replaced by a text marker noting the missing data.

    Note: this conversion is lossy. Some information (e.g. timestamps, `instructions`,
    provider details) is not preserved in the OTEL format and will use defaults.
    Content excluded by `include_content=False` will be empty strings.

    Args:
        otel_messages: A JSON string or a list of message dicts.

    Returns:
        A list of `ModelMessage` objects that can be passed as `message_history`
        to [`Agent.run`][pydantic_ai.agent.AbstractAgent.run].
    """
    parsed: list[dict[str, Any]]
    if isinstance(otel_messages, str):
        parsed = json.loads(otel_messages)
    else:
        parsed = list(otel_messages)

    if not parsed:
        return []

    # Detect format: legacy events have 'event.name' key
    first = parsed[0]
    if 'event.name' in first:
        return _legacy_events_to_model_messages(parsed)
    else:
        return _chat_messages_to_model_messages(parsed)


# ── ChatMessage format → ModelMessages ────────────────────────────────


def _chat_messages_to_model_messages(
    chat_messages: list[dict[str, Any]],
) -> list[ModelMessage]:
    """Convert ChatMessage format (`{role, parts}`) to ModelMessages.

    Merges consecutive non-assistant messages into a single ModelRequest,
    reversing the split performed by `messages_to_otel_messages`.
    """
    result: list[ModelMessage] = []
    pending_request_parts: list[ModelRequestPart] = []

    for msg in chat_messages:
        role = msg.get('role', '')
        parts_data: list[dict[str, Any]] = msg.get('parts', [])

        if role == 'assistant':
            if pending_request_parts:
                result.append(ModelRequest(parts=pending_request_parts))
                pending_request_parts = []

            response_parts = _convert_assistant_parts(parts_data)
            kwargs: dict[str, Any] = {}
            # Third-party traces may carry finish reasons outside pydantic-ai's `FinishReason` values; drop those.
            if (finish_reason := msg.get('finish_reason')) in _FINISH_REASONS:
                kwargs['finish_reason'] = finish_reason
            result.append(ModelResponse(parts=response_parts, **kwargs))
        elif role == 'system':
            pending_request_parts.extend(_convert_system_parts(parts_data))
        else:
            # 'user' or any other role
            pending_request_parts.extend(_convert_user_parts(parts_data))

    if pending_request_parts:
        result.append(ModelRequest(parts=pending_request_parts))

    return result


def _convert_system_parts(parts: list[dict[str, Any]]) -> list[ModelRequestPart]:
    result: list[ModelRequestPart] = []
    for part in parts:
        if part.get('type') == 'text':
            result.append(SystemPromptPart(part.get('content', '')))
    return result


def _convert_user_parts(parts: list[dict[str, Any]]) -> list[ModelRequestPart]:
    """Convert user-role OTEL parts to ModelRequestParts.

    Consecutive text/media/binary parts are merged into a single UserPromptPart.
    Tool call responses become separate ToolReturnParts.
    """
    result: list[ModelRequestPart] = []
    user_content_parts: list[dict[str, Any]] = []

    def flush_user_content() -> None:
        if user_content_parts:
            result.append(_make_user_prompt_part(user_content_parts))
            user_content_parts.clear()

    for part in parts:
        ptype = part.get('type', '')
        if ptype == 'tool_call_response':
            flush_user_content()
            tool_name = part.get('name', '')
            tool_call_id = part.get('id', _utils.generate_tool_call_id())
            # Support both 'result' (pydantic-ai) and 'response' (logfire semconv) field names
            content = part.get('result', part.get('response', ''))
            result.append(
                ToolReturnPart(
                    tool_name=tool_name,
                    content=content,
                    tool_call_id=tool_call_id,
                )
            )
        else:
            user_content_parts.append(part)

    flush_user_content()
    return result


_MEDIA_URL_TYPES: dict[str, type[ImageUrl] | type[AudioUrl] | type[VideoUrl] | type[DocumentUrl]] = {
    'image-url': ImageUrl,
    'audio-url': AudioUrl,
    'video-url': VideoUrl,
    'document-url': DocumentUrl,
}


def _binary_from_otel(part: dict[str, Any]) -> BinaryContent:
    """Build `BinaryContent` from a v2/v3 `binary` part or a v4+ `blob` part."""
    media_type = part.get('media_type') or part.get('mime_type') or 'application/octet-stream'
    b64_content = part.get('content', '')
    data = base64.b64decode(b64_content) if b64_content else b''
    # Match Pydantic validation of real messages, which narrows image content to `BinaryImage`.
    return BinaryContent.narrow_type(BinaryContent(data=data, media_type=media_type))


def _uri_part_to_url(part: dict[str, Any]) -> ImageUrl | AudioUrl | VideoUrl | DocumentUrl | None:
    """Convert a v4+ OTEL GenAI `uri` part to a media URL based on its modality.

    Returns `None` when no URL is present (e.g. recorded with `include_content=False`).
    """
    url = part.get('uri', '')
    if not url:
        return None
    media_type: str | None = part.get('mime_type')
    modality = part.get('modality')
    if modality == 'image':
        return ImageUrl(url, media_type=media_type)
    elif modality == 'audio':
        return AudioUrl(url, media_type=media_type)
    elif modality == 'video':
        return VideoUrl(url, media_type=media_type)
    # No modality is emitted for document URLs.
    return DocumentUrl(url, media_type=media_type)


def _file_part_to_content(part: dict[str, Any]) -> UserContent:
    """Replace a provider-hosted file reference with a marker because its provider is not recorded."""
    media_type = part.get('mime_type', 'application/octet-stream')
    return f'[unavailable file ({media_type}): provider-hosted reference not captured in OTEL]'


def _make_user_prompt_part(parts: list[dict[str, Any]]) -> UserPromptPart:
    """Create a UserPromptPart from a list of OTEL message parts."""
    if len(parts) == 1 and parts[0].get('type') == 'text':
        return UserPromptPart(parts[0].get('content', ''))

    content: list[UserContent] = []
    for part in parts:
        ptype = part.get('type', '')
        if ptype == 'text':
            content.append(part.get('content', ''))
        elif ptype in ('image-url', 'audio-url', 'video-url', 'document-url'):
            # Legacy (v2/v3) media URL parts carry the URL directly under `url`.
            if url := part.get('url', ''):
                content.append(_MEDIA_URL_TYPES[ptype](url))
        elif ptype == 'uri':
            # v4+ OTEL GenAI `uri` part — the modality determines the URL type.
            if (url_content := _uri_part_to_url(part)) is not None:
                content.append(url_content)
        elif ptype == 'file':
            # Provider-hosted file reference (`UploadedFile`), emitted at all versions.
            content.append(_file_part_to_content(part))
        elif ptype in ('binary', 'blob'):
            # `binary` is the v2/v3 inline-binary part; `blob` is its v4+ equivalent.
            content.append(_binary_from_otel(part))

    if not content:
        return UserPromptPart('')
    elif len(content) == 1 and isinstance(content[0], str):
        return UserPromptPart(content[0])
    return UserPromptPart(content)


def _convert_assistant_parts(parts: list[dict[str, Any]]) -> list[ModelResponsePart]:
    result: list[ModelResponsePart] = []
    for part in parts:
        ptype = part.get('type', '')
        if ptype == 'text':
            result.append(TextPart(part.get('content', '')))
        elif ptype == 'thinking':
            result.append(ThinkingPart(part.get('content', '')))
        elif ptype == 'tool_call':
            builtin = part.get('builtin', False)
            tool_name = part.get('name', '')
            tool_call_id = part.get('id', _utils.generate_tool_call_id())
            args = part.get('arguments')
            if builtin:
                result.append(NativeToolCallPart(tool_name=tool_name, args=args, tool_call_id=tool_call_id))
            else:
                result.append(ToolCallPart(tool_name=tool_name, args=args, tool_call_id=tool_call_id))
        elif ptype == 'tool_call_response':
            # Native tool returns can appear in assistant messages
            builtin = part.get('builtin', False)
            if builtin:
                tool_name = part.get('name', '')
                tool_call_id = part.get('id', _utils.generate_tool_call_id())
                content = part.get('result', part.get('response', ''))
                result.append(NativeToolReturnPart(tool_name=tool_name, content=content, tool_call_id=tool_call_id))
        elif ptype in ('binary', 'blob'):
            # `binary` is the v2/v3 inline-binary part; `blob` is its v4+ equivalent.
            result.append(FilePart(content=_binary_from_otel(part)))
    return result


# ── Legacy v1 events → ModelMessages ─────────────────────────────────


def _legacy_events_to_model_messages(
    events: list[dict[str, Any]],
) -> list[ModelMessage]:
    """Convert legacy v1 OTEL events to ModelMessages.

    Legacy events have `event.name` keys like `gen_ai.system.message`,
    `gen_ai.user.message`, `gen_ai.assistant.message`, `gen_ai.tool.message`,
    and `gen_ai.choice`.
    """
    result: list[ModelMessage] = []
    pending_request_parts: list[ModelRequestPart] = []

    def event_group_key(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
        position, event = item
        index = event.get('gen_ai.message.index')
        return (1, index) if isinstance(index, int) else (0, position)

    keyed_events = list(enumerate(events))
    if all(isinstance(event.get('gen_ai.message.index'), int) for event in events):
        keyed_events.sort(key=event_group_key)
    for _, event_group in itertools.groupby(keyed_events, key=event_group_key):
        event_list = [event for _, event in event_group]
        first_event = event_list[0]
        event_name = first_event.get('event.name', '')

        if event_name in ('gen_ai.choice', 'gen_ai.assistant.message'):
            # Flush pending request before an assistant/choice message
            if pending_request_parts:
                result.append(ModelRequest(parts=pending_request_parts))
                pending_request_parts = []

            response = _convert_legacy_response_events(event_name, event_list)
            if response is not None:
                result.append(response)
        else:
            pending_request_parts.extend(_convert_legacy_request_events(event_name, event_list))

    if pending_request_parts:
        result.append(ModelRequest(parts=pending_request_parts))

    return result


def _convert_legacy_response_events(event_name: str, event_list: list[dict[str, Any]]) -> ModelResponse | None:
    """Convert legacy assistant/choice events to a ModelResponse."""
    response_parts: list[ModelResponsePart] = []
    if event_name == 'gen_ai.choice':
        message_body = event_list[0].get('message', event_list[0])
        response_parts.extend(_convert_legacy_message_parts(message_body))
    else:
        for event in event_list:
            response_parts.extend(_convert_legacy_message_parts(event))
    return ModelResponse(parts=response_parts) if response_parts else None


def _convert_legacy_request_events(event_name: str, event_list: list[dict[str, Any]]) -> list[ModelRequestPart]:
    """Convert legacy system/user/tool events to ModelRequestParts."""
    parts: list[ModelRequestPart] = []
    first_event = event_list[0]

    if event_name == 'gen_ai.system.message':
        content = first_event.get('content', '')
        if isinstance(content, str) and content:
            parts.append(SystemPromptPart(content))
    elif event_name == 'gen_ai.user.message':
        content = first_event.get('content', '')
        if _utils.is_str_dict(content) and content.get('kind') == 'text':
            parts.append(UserPromptPart(content.get('text', '')))
        elif content:
            parts.append(UserPromptPart(content if isinstance(content, str) else str(content)))
    elif event_name == 'gen_ai.tool.message':
        for event in event_list:
            tool_name = event.get('name', '')
            tool_call_id = event.get('id', _utils.generate_tool_call_id())
            content = event.get('content', '')
            parts.append(ToolReturnPart(tool_name=tool_name, content=content, tool_call_id=tool_call_id))

    return parts


def _convert_legacy_message_parts(body: dict[str, Any]) -> list[ModelResponsePart]:
    """Convert a legacy assistant/choice message body to response parts.

    Shared by `gen_ai.choice` and `gen_ai.assistant.message` events, which use the same
    shape: `content` (a string or a list of `{kind, text}` items) plus optional
    OpenAI-style `tool_calls`.
    """
    parts: list[ModelResponsePart] = []
    if 'content' in body:
        content = body['content']
        if isinstance(content, str):
            parts.append(TextPart(content))
        elif isinstance(content, list):
            _extend_from_legacy_content_list(parts, cast(list[dict[str, Any]], content))

    tool_calls: list[dict[str, Any]] = body.get('tool_calls', [])
    for tool_call in tool_calls:
        tc_id: str = tool_call.get('id', _utils.generate_tool_call_id())
        func: dict[str, Any] = tool_call.get('function', {})
        tc_name: str = func.get('name', '')
        tc_args: str | dict[str, Any] | None = func.get('arguments')
        parts.append(ToolCallPart(tool_name=tc_name, args=tc_args, tool_call_id=tc_id))
    return parts


def _extend_from_legacy_content_list(parts: list[ModelResponsePart], content: list[dict[str, Any]]) -> None:
    """Parse a legacy content list (`[{kind, text}, ...]`) and extend parts."""
    for item in content:
        kind: str = item.get('kind', '')
        text: str = item.get('text', '')
        if kind == 'text':
            parts.append(TextPart(text))
        elif kind == 'thinking':
            parts.append(ThinkingPart(text))
