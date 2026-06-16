"""Utilities for converting between OTEL message formats and pydantic-ai ModelMessages.

These functions support round-tripping messages through OTEL instrumentation:
- Forward: ModelMessage -> OTEL format (handled by InstrumentationSettings)
- Reverse: OTEL format -> ModelMessage (handled by this module)
"""

from __future__ import annotations

import base64
import itertools
import json
from collections.abc import Sequence
from typing import Any, cast

from pydantic import ValidationError

from . import _utils
from .messages import (
    AudioUrl,
    BinaryContent,
    CachePoint,
    DocumentUrl,
    FilePart,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    NativeToolCallPart,
    NativeToolReturnPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UploadedFile,
    UserContent,
    UserPromptPart,
    VideoUrl,
)


def otel_messages_to_model_messages(
    otel_messages: str | Sequence[dict[str, Any]],
) -> list[ModelMessage]:
    """Convert OTEL format messages to pydantic-ai ModelMessages.

    This is the inverse of
    [`InstrumentationSettings.messages_to_otel_messages`][pydantic_ai.agent.InstrumentationSettings.messages_to_otel_messages].

    Supports the ChatMessage format used by pydantic-ai's OTEL instrumentation:

    - `pydantic_ai.all_messages` attribute on agent run spans
    - `gen_ai.input.messages` / `gen_ai.output.messages` attributes on model request spans

    Also supports the legacy v1 events format (with `event.name` keys).

    Multi-modal content is handled across instrumentation versions: the v2/v3 media parts
    (`image-url`/`audio-url`/`video-url`/`document-url`/`binary`) and the v4+ OTEL GenAI parts
    (`uri`/`blob`/`file`). A `file` part is restored to an [`UploadedFile`][pydantic_ai.messages.UploadedFile]
    when the trace carries its `provider_name`; without it (older traces, or `include_content=False`)
    the provider-hosted reference can't be rebuilt, so it's replaced by a text marker noting the missing data.

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


def model_messages_to_openai_format(
    messages: Sequence[ModelMessage],
) -> list[dict[str, Any]]:
    """Convert pydantic-ai ModelMessages to OpenAI chat completion format.

    Returns messages in the format expected by the
    [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat).

    Each [`ModelRequest`][pydantic_ai.messages.ModelRequest] is expanded into one or more messages
    (system, user, tool), and each [`ModelResponse`][pydantic_ai.messages.ModelResponse] becomes an
    assistant message with optional `tool_calls`.

    Note: [`ThinkingPart`][pydantic_ai.messages.ThinkingPart], [`FilePart`][pydantic_ai.messages.FilePart],
    [`NativeToolCallPart`][pydantic_ai.messages.NativeToolCallPart], and
    [`NativeToolReturnPart`][pydantic_ai.messages.NativeToolReturnPart] are not included in the output
    as they have no standard OpenAI representation. Assistant messages that only contain
    unsupported parts are omitted.

    Args:
        messages: A sequence of ModelMessage objects.

    Returns:
        A list of message dicts in OpenAI chat completion format.
    """
    result: list[dict[str, Any]] = []

    for message in messages:
        if isinstance(message, ModelRequest):
            result.extend(_model_request_to_openai(message))
        elif openai_message := _model_response_to_openai(message):
            result.append(openai_message)

    return result


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
            if finish_reason := msg.get('finish_reason'):
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
    return BinaryContent(data=data, media_type=media_type)


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
    """Convert a v2+ OTEL GenAI `file` part back to an `UploadedFile`.

    A provider-hosted file reference needs both its `file_id` and the hosting `provider_name`
    (file IDs aren't portable across providers). Traces recorded with `include_content=False`,
    or before pydantic-ai stored `provider_name`, omit these; rather than silently dropping the
    file, fall back to a text marker that flags the missing data.
    """
    file_id = part.get('file_id')
    provider_name = part.get('provider_name')
    media_type = part.get('mime_type', 'application/octet-stream')
    if file_id and provider_name:
        try:
            return UploadedFile(file_id=file_id, provider_name=provider_name, media_type=media_type)
        except ValidationError:
            pass  # unrecognized provider name — fall through to the marker
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

    if len(content) == 1 and isinstance(content[0], str):
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

    # `itertools.groupby` only groups *consecutive* equal keys, so events for a given message must be
    # adjacent. Sort by `gen_ai.message.index` defensively in case a trace store or exporter returned them
    # out of order. This is a stable no-op for events already in emission order (the forward path) and for
    # third-party events that lack the index entirely (all keyed `-1`, original order preserved).
    events = sorted(events, key=lambda e: e.get('gen_ai.message.index', -1))

    for _, event_group in itertools.groupby(events, key=lambda e: e.get('gen_ai.message.index')):
        event_list = list(event_group)
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
        if content:
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


# ── ModelMessages → OpenAI format ────────────────────────────────────


def _model_request_to_openai(request: ModelRequest) -> list[dict[str, Any]]:
    """Convert a ModelRequest to one or more OpenAI format messages."""
    result: list[dict[str, Any]] = []

    for part in request.parts:
        if isinstance(part, SystemPromptPart):
            result.append({'role': 'system', 'content': part.content})
        elif isinstance(part, UserPromptPart):
            result.append({'role': 'user', 'content': _user_content_to_openai(part.content)})
        elif isinstance(part, ToolReturnPart):
            result.append(
                {
                    'role': 'tool',
                    'tool_call_id': part.tool_call_id,
                    'content': part.model_response_str(),
                }
            )
        else:
            # The only remaining `ModelRequestPart` is a `RetryPromptPart`.
            if part.tool_name is not None:
                result.append(
                    {
                        'role': 'tool',
                        'tool_call_id': part.tool_call_id,
                        'content': part.model_response(),
                    }
                )
            else:
                result.append({'role': 'user', 'content': part.model_response()})

    return result


def _openai_image_url(url: str, vendor_metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Build an OpenAI `image_url` dict, preserving the `detail` fidelity hint if present."""
    image_url: dict[str, Any] = {'url': url}
    if vendor_metadata and (detail := vendor_metadata.get('detail')):
        image_url['detail'] = detail
    return image_url


def _binary_content_to_openai(item: BinaryContent) -> dict[str, Any]:
    """Convert a `BinaryContent` item to an OpenAI multimodal content part."""
    if item.is_image:
        return {'type': 'image_url', 'image_url': _openai_image_url(item.data_uri, item.vendor_metadata)}
    if item.is_audio:
        try:
            audio_format = item.format
        except ValueError:
            # Fall back to a text marker for unknown audio formats.
            return {'type': 'text', 'text': f'[Audio: {item.media_type}]'}
        if audio_format in ('wav', 'mp3'):
            return {'type': 'input_audio', 'input_audio': {'data': item.base64, 'format': audio_format}}
        # OpenAI Chat Completions only accepts wav/mp3 audio; fall back to a text marker.
        return {'type': 'text', 'text': f'[Audio: {item.media_type}]'}
    # No standard OpenAI representation for non-image/audio binary
    return {'type': 'text', 'text': f'[Binary: {item.media_type}]'}


def _user_content_to_openai(content: str | Sequence[UserContent]) -> str | list[dict[str, Any]]:
    """Convert UserPromptPart content to OpenAI multimodal content format."""
    if isinstance(content, str):
        return content

    parts: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, str | TextContent):
            parts.append({'type': 'text', 'text': item if isinstance(item, str) else item.content})
        elif isinstance(item, ImageUrl):
            parts.append({'type': 'image_url', 'image_url': _openai_image_url(item.url, item.vendor_metadata)})
        elif isinstance(item, BinaryContent):
            parts.append(_binary_content_to_openai(item))
        elif isinstance(item, AudioUrl):
            # OpenAI input_audio expects base64, not a URL — fall back to text reference
            parts.append({'type': 'text', 'text': f'[Audio: {item.url}]'})
        elif isinstance(item, (DocumentUrl, VideoUrl)):
            # No standard OpenAI representation for document/video URLs
            parts.append({'type': 'text', 'text': f'[{type(item).__name__}: {item.url}]'})
        elif isinstance(item, UploadedFile):
            # OpenAI Chat Completions' `file` type requires base64 data, but `UploadedFile.file_id` is a
            # provider-hosted reference. Fall back to a text marker so the reference isn't silently dropped.
            parts.append({'type': 'text', 'text': f'[UploadedFile: {item.file_id} ({item.provider_name})]'})
        elif isinstance(item, CachePoint):
            pass

    if len(parts) == 1 and parts[0].get('type') == 'text':
        return parts[0]['text']
    return parts


def _model_response_to_openai(response: ModelResponse) -> dict[str, Any] | None:
    """Convert a ModelResponse to an OpenAI format assistant message."""
    content = response.text
    msg: dict[str, Any] = {'role': 'assistant', 'content': content}

    tool_calls: list[dict[str, Any]] = []
    for part in response.parts:
        if isinstance(part, ToolCallPart):
            tool_calls.append(
                {
                    'id': part.tool_call_id,
                    'type': 'function',
                    'function': {
                        'name': part.tool_name,
                        'arguments': part.args_as_json_str(),
                    },
                }
            )

    if tool_calls:
        msg['tool_calls'] = tool_calls

    if content is None and not tool_calls:
        return None

    return msg
