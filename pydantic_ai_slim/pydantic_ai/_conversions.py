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

from . import _utils
from .messages import (
    AudioUrl,
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    CachePoint,
    DocumentUrl,
    FilePart,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
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

    - ``pydantic_ai.all_messages`` attribute on agent run spans
    - ``gen_ai.input.messages`` / ``gen_ai.output.messages`` attributes on model request spans

    Also supports the legacy v1 events format (with ``event.name`` keys).

    Note: this conversion is lossy. Some information (e.g. timestamps, ``instructions``,
    provider details) is not preserved in the OTEL format and will use defaults.
    Content excluded by ``include_content=False`` will be empty strings.

    Args:
        otel_messages: A JSON string or a list of message dicts.

    Returns:
        A list of ModelMessage objects that can be passed as ``message_history``
        to :meth:`Agent.run <pydantic_ai.agent.AbstractAgent.run>`.
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
    `OpenAI Chat Completions API <https://platform.openai.com/docs/api-reference/chat>`_.

    Each :class:`~pydantic_ai.messages.ModelRequest` is expanded into one or more messages
    (system, user, tool), and each :class:`~pydantic_ai.messages.ModelResponse` becomes an
    assistant message with optional ``tool_calls``.

    Note: :class:`~pydantic_ai.messages.ThinkingPart`, :class:`~pydantic_ai.messages.FilePart`,
    :class:`~pydantic_ai.messages.BuiltinToolCallPart`, and
    :class:`~pydantic_ai.messages.BuiltinToolReturnPart` are not included in the output
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
        elif isinstance(message, ModelResponse):
            if openai_message := _model_response_to_openai(message):
                result.append(openai_message)

    return result


# ── ChatMessage format → ModelMessages ────────────────────────────────


def _chat_messages_to_model_messages(
    chat_messages: list[dict[str, Any]],
) -> list[ModelMessage]:
    """Convert ChatMessage format (``{role, parts}``) to ModelMessages.

    Merges consecutive non-assistant messages into a single ModelRequest,
    reversing the split performed by ``messages_to_otel_messages``.
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


def _make_user_prompt_part(parts: list[dict[str, Any]]) -> UserPromptPart:
    """Create a UserPromptPart from a list of OTEL message parts."""
    if len(parts) == 1 and parts[0].get('type') == 'text':
        return UserPromptPart(parts[0].get('content', ''))

    content: list[UserContent] = []
    for part in parts:
        ptype = part.get('type', '')
        if ptype == 'text':
            content.append(part.get('content', ''))
        elif ptype == 'image-url':
            url = part.get('url', '')
            if url:
                content.append(ImageUrl(url))
        elif ptype == 'audio-url':
            url = part.get('url', '')
            if url:
                content.append(AudioUrl(url))
        elif ptype == 'video-url':
            url = part.get('url', '')
            if url:
                content.append(VideoUrl(url))
        elif ptype == 'document-url':
            url = part.get('url', '')
            if url:
                content.append(DocumentUrl(url))
        elif ptype == 'binary':
            media_type = part.get('media_type', 'application/octet-stream')
            b64_content = part.get('content', '')
            data = base64.b64decode(b64_content) if b64_content else b''
            content.append(BinaryContent(data=data, media_type=media_type))

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
                result.append(BuiltinToolCallPart(tool_name=tool_name, args=args, tool_call_id=tool_call_id))
            else:
                result.append(ToolCallPart(tool_name=tool_name, args=args, tool_call_id=tool_call_id))
        elif ptype == 'tool_call_response':
            # Builtin tool returns can appear in assistant messages
            builtin = part.get('builtin', False)
            if builtin:
                tool_name = part.get('name', '')
                tool_call_id = part.get('id', _utils.generate_tool_call_id())
                content = part.get('result', part.get('response', ''))
                result.append(BuiltinToolReturnPart(tool_name=tool_name, content=content, tool_call_id=tool_call_id))
        elif ptype == 'binary':
            media_type = part.get('media_type', 'application/octet-stream')
            b64_content = part.get('content', '')
            data = base64.b64decode(b64_content) if b64_content else b''
            result.append(FilePart(content=BinaryContent(data=data, media_type=media_type)))
    return result


# ── Legacy v1 events → ModelMessages ─────────────────────────────────


def _legacy_events_to_model_messages(
    events: list[dict[str, Any]],
) -> list[ModelMessage]:
    """Convert legacy v1 OTEL events to ModelMessages.

    Legacy events have ``event.name`` keys like ``gen_ai.system.message``,
    ``gen_ai.user.message``, ``gen_ai.assistant.message``, ``gen_ai.tool.message``,
    and ``gen_ai.choice``.
    """
    result: list[ModelMessage] = []
    pending_request_parts: list[ModelRequestPart] = []

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
    response_parts: list[ModelResponsePart]
    if event_name == 'gen_ai.choice':
        message_body = event_list[0].get('message', event_list[0])
        response_parts = _convert_legacy_choice(message_body)
    else:
        response_parts = []
        for event in event_list:
            response_parts.extend(_convert_legacy_assistant_event(event))
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


def _convert_legacy_choice(message_body: dict[str, Any]) -> list[ModelResponsePart]:
    """Convert a ``gen_ai.choice`` event body to response parts."""
    parts: list[ModelResponsePart] = []
    if 'content' not in message_body:
        return parts

    content = message_body['content']
    if isinstance(content, str):
        parts.append(TextPart(content))
    elif isinstance(content, list):
        _extend_from_legacy_content_list(parts, cast(list[dict[str, Any]], content))
    return parts


def _convert_legacy_assistant_event(event: dict[str, Any]) -> list[ModelResponsePart]:
    """Convert a single legacy ``gen_ai.assistant.message`` event to response parts."""
    parts: list[ModelResponsePart] = []

    if 'content' in event:
        content = event['content']
        if isinstance(content, str):
            parts.append(TextPart(content))
        elif isinstance(content, list):
            _extend_from_legacy_content_list(parts, cast(list[dict[str, Any]], content))

    tool_calls: list[dict[str, Any]] = event.get('tool_calls', [])
    for tool_call in tool_calls:
        tc_id: str = tool_call.get('id', _utils.generate_tool_call_id())
        func: dict[str, Any] = tool_call.get('function', {})
        tc_name: str = func.get('name', '')
        tc_args: str | dict[str, Any] | None = func.get('arguments')
        parts.append(ToolCallPart(tool_name=tc_name, args=tc_args, tool_call_id=tc_id))

    return parts


def _extend_from_legacy_content_list(parts: list[ModelResponsePart], content: list[dict[str, Any]]) -> None:
    """Parse a legacy content list (``[{kind, text}, ...]``) and extend parts."""
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
        elif isinstance(part, RetryPromptPart):
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


def _user_content_to_openai(content: str | Sequence[UserContent]) -> str | list[dict[str, Any]]:
    """Convert UserPromptPart content to OpenAI multimodal content format."""
    if isinstance(content, str):
        return content

    parts: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            parts.append({'type': 'text', 'text': item})
        elif isinstance(item, ImageUrl):
            parts.append({'type': 'image_url', 'image_url': {'url': item.url}})
        elif isinstance(item, BinaryContent):
            if item.is_image:
                parts.append({'type': 'image_url', 'image_url': {'url': item.data_uri}})
            elif item.is_audio:
                try:
                    audio_format = item.format
                except ValueError:
                    # Fall back to a text marker for unknown audio formats.
                    parts.append({'type': 'text', 'text': f'[Audio: {item.media_type}]'})
                else:
                    parts.append({'type': 'input_audio', 'input_audio': {'data': item.base64, 'format': audio_format}})
            else:
                # No standard OpenAI representation for non-image/audio binary
                parts.append({'type': 'text', 'text': f'[Binary: {item.media_type}]'})
        elif isinstance(item, AudioUrl):
            # OpenAI input_audio expects base64, not a URL — fall back to text reference
            parts.append({'type': 'text', 'text': f'[Audio: {item.url}]'})
        elif isinstance(item, (DocumentUrl, VideoUrl)):
            # No standard OpenAI representation for document/video URLs
            parts.append({'type': 'text', 'text': f'[{type(item).__name__}: {item.url}]'})
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
