"""Shared codec and helpers for the OpenAI Realtime wire protocol.

The OpenAI Realtime API defines a WebSocket wire protocol that other providers clone verbatim — xAI's
Grok Voice realtime API ([`pydantic_ai.realtime.xai`][pydantic_ai.realtime.xai]) is a deliberate copy.
This module holds the provider-agnostic pieces of that protocol so both
[`pydantic_ai.realtime.openai`][pydantic_ai.realtime.openai] and the xAI provider can share them
without reaching into each other's internals: the WebSocket-URL derivation, session seeding, tool-def
conversion, turn-detection/tool-choice config builders, the handshake helper, and the event mapper.

The names below have no leading underscore because they are imported across the `openai`/`xai` provider
modules (they are still private to the package: the module itself is underscore-prefixed). Helpers used
only within this module keep their underscore prefix. The stateful connection and model classes live in
the provider modules, not here.
"""

from __future__ import annotations as _annotations

import asyncio
import base64
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from ..messages import (
    ModelMessage,
    ModelRequest,
    SpeechPart,
    TextPart,
    UserPromptPart,
)
from ..settings import ToolChoice
from ..tools import ToolDefinition
from ._base import (
    AudioDelta,
    InputTranscript,
    RateLimit,
    RateLimitsEvent,
    RealtimeEvent,
    SessionErrorEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    ToolCall,
    Transcript,
    TurnCompleteEvent,
    user_prompt_text,
)

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection


def realtime_websocket_url(base_url: str) -> str:
    """Derive the realtime WebSocket URL from a provider's HTTP base URL.

    Swaps the HTTP scheme for the WebSocket one and appends the `realtime` path, so the default
    OpenAI base URL `https://api.openai.com/v1/` yields `wss://api.openai.com/v1/realtime`.
    """
    url = base_url.rstrip('/')
    if url.startswith('https://'):
        url = 'wss://' + url[len('https://') :]
    elif url.startswith('http://'):
        url = 'ws://' + url[len('http://') :]
    return f'{url}/realtime'


# The OpenAI event names differ between the GA and beta realtime surfaces; accept both.
AUDIO_DELTA_TYPES = frozenset({'response.output_audio.delta', 'response.audio.delta'})
_AUDIO_TRANSCRIPT_DELTA_TYPES = frozenset({'response.output_audio_transcript.delta', 'response.audio_transcript.delta'})
_AUDIO_TRANSCRIPT_DONE_TYPES = frozenset({'response.output_audio_transcript.done', 'response.audio_transcript.done'})
_INPUT_TRANSCRIPT_DONE_TYPES = frozenset({'conversation.item.input_audio_transcription.completed'})
_FUNCTION_CALL_DONE_TYPES = frozenset({'response.function_call_arguments.done'})


def tool_def_to_openai(tool: ToolDefinition) -> dict[str, Any]:
    """Convert a [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] to the OpenAI realtime tool format."""
    result: dict[str, Any] = {
        'type': 'function',
        'name': tool.name,
        'parameters': tool.parameters_json_schema,
    }
    if tool.description:
        result['description'] = tool.description
    return result


def seed_items(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
    """Project prior conversation to OpenAI `conversation.item.create` items (text/transcript only, v1).

    User prompts and user-spoken transcripts become `input_text` user items; assistant text and
    assistant-spoken transcripts become `output_text` assistant items. `SystemPromptPart`s are skipped (the
    `instructions` session field covers system-level guidance), and tool calls/results are skipped —
    seeding a `function_call_output` without its originating call item is invalid, and full tool-round
    replay is out of scope for v1. Content that can't be projected is dropped rather than erroring.
    """
    items: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for req_part in message.parts:
                if isinstance(req_part, UserPromptPart) and (text := user_prompt_text(req_part)):
                    items.append(_message_item('user', 'input_text', text))
                elif isinstance(req_part, SpeechPart) and req_part.transcript:
                    items.append(_message_item('user', 'input_text', req_part.transcript))
        else:
            for resp_part in message.parts:
                if isinstance(resp_part, TextPart) and resp_part.content:
                    items.append(_message_item('assistant', 'output_text', resp_part.content))
                elif isinstance(resp_part, SpeechPart) and resp_part.transcript:
                    items.append(_message_item('assistant', 'output_text', resp_part.transcript))
    return items


def _message_item(role: str, content_type: str, text: str) -> dict[str, Any]:
    return {'type': 'message', 'role': role, 'content': [{'type': content_type, 'text': text}]}


def _str_field(data: dict[str, Any], key: str, default: str = '') -> str:
    """Return `data[key]` if it is a string, otherwise `default`."""
    value = data.get(key, default)
    return value if isinstance(value, str) else default


def obj(value: Any) -> dict[str, Any]:
    """Return `value` as a `dict[str, Any]` when it is a mapping, otherwise an empty dict."""
    return cast('dict[str, Any]', value) if isinstance(value, dict) else {}


def loads_obj(raw: str) -> dict[str, Any]:
    """Parse a JSON text frame into an object, raising `ValueError` if it decodes to a non-object.

    `json.loads` can return arrays, strings, or numbers; those aren't valid realtime frames, so treat
    them as malformed (a `ValueError`, like a decode error) rather than letting a later `.get()` raise
    `AttributeError` and escape the recoverable-error handling.
    """
    data: Any = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f'expected a JSON object, got {type(data).__name__}')
    return cast('dict[str, Any]', data)


def _map_rate_limits(data: dict[str, Any]) -> RateLimitsEvent:
    """Map a `rate_limits.updated` event to a [`RateLimitsEvent`][pydantic_ai.realtime.RateLimitsEvent]."""
    entries = data.get('rate_limits')
    limits: list[RateLimit] = []
    for entry in cast('list[Any]', entries) if isinstance(entries, list) else []:
        item = obj(entry)
        name = item.get('name')
        if not isinstance(name, str):
            continue
        reset = item.get('reset_seconds')
        limits.append(
            RateLimit(
                name=name,
                limit=item.get('limit') if isinstance(item.get('limit'), int) else None,
                remaining=item.get('remaining') if isinstance(item.get('remaining'), int) else None,
                reset_seconds=float(reset) if isinstance(reset, (int, float)) and not isinstance(reset, bool) else None,
            )
        )
    return RateLimitsEvent(limits=limits)


def _is_function_call_only(output: Any) -> bool:
    """Whether a `response.done` output list contains only function calls."""
    entries = cast('list[Any]', output)
    if not isinstance(entries, list):
        return False
    return bool(entries) and all(obj(entry).get('type') == 'function_call' for entry in entries)


def _map_response_done(data: dict[str, Any]) -> RealtimeEvent | None:
    """Map a `response.done` event, returning `None` for function-call-only responses.

    A response whose only output is function calls is an intermediate step: the session executes the
    tools and the model emits a further `response.done` with the actual answer. Surfacing a
    `TurnCompleteEvent` here would prematurely signal the end of the turn.
    """
    if not isinstance(data.get('response'), dict):
        return TurnCompleteEvent(interrupted=False)
    response = obj(data.get('response'))
    output = response.get('output')
    if _is_function_call_only(output):
        return None
    return TurnCompleteEvent(interrupted=response.get('status') == 'cancelled')


def map_event(data: dict[str, Any]) -> RealtimeEvent | None:
    """Map a raw OpenAI Realtime event to a [`RealtimeEvent`][pydantic_ai.realtime.RealtimeEvent].

    Returns `None` for events that carry no session-relevant content (e.g. `session.created`).
    """
    event_type = data.get('type')

    if event_type in AUDIO_DELTA_TYPES:
        delta = data.get('delta')
        if not isinstance(delta, str):
            return None
        return AudioDelta(data=base64.b64decode(delta))

    if event_type in _AUDIO_TRANSCRIPT_DELTA_TYPES:
        return Transcript(text=_str_field(data, 'delta'), is_final=False)

    if event_type in _AUDIO_TRANSCRIPT_DONE_TYPES:
        return Transcript(text=_str_field(data, 'transcript'), is_final=True)

    if event_type == 'response.output_text.delta':
        return Transcript(text=_str_field(data, 'delta'), is_final=False, output_text=True)

    if event_type == 'response.output_text.done':
        return Transcript(text=_str_field(data, 'text'), is_final=True, output_text=True)

    if event_type == 'conversation.item.input_audio_transcription.delta':
        return InputTranscript(text=_str_field(data, 'delta'), is_final=False)

    if event_type in _INPUT_TRANSCRIPT_DONE_TYPES:
        return InputTranscript(text=_str_field(data, 'transcript'), is_final=True)

    if event_type in _FUNCTION_CALL_DONE_TYPES:
        return ToolCall(
            tool_call_id=_str_field(data, 'call_id'),
            tool_name=_str_field(data, 'name'),
            args=_str_field(data, 'arguments', '{}'),
        )

    if event_type == 'input_audio_buffer.speech_started':
        return SpeechStartedEvent()

    if event_type == 'input_audio_buffer.speech_stopped':
        return SpeechStoppedEvent()

    if event_type == 'rate_limits.updated':
        return _map_rate_limits(data)

    if event_type == 'response.done':
        return _map_response_done(data)

    if event_type == 'error':
        error = obj(data.get('error'))
        return SessionErrorEvent(
            message=_error_message(data.get('error')),
            type=_str_field(error, 'type') or None,
            code=_str_field(error, 'code') or None,
            recoverable=True,  # a protocol `error` keeps the session open; a dropped connection does not
        )

    return None


def _error_message(error: Any) -> str:
    """Extract a human-readable message from an OpenAI `error` payload."""
    if isinstance(error, dict):
        message = obj(error).get('message')
        return message if isinstance(message, str) and message else json.dumps(obj(error))
    return str(error)


@dataclass
class ServerVAD:
    """Server-side voice activity detection — the default turn-taking mode.

    The server detects when the user starts and stops speaking and (by default) commits the audio
    and triggers a response automatically. Unset fields fall back to the provider defaults.
    """

    threshold: float | None = None
    """Activation threshold (0.0-1.0). Higher requires louder audio; better in noisy environments."""
    prefix_padding_ms: int | None = None
    """Audio to include before detected speech, in milliseconds."""
    silence_duration_ms: int | None = None
    """Silence required to detect the end of speech, in milliseconds."""
    create_response: bool = True
    """Whether to automatically generate a response when the user stops speaking."""
    interrupt_response: bool = True
    """Whether to automatically interrupt an in-progress response when the user starts speaking."""
    idle_timeout_ms: int | None = None
    """If set, auto-trigger a response after this much idle time with no detected speech."""


@dataclass
class SemanticVAD:
    """Model-based semantic turn detection — uses a model to decide when the user is done speaking."""

    eagerness: Literal['low', 'medium', 'high', 'auto'] = 'auto'
    """How eagerly the model responds. `low` waits longer for the user; `high` responds sooner."""
    create_response: bool = True
    """Whether to automatically generate a response when a turn ends."""
    interrupt_response: bool = True
    """Whether to automatically interrupt an in-progress response when the user starts speaking."""


def turn_detection_config(turn_detection: ServerVAD | SemanticVAD | None) -> dict[str, Any] | None:
    """Build the OpenAI `turn_detection` payload, or `None` to disable VAD (manual turn-taking)."""
    if turn_detection is None:
        return None
    if isinstance(turn_detection, ServerVAD):
        config: dict[str, Any] = {
            'type': 'server_vad',
            'create_response': turn_detection.create_response,
            'interrupt_response': turn_detection.interrupt_response,
        }
        if turn_detection.threshold is not None:
            config['threshold'] = turn_detection.threshold
        if turn_detection.prefix_padding_ms is not None:
            config['prefix_padding_ms'] = turn_detection.prefix_padding_ms
        if turn_detection.silence_duration_ms is not None:
            config['silence_duration_ms'] = turn_detection.silence_duration_ms
        if turn_detection.idle_timeout_ms is not None:
            config['idle_timeout_ms'] = turn_detection.idle_timeout_ms
        return config
    return {
        'type': 'semantic_vad',
        'eagerness': turn_detection.eagerness,
        'create_response': turn_detection.create_response,
        'interrupt_response': turn_detection.interrupt_response,
    }


def tool_choice_config(tool_choice: ToolChoice) -> str | dict[str, Any] | None:
    """Map a pydantic-ai `tool_choice` to the OpenAI realtime `tool_choice` field.

    Realtime can't express a multi-tool restriction, so a multi-element allow-list or a
    `ToolOrOutput` is dropped (the model's default applies). A single-element allow-list forces
    that one function.
    """
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):  # 'auto' | 'required' | 'none'
        return tool_choice
    if isinstance(tool_choice, list) and len(tool_choice) == 1:
        return {'type': 'function', 'name': tool_choice[0]}
    return None  # multi-tool restriction / ToolOrOutput: not expressible in realtime


async def expect_event(ws: ClientConnection, expected_type: str, *, timeout: float) -> dict[str, Any]:
    """Read events until one of `expected_type` arrives, raising on a server error or timeout.

    Unrelated events received during the handshake (e.g. rate limit notices) are skipped rather than
    treated as a protocol violation. `timeout` bounds the total wait so `connect()` fails predictably
    instead of hanging if the expected event never arrives.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=max(0.0, deadline - time.monotonic()))
        except asyncio.TimeoutError:
            raise TimeoutError(f'Timed out waiting for OpenAI realtime {expected_type!r} event') from None
        if not isinstance(raw, str):  # pragma: no cover
            raise TypeError(f'Expected a text message from the WebSocket, got {type(raw).__name__}')
        data = loads_obj(raw)
        event_type = data.get('type')
        if event_type == expected_type:
            return data
        if event_type == 'error':
            raise RuntimeError(f'OpenAI realtime error during handshake: {_error_message(data.get("error"))}')
