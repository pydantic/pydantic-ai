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
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from typing_extensions import assert_never

from ..exceptions import UserError
from ..messages import (
    BinaryContent,
    CompactionPart,
    FilePart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponsePart,
    NativeToolCallPart,
    NativeToolReturnPart,
    RetryPromptPart,
    SpeechPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from ..profiles import DEFAULT_THINKING_TAGS
from ..settings import ToolChoice
from ..tools import ToolDefinition
from ._base import (
    AudioDelta,
    InputSpeechEndEvent,
    InputSpeechStartEvent,
    InputTranscript,
    RealtimeCodecEvent,
    RealtimeModelProfile,
    SessionErrorEvent,
    ToolCall,
    Transcript,
    TurnCompleteEvent,
    TurnDetection,
    seed_pcm_audio,
    seed_speech_content,
    seed_user_content,
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


AUTO_TRANSCRIPTION_MODEL = 'auto'
"""Sentinel `input_transcription_model` value: resolve to the provider's recommended transcription model."""


def resolve_transcription_model(value: str | None, *, default: str) -> str | None:
    """Resolve an `input_transcription_model` setting to a concrete model id, or `None` to disable transcription.

    `'auto'` (the always-supported default) resolves to `default`, the provider's current recommended
    realtime transcription model. Keeping the public default as a stable sentinel — rather than a hardcoded
    model id — lets the concrete model it maps to change over releases without silently altering the
    behavior of apps that pinned a specific id. Any other string is used verbatim; `None` disables
    transcription (no user transcripts; see `audio_retention` to retain the raw audio instead).
    """
    if value == AUTO_TRANSCRIPTION_MODEL:
        return default
    return value


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


async def seed_items(
    messages: Sequence[ModelMessage], *, profile: RealtimeModelProfile, provider_name: str
) -> list[dict[str, Any]]:
    """Map prior history to OpenAI-protocol `conversation.item.create` items.

    Text, transcripts, images, retained user audio, thinking, and function-tool rounds are replayed in
    part order. Thinking becomes tag-wrapped assistant text; its signature and `provider_details` are
    provider-session-bound and are not replayed. Native-tool parts are metadata about how an answer was
    produced and are skipped while the answer itself is retained. `SystemPromptPart`s are routed through
    the session `instructions` field, and `CachePoint`s are ignored.

    Retained user audio is decoded by the new session's configured input-audio format. Assistant audio
    cannot be inserted by the API, so assistant speech requires a transcript. Any other content that
    cannot be represented faithfully raises [`UserError`][pydantic_ai.exceptions.UserError].
    """
    items: list[dict[str, Any]] = []
    call_ids: dict[str, str] = {}
    seeded_calls: set[str] = set()
    supports_images = profile.get('supports_seeding_images', False)
    supports_audio = profile.get('supports_seeding_audio', False)

    for message in messages:
        if isinstance(message, ModelRequest):
            items.extend(
                await _seed_request_items(
                    message.parts,
                    provider_name=provider_name,
                    supports_images=supports_images,
                    supports_audio=supports_audio,
                    audio_input_sample_rate=profile.get('audio_input_sample_rate', 24000),
                    call_ids=call_ids,
                    seeded_calls=seeded_calls,
                )
            )
        else:
            items.extend(
                _seed_response_items(
                    message.parts,
                    provider_name=provider_name,
                    supports_audio=supports_audio,
                    call_ids=call_ids,
                    seeded_calls=seeded_calls,
                )
            )
    return items


async def _seed_request_items(
    parts: Sequence[ModelRequestPart],
    *,
    provider_name: str,
    supports_images: bool,
    supports_audio: bool,
    audio_input_sample_rate: int,
    call_ids: dict[str, str],
    seeded_calls: set[str],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, SystemPromptPart):
            continue
        elif isinstance(part, UserPromptPart):
            if content := _user_content_items(
                await seed_user_content(part, provider_name=provider_name, supports_images=supports_images)
            ):
                items.append(_message_item('user', content))
        elif isinstance(part, SpeechPart):
            content = seed_speech_content(part, provider_name=provider_name, supports_audio=supports_audio)
            if isinstance(content, str):
                if content:
                    items.append(_message_item('user', [_text_content('input_text', content)]))
            else:
                pcm = seed_pcm_audio(
                    content,
                    provider_name=provider_name,
                    sample_rate=audio_input_sample_rate,
                )
                items.append(_message_item('user', [{'type': 'input_audio', 'audio': base64.b64encode(pcm).decode()}]))
        elif isinstance(part, ToolReturnPart):
            _require_seeded_call(part.tool_name, part.tool_call_id, seeded_calls)
            output, user_content = part.model_response_str_and_user_content()
            items.append(
                {
                    'type': 'function_call_output',
                    'call_id': _seed_call_id(part.tool_call_id, call_ids),
                    'output': output,
                }
            )
            if user_content and (
                content := _user_content_items(
                    await seed_user_content(
                        UserPromptPart(content=user_content),
                        provider_name=provider_name,
                        supports_images=supports_images,
                    )
                )
            ):
                items.append(_message_item('user', content))
        elif isinstance(part, RetryPromptPart):
            output = part.model_response()
            if part.tool_name is None:
                items.append(_message_item('user', [_text_content('input_text', output)]))
            else:
                _require_seeded_call(part.tool_name, part.tool_call_id, seeded_calls)
                items.append(
                    {
                        'type': 'function_call_output',
                        'call_id': _seed_call_id(part.tool_call_id, call_ids),
                        'output': output,
                    }
                )
        else:
            assert_never(part)
    return items


def _seed_response_items(
    parts: Sequence[ModelResponsePart],
    *,
    provider_name: str,
    supports_audio: bool,
    call_ids: dict[str, str],
    seeded_calls: set[str],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, TextPart):
            if part.content:
                items.append(_message_item('assistant', [_text_content('output_text', part.content)]))
        elif isinstance(part, ThinkingPart):
            if part.content:
                start_tag, end_tag = DEFAULT_THINKING_TAGS
                text = '\n'.join([start_tag, part.content, end_tag])
                items.append(_message_item('assistant', [_text_content('output_text', text)]))
        elif isinstance(part, ToolCallPart):
            call_id = _seed_call_id(part.tool_call_id, call_ids)
            seeded_calls.add(part.tool_call_id)
            items.append(
                {
                    'type': 'function_call',
                    'name': part.tool_name,
                    'call_id': call_id,
                    'arguments': part.args_as_json_str(),
                }
            )
        elif isinstance(part, (NativeToolCallPart, NativeToolReturnPart)):
            continue
        elif isinstance(part, SpeechPart):
            content = seed_speech_content(part, provider_name=provider_name, supports_audio=supports_audio)
            if content:
                assert isinstance(content, str)
                items.append(_message_item('assistant', [_text_content('output_text', content)]))
        elif isinstance(part, CompactionPart):
            # Provider-session-bound compaction state can't round-trip into another session; classic
            # model adapters skip it when crossing APIs (e.g. Chat Completions), and seeding matches.
            continue
        elif isinstance(part, FilePart):
            raise UserError(
                f'`FilePart` cannot be seeded into {provider_name} realtime history. '
                'Convert it to text or filter it from `message_history` before connecting.'
            )
        else:
            assert_never(part)
    return items


def _seed_call_id(tool_call_id: str, call_ids: dict[str, str]) -> str:
    """Return a stable wire ID no longer than the OpenAI protocol's 32-character limit."""
    if wire_id := call_ids.get(tool_call_id):
        return wire_id
    wire_id = tool_call_id if len(tool_call_id) <= 32 else hashlib.sha256(tool_call_id.encode()).hexdigest()[:32]
    call_ids[tool_call_id] = wire_id
    return wire_id


def _require_seeded_call(tool_name: str, tool_call_id: str, seeded_calls: set[str]) -> None:
    if tool_call_id not in seeded_calls:
        raise UserError(
            f'Cannot seed output for tool {tool_name!r} with call ID {tool_call_id!r}: no preceding '
            '`ToolCallPart` with that ID was included in `message_history`.'
        )


def _user_content_items(content: Sequence[str | BinaryContent]) -> list[dict[str, Any]]:
    return [
        _text_content('input_text', item)
        if isinstance(item, str)
        else {'type': 'input_image', 'image_url': item.data_uri}
        for item in content
        if not isinstance(item, str) or item
    ]


def _text_content(content_type: Literal['input_text', 'output_text'], text: str) -> dict[str, Any]:
    return {'type': content_type, 'text': text}


def _message_item(role: Literal['user', 'assistant'], content: list[dict[str, Any]]) -> dict[str, Any]:
    return {'type': 'message', 'role': role, 'content': content}


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


def _is_function_call_only(output: Any) -> bool:
    """Whether a `response.done` output list contains only function calls."""
    entries = cast('list[Any]', output)
    if not isinstance(entries, list):
        return False
    return bool(entries) and all(obj(entry).get('type') == 'function_call' for entry in entries)


def _map_response_done(data: dict[str, Any]) -> RealtimeCodecEvent | None:
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


def map_event(data: dict[str, Any]) -> RealtimeCodecEvent | None:
    """Map a raw OpenAI Realtime event to a [`RealtimeCodecEvent`][pydantic_ai.realtime.RealtimeCodecEvent].

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
            response_usage_follows=True,
        )

    if event_type == 'input_audio_buffer.speech_started':
        return InputSpeechStartEvent()

    if event_type == 'input_audio_buffer.speech_stopped':
        return InputSpeechEndEvent()

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


def server_vad_from_turn_detection(turn_detection: TurnDetection) -> ServerVAD:
    """Map cross-provider turn detection to the OpenAI-compatible server-VAD shape."""
    threshold = (
        {'low': 0.7, 'medium': 0.5, 'high': 0.3}[turn_detection.sensitivity]
        if turn_detection.sensitivity is not None
        else None
    )
    return ServerVAD(
        threshold=threshold,
        prefix_padding_ms=turn_detection.prefix_padding_ms,
        silence_duration_ms=turn_detection.silence_duration_ms,
    )


def resolve_base_turn_detection(base: bool | TurnDetection) -> ServerVAD | None:
    """Resolve a cross-provider `turn_detection` value to a server-VAD config (or `None` to disable).

    `True` (or an absent setting, handled by the caller) uses the provider defaults; `False` disables
    detection (push-to-talk); a [`TurnDetection`][pydantic_ai.realtime.TurnDetection] maps its knobs.
    """
    if base is False:
        return None
    if base is True:
        return ServerVAD()
    return server_vad_from_turn_detection(base)


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
