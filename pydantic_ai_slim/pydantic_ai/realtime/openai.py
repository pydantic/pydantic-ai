"""OpenAI Realtime API provider for speech-to-speech sessions.

Connects to `wss://api.openai.com/v1/realtime` over a WebSocket and maps the OpenAI event
protocol to the shared realtime event types.

Requires the `websockets` package, available via the `realtime` optional group:

    pip install "pydantic-ai-slim[realtime]"
"""

from __future__ import annotations as _annotations

import asyncio
import base64
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

try:
    import websockets
    from websockets.asyncio.client import ClientConnection
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `websockets` package to use the OpenAI Realtime model, '
        'you can use the `realtime` and `openai` optional groups - `pip install "pydantic-ai-slim[realtime,openai]"`'
    ) from _import_error

if TYPE_CHECKING:
    # Only needed for typing: the provider supplies the concrete client at runtime, so importing the
    # protocol helpers below (e.g. from the xAI realtime provider) doesn't require the `openai` package.
    from openai import AsyncOpenAI

from ..exceptions import UserError
from ..messages import (
    AudioWithTranscriptPart,
    ModelMessage,
    ModelRequest,
    TextPart,
    UserPromptPart,
)
from ..native_tools import AbstractNativeTool
from ..providers import Provider, infer_provider
from ..settings import ModelSettings, ToolChoice
from ..tools import ToolDefinition
from ..usage import RequestUsage
from ._base import (
    AudioDelta,
    AudioInput,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    CreateResponse,
    ImageInput,
    InputTranscript,
    RateLimit,
    RateLimits,
    RealtimeCapabilities,
    RealtimeConnection,
    RealtimeEvent,
    RealtimeInput,
    RealtimeModel,
    Reconnected,
    ReconnectPolicy,
    SessionError,
    SessionUsage,
    SpeechStarted,
    SpeechStopped,
    TextInput,
    ToolCall,
    ToolResult,
    Transcript,
    TruncateOutput,
    TurnComplete,
    reconnect_with_backoff,
)


def _realtime_websocket_url(base_url: str) -> str:
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
_AUDIO_DELTA_TYPES = frozenset({'response.output_audio.delta', 'response.audio.delta'})
_AUDIO_TRANSCRIPT_DELTA_TYPES = frozenset({'response.output_audio_transcript.delta', 'response.audio_transcript.delta'})
_AUDIO_TRANSCRIPT_DONE_TYPES = frozenset({'response.output_audio_transcript.done', 'response.audio_transcript.done'})
_INPUT_TRANSCRIPT_DONE_TYPES = frozenset({'conversation.item.input_audio_transcription.completed'})
_FUNCTION_CALL_DONE_TYPES = frozenset({'response.function_call_arguments.done'})


def _tool_def_to_openai(tool: ToolDefinition) -> dict[str, Any]:
    """Convert a [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] to the OpenAI realtime tool format."""
    result: dict[str, Any] = {
        'type': 'function',
        'name': tool.name,
        'parameters': tool.parameters_json_schema,
    }
    if tool.description:
        result['description'] = tool.description
    return result


def _user_prompt_text(part: UserPromptPart) -> str:
    """Extract the plain text from a `UserPromptPart` (dropping multimodal content for text seeding)."""
    if isinstance(part.content, str):
        return part.content
    return ''.join(item for item in part.content if isinstance(item, str))


def _seed_items(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
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
                if isinstance(req_part, UserPromptPart) and (text := _user_prompt_text(req_part)):
                    items.append(_message_item('user', 'input_text', text))
                elif isinstance(req_part, AudioWithTranscriptPart) and req_part.transcript:
                    items.append(_message_item('user', 'input_text', req_part.transcript))
        else:
            for resp_part in message.parts:
                if isinstance(resp_part, TextPart) and resp_part.content:
                    items.append(_message_item('assistant', 'output_text', resp_part.content))
                elif isinstance(resp_part, AudioWithTranscriptPart) and resp_part.transcript:
                    items.append(_message_item('assistant', 'output_text', resp_part.transcript))
    return items


def _message_item(role: str, content_type: str, text: str) -> dict[str, Any]:
    return {'type': 'message', 'role': role, 'content': [{'type': content_type, 'text': text}]}


def _str_field(data: dict[str, Any], key: str, default: str = '') -> str:
    """Return `data[key]` if it is a string, otherwise `default`."""
    value = data.get(key, default)
    return value if isinstance(value, str) else default


def _obj(value: Any) -> dict[str, Any]:
    """Return `value` as a `dict[str, Any]` when it is a mapping, otherwise an empty dict."""
    return cast('dict[str, Any]', value) if isinstance(value, dict) else {}


def _int(value: Any) -> int:
    """Return `value` if it is an int (but not a bool), otherwise `0`."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _loads_obj(raw: str) -> dict[str, Any]:
    """Parse a JSON text frame into an object, raising `ValueError` if it decodes to a non-object.

    `json.loads` can return arrays, strings, or numbers; those aren't valid realtime frames, so treat
    them as malformed (a `ValueError`, like a decode error) rather than letting a later `.get()` raise
    `AttributeError` and escape the recoverable-error handling.
    """
    data: Any = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f'expected a JSON object, got {type(data).__name__}')
    return cast('dict[str, Any]', data)


def _map_usage(response: dict[str, Any]) -> RequestUsage | None:
    """Map a `response.done` `usage` payload to a [`RequestUsage`][pydantic_ai.usage.RequestUsage]."""
    usage = _obj(response.get('usage'))
    if not usage:
        return None
    inp = _obj(usage.get('input_token_details'))
    out = _obj(usage.get('output_token_details'))
    cached = _obj(inp.get('cached_tokens_details'))
    details: dict[str, int] = {}
    for key, raw in (
        ('input_text_tokens', inp.get('text_tokens')),
        ('input_image_tokens', inp.get('image_tokens')),
        ('output_text_tokens', out.get('text_tokens')),
    ):
        if isinstance(raw, int) and not isinstance(raw, bool):
            details[key] = raw
    return RequestUsage(
        input_tokens=_int(usage.get('input_tokens')),
        output_tokens=_int(usage.get('output_tokens')),
        input_audio_tokens=_int(inp.get('audio_tokens')),
        cache_read_tokens=_int(inp.get('cached_tokens')),
        cache_audio_read_tokens=_int(cached.get('audio_tokens')),
        output_audio_tokens=_int(out.get('audio_tokens')),
        details=details,
    )


def _map_rate_limits(data: dict[str, Any]) -> RateLimits:
    """Map a `rate_limits.updated` event to a [`RateLimits`][pydantic_ai.realtime.RateLimits]."""
    entries = data.get('rate_limits')
    limits: list[RateLimit] = []
    for entry in cast('list[Any]', entries) if isinstance(entries, list) else []:
        item = _obj(entry)
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
    return RateLimits(limits=limits)


def _is_function_call_only(output: Any) -> bool:
    """Whether a `response.done` output list contains only function calls."""
    entries = cast('list[Any]', output)
    if not isinstance(entries, list):
        return False
    return bool(entries) and all(_obj(entry).get('type') == 'function_call' for entry in entries)


def _map_response_done(data: dict[str, Any]) -> RealtimeEvent | None:
    """Map a `response.done` event, returning `None` for function-call-only responses.

    A response whose only output is function calls is an intermediate step: the session executes the
    tools and the model emits a further `response.done` with the actual answer. Surfacing a
    `TurnComplete` here would prematurely signal the end of the turn.
    """
    if not isinstance(data.get('response'), dict):
        return TurnComplete(interrupted=False)
    response = _obj(data.get('response'))
    output = response.get('output')
    if _is_function_call_only(output):
        return None
    return TurnComplete(interrupted=response.get('status') == 'cancelled')


def map_event(data: dict[str, Any]) -> RealtimeEvent | None:
    """Map a raw OpenAI Realtime event to a [`RealtimeEvent`][pydantic_ai.realtime.RealtimeEvent].

    Returns `None` for events that carry no session-relevant content (e.g. `session.created`).
    """
    event_type = data.get('type')

    if event_type in _AUDIO_DELTA_TYPES:
        delta = data.get('delta')
        if not isinstance(delta, str):
            return None
        return AudioDelta(data=base64.b64decode(delta))

    if event_type in _AUDIO_TRANSCRIPT_DELTA_TYPES:
        return Transcript(text=_str_field(data, 'delta'), is_final=False)

    if event_type in _AUDIO_TRANSCRIPT_DONE_TYPES:
        return Transcript(text=_str_field(data, 'transcript'), is_final=True)

    if event_type == 'response.output_text.delta':
        return Transcript(text=_str_field(data, 'delta'), is_final=False)

    if event_type == 'response.output_text.done':
        return Transcript(text=_str_field(data, 'text'), is_final=True)

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
        return SpeechStarted()

    if event_type == 'input_audio_buffer.speech_stopped':
        return SpeechStopped()

    if event_type == 'rate_limits.updated':
        return _map_rate_limits(data)

    if event_type == 'response.done':
        return _map_response_done(data)

    if event_type == 'error':
        error = _obj(data.get('error'))
        return SessionError(
            message=_error_message(data.get('error')),
            type=_str_field(error, 'type') or None,
            code=_str_field(error, 'code') or None,
            recoverable=True,  # a protocol `error` keeps the session open; a dropped connection does not
        )

    return None


def _error_message(error: Any) -> str:
    """Extract a human-readable message from an OpenAI `error` payload."""
    if isinstance(error, dict):
        message = _obj(error).get('message')
        return message if isinstance(message, str) and message else json.dumps(_obj(error))
    return str(error)


class OpenAIRealtimeConnection(RealtimeConnection):
    """A live WebSocket connection to the OpenAI Realtime API."""

    def __init__(
        self,
        ws: ClientConnection,
        *,
        dial: Callable[[], Awaitable[ClientConnection]] | None = None,
        reconnect: ReconnectPolicy | None = None,
    ) -> None:
        self._ws = ws
        # `dial` re-establishes a fully configured connection; with a `reconnect` policy it is used to
        # recover from a dropped WebSocket.
        self._dial = dial
        self._reconnect = reconnect
        # The Realtime API rejects `response.create` while a response is already being generated.
        # We track that window and defer requests (e.g. a background tool result that lands while the
        # model is mid-answer) until the active response finishes, so the model still announces it.
        self._response_active = False
        self._pending_response = False
        # The current output audio item, tracked from output-audio deltas so a `TruncateOutput` can
        # name it. These are mutated by `__aiter__` and read by `send` from a separate task; under a
        # single cooperative event loop the plain reads/writes are safe and eventually consistent.
        self._current_item_id: str | None = None
        self._current_content_index = 0

    async def send(self, content: RealtimeInput) -> None:
        """Send content to the OpenAI Realtime API.

        Accepts `AudioInput` (PCM16, 24kHz, mono), `TextInput`, `ImageInput`, `ToolResult`, and the
        control verbs `CommitAudio`, `ClearAudio`, `CreateResponse`, and `CancelResponse`.
        """
        if isinstance(content, AudioInput):
            await self._send_event(
                {
                    'type': 'input_audio_buffer.append',
                    'audio': base64.b64encode(content.data).decode('ascii'),
                }
            )
        elif isinstance(content, TextInput):
            await self._send_event(
                {
                    'type': 'conversation.item.create',
                    'item': {
                        'type': 'message',
                        'role': 'user',
                        'content': [{'type': 'input_text', 'text': content.text}],
                    },
                }
            )
            await self._request_response()
        elif isinstance(content, ToolResult):
            await self._send_event(
                {
                    'type': 'conversation.item.create',
                    'item': {
                        'type': 'function_call_output',
                        'call_id': content.tool_call_id,
                        'output': content.output,
                    },
                }
            )
            await self._request_response()
        elif isinstance(content, ImageInput):
            # An image is added as conversation context (like a video frame), not a turn of its own,
            # so it doesn't trigger a response — drive that with audio (VAD) or `CreateResponse`.
            data_uri = f'data:{content.mime_type};base64,{base64.b64encode(content.data).decode("ascii")}'
            await self._send_event(
                {
                    'type': 'conversation.item.create',
                    'item': {
                        'type': 'message',
                        'role': 'user',
                        'content': [{'type': 'input_image', 'image_url': data_uri}],
                    },
                }
            )
        elif isinstance(content, CommitAudio):
            await self._send_event({'type': 'input_audio_buffer.commit'})
        elif isinstance(content, ClearAudio):
            await self._send_event({'type': 'input_audio_buffer.clear'})
        elif isinstance(content, CreateResponse):
            await self._request_response()
        elif isinstance(content, CancelResponse):
            # Only cancel when a response is actually active: with server VAD the provider may have
            # already cancelled on the user's barge-in, and a redundant cancel raises a session error.
            if self._response_active:
                await self._send_event({'type': 'response.cancel'})
            self._response_active = False
            self._pending_response = False
        elif isinstance(content, TruncateOutput):
            # No current output item (e.g. the model wasn't speaking) → nothing to truncate.
            if self._current_item_id is not None:
                await self._send_event(
                    {
                        'type': 'conversation.item.truncate',
                        'item_id': self._current_item_id,
                        'content_index': self._current_content_index,
                        'audio_end_ms': content.audio_end_ms,
                    }
                )
        else:
            raise NotImplementedError(f'OpenAI Realtime does not support {type(content).__name__} input')

    async def _request_response(self) -> None:
        """Ask the model to respond now, or defer until the active response completes."""
        if self._response_active:
            self._pending_response = True
        else:
            self._response_active = True
            await self._send_event({'type': 'response.create'})

    async def _send_event(self, event: dict[str, Any]) -> None:
        await self._ws.send(json.dumps(event))

    def _map_event(self, data: dict[str, Any]) -> RealtimeEvent | None:
        """Map a raw provider frame to a codec event.

        A hook so protocol clones (e.g. the xAI Grok Voice provider) can reuse the whole connection while
        overriding only how frames map to events. Defaults to the OpenAI [`map_event`][pydantic_ai.realtime.openai.map_event].
        """
        return map_event(data)

    async def __aiter__(self) -> AsyncIterator[RealtimeEvent]:
        while True:
            try:
                async for raw in self._ws:
                    if not isinstance(raw, str):
                        continue
                    try:
                        events = await self._decode_frame(raw)
                    except ValueError as e:
                        # A malformed frame (bad JSON or audio payload) shouldn't tear down the whole
                        # session; surface it as a recoverable error and keep reading.
                        yield SessionError(message=f'Failed to parse OpenAI realtime event: {e}', recoverable=True)
                        continue
                    for event in events:
                        yield event
                return  # the upstream iterator ended without dropping
            except websockets.ConnectionClosed as e:
                if self._reconnect is None or self._dial is None:
                    # No reconnect policy: a dropped connection is fatal. Surface it as a
                    # non-recoverable error and end the stream cleanly, rather than raising.
                    yield SessionError(message=f'OpenAI realtime connection closed: {e}', recoverable=False)
                    return
                if await self._try_reconnect():
                    yield Reconnected()
                    continue
                yield SessionError(
                    message=f'OpenAI realtime connection closed; reconnect failed: {e}', recoverable=False
                )
                return

    async def _decode_frame(self, raw: str) -> list[RealtimeEvent]:
        """Parse one text frame into events, updating tracked response state.

        Raises `ValueError` (incl. `json.JSONDecodeError` / `binascii.Error`) on a malformed payload.
        """
        data = _loads_obj(raw)
        event_type = data.get('type')
        events: list[RealtimeEvent] = []
        if event_type == 'response.created':
            self._response_active = True
        elif event_type in _AUDIO_DELTA_TYPES:
            # Track the speaking item so a later `TruncateOutput` can name it.
            item_id = data.get('item_id')
            if isinstance(item_id, str):
                self._current_item_id = item_id
                content_index = data.get('content_index')
                self._current_content_index = content_index if isinstance(content_index, int) else 0
        elif event_type == 'response.done':
            self._response_active = False
            self._current_item_id = None
            # Emit usage for every response (including intermediate function-call-only ones)
            # so the session accounts for all tokens, then defer a pending response if needed.
            usage = _map_usage(_obj(data.get('response')))
            if usage is not None:
                events.append(SessionUsage(usage=usage))
            if self._pending_response:
                self._pending_response = False
                # A cancelled response means the user barged in: a new turn is starting, so
                # don't replay the deferred response over it.
                if _obj(data.get('response')).get('status') != 'cancelled':
                    self._response_active = True
                    await self._send_event({'type': 'response.create'})
        if (event := self._map_event(data)) is not None:
            events.append(event)
        return events

    async def _try_reconnect(self) -> bool:
        """Re-dial with exponential backoff; return whether a new connection was established."""
        assert self._reconnect is not None and self._dial is not None
        return await reconnect_with_backoff(self._reconnect, self._attempt_reconnect)

    async def _attempt_reconnect(self) -> bool:
        assert self._dial is not None
        try:
            self._ws = await self._dial()
        except Exception:
            return False
        self._response_active = False
        self._pending_response = False
        self._current_item_id = None
        return True


@dataclass
class ServerVAD:
    """Server-side voice activity detection — the default turn-taking mode.

    The server detects when the user starts and stops speaking and (by default) commits the audio
    and triggers a response automatically. Unset fields fall back to the provider defaults.
    """

    threshold: float | None = None
    """Activation threshold (0.0–1.0). Higher requires louder audio; better in noisy environments."""
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


def _turn_detection_config(turn_detection: ServerVAD | SemanticVAD | None) -> dict[str, Any] | None:
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


def _tool_choice_config(tool_choice: ToolChoice) -> str | dict[str, Any] | None:
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


@dataclass
class OpenAIRealtimeModel(RealtimeModel):
    """OpenAI Realtime API model.

    Authentication, base URL, and the HTTP/WebSocket client come from a
    [`Provider`][pydantic_ai.providers.Provider], mirroring [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel].
    Pass `provider='openai'` (the default) to read `OPENAI_API_KEY` / `OPENAI_BASE_URL` from the
    environment, or an [`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider] instance for a
    custom key, base URL, or `httpx` client. The realtime WebSocket URL is derived from the provider's
    base URL (e.g. `https://api.openai.com/v1/` → `wss://api.openai.com/v1/realtime`), so
    OpenAI-compatible endpoints that expose a realtime API work too.

    Args:
        model: The model name, e.g. `gpt-realtime` or `gpt-realtime-2.1-mini`.
        provider: The provider to use for authentication and the base URL. Defaults to `'openai'`.
            Azure OpenAI is not supported (its realtime endpoint uses a different URL and auth scheme).
        voice: Voice for audio output, e.g. `alloy`, `echo`, or `shimmer`.
        input_audio_transcription_model: Model used to transcribe the user's audio input.
        handshake_timeout: Seconds to wait for each handshake event (`session.created` / `session.updated`)
            before failing, so `connect()` doesn't hang if the server never responds.
        turn_detection: How the server decides when the user's turn ends. A [`ServerVAD`][pydantic_ai.realtime.openai.ServerVAD]
            (the default) or [`SemanticVAD`][pydantic_ai.realtime.openai.SemanticVAD] configures automatic
            detection; `None` disables it for manual turn-taking (push-to-talk), where you drive the turn
            with `commit_audio()` + `create_response()`.
        input_noise_reduction: Noise reduction tuned for `near_field` (headset) or `far_field` (laptop/conference)
            microphones. `None` disables it.
        output_modalities: The modalities the model may produce, `('audio',)` (default) or `('text',)`.
        output_speed: Playback speed multiplier for generated audio (0.25–1.5). `None` uses the default.
        reconnect: Optional [`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy] to transparently
            recover from a dropped connection. `None` (the default) surfaces a drop as a non-recoverable
            `SessionError` instead.
    """

    model: str = 'gpt-realtime'
    provider: InitVar[Provider[AsyncOpenAI] | str] = 'openai'
    voice: str | None = None
    input_audio_transcription_model: str = 'whisper-1'
    handshake_timeout: float = 30.0
    turn_detection: ServerVAD | SemanticVAD | None = field(default_factory=ServerVAD)
    input_noise_reduction: Literal['near_field', 'far_field'] | None = None
    output_modalities: tuple[Literal['audio', 'text'], ...] = ('audio',)
    output_speed: float | None = None
    reconnect: ReconnectPolicy | None = None
    _provider: Provider[AsyncOpenAI] = field(init=False, repr=False)

    def __post_init__(self, provider: Provider[AsyncOpenAI] | str) -> None:
        if isinstance(provider, str):
            provider = cast('Provider[AsyncOpenAI]', infer_provider(provider))
        if provider.name == 'azure':
            raise UserError(
                'Azure OpenAI is not supported for realtime sessions: its realtime endpoint uses a '
                'different URL and authentication scheme. Use the `openai` provider instead.'
            )
        self._provider = provider

    @property
    def client(self) -> AsyncOpenAI:
        """The underlying [`AsyncOpenAI`](https://github.com/openai/openai-python) client from the provider."""
        return self._provider.client

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def capabilities(self) -> RealtimeCapabilities:
        return RealtimeCapabilities(
            image_input=True,
            manual_turn_control=True,
            interruption=True,
            output_truncation=True,
            session_seeding=True,
        )

    def _session_config(
        self, instructions: str, tools: list[ToolDefinition] | None, model_settings: ModelSettings | None
    ) -> dict[str, Any]:
        # `turn_detection` is always set: a dict enables VAD, `None` (explicit null) disables it.
        audio_input: dict[str, Any] = {
            'format': {'type': 'audio/pcm', 'rate': 24000},
            'turn_detection': _turn_detection_config(self.turn_detection),
        }
        if self.input_audio_transcription_model:
            audio_input['transcription'] = {'model': self.input_audio_transcription_model}
        if self.input_noise_reduction is not None:
            audio_input['noise_reduction'] = {'type': self.input_noise_reduction}
        audio_output: dict[str, Any] = {'format': {'type': 'audio/pcm', 'rate': 24000}}
        if self.voice:
            audio_output['voice'] = self.voice
        if self.output_speed is not None:
            audio_output['speed'] = self.output_speed
        config: dict[str, Any] = {
            'type': 'realtime',
            'instructions': instructions,
            'output_modalities': list(self.output_modalities),
            'audio': {'input': audio_input, 'output': audio_output},
        }
        if tools:
            config['tools'] = [_tool_def_to_openai(t) for t in tools]
        if model_settings:
            # Note: GA realtime sessions have no `temperature` field, so it is intentionally not forwarded.
            if (max_tokens := model_settings.get('max_tokens')) is not None:
                config['max_output_tokens'] = max_tokens
            if (parallel_tool_calls := model_settings.get('parallel_tool_calls')) is not None:
                config['parallel_tool_calls'] = parallel_tool_calls
            if (tool_choice := _tool_choice_config(model_settings.get('tool_choice'))) is not None:
                config['tool_choice'] = tool_choice
        return config

    @asynccontextmanager
    async def connect(
        self,
        *,
        instructions: str,
        tools: list[ToolDefinition] | None = None,
        native_tools: list[AbstractNativeTool] | None = None,
        model_settings: ModelSettings | None = None,
        messages: Sequence[ModelMessage] | None = None,
    ) -> AsyncGenerator[OpenAIRealtimeConnection]:
        if native_tools:
            raise UserError(
                f'The OpenAI realtime provider does not support native tools yet (got '
                f'{", ".join(type(t).__name__ for t in native_tools)}).'
            )
        url = f'{_realtime_websocket_url(self._provider.base_url)}?model={self.model}'
        headers = {'Authorization': f'Bearer {self._provider.client.api_key}'}
        session_config = self._session_config(instructions, tools, model_settings)

        # `dial` opens and configures a fresh connection. A reconnect closes the previous connection
        # (including one left half-open by a failed handshake) before opening the next, so sockets
        # don't accumulate; teardown closes whatever is current.
        cm: AbstractAsyncContextManager[ClientConnection] | None = None

        async def dial() -> ClientConnection:
            nonlocal cm
            if cm is not None:
                previous, cm = cm, None
                await previous.__aexit__(None, None, None)
            opening = websockets.connect(url, additional_headers=headers)
            ws = await opening.__aenter__()
            cm = opening
            await _expect_event(ws, 'session.created', timeout=self.handshake_timeout)
            await ws.send(json.dumps({'type': 'session.update', 'session': session_config}))
            await _expect_event(ws, 'session.updated', timeout=self.handshake_timeout)
            return ws

        try:
            ws = await dial()
            # Seed prior conversation once, after the initial handshake. Reconnects deliberately don't
            # re-seed: server state is lost on drop and a `Reconnected` starts a fresh turn.
            for item in _seed_items(messages or ()):
                await ws.send(json.dumps({'type': 'conversation.item.create', 'item': item}))
            yield OpenAIRealtimeConnection(ws, dial=dial, reconnect=self.reconnect)
        finally:
            if cm is not None:
                await cm.__aexit__(None, None, None)


async def _expect_event(ws: ClientConnection, expected_type: str, *, timeout: float) -> dict[str, Any]:
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
        data = _loads_obj(raw)
        event_type = data.get('type')
        if event_type == expected_type:
            return data
        if event_type == 'error':
            raise RuntimeError(f'OpenAI realtime error during handshake: {_error_message(data.get("error"))}')
