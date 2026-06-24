"""OpenAI Realtime API provider for speech-to-speech sessions.

Connects to `wss://api.openai.com/v1/realtime` over a WebSocket and maps the OpenAI event
protocol to the shared realtime event types.

Requires the `websockets` package, available via the `realtime` optional group:

    pip install "pydantic-ai-slim[realtime]"
"""

from __future__ import annotations as _annotations

import base64
import json
import os
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, cast

try:
    import websockets
    from websockets.asyncio.client import ClientConnection
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `websockets` package to use the OpenAI Realtime model, '
        'you can use the `realtime` optional group - `pip install "pydantic-ai-slim[realtime]"`'
    ) from _import_error

from ..settings import ModelSettings
from ..tools import ToolDefinition
from ._base import (
    AudioDelta,
    AudioInput,
    InputTranscript,
    RealtimeConnection,
    RealtimeEvent,
    RealtimeInput,
    RealtimeModel,
    SessionError,
    SpeechStarted,
    TextInput,
    ToolCall,
    ToolResult,
    Transcript,
    TurnComplete,
)

DEFAULT_REALTIME_URL = 'wss://api.openai.com/v1/realtime'

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


def _str_field(data: dict[str, Any], key: str, default: str = '') -> str:
    """Return `data[key]` if it is a string, otherwise `default`."""
    value = data.get(key, default)
    return value if isinstance(value, str) else default


def _obj(value: Any) -> dict[str, Any]:
    """Return `value` as a `dict[str, Any]` when it is a mapping, otherwise an empty dict."""
    return cast('dict[str, Any]', value) if isinstance(value, dict) else {}


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

    if event_type == 'response.done':
        return _map_response_done(data)

    if event_type == 'error':
        return SessionError(message=_error_message(data.get('error')))

    return None


def _error_message(error: Any) -> str:
    """Extract a human-readable message from an OpenAI `error` payload."""
    if isinstance(error, dict):
        message = _obj(error).get('message')
        return message if isinstance(message, str) and message else json.dumps(_obj(error))
    return str(error)


class OpenAIRealtimeConnection(RealtimeConnection):
    """A live WebSocket connection to the OpenAI Realtime API."""

    def __init__(self, ws: ClientConnection) -> None:
        self._ws = ws
        # The Realtime API rejects `response.create` while a response is already being generated.
        # We track that window and defer requests (e.g. a background tool result that lands while the
        # model is mid-answer) until the active response finishes, so the model still announces it.
        self._response_active = False
        self._pending_response = False

    async def send(self, content: RealtimeInput) -> None:
        """Send content to the OpenAI Realtime API.

        Accepts `AudioInput` (PCM16, 24kHz, mono), `TextInput`, and `ToolResult`.
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

    async def __aiter__(self) -> AsyncIterator[RealtimeEvent]:
        async for raw in self._ws:
            if not isinstance(raw, str):
                continue
            data: dict[str, Any] = json.loads(raw)
            event_type = data.get('type')
            if event_type == 'response.created':
                self._response_active = True
            elif event_type == 'response.done':
                self._response_active = False
                if self._pending_response:
                    self._pending_response = False
                    self._response_active = True
                    await self._send_event({'type': 'response.create'})
            event = map_event(data)
            if event is not None:
                yield event


@dataclass
class OpenAIRealtimeModel(RealtimeModel):
    """OpenAI Realtime API model.

    Args:
        model: The model name, e.g. `gpt-realtime` or `gpt-4o-realtime-preview`.
        api_key: OpenAI API key. Falls back to the `OPENAI_API_KEY` environment variable.
        base_url: WebSocket base URL. Defaults to `wss://api.openai.com/v1/realtime`.
        voice: Voice for audio output, e.g. `alloy`, `echo`, or `shimmer`.
        input_audio_transcription_model: Model used to transcribe the user's audio input.
    """

    model: str = 'gpt-realtime'
    api_key: str | None = field(default=None, repr=False)
    base_url: str = DEFAULT_REALTIME_URL
    voice: str | None = None
    input_audio_transcription_model: str = 'whisper-1'

    @property
    def model_name(self) -> str:
        return self.model

    def _session_config(
        self, instructions: str, tools: list[ToolDefinition] | None, model_settings: ModelSettings | None
    ) -> dict[str, Any]:
        audio_input: dict[str, Any] = {
            'format': {'type': 'audio/pcm', 'rate': 24000},
            'turn_detection': {'type': 'server_vad'},
        }
        if self.input_audio_transcription_model:
            audio_input['transcription'] = {'model': self.input_audio_transcription_model}
        audio_output: dict[str, Any] = {'format': {'type': 'audio/pcm', 'rate': 24000}}
        if self.voice:
            audio_output['voice'] = self.voice
        config: dict[str, Any] = {
            'type': 'realtime',
            'instructions': instructions,
            'output_modalities': ['audio'],
            'audio': {'input': audio_input, 'output': audio_output},
        }
        if tools:
            config['tools'] = [_tool_def_to_openai(t) for t in tools]
        if model_settings and (max_tokens := model_settings.get('max_tokens')) is not None:
            config['max_output_tokens'] = max_tokens
        return config

    @asynccontextmanager
    async def connect(
        self,
        *,
        instructions: str,
        tools: list[ToolDefinition] | None = None,
        model_settings: ModelSettings | None = None,
    ) -> AsyncGenerator[OpenAIRealtimeConnection]:
        api_key = self.api_key or os.environ.get('OPENAI_API_KEY', '')
        url = f'{self.base_url}?model={self.model}'
        headers = {'Authorization': f'Bearer {api_key}'}

        async with websockets.connect(url, additional_headers=headers) as ws:
            await _expect_event(ws, 'session.created')

            await ws.send(
                json.dumps(
                    {'type': 'session.update', 'session': self._session_config(instructions, tools, model_settings)}
                )
            )
            await _expect_event(ws, 'session.updated')

            yield OpenAIRealtimeConnection(ws)


async def _expect_event(ws: ClientConnection, expected_type: str) -> dict[str, Any]:
    """Read events until one of `expected_type` arrives, raising on a server error.

    Unrelated events received during the handshake (e.g. rate limit notices) are skipped rather than
    treated as a protocol violation.
    """
    while True:
        raw = await ws.recv()
        if not isinstance(raw, str):  # pragma: no cover
            raise TypeError(f'Expected a text message from the WebSocket, got {type(raw).__name__}')
        data: dict[str, Any] = json.loads(raw)
        event_type = data.get('type')
        if event_type == expected_type:
            return data
        if event_type == 'error':
            raise RuntimeError(f'OpenAI realtime error during handshake: {_error_message(data.get("error"))}')
