"""OpenAI Realtime API provider for speech-to-speech sessions.

Uses WebSocket to connect to ``wss://api.openai.com/v1/realtime`` and maps
the OpenAI-specific event protocol to the shared `RealtimeEvent` types.

Requires the ``websockets`` package::

    pip install "pydantic-ai-slim[realtime]"
"""

from __future__ import annotations as _annotations

import base64
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
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
    TextInput,
    ToolCall,
    ToolResult,
    Transcript,
    TurnComplete,
)

DEFAULT_REALTIME_URL = 'wss://api.openai.com/v1/realtime'


def _tool_def_to_openai(tool: ToolDefinition) -> dict[str, object]:
    """Convert a pydantic-ai ToolDefinition to OpenAI realtime tool format."""
    result: dict[str, object] = {
        'type': 'function',
        'name': tool.name,
        'parameters': tool.parameters_json_schema,
    }
    if tool.description:
        result['description'] = tool.description
    return result


class OpenAIRealtimeConnection(RealtimeConnection):
    """Live WebSocket connection to the OpenAI Realtime API."""

    def __init__(self, ws: ClientConnection) -> None:
        self._ws = ws

    async def send(self, content: RealtimeInput) -> None:
        """Send content to the OpenAI Realtime API.

        Accepts `AudioInput` (PCM16 24kHz mono) and `ToolResult`.
        """
        if isinstance(content, AudioInput):
            event = {
                'type': 'input_audio_buffer.append',
                'audio': base64.b64encode(content.data).decode('ascii'),
            }
            await self._ws.send(json.dumps(event))
        elif isinstance(content, TextInput):
            item_event = {
                'type': 'conversation.item.create',
                'item': {
                    'type': 'message',
                    'role': 'user',
                    'content': [{'type': 'input_text', 'text': content.text}],
                },
            }
            await self._ws.send(json.dumps(item_event))
            await self._ws.send(json.dumps({'type': 'response.create'}))
        elif isinstance(content, ToolResult):
            item_event = {
                'type': 'conversation.item.create',
                'item': {
                    'type': 'function_call_output',
                    'call_id': content.tool_call_id,
                    'output': content.output,
                },
            }
            await self._ws.send(json.dumps(item_event))
            # Trigger a new model response after providing the tool result
            await self._ws.send(json.dumps({'type': 'response.create'}))
        else:
            raise NotImplementedError(f'OpenAI Realtime does not support {type(content).__name__} input')

    async def __aiter__(self) -> AsyncIterator[RealtimeEvent]:
        async for raw in self._ws:
            if not isinstance(raw, str):
                continue
            data: dict[str, Any] = json.loads(raw)
            event = map_event(data)
            if event is not None:
                yield event


def map_event(data: dict[str, Any]) -> RealtimeEvent | None:
    """Map an OpenAI Realtime event dict to a `RealtimeEvent`."""
    event_type = data.get('type')

    if event_type == 'response.audio.delta':
        audio_b64 = data.get('delta', '')
        if not isinstance(audio_b64, str):
            return None
        return AudioDelta(data=base64.b64decode(audio_b64))

    if event_type == 'response.audio_transcript.delta':
        delta = data.get('delta', '')
        if not isinstance(delta, str):
            return None
        return Transcript(text=delta, is_final=False)

    if event_type == 'response.audio_transcript.done':
        transcript = data.get('transcript', '')
        if not isinstance(transcript, str):
            return None
        return Transcript(text=transcript, is_final=True)

    if event_type == 'conversation.item.input_audio_transcription.completed':
        transcript = data.get('transcript', '')
        if not isinstance(transcript, str):
            return None
        return InputTranscript(text=transcript, is_final=True)

    if event_type == 'response.function_call_arguments.done':
        call_id = data.get('call_id', '')
        name = data.get('name', '')
        arguments = data.get('arguments', '{}')
        if not isinstance(call_id, str) or not isinstance(name, str) or not isinstance(arguments, str):
            return None
        return ToolCall(tool_call_id=call_id, tool_name=name, args=arguments)

    if event_type == 'response.done':
        response = data.get('response')
        interrupted = isinstance(response, dict) and cast(dict[str, Any], response).get('status') == 'cancelled'
        return TurnComplete(interrupted=interrupted)

    if event_type == 'error':
        error = data.get('error')
        if isinstance(error, dict):
            error_dict = cast(dict[str, Any], error)
            msg = error_dict.get('message')
            return SessionError(message=str(msg) if msg else str(error_dict))
        return SessionError(message=str(error))

    # Ignore events we don't map (session.created, session.updated, etc.)
    return None


@dataclass
class OpenAIRealtimeModel(RealtimeModel):
    """OpenAI Realtime API model.

    Args:
        model: The model name, e.g. ``'gpt-4o-realtime-preview'``.
        api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
        base_url: WebSocket base URL. Defaults to ``wss://api.openai.com/v1/realtime``.
        voice: Voice to use for audio output (e.g. ``'alloy'``, ``'echo'``, ``'shimmer'``).
    """

    model: str = 'gpt-4o-realtime-preview'
    api_key: str | None = None
    base_url: str = DEFAULT_REALTIME_URL
    voice: str | None = None

    @property
    def model_name(self) -> str:
        return self.model

    @asynccontextmanager
    async def connect(
        self,
        *,
        instructions: str,
        tools: list[ToolDefinition] | None = None,
        model_settings: ModelSettings | None = None,
    ) -> AsyncIterator[OpenAIRealtimeConnection]:
        api_key = self.api_key or os.environ.get('OPENAI_API_KEY', '')
        url = f'{self.base_url}?model={self.model}'

        headers = {
            'Authorization': f'Bearer {api_key}',
            'OpenAI-Beta': 'realtime=v1',
        }

        async with websockets.connect(url, additional_headers=headers) as ws:
            # Wait for session.created
            raw = await ws.recv()
            if not isinstance(raw, str):
                raise TypeError(f'Expected text message from WebSocket, got {type(raw).__name__}')
            created: dict[str, Any] = json.loads(raw)
            if created.get('type') != 'session.created':
                raise RuntimeError(f'Expected session.created, got {created.get("type")}')

            # Build session.update payload
            session_config: dict[str, object] = {
                'instructions': instructions,
                'input_audio_transcription': {'model': 'whisper-1'},
            }

            tool_list = tools or []
            if tool_list:
                session_config['tools'] = [_tool_def_to_openai(t) for t in tool_list]

            if self.voice:
                session_config['voice'] = self.voice

            update_event = {
                'type': 'session.update',
                'session': session_config,
            }
            await ws.send(json.dumps(update_event))

            # Wait for session.updated confirmation
            raw = await ws.recv()
            if not isinstance(raw, str):
                raise TypeError(f'Expected text message from WebSocket, got {type(raw).__name__}')
            updated: dict[str, Any] = json.loads(raw)
            if updated.get('type') != 'session.updated':
                raise RuntimeError(f'Expected session.updated, got {updated.get("type")}')

            yield OpenAIRealtimeConnection(ws)
