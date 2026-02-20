"""Gemini Live API provider for realtime speech-to-speech sessions.

Uses the ``google-genai`` SDK's async live API (``client.aio.live.connect()``)
to open a bidirectional streaming connection.

Requires the ``google-genai`` package::

    pip install "pydantic-ai-slim[google]"
"""

from __future__ import annotations as _annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

try:
    from google.genai import Client as GenaiClient, types as genai_types
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `google-genai` package to use the Gemini Realtime model, '
        'you can use the `google` optional group - `pip install "pydantic-ai-slim[google]"`'
    ) from _import_error

from ..settings import ModelSettings
from ..tools import ToolDefinition
from ._base import (
    AudioDelta,
    AudioInput,
    ImageInput,
    InputTranscript,
    RealtimeConnection,
    RealtimeEvent,
    RealtimeInput,
    RealtimeModel,
    TextInput,
    ToolCall,
    ToolResult,
    Transcript,
    TurnComplete,
)


def _tool_def_to_gemini(tool: ToolDefinition) -> genai_types.FunctionDeclaration:
    """Convert a pydantic-ai ToolDefinition to Gemini function declaration format."""
    return genai_types.FunctionDeclaration(
        name=tool.name,
        description=tool.description or '',
        parameters_json_schema=tool.parameters_json_schema,
    )


class GeminiRealtimeConnection(RealtimeConnection):
    """Live connection to the Gemini Live API."""

    def __init__(self, session: Any) -> None:
        self._session = session

    async def send(self, content: RealtimeInput) -> None:
        """Send content to the Gemini Live API.

        Accepts `AudioInput` (PCM16 16kHz mono), `ImageInput`, and `ToolResult`.
        """
        if isinstance(content, AudioInput):
            await self._session.send_realtime_input(
                audio=genai_types.Blob(data=content.data, mime_type='audio/pcm;rate=16000')
            )
        elif isinstance(content, ImageInput):
            await self._session.send_realtime_input(
                video=genai_types.Blob(data=content.data, mime_type=content.mime_type)
            )
        elif isinstance(content, TextInput):
            await self._session.send_client_content(
                turns=genai_types.Content(role='user', parts=[genai_types.Part(text=content.text)]),
                turn_complete=True,
            )
        elif isinstance(content, ToolResult):
            await self._session.send_tool_response(
                function_responses=[
                    genai_types.FunctionResponse(
                        id=content.tool_call_id,
                        name='',  # Gemini matches by id
                        response={'result': content.output},
                    )
                ]
            )
        else:
            raise NotImplementedError(f'Gemini Live does not support {type(content).__name__} input')

    async def __aiter__(self) -> AsyncIterator[RealtimeEvent]:
        # The google-genai SDK's receive() yields messages for a single model
        # turn and stops on turn_complete. We loop so the connection stays open
        # across multiple conversational turns.
        while True:
            has_messages = False
            async for msg in self._session.receive():
                has_messages = True
                for event in map_message(msg):
                    yield event
            if not has_messages:
                break


def map_message(msg: Any) -> list[RealtimeEvent]:
    """Map a Gemini LiveServerMessage to a list of RealtimeEvents.

    A single Gemini message can carry multiple events (e.g. audio parts + transcription),
    so we return a list.
    """
    events: list[RealtimeEvent] = []

    # Tool calls
    if msg.tool_call:
        events.extend(
            ToolCall(
                tool_call_id=fc.id or '',
                tool_name=fc.name or '',
                args=json.dumps(fc.args) if fc.args else '{}',
            )
            for fc in msg.tool_call.function_calls
        )
        return events

    # Tool call cancellation = interrupted turn
    if msg.tool_call_cancellation:
        events.append(TurnComplete(interrupted=True))
        return events

    # Server content (audio, text, transcriptions, turn signals)
    sc = msg.server_content
    if sc is None:
        return events

    # Model turn parts: audio and text
    if sc.model_turn and sc.model_turn.parts:
        for part in sc.model_turn.parts:
            if part.inline_data and isinstance(part.inline_data.data, bytes):
                events.append(AudioDelta(data=part.inline_data.data))
            elif part.text:
                events.append(Transcript(text=part.text, is_final=False))

    # Output transcription (model speech -> text)
    if sc.output_transcription and sc.output_transcription.text:
        events.append(Transcript(text=sc.output_transcription.text, is_final=bool(sc.output_transcription.finished)))

    # Input transcription (user speech -> text)
    if sc.input_transcription and sc.input_transcription.text:
        events.append(InputTranscript(text=sc.input_transcription.text, is_final=bool(sc.input_transcription.finished)))

    # Turn complete / interrupted
    if sc.turn_complete:
        events.append(TurnComplete(interrupted=bool(sc.interrupted)))

    return events


@dataclass
class GeminiRealtimeModel(RealtimeModel):
    """Gemini Live API model.

    Supports both Google AI (API key) and Vertex AI (project/location).

    Args:
        model: The model name, e.g. ``'gemini-2.5-flash-native-audio-preview'``.
        api_key: Google AI API key. Falls back to ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY`` env vars.
        project: Google Cloud project for Vertex AI.
        location: Google Cloud location for Vertex AI. Defaults to ``'us-central1'``.
        client: Pre-built ``google.GenaiClient`` instance. If provided, other auth args are ignored.
        voice: Voice name for audio output (e.g. ``'Kore'``, ``'Puck'``, ``'Charon'``).
        language_code: BCP-47 language code (e.g. ``'en-US'``).
        response_modalities: Output modalities. Defaults to ``[Modality.AUDIO]``.
            Pass ``[Modality.TEXT]`` for text-only or ``[Modality.AUDIO, Modality.TEXT]`` for both.
        enable_transcription: Whether to enable input/output audio transcription. Defaults to ``True``.
    """

    model: str = 'gemini-2.5-flash-native-audio-preview'
    api_key: str | None = None
    project: str | None = None
    location: str | None = None
    client: GenaiClient | None = field(default=None, repr=False)
    voice: str | None = None
    language_code: str | None = None
    response_modalities: list[genai_types.Modality] | None = None
    enable_transcription: bool = True

    @property
    def model_name(self) -> str:
        return self.model

    def _get_client(self) -> GenaiClient:
        if self.client is not None:
            return self.client

        if self.project or self.location:
            return GenaiClient(
                vertexai=True,
                project=self.project or os.environ.get('GOOGLE_CLOUD_PROJECT'),
                location=self.location or os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1'),
            )

        api_key = self.api_key or os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
        return GenaiClient(api_key=api_key)

    @asynccontextmanager
    async def connect(
        self,
        *,
        instructions: str,
        tools: list[ToolDefinition] | None = None,
        model_settings: ModelSettings | None = None,
    ) -> AsyncIterator[GeminiRealtimeConnection]:
        client = self._get_client()

        gemini_tools: genai_types.ToolListUnion | None = None
        tool_list = tools or []
        if tool_list:
            gemini_tools = [genai_types.Tool(function_declarations=[_tool_def_to_gemini(t) for t in tool_list])]

        speech_config: genai_types.SpeechConfig | None = None
        speech_kwargs: dict[str, Any] = {}
        if self.voice:
            speech_kwargs['voice_config'] = genai_types.VoiceConfig(
                prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(voice_name=self.voice)
            )
        if self.language_code:
            speech_kwargs['language_code'] = self.language_code
        if speech_kwargs:
            speech_config = genai_types.SpeechConfig(**speech_kwargs)

        modalities = self.response_modalities or [genai_types.Modality.AUDIO]

        transcription_config = genai_types.AudioTranscriptionConfig() if self.enable_transcription else None

        config = genai_types.LiveConnectConfig(
            response_modalities=modalities,
            system_instruction=instructions,
            input_audio_transcription=transcription_config,
            output_audio_transcription=transcription_config,
            tools=gemini_tools,
            speech_config=speech_config,
        )

        async with client.aio.live.connect(model=self.model, config=config) as session:
            yield GeminiRealtimeConnection(session)
