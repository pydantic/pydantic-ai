"""Azure AI Voice Live provider for realtime speech-to-speech sessions.

Voice Live uses the OpenAI Realtime beta event protocol with a different WebSocket URL,
authentication header, session configuration, and text-output event names. This module reuses the
shared OpenAI codec and connection for everything else.

Requires the `websockets` package, available via the `realtime` optional group:

    pip install "pydantic-ai-slim[realtime]"
"""

from __future__ import annotations as _annotations

import json
from collections.abc import AsyncGenerator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import InitVar, dataclass, field
from typing import Any, cast
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

try:
    import websockets
    from websockets.asyncio.client import ClientConnection
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `websockets` package to use the Azure AI Voice Live realtime model, '
        'you can use the `realtime` optional group - `pip install "pydantic-ai-slim[realtime]"`'
    ) from _import_error

from .._instrumentation import get_instructions
from ..exceptions import UserError
from ..messages import ModelMessage
from ..models import ModelRequestParameters
from ..providers import infer_provider
from ..providers.azure_voicelive import AzureVoiceLiveProvider
from ..tools import ToolDefinition
from ._base import (
    RealtimeCodecEvent,
    RealtimeModel,
    RealtimeModelSettings,
    ReconnectPolicy,
    Transcript,
    inject_trace_context,
)
from ._openai_protocol import (
    SemanticVAD,
    ServerVAD,
    expect_event,
    map_event as _map_openai_event,
    obj,
    resolve_base_turn_detection,
    resolve_transcription_model,
    seed_items,
    tool_choice_config,
    tool_def_to_openai,
    turn_detection_config,
)
from .openai import OpenAIRealtimeConnection

__all__ = (
    'AzureRealtimeConnection',
    'AzureRealtimeModel',
    'AzureRealtimeModelSettings',
    'SemanticVAD',
    'ServerVAD',
    'map_event',
)


class AzureRealtimeModelSettings(RealtimeModelSettings, total=False):
    """Settings specific to Azure AI Voice Live realtime models.

    Voice Live currently ignores the inherited `parallel_tool_calls` and `thinking` settings.
    """

    azure_turn_detection: ServerVAD | SemanticVAD
    """Azure-specific server or semantic VAD configuration.

    When present, this fully overrides the cross-provider `turn_detection` setting.
    """


def map_event(data: dict[str, Any]) -> RealtimeCodecEvent | None:
    """Map a raw Azure AI Voice Live event to a realtime codec event.

    Voice Live uses `response.text.*` for plain text output; all other supported events map through
    the shared OpenAI Realtime codec.
    """
    event_type = data.get('type')
    if event_type == 'response.text.delta':
        delta = data.get('delta')
        return Transcript(text=delta if isinstance(delta, str) else '', is_final=False, output_text=True)
    if event_type == 'response.text.done':
        text = data.get('text')
        return Transcript(text=text if isinstance(text, str) else '', is_final=True, output_text=True)
    return _map_openai_event(data)


class AzureRealtimeConnection(OpenAIRealtimeConnection):
    """A live WebSocket connection to Azure AI Voice Live."""

    def _map_event(self, data: dict[str, Any]) -> RealtimeCodecEvent | None:
        return map_event(data)


def azure_voice_live_websocket_url(endpoint: str, *, api_version: str, model: str) -> str:
    """Build an Azure AI Voice Live WebSocket URL from a resource endpoint."""
    parsed = urlparse(endpoint)
    if parsed.scheme == 'https':
        scheme = 'wss'
    elif parsed.scheme == 'http':
        scheme = 'ws'
    else:
        scheme = parsed.scheme
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update({'api-version': api_version, 'model': model})
    path = f'{parsed.path.rstrip("/")}/voice-live/realtime'
    return urlunparse((scheme, parsed.netloc, path, parsed.params, urlencode(query), parsed.fragment))


@dataclass
class AzureRealtimeModel(RealtimeModel):
    """Azure AI Voice Live realtime model.

    Pass `provider='azure-voicelive'` (the default) to read `AZURE_VOICELIVE_ENDPOINT`,
    `AZURE_VOICELIVE_API_VERSION`, and `AZURE_VOICELIVE_API_KEY`, or pass an
    [`AzureVoiceLiveProvider`][pydantic_ai.providers.azure_voicelive.AzureVoiceLiveProvider]
    constructed explicitly.

    Args:
        model: The Voice Live model identifier, e.g. `gpt-realtime`.
        provider: The provider supplying the resource endpoint, API version, and API key.
        reconnect: Optional [`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy] to recover from
            a dropped connection.
    """

    model: str = 'gpt-realtime'
    provider: InitVar[AzureVoiceLiveProvider | str] = 'azure-voicelive'
    settings: RealtimeModelSettings | None = field(default=None, kw_only=True)
    reconnect: ReconnectPolicy | None = None
    _provider: AzureVoiceLiveProvider = field(init=False, repr=False)

    def __post_init__(self, provider: AzureVoiceLiveProvider | str) -> None:
        if isinstance(provider, str):
            resolved = infer_provider(provider)
            if not isinstance(resolved, AzureVoiceLiveProvider):
                raise UserError(
                    "`AzureRealtimeModel` requires an `AzureVoiceLiveProvider` or `provider='azure-voicelive'`."
                )
            provider = resolved
        self._provider = provider

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def system(self) -> str:
        return self._provider.name

    def _session_config(
        self,
        instructions: str,
        tools: list[ToolDefinition] | None,
        model_settings: AzureRealtimeModelSettings | None,
    ) -> dict[str, Any]:
        model_settings = cast('AzureRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        if 'azure_turn_detection' in model_settings:
            turn_detection = model_settings['azure_turn_detection']
        elif 'turn_detection' in model_settings:
            turn_detection = resolve_base_turn_detection(model_settings['turn_detection'])
        else:
            turn_detection = ServerVAD()

        auto_transcription_model = 'whisper-1' if self.model.startswith('gpt-realtime') else 'azure-speech'
        transcription_model = resolve_transcription_model(
            model_settings.get('input_transcription_model', 'auto'), default=auto_transcription_model
        )
        config: dict[str, Any] = {
            'instructions': instructions,
            'modalities': ['text'] if model_settings.get('output_modality') == 'text' else ['text', 'audio'],
            'input_audio_format': 'pcm16',
            'output_audio_format': 'pcm16',
            'input_audio_sampling_rate': 24000,
            'turn_detection': turn_detection_config(turn_detection),
        }
        if transcription_model is not None:
            config['input_audio_transcription'] = {'model': transcription_model}
        if voice := model_settings.get('voice'):
            config['voice'] = {'type': 'openai', 'name': voice}
        if tools:
            config['tools'] = [tool_def_to_openai(tool) for tool in tools]
        if (max_tokens := model_settings.get('max_tokens')) is not None:
            config['max_response_output_tokens'] = max_tokens
        if (tool_choice := tool_choice_config(model_settings.get('tool_choice'))) is not None:
            config['tool_choice'] = tool_choice
        return config

    @asynccontextmanager
    async def connect(
        self,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncGenerator[AzureRealtimeConnection]:
        url = azure_voice_live_websocket_url(
            self._provider.base_url,
            api_version=self._provider.api_version,
            model=self.model,
        )
        headers = {'api-key': self._provider.api_key}
        inject_trace_context(headers)
        settings = cast('AzureRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        handshake_timeout = settings.get('handshake_timeout', 30.0)
        session_config = self._session_config(
            get_instructions(messages) or '', model_request_parameters.function_tools, settings
        )
        transcription_enabled = settings.get('input_transcription_model', 'auto') is not None

        cm: AbstractAsyncContextManager[ClientConnection] | None = None
        server_model: str | None = None

        async def dial() -> ClientConnection:
            nonlocal cm, server_model
            if cm is not None:
                previous, cm = cm, None
                await previous.__aexit__(None, None, None)
            opening = websockets.connect(url, additional_headers=headers)
            ws = await opening.__aenter__()
            cm = opening
            created = await expect_event(ws, 'session.created', timeout=handshake_timeout)
            model = obj(created.get('session')).get('model')
            if isinstance(model, str) and model:
                server_model = model
            await ws.send(json.dumps({'type': 'session.update', 'session': session_config}))
            await expect_event(ws, 'session.updated', timeout=handshake_timeout)
            return ws

        try:
            ws = await dial()
            for item in await seed_items(messages, profile=self.profile, provider_name=self.system):
                await ws.send(json.dumps({'type': 'conversation.item.create', 'item': item}))
            yield AzureRealtimeConnection(
                ws,
                dial=dial,
                reconnect=self.reconnect,
                input_transcription_enabled=transcription_enabled,
                model_name=server_model,
            )
        finally:
            if cm is not None:
                await cm.__aexit__(None, None, None)
