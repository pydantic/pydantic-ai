"""Azure realtime support using the OpenAI GA or Azure AI Voice Live protocol."""

from __future__ import annotations as _annotations

from dataclasses import InitVar, dataclass
from typing import Any, cast
from urllib.parse import urlencode, urlparse, urlunparse

from openai import AsyncOpenAI

from ..exceptions import UserError
from ..providers import Provider, infer_provider
from ..providers.azure import AzureProvider
from ..tools import ToolDefinition
from ._base import RealtimeCodecEvent, Transcript
from ._openai_protocol import (
    SemanticVAD,
    ServerVAD,
    map_event as _map_openai_event,
    resolve_base_turn_detection,
    resolve_transcription_model,
    tool_choice_config,
    tool_def_to_openai,
    turn_detection_config,
)
from .openai import OpenAIRealtimeConnection, OpenAIRealtimeModel, OpenAIRealtimeModelSettings

__all__ = ('AzureRealtimeModel', 'AzureRealtimeModelSettings')

_AZURE_VOICE_LIVE_API_VERSION = '2026-04-10'


class AzureRealtimeModelSettings(OpenAIRealtimeModelSettings, total=False):
    """Settings specific to Azure realtime models."""

    azure_voice_live: bool
    """Use the Azure AI Voice Live endpoint and beta session protocol instead of the GA endpoint."""
    azure_voice_live_turn_detection: ServerVAD | SemanticVAD
    """Voice Live server or semantic VAD config; only applies when `azure_voice_live=True`."""


def _map_voice_live_event(data: dict[str, Any]) -> RealtimeCodecEvent | None:
    """Map Voice Live's beta text events and delegate the remaining OpenAI-compatible events."""
    event_type = data.get('type')
    if event_type == 'response.text.delta':
        delta = data.get('delta')
        return Transcript(text=delta if isinstance(delta, str) else '', is_final=False, output_text=True)
    if event_type == 'response.text.done':
        text = data.get('text')
        return Transcript(text=text if isinstance(text, str) else '', is_final=True, output_text=True)
    return _map_openai_event(data)


class _AzureRealtimeConnection(OpenAIRealtimeConnection):
    """An Azure realtime connection supporting Voice Live's beta text events."""

    def _map_event(self, data: dict[str, Any]) -> RealtimeCodecEvent | None:
        return _map_voice_live_event(data)


@dataclass
class AzureRealtimeModel(OpenAIRealtimeModel):
    """Azure realtime model using the OpenAI GA protocol or Azure AI Voice Live.

    The existing [`AzureProvider`][pydantic_ai.providers.azure.AzureProvider] supplies the Azure
    resource endpoint and API key. The WebSocket transport does not use its OpenAI SDK client or
    `api_version`. By default it connects to the GA `/openai/v1/realtime` endpoint; set
    [`azure_voice_live`][pydantic_ai.realtime.azure.AzureRealtimeModelSettings.azure_voice_live]
    to connect to `/voice-live/realtime` with the Voice Live beta session protocol. Both use an
    `api-key` header.

    <!-- TODO(voice-live): Keep GA as the default until maintainers decide whether Voice Live should
    become the default Azure realtime path. -->
    <!-- TODO(voice-live): Auto-routing Voice-Live-exclusive model names requires an agreed model/path map. -->
    """

    provider: InitVar[Provider[AsyncOpenAI] | str] = 'azure'

    def __post_init__(self, provider: Provider[AsyncOpenAI] | str) -> None:
        if isinstance(provider, str):
            resolved = infer_provider(provider)
            if not isinstance(resolved, AzureProvider):
                raise UserError("`AzureRealtimeModel` requires an `AzureProvider` or `provider='azure'`.")
            provider = resolved
        self._provider = provider

    @property
    def _azure_provider(self) -> AzureProvider:
        assert isinstance(self._provider, AzureProvider)
        return self._provider

    def _realtime_url(self, model_settings: OpenAIRealtimeModelSettings | None = None) -> str:
        parsed = urlparse(self._azure_provider.azure_endpoint)
        if model_settings and model_settings.get('azure_voice_live'):
            return urlunparse(
                parsed._replace(
                    scheme='wss',
                    path='/voice-live/realtime',
                    query=urlencode({'api-version': _AZURE_VOICE_LIVE_API_VERSION, 'model': self.model}),
                )
            )
        return urlunparse(
            parsed._replace(scheme='wss', path='/openai/v1/realtime', query=urlencode({'model': self.model}))
        )

    def _session_config(
        self,
        instructions: str,
        tools: list[ToolDefinition] | None,
        model_settings: OpenAIRealtimeModelSettings | None,
    ) -> dict[str, Any]:
        settings = cast('AzureRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        if not settings.get('azure_voice_live'):
            return super()._session_config(instructions, tools, settings)

        if 'azure_voice_live_turn_detection' in settings:
            turn_detection = settings['azure_voice_live_turn_detection']
        elif 'turn_detection' in settings:
            turn_detection = resolve_base_turn_detection(settings['turn_detection'])
        else:
            turn_detection = ServerVAD()
        auto_transcription_model = 'whisper-1' if self.model.startswith('gpt-realtime') else 'azure-speech'
        transcription_model = resolve_transcription_model(
            settings.get('input_transcription_model', 'auto'), default=auto_transcription_model
        )
        config: dict[str, Any] = {
            'instructions': instructions,
            'modalities': ['text'] if settings.get('output_modality') == 'text' else ['text', 'audio'],
            'input_audio_format': 'pcm16',
            'output_audio_format': 'pcm16',
            'input_audio_sampling_rate': 24000,
            'turn_detection': turn_detection_config(turn_detection),
        }
        if transcription_model is not None:
            config['input_audio_transcription'] = {'model': transcription_model}
        if voice := settings.get('voice'):
            config['voice'] = {'type': 'openai', 'name': voice}
        if tools:
            config['tools'] = [tool_def_to_openai(tool) for tool in tools]
        if (max_tokens := settings.get('max_tokens')) is not None:
            config['max_response_output_tokens'] = max_tokens
        if (tool_choice := tool_choice_config(settings.get('tool_choice'))) is not None:
            config['tool_choice'] = tool_choice
        return config

    def _connection_class(self, model_settings: OpenAIRealtimeModelSettings) -> type[OpenAIRealtimeConnection]:
        if model_settings.get('azure_voice_live'):
            return _AzureRealtimeConnection
        return OpenAIRealtimeConnection

    async def _auth_headers(self) -> dict[str, str]:
        return {'api-key': self._azure_provider.api_key}
