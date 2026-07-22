"""Network-free tests for the Azure OpenAI realtime model."""

from __future__ import annotations as _annotations

import pytest

from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.realtime import TurnDetection
from pydantic_ai.realtime.azure import AzureRealtimeModel, AzureRealtimeModelSettings, ServerVAD
from pydantic_ai.tools import ToolDefinition


def test_model_is_exported_from_realtime_package() -> None:
    from pydantic_ai.realtime import AzureRealtimeModel as ExportedAzureRealtimeModel

    assert ExportedAzureRealtimeModel is AzureRealtimeModel


def test_default_provider() -> None:
    assert AzureRealtimeModel().system == 'azure'


@pytest.mark.anyio
async def test_url_and_auth_headers() -> None:
    provider = AzureProvider(
        azure_endpoint='https://resource.openai.azure.com/openai/v1/',
        api_key='azure-key',
    )
    model = AzureRealtimeModel('gpt realtime', provider=provider)

    assert model._realtime_url() == (  # pyright: ignore[reportPrivateUsage]
        'wss://resource.openai.azure.com/openai/v1/realtime?model=gpt+realtime'
    )
    assert await model._auth_headers() == {'api-key': 'azure-key'}  # pyright: ignore[reportPrivateUsage]


@pytest.mark.anyio
async def test_voice_live_url_and_auth_headers() -> None:
    provider = AzureProvider(
        azure_endpoint='https://resource.services.ai.azure.com',
        api_version='2026-04-10',
        api_key='azure-key',
    )
    settings = AzureRealtimeModelSettings(azure_voice_live=True)
    model = AzureRealtimeModel('gpt realtime', provider=provider, settings=settings)

    assert model._realtime_url(settings) == (  # pyright: ignore[reportPrivateUsage]
        'wss://resource.services.ai.azure.com/voice-live/realtime?api-version=2026-04-10&model=gpt+realtime'
    )
    assert await model._auth_headers() == {'api-key': 'azure-key'}  # pyright: ignore[reportPrivateUsage]


def test_voice_live_session_config_options() -> None:
    provider = AzureProvider(
        azure_endpoint='https://resource.services.ai.azure.com',
        api_version='2026-04-10',
        api_key='azure-key',
    )
    model = AzureRealtimeModel('phi-4-mm-realtime', provider=provider)
    settings = AzureRealtimeModelSettings(
        azure_voice_live=True,
        azure_voice_live_turn_detection=ServerVAD(silence_duration_ms=750),
        input_transcription_model=None,
        voice='alloy',
        max_tokens=123,
        tool_choice='required',
    )

    config = model._session_config(  # pyright: ignore[reportPrivateUsage]
        'Be concise.',
        [ToolDefinition(name='lookup', parameters_json_schema={'type': 'object'})],
        settings,
    )

    assert config['turn_detection']['silence_duration_ms'] == 750
    assert 'input_audio_transcription' not in config
    assert config['voice'] == {'type': 'openai', 'name': 'alloy'}
    assert config['max_response_output_tokens'] == 123
    assert config['tool_choice'] == 'required'
    assert config['tools'][0]['name'] == 'lookup'

    config = model._session_config(  # pyright: ignore[reportPrivateUsage]
        '', None, AzureRealtimeModelSettings(azure_voice_live=True, turn_detection=TurnDetection(sensitivity='high'))
    )
    assert config['turn_detection']['threshold'] == 0.3
    assert config['input_audio_transcription'] == {'model': 'azure-speech'}
