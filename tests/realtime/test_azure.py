"""Network-free tests for the Azure OpenAI realtime model."""

from __future__ import annotations as _annotations

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import ToolDefinition

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.realtime import TurnDetection
    from pydantic_ai.realtime._base import Transcript
    from pydantic_ai.realtime.azure import (
        AzureRealtimeModel,
        AzureRealtimeModelSettings,
        ServerVAD,
        _map_voice_live_event,  # pyright: ignore[reportPrivateUsage]
    )

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai / websockets not installed')


def test_model_is_exported_from_realtime_package() -> None:
    from pydantic_ai.realtime import AzureRealtimeModel as ExportedAzureRealtimeModel

    assert ExportedAzureRealtimeModel is AzureRealtimeModel


def test_default_provider() -> None:
    assert AzureRealtimeModel().system == 'azure'


def test_non_azure_provider_instance_is_rejected() -> None:
    # A non-Azure `Provider` *instance* (not just the `provider='...'` string) must fail fast with a clear
    # `UserError` at construction, rather than an `AssertionError` deep inside later.
    with pytest.raises(UserError, match='requires an `AzureProvider`'):
        AzureRealtimeModel('gpt-realtime', provider=OpenAIProvider(api_key='x'))


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


@pytest.mark.anyio
async def test_voice_live_uses_coherent_credential_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Voice Live targets its own endpoint/key/version as one set, never mixed with the GA resource.

    Also pins the fix for the previously hard-coded API version: the Voice Live URL now reflects the
    configured `AZURE_VOICELIVE_API_VERSION`.
    """
    monkeypatch.setenv('AZURE_VOICELIVE_ENDPOINT', 'https://vl.services.ai.azure.com')
    monkeypatch.setenv('AZURE_VOICELIVE_API_KEY', 'vl-key')
    monkeypatch.setenv('AZURE_VOICELIVE_API_VERSION', '2026-06-01-preview')
    provider = AzureProvider(azure_endpoint='https://ga.openai.azure.com/openai/v1', api_key='ga-key')
    model = AzureRealtimeModel('gpt-realtime', provider=provider)
    vl = AzureRealtimeModelSettings(azure_voice_live=True)

    # GA path → GA resource; Voice Live path → Voice Live resource + configured version.
    assert model._realtime_url() == 'wss://ga.openai.azure.com/openai/v1/realtime?model=gpt-realtime'  # pyright: ignore[reportPrivateUsage]
    assert model._realtime_url(vl) == (  # pyright: ignore[reportPrivateUsage]
        'wss://vl.services.ai.azure.com/voice-live/realtime?api-version=2026-06-01-preview&model=gpt-realtime'
    )
    assert await model._auth_headers() == {'api-key': 'ga-key'}  # pyright: ignore[reportPrivateUsage]
    assert await model._auth_headers(vl) == {'api-key': 'vl-key'}  # pyright: ignore[reportPrivateUsage]


def test_voice_live_default_api_version() -> None:
    """Without `AZURE_VOICELIVE_API_VERSION`, the Voice Live URL falls back to the supported default."""
    provider = AzureProvider(
        azure_endpoint='https://resource.services.ai.azure.com', api_version='2024-10-01', api_key='k'
    )
    model = AzureRealtimeModel('gpt-realtime', provider=provider)
    url = model._realtime_url(AzureRealtimeModelSettings(azure_voice_live=True))  # pyright: ignore[reportPrivateUsage]
    assert 'api-version=2026-04-10' in url


def test_realtime_url_ignores_endpoint_path_and_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both GA and Voice Live URLs are derived from the resource host, dropping any base path/query.

    Azure `azure_endpoint`s come in several shapes (bare host, trailing slash, `/openai/v1`, a stray
    query); the WebSocket path is fixed per service, so all should resolve to the same host + service path.
    """
    # A version in the environment so the non-`/v1` endpoints (which need one for the GA client) construct;
    # `/v1` endpoints ignore it. Neither affects the realtime WebSocket URL, which is derived from the host.
    monkeypatch.setenv('OPENAI_API_VERSION', '2024-10-01')
    vl = AzureRealtimeModelSettings(azure_voice_live=True)
    for endpoint in (
        'https://r.openai.azure.com',
        'https://r.openai.azure.com/',
        'https://r.openai.azure.com/openai/v1',
        'https://r.openai.azure.com/openai/v1/?foo=bar',
    ):
        model = AzureRealtimeModel('m', provider=AzureProvider(azure_endpoint=endpoint, api_key='k'))
        assert model._realtime_url() == 'wss://r.openai.azure.com/openai/v1/realtime?model=m'  # pyright: ignore[reportPrivateUsage]
        assert model._realtime_url(vl) == 'wss://r.openai.azure.com/voice-live/realtime?api-version=2026-04-10&model=m'  # pyright: ignore[reportPrivateUsage]


def test_voice_live_event_mapping() -> None:
    """Voice Live's beta text events map to output-text transcripts; other events delegate to the OpenAI mapper."""
    assert _map_voice_live_event({'type': 'response.text.delta', 'delta': 'hi'}) == Transcript(
        text='hi', is_final=False, output_text=True
    )
    assert _map_voice_live_event({'type': 'response.text.done', 'text': 'done'}) == Transcript(
        text='done', is_final=True, output_text=True
    )
    # Missing / non-string payloads degrade to an empty transcript rather than raising.
    assert _map_voice_live_event({'type': 'response.text.delta'}) == Transcript(
        text='', is_final=False, output_text=True
    )
    assert _map_voice_live_event({'type': 'response.text.done', 'text': 123}) == Transcript(
        text='', is_final=True, output_text=True
    )
    # A non-text event is delegated to the shared OpenAI mapper (an unknown type maps to `None`).
    assert _map_voice_live_event({'type': 'some.unknown.event'}) is None


def test_voice_live_silently_ignores_openai_only_settings() -> None:
    """OpenAI-only settings inherited by `AzureRealtimeModelSettings` are dropped on the Voice Live path."""
    provider = AzureProvider(azure_endpoint='https://r.services.ai.azure.com', api_version='2024-10-01', api_key='k')
    model = AzureRealtimeModel('gpt-realtime', provider=provider)
    config = model._session_config(  # pyright: ignore[reportPrivateUsage]
        'hi',
        None,
        AzureRealtimeModelSettings(
            azure_voice_live=True,
            openai_output_speed=1.5,
            openai_input_noise_reduction='near_field',
        ),
    )
    # The Voice Live session config is built from a fixed field set; the OpenAI-only knobs don't appear.
    assert 'speed' not in config
    assert 'output_audio' not in config
    assert 'noise_reduction' not in config


def test_infer_provider_from_bare_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    # The realtime model speaks only the GA `/openai/v1` protocol and never uses the provider's SDK
    # client, so inferring the provider from a bare resource endpoint must not demand the unrelated
    # `api_version` the SDK client would need.
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://resource.openai.azure.com')
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'azure-key')
    monkeypatch.delenv('OPENAI_API_VERSION', raising=False)

    model = AzureRealtimeModel('gpt-realtime')

    assert model._realtime_url() == (  # pyright: ignore[reportPrivateUsage]
        'wss://resource.openai.azure.com/openai/v1/realtime?model=gpt-realtime'
    )


def test_infer_provider_with_api_version_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # With `OPENAI_API_VERSION` set, the standard provider inference works and the realtime URL is
    # still derived from the endpoint's host.
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://resource.openai.azure.com')
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'azure-key')
    monkeypatch.setenv('OPENAI_API_VERSION', '2024-10-01')

    model = AzureRealtimeModel('gpt-realtime')

    assert model._realtime_url() == (  # pyright: ignore[reportPrivateUsage]
        'wss://resource.openai.azure.com/openai/v1/realtime?model=gpt-realtime'
    )


def test_infer_provider_with_v1_endpoint_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://resource.openai.azure.com/openai/v1')
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'azure-key')
    monkeypatch.delenv('OPENAI_API_VERSION', raising=False)

    model = AzureRealtimeModel('gpt-realtime')

    assert model._realtime_url() == (  # pyright: ignore[reportPrivateUsage]
        'wss://resource.openai.azure.com/openai/v1/realtime?model=gpt-realtime'
    )
