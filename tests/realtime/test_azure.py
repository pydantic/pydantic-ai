"""Network-free tests for the Azure OpenAI realtime model."""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import ToolDefinition

from ..conftest import try_import

with try_import() as imports_successful:
    from openai.types.realtime.realtime_audio_config_output import VoiceID

    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.realtime import RealtimeSessionErrorEvent, TurnDetection
    from pydantic_ai.realtime._base import OutputTranscript
    from pydantic_ai.realtime.azure import (
        AzureRealtimeConnection,
        AzureRealtimeModel,
        AzureRealtimeModelSettings,
        SemanticVAD,
        ServerVAD,
        _map_voice_live_event,  # pyright: ignore[reportPrivateUsage]
    )

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai / websockets not installed')


class _DroppedWebSocket:
    """A socket that reports an abnormal close as soon as it is iterated."""

    async def __aiter__(self) -> AsyncIterator[str]:
        raise OSError('dropped')
        yield ''  # pragma: no cover


def test_model_is_not_exported_from_the_realtime_package() -> None:
    # Concrete providers live in submodules (see the package docstring), so neither this model nor
    # `OpenAIRealtimeModel` is reachable from `pydantic_ai.realtime` itself.
    import pydantic_ai.realtime

    with pytest.raises(AttributeError):
        getattr(pydantic_ai.realtime, 'AzureRealtimeModel')


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
        'wss://resource.openai.azure.com/openai/v1/realtime?model=gpt%20realtime'
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
        openai_voice='alloy',
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


def test_voice_live_rejects_openai_custom_voice_id() -> None:
    """Voice Live addresses a voice by provider + name, so an OpenAI custom `VoiceID` fails loudly."""
    provider = AzureProvider(
        azure_endpoint='https://resource.services.ai.azure.com',
        api_version='2026-04-10',
        api_key='azure-key',
    )
    model = AzureRealtimeModel('phi-4-mm-realtime', provider=provider)
    settings = AzureRealtimeModelSettings(azure_voice_live=True, openai_voice=VoiceID(id='voice_custom'))

    with pytest.raises(UserError, match='does not accept an OpenAI custom `VoiceID`'):
        model._session_config('Be concise.', None, settings)  # pyright: ignore[reportPrivateUsage]


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
    assert _map_voice_live_event({'type': 'response.text.delta', 'delta': 'hi'}) == OutputTranscript(
        text='hi', is_final=False, output_text=True
    )
    assert _map_voice_live_event({'type': 'response.text.done', 'text': 'done'}) == OutputTranscript(
        text='done', is_final=True, output_text=True
    )
    # Missing / non-string payloads degrade to an empty transcript rather than raising.
    assert _map_voice_live_event({'type': 'response.text.delta'}) == OutputTranscript(
        text='', is_final=False, output_text=True
    )
    assert _map_voice_live_event({'type': 'response.text.done', 'text': 123}) == OutputTranscript(
        text='', is_final=True, output_text=True
    )
    # A non-text event is delegated to the shared OpenAI mapper (an unknown type maps to `None`).
    assert _map_voice_live_event({'type': 'some.unknown.event'}) is None


def test_voice_live_text_events_keep_item_id() -> None:
    """Voice Live's text frames carry `item_id`, and it must survive the mapping like OpenAI's do.

    Regression: dropping it left the recorded `TextPart` with no provider id, and — because the session
    detects a new output item by comparing `item_id` — stopped a second reply in one response from
    finalizing the first, so two replies accumulated into a single part.
    """
    assert _map_voice_live_event({'type': 'response.text.delta', 'delta': 'hi', 'item_id': 'item-1'}) == (
        OutputTranscript(text='hi', is_final=False, item_id='item-1', output_text=True)
    )
    assert _map_voice_live_event({'type': 'response.text.done', 'text': 'hi there', 'item_id': 'item-1'}) == (
        OutputTranscript(text='hi there', is_final=True, item_id='item-1', output_text=True)
    )
    # An absent or empty id stays `None` rather than becoming a falsy provider id.
    assert _map_voice_live_event({'type': 'response.text.delta', 'delta': 'x', 'item_id': ''}) == (
        OutputTranscript(text='x', is_final=False, item_id=None, output_text=True)
    )
    assert _map_voice_live_event({'type': 'response.text.done', 'text': 'x'}) == (
        OutputTranscript(text='x', is_final=True, item_id=None, output_text=True)
    )


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
            openai_truncation='auto',
            openai_turn_detection=SemanticVAD(eagerness='high'),
            thinking='low',
            parallel_tool_calls=False,
        ),
    )
    # The Voice Live session config is built from a fixed field set; the OpenAI-only knobs don't appear,
    # under their OpenAI names or the names Voice Live's own session object uses for the two that have a
    # counterpart (`input_audio_noise_reduction`, `truncation_strategy` — see the class docstring).
    assert 'speed' not in config
    assert 'output_audio' not in config
    assert 'noise_reduction' not in config
    assert 'input_audio_noise_reduction' not in config
    assert 'truncation' not in config
    assert 'truncation_strategy' not in config
    assert 'reasoning' not in config
    assert 'parallel_tool_calls' not in config
    # `openai_turn_detection` is *not* what configures Voice Live's VAD; the default server VAD stands.
    assert config['turn_detection']['type'] == 'server_vad'


def test_sideband_url_uses_the_ga_realtime_path() -> None:
    """The sideband control URL follows the resource endpoint, like the session URL.

    Regression: deriving it from the provider's `base_url` instead of the shared
    `_realtime_ws_base()` seam dropped Azure's `/openai/v1` whenever the endpoint wasn't already in
    the GA form, and the sideband dialed a path the resource doesn't serve.
    """
    provider = AzureProvider(
        azure_endpoint='https://resource.openai.azure.com', api_version='2024-10-01', api_key='azure-key'
    )
    model = AzureRealtimeModel('gpt-realtime', provider=provider)

    # The provider's own `base_url` is the SDK's `/openai/` form, which is exactly what made deriving
    # the sideband URL from it wrong.
    assert provider.base_url == 'https://resource.openai.azure.com/openai/'

    assert model._sideband_url('rtc_123') == (  # pyright: ignore[reportPrivateUsage]
        'wss://resource.openai.azure.com/openai/v1/realtime?call_id=rtc_123'
    )


@pytest.mark.anyio
async def test_connection_names_azure_not_openai() -> None:
    # The GA protocol is shared with OpenAI, but the vendor in a connection's messages must not be:
    # someone debugging a dropped or rejected Azure call would be sent to the wrong service.
    conn = AzureRealtimeConnection(_DroppedWebSocket())  # type: ignore[arg-type]
    events = [event async for event in conn]
    assert events == [
        RealtimeSessionErrorEvent(message='Azure OpenAI Realtime connection closed: dropped', recoverable=False)
    ]
    with pytest.raises(UserError, match='Azure OpenAI Realtime does not support'):
        await conn.send(cast('Any', object()))
    # The provider stamped onto content the connection can't carry names Azure too.
    assert conn._provider_name == 'azure'  # pyright: ignore[reportPrivateUsage]
    assert AzureRealtimeModel._connection_type is AzureRealtimeConnection  # pyright: ignore[reportPrivateUsage]


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
