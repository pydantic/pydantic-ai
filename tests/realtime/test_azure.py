"""Network-free tests for the Azure OpenAI realtime model."""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from pydantic import ValidationError

from pydantic_ai.exceptions import UserError
from pydantic_ai.settings import ThinkingLevel
from pydantic_ai.tools import ToolDefinition

from ..conftest import try_import

with try_import() as imports_successful:
    from openai.types.realtime.realtime_audio_config_output import VoiceID

    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.realtime import RealtimeSessionErrorEvent, TurnDetection
    from pydantic_ai.realtime.azure import (
        AzureRealtimeConnection,
        AzureRealtimeModel,
        AzureRealtimeModelProfile,
        AzureRealtimeModelSettings,
        AzureVoiceLiveVoice,
        SemanticVAD,
        ServerVAD,
        _default_azure_realtime_apis,  # pyright: ignore[reportPrivateUsage]
        _map_voice_live_event,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.realtime.codec import OutputTranscript

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
    assert AzureRealtimeModel('gpt-realtime').system == 'azure'


def test_non_azure_provider_instance_is_rejected() -> None:
    # A non-Azure `Provider` *instance* (not just the `provider='...'` string) must fail fast with a clear
    # `UserError` at construction, rather than an `AssertionError` deep inside later.
    with pytest.raises(UserError, match='requires an `AzureProvider`'):
        AzureRealtimeModel('gpt-realtime', provider=OpenAIProvider(api_key='x'))


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


async def test_voice_live_url_and_auth_headers() -> None:
    provider = AzureProvider(
        azure_endpoint='https://resource.services.ai.azure.com',
        api_version='2026-04-10',
        api_key='azure-key',
    )
    settings = AzureRealtimeModelSettings(azure_voice_live=True)
    model = AzureRealtimeModel('gpt realtime', provider=provider, settings=settings)

    assert model._realtime_url(settings) == (  # pyright: ignore[reportPrivateUsage]
        'wss://resource.services.ai.azure.com/voice-live/realtime?api-version=2026-07-15&model=gpt+realtime'
    )
    assert await model._auth_headers() == {'api-key': 'azure-key'}  # pyright: ignore[reportPrivateUsage]


def _azure_provider() -> AzureProvider:
    return AzureProvider(
        azure_endpoint='https://resource.services.ai.azure.com',
        api_version='2026-04-10',
        api_key='azure-key',
        voice_live_endpoint='https://voice.services.ai.azure.com',
        voice_live_api_key='voice-key',
    )


@pytest.mark.parametrize(
    'model_name,azure_voice_live,expected',
    [
        # Served by both APIs: defaults to the GA path, `azure_voice_live=True` selects Voice Live.
        ('gpt-realtime', None, 'ga'),
        ('gpt-realtime', True, 'voice_live'),
        ('gpt-realtime-mini', None, 'ga'),
        ('gpt-realtime-1.5', True, 'voice_live'),
        ('gpt-realtime-2', None, 'ga'),
        ('gpt-realtime-2', True, 'voice_live'),
        ('gpt-realtime-2.1-mini', True, 'voice_live'),
        # GA-only: defaults to GA, `azure_voice_live=True` is rejected before connecting.
        ('gpt-realtime-translate', None, 'ga'),
        ('gpt-4o-realtime-preview', True, 'error'),
        ('gpt-realtime-translate', True, 'error'),
        # Voice-Live-only: auto-routed to Voice Live whether or not the setting is passed.
        ('phi4-mm-realtime', None, 'voice_live'),
        ('azure-realtime', None, 'voice_live'),
        ('gpt-5', None, 'voice_live'),  # cascade chat model, served only through Voice Live
        ('gpt-4.1-mini', None, 'voice_live'),
        # Unrecognized (e.g. a future model): GA by default, Voice Live only when asked.
        ('gpt-realtime-3', None, 'ga'),
        ('gpt-realtime-3', True, 'voice_live'),
    ],
)
def test_azure_realtime_api_routing(model_name: str, azure_voice_live: bool | None, expected: str) -> None:
    settings = AzureRealtimeModelSettings(azure_voice_live=azure_voice_live) if azure_voice_live is not None else None
    model = AzureRealtimeModel(model_name, provider=_azure_provider(), settings=settings)
    if expected == 'error':
        with pytest.raises(UserError, match='cannot be used with it'):
            model._realtime_url(settings)  # pyright: ignore[reportPrivateUsage]
    else:
        url = model._realtime_url(settings)  # pyright: ignore[reportPrivateUsage]
        assert ('/voice-live/realtime' in url) == (expected == 'voice_live')


def test_azure_realtime_profile_override_routes_unconventional_deployment() -> None:
    # A deployment named after nothing recognizable is unknown (GA-default), but a `profile=` override
    # naming its serving APIs makes routing and validation work — the documented escape hatch.
    provider = _azure_provider()
    assert AzureRealtimeModel('my-voice-bot', provider=provider)._realtime_url() == (  # pyright: ignore[reportPrivateUsage]
        'wss://resource.services.ai.azure.com/openai/v1/realtime?model=my-voice-bot'
    )
    voice_live_only = AzureRealtimeModelProfile(azure_realtime_apis=frozenset({'voice_live'}))
    model = AzureRealtimeModel('my-voice-bot', provider=provider, profile=voice_live_only)
    assert '/voice-live/realtime' in model._realtime_url()  # pyright: ignore[reportPrivateUsage]
    # A Voice-Live-only model can't do GA-style browser WebRTC, so the profile drops the flag.
    assert model.profile.get('supports_webrtc') is False


def test_azure_realtime_apis_default_absent_for_unknown_model() -> None:
    model = AzureRealtimeModel('gpt-realtime-3', provider=_azure_provider())
    assert 'azure_realtime_apis' not in model.profile


@pytest.mark.parametrize(
    'model_name,expected',
    [
        # Served by both APIs, so unconstrained — including point releases and dated snapshots.
        ('gpt-realtime-2', None),
        ('gpt-realtime-2.1-mini', None),
        # A GA-only base is matched at a boundary: its dated snapshot is covered too.
        ('gpt-realtime-translate-2026-05-07', frozenset({'azure_openai'})),
        # ...but not a longer name that merely starts with it.
        ('gpt-50', None),
        # Cascade families cover their point releases (`.`-delimited) as well as `-`-suffixed variants.
        ('gpt-5.2-chat', frozenset({'voice_live'})),
        ('gpt-4o-mini', frozenset({'voice_live'})),
        # A GA-only `-realtime` variant is matched before the bare cascade name it also starts with.
        ('gpt-4o-realtime-preview', frozenset({'azure_openai'})),
        # The data-zone `gpt-realtime` deployments exist only on Voice Live.
        ('gpt-realtime-datazone', frozenset({'voice_live'})),
        ('gpt-realtime-1.5-datazone', frozenset({'voice_live'})),
    ],
)
def test_azure_realtime_apis_name_boundary_matching(model_name: str, expected: frozenset[str] | None) -> None:
    assert _default_azure_realtime_apis(model_name) == expected


def test_voice_live_session_config_options() -> None:
    provider = AzureProvider(
        azure_endpoint='https://resource.services.ai.azure.com',
        api_version='2026-04-10',
        api_key='azure-key',
    )
    model = AzureRealtimeModel('phi-4-mm-realtime', provider=provider)
    settings = AzureRealtimeModelSettings(
        azure_voice_live=True,
        azure_voice_live_turn_detection=ServerVAD(type='server_vad', silence_duration_ms=750),
        input_transcription_model=None,
        openai_voice='alloy',
        max_tokens=123,
        tool_choice='auto',
    )

    config = model._session_config(  # pyright: ignore[reportPrivateUsage]
        'Be concise.',
        [ToolDefinition(name='lookup', parameters_json_schema={'type': 'object'})],
        model_settings=settings,
    )

    assert config['turn_detection']['silence_duration_ms'] == 750
    assert 'input_audio_transcription' not in config
    assert config['voice'] == {'type': 'openai', 'name': 'alloy'}
    assert config['max_response_output_tokens'] == 123
    assert config['tool_choice'] == 'auto'
    assert config['tools'][0]['name'] == 'lookup'

    config = model._session_config(  # pyright: ignore[reportPrivateUsage]
        '',
        None,
        model_settings=AzureRealtimeModelSettings(
            azure_voice_live=True, turn_detection=TurnDetection(sensitivity='high')
        ),
    )
    assert config['turn_detection']['threshold'] == 0.3
    assert config['input_audio_transcription'] == {'model': 'azure-speech'}


def test_voice_live_maps_session_settings() -> None:
    """The settings Voice Live has a counterpart for reach its session config; unset, they're left out."""
    provider = AzureProvider(
        azure_endpoint='https://resource.services.ai.azure.com',
        api_version='2026-04-10',
        api_key='azure-key',
    )
    model = AzureRealtimeModel('gpt-realtime', provider=provider)

    def config(**settings: Any) -> dict[str, Any]:
        return model._session_config(  # pyright: ignore[reportPrivateUsage]
            '', None, model_settings=AzureRealtimeModelSettings(azure_voice_live=True, **settings)
        )

    baseline = config()
    for key in ('input_audio_noise_reduction', 'temperature', 'reasoning_effort'):
        assert key not in baseline

    mapped = config(
        openai_input_noise_reduction='far_field',
        openai_turn_detection=SemanticVAD(type='semantic_vad', eagerness='high'),
        azure_voice_live_temperature=0.7,
    )
    assert mapped['input_audio_noise_reduction'] == {'type': 'far_field'}
    assert mapped['turn_detection'] == {
        'type': 'semantic_vad',
        'eagerness': 'high',
        'create_response': True,
        'interrupt_response': True,
    }
    assert mapped['temperature'] == 0.7

    # `openai_turn_detection` overrides the shared `turn_detection`, and the Voice Live setting overrides both.
    assert config(openai_turn_detection=None, turn_detection=True)['turn_detection'] is None
    assert (
        config(
            azure_voice_live_turn_detection=ServerVAD(type='server_vad'),
            openai_turn_detection=SemanticVAD(type='semantic_vad'),
        )['turn_detection']['type']
        == 'server_vad'
    )

    # `gpt-realtime` doesn't reason (Voice Live rejects `reasoning_effort` for it), so `thinking` is dropped.
    assert 'reasoning_effort' not in config(thinking='high')


@pytest.mark.parametrize(
    'model_name,profile,expected_type',
    [
        # Cascade models reject OpenAI's semantic VAD, so Voice Live's own is used instead.
        ('gpt-5', None, 'azure_semantic_vad'),
        ('phi4-mm-realtime', None, 'azure_semantic_vad'),
        # Native-audio models keep it, `eagerness` included.
        ('gpt-realtime', None, 'semantic_vad'),
        ('azure-realtime', None, 'semantic_vad'),
        # Decided by the profile, not the deployment name: an unrecognized name is left alone...
        ('voice-production', None, 'semantic_vad'),
        # ...and a cascade deployed under another name is marked through `profile=`.
        ('my-voice-bot', AzureRealtimeModelProfile(azure_voice_live_cascade=True), 'azure_semantic_vad'),
    ],
)
def test_voice_live_cascade_semantic_vad_uses_azure_semantic_vad(
    model_name: str, profile: AzureRealtimeModelProfile | None, expected_type: str
) -> None:
    provider = AzureProvider(
        azure_endpoint='https://resource.services.ai.azure.com',
        api_version='2026-04-10',
        api_key='azure-key',
    )
    model = AzureRealtimeModel(model_name, provider=provider, profile=profile)
    settings = AzureRealtimeModelSettings(
        azure_voice_live=True,
        openai_turn_detection=SemanticVAD(type='semantic_vad', eagerness='high', interrupt_response=False),
    )
    config = model._session_config('', None, model_settings=settings)  # pyright: ignore[reportPrivateUsage]
    if expected_type == 'azure_semantic_vad':
        assert config['turn_detection'] == {
            'type': 'azure_semantic_vad',
            'create_response': True,
            'interrupt_response': False,
        }
    else:
        assert config['turn_detection'] == {
            'type': 'semantic_vad',
            'eagerness': 'high',
            'create_response': True,
            'interrupt_response': False,
        }
    # Server VAD and disabled detection pass through unchanged.
    config = model._session_config(  # pyright: ignore[reportPrivateUsage]
        '', None, model_settings=AzureRealtimeModelSettings(azure_voice_live=True, turn_detection=False)
    )
    assert config['turn_detection'] is None


@pytest.mark.parametrize(
    'model_name,thinking,temperature_sent',
    [
        # Can't turn reasoning off, so the temperature is always dropped.
        ('gpt-5', None, False),
        ('gpt-5', False, False),
        # Can turn reasoning off, and doesn't reason by default: kept unless reasoning is asked for.
        ('gpt-5.2', None, True),
        ('gpt-5.2', False, True),
        ('gpt-5.2', 'low', False),
        # Doesn't reason at all.
        ('gpt-4.1', 'high', True),
        # A native-audio reasoning model takes a temperature while reasoning (verified live).
        ('gpt-realtime-2', 'high', True),
    ],
)
def test_voice_live_temperature_dropped_while_reasoning(
    model_name: str, thinking: ThinkingLevel | None, temperature_sent: bool
) -> None:
    """Like a standard OpenAI run, a temperature is dropped with a warning while the model reasons."""
    provider = AzureProvider(
        azure_endpoint='https://resource.services.ai.azure.com',
        api_version='2026-04-10',
        api_key='azure-key',
    )
    model = AzureRealtimeModel(model_name, provider=provider)
    settings = AzureRealtimeModelSettings(azure_voice_live=True, azure_voice_live_temperature=0.5)
    if thinking is not None:
        settings['thinking'] = thinking
    if temperature_sent:
        config = model._session_config('', None, model_settings=settings)  # pyright: ignore[reportPrivateUsage]
        assert config['temperature'] == 0.5
    else:
        with pytest.warns(UserWarning, match='`azure_voice_live_temperature` is not supported while') as record:
            config = model._session_config('', None, model_settings=settings)  # pyright: ignore[reportPrivateUsage]
        assert 'temperature' not in config
        # Only a model that can turn reasoning off is pointed at `thinking=False`.
        remedy = 'Set `thinking=False`' if model_name == 'gpt-5.2' else 'cannot turn reasoning off'
        assert remedy in str(record[0].message)


@pytest.mark.parametrize(
    'model_name,thinking,expected',
    [
        # Reasoning cascade chat models take their thinking support from the chat model profile.
        ('gpt-5', 'high', 'high'),
        ('gpt-5', False, 'none'),
        ('gpt-5.2', True, 'medium'),
        # Non-reasoning models: Voice Live accepts the effort for `gpt-4.1`, then fails every response.
        ('gpt-4.1', 'high', None),
        ('gpt-5-chat', 'high', None),
        ('phi4-mm-realtime', 'high', None),
    ],
)
def test_voice_live_thinking_maps_to_reasoning_effort(
    model_name: str, thinking: ThinkingLevel, expected: str | None
) -> None:
    provider = AzureProvider(
        azure_endpoint='https://resource.services.ai.azure.com',
        api_version='2026-04-10',
        api_key='azure-key',
    )
    model = AzureRealtimeModel(model_name, provider=provider)
    settings = AzureRealtimeModelSettings(azure_voice_live=True, thinking=thinking)
    config = model._session_config('', None, model_settings=settings)  # pyright: ignore[reportPrivateUsage]
    assert config.get('reasoning_effort') == expected
    assert model.profile.get('supports_thinking') is (expected is not None)


def test_voice_live_cascade_thinking_respects_profile_override() -> None:
    """A user `profile=` still has the last word on a cascade model's thinking support."""
    provider = AzureProvider(
        azure_endpoint='https://resource.services.ai.azure.com',
        api_version='2026-04-10',
        api_key='azure-key',
    )
    model = AzureRealtimeModel('gpt-5', provider=provider, profile={'supports_thinking': False})
    assert model.profile.get('supports_thinking') is False
    config = model._session_config('', None, model_settings=AzureRealtimeModelSettings(thinking='high'))  # pyright: ignore[reportPrivateUsage]
    assert 'reasoning_effort' not in config


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
        model._session_config('Be concise.', None, model_settings=settings)  # pyright: ignore[reportPrivateUsage]


def test_voice_live_azure_voice() -> None:
    """`azure_voice_live_voice` selects an Azure voice, taking precedence over `openai_voice`."""
    provider = AzureProvider(
        azure_endpoint='https://resource.services.ai.azure.com',
        api_version='2026-04-10',
        api_key='azure-key',
    )
    model = AzureRealtimeModel('gpt-5', provider=provider)

    def voice(settings: AzureRealtimeModelSettings) -> Any:
        return model._session_config('', None, model_settings=settings).get('voice')  # pyright: ignore[reportPrivateUsage]

    # A bare name is a standard Azure text-to-speech voice.
    assert voice(AzureRealtimeModelSettings(azure_voice_live_voice='en-US-AvaMultilingualNeural')) == {
        'type': 'azure-standard',
        'name': 'en-US-AvaMultilingualNeural',
    }
    # The full object is forwarded as-is, and wins over `openai_voice` (which cascade models reject).
    custom = AzureVoiceLiveVoice(type='azure-custom', name='my-voice', endpoint_id='endpoint', rate='1.2')
    assert voice(AzureRealtimeModelSettings(azure_voice_live_voice=custom, openai_voice='alloy')) == custom
    # Without it, the session sends no voice and Voice Live uses the model's default.
    assert voice(AzureRealtimeModelSettings()) is None
    # It has no GA counterpart, so the GA session config ignores it.
    ga_model = AzureRealtimeModel('gpt-realtime', provider=provider)
    ga_config = ga_model._session_config(  # pyright: ignore[reportPrivateUsage]
        '', None, model_settings=AzureRealtimeModelSettings(azure_voice_live_voice='en-US-AvaNeural')
    )
    assert 'voice' not in ga_config['audio']['output']


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
    assert 'api-version=2026-07-15' in url


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
        assert model._realtime_url(vl) == 'wss://r.openai.azure.com/voice-live/realtime?api-version=2026-07-15&model=m'  # pyright: ignore[reportPrivateUsage]


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


def test_voice_live_warning_event_is_surfaced() -> None:
    """Voice Live's informational `warning` event becomes a Python warning, and the session goes on."""
    event = {'type': 'warning', 'warning': {'message': 'Heads up', 'code': 'some_code', 'param': 'session.voice'}}
    with pytest.warns(
        UserWarning, match=r"^Azure AI Voice Live warning: Heads up \(code='some_code', param='session.voice'\)$"
    ):
        assert _map_voice_live_event(event) is None
    with pytest.warns(UserWarning, match=r'^Azure AI Voice Live warning: Heads up$'):
        assert _map_voice_live_event({'type': 'warning', 'warning': {'message': 'Heads up'}}) is None
    # A malformed one is reported like any other malformed frame, rather than warning with no message.
    with pytest.raises(ValidationError):
        _map_voice_live_event({'type': 'warning', 'warning': {}})


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


def test_voice_live_profile_layers_over_the_user_profile() -> None:
    """Voice Live's profile correction is applied *after* the user's `profile=`, so it wins.

    `RealtimeModel.profile` now ends with a user layer, and the Voice Live override sits on top of it
    via `super().profile`. That order is the safe one: `supports_webrtc=False` under Voice Live is a
    fact about the transport, not a default, so a user claiming otherwise must not produce a session
    that then fails at the signaling call.
    """
    provider = AzureProvider(azure_endpoint='https://r.openai.azure.com/openai/v1', api_key='k')
    voice_live = AzureRealtimeModelSettings(azure_voice_live=True)

    # The user layer reaches keys Voice Live doesn't speak for.
    model = AzureRealtimeModel('gpt-realtime', provider=provider, profile={'supports_text_output': False})
    assert model.profile.get('supports_text_output') is False
    assert model.profile.get('supports_webrtc') is True

    # ...but not `supports_webrtc` under Voice Live, in either `profile=` form.
    forced = AzureRealtimeModel(
        'gpt-realtime', provider=provider, settings=voice_live, profile={'supports_webrtc': True}
    )
    assert forced.profile.get('supports_webrtc') is False
    callable_form = AzureRealtimeModel(
        'gpt-realtime',
        provider=provider,
        settings=voice_live,
        profile=lambda resolved: {**resolved, 'supports_webrtc': True},
    )
    assert callable_form.profile.get('supports_webrtc') is False

    # Unrelated user keys still apply under Voice Live.
    both = AzureRealtimeModel(
        'gpt-realtime', provider=provider, settings=voice_live, profile={'supports_text_output': False}
    )
    assert both.profile.get('supports_text_output') is False
    assert both.profile.get('supports_webrtc') is False


def test_voice_live_silently_ignores_openai_only_settings() -> None:
    """The OpenAI-only settings Voice Live has no counterpart for are dropped on the Voice Live path."""
    provider = AzureProvider(azure_endpoint='https://r.services.ai.azure.com', api_version='2024-10-01', api_key='k')
    model = AzureRealtimeModel('gpt-realtime', provider=provider)
    config = model._session_config(  # pyright: ignore[reportPrivateUsage]
        'hi',
        None,
        model_settings=AzureRealtimeModelSettings(
            azure_voice_live=True,
            openai_output_speed=1.5,
            openai_truncation='auto',
        ),
    )
    # The Voice Live session config is built from a fixed field set; these knobs don't appear, under
    # their OpenAI names or under Voice Live's own `truncation_strategy` (see the class docstring).
    assert 'speed' not in config
    assert 'output_audio' not in config
    assert 'truncation' not in config
    assert 'truncation_strategy' not in config


@pytest.mark.parametrize('parallel_tool_calls,expected', [(False, False), (True, None), (None, None)])
def test_voice_live_parallel_tool_calls(parallel_tool_calls: bool | None, expected: bool | None) -> None:
    """Only `False` is sent: `True` is the server default, and some `gpt-realtime` models reject it."""
    provider = AzureProvider(azure_endpoint='https://r.services.ai.azure.com', api_version='2024-10-01', api_key='k')
    model = AzureRealtimeModel('gpt-realtime', provider=provider)
    settings = AzureRealtimeModelSettings(azure_voice_live=True)
    if parallel_tool_calls is not None:
        settings['parallel_tool_calls'] = parallel_tool_calls
    config = model._session_config('', None, model_settings=settings)  # pyright: ignore[reportPrivateUsage]
    assert config.get('parallel_tool_calls') == expected


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


def test_profile_override_corrects_a_deployment_name(monkeypatch: pytest.MonkeyPatch) -> None:
    # Azure's `model` is the *deployment* name, a user-chosen string that need not name the model, and
    # the profile is inferred from it (reasoning effort is only accepted by `gpt-realtime-2*`). A
    # deployment named anything else therefore loses `thinking` — `profile=` is the way to correct it,
    # mirroring `profile=` on a standard `Model`.
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://resource.openai.azure.com/openai/v1')
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'azure-key')

    inferred = AzureRealtimeModel('voice-prod', settings={'thinking': 'low'})
    assert inferred.profile.get('supports_thinking') is False
    assert 'reasoning' not in inferred._session_config('', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]

    corrected = AzureRealtimeModel('voice-prod', settings={'thinking': 'low'}, profile={'supports_thinking': True})
    assert corrected.profile.get('supports_thinking') is True
    assert corrected._session_config('', None, model_settings=None)['reasoning'] == {'effort': 'low'}  # pyright: ignore[reportPrivateUsage]
    # Everything the provider said is still there — `profile=` is a layer, not a replacement.
    assert corrected.profile.get('supports_image_input') is True


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
