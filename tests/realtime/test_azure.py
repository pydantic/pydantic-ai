"""Network-free tests for the Azure AI Voice Live realtime provider."""

from __future__ import annotations as _annotations

import base64
import json
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.realtime import AudioDelta, RealtimeModelProfile, ToolCall, Transcript, TurnDetection
from pydantic_ai.tools import ToolDefinition

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.providers.azure_voicelive import AzureVoiceLiveProvider
    from pydantic_ai.realtime import azure as rt_azure
    from pydantic_ai.realtime.azure import (
        AzureRealtimeConnection,
        AzureRealtimeModel,
        AzureRealtimeModelSettings,
        map_event,
    )

pytestmark = pytest.mark.skipif(not imports_successful(), reason='websockets not installed')


def _provider(**kwargs: Any) -> AzureVoiceLiveProvider:
    return AzureVoiceLiveProvider(
        endpoint='https://resource.services.ai.azure.com',
        api_version='2026-04-10',
        api_key='k',
        **kwargs,
    )


def _model(settings: AzureRealtimeModelSettings | None = None, **kwargs: Any) -> AzureRealtimeModel:
    return AzureRealtimeModel(provider=_provider(), settings=settings, **kwargs)


def _connect(
    model: AzureRealtimeModel,
    instructions: str,
    *,
    messages: Sequence[ModelMessage] | None = None,
    tools: list[ToolDefinition] | None = None,
) -> AbstractAsyncContextManager[AzureRealtimeConnection]:
    return model.connect(
        messages=[*(messages or ()), ModelRequest(parts=[], instructions=instructions)],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(function_tools=tools or []),
    )


def test_map_text_output_events() -> None:
    assert map_event({'type': 'response.text.delta', 'delta': 'hel'}) == Transcript(
        text='hel', is_final=False, output_text=True
    )
    assert map_event({'type': 'response.text.done', 'text': 'hello'}) == Transcript(
        text='hello', is_final=True, output_text=True
    )
    assert map_event({'type': 'response.text.delta', 'delta': 1}) == Transcript(
        text='', is_final=False, output_text=True
    )


def test_map_delegates_openai_compatible_events() -> None:
    payload = base64.b64encode(b'\x01\x02').decode()
    assert map_event({'type': 'response.audio.delta', 'delta': payload}) == AudioDelta(data=b'\x01\x02')
    assert map_event(
        {'type': 'response.function_call_arguments.done', 'call_id': 'c1', 'name': 'weather', 'arguments': '{}'}
    ) == ToolCall(tool_call_id='c1', tool_name='weather', args='{}', response_usage_follows=True)


def test_connection_uses_azure_event_mapper() -> None:
    connection = AzureRealtimeConnection.__new__(AzureRealtimeConnection)
    assert connection._map_event({'type': 'response.text.done', 'text': 'hello'}) == Transcript(  # pyright: ignore[reportPrivateUsage]
        text='hello', is_final=True, output_text=True
    )


def test_profile() -> None:
    assert _model().profile == RealtimeModelProfile(
        supports_image_input=True,
        supports_manual_turn_control=True,
        supports_interruption=True,
        supports_output_truncation=True,
        supports_session_seeding=True,
        supports_seeding_images=True,
        supports_seeding_audio=True,
        audio_input_sample_rate=24000,
        audio_output_sample_rate=24000,
        supported_native_tools=frozenset(),
    )


def test_model_is_exported_from_realtime_package() -> None:
    from pydantic_ai.realtime import AzureRealtimeModel as ExportedAzureRealtimeModel

    assert ExportedAzureRealtimeModel is AzureRealtimeModel


def test_session_config_shape() -> None:
    settings = AzureRealtimeModelSettings(
        voice='alloy',
        output_modality='text',
        max_tokens=256,
        parallel_tool_calls=False,
        tool_choice='required',
        azure_turn_detection=rt_azure.SemanticVAD(eagerness='high'),
    )
    tool = ToolDefinition(name='weather', description='Weather', parameters_json_schema={'type': 'object'})
    config = _model(settings)._session_config('Be concise', [tool], None)  # pyright: ignore[reportPrivateUsage]
    assert config == {
        'instructions': 'Be concise',
        'modalities': ['text'],
        'input_audio_format': 'pcm16',
        'output_audio_format': 'pcm16',
        'input_audio_sampling_rate': 24000,
        'turn_detection': {
            'type': 'semantic_vad',
            'eagerness': 'high',
            'create_response': True,
            'interrupt_response': True,
        },
        'input_audio_transcription': {'model': 'whisper-1'},
        'voice': {'type': 'openai', 'name': 'alloy'},
        'tools': [
            {
                'type': 'function',
                'name': 'weather',
                'description': 'Weather',
                'parameters': {'type': 'object'},
            }
        ],
        'max_response_output_tokens': 256,
        'tool_choice': 'required',
    }


def test_session_config_defaults_to_audio_and_server_vad() -> None:
    config = _model()._session_config('hi', None, None)  # pyright: ignore[reportPrivateUsage]
    assert config['modalities'] == ['text', 'audio']
    assert config['turn_detection'] == {
        'type': 'server_vad',
        'create_response': True,
        'interrupt_response': True,
    }
    assert config['input_audio_transcription'] == {'model': 'whisper-1'}
    assert 'voice' not in config


def test_session_config_uses_azure_speech_for_cascaded_model() -> None:
    config = _model(model='gpt-4o')._session_config('hi', None, None)  # pyright: ignore[reportPrivateUsage]
    assert config['input_audio_transcription'] == {'model': 'azure-speech'}


def test_session_config_common_turn_detection_and_disabled_transcription() -> None:
    settings = AzureRealtimeModelSettings(
        turn_detection=TurnDetection(sensitivity='low', prefix_padding_ms=200, silence_duration_ms=600),
        input_transcription_model=None,
    )
    config = _model()._session_config('hi', None, settings)  # pyright: ignore[reportPrivateUsage]
    assert config['turn_detection'] == {
        'type': 'server_vad',
        'create_response': True,
        'interrupt_response': True,
        'threshold': 0.7,
        'prefix_padding_ms': 200,
        'silence_duration_ms': 600,
    }
    assert 'input_audio_transcription' not in config


def test_voice_live_websocket_url() -> None:
    assert rt_azure.azure_voice_live_websocket_url(
        'https://resource.services.ai.azure.com/custom?feature=yes',
        api_version='2026-04-10',
        model='gpt realtime',
    ) == (
        'wss://resource.services.ai.azure.com/custom/voice-live/realtime?'
        'feature=yes&api-version=2026-04-10&model=gpt+realtime'
    )


@pytest.mark.parametrize(
    ('endpoint', 'expected_scheme'),
    [('http://localhost:8080', 'ws'), ('ws://localhost:8080', 'ws')],
)
def test_voice_live_websocket_url_preserves_transport(endpoint: str, expected_scheme: str) -> None:
    url = rt_azure.azure_voice_live_websocket_url(endpoint, api_version='v1', model='voice')
    assert url == f'{expected_scheme}://localhost:8080/voice-live/realtime?api-version=v1&model=voice'


class FakeWebSocket:
    def __init__(self, incoming: list[Any]) -> None:
        self._incoming = list(incoming)
        self.sent: list[str] = []

    async def recv(self) -> Any:
        return self._incoming.pop(0)

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def __aiter__(self) -> AsyncIterator[Any]:
        while self._incoming:
            yield self._incoming.pop(0)


class FakeConnect:
    def __init__(self, websocket: FakeWebSocket) -> None:
        self.websocket = websocket
        self.url: str | None = None
        self.headers: dict[str, str] | None = None

    def __call__(self, url: str, *, additional_headers: dict[str, str] | None = None) -> FakeConnect:
        self.url = url
        self.headers = additional_headers
        return self

    async def __aenter__(self) -> FakeWebSocket:
        return self.websocket

    async def __aexit__(self, *exc: object) -> bool:
        return False


@pytest.mark.anyio
async def test_connect_url_auth_handshake_and_server_model(monkeypatch: pytest.MonkeyPatch) -> None:
    created = json.dumps({'type': 'session.created', 'session': {'model': 'gpt-realtime-global-standard'}})
    updated = json.dumps({'type': 'session.updated'})
    text = json.dumps({'type': 'response.text.done', 'text': 'hello'})
    websocket = FakeWebSocket([created, updated, text])
    fake_connect = FakeConnect(websocket)
    monkeypatch.setattr(rt_azure.websockets, 'connect', fake_connect)

    async with _connect(_model(model='gpt-realtime'), 'Be concise') as connection:
        assert connection.model_name == 'gpt-realtime-global-standard'
        events = [event async for event in connection]

    assert events == [Transcript(text='hello', is_final=True, output_text=True)]
    assert fake_connect.url == (
        'wss://resource.services.ai.azure.com/voice-live/realtime?api-version=2026-04-10&model=gpt-realtime'
    )
    assert fake_connect.headers == {'api-key': 'k'}
    assert json.loads(websocket.sent[0]) == {
        'type': 'session.update',
        'session': {
            'instructions': 'Be concise',
            'modalities': ['text', 'audio'],
            'input_audio_format': 'pcm16',
            'output_audio_format': 'pcm16',
            'input_audio_sampling_rate': 24000,
            'turn_detection': {
                'type': 'server_vad',
                'create_response': True,
                'interrupt_response': True,
            },
            'input_audio_transcription': {'model': 'whisper-1'},
        },
    }


@pytest.mark.anyio
async def test_connect_seeds_history_and_reconnects_without_server_model(monkeypatch: pytest.MonkeyPatch) -> None:
    handshake = [
        json.dumps({'type': 'session.created', 'session': {'model': ''}}),
        json.dumps({'type': 'session.updated'}),
    ]
    first = FakeWebSocket(handshake.copy())
    second = FakeWebSocket(handshake.copy())

    class _Reconnect:
        def __init__(self) -> None:
            self.websockets = [first, second]
            self.closed: list[FakeWebSocket] = []

        def __call__(self, url: str, *, additional_headers: dict[str, str] | None = None) -> _Reconnect:
            del url, additional_headers
            return self

        async def __aenter__(self) -> FakeWebSocket:
            return self.websockets.pop(0)

        async def __aexit__(self, *exc: object) -> bool:
            self.closed.append(first if not self.closed else second)
            return False

    reconnect = _Reconnect()
    monkeypatch.setattr(rt_azure.websockets, 'connect', reconnect)
    history = [ModelRequest(parts=[UserPromptPart(content='Earlier question')])]

    async with _connect(_model(), 'Be concise', messages=history) as connection:
        assert connection.model_name is None
        dial = connection._dial  # pyright: ignore[reportPrivateUsage]
        assert dial is not None
        await dial()

    assert reconnect.closed == [first, second]
    assert json.loads(first.sent[1])['item']['content'][0]['text'] == 'Earlier question'


@pytest.mark.anyio
async def test_connect_open_failure_has_no_context_to_close(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingConnect:
        def __call__(self, url: str, *, additional_headers: dict[str, str] | None = None) -> _FailingConnect:
            del url, additional_headers
            return self

        async def __aenter__(self) -> FakeWebSocket:
            raise OSError('refused')

        async def __aexit__(self, *exc: object) -> bool:  # pragma: no cover - never entered
            raise AssertionError

    monkeypatch.setattr(rt_azure.websockets, 'connect', _FailingConnect())
    with pytest.raises(OSError, match='refused'):
        async with _connect(_model(), 'x'):
            pass  # pragma: no cover


@pytest.mark.anyio
async def test_failed_redial_closes_previous_context_and_leaves_no_current_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = FakeWebSocket([json.dumps({'type': 'session.created'}), json.dumps({'type': 'session.updated'})])

    class _ReconnectThenFail:
        def __init__(self) -> None:
            self.calls = 0
            self.closed = False

        def __call__(self, url: str, *, additional_headers: dict[str, str] | None = None) -> _ReconnectThenFail:
            del url, additional_headers
            self.calls += 1
            return self

        async def __aenter__(self) -> FakeWebSocket:
            if self.calls == 2:
                raise OSError('redial refused')
            return websocket

        async def __aexit__(self, *exc: object) -> bool:
            self.closed = True
            return False

    reconnect = _ReconnectThenFail()
    monkeypatch.setattr(rt_azure.websockets, 'connect', reconnect)

    async with _connect(_model(), 'x') as connection:
        with pytest.raises(OSError, match='redial refused'):
            dial = connection._dial  # pyright: ignore[reportPrivateUsage]
            assert dial is not None
            await dial()

    assert reconnect.closed


def test_provider_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('AZURE_VOICELIVE_ENDPOINT', 'https://env-resource.services.ai.azure.com/')
    monkeypatch.setenv('AZURE_VOICELIVE_API_VERSION', '2026-04-10')
    monkeypatch.setenv('AZURE_VOICELIVE_API_KEY', 'env-key')
    provider = AzureVoiceLiveProvider()
    assert provider.base_url == 'https://env-resource.services.ai.azure.com'
    assert provider.api_version == '2026-04-10'
    assert provider.api_key == 'env-key'
    assert provider.client is None


@pytest.mark.parametrize(
    ('missing', 'message'),
    [
        ('AZURE_VOICELIVE_ENDPOINT', 'AZURE_VOICELIVE_ENDPOINT'),
        ('AZURE_VOICELIVE_API_VERSION', 'AZURE_VOICELIVE_API_VERSION'),
        ('AZURE_VOICELIVE_API_KEY', 'AZURE_VOICELIVE_API_KEY'),
    ],
)
def test_provider_requires_configuration(monkeypatch: pytest.MonkeyPatch, missing: str, message: str) -> None:
    values = {
        'AZURE_VOICELIVE_ENDPOINT': 'https://resource.services.ai.azure.com',
        'AZURE_VOICELIVE_API_VERSION': '2026-04-10',
        'AZURE_VOICELIVE_API_KEY': 'k',
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(missing)
    with pytest.raises(UserError, match=message):
        AzureVoiceLiveProvider()


def test_model_rejects_other_provider() -> None:
    with pytest.raises(UserError, match='requires an `AzureVoiceLiveProvider`'):
        AzureRealtimeModel(provider='openai')
