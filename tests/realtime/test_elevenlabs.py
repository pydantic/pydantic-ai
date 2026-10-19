"""Tests for the ElevenLabs realtime provider (provider, handshake, session config), all network-free.

ElevenLabs' realtime API is modelled on the OpenAI Realtime protocol, so these tests focus on what the
ElevenLabs provider adds on top of the shared OpenAI codec (exercised in `test_openai.py`).
"""

# TODO(elevenlabs-launch): the snapshots below pin the placeholder URL, auth header, and session shape; update
# them (`--inline-snapshot=fix`) once the constants in `realtime/elevenlabs.py` are filled, and add a test per
# divergence found (transcription events, usage extras, session-ending errors), as `test_xai.py` has.

from __future__ import annotations as _annotations

import json
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager

import pytest
from inline_snapshot import snapshot

from pydantic_ai.exceptions import ModelAPIError, UserError
from pydantic_ai.messages import ModelMessage, ModelRequest
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.realtime import WebRTCSession
from pydantic_ai.realtime.codec import OutputTranscript
from pydantic_ai.tools import ToolDefinition

from ..conftest import TestEnv, try_import
from .ws_helpers import collect_codec_events

with try_import() as imports_successful:
    from pydantic_ai.providers import infer_provider
    from pydantic_ai.providers.elevenlabs import ElevenLabsProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.realtime import elevenlabs as rt_elevenlabs
    from pydantic_ai.realtime.elevenlabs import (
        ElevenLabsRealtimeConnection,
        ElevenLabsRealtimeModel,
        ElevenLabsRealtimeModelSettings,
        map_event,
    )
    from pydantic_ai.realtime.openai import OpenAIRealtimeConnection

    from .test_openai import FakeConnect, FakeWebSocket, sdk_frame

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai / websockets not installed'),
    pytest.mark.anyio,
]


def _model(
    settings: ElevenLabsRealtimeModelSettings | None = None, model: str = 'eleven-voice'
) -> ElevenLabsRealtimeModel:
    return ElevenLabsRealtimeModel(model, provider=ElevenLabsProvider(api_key='k'), settings=settings)


def _connect(
    model: ElevenLabsRealtimeModel, instructions: str, *, messages: Sequence[ModelMessage] | None = None
) -> AbstractAsyncContextManager[OpenAIRealtimeConnection]:
    return model.connect(
        messages=[*(messages or ()), ModelRequest(parts=[], instructions=instructions)],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )


# --- provider -----------------------------------------------------------------------------------


def test_provider_reads_env_and_normalizes_base_url(env: TestEnv) -> None:
    env.set('ELEVENLABS_API_KEY', 'env-key')
    provider = infer_provider('elevenlabs')
    assert isinstance(provider, ElevenLabsProvider)
    assert provider.name == 'elevenlabs'
    assert provider.api_key == 'env-key'
    assert provider.base_url == 'https://api.elevenlabs.io'
    assert ElevenLabsProvider(api_key='k', base_url='https://api.eu.residency.elevenlabs.io/').base_url == (
        'https://api.eu.residency.elevenlabs.io'
    )


def test_provider_requires_api_key(env: TestEnv) -> None:
    env.remove('ELEVENLABS_API_KEY')
    with pytest.raises(UserError, match='ELEVENLABS_API_KEY'):
        ElevenLabsProvider()


def test_model_rejects_foreign_provider() -> None:
    with pytest.raises(UserError, match="requires an `ElevenLabsProvider` or `provider='elevenlabs'`; got 'openai'"):
        ElevenLabsRealtimeModel('eleven-voice', provider=OpenAIProvider(api_key='k'))  # pyright: ignore[reportArgumentType]


def test_model_identity() -> None:
    model = _model()
    assert model.model_name == 'eleven-voice'
    assert model.system == 'elevenlabs'
    assert model.base_url == 'https://api.elevenlabs.io'


def test_profile() -> None:
    assert _model().profile == snapshot(
        {
            'supports_image_input': False,
            'image_input_requires_response': False,
            'supports_manual_turn_control': True,
            'supports_interruption': True,
            'supports_output_truncation': False,
            'supports_text_output': False,
            'supports_session_seeding': True,
            'supports_webrtc': False,
            'supports_seeding_images': False,
            'supports_seeding_audio': False,
            'async_tool_call_mode': 'always',
            'supports_tool_return_schema': False,
            'supported_native_tools': frozenset(),
            'emits_input_speech_events': True,
            'synthesizes_turn_boundary': False,
            'responses_are_requests': True,
            'response_usage_covers_context': False,
            'audio_input_sample_rate': 24000,
            'audio_output_sample_rate': 24000,
            'context_window': None,
            'supports_thinking': False,
            'supports_async_tool_calls': True,
        }
    )


# --- session config -------------------------------------------------------------------------------


def test_session_config_shape() -> None:
    tools = [ToolDefinition(name='lookup', description='Look up', parameters_json_schema={'type': 'object'})]
    config = _model()._session_config('Be nice', tools, model_settings=None)  # pyright: ignore[reportPrivateUsage]
    assert config == snapshot(
        {
            'type': 'realtime',
            'instructions': 'Be nice',
            'output_modalities': ['audio'],
            'audio': {
                'input': {
                    'format': {'type': 'audio/pcm', 'rate': 24000},
                    'turn_detection': {'type': 'server_vad', 'create_response': True, 'interrupt_response': True},
                    'transcription': {'model': 'scribe_v2_realtime'},
                },
                'output': {'format': {'type': 'audio/pcm', 'rate': 24000}},
            },
            'tools': [
                {
                    'type': 'function',
                    'name': 'lookup',
                    'description': 'Look up',
                    'parameters': {'type': 'object'},
                }
            ],
        }
    )


def test_session_config_settings() -> None:
    settings = ElevenLabsRealtimeModelSettings(
        elevenlabs_voice='voice-id',
        openai_turn_detection={'type': 'server_vad', 'threshold': 0.4},
        input_transcription_model=None,
        max_tokens=100,
        parallel_tool_calls=False,
    )
    config = _model(settings)._session_config('', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]
    assert config['audio'] == snapshot(
        {
            'input': {
                'format': {'type': 'audio/pcm', 'rate': 24000},
                'turn_detection': {
                    'type': 'server_vad',
                    'create_response': True,
                    'interrupt_response': True,
                    'threshold': 0.4,
                },
            },
            'output': {'format': {'type': 'audio/pcm', 'rate': 24000}, 'voice': 'voice-id'},
        }
    )
    assert config['max_output_tokens'] == 100
    assert config['parallel_tool_calls'] is False


def test_session_config_ignores_openai_voice() -> None:
    settings = ElevenLabsRealtimeModelSettings(openai_voice='alloy')
    config = _model(settings)._session_config('', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]
    assert 'voice' not in config['audio']['output']
    settings = ElevenLabsRealtimeModelSettings(openai_voice='alloy', elevenlabs_voice='voice-id')
    config = _model(settings)._session_config('', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]
    assert config['audio']['output']['voice'] == 'voice-id'


def test_session_config_push_to_talk_and_cross_provider_turn_detection() -> None:
    manual = _model(ElevenLabsRealtimeModelSettings(turn_detection=False))
    assert manual._session_config('', None, model_settings=None)['audio']['input']['turn_detection'] is None  # pyright: ignore[reportPrivateUsage]
    sensitive = _model(ElevenLabsRealtimeModelSettings(turn_detection={'sensitivity': 'high'}))
    config = sensitive._session_config('', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]
    assert config['audio']['input']['turn_detection'] == snapshot(
        {'type': 'server_vad', 'create_response': True, 'interrupt_response': True, 'threshold': 0.3}
    )


def test_session_config_tool_choice() -> None:
    tools = [ToolDefinition(name='lookup', parameters_json_schema={'type': 'object'})]
    config = _model(ElevenLabsRealtimeModelSettings(tool_choice='none'))._session_config(  # pyright: ignore[reportPrivateUsage]
        '', tools, model_settings=None
    )
    assert config['tool_choice'] == 'none'


# --- event mapping --------------------------------------------------------------------------------


def test_map_event_delegates_to_openai_codec() -> None:
    event = map_event(
        sdk_frame({'type': 'response.output_audio_transcript.done', 'transcript': 'hi', 'response_id': 'r'})
    )
    assert event == OutputTranscript(text='hi', is_final=True, response_id='r')


# --- connect: handshake, URL, auth ----------------------------------------------------------------


def _created(model: str | None = None) -> str:
    return json.dumps({'type': 'session.created', 'session': {'model': model}})


def _updated() -> str:
    return json.dumps({'type': 'session.updated'})


async def test_connect_handshake_url_auth_and_session_config(monkeypatch: pytest.MonkeyPatch) -> None:
    transcript = json.dumps({'type': 'response.output_audio_transcript.done', 'transcript': 'hi'})
    ws = FakeWebSocket([_created('eleven-voice-served'), _updated(), transcript])
    fake_connect = FakeConnect(ws)
    monkeypatch.setattr(rt_elevenlabs.websockets, 'connect', fake_connect)

    async with _connect(_model(ElevenLabsRealtimeModelSettings(elevenlabs_voice='voice-id')), 'Be nice') as conn:
        assert isinstance(conn, ElevenLabsRealtimeConnection)
        assert conn.model_name == 'eleven-voice-served'
        events = await collect_codec_events(conn)

    assert events == [OutputTranscript(text='hi', is_final=True, response_id='response')]
    assert fake_connect.url == snapshot('wss://api.elevenlabs.io/v1/realtime?model=eleven-voice')
    assert fake_connect.headers == snapshot({'Authorization': 'Bearer k'})
    update = json.loads(ws.sent[0])
    assert update['type'] == 'session.update'
    assert update['session']['instructions'] == 'Be nice'
    assert update['session']['audio']['output']['voice'] == 'voice-id'


async def test_connect_derives_url_from_residency_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_connect = FakeConnect(FakeWebSocket([_created(), _updated()]))
    monkeypatch.setattr(rt_elevenlabs.websockets, 'connect', fake_connect)
    model = ElevenLabsRealtimeModel(
        'eleven-voice', provider=ElevenLabsProvider(api_key='k', base_url='https://api.eu.residency.elevenlabs.io')
    )
    async with _connect(model, 'x') as conn:
        assert conn.model_name is None

    assert fake_connect.url == snapshot('wss://api.eu.residency.elevenlabs.io/v1/realtime?model=eleven-voice')


async def test_connect_surfaces_handshake_error(monkeypatch: pytest.MonkeyPatch) -> None:
    error = json.dumps({'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'bad voice'}})
    monkeypatch.setattr(rt_elevenlabs.websockets, 'connect', FakeConnect(FakeWebSocket([_created(), error])))
    with pytest.raises(ModelAPIError, match='bad voice'):
        async with _connect(_model(), 'x'):
            pass  # pragma: no cover


async def test_webrtc_is_unsupported() -> None:
    model = _model()
    with pytest.raises(UserError, match=r'does not support WebRTC, so `create_client_secret\(\)` is unavailable'):
        await model.create_client_secret()
    with pytest.raises(UserError, match=r'`answer_webrtc_offer\(\)` is unavailable'):
        await model.answer_webrtc_offer('v=0')
    session = WebRTCSession(provider_name='elevenlabs', session_id='call')
    with pytest.raises(UserError, match=r'`connect_webrtc\(\)` is unavailable'):
        model.connect_webrtc(
            session, messages=[], model_settings=None, model_request_parameters=ModelRequestParameters()
        )
    with pytest.raises(UserError, match='cannot end a call from the server'):
        await model.hang_up(session)


async def test_connection_names_elevenlabs_when_closed() -> None:
    conn = ElevenLabsRealtimeConnection(FakeWebSocket([]))  # type: ignore[arg-type]
    assert conn._provider_label == 'ElevenLabs Realtime'  # pyright: ignore[reportPrivateUsage]
    assert await collect_codec_events(conn) == []
