"""Tests for the ElevenLabs realtime provider (provider, handshake, session config), all network-free.

Eleven Realtime speaks the OpenAI Realtime protocol, so these tests focus on what the ElevenLabs provider
adds on top of the shared OpenAI codec (exercised in `test_openai.py`). Each divergence pinned here was
observed against the live API; `test_elevenlabs_ws.py` replays the recordings.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any

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
    import httpx2

    from pydantic_ai.providers import infer_provider
    from pydantic_ai.providers.elevenlabs import ElevenLabsProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.realtime import elevenlabs as rt_elevenlabs
    from pydantic_ai.realtime.elevenlabs import (
        ElevenLabsRealtimeConnection,
        ElevenLabsRealtimeModel,
        ElevenLabsRealtimeModelSettings,
    )
    from pydantic_ai.realtime.openai import OpenAIRealtimeConnection

    from .test_openai import FakeConnect, FakeWebSocket

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai / websockets not installed'),
    pytest.mark.anyio,
]


def _model(
    settings: ElevenLabsRealtimeModelSettings | None = None, model: str = 'eleven_realtime_v1_mini'
) -> ElevenLabsRealtimeModel:
    return ElevenLabsRealtimeModel(model, provider=ElevenLabsProvider(api_key='k'), settings=settings)


def _config(settings: ElevenLabsRealtimeModelSettings, tools: list[ToolDefinition] | None = None) -> dict[str, Any]:
    return _model(settings)._session_config('', tools, model_settings=None)  # pyright: ignore[reportPrivateUsage]


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
        ElevenLabsRealtimeModel('eleven_realtime_v1_mini', provider=OpenAIProvider(api_key='k'))  # pyright: ignore[reportArgumentType]


def test_model_identity() -> None:
    model = _model()
    assert model.model_name == 'eleven_realtime_v1_mini'
    assert model.system == 'elevenlabs'
    assert model.base_url == 'https://api.elevenlabs.io'


def test_profile() -> None:
    assert _model().profile == snapshot(
        {
            'supports_image_input': False,
            'image_input_requires_response': False,
            'supports_manual_turn_control': True,
            'supports_interruption': True,
            'supports_output_truncation': True,
            'supports_text_output': False,
            'supports_session_seeding': True,
            'supports_webrtc': False,
            'supports_seeding_images': False,
            'supports_seeding_audio': False,
            'async_tool_call_mode': 'never',
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
            'supports_async_tool_calls': False,
        }
    )


# --- session config -------------------------------------------------------------------------------


def test_session_config_shape() -> None:
    tools = [ToolDefinition(name='lookup', description='Look up', parameters_json_schema={'type': 'object'})]
    config = _model()._session_config('Be nice', tools, model_settings=None)  # pyright: ignore[reportPrivateUsage]
    # The model goes in `session.model`, VAD is semantic, and transcription takes no model.
    assert config == snapshot(
        {
            'type': 'realtime',
            'instructions': 'Be nice',
            'output_modalities': ['audio'],
            'audio': {
                'input': {'format': {'type': 'audio/pcm', 'rate': 24000}, 'turn_detection': {'type': 'semantic_vad'}},
                'output': {'format': {'type': 'audio/pcm'}},
            },
            'tools': [
                {'type': 'function', 'name': 'lookup', 'parameters': {'type': 'object'}, 'description': 'Look up'}
            ],
            'model': 'eleven_realtime_v1_mini',
        }
    )


def test_session_config_elevenlabs_settings() -> None:
    config = _config(
        ElevenLabsRealtimeModelSettings(
            elevenlabs_voice='JBFqnCBsd6RMkjVDRZzb', elevenlabs_language='es', elevenlabs_max_duration_seconds=1800
        )
    )
    assert config['audio'] == snapshot(
        {
            'input': {
                'format': {'type': 'audio/pcm', 'rate': 24000},
                'turn_detection': {'type': 'semantic_vad'},
                'transcription': {'language': 'es'},
            },
            'output': {'format': {'type': 'audio/pcm'}, 'voice': 'JBFqnCBsd6RMkjVDRZzb'},
        }
    )
    assert config['max_duration_seconds'] == 1800


def test_session_config_drops_settings_elevenlabs_ignores() -> None:
    config = _config(
        ElevenLabsRealtimeModelSettings(
            openai_voice='alloy',
            openai_input_noise_reduction='near_field',
            openai_output_speed=1.2,
            openai_truncation='disabled',
            parallel_tool_calls=False,
            thinking='high',
            input_transcription_model=None,
            max_tokens=2048,
        )
    )
    assert config == snapshot(
        {
            'type': 'realtime',
            'instructions': '',
            'output_modalities': ['audio'],
            'audio': {
                'input': {'format': {'type': 'audio/pcm', 'rate': 24000}, 'turn_detection': {'type': 'semantic_vad'}},
                'output': {'format': {'type': 'audio/pcm'}},
            },
            'max_output_tokens': 2048,
            'model': 'eleven_realtime_v1_mini',
        }
    )


def test_session_config_rejects_a_transcription_model() -> None:
    with pytest.raises(UserError, match=r"takes no transcription model.*got 'scribe_v2_realtime'"):
        _config(ElevenLabsRealtimeModelSettings(input_transcription_model='scribe_v2_realtime'))


def test_session_config_cross_provider_turn_detection() -> None:
    assert _config(ElevenLabsRealtimeModelSettings(turn_detection=False))['audio']['input']['turn_detection'] is None
    assert _config(ElevenLabsRealtimeModelSettings(turn_detection=True))['audio']['input']['turn_detection'] == {
        'type': 'semantic_vad'
    }
    config = _config(
        ElevenLabsRealtimeModelSettings(
            turn_detection={'sensitivity': 'high', 'prefix_padding_ms': 200, 'silence_duration_ms': 800}
        )
    )
    assert config['audio']['input']['turn_detection'] == snapshot(
        {'type': 'semantic_vad', 'eagerness': 'high', 'prefix_padding_ms': 200, 'silence_duration_ms': 800}
    )


def test_session_config_openai_turn_detection() -> None:
    config = _config(ElevenLabsRealtimeModelSettings(openai_turn_detection={'type': 'semantic_vad'}))
    # OpenAI's `eagerness='auto'` default isn't one ElevenLabs takes, so it's left to the server's default.
    assert config['audio']['input']['turn_detection'] == snapshot(
        {'type': 'semantic_vad', 'create_response': True, 'interrupt_response': True}
    )
    config = _config(
        ElevenLabsRealtimeModelSettings(
            openai_turn_detection={'type': 'semantic_vad', 'eagerness': 'low', 'interrupt_response': False}
        )
    )
    assert config['audio']['input']['turn_detection'] == snapshot(
        {'type': 'semantic_vad', 'eagerness': 'low', 'create_response': True, 'interrupt_response': False}
    )
    with pytest.raises(UserError, match='only supports semantic VAD'):
        _config(ElevenLabsRealtimeModelSettings(openai_turn_detection={'type': 'server_vad'}))


def test_session_config_tool_choice() -> None:
    tools = [ToolDefinition(name='lookup', parameters_json_schema={'type': 'object'})]
    assert _config(ElevenLabsRealtimeModelSettings(tool_choice='none'), tools)['tool_choice'] == 'none'


def test_connection_always_transcribes() -> None:
    # ElevenLabs transcribes every spoken turn, whatever `input_transcription_model` asked for.
    conn = ElevenLabsRealtimeConnection(FakeWebSocket([]), input_transcription_enabled=False)  # type: ignore[arg-type]
    assert conn.input_transcription_enabled is True


def test_connection_never_prunes_tool_items() -> None:
    conn = ElevenLabsRealtimeConnection(FakeWebSocket([]), prune_tool_items=True)  # type: ignore[arg-type]
    assert conn._prunes_tool_calls is False  # pyright: ignore[reportPrivateUsage]


# --- connect: handshake, URL, auth ----------------------------------------------------------------


def _created() -> str:
    # `session.created` always names the default model, whichever one the session goes on to request.
    return json.dumps({'type': 'session.created', 'session': {'model': 'eleven_realtime_v1_mini'}})


def _conversation_created() -> str:
    return json.dumps({'type': 'conversation.created', 'conversation': {'id': 'conv_1'}})


def _updated() -> str:
    return json.dumps({'type': 'session.updated'})


async def test_connect_handshake_url_auth_and_session_config(monkeypatch: pytest.MonkeyPatch) -> None:
    transcript = json.dumps({'type': 'response.output_audio_transcript.done', 'transcript': 'hi'})
    # ElevenLabs announces the conversation between `session.created` and `session.updated`.
    ws = FakeWebSocket([_created(), _conversation_created(), _updated(), transcript])
    fake_connect = FakeConnect(ws)
    monkeypatch.setattr(rt_elevenlabs.websockets, 'connect', fake_connect)

    model = _model(ElevenLabsRealtimeModelSettings(elevenlabs_voice='george'), model='eleven_realtime_v1_max')
    async with _connect(model, 'Be nice') as conn:
        assert isinstance(conn, ElevenLabsRealtimeConnection)
        assert conn.model_name == 'eleven_realtime_v1_max'
        events = await collect_codec_events(conn)

    assert events == [OutputTranscript(text='hi', is_final=True, response_id='response')]
    assert fake_connect.url == snapshot('wss://api.elevenlabs.io/v1/convai/realtime')
    assert fake_connect.headers == snapshot({'xi-api-key': 'k'})
    update = json.loads(ws.sent[0])
    assert update['type'] == 'session.update'
    assert update['session']['model'] == 'eleven_realtime_v1_max'
    assert update['session']['instructions'] == 'Be nice'
    assert update['session']['audio']['output']['voice'] == 'george'


async def test_connect_derives_url_from_residency_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_connect = FakeConnect(FakeWebSocket([_created(), _updated()]))
    monkeypatch.setattr(rt_elevenlabs.websockets, 'connect', fake_connect)
    model = ElevenLabsRealtimeModel(
        'eleven_realtime_v1_mini',
        provider=ElevenLabsProvider(api_key='k', base_url='https://api.eu.residency.elevenlabs.io'),
    )
    async with _connect(model, 'x'):
        pass

    assert fake_connect.url == snapshot('wss://api.eu.residency.elevenlabs.io/v1/convai/realtime')


async def test_connect_surfaces_handshake_error(monkeypatch: pytest.MonkeyPatch) -> None:
    error = json.dumps(
        {
            'type': 'error',
            'error': {
                'type': 'invalid_request_error',
                'code': 'invalid_value',
                'message': "Invalid value for 'session.model': Input should be 'eleven_realtime_v1_mini' or "
                "'eleven_realtime_v1_max'.",
                'param': 'session.model',
            },
        }
    )
    monkeypatch.setattr(
        rt_elevenlabs.websockets, 'connect', FakeConnect(FakeWebSocket([_created(), _conversation_created(), error]))
    )
    with pytest.raises(ModelAPIError, match=r"Invalid value for 'session\.model'"):
        async with _connect(_model(model='eleven_realtime_v2'), 'x'):
            pass  # pragma: no cover


# --- client secrets and WebRTC --------------------------------------------------------------------


async def test_create_client_secret_request() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured['url'] = str(request.url)
        captured['api_key'] = request.headers.get('xi-api-key')
        captured['authorization'] = request.headers.get('authorization')
        captured['body'] = json.loads(request.content)
        return httpx2.Response(200, json={'value': 'ek_realtime_x', 'expires_at': 1_700_000_060})

    provider = ElevenLabsProvider(
        api_key='k',
        base_url='https://api.eu.residency.elevenlabs.io',
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    model = ElevenLabsRealtimeModel('eleven_realtime_v1_mini', provider=provider)
    secret = await model.create_client_secret(
        instructions='Be brief.',
        model_settings=ElevenLabsRealtimeModelSettings(elevenlabs_voice='george'),
        expires_after_seconds=60,
    )

    assert secret.value == 'ek_realtime_x'
    assert captured['url'] == snapshot('https://api.eu.residency.elevenlabs.io/v1/convai/realtime/client_secrets')
    assert captured['api_key'] == 'k'
    assert captured['authorization'] is None
    assert captured['body'] == snapshot(
        {
            'session': {
                'model': 'eleven_realtime_v1_mini',
                'type': 'realtime',
                'instructions': 'Be brief.',
                'output_modalities': ['audio'],
                'audio': {
                    'input': {
                        'format': {'type': 'audio/pcm', 'rate': 24000},
                        'turn_detection': {'type': 'semantic_vad'},
                    },
                    'output': {'format': {'type': 'audio/pcm'}, 'voice': 'george'},
                },
            },
            'expires_after': {'anchor': 'created_at', 'seconds': 60},
        }
    )


async def test_webrtc_is_unsupported() -> None:
    model = _model()
    with pytest.raises(UserError, match=r'does not support WebRTC, so `answer_webrtc_offer\(\)` is unavailable'):
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
