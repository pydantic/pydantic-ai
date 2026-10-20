"""Tests for the ElevenLabs realtime provider (provider, handshake, session config), all network-free.

Eleven Realtime speaks the OpenAI Realtime protocol, so these tests focus on what the ElevenLabs provider
adds on top of the shared OpenAI codec (exercised in `test_openai.py`). Each divergence pinned here was
observed against the live API; `test_elevenlabs_ws.py` replays the recordings.
"""

from __future__ import annotations as _annotations

import asyncio
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
from pydantic_ai.realtime._lifecycle import InputAdded, InputLost, ResponseEnded, ResponseStarted
from pydantic_ai.realtime.codec import OutputTranscript, ResponseDone
from pydantic_ai.tools import ToolDefinition

from ..conftest import TestEnv, try_import
from ..realtime_connections import lifecycle_events
from .ws_helpers import collect_codec_events

with try_import() as imports_successful:
    import httpx2

    from pydantic_ai.providers import infer_provider
    from pydantic_ai.providers.elevenlabs import ElevenLabsProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.realtime import elevenlabs as rt_elevenlabs
    from pydantic_ai.realtime.codec import ToolResult
    from pydantic_ai.realtime.elevenlabs import (
        ElevenLabsRealtimeConnection,
        ElevenLabsRealtimeModel,
        ElevenLabsRealtimeModelSettings,
        _ElevenLabsLifecycle,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.realtime.openai import OpenAIRealtimeConnection

    from .test_openai import FakeConnect, FakeWebSocket, PushWebSocket
    from .test_openai_lifecycle import created, describe, done, frames, tool_output_added, transcript

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


def test_model_identity(env: TestEnv) -> None:
    env.set('ELEVENLABS_API_KEY', 'env-key')
    assert ElevenLabsRealtimeModel('eleven_realtime_v1_max').base_url == 'https://api.elevenlabs.io'
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
                'input': {
                    'format': {'type': 'audio/pcm', 'rate': 24000},
                    'turn_detection': {'type': 'semantic_vad', 'create_response': True, 'interrupt_response': True},
                },
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
                'turn_detection': {'type': 'semantic_vad', 'create_response': True, 'interrupt_response': True},
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
                'input': {
                    'format': {'type': 'audio/pcm', 'rate': 24000},
                    'turn_detection': {'type': 'semantic_vad', 'create_response': True, 'interrupt_response': True},
                },
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
        'type': 'semantic_vad',
        'create_response': True,
        'interrupt_response': True,
    }
    # Only the knobs given are sent.
    config = _config(ElevenLabsRealtimeModelSettings(turn_detection={'silence_duration_ms': 500}))
    assert config['audio']['input']['turn_detection'] == snapshot(
        {'type': 'semantic_vad', 'create_response': True, 'interrupt_response': True, 'silence_duration_ms': 500}
    )
    config = _config(
        ElevenLabsRealtimeModelSettings(
            turn_detection={'sensitivity': 'high', 'prefix_padding_ms': 200, 'silence_duration_ms': 800}
        )
    )
    assert config['audio']['input']['turn_detection'] == snapshot(
        {
            'type': 'semantic_vad',
            'create_response': True,
            'interrupt_response': True,
            'eagerness': 'high',
            'prefix_padding_ms': 200,
            'silence_duration_ms': 800,
        }
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
                        'turn_detection': {'type': 'semantic_vad', 'create_response': True, 'interrupt_response': True},
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


# --- connection behavior --------------------------------------------------------------------------


async def test_keep_alive_sends_an_empty_append_when_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    # ElevenLabs closes a session that hasn't heard from the client for 60 seconds, and pings don't count.
    monkeypatch.setattr(ElevenLabsRealtimeConnection, '_keep_alive_after', 0.01)
    ws = PushWebSocket()
    conn = ElevenLabsRealtimeConnection(ws)  # type: ignore[arg-type]

    async def read() -> None:
        async for _ in conn.frames():
            pass  # pragma: no cover

    reading = asyncio.create_task(read())
    await asyncio.sleep(0.05)
    assert {'type': 'input_audio_buffer.append', 'audio': ''} in [json.loads(sent) for sent in ws.sent]
    reading.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reading


def _call(response_id: str, call_id: str) -> dict[str, Any]:
    return {
        'type': 'response.function_call_arguments.done',
        'response_id': response_id,
        'item_id': f'item_{call_id}',
        'output_index': 0,
        'call_id': call_id,
        'name': 'lookup',
        'arguments': '{}',
    }


async def test_tool_calls_abandoned_by_a_later_response() -> None:
    """A response that starts before a tool call's result is in makes ElevenLabs give up on that call.

    Its result never reaches the model, and a `response.create` after it goes unanswered, so the connection
    doesn't ask for one: the result's input is settled as lost instead of leaving `wait_for_reply()` waiting.
    """
    ws = FakeWebSocket(
        frames(
            created('resp_1'),
            _call('resp_1', 'call_1'),
            done('resp_1'),
            created('resp_2'),
            done('resp_2'),
            tool_output_added('call_1'),
        )
    )
    conn = ElevenLabsRealtimeConnection(ws)  # type: ignore[arg-type]
    events = aiter(lifecycle_events(conn))
    taken = [describe(await anext(events)) for _ in range(8)]
    await conn.send(ToolResult(tool_call_id='call_1', output='42'))
    rest = [describe(event) async for event in events]

    assert taken + rest == snapshot(
        [
            ResponseStarted(response_id='resp_1'),
            'ToolCall',
            'ResponseDone',
            ResponseEnded(
                response_id='resp_1', status='completed', finish_reason='stop', provider_details={'status': 'completed'}
            ),
            'RealtimeSessionErrorEvent',
            ResponseStarted(response_id='resp_2'),
            'ResponseDone',
            ResponseEnded(
                response_id='resp_2', status='completed', finish_reason='stop', provider_details={'status': 'completed'}
            ),
            InputAdded(input_id=0),
            InputLost(input_ids=(0,)),
            'RealtimeSessionErrorEvent',
        ]
    )
    assert [json.loads(sent)['type'] for sent in ws.sent] == snapshot(['conversation.item.create'])


async def test_audio_tags_are_stripped_from_transcripts() -> None:
    ws = FakeWebSocket(
        frames(
            created('resp_1'),
            {**transcript('resp_1', '[cheerful]'), 'item_id': 'item_1'},
            {**transcript('resp_1', ' Hello'), 'item_id': 'item_1'},
            {**transcript('resp_1', ' there! [happy'), 'item_id': 'item_1'},
            {**transcript('resp_1', '] How are you?'), 'item_id': 'item_1'},
            {
                'type': 'response.output_audio_transcript.done',
                'response_id': 'resp_1',
                'item_id': 'item_1',
                'output_index': 0,
                'content_index': 0,
                'transcript': '[cheerful] Hello there! [happy] How are you?',
            },
            done('resp_1'),
        )
    )
    events = await collect_codec_events(ElevenLabsRealtimeConnection(ws))  # type: ignore[arg-type]
    assert events == snapshot(
        [
            OutputTranscript(text='Hello', item_id='item_1', response_id='resp_1'),
            OutputTranscript(text=' there!', item_id='item_1', response_id='resp_1'),
            OutputTranscript(text=' How are you?', item_id='item_1', response_id='resp_1'),
            OutputTranscript(text='Hello there! How are you?', is_final=True, item_id='item_1', response_id='resp_1'),
            ResponseDone(
                provider_response_id='resp_1',
                finish_reason='stop',
                provider_details={'status': 'completed', 'audio_tags': ['cheerful', 'happy']},
            ),
        ]
    )


async def test_audio_tags_kept_when_stripping_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    transcript_done = json.dumps(
        {
            'type': 'response.output_audio_transcript.done',
            'response_id': 'resp_1',
            'item_id': 'item_1',
            'output_index': 0,
            'content_index': 0,
            'transcript': '[cheerful] Hi.',
        }
    )
    ws = FakeWebSocket([_created(), _updated(), transcript_done])
    monkeypatch.setattr(rt_elevenlabs.websockets, 'connect', FakeConnect(ws))
    model = _model(ElevenLabsRealtimeModelSettings(elevenlabs_strip_audio_tags=False))
    async with _connect(model, 'x') as conn:
        events = await collect_codec_events(conn)
    assert [event.text for event in events if isinstance(event, OutputTranscript)] == ['[cheerful] Hi.']


async def test_keep_alive_can_be_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ElevenLabsRealtimeConnection, '_keep_alive_after', None)
    ws = FakeWebSocket([])
    assert await collect_codec_events(ElevenLabsRealtimeConnection(ws)) == []  # type: ignore[arg-type]
    assert ws.sent == []


async def test_tool_calls_of_a_response_under_way_are_not_abandoned() -> None:
    ws = FakeWebSocket(frames(created('resp_1'), _call('resp_1', 'call_1'), created('resp_2')))
    events = await collect_codec_events(ElevenLabsRealtimeConnection(ws))  # type: ignore[arg-type]
    assert [event for event in events if getattr(event, 'type', None) == 'tool_calls_abandoned'] == []


def test_semantic_vad_stop_is_not_a_commit() -> None:
    """A stop doesn't place the turn: semantic VAD may hear the user carry on, and commits the turn itself."""
    stopped = {'type': 'input_audio_buffer.speech_stopped', 'event_id': 'e', 'audio_end_ms': 1, 'item_id': 'item_1'}
    assert _ElevenLabsLifecycle(server_vad=True).speech_stopped(stopped) == []
    manual = _ElevenLabsLifecycle(server_vad=False)
    assert manual.speech_stopped(stopped) == []
    assert manual._vad_stopped == set()  # pyright: ignore[reportPrivateUsage]
