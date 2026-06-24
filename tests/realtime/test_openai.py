"""Tests for the OpenAI realtime provider (event mapping, handshake, send), all network-free."""

from __future__ import annotations as _annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from pydantic_ai.realtime import (
    AudioDelta,
    AudioInput,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    CreateResponse,
    ImageInput,
    InputTranscript,
    RateLimit,
    RateLimits,
    Reconnected,
    SessionError,
    SpeechStarted,
    SpeechStopped,
    TextInput,
    ToolCall,
    ToolResult,
    Transcript,
    TruncateOutput,
    TurnComplete,
    Usage,
    openai as rt_openai,
)
from pydantic_ai.realtime.openai import (
    OpenAIRealtimeConnection,
    OpenAIRealtimeModel,
    map_event,
)
from pydantic_ai.settings import ModelSettings, ToolOrOutput
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage


def test_map_audio_delta() -> None:
    payload = base64.b64encode(b'\x01\x02').decode('ascii')
    for event_type in ('response.output_audio.delta', 'response.audio.delta'):
        event = map_event({'type': event_type, 'delta': payload})
        assert event == AudioDelta(data=b'\x01\x02')


def test_map_audio_delta_non_string_delta() -> None:
    assert map_event({'type': 'response.output_audio.delta', 'delta': 123}) is None


def test_map_transcript_delta_and_done() -> None:
    for event_type in ('response.output_audio_transcript.delta', 'response.audio_transcript.delta'):
        assert map_event({'type': event_type, 'delta': 'hel'}) == Transcript(text='hel', is_final=False)
    for event_type in ('response.output_audio_transcript.done', 'response.audio_transcript.done'):
        assert map_event({'type': event_type, 'transcript': 'hello'}) == Transcript(text='hello', is_final=True)


def test_map_text_output_delta_and_done() -> None:
    assert map_event({'type': 'response.output_text.delta', 'delta': 'hel'}) == Transcript(text='hel', is_final=False)
    assert map_event({'type': 'response.output_text.done', 'text': 'hello'}) == Transcript(text='hello', is_final=True)


def test_map_transcript_missing_field_defaults_to_empty() -> None:
    assert map_event({'type': 'response.output_audio_transcript.delta'}) == Transcript(text='', is_final=False)


def test_map_input_transcript() -> None:
    event = map_event({'type': 'conversation.item.input_audio_transcription.completed', 'transcript': 'weather?'})
    assert event == InputTranscript(text='weather?', is_final=True)


def test_map_input_transcript_delta() -> None:
    event = map_event({'type': 'conversation.item.input_audio_transcription.delta', 'delta': 'wea'})
    assert event == InputTranscript(text='wea', is_final=False)


def test_map_function_call() -> None:
    event = map_event(
        {
            'type': 'response.function_call_arguments.done',
            'call_id': 'call_1',
            'name': 'get_weather',
            'arguments': '{"city": "Paris"}',
        }
    )
    assert event == ToolCall(tool_call_id='call_1', tool_name='get_weather', args='{"city": "Paris"}')


def test_map_function_call_missing_arguments_defaults_to_empty_object() -> None:
    event = map_event({'type': 'response.function_call_arguments.done', 'call_id': 'c', 'name': 'n'})
    assert isinstance(event, ToolCall)
    assert event.args == '{}'


def _response_done(response: Any) -> dict[str, Any]:
    return {'type': 'response.done', 'response': response}


def test_map_response_done_normal() -> None:
    assert map_event(_response_done({'status': 'completed', 'output': []})) == TurnComplete(interrupted=False)


def test_map_response_done_cancelled() -> None:
    assert map_event(_response_done({'status': 'cancelled'})) == TurnComplete(interrupted=True)


def test_map_response_done_function_call_only_is_skipped() -> None:
    assert (
        map_event(_response_done({'status': 'completed', 'output': [{'type': 'function_call', 'name': 'x'}]})) is None
    )


def test_map_response_done_mixed_output_is_turn_complete() -> None:
    data = _response_done({'status': 'completed', 'output': [{'type': 'function_call'}, {'type': 'message'}]})
    assert map_event(data) == TurnComplete(interrupted=False)


def test_map_response_done_without_response_object() -> None:
    assert map_event({'type': 'response.done'}) == TurnComplete(interrupted=False)


def test_map_error_event_with_message() -> None:
    assert map_event({'type': 'error', 'error': {'message': 'bad'}}) == SessionError(message='bad')


def test_map_error_event_without_message_serializes_payload() -> None:
    assert map_event({'type': 'error', 'error': {'code': 'x'}}) == SessionError(
        message=json.dumps({'code': 'x'}), code='x'
    )


def test_map_error_event_non_dict_payload() -> None:
    assert map_event({'type': 'error', 'error': 'plain'}) == SessionError(message='plain')


def test_map_error_event_with_type_and_code_is_recoverable() -> None:
    event = map_event({'type': 'error', 'error': {'message': 'bad', 'type': 'invalid_request_error', 'code': 'c1'}})
    assert event == SessionError(message='bad', type='invalid_request_error', code='c1', recoverable=True)


def test_map_rate_limits() -> None:
    event = map_event(
        {
            'type': 'rate_limits.updated',
            'rate_limits': [
                {'name': 'requests', 'limit': 100, 'remaining': 99, 'reset_seconds': 1.5},
                {'name': 'tokens', 'reset_seconds': 2},  # int reset → float; limit/remaining missing → None
                {'limit': 1},  # no name → skipped
            ],
        }
    )
    assert event == RateLimits(
        limits=[
            RateLimit(name='requests', limit=100, remaining=99, reset_seconds=1.5),
            RateLimit(name='tokens', limit=None, remaining=None, reset_seconds=2.0),
        ]
    )


def test_map_rate_limits_non_list() -> None:
    assert map_event({'type': 'rate_limits.updated'}) == RateLimits(limits=[])


def test_map_usage_full_payload() -> None:
    response = {
        'usage': {
            'input_tokens': 100,
            'output_tokens': 50,
            'input_token_details': {
                'audio_tokens': 80,
                'cached_tokens': 30,
                'text_tokens': 20,
                'image_tokens': 5,
                'cached_tokens_details': {'audio_tokens': 10},
            },
            'output_token_details': {'audio_tokens': 40, 'text_tokens': 10},
        }
    }
    usage = rt_openai._map_usage(response)  # pyright: ignore[reportPrivateUsage]
    assert usage == RequestUsage(
        input_tokens=100,
        output_tokens=50,
        input_audio_tokens=80,
        cache_read_tokens=30,
        cache_audio_read_tokens=10,
        output_audio_tokens=40,
        details={'input_text_tokens': 20, 'input_image_tokens': 5, 'output_text_tokens': 10},
    )


def test_map_usage_minimal_and_missing() -> None:
    assert rt_openai._map_usage({'usage': {'input_tokens': 7}}) == RequestUsage(input_tokens=7)  # pyright: ignore[reportPrivateUsage]
    assert rt_openai._map_usage({}) is None  # pyright: ignore[reportPrivateUsage]
    assert rt_openai._map_usage({'usage': 'nope'}) is None  # pyright: ignore[reportPrivateUsage]


def test_map_speech_started() -> None:
    assert map_event({'type': 'input_audio_buffer.speech_started'}) == SpeechStarted()


def test_map_speech_stopped() -> None:
    assert map_event({'type': 'input_audio_buffer.speech_stopped'}) == SpeechStopped()


def test_map_unknown_event_returns_none() -> None:
    assert map_event({'type': 'session.created'}) is None


def test_model_repr_hides_api_key() -> None:
    model = OpenAIRealtimeModel('gpt-realtime', api_key='super-secret')
    assert 'super-secret' not in repr(model)
    assert model.model_name == 'gpt-realtime'


class FakeWebSocket:
    """A minimal stand-in for a `websockets` client connection."""

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
    """Stand-in for `websockets.connect`, returning a fixed websocket."""

    def __init__(self, ws: FakeWebSocket) -> None:
        self.ws = ws
        self.url: str | None = None
        self.headers: dict[str, str] | None = None

    def __call__(self, url: str, *, additional_headers: dict[str, str] | None = None) -> FakeConnect:
        self.url = url
        self.headers = additional_headers
        return self

    async def __aenter__(self) -> FakeWebSocket:
        return self.ws

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _created() -> str:
    return json.dumps({'type': 'session.created'})


def _updated() -> str:
    return json.dumps({'type': 'session.updated'})


@pytest.mark.anyio
async def test_connect_handshake_and_session_config(monkeypatch: pytest.MonkeyPatch) -> None:
    transcript = json.dumps({'type': 'response.audio_transcript.done', 'transcript': 'hi'})
    ws = FakeWebSocket([_created(), _updated(), transcript])
    fake_connect = FakeConnect(ws)
    monkeypatch.setattr(rt_openai.websockets, 'connect', fake_connect)

    model = OpenAIRealtimeModel('gpt-realtime', api_key='k', voice='alloy')
    tools = [ToolDefinition(name='get_weather', description='Weather', parameters_json_schema={'type': 'object'})]

    async with model.connect(instructions='Be nice', tools=tools) as conn:
        events = [e async for e in conn]

    assert events == [Transcript(text='hi', is_final=True)]
    assert fake_connect.url == 'wss://api.openai.com/v1/realtime?model=gpt-realtime'
    assert fake_connect.headers == {'Authorization': 'Bearer k'}

    update = json.loads(ws.sent[0])
    assert update['type'] == 'session.update'
    session = update['session']
    assert session['type'] == 'realtime'
    assert session['instructions'] == 'Be nice'
    assert session['output_modalities'] == ['audio']
    assert session['audio']['input']['format'] == {'type': 'audio/pcm', 'rate': 24000}
    assert session['audio']['input']['turn_detection'] == {
        'type': 'server_vad',
        'create_response': True,
        'interrupt_response': True,
    }
    assert session['audio']['input']['transcription'] == {'model': 'whisper-1'}
    assert session['audio']['output']['voice'] == 'alloy'
    assert session['tools'][0]['name'] == 'get_weather'
    assert session['tools'][0]['type'] == 'function'


def test_session_config_server_vad_params() -> None:
    model = OpenAIRealtimeModel(
        turn_detection=rt_openai.ServerVAD(
            threshold=0.7,
            prefix_padding_ms=200,
            silence_duration_ms=400,
            create_response=False,
            interrupt_response=False,
            idle_timeout_ms=5000,
        ),
    )
    config = model._session_config('hi', None, None)  # pyright: ignore[reportPrivateUsage]
    assert config['audio']['input']['turn_detection'] == {
        'type': 'server_vad',
        'create_response': False,
        'interrupt_response': False,
        'threshold': 0.7,
        'prefix_padding_ms': 200,
        'silence_duration_ms': 400,
        'idle_timeout_ms': 5000,
    }


def test_session_config_semantic_vad() -> None:
    model = OpenAIRealtimeModel(turn_detection=rt_openai.SemanticVAD(eagerness='high'))
    config = model._session_config('hi', None, None)  # pyright: ignore[reportPrivateUsage]
    assert config['audio']['input']['turn_detection'] == {
        'type': 'semantic_vad',
        'eagerness': 'high',
        'create_response': True,
        'interrupt_response': True,
    }


def test_session_config_manual_turn_detection_is_null() -> None:
    model = OpenAIRealtimeModel(turn_detection=None)
    config = model._session_config('hi', None, None)  # pyright: ignore[reportPrivateUsage]
    assert config['audio']['input']['turn_detection'] is None


def test_session_config_noise_reduction_and_speed_and_modalities() -> None:
    model = OpenAIRealtimeModel(
        input_noise_reduction='near_field',
        output_speed=1.25,
        output_modalities=('text',),
    )
    config = model._session_config('hi', None, None)  # pyright: ignore[reportPrivateUsage]
    assert config['audio']['input']['noise_reduction'] == {'type': 'near_field'}
    assert config['audio']['output']['speed'] == 1.25
    assert config['output_modalities'] == ['text']


def test_session_config_forwards_parallel_tool_calls_and_tool_choice() -> None:
    model = OpenAIRealtimeModel()
    settings = ModelSettings(parallel_tool_calls=True, tool_choice='required', temperature=0.5)
    config = model._session_config('hi', None, settings)  # pyright: ignore[reportPrivateUsage]
    assert config['parallel_tool_calls'] is True
    assert config['tool_choice'] == 'required'
    assert 'temperature' not in config  # GA realtime has no temperature field


def test_session_config_tool_choice_single_function() -> None:
    model = OpenAIRealtimeModel()
    config = model._session_config('hi', None, ModelSettings(tool_choice=['get_weather']))  # pyright: ignore[reportPrivateUsage]
    assert config['tool_choice'] == {'type': 'function', 'name': 'get_weather'}


def test_session_config_tool_choice_multi_tool_dropped() -> None:
    model = OpenAIRealtimeModel()
    config = model._session_config('hi', None, ModelSettings(tool_choice=['a', 'b']))  # pyright: ignore[reportPrivateUsage]
    assert 'tool_choice' not in config  # realtime can't express a multi-tool restriction


def test_session_config_tool_choice_tool_or_output_dropped() -> None:
    model = OpenAIRealtimeModel()
    settings = ModelSettings(tool_choice=ToolOrOutput(function_tools=['a']))
    config = model._session_config('hi', None, settings)  # pyright: ignore[reportPrivateUsage]
    assert 'tool_choice' not in config  # ToolOrOutput restriction isn't expressible in realtime


@pytest.mark.anyio
async def test_connect_skips_unrelated_events_during_handshake(monkeypatch: pytest.MonkeyPatch) -> None:
    rate_limits = json.dumps({'type': 'rate_limits.updated'})
    ws = FakeWebSocket([rate_limits, _created(), _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    model = OpenAIRealtimeModel('gpt-realtime', api_key='k')
    async with model.connect(instructions='x') as conn:
        assert [e async for e in conn] == []


@pytest.mark.anyio
async def test_connect_surfaces_handshake_error(monkeypatch: pytest.MonkeyPatch) -> None:
    error = json.dumps({'type': 'error', 'error': {'message': 'invalid model'}})
    ws = FakeWebSocket([error])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    model = OpenAIRealtimeModel('gpt-realtime', api_key='k')
    with pytest.raises(RuntimeError, match='invalid model'):
        async with model.connect(instructions='x'):
            pass  # pragma: no cover


class HangingWebSocket(FakeWebSocket):
    """A websocket whose `recv` never returns, to exercise the handshake timeout."""

    async def recv(self) -> Any:
        await asyncio.Event().wait()  # pragma: no cover


@pytest.mark.anyio
async def test_connect_handshake_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = HangingWebSocket([])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    model = OpenAIRealtimeModel('gpt-realtime', api_key='k', handshake_timeout=0.02)
    with pytest.raises(TimeoutError, match="'session.created'"):
        async with model.connect(instructions='x'):
            pass  # pragma: no cover


@pytest.mark.anyio
async def test_connection_iter_skips_non_string_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = json.dumps({'type': 'response.output_audio.delta', 'delta': base64.b64encode(b'\x09').decode('ascii')})
    ws = FakeWebSocket([_created(), _updated(), b'\x00binary', audio])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    model = OpenAIRealtimeModel('gpt-realtime', api_key='k')
    async with model.connect(instructions='x') as conn:
        events = [e async for e in conn]
    assert events == [AudioDelta(data=b'\x09')]


@pytest.mark.anyio
async def test_connect_without_tools_omits_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    model = OpenAIRealtimeModel('gpt-realtime', api_key='k')
    async with model.connect(instructions='x'):
        pass
    session = json.loads(ws.sent[0])['session']
    assert 'tools' not in session
    assert 'voice' not in session['audio']['output']


@pytest.mark.anyio
async def test_connection_send_audio() -> None:
    ws = FakeWebSocket([])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    await conn.send(AudioInput(data=b'\x01\x02'))
    event = json.loads(ws.sent[0])
    assert event['type'] == 'input_audio_buffer.append'
    assert base64.b64decode(event['audio']) == b'\x01\x02'


@pytest.mark.anyio
async def test_connection_send_text() -> None:
    ws = FakeWebSocket([])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    await conn.send(TextInput(text='hello'))
    create = json.loads(ws.sent[0])
    assert create['item']['content'][0]['text'] == 'hello'
    assert json.loads(ws.sent[1]) == {'type': 'response.create'}


@pytest.mark.anyio
async def test_connection_send_tool_result_triggers_response() -> None:
    ws = FakeWebSocket([])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    await conn.send(ToolResult(tool_call_id='call_1', output='42'))
    item = json.loads(ws.sent[0])
    assert item['item'] == {'type': 'function_call_output', 'call_id': 'call_1', 'output': '42'}
    assert json.loads(ws.sent[1]) == {'type': 'response.create'}


@pytest.mark.anyio
async def test_connection_send_image() -> None:
    ws = FakeWebSocket([])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    await conn.send(ImageInput(data=b'\x01\x02', mime_type='image/png'))
    item = json.loads(ws.sent[0])
    assert item['type'] == 'conversation.item.create'
    content = item['item']['content'][0]
    assert content['type'] == 'input_image'
    assert content['image_url'] == 'data:image/png;base64,' + base64.b64encode(b'\x01\x02').decode('ascii')
    assert len(ws.sent) == 1  # image is context only → no response.create


@pytest.mark.anyio
async def test_connection_send_unsupported_raises() -> None:
    ws = FakeWebSocket([])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    # Every member of the RealtimeInput union is handled, so the defensive branch needs a non-member.
    with pytest.raises(NotImplementedError, match='object'):
        await conn.send(object())  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_connection_send_commit_and_clear_audio() -> None:
    ws = FakeWebSocket([])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    await conn.send(CommitAudio())
    await conn.send(ClearAudio())
    assert json.loads(ws.sent[0]) == {'type': 'input_audio_buffer.commit'}
    assert json.loads(ws.sent[1]) == {'type': 'input_audio_buffer.clear'}


@pytest.mark.anyio
async def test_connection_send_create_response() -> None:
    ws = FakeWebSocket([])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    await conn.send(CreateResponse())
    assert json.loads(ws.sent[0]) == {'type': 'response.create'}


@pytest.mark.anyio
async def test_connection_send_cancel_when_response_active() -> None:
    ws = FakeWebSocket([])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    conn._response_active = True  # pyright: ignore[reportPrivateUsage]
    conn._pending_response = True  # pyright: ignore[reportPrivateUsage]
    await conn.send(CancelResponse())
    assert json.loads(ws.sent[0]) == {'type': 'response.cancel'}
    assert conn._response_active is False  # pyright: ignore[reportPrivateUsage]
    assert conn._pending_response is False  # pyright: ignore[reportPrivateUsage]


@pytest.mark.anyio
async def test_connection_send_cancel_when_idle_does_not_send() -> None:
    ws = FakeWebSocket([])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    await conn.send(CancelResponse())  # no active response → no cancel event
    assert ws.sent == []
    assert conn._response_active is False  # pyright: ignore[reportPrivateUsage]


@pytest.mark.anyio
async def test_response_done_emits_usage_then_turn_complete() -> None:
    done = json.dumps(
        {
            'type': 'response.done',
            'response': {'status': 'completed', 'output': [], 'usage': {'input_tokens': 3, 'output_tokens': 2}},
        }
    )
    ws = FakeWebSocket([done])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    events = [e async for e in conn]
    assert events == [Usage(usage=RequestUsage(input_tokens=3, output_tokens=2)), TurnComplete(interrupted=False)]


@pytest.mark.anyio
async def test_response_done_function_call_only_still_emits_usage() -> None:
    done = json.dumps(
        {
            'type': 'response.done',
            'response': {'status': 'completed', 'output': [{'type': 'function_call'}], 'usage': {'output_tokens': 5}},
        }
    )
    ws = FakeWebSocket([done])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    events = [e async for e in conn]
    # function-call-only → no TurnComplete, but usage is still surfaced
    assert events == [Usage(usage=RequestUsage(output_tokens=5))]


class DroppingWebSocket(FakeWebSocket):
    """A websocket whose iteration raises `ConnectionClosed`, simulating a dropped connection."""

    async def __aiter__(self) -> AsyncIterator[Any]:
        raise rt_openai.websockets.ConnectionClosed(None, None)
        yield  # pragma: no cover  (makes this an async generator)


@pytest.mark.anyio
async def test_connection_closed_yields_fatal_error() -> None:
    ws = DroppingWebSocket([])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    events = [e async for e in conn]
    assert len(events) == 1
    error = events[0]
    assert isinstance(error, SessionError)
    assert error.recoverable is False


@pytest.mark.anyio
async def test_reconnects_on_drop_and_resumes() -> None:
    transcript = json.dumps({'type': 'response.audio_transcript.done', 'transcript': 'hi'})
    good = FakeWebSocket([transcript])

    async def dial() -> Any:
        return good

    # The initial connection drops; reconnect re-dials to `good` and resumes streaming.
    conn = OpenAIRealtimeConnection(
        DroppingWebSocket([]),  # type: ignore[arg-type]
        dial=dial,
        reconnect=rt_openai.ReconnectPolicy(base_delay=0.0),
    )
    events = [e async for e in conn]
    assert events == [Reconnected(), Transcript(text='hi', is_final=True)]


@pytest.mark.anyio
async def test_reconnect_gives_up_after_max_attempts() -> None:
    async def dial() -> Any:
        raise RuntimeError('still down')

    conn = OpenAIRealtimeConnection(
        DroppingWebSocket([]),  # type: ignore[arg-type]
        dial=dial,
        reconnect=rt_openai.ReconnectPolicy(max_attempts=2, base_delay=0.0, jitter=False),
    )
    events = [e async for e in conn]
    assert len(events) == 1
    error = events[0]
    assert isinstance(error, SessionError)
    assert error.recoverable is False
    assert 'reconnect failed' in error.message


def _audio_delta(item_id: str, content_index: int | None = None) -> str:
    data: dict[str, Any] = {
        'type': 'response.output_audio.delta',
        'item_id': item_id,
        'delta': base64.b64encode(b'\x01').decode('ascii'),
    }
    if content_index is not None:
        data['content_index'] = content_index
    return json.dumps(data)


@pytest.mark.anyio
async def test_truncate_uses_item_tracked_from_audio_delta() -> None:
    ws = FakeWebSocket([_audio_delta('item_7', content_index=2)])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    _ = [e async for e in conn]  # consume the delta → captures the current output item
    await conn.send(TruncateOutput(audio_end_ms=1200))
    assert json.loads(ws.sent[0]) == {
        'type': 'conversation.item.truncate',
        'item_id': 'item_7',
        'content_index': 2,
        'audio_end_ms': 1200,
    }


@pytest.mark.anyio
async def test_truncate_defaults_content_index_when_absent() -> None:
    ws = FakeWebSocket([_audio_delta('item_x')])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    _ = [e async for e in conn]
    await conn.send(TruncateOutput(audio_end_ms=10))
    assert json.loads(ws.sent[0])['content_index'] == 0


@pytest.mark.anyio
async def test_truncate_without_current_item_is_noop() -> None:
    ws = FakeWebSocket([])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    await conn.send(TruncateOutput(audio_end_ms=500))
    assert ws.sent == []


@pytest.mark.anyio
async def test_response_done_resets_tracked_item() -> None:
    done = json.dumps({'type': 'response.done', 'response': {'status': 'completed', 'output': []}})
    ws = FakeWebSocket([_audio_delta('item_9'), done])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    _ = [e async for e in conn]  # delta sets the item, response.done clears it
    await conn.send(TruncateOutput(audio_end_ms=500))
    assert ws.sent == []


class PushWebSocket:
    """A websocket fake you can push events into while iterating concurrently."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self._q: asyncio.Queue[str] = asyncio.Queue()

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def __aiter__(self) -> AsyncIterator[str]:
        while True:
            yield await self._q.get()

    def push(self, obj: dict[str, Any]) -> None:
        self._q.put_nowait(json.dumps(obj))

    def sent_types(self) -> list[str]:
        return [json.loads(s).get('type') for s in self.sent]


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.anyio
async def test_tool_result_deferred_until_active_response_done() -> None:
    ws = PushWebSocket()
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    task = asyncio.create_task(_drain(conn))

    ws.push({'type': 'response.created'})  # a response is now generating
    await _settle()

    await conn.send(ToolResult(tool_call_id='c1', output='done'))
    await _settle()
    # the tool output is submitted, but the response is deferred (would collide otherwise)
    assert 'conversation.item.create' in ws.sent_types()
    assert 'response.create' not in ws.sent_types()

    ws.push({'type': 'response.done', 'response': {'status': 'completed', 'output': []}})
    await _settle()
    # once the active response finishes, the deferred response.create fires
    assert 'response.create' in ws.sent_types()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.anyio
async def test_deferred_response_dropped_when_active_response_cancelled() -> None:
    ws = PushWebSocket()
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    task = asyncio.create_task(_drain(conn))

    ws.push({'type': 'response.created'})  # a response is now generating
    await _settle()

    await conn.send(ToolResult(tool_call_id='c1', output='done'))
    await _settle()
    assert 'response.create' not in ws.sent_types()

    # The user barges in: the active response is cancelled, so the deferred response must not replay.
    ws.push({'type': 'response.done', 'response': {'status': 'cancelled', 'output': []}})
    await _settle()
    assert 'response.create' not in ws.sent_types()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.anyio
async def test_tool_result_triggers_response_when_idle() -> None:
    ws = PushWebSocket()
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    await conn.send(ToolResult(tool_call_id='c1', output='done'))
    # no active response, so the response is requested immediately
    assert 'response.create' in ws.sent_types()


async def _drain(conn: OpenAIRealtimeConnection) -> None:
    async for _ in conn:
        pass


@pytest.mark.anyio
async def test_connect_tool_without_description(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    model = OpenAIRealtimeModel('gpt-realtime', api_key='k')
    tools = [ToolDefinition(name='ping', parameters_json_schema={'type': 'object'})]
    async with model.connect(instructions='x', tools=tools):
        pass
    tool = json.loads(ws.sent[0])['session']['tools'][0]
    assert tool == {'type': 'function', 'name': 'ping', 'parameters': {'type': 'object'}}


@pytest.mark.anyio
async def test_connect_without_transcription_model_omits_transcription(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    model = OpenAIRealtimeModel('gpt-realtime', api_key='k', input_audio_transcription_model='')
    async with model.connect(instructions='x'):
        pass
    assert 'transcription' not in json.loads(ws.sent[0])['session']['audio']['input']


@pytest.mark.anyio
async def test_connect_applies_max_tokens_without_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    model = OpenAIRealtimeModel('gpt-realtime', api_key='k')
    async with model.connect(instructions='x', model_settings=ModelSettings(max_tokens=256)):
        pass
    session = json.loads(ws.sent[0])['session']
    assert session['max_output_tokens'] == 256
    assert 'temperature' not in session


@pytest.mark.anyio
async def test_connection_iter_skips_unmapped_events(monkeypatch: pytest.MonkeyPatch) -> None:
    unmapped = json.dumps({'type': 'response.created'})
    done = json.dumps({'type': 'response.done', 'response': {'status': 'completed', 'output': []}})
    ws = FakeWebSocket([_created(), _updated(), unmapped, done])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    model = OpenAIRealtimeModel('gpt-realtime', api_key='k')
    async with model.connect(instructions='x') as conn:
        events = [e async for e in conn]
    assert events == [TurnComplete(interrupted=False)]
