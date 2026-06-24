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
    ImageInput,
    InputTranscript,
    SessionError,
    SpeechStarted,
    TextInput,
    ToolCall,
    ToolResult,
    Transcript,
    TurnComplete,
    openai as rt_openai,
)
from pydantic_ai.realtime.openai import (
    OpenAIRealtimeConnection,
    OpenAIRealtimeModel,
    map_event,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition


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


def test_map_transcript_missing_field_defaults_to_empty() -> None:
    assert map_event({'type': 'response.output_audio_transcript.delta'}) == Transcript(text='', is_final=False)


def test_map_input_transcript() -> None:
    event = map_event({'type': 'conversation.item.input_audio_transcription.completed', 'transcript': 'weather?'})
    assert event == InputTranscript(text='weather?', is_final=True)


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
    assert map_event({'type': 'error', 'error': {'code': 'x'}}) == SessionError(message=json.dumps({'code': 'x'}))


def test_map_error_event_non_dict_payload() -> None:
    assert map_event({'type': 'error', 'error': 'plain'}) == SessionError(message='plain')


def test_map_speech_started() -> None:
    assert map_event({'type': 'input_audio_buffer.speech_started'}) == SpeechStarted()


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
    assert session['audio']['input']['turn_detection'] == {'type': 'server_vad'}
    assert session['audio']['input']['transcription'] == {'model': 'whisper-1'}
    assert session['audio']['output']['voice'] == 'alloy'
    assert session['tools'][0]['name'] == 'get_weather'
    assert session['tools'][0]['type'] == 'function'


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
async def test_connection_send_unsupported_raises() -> None:
    ws = FakeWebSocket([])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match='ImageInput'):
        await conn.send(ImageInput(data=b'\x00'))


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
