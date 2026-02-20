"""Tests for the Gemini Live realtime model and connection."""

from __future__ import annotations as _annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai import Agent
    from pydantic_ai.realtime import (
        AudioDelta,
        AudioInput,
        ImageInput,
        InputTranscript,
        RealtimeSessionEvent,
        TextInput,
        ToolCall,
        ToolCallCompleted,
        ToolCallStarted,
        ToolResult,
        Transcript,
        TurnComplete,
    )
    from pydantic_ai.realtime.gemini import GeminiRealtimeConnection, GeminiRealtimeModel, map_message

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
]


# ---------------------------------------------------------------------------
# Stub types to simulate google.genai LiveServerMessage structures
# ---------------------------------------------------------------------------


class StubInlineData:
    def __init__(self, data: bytes) -> None:
        self.data = data


class StubPart:
    def __init__(self, *, inline_data: StubInlineData | None = None, text: str | None = None) -> None:
        self.inline_data = inline_data
        self.text = text


class StubModelTurn:
    def __init__(self, parts: list[StubPart]) -> None:
        self.parts = parts


class StubTranscription:
    def __init__(self, text: str, finished: bool = False) -> None:
        self.text = text
        self.finished = finished


class StubServerContent:
    def __init__(
        self,
        *,
        model_turn: StubModelTurn | None = None,
        turn_complete: bool = False,
        interrupted: bool = False,
        output_transcription: StubTranscription | None = None,
        input_transcription: StubTranscription | None = None,
    ) -> None:
        self.model_turn = model_turn
        self.turn_complete = turn_complete
        self.interrupted = interrupted
        self.output_transcription = output_transcription
        self.input_transcription = input_transcription


class StubFunctionCall:
    def __init__(self, *, id: str, name: str, args: dict[str, Any] | None = None) -> None:
        self.id = id
        self.name = name
        self.args = args


class StubToolCall:
    def __init__(self, function_calls: list[StubFunctionCall]) -> None:
        self.function_calls = function_calls


class StubToolCallCancellation:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids


class StubMessage:
    def __init__(
        self,
        *,
        server_content: StubServerContent | None = None,
        tool_call: StubToolCall | None = None,
        tool_call_cancellation: StubToolCallCancellation | None = None,
    ) -> None:
        self.server_content = server_content
        self.tool_call = tool_call
        self.tool_call_cancellation = tool_call_cancellation


# ---------------------------------------------------------------------------
# map_message tests
# ---------------------------------------------------------------------------


def test_map_audio_delta() -> None:
    msg = StubMessage(
        server_content=StubServerContent(
            model_turn=StubModelTurn(parts=[StubPart(inline_data=StubInlineData(data=b'\x00\x01\x02'))])
        )
    )
    events = map_message(msg)
    assert len(events) == 1
    assert isinstance(events[0], AudioDelta)
    assert events[0].data == b'\x00\x01\x02'


def test_map_text_part() -> None:
    msg = StubMessage(server_content=StubServerContent(model_turn=StubModelTurn(parts=[StubPart(text='Hello')])))
    events = map_message(msg)
    assert len(events) == 1
    assert isinstance(events[0], Transcript)
    assert events[0].text == 'Hello'
    assert events[0].is_final is False


def test_map_output_transcription() -> None:
    msg = StubMessage(
        server_content=StubServerContent(output_transcription=StubTranscription(text='Hi there', finished=True))
    )
    events = map_message(msg)
    assert len(events) == 1
    assert isinstance(events[0], Transcript)
    assert events[0].text == 'Hi there'
    assert events[0].is_final is True


def test_map_input_transcription() -> None:
    msg = StubMessage(
        server_content=StubServerContent(
            input_transcription=StubTranscription(text='What is the weather?', finished=True)
        )
    )
    events = map_message(msg)
    assert len(events) == 1
    assert isinstance(events[0], InputTranscript)
    assert events[0].text == 'What is the weather?'
    assert events[0].is_final is True


def test_map_turn_complete() -> None:
    msg = StubMessage(server_content=StubServerContent(turn_complete=True))
    events = map_message(msg)
    assert len(events) == 1
    assert isinstance(events[0], TurnComplete)
    assert events[0].interrupted is False


def test_map_turn_complete_interrupted() -> None:
    msg = StubMessage(server_content=StubServerContent(turn_complete=True, interrupted=True))
    events = map_message(msg)
    assert len(events) == 1
    assert isinstance(events[0], TurnComplete)
    assert events[0].interrupted is True


def test_map_tool_call() -> None:
    msg = StubMessage(
        tool_call=StubToolCall(
            function_calls=[StubFunctionCall(id='tc_1', name='get_weather', args={'city': 'London'})]
        )
    )
    events = map_message(msg)
    assert len(events) == 1
    assert isinstance(events[0], ToolCall)
    assert events[0].tool_call_id == 'tc_1'
    assert events[0].tool_name == 'get_weather'
    assert json.loads(events[0].args) == {'city': 'London'}


def test_map_tool_call_no_args() -> None:
    msg = StubMessage(tool_call=StubToolCall(function_calls=[StubFunctionCall(id='tc_2', name='noop', args=None)]))
    events = map_message(msg)
    assert len(events) == 1
    assert isinstance(events[0], ToolCall)
    assert events[0].args == '{}'


def test_map_multiple_tool_calls() -> None:
    msg = StubMessage(
        tool_call=StubToolCall(
            function_calls=[
                StubFunctionCall(id='tc_a', name='foo', args={'x': 1}),
                StubFunctionCall(id='tc_b', name='bar', args={'y': 2}),
            ]
        )
    )
    events = map_message(msg)
    assert len(events) == 2
    assert isinstance(events[0], ToolCall)
    assert events[0].tool_name == 'foo'
    assert isinstance(events[1], ToolCall)
    assert events[1].tool_name == 'bar'


def test_map_tool_call_cancellation() -> None:
    msg = StubMessage(tool_call_cancellation=StubToolCallCancellation(ids=['tc_1']))
    events = map_message(msg)
    assert len(events) == 1
    assert isinstance(events[0], TurnComplete)
    assert events[0].interrupted is True


def test_map_combined_audio_and_transcription() -> None:
    msg = StubMessage(
        server_content=StubServerContent(
            model_turn=StubModelTurn(parts=[StubPart(inline_data=StubInlineData(data=b'\xff'))]),
            output_transcription=StubTranscription(text='response text', finished=False),
        )
    )
    events = map_message(msg)
    assert len(events) == 2
    assert isinstance(events[0], AudioDelta)
    assert isinstance(events[1], Transcript)
    assert events[1].text == 'response text'


def test_map_empty_server_content() -> None:
    msg = StubMessage(server_content=StubServerContent())
    events = map_message(msg)
    assert events == []


def test_map_no_content() -> None:
    msg = StubMessage()
    events = map_message(msg)
    assert events == []


# ---------------------------------------------------------------------------
# Fake Gemini session for connection tests
# ---------------------------------------------------------------------------


class FakeGeminiSession:
    """Simulates the google.genai async live session."""

    def __init__(self, messages: list[Any] | None = None) -> None:
        self._messages = messages or []
        self.sent_audio: list[Any] = []
        self.sent_video: list[Any] = []
        self.sent_tool_responses: list[Any] = []
        self.sent_client_content: list[tuple[Any, bool]] = []

    async def send_realtime_input(self, *, audio: Any = None, video: Any = None) -> None:
        if audio is not None:
            self.sent_audio.append(audio)
        if video is not None:
            self.sent_video.append(video)

    async def send_client_content(self, *, turns: Any = None, turn_complete: bool = True) -> None:
        self.sent_client_content.append((turns, turn_complete))

    async def send_tool_response(self, *, function_responses: list[Any]) -> None:
        self.sent_tool_responses.extend(function_responses)

    async def receive(self) -> AsyncIterator[Any]:
        messages = self._messages
        self._messages = []
        for msg in messages:
            yield msg


# ---------------------------------------------------------------------------
# GeminiRealtimeConnection.send tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_send_audio() -> None:
    session = FakeGeminiSession()
    conn = GeminiRealtimeConnection(session)

    await conn.send(AudioInput(data=b'\x01\x02\x03'))

    assert len(session.sent_audio) == 1
    assert session.sent_audio[0].data == b'\x01\x02\x03'
    assert session.sent_audio[0].mime_type == 'audio/pcm;rate=16000'


@pytest.mark.anyio
async def test_send_image() -> None:
    session = FakeGeminiSession()
    conn = GeminiRealtimeConnection(session)

    await conn.send(ImageInput(data=b'\xff\xd8', mime_type='image/jpeg'))

    assert len(session.sent_video) == 1
    assert session.sent_video[0].data == b'\xff\xd8'
    assert session.sent_video[0].mime_type == 'image/jpeg'


@pytest.mark.anyio
async def test_send_text() -> None:
    session = FakeGeminiSession()
    conn = GeminiRealtimeConnection(session)

    await conn.send(TextInput(text='Hello'))

    assert len(session.sent_client_content) == 1
    turns, turn_complete = session.sent_client_content[0]
    assert turns.role == 'user'
    assert turns.parts[0].text == 'Hello'
    assert turn_complete is True


@pytest.mark.anyio
async def test_send_tool_result() -> None:
    session = FakeGeminiSession()
    conn = GeminiRealtimeConnection(session)

    await conn.send(ToolResult(tool_call_id='tc_1', output='sunny'))

    assert len(session.sent_tool_responses) == 1
    resp = session.sent_tool_responses[0]
    assert resp.id == 'tc_1'
    assert resp.response == {'result': 'sunny'}


# ---------------------------------------------------------------------------
# GeminiRealtimeConnection iteration
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_iterates_and_maps_events() -> None:
    messages = [
        StubMessage(
            server_content=StubServerContent(output_transcription=StubTranscription(text='Hi', finished=False))
        ),
        StubMessage(server_content=StubServerContent(turn_complete=True)),
    ]
    session = FakeGeminiSession(messages)
    conn = GeminiRealtimeConnection(session)

    events = [event async for event in conn]
    assert len(events) == 2
    assert isinstance(events[0], Transcript)
    assert isinstance(events[1], TurnComplete)


# ---------------------------------------------------------------------------
# GeminiRealtimeModel.model_name
# ---------------------------------------------------------------------------


def test_model_name() -> None:
    model = GeminiRealtimeModel(model='gemini-2.5-flash-native-audio-preview')
    assert model.model_name == 'gemini-2.5-flash-native-audio-preview'


def test_default_model_name() -> None:
    model = GeminiRealtimeModel()
    assert model.model_name == 'gemini-2.5-flash-native-audio-preview'


# ---------------------------------------------------------------------------
# Integration tests (WebSocket cassette recording/replay)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_gemini_connect_and_transcript(gemini_realtime_model: GeminiRealtimeModel) -> None:
    """Connect via Agent.realtime_session and verify transcript + turn complete events."""
    agent: Agent[None, str] = Agent(instructions='You are a helpful assistant. Be brief.')

    events: list[RealtimeSessionEvent] = []
    async with agent.realtime_session(model=gemini_realtime_model) as session:
        await session.send_text('Say hello in exactly three words.')
        async for event in session:
            if not isinstance(event, AudioDelta):
                events.append(event)
            if isinstance(event, TurnComplete):
                break

    assert events == snapshot([Transcript(text='Hello there!'), TurnComplete()])


@pytest.mark.anyio
async def test_gemini_tool_call(gemini_realtime_model: GeminiRealtimeModel) -> None:
    """Connect with a tool via Agent.realtime_session and verify tool dispatch."""
    agent: Agent[None, str] = Agent(
        instructions='You are a weather assistant. Always use the get_weather tool when asked about weather.',
    )

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'Sunny in {city}'

    events: list[RealtimeSessionEvent] = []
    async with agent.realtime_session(model=gemini_realtime_model) as session:
        await session.send_text('What is the weather in London?')
        async for event in session:
            if not isinstance(event, AudioDelta):
                events.append(event)
            if isinstance(event, (ToolCallCompleted, TurnComplete)):
                break

    assert events == snapshot(
        [
            ToolCallStarted(tool_name='get_weather', tool_call_id='386960d7-689d-42b1-a7c7-43962a0c8281'),
            ToolCallCompleted(
                tool_name='get_weather', tool_call_id='386960d7-689d-42b1-a7c7-43962a0c8281', result='Sunny in London'
            ),
        ]
    )
