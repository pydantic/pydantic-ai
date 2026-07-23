"""Tests for the xAI Grok Voice realtime provider (event mapping, handshake, config), all network-free.

xAI's realtime API clones the OpenAI Realtime protocol, so these tests focus on the divergences the
xAI provider adds on top of the shared OpenAI codec (exercised in `test_openai.py`): the session-config
shape, input-transcription events, capabilities, and provider/auth resolution.
"""

from __future__ import annotations as _annotations

import base64
import json
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, Literal

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SpeechPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.realtime import (
    RealtimeModelProfile,
    ReconnectedEvent,
    TurnDetection,
)
from pydantic_ai.realtime._base import ConversationCreated, ConversationItemCreated, SessionErrorEvent
from pydantic_ai.realtime.codec import (
    AudioDelta,
    InputTranscript,
    ToolCall,
    Transcript,
)
from pydantic_ai.tools import ToolDefinition

from ..conftest import try_import

with try_import() as imports_successful:
    from xai_sdk import AsyncClient

    from pydantic_ai.providers.xai import XaiProvider
    from pydantic_ai.realtime import xai as rt_xai
    from pydantic_ai.realtime.xai import XaiRealtimeConnection, XaiRealtimeModel, map_event

pytestmark = pytest.mark.skipif(not imports_successful(), reason='xai-sdk / websockets not installed')


def _model(settings: rt_xai.XaiRealtimeModelSettings | None = None, **kwargs: Any) -> XaiRealtimeModel:
    return XaiRealtimeModel(provider=XaiProvider(api_key='k'), settings=settings, **kwargs)


def test_realtime_rejects_custom_api_host() -> None:
    """A custom `api_host` sets the gRPC channel target, which the realtime WebSocket can't honor (it
    derives its URL from `base_url`), so construction fails loudly rather than dialing the wrong host."""
    with pytest.raises(UserError, match='does not support a custom `api_host`'):
        XaiRealtimeModel(provider=XaiProvider(api_key='k', api_host='grpc.custom.example.com'))


def _connect(
    model: XaiRealtimeModel,
    instructions: str,
    *,
    messages: Sequence[ModelMessage] | None = None,
) -> AbstractAsyncContextManager[XaiRealtimeConnection]:
    return model.connect(
        messages=[*(messages or ()), ModelRequest(parts=[], instructions=instructions)],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )


# --- event mapping: the one divergence from the OpenAI codec -------------------------------------


def test_map_input_transcription_updated_is_dropped() -> None:
    """xAI's cumulative `.updated` partials are dropped; the final `.completed` snapshot is authoritative."""
    assert map_event({'type': 'conversation.item.input_audio_transcription.updated', 'delta': 'weath'}) is None


def test_map_input_transcription_completed_delegates_to_openai_codec() -> None:
    event = map_event({'type': 'conversation.item.input_audio_transcription.completed', 'transcript': 'weather?'})
    assert event == InputTranscript(text='weather?', is_final=True)


def test_map_tool_call_preserves_xai_item_id() -> None:
    assert map_event(
        {
            'type': 'response.function_call_arguments.done',
            'call_id': 'call-1',
            'name': 'weather',
            'arguments': '{}',
            'item_id': 'item-1',
        }
    ) == ToolCall(
        tool_call_id='call-1',
        tool_name='weather',
        args='{}',
        item_id='item-1',
        response_usage_follows=True,
    )

    event = map_event(
        {
            'type': 'response.function_call_arguments.done',
            'call_id': 'call-2',
            'name': 'weather',
            'arguments': '{}',
            'item_id': '',
        }
    )
    assert isinstance(event, ToolCall) and event.item_id is None


def test_map_input_transcription_completed_respects_status() -> None:
    base = {
        'type': 'conversation.item.input_audio_transcription.completed',
        'item_id': 'item-1',
        'transcript': 'weather?',
    }
    assert map_event({**base, 'status': 'in_progress'}) is None
    assert map_event({**base, 'status': 'completed'}) == InputTranscript(
        text='weather?', is_final=True, item_id='item-1'
    )


def test_map_delegates_audio_and_transcript_and_tool_calls() -> None:
    payload = base64.b64encode(b'\x01\x02').decode('ascii')
    assert map_event({'type': 'response.output_audio.delta', 'delta': payload}) == AudioDelta(data=b'\x01\x02')
    assert map_event({'type': 'response.output_audio_transcript.delta', 'delta': 'hel'}) == Transcript(
        text='hel', is_final=False
    )
    assert map_event(
        {
            'type': 'response.function_call_arguments.done',
            'item_id': 'item-call',
            'call_id': 'c1',
            'name': 'get_weather',
            'arguments': '{}',
        }
    ) == ToolCall(
        tool_call_id='c1',
        tool_name='get_weather',
        args='{}',
        response_usage_follows=True,
        item_id='item-call',
    )


def test_map_conversation_resumption_events() -> None:
    assert map_event({'type': 'conversation.created', 'conversation': {'id': 'conversation-1'}}) == ConversationCreated(
        'conversation-1'
    )
    # A live-stream item lifecycle event is never a resumption replay (only the reconnect handshake's
    # burst-capture marks items `replayed=True`), so it maps with `replayed=False` and is not suppressed.
    assert map_event(
        {
            'type': 'conversation.item.created',
            'item': {'id': 'item-1', 'type': 'function_call', 'call_id': 'call-1'},
        }
    ) == ConversationItemCreated(item_id='item-1', tool_call_id='call-1', replayed=False)


def test_connection_map_event_override_matches_module() -> None:
    """`XaiRealtimeConnection` routes frame decoding through the xAI `map_event` (dropping `.updated`)."""
    conn = XaiRealtimeConnection.__new__(XaiRealtimeConnection)
    assert conn._map_event({'type': 'conversation.item.input_audio_transcription.updated', 'delta': 'x'}) is None  # pyright: ignore[reportPrivateUsage]
    assert conn._map_event({'type': 'response.output_audio_transcript.delta', 'delta': 'hi'}) == Transcript(  # pyright: ignore[reportPrivateUsage]
        text='hi', is_final=False
    )


# --- capabilities --------------------------------------------------------------------------------


def test_profile() -> None:
    """xAI supports cancellation-based interruption but not output truncation, and no image input."""
    assert _model().profile == RealtimeModelProfile(
        supports_image_input=False,
        supports_manual_turn_control=True,
        supports_interruption=True,
        supports_output_truncation=False,
        supports_session_seeding=True,
        supports_seeding_images=False,
        supports_seeding_audio=False,
        audio_input_sample_rate=24000,
        audio_output_sample_rate=24000,
        supported_native_tools=frozenset(),
    )


# --- session config: xAI's shape diverges from OpenAI's GA surface -------------------------------


def test_session_config_shape() -> None:
    """`voice` and `turn_detection` sit at the session top level (unlike OpenAI's nested GA shape)."""
    model = _model(rt_xai.XaiRealtimeModelSettings(voice='ara'))
    tools = [ToolDefinition(name='get_weather', description='Weather', parameters_json_schema={'type': 'object'})]
    config = model._session_config('Be nice', tools, None)  # pyright: ignore[reportPrivateUsage]
    assert config == {
        'instructions': 'Be nice',
        'turn_detection': {'type': 'server_vad', 'create_response': True, 'interrupt_response': True},
        'audio': {
            'input': {
                'format': {'type': 'audio/pcm', 'rate': 24000},
                'transcription': {'model': 'grok-transcribe'},  # on by default
            },
            'output': {'format': {'type': 'audio/pcm', 'rate': 24000}},
        },
        'voice': 'ara',
        'tools': [
            {'type': 'function', 'name': 'get_weather', 'description': 'Weather', 'parameters': {'type': 'object'}}
        ],
    }


def test_session_config_resumption_follows_reconnect_policy() -> None:
    assert 'resumption' not in _model()._session_config('hi', None, None)  # pyright: ignore[reportPrivateUsage]
    config = _model(reconnect=rt_xai.ReconnectPolicy())._session_config('hi', None, None)  # pyright: ignore[reportPrivateUsage]
    assert config['resumption'] == {'enabled': True}


def test_session_config_transcription_auto_by_default() -> None:
    """The default `input_transcription_model='auto'` resolves to xAI's recommended transcription model
    (`grok-transcribe`) → `audio.input.transcription.model`, so the user's audio turns are transcribed
    into history under the default `transcript_only` retention (they'd otherwise be dropped)."""
    config = _model()._session_config('hi', None, None)  # pyright: ignore[reportPrivateUsage]
    assert config['audio']['input']['transcription'] == {'model': 'grok-transcribe'}


def test_session_config_transcription_explicit_override() -> None:
    """An explicit model id is used verbatim, overriding the `'auto'` default."""
    config = _model()._session_config(  # pyright: ignore[reportPrivateUsage]
        'hi', None, rt_xai.XaiRealtimeModelSettings(input_transcription_model='grok-transcribe-next')
    )
    assert config['audio']['input']['transcription'] == {'model': 'grok-transcribe-next'}


def test_session_config_transcription_disabled() -> None:
    """`input_transcription_model=None` opts out of transcription."""
    config = _model()._session_config(  # pyright: ignore[reportPrivateUsage]
        'hi', None, rt_xai.XaiRealtimeModelSettings(input_transcription_model=None)
    )
    assert 'transcription' not in config['audio']['input']


def test_session_config_manual_turn_detection_is_null() -> None:
    """`turn_detection=False` disables VAD (push-to-talk), sent as an explicit null."""
    config = _model()._session_config(  # pyright: ignore[reportPrivateUsage]
        'hi', None, rt_xai.XaiRealtimeModelSettings(turn_detection=False)
    )
    assert config['turn_detection'] is None


@pytest.mark.parametrize(('sensitivity', 'threshold'), [('low', 0.7), ('medium', 0.5), ('high', 0.3)])
def test_session_config_cross_provider_turn_detection_sensitivity(
    sensitivity: Literal['low', 'medium', 'high'], threshold: float
) -> None:
    config = _model()._session_config(  # pyright: ignore[reportPrivateUsage]
        'hi',
        None,
        rt_xai.XaiRealtimeModelSettings(turn_detection=TurnDetection(sensitivity=sensitivity)),
    )
    assert config['turn_detection']['threshold'] == threshold


def test_session_config_xai_turn_detection_overrides_base() -> None:
    config = _model()._session_config(  # pyright: ignore[reportPrivateUsage]
        'hi',
        None,
        rt_xai.XaiRealtimeModelSettings(
            turn_detection=TurnDetection(sensitivity='high'),
            xai_turn_detection=rt_xai.ServerVAD(threshold=0.9, create_response=False),
        ),
    )
    assert config['turn_detection'] == {
        'type': 'server_vad',
        'create_response': False,
        'interrupt_response': True,
        'threshold': 0.9,
    }


def test_session_config_no_voice_by_default() -> None:
    """Without an explicit voice, none is sent and the server default (`eve`) applies."""
    assert 'voice' not in _model()._session_config('hi', None, None)  # pyright: ignore[reportPrivateUsage]


def test_session_config_forwards_model_settings() -> None:
    settings = rt_xai.XaiRealtimeModelSettings(max_tokens=256, parallel_tool_calls=False, tool_choice='required')
    model = _model(settings=settings)
    assert model.settings == settings
    config = model._session_config('hi', None, settings)  # pyright: ignore[reportPrivateUsage]
    assert config['max_output_tokens'] == 256
    assert config['parallel_tool_calls'] is False
    assert config['tool_choice'] == 'required'


def test_session_config_omits_absent_model_settings() -> None:
    """Absent realtime settings are omitted from the session config."""
    config = _model()._session_config('hi', None, rt_xai.XaiRealtimeModelSettings())  # pyright: ignore[reportPrivateUsage]
    assert 'max_output_tokens' not in config
    assert 'parallel_tool_calls' not in config
    assert 'tool_choice' not in config


# --- connect: handshake, URL, auth, seeding ------------------------------------------------------


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


class _DropAfterHandshake(FakeWebSocket):
    """Completes the handshake (via `recv`), then drops when iterated."""

    async def __aiter__(self) -> AsyncIterator[Any]:
        raise rt_xai.websockets.ConnectionClosed(None, None)
        yield  # pragma: no cover  (makes this an async generator)


class _DropAfterFrames(FakeWebSocket):
    """Yields all post-handshake frames, then simulates an abnormal connection loss."""

    async def __aiter__(self) -> AsyncIterator[Any]:
        while self._incoming:
            yield self._incoming.pop(0)
        raise rt_xai.websockets.ConnectionClosed(None, None)


class _RecordingConnect:
    """Stand-in for `websockets.connect` that hands out sockets in order and records closes."""

    def __init__(self, sockets: list[FakeWebSocket]) -> None:
        self._sockets = iter(sockets)
        self.closed: list[FakeWebSocket] = []
        self.urls: list[str] = []

    def __call__(self, url: str, *, additional_headers: dict[str, str] | None = None) -> Any:
        self.urls.append(url)
        ws = next(self._sockets)
        recorder = self

        class _CM:
            async def __aenter__(self) -> FakeWebSocket:
                return ws

            async def __aexit__(self, *exc: object) -> bool:
                recorder.closed.append(ws)
                return False

        return _CM()


def _created() -> str:
    return json.dumps({'type': 'session.created'})


def _updated() -> str:
    return json.dumps({'type': 'session.updated'})


def _conversation_created(conversation_id: str = 'conversation-1') -> str:
    return json.dumps({'type': 'conversation.created', 'conversation': {'id': conversation_id}})


@pytest.mark.anyio
async def test_connect_captures_substituted_server_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # xAI accepts any model slug — even a retired or misspelled one — and silently substitutes its
    # current default, reporting the actually-served model only in `session.created`. Capturing it is
    # the only way a session's history can show what model really answered.
    created = json.dumps({'type': 'session.created', 'session': {'model': 'grok-voice-latest'}})
    ws = FakeWebSocket([created, _updated()])
    monkeypatch.setattr(rt_xai.websockets, 'connect', FakeConnect(ws))
    async with _connect(_model(model='grok-voice-retired-1.0'), 'x') as conn:
        assert conn.model_name == 'grok-voice-latest'


@pytest.mark.anyio
async def test_connect_handshake_url_auth_and_session_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The URL, bearer auth, and `session.update` frame are derived from the xAI provider."""
    # A dropped `.updated` partial followed by a real transcript proves the xAI codec is wired in.
    updated_partial = json.dumps({'type': 'conversation.item.input_audio_transcription.updated', 'delta': 'ignore'})
    transcript = json.dumps({'type': 'response.output_audio_transcript.done', 'transcript': 'hi'})
    ws = FakeWebSocket([_created(), _updated(), updated_partial, transcript])
    fake_connect = FakeConnect(ws)
    monkeypatch.setattr(rt_xai.websockets, 'connect', fake_connect)

    model = XaiRealtimeModel(
        'grok-voice-latest',
        provider=XaiProvider(api_key='k'),
        settings=rt_xai.XaiRealtimeModelSettings(voice='eve'),
    )
    async with _connect(model, 'Be nice') as conn:
        assert isinstance(conn, XaiRealtimeConnection)
        events = [e async for e in conn]

    assert events == [Transcript(text='hi', is_final=True)]  # the `.updated` partial was dropped
    assert fake_connect.url == 'wss://api.x.ai/v1/realtime?model=grok-voice-latest'
    assert fake_connect.headers == {'Authorization': 'Bearer k'}

    update = json.loads(ws.sent[0])
    assert update['type'] == 'session.update'
    assert update['session']['instructions'] == 'Be nice'
    assert update['session']['voice'] == 'eve'


@pytest.mark.anyio
async def test_connect_injects_trace_context_into_handshake(monkeypatch: pytest.MonkeyPatch) -> None:
    """An active span propagates `traceparent` into the handshake headers (see the OpenAI provider test)."""
    pytest.importorskip('opentelemetry.sdk')
    from opentelemetry.sdk.trace import TracerProvider

    ws = FakeWebSocket([_created(), _updated()])
    fake_connect = FakeConnect(ws)
    monkeypatch.setattr(rt_xai.websockets, 'connect', fake_connect)

    model = XaiRealtimeModel('grok-voice-latest', provider=XaiProvider(api_key='k'))
    tracer = TracerProvider().get_tracer('test')
    with tracer.start_as_current_span('root'):
        async with _connect(model, 'hi') as conn:
            _ = [e async for e in conn]

    assert fake_connect.headers is not None
    assert fake_connect.headers['Authorization'] == 'Bearer k'
    assert 'traceparent' in fake_connect.headers


@pytest.mark.anyio
async def test_agent_realtime_session_rejects_native_tools() -> None:
    # xAI Grok Voice supports no native tools, so any native tool fails up front with the uniform
    # error, before dialing — the check lives in `Agent.realtime_session`, keyed on the model profile.
    agent: Agent[None, str] = Agent()
    with pytest.raises(
        UserError,
        match=r'does not support the WebSearchTool native tool\(s\)\. Supported native tools: none\.',
    ):
        async with agent.realtime_session(model=_model(), capabilities=[NativeTool(WebSearchTool())]):
            pass  # pragma: no cover - validation raises before yielding


@pytest.mark.anyio
async def test_connect_seeds_message_history_as_output_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Seeded assistant turns are sent as `output_text` items (as xAI, like OpenAI, expects)."""
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_xai.websockets, 'connect', FakeConnect(ws))
    history = [
        ModelRequest(parts=[UserPromptPart(content='My name is Alice.')]),
        ModelResponse(parts=[TextPart(content='Hi Alice!')]),
    ]

    model = _model()
    async with _connect(model, 'hi', messages=history) as conn:
        assert isinstance(conn, XaiRealtimeConnection)

    seeded = [json.loads(frame) for frame in ws.sent[1:]]  # ws.sent[0] is the session.update handshake
    assert seeded == [
        {
            'type': 'conversation.item.create',
            'item': {
                'type': 'message',
                'role': 'user',
                'content': [{'type': 'input_text', 'text': 'My name is Alice.'}],
            },
        },
        {
            'type': 'conversation.item.create',
            'item': {
                'type': 'message',
                'role': 'assistant',
                'content': [{'type': 'output_text', 'text': 'Hi Alice!'}],
            },
        },
    ]


@pytest.mark.anyio
@pytest.mark.parametrize('image_kind', ['url', 'binary'])
async def test_connect_rejects_seeded_image(monkeypatch: pytest.MonkeyPatch, image_kind: str) -> None:
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_xai.websockets, 'connect', FakeConnect(ws))
    image = (
        ImageUrl(url='https://example.com/image.png')
        if image_kind == 'url'
        else BinaryContent(data=b'image', media_type='image/png')
    )
    history = [ModelRequest(parts=[UserPromptPart(content=[image])])]

    with pytest.raises(UserError, match='xai realtime history seeding does not support images'):
        async with _connect(_model(), 'x', messages=history):
            pass  # pragma: no cover


@pytest.mark.anyio
async def test_connect_rejects_seeded_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_xai.websockets, 'connect', FakeConnect(ws))
    history = [
        ModelRequest(parts=[SpeechPart(speaker='user', audio=BinaryContent(data=b'audio', media_type='audio/pcm'))])
    ]

    with pytest.raises(UserError, match='xai realtime history seeding does not support retained user audio'):
        async with _connect(_model(), 'x', messages=history):
            pass  # pragma: no cover


@pytest.mark.anyio
async def test_connect_reconnect_closes_previous_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reconnect through `connect()`'s own dial closes the dropped socket before opening the next."""
    transcript = json.dumps({'type': 'response.output_audio_transcript.done', 'transcript': 'hi'})
    dropped = _DropAfterHandshake([_created(), _conversation_created(), _updated()])
    good = FakeWebSocket([_created(), _conversation_created(), _updated(), transcript])
    connect = _RecordingConnect([dropped, good])
    monkeypatch.setattr(rt_xai.websockets, 'connect', connect)

    model = _model(reconnect=rt_xai.ReconnectPolicy(base_delay=0.0))
    async with _connect(model, 'x') as conn:
        events = [e async for e in conn]

    assert events == [ReconnectedEvent(state_restored=True), Transcript(text='hi', is_final=True)]
    assert connect.closed == [dropped, good]  # both the dropped and the current socket are closed
    assert connect.urls == [
        'wss://api.x.ai/v1/realtime?model=grok-voice-latest',
        'wss://api.x.ai/v1/realtime?model=grok-voice-latest&conversation_id=conversation-1',
    ]
    assert json.loads(dropped.sent[0])['session']['resumption'] == {'enabled': True}
    assert json.loads(good.sent[0])['session']['resumption'] == {'enabled': True}


@pytest.mark.anyio
async def test_reconnect_replay_burst_is_deduplicated_from_session_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resumed items are suppressed even when xAI assigns new IDs to the replayed copies."""
    dropped = _DropAfterFrames(
        [
            _created(),
            _conversation_created(),
            _updated(),
            json.dumps(
                {
                    'type': 'conversation.item.added',
                    'item': {'id': 'item-user', 'type': 'message', 'role': 'user'},
                }
            ),
            json.dumps(
                {
                    'type': 'response.output_audio_transcript.done',
                    'item_id': 'item-assistant',
                    'transcript': 'Hello back.',
                }
            ),
            json.dumps({'type': 'response.done', 'response': {'id': 'response-1', 'status': 'completed'}}),
        ]
    )
    resumed = FakeWebSocket(
        [
            _created(),
            _conversation_created(),
            json.dumps(
                {
                    'type': 'conversation.item.added',
                    'item': {'id': 'replayed-item-user', 'type': 'message', 'role': 'user'},
                }
            ),
            json.dumps(
                {
                    'type': 'conversation.item.added',
                    'item': {'id': 'replayed-item-assistant', 'type': 'message', 'role': 'assistant'},
                }
            ),
            _updated(),
            # Defensive duplicate content after the replay marker proves suppression happens by ID,
            # rather than merely because `conversation.item.created` itself has no history mapping.
            json.dumps(
                {
                    'type': 'response.output_audio_transcript.done',
                    'item_id': 'replayed-item-assistant',
                    'transcript': 'Hello back.',
                }
            ),
        ]
    )
    monkeypatch.setattr(rt_xai.websockets, 'connect', _RecordingConnect([dropped, resumed]))

    agent = Agent()
    async with agent.realtime_session(model=_model(reconnect=rt_xai.ReconnectPolicy(base_delay=0.0))) as session:
        await session.send('Hello.')
        events = [event async for event in session]

    assert sum(isinstance(event, ReconnectedEvent) for event in events) == 1
    messages = session.all_messages()
    assert len(messages) == 2
    assert isinstance(messages[0], ModelRequest)
    assert isinstance(messages[0].parts[0], UserPromptPart)
    assert messages[0].parts[0].content == 'Hello.'
    assert isinstance(messages[1], ModelResponse)
    assert messages[1].parts == [
        SpeechPart(
            speaker='assistant',
            transcript='Hello back.',
            id='item-assistant',
            provider_name='xai',
        )
    ]


@pytest.mark.anyio
async def test_connect_reconnect_failure_leaves_nothing_to_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed reconnect through `connect()`'s dial leaves nothing to close on teardown.

    The dial nulls `cm` before re-dialing, so when the re-dial fails (an expected `OSError`) and the
    session ends via a `SessionErrorEvent`, teardown finds `cm` already `None` and skips the close.
    """
    dropped = _DropAfterHandshake([_created(), _conversation_created(), _updated()])

    class _DropThenFail:
        """First `connect()` yields a socket that drops after the handshake; the re-dial refuses."""

        def __init__(self) -> None:
            self.calls = 0
            self.closed: list[str] = []

        def __call__(self, url: str, *, additional_headers: dict[str, str] | None = None) -> Any:
            self.calls += 1
            first = self.calls == 1
            recorder = self

            class _CM:
                async def __aenter__(self) -> FakeWebSocket:
                    if first:
                        return dropped
                    raise OSError('refused')  # an expected dial failure → reconnect gives up

                async def __aexit__(self, *exc: object) -> bool:
                    recorder.closed.append('dropped' if first else 'refused')
                    return False

            return _CM()

    connect = _DropThenFail()
    monkeypatch.setattr(rt_xai.websockets, 'connect', connect)
    model = _model(reconnect=rt_xai.ReconnectPolicy(max_attempts=1, base_delay=0.0, jitter=False))
    async with _connect(model, 'x') as conn:
        events = [e async for e in conn]

    assert any(isinstance(e, SessionErrorEvent) and not e.recoverable for e in events)
    # The dropped socket is closed as the reconnect nulls `cm` before re-dialing; the refused re-dial
    # never enters its context manager, so `cm` stays `None` and teardown closes nothing further. A
    # regression that assigned `cm` before awaiting `__aenter__` would leave `'refused'` here.
    assert connect.closed == ['dropped']


@pytest.mark.anyio
async def test_connect_open_failure_propagates_without_teardown(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the very first connection fails to open, there is nothing to close on teardown."""

    class _FailingConnect:
        def __call__(self, url: str, *, additional_headers: dict[str, str] | None = None) -> Any:
            return self

        async def __aenter__(self) -> Any:
            raise ConnectionError('refused')

        async def __aexit__(self, *exc: object) -> bool:  # pragma: no cover — never entered
            return False

    monkeypatch.setattr(rt_xai.websockets, 'connect', _FailingConnect())
    with pytest.raises(ConnectionError, match='refused'):
        async with _connect(_model(), 'x'):
            pass  # pragma: no cover


@pytest.mark.anyio
async def test_connect_rejects_conversation_created_without_id(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeWebSocket([_created(), json.dumps({'type': 'conversation.created', 'conversation': {}})])
    monkeypatch.setattr(rt_xai.websockets, 'connect', FakeConnect(ws))

    with pytest.raises(RuntimeError, match=r'did not include `conversation\.id`'):
        async with _connect(_model(reconnect=rt_xai.ReconnectPolicy()), 'x'):
            pass  # pragma: no cover


# --- provider / auth resolution ------------------------------------------------------------------


@pytest.mark.anyio
async def test_provider_str_resolves_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default `provider='xai'` reads `XAI_API_KEY`, which becomes the WebSocket bearer token."""
    monkeypatch.setenv('XAI_API_KEY', 'env-key')
    ws = FakeWebSocket([_created(), _updated()])
    fake_connect = FakeConnect(ws)
    monkeypatch.setattr(rt_xai.websockets, 'connect', fake_connect)

    model = XaiRealtimeModel()
    assert model.model_name == 'grok-voice-latest'
    async with _connect(model, 'hi'):
        pass
    assert fake_connect.headers == {'Authorization': 'Bearer env-key'}


def test_provider_from_xai_client_without_exposed_key_raises() -> None:
    """A provider built from a pre-configured `xai_client` can't expose its key, so realtime errors clearly."""
    provider = XaiProvider(xai_client=AsyncClient(api_key='hidden'))
    with pytest.raises(UserError, match='pre-configured `xai_client`'):
        XaiRealtimeModel(provider=provider)
