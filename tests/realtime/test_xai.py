"""Tests for the xAI Grok Voice realtime provider (event mapping, handshake, config), all network-free.

xAI's realtime API clones the OpenAI Realtime protocol, so these tests focus on the divergences the
xAI provider adds on top of the shared OpenAI codec (exercised in `test_openai.py`): the session-config
shape, input-transcription events, capabilities, and provider/auth resolution.
"""

from __future__ import annotations as _annotations

import base64
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.realtime import (
    AudioDelta,
    InputTranscript,
    RealtimeCapabilities,
    ToolCall,
    Transcript,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from ..conftest import try_import

with try_import() as imports_successful:
    from xai_sdk import AsyncClient

    from pydantic_ai.providers.xai import XaiProvider
    from pydantic_ai.realtime import xai as rt_xai
    from pydantic_ai.realtime.xai import XaiRealtimeConnection, XaiRealtimeModel, map_event

pytestmark = pytest.mark.skipif(not imports_successful(), reason='xai-sdk / websockets not installed')


def _model(**kwargs: Any) -> XaiRealtimeModel:
    return XaiRealtimeModel(provider=XaiProvider(api_key='k'), **kwargs)


# --- event mapping: the one divergence from the OpenAI codec -------------------------------------


def test_map_input_transcription_updated_is_dropped() -> None:
    """xAI's cumulative `.updated` partials are dropped; the final `.completed` snapshot is authoritative."""
    assert map_event({'type': 'conversation.item.input_audio_transcription.updated', 'delta': 'weath'}) is None


def test_map_input_transcription_completed_delegates_to_openai_codec() -> None:
    event = map_event({'type': 'conversation.item.input_audio_transcription.completed', 'transcript': 'weather?'})
    assert event == InputTranscript(text='weather?', is_final=True)


def test_map_delegates_audio_and_transcript_and_tool_calls() -> None:
    payload = base64.b64encode(b'\x01\x02').decode('ascii')
    assert map_event({'type': 'response.output_audio.delta', 'delta': payload}) == AudioDelta(data=b'\x01\x02')
    assert map_event({'type': 'response.output_audio_transcript.delta', 'delta': 'hel'}) == Transcript(
        text='hel', is_final=False
    )
    assert map_event(
        {'type': 'response.function_call_arguments.done', 'call_id': 'c1', 'name': 'get_weather', 'arguments': '{}'}
    ) == ToolCall(tool_call_id='c1', tool_name='get_weather', args='{}')


def test_connection_map_event_override_matches_module() -> None:
    """`XaiRealtimeConnection` routes frame decoding through the xAI `map_event` (dropping `.updated`)."""
    conn = XaiRealtimeConnection.__new__(XaiRealtimeConnection)
    assert conn._map_event({'type': 'conversation.item.input_audio_transcription.updated', 'delta': 'x'}) is None  # pyright: ignore[reportPrivateUsage]
    assert conn._map_event({'type': 'response.output_audio_transcript.delta', 'delta': 'hi'}) == Transcript(  # pyright: ignore[reportPrivateUsage]
        text='hi', is_final=False
    )


# --- capabilities --------------------------------------------------------------------------------


def test_capabilities() -> None:
    """xAI supports cancellation-based interruption but not output truncation, and no image input."""
    assert _model().capabilities == RealtimeCapabilities(
        image_input=False,
        manual_turn_control=True,
        interruption=True,
        output_truncation=False,
        session_seeding=True,
    )


# --- session config: xAI's shape diverges from OpenAI's GA surface -------------------------------


def test_session_config_shape() -> None:
    """`voice` and `turn_detection` sit at the session top level (unlike OpenAI's nested GA shape)."""
    model = _model(voice='ara')
    tools = [ToolDefinition(name='get_weather', description='Weather', parameters_json_schema={'type': 'object'})]
    config = model._session_config('Be nice', tools, None)  # pyright: ignore[reportPrivateUsage]
    assert config == {
        'instructions': 'Be nice',
        'turn_detection': {'type': 'server_vad', 'create_response': True, 'interrupt_response': True},
        'audio': {
            'input': {'format': {'type': 'audio/pcm', 'rate': 24000}},
            'output': {'format': {'type': 'audio/pcm', 'rate': 24000}},
        },
        'voice': 'ara',
        'tools': [
            {'type': 'function', 'name': 'get_weather', 'description': 'Weather', 'parameters': {'type': 'object'}}
        ],
    }


def test_session_config_transcription_opt_in() -> None:
    """Input transcription is opt-in via `input_transcription_model` → `audio.input.transcription.model`."""
    config = _model(input_transcription_model='grok-transcribe')._session_config('hi', None, None)  # pyright: ignore[reportPrivateUsage]
    assert config['audio']['input']['transcription'] == {'model': 'grok-transcribe'}


def test_session_config_no_transcription_by_default() -> None:
    config = _model()._session_config('hi', None, None)  # pyright: ignore[reportPrivateUsage]
    assert 'transcription' not in config['audio']['input']


def test_session_config_manual_turn_detection_is_null() -> None:
    """`turn_detection=None` disables VAD (push-to-talk), sent as an explicit null."""
    config = _model(turn_detection=None)._session_config('hi', None, None)  # pyright: ignore[reportPrivateUsage]
    assert config['turn_detection'] is None


def test_session_config_no_voice_by_default() -> None:
    """Without an explicit voice, none is sent and the server default (`eve`) applies."""
    assert 'voice' not in _model()._session_config('hi', None, None)  # pyright: ignore[reportPrivateUsage]


def test_session_config_forwards_model_settings() -> None:
    settings = ModelSettings(max_tokens=256, parallel_tool_calls=False, tool_choice='required')
    config = _model()._session_config('hi', None, settings)  # pyright: ignore[reportPrivateUsage]
    assert config['max_output_tokens'] == 256
    assert config['parallel_tool_calls'] is False
    assert config['tool_choice'] == 'required'


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


def _created() -> str:
    return json.dumps({'type': 'session.created'})


def _updated() -> str:
    return json.dumps({'type': 'session.updated'})


@pytest.mark.anyio
async def test_connect_handshake_url_auth_and_session_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The URL, bearer auth, and `session.update` frame are derived from the xAI provider."""
    # A dropped `.updated` partial followed by a real transcript proves the xAI codec is wired in.
    updated_partial = json.dumps({'type': 'conversation.item.input_audio_transcription.updated', 'delta': 'ignore'})
    transcript = json.dumps({'type': 'response.output_audio_transcript.done', 'transcript': 'hi'})
    ws = FakeWebSocket([_created(), _updated(), updated_partial, transcript])
    fake_connect = FakeConnect(ws)
    monkeypatch.setattr(rt_xai.websockets, 'connect', fake_connect)

    model = XaiRealtimeModel('grok-voice-latest', provider=XaiProvider(api_key='k'), voice='eve')
    async with model.connect(instructions='Be nice') as conn:
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
async def test_connect_rejects_native_tools() -> None:
    from pydantic_ai.native_tools import WebSearchTool

    model = _model()
    with pytest.raises(UserError, match='does not support native tools'):
        async with model.connect(instructions='hi', native_tools=[WebSearchTool()]):
            pass  # pragma: no cover


@pytest.mark.anyio
async def test_connect_seeds_message_history_as_output_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Seeded assistant turns are sent as `output_text` items (as xAI, like OpenAI, expects)."""
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_xai.websockets, 'connect', FakeConnect(ws))
    history = [
        ModelRequest(parts=[UserPromptPart(content='My name is Alice.')]),
        ModelResponse(parts=[TextPart(content='Hi Alice!')]),
    ]

    model = _model()
    async with model.connect(instructions='hi', messages=history) as conn:
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
    async with model.connect(instructions='hi'):
        pass
    assert fake_connect.headers == {'Authorization': 'Bearer env-key'}


def test_provider_from_xai_client_without_exposed_key_raises() -> None:
    """A provider built from a pre-configured `xai_client` can't expose its key, so realtime errors clearly."""
    provider = XaiProvider(xai_client=AsyncClient(api_key='hidden'))
    with pytest.raises(UserError, match='pre-configured `xai_client`'):
        XaiRealtimeModel(provider=provider)
