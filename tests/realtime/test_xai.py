"""Tests for the xAI Grok Voice realtime provider (event mapping, handshake, config), all network-free.

xAI's realtime API clones the OpenAI Realtime protocol, so these tests focus on the divergences the
xAI provider adds on top of the shared OpenAI codec (exercised in `test_openai.py`): the session-config
shape, input-transcription events, capabilities, and provider/auth resolution.
"""

from __future__ import annotations as _annotations

import asyncio
import base64
import json
from collections import deque
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, Literal, cast

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.exceptions import ModelAPIError, UserError
from pydantic_ai.messages import (
    BinaryAudio,
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    RealtimeInputSpeechStartEvent,
    RealtimeSessionErrorEvent,
    SpeechPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.realtime import (
    RealtimeModelProfile,
    RealtimeSessionReconnectEvent,
)
from pydantic_ai.realtime.codec import (
    AudioDelta,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    ConversationCreated,
    ConversationItemCreated,
    CreateResponse,
    InputRejected,
    InputTranscript,
    OutputTranscript,
    RealtimeInput,
    ResponseDone,
    SessionUsage,
    TextContext,
    ToolCall,
    ToolResult,
)
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from ..conftest import IsStr, try_import
from .ws_helpers import collect_codec_events, collect_session_events

with try_import() as imports_successful:
    from openai.types.realtime import RealtimeResponseUsage
    from xai_sdk import AsyncClient

    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.xai import XaiProvider
    from pydantic_ai.realtime import xai as rt_xai
    from pydantic_ai.realtime.xai import XaiRealtimeConnection, XaiRealtimeModel, map_event as _map_wire_event

from .test_openai import sdk_frame

pytestmark = pytest.mark.skipif(not imports_successful(), reason='xai-sdk / websockets not installed')


def map_event(frame: dict[str, Any]) -> object:
    return _map_wire_event(sdk_frame(frame))


def test_xai_public_exports_are_curated() -> None:
    assert rt_xai.__all__ == (
        'XaiRealtimeModel',
        'XaiRealtimeModelSettings',
        'XaiRealtimeConnection',
        'map_event',
    )


def _model(settings: rt_xai.XaiRealtimeModelSettings | None = None, **kwargs: Any) -> XaiRealtimeModel:
    model = kwargs.pop('model', 'grok-voice-latest')
    return XaiRealtimeModel(model, provider=XaiProvider(api_key='k'), settings=settings, **kwargs)


def test_realtime_rejects_custom_api_host() -> None:
    """A custom `api_host` sets the gRPC channel target, which the realtime WebSocket can't honor (it
    derives its URL from `base_url`), so construction fails loudly rather than dialing the wrong host."""
    with pytest.raises(UserError, match='does not support a custom `api_host`'):
        XaiRealtimeModel('grok-voice-latest', provider=XaiProvider(api_key='k', api_host='grpc.custom.example.com'))


async def test_connection_send_audio_rejects_non_pcm_media_type() -> None:
    ws = FakeWebSocket([])
    conn = XaiRealtimeConnection(ws)  # type: ignore[arg-type]
    with pytest.raises(UserError, match='require raw PCM audio'):
        await conn.send(BinaryAudio(data=b'RIFF', media_type='audio/wav'))
    assert ws.sent == []


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


def test_map_input_transcription_updated_is_a_cumulative_partial() -> None:
    """xAI's `.updated` partials carry the whole transcript so far, not an incremental piece."""
    assert map_event(
        {
            'type': 'conversation.item.input_audio_transcription.updated',
            'transcript': 'Hello, my name is',
            'item_id': 'item-1',
        }
    ) == InputTranscript(text='Hello, my name is', cumulative=True, item_id='item-1')


@pytest.mark.parametrize(
    'frame,expected',
    [
        pytest.param({}, InputTranscript(text='', cumulative=True), id='no-transcript'),
        pytest.param({'transcript': None}, InputTranscript(text='', cumulative=True), id='null-transcript'),
        pytest.param({'item_id': 7, 'transcript': 'hi'}, InputTranscript(text='hi', cumulative=True), id='bad-item-id'),
    ],
)
def test_map_input_transcription_updated_tolerates_a_thin_frame(frame: dict[str, Any], expected: object) -> None:
    """The `.updated` frame has no SDK model behind it, so it is read defensively off the wire."""
    if frame.get('item_id') == 7:
        with pytest.raises(ValueError):
            map_event({'type': 'conversation.item.input_audio_transcription.updated', **frame})
    else:
        assert map_event({'type': 'conversation.item.input_audio_transcription.updated', **frame}) == expected


def test_map_input_transcription_completed_delegates_to_openai_codec() -> None:
    """The final snapshot is read through the OpenAI codec, but still marked cumulative.

    xAI's `.completed` carries the whole transcript, like its `.updated` partials. Read as an increment
    it would be appended to the snapshots it supersedes, so a turn xAI revised mid-flight ends up saying
    everything twice (measured live: `'Hello, my name.'` then `'Hello, my name is Marcelo.'` became
    `'Hello, my name.Hello, my name is Marcelo.'`). `test_session`'s
    `test_cumulative_transcripts_revise_the_turn_instead_of_doubling_up` pins the session half.
    """
    event = map_event({'type': 'conversation.item.input_audio_transcription.completed', 'transcript': 'weather?'})
    assert event == InputTranscript(text='weather?', is_final=True, cumulative=True)


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
        response_id='response',
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
        text='weather?', is_final=True, item_id='item-1', cumulative=True
    )


def test_map_delegates_audio_and_transcript_and_tool_calls() -> None:
    payload = base64.b64encode(b'\x01\x02').decode('ascii')
    assert map_event({'type': 'response.output_audio.delta', 'delta': payload}) == AudioDelta(
        data=b'\x01\x02', response_id='response'
    )
    assert map_event({'type': 'response.output_audio_transcript.delta', 'delta': 'hel'}) == OutputTranscript(
        text='hel', is_final=False, response_id='response'
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
        response_id='response',
    )


def _error_frame(error_type: str, message: str) -> dict[str, Any]:
    """An `error` frame shaped as in xAI's realtime schema, whose `code` repeats the `type`."""
    return {
        'type': 'error',
        'event_id': 'event_err01',
        'error': {'type': error_type, 'code': error_type, 'message': message},
    }


def test_map_max_duration_error_ends_the_session() -> None:
    # xAI ends a conversation that runs past its maximum duration, so resuming it would hit the same limit.
    assert map_event(_error_frame('max_duration', 'Maximum conversation duration exceeded.')) == (
        RealtimeSessionErrorEvent(
            message='Maximum conversation duration exceeded.',
            type='max_duration',
            code='max_duration',
            recoverable=False,
        )
    )
    # The inactivity `timeout` keeps the session open, as the schema says most errors do.
    assert map_event(_error_frame('timeout', 'Inactivity timeout.')) == RealtimeSessionErrorEvent(
        message='Inactivity timeout.', type='timeout', code='timeout', recoverable=True
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
    """`XaiRealtimeConnection` routes frame decoding through the xAI `map_event` (cumulative `.updated`)."""
    conn = XaiRealtimeConnection(FakeWebSocket([]))  # type: ignore[arg-type]
    assert conn._map_event(  # pyright: ignore[reportPrivateUsage]
        {'type': 'conversation.item.input_audio_transcription.updated', 'transcript': 'x'}
    ) == InputTranscript(text='x', cumulative=True)
    assert conn._map_event(  # pyright: ignore[reportPrivateUsage]
        sdk_frame({'type': 'response.output_audio_transcript.delta', 'delta': 'hi'})
    ) == OutputTranscript(text='hi', is_final=False, response_id='response')


async def test_connection_send_tool_result_image_raises_with_nothing_sent() -> None:
    """Grok Voice has no image input, so an image attached to a tool result raises before any frame
    goes out — instead of the shared codec's follow-up user message — rather than degrading silently."""
    ws = FakeWebSocket([])
    conn = XaiRealtimeConnection(ws)  # type: ignore[arg-type]
    with pytest.raises(UserError, match='xai realtime sessions do not support images'):
        await conn.send(
            ToolResult(
                tool_call_id='call_1',
                output='See file result.png.',
                content=['This is file result.png:', BinaryContent(data=b'png', media_type='image/png')],
            )
        )
    assert ws.sent == []


# --- capabilities --------------------------------------------------------------------------------


@pytest.mark.usefixtures('no_genai_prices_context_window')
def test_profile() -> None:
    """xAI supports cancellation-based interruption but not output truncation, and no image input."""
    assert _model().profile == RealtimeModelProfile(
        supports_image_input=False,
        image_input_requires_response=False,
        supports_manual_turn_control=True,
        supports_interruption=True,
        supports_output_truncation=False,
        supports_text_output=False,  # Grok Voice always speaks
        supports_session_seeding=True,
        supports_webrtc=False,
        supports_seeding_images=False,
        supports_seeding_audio=False,
        supports_thinking=True,
        # Grok Voice answers the user while a tool call is outstanding (verified live).
        async_tool_call_mode='always',
        supports_async_tool_calls=True,  # deprecated, derived from `async_tool_call_mode`
        supports_tool_return_schema=False,
        emits_input_speech_events=True,
        synthesizes_turn_boundary=False,
        responses_are_requests=True,
        response_usage_covers_context=False,
        audio_input_sample_rate=24000,
        audio_output_sample_rate=24000,
        supported_native_tools=frozenset(),
        context_window=None,
    )


# --- session config: xAI's shape diverges from OpenAI's GA surface -------------------------------


def test_session_config_shape() -> None:
    """`xai_voice` maps to top-level `voice`, alongside `turn_detection`, in xAI's session shape."""
    model = _model(rt_xai.XaiRealtimeModelSettings(xai_voice='ara'))
    tools = [ToolDefinition(name='get_weather', description='Weather', parameters_json_schema={'type': 'object'})]
    config = model._session_config('Be nice', tools, model_settings=None)  # pyright: ignore[reportPrivateUsage]
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


def test_session_config_uses_profile_sample_rates() -> None:
    model = _model(profile=RealtimeModelProfile(audio_input_sample_rate=16000, audio_output_sample_rate=32000))

    config = model._session_config('', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]

    assert config['audio']['input']['format']['rate'] == 16000
    assert config['audio']['output']['format']['rate'] == 32000


def test_session_config_resumption_follows_reconnect_policy() -> None:
    assert 'resumption' not in _model()._session_config('hi', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]
    # A model-level default policy (via `settings=`) enables native resumption...
    model_level = _model(rt_xai.XaiRealtimeModelSettings(reconnect={}))
    assert model_level._session_config('hi', None, model_settings=None)['resumption'] == {'enabled': True}  # pyright: ignore[reportPrivateUsage]
    # ...and so does a per-session policy on a model with no defaults.
    per_session = rt_xai.XaiRealtimeModelSettings(reconnect={})
    assert _model()._session_config('hi', None, model_settings=per_session)['resumption'] == {'enabled': True}  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ('model_name', 'thinking', 'expected'),
    [
        ('grok-voice-latest', True, 'high'),
        ('grok-voice-think-fast-1.0', 'low', 'high'),
        ('grok-voice-think-fast-1.0', False, 'none'),
        # Shipped a week after 1.0 and becomes what `grok-voice-latest` resolves to on 2026-08-05.
        ('grok-voice-think-fast-2.0', 'high', 'high'),
    ],
)
def test_session_config_thinking(model_name: str, thinking: object, expected: str) -> None:
    model = _model(model=model_name)
    settings = rt_xai.XaiRealtimeModelSettings(thinking=thinking)  # type: ignore[typeddict-item]
    config = model._session_config('hi', None, model_settings=settings)  # pyright: ignore[reportPrivateUsage]
    assert config['reasoning'] == {'effort': expected}
    assert model.profile.get('supports_thinking') is True


def test_session_config_thinking_is_ignored_by_legacy_model() -> None:
    model = _model(model='grok-voice-fast-1.0')
    config = model._session_config(  # pyright: ignore[reportPrivateUsage]
        'hi', None, model_settings=rt_xai.XaiRealtimeModelSettings(thinking='high')
    )
    assert 'reasoning' not in config
    assert model.profile.get('supports_thinking') is False


def test_session_config_transcription_auto_by_default() -> None:
    """The default `input_transcription_model='auto'` resolves to xAI's recommended transcription model
    (`grok-transcribe`) → `audio.input.transcription.model`, so the user's audio turns are transcribed
    into history under the default `transcript_only` retention (they'd otherwise be dropped)."""
    config = _model()._session_config('hi', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]
    assert config['audio']['input']['transcription'] == {'model': 'grok-transcribe'}


def test_session_config_transcription_explicit_override() -> None:
    """An explicit model id is used verbatim, overriding the `'auto'` default."""
    config = _model()._session_config(  # pyright: ignore[reportPrivateUsage]
        'hi', None, model_settings=rt_xai.XaiRealtimeModelSettings(input_transcription_model='grok-transcribe-next')
    )
    assert config['audio']['input']['transcription'] == {'model': 'grok-transcribe-next'}


def test_session_config_transcription_disabled() -> None:
    """`input_transcription_model=None` opts out of transcription."""
    config = _model()._session_config(  # pyright: ignore[reportPrivateUsage]
        'hi', None, model_settings=rt_xai.XaiRealtimeModelSettings(input_transcription_model=None)
    )
    assert 'transcription' not in config['audio']['input']


def test_session_config_manual_turn_detection_is_null() -> None:
    """`turn_detection=False` disables VAD (push-to-talk), sent as an explicit null."""
    config = _model()._session_config(  # pyright: ignore[reportPrivateUsage]
        'hi', None, model_settings=rt_xai.XaiRealtimeModelSettings(turn_detection=False)
    )
    assert config['turn_detection'] is None


@pytest.mark.parametrize(('sensitivity', 'threshold'), [('low', 0.7), ('medium', 0.5), ('high', 0.3)])
def test_session_config_cross_provider_turn_detection_sensitivity(
    sensitivity: Literal['low', 'medium', 'high'], threshold: float
) -> None:
    config = _model()._session_config(  # pyright: ignore[reportPrivateUsage]
        'hi',
        None,
        model_settings=rt_xai.XaiRealtimeModelSettings(turn_detection={'sensitivity': sensitivity}),
    )
    assert config['turn_detection']['threshold'] == threshold


def test_session_config_xai_turn_detection_overrides_base() -> None:
    config = _model()._session_config(  # pyright: ignore[reportPrivateUsage]
        'hi',
        None,
        model_settings=rt_xai.XaiRealtimeModelSettings(
            turn_detection={'sensitivity': 'high'},
            xai_turn_detection={'type': 'server_vad', 'threshold': 0.9, 'create_response': False},
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
    assert 'voice' not in _model()._session_config('hi', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]


def test_session_config_forwards_model_settings() -> None:
    settings = rt_xai.XaiRealtimeModelSettings(max_tokens=256, parallel_tool_calls=False, tool_choice='none')
    model = _model(settings=settings)
    assert model.settings == settings
    tools = [ToolDefinition(name='get_weather', parameters_json_schema={'type': 'object'})]
    config = model._session_config('hi', tools, model_settings=settings)  # pyright: ignore[reportPrivateUsage]
    assert config['max_output_tokens'] == 256
    assert config['parallel_tool_calls'] is False
    assert config['tool_choice'] == 'none'
    assert 'tools' not in config


def test_session_config_rejects_forced_tool_choice() -> None:
    tools = [ToolDefinition(name='get_weather', parameters_json_schema={'type': 'object'})]
    with pytest.raises(UserError, match="A realtime session can't force a tool call"):
        _model()._session_config(  # pyright: ignore[reportPrivateUsage]
            'hi', tools, model_settings=rt_xai.XaiRealtimeModelSettings(tool_choice='required')
        )


def test_session_config_omits_absent_model_settings() -> None:
    """Absent realtime settings are omitted from the session config."""
    config = _model()._session_config('hi', None, model_settings=rt_xai.XaiRealtimeModelSettings())  # pyright: ignore[reportPrivateUsage]
    assert 'max_output_tokens' not in config
    assert 'parallel_tool_calls' not in config
    assert 'tool_choice' not in config


# --- connect: handshake, URL, auth, seeding ------------------------------------------------------


class FakeWebSocket:
    """A minimal stand-in for a `websockets` client connection.

    Running out of scripted frames stands in for the server closing the connection normally, which is
    how `websockets` reports a 1000/1001 close: iteration ends rather than raising.
    """

    close_code: int | None = 1000
    close_reason: str = ''

    def __init__(self, incoming: list[Any]) -> None:
        self._incoming = [self._normalize_frame(frame) for frame in incoming]
        self.sent: list[str] = []

    @staticmethod
    def _normalize_frame(frame: str) -> str:
        data = json.loads(frame)
        return json.dumps(sdk_frame(cast('dict[str, Any]', data))) if isinstance(data, dict) else frame

    async def recv(self) -> Any:
        return self._incoming.pop(0)

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def __aiter__(self) -> AsyncIterator[Any]:
        while self._incoming:
            yield self._incoming.pop(0)


def test_xai_connection_restores_in_flight_state_on_reconnect() -> None:
    # xAI resumes the conversation server-side, so the session keeps its in-flight state rather than
    # settling it (unlike the OpenAI base this connection is cloned from).
    conn = XaiRealtimeConnection(FakeWebSocket([]))  # type: ignore[arg-type]
    assert conn.reconnect_restores_in_flight_state is True


async def test_reconnect_does_not_re_solicit_an_unstarted_response() -> None:
    # xAI inherits the OpenAI `_attempt_reconnect`, but because it resumes in-flight state server-side
    # a response solicited before the drop is resumed by the server — re-soliciting it would duplicate
    # the turn, so the re-solicit is gated off for this connection.
    replacement = FakeWebSocket([])
    replacements = iter([replacement])

    async def dial() -> Any:
        try:
            return next(replacements)
        except StopIteration:
            raise OSError('server is down')

    conn = XaiRealtimeConnection(
        _DropAfterFrames([]),  # type: ignore[arg-type]
        dial=dial,
        reconnect={'base_delay': 0.0, 'max_attempts': 1},
    )
    conn._response_active = True  # pyright: ignore[reportPrivateUsage]
    conn._response_started = False  # pyright: ignore[reportPrivateUsage]

    events = [e async for e in conn]
    assert any(isinstance(e, RealtimeSessionReconnectEvent) for e in events)
    assert not any(json.loads(s).get('type') == 'response.create' for s in replacement.sent)


async def test_response_done_maps_xai_usage_extras() -> None:
    done = json.dumps(
        {
            'type': 'response.done',
            'response': {'id': 'resp-xai', 'status': 'completed', 'output': [], 'usage': None},
            'usage': {
                'input_tokens': 8,
                'output_tokens': 5,
                'input_token_details': {'audio_tokens': 6, 'grok_tokens': 2},
                'output_token_details': {'audio_tokens': 4, 'grok_tokens': 1},
                'billable_audio_seconds': 3,
                'output_audio_seconds': 2,
            },
        }
    )
    conn = XaiRealtimeConnection(FakeWebSocket([done]))  # type: ignore[arg-type]
    events = await collect_codec_events(conn)

    expected = RequestUsage(
        input_tokens=8,
        output_tokens=5,
        input_audio_tokens=6,
        output_audio_tokens=4,
        # Reported twice on purpose: in `details` under xAI's own name, and as the meter that prices it.
        audio_seconds=3,
        details={
            'audio_tokens': 4,
            'input_grok_tokens': 2,
            'output_grok_tokens': 1,
            'billable_audio_seconds': 3,
        },
    )
    assert events[0] == SessionUsage(
        usage=expected,
        provider_response_id='resp-xai',
        finish_reason='stop',
        provider_details={'status': 'completed'},
    )


async def test_response_done_without_billable_seconds_leaves_audio_seconds_unset() -> None:
    """A usage frame with no billed duration reports none, rather than guessing one from the tokens."""
    done = json.dumps(
        {
            'type': 'response.done',
            'response': {'id': 'resp-xai', 'status': 'completed', 'output': [], 'usage': None},
            'usage': {'input_tokens': 8, 'output_tokens': 5},
        }
    )
    conn = XaiRealtimeConnection(FakeWebSocket([done]))  # type: ignore[arg-type]
    events = await collect_codec_events(conn)

    usage_event = events[0]
    assert isinstance(usage_event, SessionUsage)
    assert usage_event.usage.audio_seconds == 0


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
        # Unreachable; it is what makes this an async generator.
        yield  # pragma: no cover


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
        try:
            ws = next(self._sockets)
        except StopIteration:
            raise OSError('server is down')  # no more sockets scripted: the server stays down
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


async def test_connect_captures_substituted_server_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # xAI accepts any model slug — even a retired or misspelled one — and silently substitutes its
    # current default, reporting the actually-served model only in `session.created`. Capturing it is
    # the only way a session's history can show what model really answered.
    created = json.dumps({'type': 'session.created', 'session': {'model': 'grok-voice-latest'}})
    ws = FakeWebSocket([created, _updated()])
    monkeypatch.setattr(rt_xai.websockets, 'connect', FakeConnect(ws))
    async with _connect(_model(model='grok-voice-retired-1.0'), 'x') as conn:
        assert conn.model_name == 'grok-voice-latest'


async def test_connect_handshake_url_auth_and_session_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The URL, bearer auth, and `session.update` frame are derived from the xAI provider."""
    # A cumulative `.updated` partial ahead of a real transcript proves the xAI codec is wired in:
    # the shared OpenAI codec has no mapping for that frame and would drop it.
    updated_partial = json.dumps(
        {'type': 'conversation.item.input_audio_transcription.updated', 'transcript': 'partial'}
    )
    transcript = json.dumps({'type': 'response.output_audio_transcript.done', 'transcript': 'hi'})
    ws = FakeWebSocket([_created(), _updated(), updated_partial, transcript])
    fake_connect = FakeConnect(ws)
    monkeypatch.setattr(rt_xai.websockets, 'connect', fake_connect)

    model = XaiRealtimeModel(
        'grok-voice-latest',
        provider=XaiProvider(api_key='k'),
        settings=rt_xai.XaiRealtimeModelSettings(xai_voice='eve'),
    )
    async with _connect(model, 'Be nice') as conn:
        assert isinstance(conn, XaiRealtimeConnection)
        events = await collect_codec_events(conn)

    assert events == [
        InputTranscript(text='partial', cumulative=True),
        OutputTranscript(text='hi', is_final=True, response_id='response'),
    ]
    assert fake_connect.url == 'wss://api.x.ai/v1/realtime?model=grok-voice-latest'
    assert fake_connect.headers == {'Authorization': 'Bearer k'}

    update = json.loads(ws.sent[0])
    assert update['type'] == 'session.update'
    assert update['session']['instructions'] == 'Be nice'
    assert update['session']['voice'] == 'eve'


async def test_connect_url_encodes_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeWebSocket([_created(), _updated()])
    fake_connect = FakeConnect(ws)
    monkeypatch.setattr(rt_xai.websockets, 'connect', fake_connect)

    async with _connect(_model(model='voice&conversation_id=stolen#fragment'), 'x'):
        pass

    assert fake_connect.url == ('wss://api.x.ai/v1/realtime?model=voice%26conversation_id%3Dstolen%23fragment')


async def test_connect_surfaces_handshake_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # xAI shares the OpenAI-protocol handshake, so a rejected config surfaces as a `ModelAPIError`
    # carrying the provider's message (not a raw protocol error), same as the OpenAI provider.
    error = json.dumps({'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'bad voice'}})
    ws = FakeWebSocket([_created(), error])
    monkeypatch.setattr(rt_xai.websockets, 'connect', FakeConnect(ws))
    model = XaiRealtimeModel('grok-voice-latest', provider=XaiProvider(api_key='k'))
    with pytest.raises(ModelAPIError, match='bad voice') as exc_info:
        async with _connect(model, 'x'):
            pass  # pragma: no cover
    assert exc_info.value.model_name == 'grok-voice-latest'


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


async def test_agent_realtime_session_rejects_native_tools() -> None:
    # xAI Grok Voice supports no native tools, so a native tool with no local fallback fails up front,
    # before dialing — via the same native ↔ local-tool swap the classic agent-run path applies, so the
    # error points at `local=`.
    agent: Agent[None, str] = Agent()
    with pytest.raises(
        UserError,
        match=r"not supported by this model.*WebSearch\(local='duckduckgo'\)",
    ):
        async with agent.realtime(_model(), capabilities=[NativeTool(WebSearchTool())]).session():
            pass  # pragma: no cover


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

    with pytest.raises(UserError, match='xai realtime sessions do not support images'):
        async with _connect(_model(), 'x', messages=history):
            pass  # pragma: no cover


async def test_connect_rejects_seeded_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_xai.websockets, 'connect', FakeConnect(ws))
    history = [
        ModelRequest(parts=[SpeechPart(speaker='user', audio=BinaryContent(data=b'audio', media_type='audio/pcm'))])
    ]

    with pytest.raises(UserError, match='xai realtime history seeding does not support retained user audio'):
        async with _connect(_model(), 'x', messages=history):
            pass  # pragma: no cover


async def test_connect_reconnect_closes_previous_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reconnect through `connect()`'s own dial closes the dropped socket before opening the next."""
    transcript = json.dumps({'type': 'response.output_audio_transcript.done', 'transcript': 'hi'})
    dropped = _DropAfterHandshake([_created(), _conversation_created(), _updated()])
    good = FakeWebSocket([_created(), _conversation_created(), _updated(), transcript])
    connect = _RecordingConnect([dropped, good])
    monkeypatch.setattr(rt_xai.websockets, 'connect', connect)

    model = _model(rt_xai.XaiRealtimeModelSettings(reconnect={'base_delay': 0.0, 'max_attempts': 1}))
    async with _connect(model, 'x') as conn:
        events = await collect_codec_events(conn)

    assert events == [
        RealtimeSessionReconnectEvent(state_restored=True),
        OutputTranscript(text='hi', is_final=True, response_id='response'),
    ]
    assert connect.closed == [dropped, good]  # both the dropped and the current socket are closed
    # The last URL is the re-dial attempted after `good` hung up, which the stand-in refuses.
    assert connect.urls == [
        'wss://api.x.ai/v1/realtime?model=grok-voice-latest',
        'wss://api.x.ai/v1/realtime?model=grok-voice-latest&conversation_id=conversation-1',
        'wss://api.x.ai/v1/realtime?model=grok-voice-latest&conversation_id=conversation-1',
    ]
    assert json.loads(dropped.sent[0])['session']['resumption'] == {'enabled': True}
    assert json.loads(good.sent[0])['session']['resumption'] == {'enabled': True}


async def test_max_duration_error_is_not_reconnected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The close after a `max_duration` error ends the session: a re-dial would resume into the same limit.

    The error already reports why the session ended, so the close isn't reported again.
    """
    ended = FakeWebSocket(
        [
            _created(),
            _conversation_created(),
            _updated(),
            json.dumps(_error_frame('max_duration', 'Maximum conversation duration exceeded.')),
        ]
    )
    connect = _RecordingConnect([ended, FakeWebSocket([_created(), _conversation_created(), _updated()])])
    monkeypatch.setattr(rt_xai.websockets, 'connect', connect)

    model = _model(rt_xai.XaiRealtimeModelSettings(reconnect={'base_delay': 0.0, 'max_attempts': 1}))
    async with _connect(model, 'x') as conn:
        events = [event async for event in conn]
        assert not conn._can_reconnect  # pyright: ignore[reportPrivateUsage]

    assert events == [
        RealtimeSessionErrorEvent(
            message='Maximum conversation duration exceeded.',
            type='max_duration',
            code='max_duration',
            recoverable=False,
        ),
    ]
    assert connect.urls == ['wss://api.x.ai/v1/realtime?model=grok-voice-latest']


@pytest.mark.shadow_divergence(
    reason=(
        'synthetic frames: the transcript and the terminal name different responses, and the input is '
        'acknowledged before it is sent'
    )
)
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
    model = _model(rt_xai.XaiRealtimeModelSettings(reconnect={'base_delay': 0.0, 'max_attempts': 1}))
    async with agent.realtime(model).session() as session:
        await session.send('Hello.')
        events = await collect_session_events(session)

    assert sum(isinstance(event, RealtimeSessionReconnectEvent) for event in events) == 1
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
        )
    ]


async def test_connect_reconnect_failure_leaves_nothing_to_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed reconnect through `connect()`'s dial leaves nothing to close on teardown.

    The dial nulls `cm` before re-dialing, so when the re-dial fails (an expected `OSError`) and the
    session ends via a `RealtimeSessionErrorEvent`, teardown finds `cm` already `None` and skips the close.
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
    model = _model(rt_xai.XaiRealtimeModelSettings(reconnect={'max_attempts': 1, 'base_delay': 0.0, 'jitter': False}))
    async with _connect(model, 'x') as conn:
        events = [e async for e in conn]

    # The message names xAI, not the OpenAI protocol whose connection class this reuses.
    fatal = [e for e in events if isinstance(e, RealtimeSessionErrorEvent) and not e.recoverable]
    assert [e.message for e in fatal] == [IsStr(regex=r'xAI Grok Voice connection closed; reconnect failed: .*')]
    # The dropped socket is closed as the reconnect nulls `cm` before re-dialing; the refused re-dial
    # never enters its context manager, so `cm` stays `None` and teardown closes nothing further. A
    # regression that assigned `cm` before awaiting `__aenter__` would leave `'refused'` here.
    assert connect.closed == ['dropped']


async def test_reconnect_handshake_error_is_retryable() -> None:
    async def dial() -> rt_xai.ClientConnection:
        raise rt_xai.RealtimeHandshakeError('expired conversation')

    conn = XaiRealtimeConnection(FakeWebSocket([]), dial=dial)  # type: ignore[arg-type]

    assert await conn._attempt_reconnect() is False  # pyright: ignore[reportPrivateUsage]


async def test_connect_open_failure_propagates_without_teardown(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the very first connection fails to open, there is nothing to close on teardown."""

    class _FailingConnect:
        def __call__(self, url: str, *, additional_headers: dict[str, str] | None = None) -> Any:
            return self

        async def __aenter__(self) -> Any:
            raise ConnectionError('refused')

        async def __aexit__(self, *exc: object) -> bool:  # pragma: no cover
            return False

    monkeypatch.setattr(rt_xai.websockets, 'connect', _FailingConnect())
    with pytest.raises(ModelAPIError, match='Could not reach the realtime API: refused'):
        async with _connect(_model(), 'x'):
            pass  # pragma: no cover


async def test_connect_rejects_conversation_created_without_id(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeWebSocket([_created(), json.dumps({'type': 'conversation.created', 'conversation': {}})])
    monkeypatch.setattr(rt_xai.websockets, 'connect', FakeConnect(ws))

    with pytest.raises(RuntimeError, match=r'did not include a `conversation\.id`'):
        async with _connect(_model(rt_xai.XaiRealtimeModelSettings(reconnect={})), 'x'):
            pass  # pragma: no cover


# --- provider / auth resolution ------------------------------------------------------------------


async def test_provider_str_resolves_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default `provider='xai'` reads `XAI_API_KEY`, which becomes the WebSocket bearer token."""
    monkeypatch.setenv('XAI_API_KEY', 'env-key')
    ws = FakeWebSocket([_created(), _updated()])
    fake_connect = FakeConnect(ws)
    monkeypatch.setattr(rt_xai.websockets, 'connect', fake_connect)

    model = XaiRealtimeModel('grok-voice-latest')
    assert model.model_name == 'grok-voice-latest'
    async with _connect(model, 'hi'):
        pass
    assert fake_connect.headers == {'Authorization': 'Bearer env-key'}


def test_non_xai_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # A foreign provider (instance or `provider='...'` string) must fail fast with a clear `UserError`,
    # rather than an `AttributeError` about an xAI-only field the user never set — matching how
    # `AzureRealtimeModel` rejects a non-Azure provider.
    monkeypatch.setenv('OPENAI_API_KEY', 'test')
    with pytest.raises(UserError, match='requires an `XaiProvider`'):
        XaiRealtimeModel('grok-voice-latest', provider='openai')
    with pytest.raises(UserError, match='requires an `XaiProvider`'):
        # Deliberately a foreign provider, so the cast is what lets the runtime guard be exercised.
        XaiRealtimeModel('grok-voice-latest', provider=cast('Any', OpenAIProvider(api_key='x')))


async def test_reconnect_reports_the_newly_served_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # xAI substitutes its current default for any slug, so a re-dial can land on a different model than
    # the one that started the session. `model_name` reads the latest `session.created` through the
    # dial closure; snapshotting it at construction would keep reporting the original forever.
    def _session_created(model_name: str) -> str:
        return json.dumps({'type': 'session.created', 'session': {'model': model_name}})

    connects = iter(
        [
            FakeConnect(FakeWebSocket([_session_created('grok-voice-1.0'), _updated()])),
            FakeConnect(FakeWebSocket([_session_created('grok-voice-2.0'), _updated()])),
        ]
    )

    def connect(url: str, *, additional_headers: dict[str, str] | None = None) -> FakeConnect:
        return next(connects)(url, additional_headers=additional_headers)

    monkeypatch.setattr(rt_xai.websockets, 'connect', connect)

    async with _connect(_model(), 'x') as conn:
        assert conn.model_name == 'grok-voice-1.0'
        assert await conn._attempt_reconnect() is True  # pyright: ignore[reportPrivateUsage]
        assert conn.model_name == 'grok-voice-2.0'


def test_provider_from_xai_client_without_exposed_key_raises() -> None:
    """A provider built from a pre-configured `xai_client` can't expose its key, so realtime errors clearly."""
    provider = XaiProvider(xai_client=AsyncClient(api_key='hidden'))
    with pytest.raises(UserError, match='pre-configured `xai_client`'):
        XaiRealtimeModel('grok-voice-latest', provider=provider)


def _done_with_billed_seconds(total: int) -> str:
    return json.dumps(
        {
            'type': 'response.done',
            'response': {'id': f'resp-{total}', 'status': 'completed', 'output': [], 'usage': None},
            'usage': {'input_tokens': 3, 'output_tokens': 40, 'billable_audio_seconds': total},
        }
    )


async def test_billable_audio_seconds_running_total_is_split_per_response() -> None:
    """xAI reports the session's running total; each response is credited only its own increase.

    Recorded live against `grok-voice-latest`: three turns of 0.71s, 0.71s and 0.87s of audio reported
    `billable_audio_seconds` of 1, 2 and 3. Summing those as-is would bill 6 seconds for a 3-second
    session, and the overcount grows with every turn.
    """
    frames = [_done_with_billed_seconds(total) for total in (1, 2, 3)]
    conn = XaiRealtimeConnection(FakeWebSocket(frames))  # type: ignore[arg-type]
    events = await collect_codec_events(conn)

    usages = [event.usage for event in events if isinstance(event, SessionUsage)]
    assert [usage.audio_seconds for usage in usages] == [1, 1, 1]
    assert [usage.details['billable_audio_seconds'] for usage in usages] == [1, 1, 1]


async def test_billable_audio_seconds_lower_total_in_the_same_conversation_adds_nothing() -> None:
    """xAI's total only grows within a conversation, so a lower one is not new usage to bill again."""
    frames = [_done_with_billed_seconds(total) for total in (2, 4, 2, 5)]
    conn = XaiRealtimeConnection(FakeWebSocket(frames), conversation_id='conv-1')  # type: ignore[arg-type]
    events = await collect_codec_events(conn)

    usages = [event.usage for event in events if isinstance(event, SessionUsage)]
    assert [usage.audio_seconds for usage in usages] == [2, 2, 0, 1]


async def test_billable_audio_seconds_restart_with_a_new_conversation() -> None:
    """A new conversation is the one place xAI's count starts again from zero."""
    conn = XaiRealtimeConnection(FakeWebSocket([]), conversation_id='conv-1')  # type: ignore[arg-type]
    usage = RealtimeResponseUsage.model_validate({'input_tokens': 1, 'output_tokens': 1, 'billable_audio_seconds': 4})
    first = conn._map_response_usage(usage)  # pyright: ignore[reportPrivateUsage]

    conn.conversation_id = 'conv-2'
    restarted = RealtimeResponseUsage.model_validate(
        {'input_tokens': 1, 'output_tokens': 1, 'billable_audio_seconds': 1}
    )
    second = conn._map_response_usage(restarted)  # pyright: ignore[reportPrivateUsage]

    assert first is not None and second is not None
    assert (first.audio_seconds, second.audio_seconds) == (4, 1)


@pytest.mark.anyio
async def test_reconnect_keeps_a_tool_call_batch_the_server_restores() -> None:
    """xAI resumes the response and its tool calls on reconnect, so an early result still gets its answer."""
    call = {
        'id': 'item-c1',
        'type': 'function_call',
        'call_id': 'c1',
        'name': 'w',
        'arguments': '{}',
        'status': 'completed',
    }
    first = _DropAfterFrames(
        [
            json.dumps({'type': 'response.created', 'response': {'id': 'r1', 'status': 'in_progress', 'output': []}}),
            json.dumps(
                {
                    'type': 'response.function_call_arguments.done',
                    'response_id': 'r1',
                    'item_id': 'item-c1',
                    'output_index': 0,
                    'call_id': 'c1',
                    'name': 'w',
                    'arguments': '{}',
                }
            ),
        ]
    )
    replacement = FakeWebSocket(
        [json.dumps({'type': 'response.done', 'response': {'id': 'r1', 'status': 'completed', 'output': [call]}})]
    )
    replacements = iter([replacement])

    async def dial() -> Any:
        try:
            return next(replacements)
        except StopIteration:
            raise OSError('server is down')

    conn = XaiRealtimeConnection(first, dial=dial, reconnect={'base_delay': 0.0, 'max_attempts': 1})  # type: ignore[arg-type]
    events = conn.__aiter__()
    await events.__anext__()  # the tool call
    await conn.send(ToolResult(tool_call_id='c1', output='sunny'))  # out before r1 is done
    _ = [event async for event in events]

    assert [json.loads(frame)['type'] for frame in replacement.sent] == ['response.create']


def _response_frame(event_type: str, response_id: str = 'r1', status: str | None = None) -> str:
    status = status or ('in_progress' if event_type == 'response.created' else 'completed')
    return json.dumps({'type': event_type, 'response': {'id': response_id, 'status': status, 'output': []}})


def _transcript_delta(response_id: str = 'r1') -> str:
    return json.dumps(
        {
            'type': 'response.output_audio_transcript.delta',
            'response_id': response_id,
            'item_id': 'item-a1',
            'output_index': 0,
            'content_index': 0,
            'delta': 'Four.',
        }
    )


def _sent(ws: FakeWebSocket) -> list[dict[str, Any]]:
    return [json.loads(frame) for frame in ws.sent]


def _sent_types(ws: FakeWebSocket) -> list[str]:
    return [frame['type'] for frame in _sent(ws)]


def _manual(ws: FakeWebSocket, **kwargs: Any) -> XaiRealtimeConnection:
    return XaiRealtimeConnection(ws, manual_turns=True, **kwargs)  # type: ignore[arg-type]


# Stands for speech: xAI answers a request with speech in the buffer, but drops one with only silence.
_AUDIO = BinaryAudio(data=b'\x10\x27' * 240, media_type='audio/pcm')
_OTHER_AUDIO = BinaryAudio(data=b'\x20\x4e' * 240, media_type='audio/pcm')


def _appended(ws: FakeWebSocket) -> list[bytes]:
    return [base64.b64decode(frame['audio']) for frame in _sent(ws) if frame['type'] == 'input_audio_buffer.append']


async def test_commit_is_held_until_a_response_is_asked_for() -> None:
    """With turn detection off, the commit goes out only with `create_response()`, just ahead of `response.create`.

    xAI answers a commit by itself, so a commit sent at once would reply whether or not one was asked for.
    """
    ws = FakeWebSocket([_response_frame('response.created'), _response_frame('response.done')])
    conn = _manual(ws)
    await conn.send(_AUDIO)
    await conn.send(CommitAudio())
    assert _sent_types(ws) == ['input_audio_buffer.append']

    await conn.send(TextContext('The user is called Ada.'))
    await conn.send(CreateResponse())
    # Nothing new: the commit went out after the text, so the audio is what xAI last answered, and the
    # request clears the empty buffer to be answered again.
    await conn.send(CreateResponse())
    _ = [event async for event in conn]
    assert _sent(ws)[1:] == [
        {
            'type': 'conversation.item.create',
            'event_id': 'pydantic_ai.content.2',
            'item': {
                'id': 'pydantic_ai_item_2',
                'type': 'message',
                'role': 'user',
                'content': [{'type': 'input_text', 'text': 'The user is called Ada.'}],
            },
        },
        {'type': 'input_audio_buffer.commit'},
        {
            'type': 'response.create',
            'event_id': 'pydantic_ai.response.3',
            'response': {'metadata': {'pydantic_ai_inputs': '3'}},
        },
        {'type': 'input_audio_buffer.clear'},
        {
            'type': 'response.create',
            'event_id': 'pydantic_ai.response.4',
            'response': {'metadata': {'pydantic_ai_inputs': '4'}},
        },
    ]


async def test_commit_is_sent_at_once_with_turn_detection_on() -> None:
    """With server VAD nothing is held back: a commit and a request go out as they are made."""
    ws = FakeWebSocket([])
    conn = XaiRealtimeConnection(ws)  # type: ignore[arg-type]
    await conn.send(_AUDIO)
    await conn.send(CommitAudio())
    await conn.send(CreateResponse())
    await conn.send(ClearAudio())
    assert _sent_types(ws) == [
        'input_audio_buffer.append',
        'input_audio_buffer.commit',
        'response.create',
        'input_audio_buffer.clear',
    ]


async def test_text_turn_sends_the_held_commit() -> None:
    """A text turn asks for a response, so the held commit goes out with it and its reply answers both."""
    ws = FakeWebSocket([])
    conn = _manual(ws)
    await conn.send(_AUDIO)
    await conn.send(CommitAudio())
    await conn.send('And answer in French.')
    assert _sent_types(ws) == [
        'input_audio_buffer.append',
        'conversation.item.create',
        'input_audio_buffer.commit',
        'response.create',
    ]


async def test_audio_after_a_held_commit_is_kept_back() -> None:
    """Audio after a held commit stays local: a second commit adds it to the same turn, and a clear drops it."""
    ws = FakeWebSocket([])
    conn = _manual(ws)
    await conn.send(_AUDIO)
    await conn.send(CommitAudio())
    await conn.send(_OTHER_AUDIO)
    await conn.send(CommitAudio())
    await conn.send(_AUDIO)
    await conn.send(ClearAudio())
    assert _sent_types(ws) == ['input_audio_buffer.append']

    await conn.send(CreateResponse())
    assert _sent_types(ws) == [
        'input_audio_buffer.append',
        'input_audio_buffer.append',
        'input_audio_buffer.commit',
        'response.create',
    ]
    assert _appended(ws) == [_AUDIO.data, _OTHER_AUDIO.data]


async def test_commit_with_nothing_buffered_is_sent() -> None:
    """A commit with no audio behind it is passed on, as before, while a clear with none isn't sent.

    A clear right after a commit cancels the commit's reply (checked live), and has nothing to discard.
    """
    ws = FakeWebSocket([])
    conn = _manual(ws)
    await conn.send(CommitAudio())
    await conn.send(ClearAudio())
    assert _sent_types(ws) == ['input_audio_buffer.commit']


async def test_audio_after_the_commit_waits_for_its_reply_to_end() -> None:
    """Audio kept back when the commit goes out is sent once its reply ends.

    Speech reaching xAI during a reply stops the reply without ending it (checked live).
    """
    ws = FakeWebSocket([_response_frame('response.created'), _transcript_delta(), _response_frame('response.done')])
    conn = _manual(ws)
    await conn.send(_AUDIO)
    await conn.send(CommitAudio())
    await conn.send(_OTHER_AUDIO)
    await conn.send(CreateResponse())
    events = conn.__aiter__()
    assert isinstance(await events.__anext__(), OutputTranscript)
    await conn.send(_AUDIO)
    assert _sent_types(ws) == ['input_audio_buffer.append', 'input_audio_buffer.commit', 'response.create']

    rest = [event async for event in events]
    assert isinstance(rest[0], ResponseDone)
    assert _appended(ws) == [_AUDIO.data, _OTHER_AUDIO.data, _AUDIO.data]


async def test_request_behind_a_reply_sends_the_audio_kept_back_first() -> None:
    """A request held behind the reply goes out after the audio kept back, which it then commits."""
    ws = FakeWebSocket([_response_frame('response.created'), _transcript_delta(), _response_frame('response.done')])
    conn = _manual(ws)
    await conn.send(_AUDIO)
    await conn.send(CommitAudio())
    await conn.send(_OTHER_AUDIO)
    await conn.send(CreateResponse())
    events = conn.__aiter__()
    assert isinstance(await events.__anext__(), OutputTranscript)
    await conn.send(CreateResponse())
    rest = [event async for event in events]

    assert not any(isinstance(event, InputRejected) for event in rest)
    assert _sent_types(ws)[-2:] == ['input_audio_buffer.append', 'response.create']


async def test_held_commit_waits_for_the_reply_in_flight() -> None:
    """Audio, a commit and a request made during a reply all wait for that reply to end.

    Speech reaching xAI during a reply stops the reply without ending it (checked live).
    """
    ws = FakeWebSocket(
        [_response_frame('response.created', 'r0'), _transcript_delta('r0'), _response_frame('response.done', 'r0')]
    )
    conn = _manual(ws)
    events = conn.__aiter__()
    assert isinstance(await events.__anext__(), OutputTranscript)
    await conn.send(_AUDIO)
    await conn.send(CommitAudio())
    await conn.send(CreateResponse())
    assert _sent_types(ws) == []
    _ = [event async for event in events]
    assert _sent_types(ws) == ['input_audio_buffer.append', 'input_audio_buffer.commit', 'response.create']


async def test_create_response_during_the_reply_is_answered_again_when_it_ends() -> None:
    """A request with nothing new behind it, made during the reply, is answered again once that reply ends.

    xAI would silently drop a bare `response.create` then, since the reply already answered the committed
    audio, so the request clears the empty buffer first (checked live), as a request on OpenAI is answered.
    """
    ws = FakeWebSocket([_response_frame('response.created'), _transcript_delta(), _response_frame('response.done')])
    conn = _manual(ws)
    await conn.send(_AUDIO)
    await conn.send(CommitAudio())
    await conn.send(CreateResponse())
    events = conn.__aiter__()
    assert isinstance(await events.__anext__(), OutputTranscript)
    await conn.send(CreateResponse())
    assert _sent_types(ws) == ['input_audio_buffer.append', 'input_audio_buffer.commit', 'response.create']
    _ = [event async for event in events]

    assert _sent(ws)[3:] == [
        {'type': 'input_audio_buffer.clear'},
        {
            'type': 'response.create',
            'event_id': 'pydantic_ai.response.3',
            'response': {'metadata': {'pydantic_ai_inputs': '3'}},
        },
    ]


_CLEAR = 'input_audio_buffer.clear'
_CREATE = 'response.create'


@pytest.mark.parametrize(
    ('between', 'expected'),
    [
        pytest.param([], [_CLEAR, _CREATE], id='nothing'),
        pytest.param([CancelResponse()], [_CLEAR, _CREATE], id='cancel'),
        pytest.param([CommitAudio()], ['input_audio_buffer.commit', _CLEAR, _CREATE], id='empty-commit'),
        pytest.param([ClearAudio()], [_CLEAR, _CREATE], id='clear-with-nothing-buffered'),
        pytest.param([TextContext('And three plus three?')], ['conversation.item.create', _CREATE], id='text'),
        pytest.param([_AUDIO, ClearAudio()], ['input_audio_buffer.append', _CLEAR, _CREATE], id='clear'),
        pytest.param([_AUDIO], ['input_audio_buffer.append', _CREATE], id='uncommitted-audio'),
    ],
)
async def test_create_response_after_the_reply(between: list[RealtimeInput], expected: list[str]) -> None:
    """After xAI has answered committed audio, a request is always answered, as on other providers.

    Only new input makes xAI answer a bare `response.create` (checked live). Without it, the request clears
    the empty buffer first, which makes xAI answer again.
    """
    ws = FakeWebSocket([_response_frame('response.created'), _response_frame('response.done')])
    conn = _manual(ws)
    await conn.send(_AUDIO)
    await conn.send(CommitAudio())
    await conn.send(CreateResponse())
    _ = [event async for event in conn]
    for content in between:
        await conn.send(content)
    await conn.send(CreateResponse())

    assert _sent_types(ws)[3:] == expected


async def test_audio_that_fails_to_send_is_not_counted_as_buffered() -> None:
    """Audio `send()` rejects never reached the buffer, so a commit after it has nothing to hold."""
    ws = FakeWebSocket([])
    conn = _manual(ws)
    with pytest.raises(UserError, match='require raw PCM audio'):
        await conn.send(BinaryAudio(data=b'RIFF', media_type='audio/wav'))
    await conn.send(CommitAudio())
    assert _sent_types(ws) == ['input_audio_buffer.commit']


async def test_clear_before_the_reply_starts_is_not_sent() -> None:
    """A clear right after the commit went out would cancel its reply (checked live), and has nothing to discard."""
    ws = FakeWebSocket([])
    conn = _manual(ws)
    await conn.send(_AUDIO)
    await conn.send(CommitAudio())
    await conn.send(CreateResponse())
    await conn.send(ClearAudio())
    assert _sent_types(ws) == ['input_audio_buffer.append', 'input_audio_buffer.commit', 'response.create']


async def test_reconnect_sends_the_audio_a_held_commit_covers_again() -> None:
    """A dropped connection takes the buffered audio with it, so a held commit sends that audio again."""
    replacement = FakeWebSocket([])
    replacements = iter([replacement])

    async def dial() -> Any:
        try:
            return next(replacements)
        except StopIteration:
            raise OSError('server is down')

    conn = _manual(_DropAfterFrames([]), dial=dial, reconnect={'base_delay': 0.0, 'max_attempts': 1})
    await conn.send(_AUDIO)
    await conn.send(CommitAudio())
    await conn.send(_OTHER_AUDIO)
    events = [event async for event in conn]
    assert any(isinstance(event, RealtimeSessionReconnectEvent) for event in events)
    await conn.send(CreateResponse())

    assert _sent_types(replacement) == [
        'input_audio_buffer.append',
        'input_audio_buffer.commit',
        'response.create',
    ]
    assert _appended(replacement) == [_AUDIO.data]


async def test_commit_of_speech_is_sent_without_a_request() -> None:
    """When xAI has heard speech, the commit alone asks for the reply.

    A `response.create` sent with it would be answered before the speech joins the conversation (checked
    live). Speech reported after the commit went out belongs to that commit, not to the next turn.
    """
    ws = FakeWebSocket(
        [
            json.dumps({'type': 'input_audio_buffer.speech_started', 'item_id': 'item-u1', 'audio_start_ms': 0}),
            json.dumps({'type': 'input_audio_buffer.committed', 'item_id': 'item-u1'}),
            _response_frame('response.created'),
            _response_frame('response.done'),
            json.dumps({'type': 'input_audio_buffer.speech_started', 'item_id': 'item-u2', 'audio_start_ms': 0}),
            json.dumps({'type': 'input_audio_buffer.committed', 'item_id': 'item-u2'}),
        ]
    )
    conn = _manual(ws)
    await conn.send(_AUDIO)
    events = conn.__aiter__()
    await events.__anext__()  # xAI heard speech
    await conn.send(CommitAudio())
    await conn.send(CreateResponse())
    assert _sent_types(ws) == ['input_audio_buffer.append', 'input_audio_buffer.commit']

    rest = [event async for event in events]
    assert not any(isinstance(event, InputRejected) for event in rest)
    # The commit's reply answered the request, so the connection is free for the next one.
    await conn.send(_AUDIO)
    await conn.send(CommitAudio())
    await conn.send(CreateResponse())
    assert _sent_types(ws)[2:] == ['input_audio_buffer.append', 'input_audio_buffer.commit', 'response.create']


async def test_refused_commit_of_speech_releases_the_request() -> None:
    """A commit sent in place of `response.create` carries its id, so an error echoing it refuses the request."""
    ws = FakeWebSocket(
        [
            json.dumps({'type': 'input_audio_buffer.speech_started', 'item_id': 'item-u1', 'audio_start_ms': 0}),
            json.dumps(
                {
                    'type': 'error',
                    'error': {'type': 'invalid_request_error', 'message': 'no', 'event_id': 'pydantic_ai.response.2'},
                }
            ),
        ]
    )
    conn = _manual(ws)
    await conn.send(_AUDIO)
    events = conn.__aiter__()
    await events.__anext__()  # xAI heard speech
    await conn.send(CommitAudio())
    await conn.send(CreateResponse())
    assert _sent(ws)[-1] == {'type': 'input_audio_buffer.commit', 'event_id': 'pydantic_ai.response.2'}
    rest = [event async for event in events]
    assert InputRejected(2, refused='response') in rest


class _FailingSend(FakeWebSocket):
    """Fails every send after the first `sends` ones, as a connection dropping mid-send would."""

    def __init__(self, incoming: list[Any], *, sends: int) -> None:
        super().__init__(incoming)
        self._sends = sends

    async def send(self, data: str) -> None:
        if len(self.sent) >= self._sends:
            raise OSError('connection reset')
        await super().send(data)


async def test_reconnect_during_the_held_commit_sends_the_whole_turn_again() -> None:
    """A connection dropping while the held commit goes out keeps the turn, which is sent again after the reconnect."""
    replacement = FakeWebSocket([])
    replacements = iter([replacement])

    async def dial() -> Any:
        try:
            return next(replacements)
        except StopIteration:
            raise OSError('server is down')

    ws = _FailingSend([], sends=2)
    conn = _manual(ws, dial=dial, reconnect={'base_delay': 0.0, 'max_attempts': 1})
    commits_announced: list[None] = []
    conn._set_audio_commit_listener(lambda: commits_announced.append(None))  # pyright: ignore[reportPrivateUsage]
    await conn.send(_AUDIO)
    await conn.send(CommitAudio())
    await conn.send(_OTHER_AUDIO)
    await conn.send(CommitAudio())
    with pytest.raises(OSError):
        await conn.send(CreateResponse())
    events = [event async for event in conn]
    assert any(isinstance(event, RealtimeSessionReconnectEvent) for event in events)
    await conn.send(CreateResponse())

    assert _sent_types(replacement) == [
        'input_audio_buffer.append',
        'input_audio_buffer.append',
        'input_audio_buffer.commit',
        'response.create',
    ]
    assert _appended(replacement) == [_AUDIO.data, _OTHER_AUDIO.data]
    # The session placed the turn when the commit first started going out; sending it again doesn't move it.
    assert len(commits_announced) == 1


async def test_audio_kept_back_during_a_reply_is_the_latest_input_once_sent() -> None:
    """Audio kept back behind a reply lands after text sent meanwhile, so once a request commits it, a bare
    request after that clears the empty buffer rather than going out alone and being dropped."""
    ws = FakeWebSocket(
        [
            _response_frame('response.created', 'r0'),
            _transcript_delta('r0'),
            _response_frame('response.done', 'r0'),
            _response_frame('response.created'),
            _response_frame('response.done'),
        ]
    )
    conn = _manual(ws)
    events = conn.__aiter__()
    assert isinstance(await events.__anext__(), OutputTranscript)
    await conn.send(_AUDIO)
    await conn.send(TextContext('Some context.'))
    assert isinstance(await events.__anext__(), ResponseDone)
    await conn.send(CreateResponse())  # commits the audio sent once the reply ended
    assert isinstance(await events.__anext__(), ResponseDone)
    await conn.send(CreateResponse())

    assert _sent_types(ws) == [
        'conversation.item.create',
        'input_audio_buffer.append',
        'response.create',
        'input_audio_buffer.clear',
        'response.create',
    ]


class _PhasedWebSocket(FakeWebSocket):
    """A socket whose frames after the handshake arrive in phases, each once the test lets it through.

    The first arrives once the session sent something past the handshake, as an answer would.
    """

    def __init__(self, handshake: list[str], *phases: list[str]) -> None:
        super().__init__(handshake)
        self._phases = [[self._normalize_frame(frame) for frame in phase] for phase in phases]
        self._gate = asyncio.Event()
        self._handshake_sent = 0

    async def recv(self) -> Any:
        frame = await super().recv()
        # What the session sent by the time it read the last handshake frame is the handshake's.
        self._handshake_sent = len(self.sent)
        return frame

    def advance(self) -> None:
        self._gate.set()

    async def __aiter__(self) -> AsyncIterator[Any]:
        for index, phase in enumerate(self._phases):
            if index:
                await self._gate.wait()
                self._gate.clear()
            else:
                while len(self.sent) == self._handshake_sent:
                    await asyncio.sleep(0)
            for frame in phase:
                yield frame
        await asyncio.Event().wait()


@pytest.mark.parametrize('transcription', [True, False], ids=['transcribed', 'untranscribed'])
async def test_held_spoken_turn_is_recorded_after_what_was_sent_first(
    monkeypatch: pytest.MonkeyPatch, transcription: bool
) -> None:
    """History follows the order xAI has the turns in: a spoken turn whose commit was held goes after the text
    sent before its release, not where its audio started.

    The text is sent while the commit is held, so it reaches xAI first, and the spoken turn only joins the
    conversation when the text turn's request releases the commit.
    """
    transcript = [
        json.dumps(
            {
                'type': 'conversation.item.input_audio_transcription.completed',
                'item_id': 'item-u1',
                'content_index': 0,
                'transcript': 'My name is Ada.',
            }
        )
    ]
    ws = _PhasedWebSocket(
        [_created(), _updated()],
        [json.dumps({'type': 'input_audio_buffer.speech_started', 'item_id': 'item-u1', 'audio_start_ms': 0})],
        [
            json.dumps({'type': 'input_audio_buffer.committed', 'item_id': 'item-u1'}),
            *(transcript if transcription else []),
            _response_frame('response.created'),
            json.dumps(
                {
                    'type': 'response.output_audio_transcript.done',
                    'response_id': 'r1',
                    'item_id': 'item-a1',
                    'output_index': 0,
                    'content_index': 0,
                    'transcript': 'Hello Ada.',
                }
            ),
            _response_frame('response.done'),
        ],
    )
    monkeypatch.setattr(rt_xai.websockets, 'connect', _RecordingConnect([ws]))
    settings = rt_xai.XaiRealtimeModelSettings(turn_detection=False)
    if not transcription:
        settings['input_transcription_model'] = None

    async with Agent().realtime(_model(settings)).session() as session:
        await session.send_audio(_AUDIO.data)
        async for event in session:  # pragma: no branch
            if isinstance(event, RealtimeInputSpeechStartEvent):  # pragma: no branch
                break
        await session.commit_audio()
        await session.send('Greet me by name.')
        ws.advance()
        await session.wait_for_reply()

    assert _sent_types(ws)[-3:] == [
        'input_audio_buffer.append',
        'conversation.item.create',
        'input_audio_buffer.commit',
    ]
    messages = session.all_messages()
    assert [[type(part).__name__ for part in message.parts] for message in messages] == [
        ['UserPromptPart'],
        ['SpeechPart'],
        ['SpeechPart'],
    ]
    assert isinstance(messages[1], ModelRequest)
    assert isinstance(messages[2], ModelResponse)


def _turns(messages: Sequence[ModelMessage]) -> list[str]:
    """Each message as who said what, to read history order at a glance."""
    turns: list[str] = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                turns.append(f'user: {part.content}')
            else:
                assert isinstance(part, SpeechPart)
                turns.append(f'{part.speaker} speech: {part.transcript}')
    return turns


def _user_transcript(item_id: str, transcript: str) -> str:
    return json.dumps(
        {
            'type': 'conversation.item.input_audio_transcription.completed',
            'item_id': item_id,
            'content_index': 0,
            'transcript': transcript,
        }
    )


def _reply(response_id: str, transcript: str) -> list[str]:
    return [
        _response_frame('response.created', response_id),
        json.dumps(
            {
                'type': 'response.output_audio_transcript.done',
                'response_id': response_id,
                'item_id': f'item-{response_id}',
                'output_index': 0,
                'content_index': 0,
                'transcript': transcript,
            }
        ),
        _response_frame('response.done', response_id),
    ]


async def test_late_transcript_keeps_an_earlier_turn_in_its_place(monkeypatch: pytest.MonkeyPatch) -> None:
    """A turn whose commit went out earlier keeps its place when its transcript arrives after the next commit is held."""
    ws = _PhasedWebSocket(
        [_created(), _updated()],
        [],
        [
            json.dumps({'type': 'input_audio_buffer.committed', 'item_id': 'item-u1'}),
            json.dumps(
                {
                    'type': 'conversation.item.input_audio_transcription.delta',
                    'item_id': 'item-u1',
                    'content_index': 0,
                    'delta': 'First',
                }
            ),
            *_reply('r1', 'Answer one.'),
        ],
        [
            _user_transcript('item-u1', 'First turn.'),
            json.dumps({'type': 'input_audio_buffer.committed', 'item_id': 'item-u2'}),
            _user_transcript('item-u2', 'Second turn.'),
            *_reply('r2', 'Answer two.'),
        ],
    )
    monkeypatch.setattr(rt_xai.websockets, 'connect', _RecordingConnect([ws]))

    async with Agent().realtime(_model(rt_xai.XaiRealtimeModelSettings(turn_detection=False))).session() as session:
        await session.send_audio(_AUDIO.data)
        await session.commit_audio()
        await session.create_response()
        ws.advance()
        await session.wait_for_reply()
        await session.send_audio(_AUDIO.data)
        await session.commit_audio()
        await session.send('And this.')
        ws.advance()
        await session.wait_for_reply()

    assert _turns(session.all_messages()) == [
        'user speech: First turn.',
        'assistant speech: Answer one.',
        'user: And this.',
        'user speech: Second turn.',
        'assistant speech: Answer two.',
    ]


def _spoken_reply(response_id: str, pcm: bytes, transcript: str) -> list[str]:
    return [
        _response_frame('response.created', response_id),
        json.dumps(
            {
                'type': 'response.output_audio.delta',
                'response_id': response_id,
                'item_id': f'item-{response_id}',
                'output_index': 0,
                'content_index': 0,
                'delta': base64.b64encode(pcm).decode('ascii'),
            }
        ),
        *_reply(response_id, transcript)[1:],
    ]


@pytest.mark.shadow_divergence('the new session core does not apply `retain_audio_max_seconds` yet')
async def test_retained_audio_eviction_keeps_a_committed_turn_in_its_place(monkeypatch: pytest.MonkeyPatch) -> None:
    """A spoken turn goes after what its commit followed, even once the retained-audio budget evicted that answer's audio.

    The second answer arrives before the second turn's transcript and evicts the first answer's audio, which the
    second commit was placed after.
    """
    tenth_of_a_second = b'\x10\x27' * 2400
    ws = _PhasedWebSocket(
        [_created(), _updated()],
        [],
        [
            json.dumps({'type': 'input_audio_buffer.committed', 'item_id': 'item-u1'}),
            _user_transcript('item-u1', 'First turn.'),
            *_spoken_reply('r1', tenth_of_a_second, 'Answer one.'),
        ],
        [
            json.dumps({'type': 'input_audio_buffer.committed', 'item_id': 'item-u2'}),
            *_spoken_reply('r2', tenth_of_a_second, 'Answer two.'),
            _user_transcript('item-u2', 'Second turn.'),
        ],
    )
    monkeypatch.setattr(rt_xai.websockets, 'connect', _RecordingConnect([ws]))

    async with (
        Agent()
        .realtime(_model(rt_xai.XaiRealtimeModelSettings(turn_detection=False)))
        .session(audio_retention='all', retain_audio_max_seconds=0.2) as session
    ):
        for _ in range(2):
            await session.send_audio(tenth_of_a_second)
            await session.commit_audio()
            await session.create_response()
            ws.advance()
            await session.wait_for_reply()
        async for event in session:  # pragma: no branch
            if isinstance(event, PartEndEvent) and isinstance(event.part, SpeechPart):
                if event.part.transcript == 'Second turn.':
                    break

    assert [
        (part.speaker, part.transcript, part.audio is not None)
        for message in session.all_messages()
        for part in message.parts
        if isinstance(part, SpeechPart)
    ] == [
        ('user', 'First turn.', False),
        ('assistant', 'Answer one.', False),
        ('user', 'Second turn.', True),
        ('assistant', 'Answer two.', True),
    ]


@pytest.mark.parametrize(
    ('transcription', 'commits', 'text'),
    [
        pytest.param(True, 1, True, id='transcribed'),
        pytest.param(False, 1, True, id='untranscribed'),
        pytest.param(
            False,
            2,
            True,
            id='untranscribed-twice',
            marks=pytest.mark.shadow_divergence(
                reason='the connection folds the two held commits into one, which xAI makes one turn; the new core '
                'records that one'
            ),
        ),
        pytest.param(True, 1, False, id='transcribed-no-text'),
    ],
)
async def test_commit_held_behind_a_reply_goes_after_what_was_sent_meanwhile(
    monkeypatch: pytest.MonkeyPatch, transcription: bool, commits: int, text: bool
) -> None:
    """A commit held until the reply in flight ends reaches xAI after text sent during that reply, and so does history."""
    ws = _PhasedWebSocket(
        [_created(), _updated()],
        [_response_frame('response.created', 'r0'), _transcript_delta('r0')],
        [
            json.dumps(
                {
                    'type': 'response.output_audio_transcript.done',
                    'response_id': 'r0',
                    'item_id': 'item-a1',
                    'output_index': 0,
                    'content_index': 0,
                    'transcript': 'Four.',
                }
            ),
            _response_frame('response.done', 'r0'),
            json.dumps({'type': 'input_audio_buffer.committed', 'item_id': 'item-u1'}),
            *([_user_transcript('item-u1', 'Spoken.')] if transcription else []),
            *_reply('r1', 'Answer.'),
        ],
    )
    monkeypatch.setattr(rt_xai.websockets, 'connect', _RecordingConnect([ws]))
    settings = rt_xai.XaiRealtimeModelSettings(turn_detection=False)
    if not transcription:
        settings['input_transcription_model'] = None

    async with Agent().realtime(_model(settings)).session() as session:
        await session.send('What is two plus two?')
        async for event in session:  # pragma: no branch
            if isinstance(event, PartDeltaEvent):
                break
        for _ in range(commits):
            await session.send_audio(_AUDIO.data)
            await session.commit_audio()
        if text:
            await session.send('Then this.')
            await session.send('Some context.', respond=False)
        else:
            await session.create_response()
        ws.advance()
        await session.wait_for_reply()

    sent = ['conversation.item.create'] * 2 if text else []
    assert _sent_types(ws)[-(2 + commits + len(sent)) :] == [
        *sent,
        *['input_audio_buffer.append'] * commits,
        'input_audio_buffer.commit',
        'response.create',
    ]
    spoken = 'user speech: Spoken.' if transcription else 'user speech: None'
    assert _turns(session.all_messages()) == [
        'user: What is two plus two?',
        'assistant speech: Four.',
        *(['user: Then this.', 'user: Some context.'] if text else []),
        *[spoken] * commits,
        'assistant speech: Answer.',
    ]


@pytest.mark.shadow_divergence(
    reason='the commit never went out, so xAI made no turn of it; the new core records only turns the provider made'
)
async def test_untranscribed_turn_with_a_held_commit_is_recorded_on_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """A spoken turn whose commit never went out still ends up in history when the session closes."""
    monkeypatch.setattr(rt_xai.websockets, 'connect', _RecordingConnect([_PhasedWebSocket([_created(), _updated()])]))
    settings = rt_xai.XaiRealtimeModelSettings(turn_detection=False, input_transcription_model=None)

    async with Agent().realtime(_model(settings)).session() as session:
        await session.send_audio(_AUDIO.data)
        await session.commit_audio()
        assert session.all_messages() == []

    assert _turns(session.all_messages()) == ['user speech: None']


async def test_request_replayed_after_a_reconnect_sends_no_clear() -> None:
    """A request held behind the commit's reply goes out alone after a reconnect.

    The resumed conversation may still owe that reply, which a clear would cancel.
    """
    replacement = FakeWebSocket([])
    replacements = iter([replacement])

    async def dial() -> Any:
        try:
            return next(replacements)
        except StopIteration:
            raise OSError('server is down')

    conn = _manual(
        _DropAfterFrames([_response_frame('response.created')]),
        dial=dial,
        reconnect={'base_delay': 0.0, 'max_attempts': 1},
    )
    await conn.send(_AUDIO)
    await conn.send(CommitAudio())
    await conn.send(CreateResponse())
    events = conn.__aiter__()
    await events.__anext__()
    await conn.send(CreateResponse())
    _ = [event async for event in events]

    assert _sent_types(replacement) == ['response.create']


async def test_speech_reported_after_the_commit_went_out_stays_with_that_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """xAI can report speech for audio only once its commit has gone out, which mustn't move the turn to the next commit."""

    def turn(item_id: str, transcript: str, response_id: str, answer: str) -> list[str]:
        return [
            json.dumps({'type': 'input_audio_buffer.speech_started', 'item_id': item_id, 'audio_start_ms': 0}),
            json.dumps({'type': 'input_audio_buffer.committed', 'item_id': item_id}),
            _user_transcript(item_id, transcript),
            *_reply(response_id, answer),
        ]

    ws = _PhasedWebSocket(
        [_created(), _updated()],
        [],
        turn('item-u1', 'First turn.', 'r1', 'Answer one.'),
        turn('item-u2', 'Second turn.', 'r2', 'Answer two.'),
    )
    monkeypatch.setattr(rt_xai.websockets, 'connect', _RecordingConnect([ws]))

    async with Agent().realtime(_model(rt_xai.XaiRealtimeModelSettings(turn_detection=False))).session() as session:
        for _ in range(2):
            await session.send_audio(_AUDIO.data)
            await session.commit_audio()
            await session.create_response()
            ws.advance()
            await session.wait_for_reply()

    assert _turns(session.all_messages()) == [
        'user speech: First turn.',
        'assistant speech: Answer one.',
        'user speech: Second turn.',
        'assistant speech: Answer two.',
    ]


@pytest.mark.parametrize('transcription', [True, False], ids=['transcribed', 'untranscribed'])
@pytest.mark.parametrize('request_by', ['create_response', 'text'])
async def test_audio_committed_by_a_request_is_recorded_in_place(
    monkeypatch: pytest.MonkeyPatch, request_by: str, transcription: bool
) -> None:
    """Audio never committed is committed by the next request, which is when it joins the conversation."""
    ws = _PhasedWebSocket(
        [_created(), _updated()],
        [],
        [
            json.dumps({'type': 'input_audio_buffer.committed', 'item_id': 'item-u1'}),
            *([_user_transcript('item-u1', 'Spoken.')] if transcription else []),
            *_reply('r1', 'Answer.'),
        ],
    )
    monkeypatch.setattr(rt_xai.websockets, 'connect', _RecordingConnect([ws]))
    settings = rt_xai.XaiRealtimeModelSettings(turn_detection=False)
    if not transcription:
        settings['input_transcription_model'] = None

    async with Agent().realtime(_model(settings)).session() as session:
        await session.send_audio(_AUDIO.data)
        if request_by == 'text':
            await session.send('Text.')
        else:
            await session.create_response()
        ws.advance()
        await session.wait_for_reply()
        turns = _turns(session.all_messages())

    assert turns == [
        *(['user: Text.'] if request_by == 'text' else []),
        f'user speech: {"Spoken." if transcription else None}',
        'assistant speech: Answer.',
    ]


async def test_speech_for_audio_already_committed_is_not_held_for_the_next_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """xAI reporting speech for a turn whose commit went out doesn't tie that turn to the next commit.

    The next turn's audio was sent before xAI reported on the first, so it reserved no place of its own.
    """

    def turn(item_id: str, transcript: str, response_id: str, answer: str) -> list[str]:
        return [
            json.dumps({'type': 'input_audio_buffer.speech_started', 'item_id': item_id, 'audio_start_ms': 0}),
            json.dumps({'type': 'input_audio_buffer.committed', 'item_id': item_id}),
            _user_transcript(item_id, transcript),
            *_reply(response_id, answer),
        ]

    ws = _PhasedWebSocket(
        [_created(), _updated()],
        [],
        turn('item-u1', 'First.', 'r1', 'One.'),
        turn('item-u2', 'Second.', 'r2', 'Two.'),
    )
    monkeypatch.setattr(rt_xai.websockets, 'connect', _RecordingConnect([ws]))

    async with Agent().realtime(_model(rt_xai.XaiRealtimeModelSettings(turn_detection=False))).session() as session:
        await session.send_audio(_AUDIO.data)
        await session.commit_audio()
        await session.create_response()
        await session.send_audio(_AUDIO.data)
        await session.commit_audio()
        ws.advance()
        await session.wait_for_reply()
        await session.create_response()
        ws.advance()
        await session.wait_for_reply()
        turns = _turns(session.all_messages())

    assert turns == ['user speech: First.', 'assistant speech: One.', 'user speech: Second.', 'assistant speech: Two.']


@pytest.mark.shadow_divergence(
    reason=(
        'synthetic frames: xAI reports speech in audio the connection still holds back, and adds no item at speech '
        'start (recorded: `test_xai_ws/test_push_to_talk_replies_only_when_asked`)'
    )
)
async def test_turns_of_a_commit_held_behind_a_reply_keep_their_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two utterances xAI transcribed before their held commit went out are recorded in the order spoken."""

    def utterance(item_id: str, transcript: str) -> list[str]:
        return [
            json.dumps({'type': 'input_audio_buffer.speech_started', 'item_id': item_id, 'audio_start_ms': 0}),
            _user_transcript(item_id, transcript),
        ]

    ws = _PhasedWebSocket(
        [_created(), _updated()],
        [_response_frame('response.created', 'r0'), _transcript_delta('r0')],
        [*utterance('item-u1', 'One.'), *utterance('item-u2', 'Two.')],
        [
            _response_frame('response.done', 'r0'),
            json.dumps({'type': 'input_audio_buffer.committed', 'item_id': 'item-u2'}),
            *_reply('r1', 'Answer.'),
        ],
    )
    monkeypatch.setattr(rt_xai.websockets, 'connect', _RecordingConnect([ws]))

    async with Agent().realtime(_model(rt_xai.XaiRealtimeModelSettings(turn_detection=False))).session() as session:
        await session.send('What is two plus two?')
        async for event in session:  # pragma: no branch
            if isinstance(event, PartDeltaEvent):
                break
        await session.send_audio(_AUDIO.data)
        await session.commit_audio()
        await session.send('Then this.')
        ws.advance()
        async for event in session:  # pragma: no branch
            if isinstance(event, PartEndEvent) and isinstance(event.part, SpeechPart):
                if event.part.transcript == 'Two.':
                    break
        ws.advance()
        await session.wait_for_reply()

    assert _turns(session.all_messages()) == [
        'user: What is two plus two?',
        'assistant speech: Four.',
        'user: Then this.',
        'user speech: One.',
        'user speech: Two.',
        'assistant speech: Answer.',
    ]


async def test_a_request_that_fails_to_commit_the_buffer_places_nothing_ahead_of_the_next_commit() -> None:
    """The request that would have committed the buffered speech never went out, so neither did the commit."""

    class _FailingRequest(FakeWebSocket):
        async def send(self, data: str) -> None:
            if '"response.create"' in data:
                raise OSError('gone')
            await super().send(data)

    conn = _manual(_FailingRequest([]))
    await conn.send(_AUDIO)
    with pytest.raises(OSError):
        await conn.send(CreateResponse())
    assert conn._lifecycle._sent_before_commits == deque()  # pyright: ignore[reportPrivateUsage]
    # A request committing nothing has no snapshot to drop.
    with pytest.raises(OSError):
        await conn.send(CreateResponse())
    assert conn._lifecycle._sent_before_commits == deque()  # pyright: ignore[reportPrivateUsage]
