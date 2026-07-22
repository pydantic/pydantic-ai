"""Tests for the OpenAI realtime provider (event mapping, handshake, send), all network-free."""

from __future__ import annotations as _annotations

import asyncio
import base64
import io
import json
import re
import wave
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, Literal

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    CachePoint,
    CompactionPart,
    DocumentUrl,
    FilePart,
    FinishReason,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    RetryPromptPart,
    SpeechPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UploadedFile,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.realtime import (
    AudioDelta,
    AudioInput,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    CreateResponse,
    InputSpeechEndEvent,
    InputSpeechStartEvent,
    InputTranscript,
    InputTranscriptionFailedEvent,
    RealtimeModelProfile,
    RealtimeModelSettings,
    RealtimeSession,
    ReconnectedEvent,
    SessionUsageEvent,
    ToolCall,
    ToolResult,
    Transcript,
    TruncateOutput,
    TurnCompleteEvent,
    TurnDetection,
)
from pydantic_ai.realtime._base import ImageInput, SessionErrorEvent, TextInput, merge_realtime_profile
from pydantic_ai.realtime._openai_protocol import map_conversation_event, realtime_websocket_url
from pydantic_ai.settings import ThinkingLevel, ToolOrOutput
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from ..conftest import IsStr, try_import
from .test_session import make_tool_manager

with try_import() as imports_successful:
    from openai import AsyncOpenAI
    from openai.types.realtime import RealtimeResponseUsage
    from openai.types.realtime.conversation_item_input_audio_transcription_completed_event import (
        UsageTranscriptTextUsageDuration,
        UsageTranscriptTextUsageTokens,
        UsageTranscriptTextUsageTokensInputTokenDetails,
    )

    from pydantic_ai.providers.gateway import gateway_provider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.realtime import openai as rt_openai
    from pydantic_ai.realtime.openai import OpenAIRealtimeConnection, OpenAIRealtimeModel, map_event

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai / websockets not installed')


def _wav_bytes(pcm: bytes, sample_rate: int = 24000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def test_map_transcription_usage() -> None:
    assert rt_openai._map_transcription_usage(None) is None  # pyright: ignore[reportPrivateUsage]
    assert rt_openai._map_transcription_usage(UsageTranscriptTextUsageDuration(type='duration', seconds=0.5)) is None  # pyright: ignore[reportPrivateUsage]
    assert rt_openai._map_transcription_usage(  # pyright: ignore[reportPrivateUsage]
        UsageTranscriptTextUsageDuration(type='duration', seconds=3)
    ) == RequestUsage(details={'input_transcription_seconds': 3})
    assert rt_openai._map_transcription_usage(  # pyright: ignore[reportPrivateUsage]
        UsageTranscriptTextUsageTokens(
            type='tokens',
            input_tokens=5,
            output_tokens=2,
            total_tokens=7,
            input_token_details=UsageTranscriptTextUsageTokensInputTokenDetails(audio_tokens=4, text_tokens=1),
        )
    ) == RequestUsage(
        details={
            'input_transcription_tokens': 7,
            'input_transcription_audio_tokens': 4,
            'input_transcription_text_tokens': 1,
        }
    )
    assert rt_openai._map_transcription_usage(  # pyright: ignore[reportPrivateUsage]
        UsageTranscriptTextUsageTokens(
            type='tokens', input_tokens=5, output_tokens=2, total_tokens=7, input_token_details=None
        )
    ) == RequestUsage(details={'input_transcription_tokens': 7})


@pytest.mark.parametrize(
    'usage',
    [
        RealtimeResponseUsage.construct(input_token_details='bad'),
        RealtimeResponseUsage.construct(output_token_details='bad'),
        RealtimeResponseUsage.construct(
            input_token_details={'cached_tokens_details': 'bad'},
        ),
    ],
)
def test_map_usage_rejects_malformed_constructed_details(usage: RealtimeResponseUsage) -> None:
    with pytest.raises(ValueError, match='must be an object'):
        rt_openai._map_usage(usage)  # pyright: ignore[reportPrivateUsage]


def test_map_transcription_usage_rejects_malformed_constructed_details() -> None:
    usage = UsageTranscriptTextUsageTokens.construct(type='tokens', input_token_details='bad')
    with pytest.raises(ValueError, match='must be an object'):
        rt_openai._map_transcription_usage(usage)  # pyright: ignore[reportPrivateUsage]


def test_merge_realtime_profile_skips_empty_layers_and_applies_overrides() -> None:
    assert merge_realtime_profile(None, None, {}) == {}
    assert merge_realtime_profile(
        RealtimeModelProfile(supports_image_input=False),
        None,
        RealtimeModelProfile(supports_image_input=True),
    ) == {'supports_image_input': True}


def _connect(
    model: OpenAIRealtimeModel,
    instructions: str,
    *,
    messages: Sequence[ModelMessage] | None = None,
    tools: list[ToolDefinition] | None = None,
    model_settings: RealtimeModelSettings | None = None,
) -> AbstractAsyncContextManager[OpenAIRealtimeConnection]:
    return model.connect(
        messages=[*(messages or ()), ModelRequest(parts=[], instructions=instructions)],
        model_settings=model_settings,
        model_request_parameters=ModelRequestParameters(function_tools=tools or []),
    )


def test_realtime_url_for_gateway_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('PYDANTIC_AI_GATEWAY_API_KEY', 'gw-key')
    monkeypatch.setenv('PYDANTIC_AI_GATEWAY_BASE_URL', 'https://gateway.pydantic.dev/proxy')
    string_model = OpenAIRealtimeModel('gpt-realtime', provider='gateway/openai')
    instance_model = OpenAIRealtimeModel(
        'gpt-realtime',
        provider=gateway_provider('openai', api_key='gw-key', base_url='https://gateway.pydantic.dev/proxy'),
    )
    plain_model = OpenAIRealtimeModel('gpt-realtime', provider=OpenAIProvider(api_key='k'))

    assert '/v1/realtime' in string_model._realtime_url()  # pyright: ignore[reportPrivateUsage]
    assert '/v1/realtime' in instance_model._realtime_url()  # pyright: ignore[reportPrivateUsage]
    assert plain_model._realtime_url().count('/v1') == 1  # pyright: ignore[reportPrivateUsage]


def test_map_audio_delta() -> None:
    payload = base64.b64encode(b'\x01\x02').decode('ascii')
    for event_type in ('response.output_audio.delta', 'response.audio.delta'):
        event = map_event({'type': event_type, 'delta': payload, 'item_id': 'item-a'})
        assert event == AudioDelta(data=b'\x01\x02', item_id='item-a')


def test_map_audio_delta_non_string_delta() -> None:
    assert map_event({'type': 'response.output_audio.delta', 'delta': 123}) is None


def test_map_transcript_delta_and_done() -> None:
    for event_type in ('response.output_audio_transcript.delta', 'response.audio_transcript.delta'):
        assert map_event({'type': event_type, 'delta': 'hel', 'item_id': 'item-a'}) == Transcript(
            text='hel', is_final=False, item_id='item-a'
        )
    for event_type in ('response.output_audio_transcript.done', 'response.audio_transcript.done'):
        assert map_event({'type': event_type, 'transcript': 'hello', 'item_id': 'item-a'}) == Transcript(
            text='hello', is_final=True, item_id='item-a'
        )


def test_map_text_output_delta_and_done() -> None:
    # `output_text=True` distinguishes plain text output from an audio transcript, so the session
    # persists it as a `TextPart` rather than a `SpeechPart`.
    assert map_event({'type': 'response.output_text.delta', 'delta': 'hel'}) == Transcript(
        text='hel', is_final=False, output_text=True
    )
    assert map_event({'type': 'response.output_text.done', 'text': 'hello'}) == Transcript(
        text='hello', is_final=True, output_text=True
    )


def test_map_transcript_missing_field_defaults_to_empty() -> None:
    assert map_event({'type': 'response.output_audio_transcript.delta'}) == Transcript(text='', is_final=False)


@pytest.mark.parametrize('status', ['completed', None])
def test_map_input_transcript_completed(status: str | None) -> None:
    data = {
        'type': 'conversation.item.input_audio_transcription.completed',
        'transcript': 'weather?',
        'item_id': 'item-u',
    }
    if status is not None:
        data['status'] = status
    assert map_event(data) == InputTranscript(text='weather?', is_final=True, item_id='item-u')


def test_map_input_transcript_completed_drops_interim_status() -> None:
    assert (
        map_event(
            {
                'type': 'conversation.item.input_audio_transcription.completed',
                'status': 'in_progress',
                'transcript': 'wea',
                'item_id': 'item-u',
            }
        )
        is None
    )


def test_map_input_transcript_delta() -> None:
    event = map_event(
        {'type': 'conversation.item.input_audio_transcription.delta', 'delta': 'wea', 'item_id': 'item-u'}
    )
    assert event == InputTranscript(text='wea', is_final=False, item_id='item-u')


def test_map_function_call() -> None:
    event = map_event(
        {
            'type': 'response.function_call_arguments.done',
            'call_id': 'call_1',
            'name': 'get_weather',
            'arguments': '{"city": "Paris"}',
        }
    )
    assert event == ToolCall(
        tool_call_id='call_1',
        tool_name='get_weather',
        args='{"city": "Paris"}',
        response_usage_follows=True,
    )


def test_map_function_call_missing_arguments_defaults_to_empty_object() -> None:
    event = map_event({'type': 'response.function_call_arguments.done', 'call_id': 'c', 'name': 'n'})
    assert isinstance(event, ToolCall)
    assert event.args == '{}'


def _response_done(response: Any) -> dict[str, Any]:
    return {'type': 'response.done', 'response': response}


def test_map_response_done_normal() -> None:
    assert map_event(_response_done({'id': 'resp-1', 'status': 'completed', 'output': []})) == TurnCompleteEvent(
        interrupted=False,
        provider_response_id='resp-1',
        finish_reason='stop',
        provider_details={'status': 'completed'},
    )


def test_map_response_done_cancelled() -> None:
    # A cancelled response is a barge-in, not an error: `interrupted=True` (→ `state='interrupted'`)
    # carries the meaning and `finish_reason` is left unset, matching a classic cancelled stream.
    assert map_event(_response_done({'id': 'resp-2', 'status': 'cancelled'})) == TurnCompleteEvent(
        interrupted=True,
        provider_response_id='resp-2',
        finish_reason=None,
        provider_details={'status': 'cancelled'},
    )


@pytest.mark.parametrize(
    ('reason', 'finish_reason'),
    [('max_output_tokens', 'length'), ('content_filter', 'content_filter')],
)
def test_map_response_done_incomplete_reason(reason: str, finish_reason: FinishReason) -> None:
    response: dict[str, Any] = {
        'id': 'resp-incomplete',
        'status': 'incomplete',
        'status_details': {'reason': reason},
        'output': [],
    }
    assert map_event(_response_done(response)) == TurnCompleteEvent(
        interrupted=False,
        provider_response_id='resp-incomplete',
        finish_reason=finish_reason,
        provider_details={'status': 'incomplete', 'finish_reason': reason},
    )


def test_map_response_done_function_call_only_is_skipped() -> None:
    assert (
        map_event(_response_done({'status': 'completed', 'output': [{'type': 'function_call', 'name': 'x'}]})) is None
    )


def test_map_response_done_mixed_output_is_turn_complete() -> None:
    data = _response_done({'status': 'completed', 'output': [{'type': 'function_call'}, {'type': 'message'}]})
    assert map_event(data) == TurnCompleteEvent(
        interrupted=False, finish_reason='stop', provider_details={'status': 'completed'}
    )


def test_map_response_done_without_response_object() -> None:
    assert map_event({'type': 'response.done'}) == TurnCompleteEvent(
        interrupted=False, provider_details={'status': None}
    )


def test_map_response_done_failed_and_unknown_incomplete_reason() -> None:
    assert map_event(_response_done({'status': 'failed'})) == TurnCompleteEvent(
        interrupted=False, finish_reason='error', provider_details={'status': 'failed'}
    )
    assert map_event(_response_done({'status': 'incomplete', 'status_details': {'reason': 'network'}})) == (
        TurnCompleteEvent(
            interrupted=False,
            provider_details={'status': 'incomplete', 'finish_reason': 'network'},
        )
    )


def test_map_conversation_item_without_identifiers_is_ignored() -> None:
    assert map_conversation_event({'type': 'conversation.item.created', 'item': {}}) is None
    assert map_conversation_event({'type': 'conversation.item.created'}) is None


@pytest.mark.parametrize(
    'frame',
    [
        {'type': 'conversation.created', 'conversation': 'bad'},
        {'type': 'conversation.item.created', 'item': 'bad'},
    ],
)
def test_map_conversation_event_rejects_malformed_nested_object(frame: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match='must be an object'):
        map_conversation_event(frame)
    assert map_conversation_event({'type': 'conversation.created'}) is None


def test_map_error_event_with_message() -> None:
    assert map_event({'type': 'error', 'error': {'message': 'bad'}}) == SessionErrorEvent(message='bad')


def test_map_error_event_without_message_serializes_payload() -> None:
    assert map_event({'type': 'error', 'error': {'code': 'x'}}) == SessionErrorEvent(
        message=json.dumps({'code': 'x'}), code='x'
    )


def test_map_error_event_non_dict_payload() -> None:
    assert map_event({'type': 'error', 'error': 'plain'}) == SessionErrorEvent(message='plain')


def test_map_error_event_with_type_and_code_is_recoverable() -> None:
    event = map_event({'type': 'error', 'error': {'message': 'bad', 'type': 'invalid_request_error', 'code': 'c1'}})
    assert event == SessionErrorEvent(message='bad', type='invalid_request_error', code='c1', recoverable=True)


def test_map_usage_full_payload() -> None:
    sdk_usage = RealtimeResponseUsage.construct(
        input_tokens=100,
        output_tokens=50,
        input_token_details={
            'audio_tokens': 80,
            'cached_tokens': 30,
            'text_tokens': 20,
            'image_tokens': 5,
            'cached_tokens_details': {'audio_tokens': 10},
        },
        output_token_details={'audio_tokens': 40, 'text_tokens': 10},
    )
    usage = rt_openai._map_usage(sdk_usage)  # pyright: ignore[reportPrivateUsage]
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
    sdk_usage = RealtimeResponseUsage.construct(input_tokens=7)
    assert rt_openai._map_usage(sdk_usage) == RequestUsage(input_tokens=7)  # pyright: ignore[reportPrivateUsage]
    assert rt_openai._map_usage(None) is None  # pyright: ignore[reportPrivateUsage]


def test_map_speech_started() -> None:
    assert map_event({'type': 'input_audio_buffer.speech_started'}) == InputSpeechStartEvent()


def test_map_speech_stopped() -> None:
    assert map_event({'type': 'input_audio_buffer.speech_stopped'}) == InputSpeechEndEvent()


def test_map_unhandled_event_returns_none() -> None:
    # Frames we don't surface as user-facing events (lifecycle acks like `session.created`, and
    # `rate_limits.updated`, which has no `RealtimeEvent` representation) fall through to `None`.
    assert map_event({'type': 'session.created'}) is None
    assert map_event({'type': 'rate_limits.updated', 'rate_limits': [{'name': 'requests', 'limit': 100}]}) is None


@pytest.mark.parametrize(
    ('frame', 'expected'),
    [
        ({'type': 'response.output_audio.delta', 'delta': 'AQI=', 'item_id': 'a'}, AudioDelta(b'\x01\x02', 'a')),
        ({'type': 'response.audio.delta', 'delta': 'AQI=', 'item_id': 'a'}, AudioDelta(b'\x01\x02', 'a')),
        (
            {'type': 'response.output_audio_transcript.delta', 'delta': 'hel', 'item_id': 'a'},
            Transcript('hel', is_final=False, item_id='a'),
        ),
        (
            {'type': 'response.audio_transcript.delta', 'delta': 'hel', 'item_id': 'a'},
            Transcript('hel', is_final=False, item_id='a'),
        ),
        (
            {'type': 'response.output_audio_transcript.done', 'transcript': 'hello', 'item_id': 'a'},
            Transcript('hello', is_final=True, item_id='a'),
        ),
        (
            {'type': 'response.audio_transcript.done', 'transcript': 'hello', 'item_id': 'a'},
            Transcript('hello', is_final=True, item_id='a'),
        ),
        ({'type': 'response.output_text.delta', 'delta': 'hel'}, Transcript('hel', False, output_text=True)),
        ({'type': 'response.output_text.done', 'text': 'hello'}, Transcript('hello', True, output_text=True)),
        (
            {'type': 'conversation.item.input_audio_transcription.delta', 'delta': 'hel', 'item_id': 'u'},
            InputTranscript('hel', is_final=False, item_id='u'),
        ),
        (
            {
                'type': 'conversation.item.input_audio_transcription.completed',
                'transcript': 'hello',
                'item_id': 'u',
            },
            InputTranscript('hello', is_final=True, item_id='u'),
        ),
        (
            {
                'type': 'response.function_call_arguments.done',
                'call_id': 'call-1',
                'name': 'weather',
                'arguments': '{}',
            },
            ToolCall('call-1', 'weather', '{}', response_usage_follows=True),
        ),
        ({'type': 'input_audio_buffer.speech_started'}, InputSpeechStartEvent()),
        ({'type': 'input_audio_buffer.speech_stopped'}, InputSpeechEndEvent()),
        (
            {'type': 'response.done', 'response': {'id': 'r', 'status': 'completed', 'output': []}},
            TurnCompleteEvent(False, 'r', 'stop', {'status': 'completed'}),
        ),
        ({'type': 'error', 'error': {'message': 'bad'}}, SessionErrorEvent('bad')),
        (
            {'type': 'conversation.item.input_audio_transcription.failed', 'error': {'message': 'bad'}},
            InputTranscriptionFailedEvent(message='bad'),
        ),
        (
            {
                'type': 'conversation.item.input_audio_transcription.failed',
                'error': {'message': 'bad', 'type': 'transcription_error', 'code': 'audio_unintelligible'},
                'item_id': 'u',
                'content_index': 2,
            },
            InputTranscriptionFailedEvent(
                message='bad',
                type='transcription_error',
                code='audio_unintelligible',
                item_id='u',
                content_index=2,
            ),
        ),
        (
            # A `DeploymentNotFound` transcription failure is a misconfiguration (the transcription model
            # isn't deployed on the Azure resource), not a transient failure, so it maps to a
            # non-recoverable error the session raises — unlike `audio_unintelligible` above.
            {
                'type': 'conversation.item.input_audio_transcription.failed',
                'error': {'message': 'x', 'type': 'server_error', 'code': 'DeploymentNotFound'},
            },
            SessionErrorEvent(message=IsStr(regex=r'.*transcription model is not deployed.*'), recoverable=False),
        ),
    ],
)
def test_sdk_typed_event_mapping_guard(frame: dict[str, Any], expected: object) -> None:
    """Pin the SDK event classes and attributes used by the shared protocol mapper."""
    assert map_event(frame) == expected


def test_model_repr_hides_api_key() -> None:
    model = OpenAIRealtimeModel('gpt-realtime', provider=OpenAIProvider(api_key='super-secret'))
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

    model = OpenAIRealtimeModel(
        'gpt-realtime',
        provider=OpenAIProvider(api_key='k'),
        settings=rt_openai.OpenAIRealtimeModelSettings(voice='alloy'),
    )
    tools = [ToolDefinition(name='get_weather', description='Weather', parameters_json_schema={'type': 'object'})]

    async with _connect(model, 'Be nice', tools=tools) as conn:
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
    assert session['audio']['input']['transcription'] == {'model': 'gpt-realtime-whisper'}  # `'auto'` resolved
    assert session['audio']['output']['voice'] == 'alloy'
    assert session['tools'][0]['name'] == 'get_weather'
    assert session['tools'][0]['type'] == 'function'


@pytest.mark.anyio
async def test_connect_injects_trace_context_into_handshake(monkeypatch: pytest.MonkeyPatch) -> None:
    """An active span propagates `traceparent` into the handshake headers, for gateway/OTel-proxy correlation.

    The realtime WebSocket bypasses the provider's `httpx` client, so `connect()` injects trace context
    itself (the analogue of the gateway provider's HTTP request hook). A unit test because a cassette's
    request matcher ignores handshake headers, so it wouldn't catch a regression here.
    """
    pytest.importorskip('opentelemetry.sdk')
    from opentelemetry.sdk.trace import TracerProvider

    ws = FakeWebSocket([_created(), _updated()])
    fake_connect = FakeConnect(ws)
    monkeypatch.setattr(rt_openai.websockets, 'connect', fake_connect)

    model = OpenAIRealtimeModel('gpt-realtime', provider=OpenAIProvider(api_key='k'))
    tracer = TracerProvider().get_tracer('test')
    with tracer.start_as_current_span('root'):
        async with _connect(model, 'hi') as conn:
            _ = [e async for e in conn]

    assert fake_connect.headers is not None
    assert fake_connect.headers['Authorization'] == 'Bearer k'
    # W3C `traceparent` names the active span, so a proxy (e.g. the Pydantic AI Gateway) can nest its
    # own realtime spans under this trace.
    assert 'traceparent' in fake_connect.headers


async def test_connect_resolves_async_api_key_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """The handshake resolves an async `api_key` provider via the SDK, not the empty static field.

    `AsyncOpenAI` accepts a `Callable[[], Awaitable[str]]` for `api_key`, leaving `client.api_key` empty
    until the SDK refreshes it per request. The raw WebSocket handshake bypasses that request path, so a
    regression would send `Authorization: Bearer ` (empty). A unit test because a cassette's request
    matcher ignores handshake headers.
    """

    async def provide_key() -> str:
        return 'sk-resolved'

    client = AsyncOpenAI(api_key=provide_key)
    assert not client.api_key  # unresolved until the SDK refreshes it

    ws = FakeWebSocket([_created(), _updated()])
    fake_connect = FakeConnect(ws)
    monkeypatch.setattr(rt_openai.websockets, 'connect', fake_connect)

    model = OpenAIRealtimeModel('gpt-realtime', provider=OpenAIProvider(openai_client=client))
    async with _connect(model, 'hi') as conn:
        _ = [e async for e in conn]

    assert fake_connect.headers is not None
    assert fake_connect.headers['Authorization'] == 'Bearer sk-resolved'


def test_session_config_server_vad_params() -> None:
    model = OpenAIRealtimeModel(
        settings=rt_openai.OpenAIRealtimeModelSettings(
            openai_turn_detection=rt_openai.ServerVAD(
                threshold=0.7,
                prefix_padding_ms=200,
                silence_duration_ms=400,
                create_response=False,
                interrupt_response=False,
                idle_timeout_ms=5000,
            ),
        )
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


def test_session_config_truncation_modes() -> None:
    # A plain mode passes through as-is; a retention ratio maps to the retention_ratio truncation shape.
    auto = OpenAIRealtimeModel(settings=rt_openai.OpenAIRealtimeModelSettings(openai_truncation='disabled'))
    assert auto._session_config('hi', None, None)['truncation'] == 'disabled'  # pyright: ignore[reportPrivateUsage]

    ratio = OpenAIRealtimeModel(
        settings=rt_openai.OpenAIRealtimeModelSettings(
            openai_truncation={'type': 'retention_ratio', 'retention_ratio': 0.8}
        )
    )
    assert ratio._session_config('hi', None, None)['truncation'] == {  # pyright: ignore[reportPrivateUsage]
        'type': 'retention_ratio',
        'retention_ratio': 0.8,
    }
    # Absent by default so the wire stays byte-identical for sessions that don't set it.
    assert 'truncation' not in OpenAIRealtimeModel()._session_config('hi', None, None)  # pyright: ignore[reportPrivateUsage]


def test_session_config_thinking_maps_to_reasoning_on_reasoning_models() -> None:
    # `thinking` maps to reasoning effort on the `gpt-realtime-2*` reasoning models (live-verified
    # that these accept `reasoning.effort` while the GA `gpt-realtime` rejects it).
    def reasoning(thinking: ThinkingLevel) -> object:
        model = OpenAIRealtimeModel(
            'gpt-realtime-2.1', settings=rt_openai.OpenAIRealtimeModelSettings(thinking=thinking)
        )
        return model._session_config('hi', None, None).get('reasoning')  # pyright: ignore[reportPrivateUsage]

    assert reasoning('low') == {'effort': 'low'}
    assert reasoning('high') == {'effort': 'high'}
    assert reasoning(True) == {'effort': 'medium'}
    # `thinking=False` maps to effort `'none'`, which the realtime `reasoning.effort` doesn't accept,
    # so it's omitted (a reasoning model falls back to its default rather than erroring).
    assert reasoning(False) is None


def test_session_config_thinking_on_non_reasoning_model_warns() -> None:
    # The GA `gpt-realtime` isn't a reasoning model, so `thinking` is dropped with a warning rather
    # than sent (which the server rejects with "Unsupported option for this model").
    model = OpenAIRealtimeModel('gpt-realtime', settings=rt_openai.OpenAIRealtimeModelSettings(thinking='high'))
    with pytest.warns(UserWarning, match='does not support the `thinking` setting'):
        config = model._session_config('hi', None, None)  # pyright: ignore[reportPrivateUsage]
    assert 'reasoning' not in config


def test_session_config_openai_turn_detection_overrides_base() -> None:
    model = OpenAIRealtimeModel(
        settings=rt_openai.OpenAIRealtimeModelSettings(
            turn_detection=TurnDetection(sensitivity='low'),
            openai_turn_detection=rt_openai.SemanticVAD(eagerness='high'),
        )
    )
    config = model._session_config('hi', None, None)  # pyright: ignore[reportPrivateUsage]
    assert config['audio']['input']['turn_detection'] == {
        'type': 'semantic_vad',
        'eagerness': 'high',
        'create_response': True,
        'interrupt_response': True,
    }


@pytest.mark.parametrize(('sensitivity', 'threshold'), [('low', 0.7), ('medium', 0.5), ('high', 0.3)])
def test_session_config_cross_provider_turn_detection_sensitivity(
    sensitivity: Literal['low', 'medium', 'high'], threshold: float
) -> None:
    settings = rt_openai.OpenAIRealtimeModelSettings(turn_detection=TurnDetection(sensitivity=sensitivity))
    config = OpenAIRealtimeModel(settings=settings)._session_config(  # pyright: ignore[reportPrivateUsage]
        'hi', None, None
    )
    assert config['audio']['input']['turn_detection']['threshold'] == threshold


def test_session_config_manual_turn_detection_is_null() -> None:
    """`turn_detection=False` disables VAD (push-to-talk), sent as an explicit null."""
    model = OpenAIRealtimeModel(settings=rt_openai.OpenAIRealtimeModelSettings(turn_detection=False))
    config = model._session_config('hi', None, None)  # pyright: ignore[reportPrivateUsage]
    assert config['audio']['input']['turn_detection'] is None


def test_session_config_turn_detection_true_matches_default() -> None:
    """`turn_detection=True` enables server VAD at the provider defaults — identical to an absent setting."""
    enabled = OpenAIRealtimeModel(settings=rt_openai.OpenAIRealtimeModelSettings(turn_detection=True))
    default = OpenAIRealtimeModel()
    assert (
        enabled._session_config('hi', None, None)['audio']['input']['turn_detection']  # pyright: ignore[reportPrivateUsage]
        == default._session_config('hi', None, None)['audio']['input']['turn_detection']  # pyright: ignore[reportPrivateUsage]
    )


def test_session_config_noise_reduction_and_speed_and_modalities() -> None:
    model = OpenAIRealtimeModel(
        settings=rt_openai.OpenAIRealtimeModelSettings(
            openai_input_noise_reduction='near_field', openai_output_speed=1.25, output_modality='text'
        )
    )
    config = model._session_config('hi', None, None)  # pyright: ignore[reportPrivateUsage]
    assert config['audio']['input']['noise_reduction'] == {'type': 'near_field'}
    assert config['audio']['output']['speed'] == 1.25
    assert config['output_modalities'] == ['text']


def test_session_config_forwards_parallel_tool_calls_and_tool_choice() -> None:
    settings = rt_openai.OpenAIRealtimeModelSettings(parallel_tool_calls=True, tool_choice='required')
    model = OpenAIRealtimeModel(settings=settings)
    assert model.settings == settings
    config = model._session_config('hi', None, settings)  # pyright: ignore[reportPrivateUsage]
    assert config['parallel_tool_calls'] is True
    assert config['tool_choice'] == 'required'


def test_session_config_merges_model_defaults_and_connection_overrides() -> None:
    model = OpenAIRealtimeModel(settings=rt_openai.OpenAIRealtimeModelSettings(voice='alloy', max_tokens=128))
    config = model._session_config(  # pyright: ignore[reportPrivateUsage]
        'hi', None, rt_openai.OpenAIRealtimeModelSettings(voice='echo')
    )

    assert config['audio']['output']['voice'] == 'echo'
    assert config['max_output_tokens'] == 128


def test_session_config_tool_choice_single_function() -> None:
    model = OpenAIRealtimeModel()
    config = model._session_config(  # pyright: ignore[reportPrivateUsage]
        'hi', None, rt_openai.OpenAIRealtimeModelSettings(tool_choice=['get_weather'])
    )
    assert config['tool_choice'] == {'type': 'function', 'name': 'get_weather'}


def test_session_config_tool_choice_multi_tool_dropped() -> None:
    model = OpenAIRealtimeModel()
    config = model._session_config(  # pyright: ignore[reportPrivateUsage]
        'hi', None, rt_openai.OpenAIRealtimeModelSettings(tool_choice=['a', 'b'])
    )
    assert 'tool_choice' not in config  # realtime can't express a multi-tool restriction


def test_session_config_tool_choice_tool_or_output_dropped() -> None:
    model = OpenAIRealtimeModel()
    settings = rt_openai.OpenAIRealtimeModelSettings(tool_choice=ToolOrOutput(function_tools=['a']))
    config = model._session_config('hi', None, settings)  # pyright: ignore[reportPrivateUsage]
    assert 'tool_choice' not in config  # ToolOrOutput restriction isn't expressible in realtime


@pytest.mark.anyio
async def test_connect_skips_unrelated_events_during_handshake(monkeypatch: pytest.MonkeyPatch) -> None:
    rate_limits = json.dumps({'type': 'rate_limits.updated'})
    ws = FakeWebSocket([rate_limits, _created(), _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    model = OpenAIRealtimeModel('gpt-realtime')
    async with _connect(model, 'x') as conn:
        assert [e async for e in conn] == []


@pytest.mark.anyio
async def test_connect_surfaces_handshake_error(monkeypatch: pytest.MonkeyPatch) -> None:
    error = json.dumps({'type': 'error', 'error': {'message': 'invalid model'}})
    ws = FakeWebSocket([error])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    model = OpenAIRealtimeModel('gpt-realtime')
    with pytest.raises(RuntimeError, match='invalid model'):
        async with _connect(model, 'x'):
            pass  # pragma: no cover


class HangingWebSocket(FakeWebSocket):
    """A websocket whose `recv` never returns, to exercise the handshake timeout."""

    async def recv(self) -> Any:
        await asyncio.Event().wait()


@pytest.mark.anyio
async def test_connect_handshake_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = HangingWebSocket([])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    model = OpenAIRealtimeModel('gpt-realtime', settings=rt_openai.OpenAIRealtimeModelSettings(handshake_timeout=0.02))
    with pytest.raises(TimeoutError, match=re.escape("'session.created'")):
        async with _connect(model, 'x'):
            pass  # pragma: no cover


@pytest.mark.anyio
async def test_connect_open_failure_propagates_without_teardown(monkeypatch: pytest.MonkeyPatch) -> None:
    # If the very first connection fails to open, there is nothing to close on teardown.
    class _FailingConnect:
        def __call__(self, url: str, *, additional_headers: dict[str, str] | None = None) -> Any:
            return self

        async def __aenter__(self) -> Any:
            raise ConnectionError('refused')

        async def __aexit__(self, *exc: object) -> bool:  # pragma: no cover — never entered
            return False

    monkeypatch.setattr(rt_openai.websockets, 'connect', _FailingConnect())
    model = OpenAIRealtimeModel('gpt-realtime')
    with pytest.raises(ConnectionError, match='refused'):
        async with _connect(model, 'x'):
            pass  # pragma: no cover


@pytest.mark.anyio
async def test_connection_iter_skips_non_string_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = json.dumps({'type': 'response.output_audio.delta', 'delta': base64.b64encode(b'\x09').decode('ascii')})
    ws = FakeWebSocket([_created(), _updated(), b'\x00binary', audio])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    model = OpenAIRealtimeModel('gpt-realtime')
    async with _connect(model, 'x') as conn:
        events = [e async for e in conn]
    assert events == [AudioDelta(data=b'\x09')]


@pytest.mark.anyio
async def test_connection_iter_recovers_from_malformed_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    # A malformed frame (invalid JSON, a valid-JSON-but-non-object frame, then a bad base64 audio
    # payload) surfaces as a recoverable SessionErrorEvent and the session keeps streaming rather than
    # tearing down. The non-object case guards against `json.loads` returning a list/str/number, which
    # would otherwise raise `AttributeError` from a later `.get()` and escape the recoverable path.
    bad_json = 'not json'
    non_object = json.dumps(['not', 'an', 'object'])
    bad_audio = json.dumps({'type': 'response.output_audio.delta', 'delta': 'not-base64!!'})
    malformed_nested_frames = [
        json.dumps({'type': 'conversation.item.input_audio_transcription.failed', 'error': 'bad'}),
        json.dumps({'type': 'response.created', 'response': 'bad'}),
        json.dumps({'type': 'response.done', 'response': 'bad'}),
        json.dumps({'type': 'response.done', 'response': {'output': 'bad'}}),
        json.dumps({'type': 'response.done', 'response': {'output': ['bad']}}),
        json.dumps({'type': 'response.done', 'response': {'usage': 'bad'}}),
        json.dumps(
            {
                'type': 'response.done',
                'response': {
                    'id': 'bad-usage',
                    'status': 'completed',
                    'output': [],
                    'usage': {'input_tokens': 1, 'input_token_details': 'bad'},
                },
            }
        ),
        json.dumps(
            {
                'type': 'response.done',
                'response': {
                    'id': 'bad-cached-usage',
                    'status': 'completed',
                    'output': [],
                    'usage': {'input_token_details': {'cached_tokens_details': 'bad'}},
                },
            }
        ),
        json.dumps(
            {
                'type': 'conversation.item.input_audio_transcription.completed',
                'transcript': 'hello',
                'usage': {'type': 'tokens', 'total_tokens': 1, 'input_token_details': 'bad'},
            }
        ),
        # A `duration` transcription usage with no numeric `seconds` (the SDK's lenient union fallback would
        # otherwise construct the wrong variant and crash on `usage.seconds`).
        json.dumps(
            {
                'type': 'conversation.item.input_audio_transcription.completed',
                'transcript': 'hi',
                'usage': {'type': 'duration'},
            }
        ),
        json.dumps(
            {
                'type': 'conversation.item.input_audio_transcription.completed',
                'transcript': 'hi',
                'usage': {'type': 'duration', 'seconds': 'nope'},
            }
        ),
        json.dumps(
            {
                'type': 'conversation.item.input_audio_transcription.completed',
                'transcript': 'hi',
                'usage': {'type': 'mystery'},
            }
        ),
    ]
    good = json.dumps({'type': 'response.output_audio.delta', 'delta': base64.b64encode(b'\x09').decode('ascii')})
    frames = [bad_json, non_object, bad_audio]
    for malformed in malformed_nested_frames:
        frames.extend((malformed, good))
    ws = FakeWebSocket([_created(), _updated(), *frames])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    model = OpenAIRealtimeModel('gpt-realtime')
    async with _connect(model, 'x') as conn:
        events = [e async for e in conn]
    assert [type(e).__name__ for e in events] == [
        'SessionErrorEvent',
        'SessionErrorEvent',
        'SessionErrorEvent',
        *['SessionErrorEvent', 'AudioDelta'] * len(malformed_nested_frames),
    ]
    errors = [event for event in events if isinstance(event, SessionErrorEvent)]
    assert len(errors) == 3 + len(malformed_nested_frames)
    assert all(event.recoverable for event in errors)
    assert events[-1] == AudioDelta(data=b'\x09')


@pytest.mark.anyio
async def test_connect_without_tools_omits_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    model = OpenAIRealtimeModel('gpt-realtime')
    async with _connect(model, 'x'):
        pass
    session = json.loads(ws.sent[0])['session']
    assert 'tools' not in session
    assert 'voice' not in session['audio']['output']


@pytest.mark.anyio
async def test_connect_seeds_message_history(monkeypatch: pytest.MonkeyPatch) -> None:
    async def download_image(*args: Any, **kwargs: Any) -> Any:
        return {'data': b'url-image', 'data_type': 'image/png'}

    monkeypatch.setattr('pydantic_ai.realtime._base.download_item', download_image)
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    history = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='sys'),
                UserPromptPart(content=['earlier question', TextContent(' with context'), CachePoint()]),
                UserPromptPart(content=[CachePoint(), '']),
                SpeechPart(speaker='user', transcript=''),
            ]
        ),
        ModelResponse(
            parts=[
                ThinkingPart(
                    content='reasoning',
                    signature='session-bound',
                    provider_name='openai',
                    provider_details={'encrypted_content': 'secret'},
                ),
                ThinkingPart(content='', signature='signature-only', provider_name='openai'),
                TextPart(content=''),
                TextPart(content='earlier answer'),
                SpeechPart(speaker='assistant', transcript=''),
                NativeToolCallPart(tool_name='web_search', args={}, tool_call_id='native-call'),
                NativeToolReturnPart(tool_name='web_search', content='native metadata', tool_call_id='native-call'),
                ToolCallPart(tool_name='weather', args={'city': 'Paris'}, tool_call_id='call-1'),
                ToolCallPart(tool_name='lookup', args='{"id":1}', tool_call_id='call-2'),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='weather', content='sunny', tool_call_id='call-1'),
                RetryPromptPart(tool_name='lookup', content='invalid id', tool_call_id='call-2'),
                RetryPromptPart(content='answer in prose'),
                UserPromptPart(
                    content=[
                        ImageUrl(url='https://example.com/a.png'),
                        BinaryContent(data=b'inline-image', media_type='image/png'),
                    ]
                ),
                SpeechPart(speaker='user', transcript='spoken question'),
            ]
        ),
        ModelResponse(parts=[SpeechPart(speaker='assistant', transcript='spoken answer')]),
    ]
    model = OpenAIRealtimeModel('gpt-realtime')
    async with _connect(model, 'x', messages=history):
        pass

    items = [json.loads(frame) for frame in ws.sent[1:]]
    assert items == snapshot(
        [
            {
                'type': 'conversation.item.create',
                'item': {
                    'type': 'message',
                    'role': 'user',
                    'content': [
                        {'type': 'input_text', 'text': 'earlier question'},
                        {'type': 'input_text', 'text': ' with context'},
                    ],
                },
            },
            {
                'type': 'conversation.item.create',
                'item': {
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'output_text', 'text': '<think>\nreasoning\n</think>'}],
                },
            },
            {
                'type': 'conversation.item.create',
                'item': {
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'output_text', 'text': 'earlier answer'}],
                },
            },
            {
                'type': 'conversation.item.create',
                'item': {
                    'type': 'function_call',
                    'name': 'weather',
                    'call_id': 'call-1',
                    'arguments': '{"city":"Paris"}',
                },
            },
            {
                'type': 'conversation.item.create',
                'item': {
                    'type': 'function_call',
                    'name': 'lookup',
                    'call_id': 'call-2',
                    'arguments': '{"id":1}',
                },
            },
            {
                'type': 'conversation.item.create',
                'item': {'type': 'function_call_output', 'call_id': 'call-1', 'output': 'sunny'},
            },
            {
                'type': 'conversation.item.create',
                'item': {
                    'type': 'function_call_output',
                    'call_id': 'call-2',
                    'output': 'invalid id\n\nFix the errors and try again.',
                },
            },
            {
                'type': 'conversation.item.create',
                'item': {
                    'type': 'message',
                    'role': 'user',
                    'content': [
                        {
                            'type': 'input_text',
                            'text': 'Validation feedback:\nanswer in prose\n\nFix the errors and try again.',
                        }
                    ],
                },
            },
            {
                'type': 'conversation.item.create',
                'item': {
                    'type': 'message',
                    'role': 'user',
                    'content': [
                        {'type': 'input_image', 'image_url': 'data:image/png;base64,dXJsLWltYWdl'},
                        {'type': 'input_image', 'image_url': 'data:image/png;base64,aW5saW5lLWltYWdl'},
                    ],
                },
            },
            {
                'type': 'conversation.item.create',
                'item': {
                    'type': 'message',
                    'role': 'user',
                    'content': [{'type': 'input_text', 'text': 'spoken question'}],
                },
            },
            {
                'type': 'conversation.item.create',
                'item': {
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'output_text', 'text': 'spoken answer'}],
                },
            },
        ]
    )
    assert 'session-bound' not in json.dumps(items)
    assert 'encrypted_content' not in json.dumps(items)


async def test_connect_seeds_multimodal_user_prompt_as_native_image(monkeypatch: pytest.MonkeyPatch) -> None:
    async def download_image(*args: Any, **kwargs: Any) -> Any:
        return {'data': b'png', 'data_type': 'image/png'}

    monkeypatch.setattr('pydantic_ai.realtime._base.download_item', download_image)
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    history = [
        ModelRequest(parts=[UserPromptPart(content=[ImageUrl(url='https://example.com/a.png'), 'describe this'])])
    ]
    model = OpenAIRealtimeModel('gpt-realtime')
    async with _connect(model, 'x', messages=history):
        pass
    items = [json.loads(frame) for frame in ws.sent[1:]]
    assert items[0]['item']['content'] == [
        {'type': 'input_image', 'image_url': 'data:image/png;base64,cG5n'},
        {'type': 'input_text', 'text': 'describe this'},
    ]


@pytest.mark.anyio
async def test_connect_seeds_multimodal_tool_return(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    history = [
        ModelResponse(parts=[ToolCallPart(tool_name='inspect', args={}, tool_call_id='call-image')]),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='inspect',
                    tool_call_id='call-image',
                    content=[
                        'done',
                        BinaryContent(data=b'result-image', media_type='image/png', identifier='result.png'),
                    ],
                )
            ]
        ),
    ]

    async with _connect(OpenAIRealtimeModel('gpt-realtime'), 'x', messages=history):
        pass

    assert [json.loads(frame) for frame in ws.sent[1:]] == snapshot(
        [
            {
                'type': 'conversation.item.create',
                'item': {
                    'type': 'function_call',
                    'name': 'inspect',
                    'call_id': 'call-image',
                    'arguments': '{}',
                },
            },
            {
                'type': 'conversation.item.create',
                'item': {
                    'type': 'function_call_output',
                    'call_id': 'call-image',
                    'output': '["done","See file result.png."]',
                },
            },
            {
                'type': 'conversation.item.create',
                'item': {
                    'type': 'message',
                    'role': 'user',
                    'content': [
                        {'type': 'input_text', 'text': 'This is file result.png:'},
                        {'type': 'input_image', 'image_url': 'data:image/png;base64,cmVzdWx0LWltYWdl'},
                    ],
                },
            },
        ]
    )


@pytest.mark.anyio
async def test_connect_remaps_long_tool_call_id_and_keeps_pending_call(monkeypatch: pytest.MonkeyPatch) -> None:
    long_id = 'pyd_ai_0123456789abcdef0123456789abcdef'
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    history = [
        ModelResponse(
            parts=[
                ToolCallPart(tool_name='done', args={}, tool_call_id=long_id),
                ToolCallPart(tool_name='pending', args={}, tool_call_id='pending-call'),
            ]
        ),
        ModelRequest(parts=[ToolReturnPart(tool_name='done', content='ok', tool_call_id=long_id)]),
    ]

    async with _connect(OpenAIRealtimeModel('gpt-realtime'), 'x', messages=history):
        pass

    items = [json.loads(frame)['item'] for frame in ws.sent[1:]]
    assert items == snapshot(
        [
            {
                'type': 'function_call',
                'name': 'done',
                'call_id': 'dc48ed0580f3898b7fe60753ced829ff',
                'arguments': '{}',
            },
            {
                'type': 'function_call',
                'name': 'pending',
                'call_id': 'pending-call',
                'arguments': '{}',
            },
            {
                'type': 'function_call_output',
                'call_id': 'dc48ed0580f3898b7fe60753ced829ff',
                'output': 'ok',
            },
        ]
    )


@pytest.mark.anyio
async def test_connect_rejects_orphan_tool_return(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    history = [ModelRequest(parts=[ToolReturnPart(tool_name='weather', content='sunny', tool_call_id='missing')])]

    with pytest.raises(UserError, match=r"tool 'weather' with call ID 'missing'.*no preceding `ToolCallPart`"):
        async with _connect(OpenAIRealtimeModel('gpt-realtime'), 'x', messages=history):
            pass  # pragma: no cover


@pytest.mark.anyio
async def test_connect_seeds_retained_user_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    history = [
        ModelRequest(
            parts=[
                SpeechPart(
                    speaker='user',
                    audio=BinaryContent(data=_wav_bytes(b'pcm-audio!'), media_type='audio/wav'),
                )
            ]
        )
    ]

    async with _connect(OpenAIRealtimeModel('gpt-realtime'), 'x', messages=history):
        pass

    assert json.loads(ws.sent[1])['item'] == {
        'type': 'message',
        'role': 'user',
        'content': [{'type': 'input_audio', 'audio': 'cGNtLWF1ZGlvIQ=='}],
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    ('audio', 'match'),
    [
        (BinaryContent(data=_wav_bytes(b'pcm-audio!', 16000), media_type='audio/wav'), 'recorded at 16000 Hz'),
        (BinaryContent(data=b'pcm-audio', media_type='audio/pcm'), "media type 'audio/pcm'"),
        (BinaryContent(data=b'not a wav', media_type='audio/wav'), 'not valid WAV audio'),
    ],
)
async def test_connect_rejects_retained_audio_incompatible_with_input_format(
    monkeypatch: pytest.MonkeyPatch,
    audio: BinaryContent,
    match: str,
) -> None:
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    history = [ModelRequest(parts=[SpeechPart(speaker='user', audio=audio)])]

    with pytest.raises(UserError, match=re.escape(match)):
        async with _connect(OpenAIRealtimeModel('gpt-realtime'), 'x', messages=history):
            pass  # pragma: no cover


@pytest.mark.anyio
async def test_connect_rejects_non_mono_retained_wav(monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b'\x00' * 8)
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    history = [
        ModelRequest(
            parts=[SpeechPart(speaker='user', audio=BinaryContent(data=buffer.getvalue(), media_type='audio/wav'))]
        )
    ]

    with pytest.raises(UserError, match='expected mono 16-bit PCM WAV'):
        async with _connect(OpenAIRealtimeModel('gpt-realtime'), 'x', messages=history):
            pass  # pragma: no cover


@pytest.mark.anyio
@pytest.mark.parametrize('content_kind', ['audio-url', 'video-url', 'document-url', 'binary', 'uploaded'])
async def test_connect_rejects_unseedable_user_content(monkeypatch: pytest.MonkeyPatch, content_kind: str) -> None:
    content = {
        'audio-url': AudioUrl(url='https://example.com/a.mp3'),
        'video-url': VideoUrl(url='https://example.com/a.mp4'),
        'document-url': DocumentUrl(url='https://example.com/a.pdf'),
        'binary': BinaryContent(data=b'pdf', media_type='application/pdf'),
        'uploaded': UploadedFile(file_id='file-1', provider_name='openai', media_type='application/pdf'),
    }[content_kind]
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    history = [ModelRequest(parts=[UserPromptPart(content=[content])])]

    with pytest.raises(UserError, match='cannot be seeded into openai realtime history'):
        async with _connect(OpenAIRealtimeModel('gpt-realtime'), 'x', messages=history):
            pass  # pragma: no cover


@pytest.mark.anyio
async def test_connect_rejects_image_url_returning_non_image(monkeypatch: pytest.MonkeyPatch) -> None:
    async def download_document(*args: Any, **kwargs: Any) -> Any:
        return {'data': b'not-image', 'data_type': 'application/pdf'}

    monkeypatch.setattr('pydantic_ai.realtime._base.download_item', download_document)
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    history = [ModelRequest(parts=[UserPromptPart(content=[ImageUrl(url='https://example.com/a.png')])])]

    with pytest.raises(UserError, match='`ImageUrl` resolved to unsupported media type'):
        async with _connect(OpenAIRealtimeModel('gpt-realtime'), 'x', messages=history):
            pass  # pragma: no cover


@pytest.mark.anyio
async def test_connect_rejects_unseedable_speech_and_response_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    histories = [
        (
            [ModelRequest(parts=[SpeechPart(speaker='user')])],
            'without a transcript or retained audio',
        ),
        (
            [
                ModelRequest(
                    parts=[
                        SpeechPart(
                            speaker='user',
                            audio=BinaryContent(data=b'not-audio', media_type='application/pdf'),
                        )
                    ]
                )
            ],
            '`SpeechPart.audio` with media type',
        ),
        (
            [
                ModelResponse(
                    parts=[
                        SpeechPart(
                            speaker='assistant',
                            audio=BinaryContent(data=b'audio', media_type='audio/pcm'),
                        )
                    ]
                )
            ],
            'assistant `SpeechPart` without a transcript',
        ),
        (
            [ModelResponse(parts=[FilePart(content=BinaryContent(data=b'file', media_type='application/pdf'))])],
            '`FilePart`',
        ),
    ]
    for history, match in histories:
        ws = FakeWebSocket([_created(), _updated()])
        monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
        with pytest.raises(UserError, match=re.escape(match)):
            async with _connect(OpenAIRealtimeModel('gpt-realtime'), 'x', messages=history):
                pass  # pragma: no cover


@pytest.mark.anyio
async def test_connect_captures_server_reported_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # `session.created` reports the model actually serving the session; the connection captures it so
    # the session can stamp it on `ModelResponse.model_name` (it can differ from the requested id).
    created = json.dumps({'type': 'session.created', 'session': {'model': 'gpt-realtime-2025-06-03'}})
    ws = FakeWebSocket([created, _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    async with _connect(OpenAIRealtimeModel('gpt-realtime'), 'x') as conn:
        assert conn.model_name == 'gpt-realtime-2025-06-03'


@pytest.mark.anyio
async def test_connect_without_server_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # A handshake that doesn't report a model (like these bare test frames) leaves `model_name` unset,
    # so the session falls back to the configured id.
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    async with _connect(OpenAIRealtimeModel('gpt-realtime'), 'x') as conn:
        assert conn.model_name is None


@pytest.mark.anyio
async def test_connect_seed_skips_compaction_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    # Provider-session-bound compaction state can't round-trip into another session; like the classic
    # model adapters crossing APIs, seeding skips it silently rather than erroring.
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    history = [ModelResponse(parts=[CompactionPart(content='summary'), TextPart(content='the answer')])]
    async with _connect(OpenAIRealtimeModel('gpt-realtime'), 'x', messages=history):
        pass
    items = [json.loads(frame)['item'] for frame in ws.sent[1:]]
    assert [c['text'] for item in items for c in item['content']] == ['the answer']


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
async def test_connection_send_tool_result_with_follow_up_user_content() -> None:
    ws = FakeWebSocket([])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    await conn.send(
        ToolResult(
            tool_call_id='call_1',
            output='See file result.png.',
            content=[
                'This is file result.png:',
                BinaryContent(data=b'png', media_type='image/png'),
                'extra context',
            ],
        )
    )
    assert [json.loads(frame) for frame in ws.sent] == [
        {
            'type': 'conversation.item.create',
            'item': {'type': 'function_call_output', 'call_id': 'call_1', 'output': 'See file result.png.'},
        },
        {
            'type': 'conversation.item.create',
            'item': {
                'type': 'message',
                'role': 'user',
                'content': [
                    {'type': 'input_text', 'text': 'This is file result.png:'},
                    {'type': 'input_image', 'image_url': 'data:image/png;base64,cG5n'},
                    {'type': 'input_text', 'text': 'extra context'},
                ],
            },
        },
        {'type': 'response.create'},
    ]


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
async def test_connection_drops_deltas_from_a_cancelled_response() -> None:
    # After a barge-in cancel, the server keeps streaming the cancelled response's trailing audio and
    # transcript deltas before its `response.done`. Those must be dropped (the user already interrupted
    # that speech), while the cancelled response's own `response.done` still closes the turn, and a fresh
    # response that follows streams normally.
    audio_straggler = json.dumps(
        {
            'type': 'response.output_audio.delta',
            'response_id': 'resp-1',
            'item_id': 'item-1',
            'delta': base64.b64encode(b'\x01').decode('ascii'),
        }
    )
    transcript_straggler = json.dumps(
        {'type': 'response.output_audio_transcript.delta', 'response_id': 'resp-1', 'item_id': 'item-1', 'delta': 'no'}
    )
    cancelled_done = json.dumps({'type': 'response.done', 'response': {'id': 'resp-1', 'status': 'cancelled'}})
    new_created = json.dumps({'type': 'response.created', 'response': {'id': 'resp-2'}})
    new_audio = json.dumps(
        {
            'type': 'response.output_audio.delta',
            'response_id': 'resp-2',
            'item_id': 'item-2',
            'delta': base64.b64encode(b'\x02').decode('ascii'),
        }
    )
    ws = FakeWebSocket([audio_straggler, transcript_straggler, cancelled_done, new_created, new_audio])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    conn._response_active = True  # pyright: ignore[reportPrivateUsage]
    conn._active_response_id = 'resp-1'  # pyright: ignore[reportPrivateUsage]
    await conn.send(CancelResponse())  # cancels resp-1 and starts suppressing its stragglers

    events = [event async for event in conn]
    assert events == [
        TurnCompleteEvent(
            interrupted=True,
            provider_response_id='resp-1',
            finish_reason=None,
            provider_details={'status': 'cancelled'},
        ),
        AudioDelta(data=b'\x02', item_id='item-2'),  # the next response is unaffected
    ]
    assert conn._cancelled_response_id is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.anyio
async def test_superseded_cancelled_response_done_suppresses_turn_complete() -> None:
    # A barge-in cancels response A; a new response B then becomes active before A's late `response.done`
    # arrives. A's usage is still accounted, but its `TurnCompleteEvent` must be suppressed — otherwise the
    # session would finalize B's in-flight output under A's (interrupted) boundary.
    created_b = json.dumps({'type': 'response.created', 'response': {'id': 'B'}})
    late_a_done = json.dumps(
        {'type': 'response.done', 'response': {'id': 'A', 'status': 'cancelled', 'usage': {'input_tokens': 1}}}
    )
    b_audio = json.dumps(
        {
            'type': 'response.output_audio.delta',
            'response_id': 'B',
            'item_id': 'b-item',
            'delta': base64.b64encode(b'\x02').decode('ascii'),
        }
    )
    ws = FakeWebSocket([created_b, late_a_done, b_audio])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    conn._response_active = True  # pyright: ignore[reportPrivateUsage]
    conn._active_response_id = 'A'  # pyright: ignore[reportPrivateUsage]
    await conn.send(CancelResponse())  # cancel A (barge-in); B is created below and becomes active

    events = [event async for event in conn]
    # A's usage is recorded, B keeps streaming, and no `TurnCompleteEvent` fired for the superseded A.
    assert [type(event).__name__ for event in events] == ['SessionUsageEvent', 'AudioDelta']
    assert isinstance(events[0], SessionUsageEvent) and events[0].provider_response_id == 'A'
    assert events[1] == AudioDelta(data=b'\x02', item_id='b-item')
    assert not any(isinstance(event, TurnCompleteEvent) for event in events)


@pytest.mark.anyio
async def test_response_done_without_response_object_is_recoverable() -> None:
    # A malformed `response.done` with no `response` object must not raise `AttributeError` (escaping the
    # recoverable path); it degrades to a graceful `TurnCompleteEvent` with an unknown status.
    ws = FakeWebSocket([json.dumps({'type': 'response.done'})])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    events = [event async for event in conn]
    assert events == [TurnCompleteEvent(interrupted=False, provider_details={'status': None})]


@pytest.mark.anyio
async def test_transcription_completed_token_usage_emits_run_level_usage() -> None:
    # A final input transcription with well-formed token usage yields the transcript plus a run-level
    # (non-response-scoped) ASR usage event with the per-modality token breakdown in `details`.
    frame = json.dumps(
        {
            'type': 'conversation.item.input_audio_transcription.completed',
            'item_id': 'u1',
            'transcript': 'hi',
            'usage': {
                'type': 'tokens',
                'total_tokens': 5,
                'input_token_details': {'audio_tokens': 4, 'text_tokens': 1},
            },
        }
    )
    ws = FakeWebSocket([frame])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    events = [event async for event in conn]
    assert events == [
        InputTranscript(text='hi', is_final=True, item_id='u1'),
        SessionUsageEvent(
            usage=RequestUsage(
                details={
                    'input_transcription_tokens': 5,
                    'input_transcription_audio_tokens': 4,
                    'input_transcription_text_tokens': 1,
                }
            ),
            response_scoped=False,
        ),
    ]


@pytest.mark.anyio
async def test_response_done_emits_usage_then_turn_complete() -> None:
    done = json.dumps(
        {
            'type': 'response.done',
            'response': {
                'id': 'resp-1',
                'status': 'completed',
                'output': [],
                'usage': {'input_tokens': 3, 'output_tokens': 2},
            },
        }
    )
    ws = FakeWebSocket([done])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    events = [e async for e in conn]
    assert events == [
        SessionUsageEvent(
            usage=RequestUsage(input_tokens=3, output_tokens=2),
            provider_response_id='resp-1',
            finish_reason='stop',
        ),
        TurnCompleteEvent(
            interrupted=False,
            provider_response_id='resp-1',
            finish_reason='stop',
            provider_details={'status': 'completed'},
        ),
    ]


@pytest.mark.anyio
async def test_response_done_maps_xai_top_level_usage_extras() -> None:
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
    conn = OpenAIRealtimeConnection(FakeWebSocket([done]))  # type: ignore[arg-type]
    events = [event async for event in conn]
    assert events[0] == SessionUsageEvent(
        usage=RequestUsage(
            input_tokens=8,
            output_tokens=5,
            input_audio_tokens=6,
            output_audio_tokens=4,
            details={'input_grok_tokens': 2, 'output_grok_tokens': 1, 'billable_audio_seconds': 3},
        ),
        provider_response_id='resp-xai',
        finish_reason='stop',
    )


@pytest.mark.anyio
async def test_response_done_function_call_only_still_emits_usage() -> None:
    done = json.dumps(
        {
            'type': 'response.done',
            'response': {
                'id': 'resp-tool',
                'status': 'completed',
                'output': [{'type': 'function_call'}],
                'usage': {'output_tokens': 5},
            },
        }
    )
    ws = FakeWebSocket([done])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    events = [e async for e in conn]
    # function-call-only → no TurnCompleteEvent, but usage is still surfaced
    assert events == [
        SessionUsageEvent(
            usage=RequestUsage(output_tokens=5),
            provider_response_id='resp-tool',
            finish_reason='tool_call',
        )
    ]


@pytest.mark.anyio
async def test_function_call_only_response_without_usage_finalizes_before_answer() -> None:
    frames = [
        json.dumps({'type': 'response.created', 'response': {'id': 'resp-tool'}}),
        json.dumps(
            {
                'type': 'response.function_call_arguments.done',
                'call_id': 'call-1',
                'name': 'get_weather',
                'arguments': '{}',
            }
        ),
        json.dumps(
            {
                'type': 'response.done',
                'response': {
                    'id': 'resp-tool',
                    'status': 'completed',
                    'output': [{'type': 'function_call'}],
                },
            }
        ),
        json.dumps({'type': 'response.created', 'response': {'id': 'resp-answer'}}),
        json.dumps(
            {
                'type': 'response.output_audio_transcript.done',
                'item_id': 'answer-1',
                'transcript': 'Sunny',
            }
        ),
        json.dumps(
            {
                'type': 'response.done',
                'response': {
                    'id': 'resp-answer',
                    'status': 'completed',
                    'output': [],
                    'usage': {'output_tokens': 3},
                },
            }
        ),
    ]

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        return 'sunny'

    connection = OpenAIRealtimeConnection(FakeWebSocket(frames))  # type: ignore[arg-type]
    session = RealtimeSession(connection, make_tool_manager(runner), provider_name='openai')
    async with session:
        _ = [event async for event in session]

    messages = session.all_messages()
    assert len(messages) == 3
    tool_response, tool_result, answer = messages
    assert isinstance(tool_response, ModelResponse)
    assert tool_response.parts == [ToolCallPart(tool_name='get_weather', args='{}', tool_call_id='call-1')]
    assert tool_response.usage == RequestUsage()
    assert isinstance(tool_result, ModelRequest)
    assert isinstance(tool_result.parts[0], ToolReturnPart)
    assert isinstance(answer, ModelResponse)
    assert answer.parts == [SpeechPart(speaker='assistant', transcript='Sunny', id='answer-1', provider_name='openai')]
    assert answer.usage == RequestUsage(output_tokens=3)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ('status', 'raw_reason', 'finish_reason', 'state'),
    # A cancelled (barge-in) turn is interrupted, not an error, so `finish_reason` stays unset.
    [
        ('completed', None, 'stop', 'complete'),
        ('cancelled', None, None, 'interrupted'),
        ('incomplete', 'max_output_tokens', 'length', 'complete'),
        ('incomplete', 'content_filter', 'content_filter', 'complete'),
    ],
)
async def test_session_stamps_openai_response_metadata(
    status: str, raw_reason: str | None, finish_reason: FinishReason | None, state: str
) -> None:
    transcript = json.dumps(
        {
            'type': 'response.output_audio_transcript.done',
            'item_id': 'item-1',
            'transcript': 'hello',
        }
    )
    response_data: dict[str, Any] = {'id': 'resp-1', 'status': status, 'output': []}
    if raw_reason is not None:
        response_data['status_details'] = {'reason': raw_reason}
    done = json.dumps({'type': 'response.done', 'response': response_data})
    connection = OpenAIRealtimeConnection(FakeWebSocket([transcript, done]))  # type: ignore[arg-type]
    session = RealtimeSession(
        connection,
        make_tool_manager(),
        model_name='gpt-realtime',
        provider_name='openai',
    )
    async with session:
        _ = [event async for event in session]

    response = next(message for message in session.new_messages() if isinstance(message, ModelResponse))
    assert response.provider_name == 'openai'
    assert response.provider_response_id == 'resp-1'
    assert response.finish_reason == finish_reason
    assert response.state == state
    expected_details: dict[str, Any] = {'status': status}
    if raw_reason is not None:
        expected_details['finish_reason'] = raw_reason
    assert response.provider_details == expected_details
    speech = response.parts[0]
    assert isinstance(speech, SpeechPart)
    assert (speech.id, speech.provider_name) == ('item-1', 'openai')


@pytest.mark.anyio
@pytest.mark.parametrize(
    ('status', 'raw_reason', 'finish_reason', 'state'),
    [
        ('incomplete', 'max_output_tokens', 'length', 'complete'),
        ('failed', None, 'error', 'complete'),
        ('cancelled', None, None, 'interrupted'),
    ],
)
async def test_session_records_empty_openai_response(
    status: str, raw_reason: str | None, finish_reason: FinishReason | None, state: str
) -> None:
    response_data: dict[str, Any] = {'id': 'resp-empty', 'status': status, 'output': []}
    if raw_reason is not None:
        response_data['status_details'] = {'reason': raw_reason}
    done = json.dumps({'type': 'response.done', 'response': response_data})
    connection = OpenAIRealtimeConnection(FakeWebSocket([done]))  # type: ignore[arg-type]
    session = RealtimeSession(connection, make_tool_manager(), provider_name='openai')

    async with session:
        _ = [event async for event in session]

    response = session.new_messages()[0]
    assert isinstance(response, ModelResponse)
    assert response.parts == []
    assert response.provider_name == 'openai'
    assert response.provider_response_id == 'resp-empty'
    assert response.finish_reason == finish_reason
    assert response.provider_details == {
        'status': status,
        **({'finish_reason': raw_reason} if raw_reason is not None else {}),
    }
    assert response.state == state


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
    assert isinstance(error, SessionErrorEvent)
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
    assert events == [ReconnectedEvent(state_restored=False), Transcript(text='hi', is_final=True)]


class _DropAfterHandshake(FakeWebSocket):
    """Completes the handshake (via `recv`), then drops when iterated."""

    async def __aiter__(self) -> AsyncIterator[Any]:
        raise rt_openai.websockets.ConnectionClosed(None, None)
        yield  # pragma: no cover  (makes this an async generator)


class _RecordingConnect:
    """Stand-in for `websockets.connect` that hands out sockets in order and records closes."""

    def __init__(self, sockets: list[FakeWebSocket]) -> None:
        self._sockets = iter(sockets)
        self.closed: list[FakeWebSocket] = []

    def __call__(self, url: str, *, additional_headers: dict[str, str] | None = None) -> Any:
        ws = next(self._sockets)
        recorder = self

        class _CM:
            async def __aenter__(self) -> FakeWebSocket:
                return ws

            async def __aexit__(self, *exc: object) -> bool:
                recorder.closed.append(ws)
                return False

        return _CM()


@pytest.mark.anyio
async def test_connect_reconnect_closes_previous_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    # A reconnect through `connect()`'s own dial must close the dropped connection before opening the
    # next, and teardown closes the current one — so sockets don't accumulate across drops.
    transcript = json.dumps({'type': 'response.audio_transcript.done', 'transcript': 'hi'})
    dropped = _DropAfterHandshake([_created(), _updated()])
    good = FakeWebSocket([_created(), _updated(), transcript])
    connect = _RecordingConnect([dropped, good])
    monkeypatch.setattr(rt_openai.websockets, 'connect', connect)

    model = OpenAIRealtimeModel('gpt-realtime', reconnect=rt_openai.ReconnectPolicy(base_delay=0.0))
    async with _connect(model, 'x') as conn:
        events = [e async for e in conn]

    assert events == [ReconnectedEvent(state_restored=False), Transcript(text='hi', is_final=True)]
    assert connect.closed == [dropped, good]


@pytest.mark.anyio
async def test_reconnect_gives_up_after_max_attempts() -> None:
    async def dial() -> Any:
        raise OSError('still down')  # an expected dial failure (network unreachable)

    conn = OpenAIRealtimeConnection(
        DroppingWebSocket([]),  # type: ignore[arg-type]
        dial=dial,
        reconnect=rt_openai.ReconnectPolicy(max_attempts=2, base_delay=0.0, jitter=False),
    )
    events = [e async for e in conn]
    assert len(events) == 1
    error = events[0]
    assert isinstance(error, SessionErrorEvent)
    assert error.recoverable is False
    assert 'reconnect failed' in error.message


@pytest.mark.anyio
async def test_reconnect_propagates_unexpected_dial_error() -> None:
    # An unexpected error while re-dialing (a bug, not a network/protocol failure) propagates instead
    # of being swallowed as a failed reconnect, so it surfaces rather than looking like the server went
    # away.
    async def dial() -> Any:
        raise RuntimeError('boom')

    conn = OpenAIRealtimeConnection(
        DroppingWebSocket([]),  # type: ignore[arg-type]
        dial=dial,
        reconnect=rt_openai.ReconnectPolicy(base_delay=0.0),
    )
    with pytest.raises(RuntimeError, match='boom'):
        _ = [e async for e in conn]


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


@pytest.mark.anyio
async def test_cancel_clears_tracked_item_so_later_truncate_is_noop() -> None:
    # A client-driven `CancelResponse` forgets the cancelled response's output item, so a second
    # `interrupt(audio_end_ms=...)` before the next turn's first audio delta doesn't truncate a stale item.
    ws = FakeWebSocket([_audio_delta('item_5')])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    conn._response_active = True  # pyright: ignore[reportPrivateUsage]
    _ = [e async for e in conn]  # the delta sets the current output item
    await conn.send(CancelResponse())
    await conn.send(TruncateOutput(audio_end_ms=500))
    assert json.loads(ws.sent[0]) == {'type': 'response.cancel'}
    assert len(ws.sent) == 1  # no truncate for the cleared, cancelled item


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
async def test_late_cancelled_response_done_does_not_clear_new_response() -> None:
    ws = PushWebSocket()
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    task = asyncio.create_task(_drain(conn))

    ws.push({'type': 'response.created', 'response': {'id': 'resp_old'}})
    await _settle()
    await conn.send(CancelResponse())
    await conn.send(TextInput(text='new turn'))
    ws.push({'type': 'response.created', 'response': {'id': 'resp_new'}})
    await _settle()

    ws.push({'type': 'response.done', 'response': {'id': 'resp_old', 'status': 'cancelled', 'output': []}})
    await _settle()
    await conn.send(CreateResponse())
    assert ws.sent_types().count('response.create') == 1

    ws.push({'type': 'response.done', 'response': {'id': 'resp_new', 'status': 'completed', 'output': []}})
    await _settle()
    assert ws.sent_types().count('response.create') == 2

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
    model = OpenAIRealtimeModel('gpt-realtime')
    tools = [ToolDefinition(name='ping', parameters_json_schema={'type': 'object'})]
    async with _connect(model, 'x', tools=tools):
        pass
    tool = json.loads(ws.sent[0])['session']['tools'][0]
    assert tool == {'type': 'function', 'name': 'ping', 'parameters': {'type': 'object'}}


@pytest.mark.anyio
async def test_connect_without_transcription_model_omits_transcription(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    model = OpenAIRealtimeModel(
        'gpt-realtime', settings=rt_openai.OpenAIRealtimeModelSettings(input_transcription_model=None)
    )
    async with _connect(model, 'x') as conn:
        # A disabled transcription model reports `input_transcription_enabled=False`, so the session
        # finalizes user turns from retained audio instead of waiting for transcripts that never arrive.
        assert conn.input_transcription_enabled is False
    assert 'transcription' not in json.loads(ws.sent[0])['session']['audio']['input']


@pytest.mark.anyio
async def test_connect_transcription_model_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    # An explicit id is used verbatim, overriding the `'auto'` default; the connection reports transcription on.
    model = OpenAIRealtimeModel(
        'gpt-realtime',
        settings=rt_openai.OpenAIRealtimeModelSettings(input_transcription_model='gpt-4o-transcribe'),
    )
    async with _connect(model, 'x') as conn:
        assert conn.input_transcription_enabled is True
    assert json.loads(ws.sent[0])['session']['audio']['input']['transcription'] == {'model': 'gpt-4o-transcribe'}


@pytest.mark.anyio
async def test_connect_applies_max_tokens_without_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    model = OpenAIRealtimeModel('gpt-realtime')
    async with _connect(model, 'x', model_settings=RealtimeModelSettings(max_tokens=256)):
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
    model = OpenAIRealtimeModel('gpt-realtime')
    async with _connect(model, 'x') as conn:
        events = [e async for e in conn]
    assert events == [
        TurnCompleteEvent(
            interrupted=False,
            finish_reason='stop',
            provider_details={'status': 'completed'},
        )
    ]


async def test_agent_realtime_session_rejects_native_tools() -> None:
    # OpenAI realtime supports no native tools, so any native tool fails up front with the uniform
    # error, before dialing — the check lives in `Agent.realtime_session`, keyed on the model profile.
    agent: Agent[None, str] = Agent()
    with pytest.raises(
        UserError,
        match=r'does not support the WebSearchTool native tool\(s\)\. Supported native tools: none\.',
    ):
        async with agent.realtime_session(
            model=OpenAIRealtimeModel('gpt-realtime'), capabilities=[NativeTool(WebSearchTool())]
        ):
            pass  # pragma: no cover - validation raises before yielding


# --- provider resolution & capabilities -------------------------------------------------------


def test_realtime_websocket_url_derivation() -> None:
    # The default OpenAI HTTP base URL maps to the documented realtime WebSocket URL.
    assert realtime_websocket_url('https://api.openai.com/v1/') == 'wss://api.openai.com/v1/realtime'
    # A custom (e.g. self-hosted, non-TLS) base URL keeps its host/path and swaps the scheme.
    assert realtime_websocket_url('http://localhost:8000/v1') == 'ws://localhost:8000/v1/realtime'
    # A base URL with neither scheme is left untouched apart from the appended path.
    assert realtime_websocket_url('localhost:8000/v1') == 'localhost:8000/v1/realtime'


def test_default_provider_is_openai() -> None:
    model = OpenAIRealtimeModel('gpt-realtime')
    assert model.client.api_key == 'mock-api-key'  # from the autouse OPENAI_API_KEY fixture


def test_provider_instance_is_reused() -> None:
    provider = OpenAIProvider(api_key='k')
    model = OpenAIRealtimeModel('gpt-realtime', provider=provider)
    assert model.client is provider.client


@pytest.mark.anyio
async def test_custom_provider_base_url_derives_websocket_url(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeWebSocket([_created(), _updated()])
    fake_connect = FakeConnect(ws)
    monkeypatch.setattr(rt_openai.websockets, 'connect', fake_connect)
    model = OpenAIRealtimeModel(
        'gpt-realtime', provider=OpenAIProvider(base_url='https://proxy.example/v1', api_key='k')
    )
    async with _connect(model, 'x'):
        pass
    assert fake_connect.url == 'wss://proxy.example/v1/realtime?model=gpt-realtime'
    assert fake_connect.headers == {'Authorization': 'Bearer k'}


def test_azure_provider_is_rejected() -> None:
    from pydantic_ai.providers.azure import AzureProvider

    provider = AzureProvider(azure_endpoint='https://res.openai.azure.com/openai/v1/', api_key='k')
    with pytest.raises(UserError, match='Azure OpenAI is not supported'):
        OpenAIRealtimeModel('gpt-realtime', provider=provider)


def test_profile() -> None:
    profile = OpenAIRealtimeModel('gpt-realtime').profile
    assert (
        profile.get('supports_image_input'),
        profile.get('supports_manual_turn_control'),
        profile.get('supports_interruption'),
        profile.get('supports_output_truncation'),
        profile.get('supports_session_seeding'),
        profile.get('supports_seeding_images'),
        profile.get('supports_seeding_audio'),
    ) == (True, True, True, True, True, True, True)
    assert profile.get('supported_native_tools') == frozenset()
    assert profile.get('audio_input_sample_rate') == 24000
    assert profile.get('audio_output_sample_rate') == 24000


def test_provider_driven_profile_merges_defaults_varies_by_model_and_intersects_native_tools() -> None:
    class ProfileProvider(OpenAIProvider):
        @staticmethod
        def realtime_model_profile(model_name: str) -> RealtimeModelProfile:
            return RealtimeModelProfile(
                supports_image_input=model_name == 'image-model',
                supported_native_tools=frozenset({WebSearchTool}),
            )

    provider = ProfileProvider(api_key='k')
    image_profile = OpenAIRealtimeModel('image-model', provider=provider).profile
    text_profile = OpenAIRealtimeModel('text-model', provider=provider).profile

    assert image_profile.get('supports_image_input') is True
    assert text_profile.get('supports_image_input') is False
    assert image_profile.get('supports_interruption') is False  # merged from the default
    assert image_profile.get('supported_native_tools') == frozenset()  # model class implements none
