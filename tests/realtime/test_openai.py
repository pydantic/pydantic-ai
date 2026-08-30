"""Tests for the OpenAI realtime provider (event mapping, handshake, send), all network-free."""

from __future__ import annotations as _annotations

import asyncio
import base64
import hashlib
import io
import json
import re
import wave
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, Literal, cast, get_args, get_origin

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UserError
from pydantic_ai.messages import (
    AudioUrl,
    BinaryAudio,
    BinaryContent,
    BinaryImage,
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
    RealtimeSessionErrorEvent,
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
    RealtimeInputSpeechEndEvent,
    RealtimeInputSpeechStartEvent,
    RealtimeInputTranscriptionErrorEvent,
    RealtimeModelProfile,
    RealtimeModelSettings,
    RealtimeOutputSpeechEndEvent,
    RealtimeOutputSpeechStartEvent,
    RealtimeSession,
    RealtimeSessionReconnectEvent,
    WebRTCSession,
)
from pydantic_ai.realtime._openai_protocol import (
    RealtimeHandshakeError,
    _user_content_items,  # pyright: ignore[reportPrivateUsage]
    expect_event,
    realtime_websocket_url,
    replay_items,
    server_vad_from_turn_detection,
)
from pydantic_ai.realtime.codec import (
    AudioDelta,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    ConversationCreated,
    ConversationItemCreated,
    CreateResponse,
    InputTranscript,
    OutputTranscript,
    RealtimeCodecEvent,
    ResponseDone,
    SessionUsage,
    ToolCall,
    ToolResult,
    TruncateOutput,
)
from pydantic_ai.realtime.profiles import merge_realtime_profile
from pydantic_ai.realtime.xai import map_conversation_event as _map_conversation_wire_event
from pydantic_ai.settings import ThinkingLevel, ToolOrOutput
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from ..conftest import try_import
from .test_session import FakeRealtimeModel, make_tool_manager
from .ws_helpers import collect_codec_events, collect_session_events

with try_import() as imports_successful:
    from openai import AsyncOpenAI
    from openai.auth import WorkloadIdentity
    from openai.types import CompletionUsage
    from openai.types.chat import ChatCompletion
    from openai.types.completion_usage import CompletionTokensDetails, PromptTokensDetails
    from openai.types.realtime import RealtimeResponseUsage
    from openai.types.realtime.conversation_item_input_audio_transcription_completed_event import (
        UsageTranscriptTextUsageDuration,
        UsageTranscriptTextUsageTokens,
        UsageTranscriptTextUsageTokensInputTokenDetails,
    )
    from openai.types.realtime.realtime_audio_config_output import Voice as SDKVoice, VoiceID

    from pydantic_ai.models.openai import _map_usage as _map_standard_usage  # pyright: ignore[reportPrivateUsage]
    from pydantic_ai.providers.gateway import gateway_provider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.realtime import openai as rt_openai
    from pydantic_ai.realtime.openai import (
        KnownOpenAIRealtimeVoiceName,
        OpenAIRealtimeConnection,
        OpenAIRealtimeModel,
        map_event as _map_wire_event,
    )

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai / websockets not installed')


def sdk_frame(frame: dict[str, Any]) -> dict[str, Any]:  # noqa: C901
    """Fill protocol bookkeeping on concise synthetic frames; semantic payloads stay test-owned."""
    frame = dict(frame)
    event_type = cast('str', frame.get('type'))
    if event_type == 'session.created':
        frame.setdefault('event_id', 'event')
        session = cast('dict[str, Any]', frame.setdefault('session', {'type': 'realtime'}))
        session.setdefault('type', 'realtime')
        return frame
    if event_type in {'session.updated', 'rate_limits.updated'}:
        return frame
    frame.setdefault('event_id', 'event')
    if event_type.startswith('response.') and event_type != 'response.done':
        frame.setdefault('response_id', 'response')
    if event_type == 'response.created':
        frame.setdefault('response', {})
    if event_type in {
        'response.output_audio.delta',
        'response.audio.delta',
        'response.output_audio_transcript.delta',
        'response.audio_transcript.delta',
        'response.output_audio_transcript.done',
        'response.audio_transcript.done',
        'response.output_text.delta',
        'response.output_text.done',
    }:
        frame.setdefault('content_index', 0)
        frame.setdefault('output_index', 0)
        frame.setdefault('item_id', '')
    if event_type in {'response.output_audio_transcript.delta', 'response.audio_transcript.delta'}:
        frame.setdefault('delta', '')
    elif event_type in {'response.output_audio_transcript.done', 'response.audio_transcript.done'}:
        frame.setdefault('transcript', '')
    elif event_type == 'response.output_text.delta':
        frame.setdefault('delta', '')
    elif event_type == 'response.output_text.done':
        frame.setdefault('text', '')
    elif event_type == 'response.function_call_arguments.done':
        frame.setdefault('arguments', '{}')
        frame.setdefault('call_id', '')
        frame.setdefault('item_id', '')
        frame.setdefault('name', '')
        frame.setdefault('output_index', 0)
    elif event_type in {'input_audio_buffer.speech_started', 'input_audio_buffer.speech_stopped'}:
        frame.setdefault('audio_start_ms' if event_type.endswith('started') else 'audio_end_ms', 0)
        frame.setdefault('item_id', '')
    elif event_type in {
        'conversation.item.input_audio_transcription.delta',
        'conversation.item.input_audio_transcription.completed',
        'conversation.item.input_audio_transcription.failed',
    }:
        frame.setdefault('content_index', 0)
        frame.setdefault('item_id', '')
        if event_type.endswith('completed'):
            frame.setdefault('usage', {'type': 'duration', 'seconds': 0})
    elif event_type == 'error' and isinstance(error := frame.get('error'), dict):
        error = cast('dict[str, Any]', error)
        error.setdefault('message', '')
        error.setdefault('type', '')
        error.setdefault('code', None)
        error.setdefault('event_id', None)
        error.setdefault('param', None)
    elif event_type in {'conversation.item.added', 'conversation.item.created'} and isinstance(
        item := frame.get('item'), dict
    ):
        item = cast('dict[str, Any]', item)
        item.setdefault('type', 'message')
        if item['type'] == 'message':
            item.setdefault('role', 'assistant')
            item.setdefault('content', [])
        else:  # function_call
            item.setdefault('name', '')
            item.setdefault('arguments', '')
    if event_type == 'conversation.created' and isinstance(conversation := frame.get('conversation'), dict):
        conversation = cast('dict[str, Any]', conversation)
        conversation.setdefault('id', '')
        conversation.setdefault('object', 'realtime.conversation')
    return frame


def test_user_content_items_rejects_non_image_binary_content():
    """Upstream normalization only ever yields text and images; anything else must fail loudly."""
    with pytest.raises(
        UserError, match="Expected image content after realtime user-content normalization, got 'application/pdf'"
    ):
        _user_content_items([BinaryContent(data=b'%PDF', media_type='application/pdf')])


def map_event(frame: dict[str, Any]) -> RealtimeCodecEvent | None:
    return _map_wire_event(sdk_frame(frame))


def map_conversation_event(
    frame: dict[str, Any], *, replayed: bool | None = None
) -> ConversationCreated | ConversationItemCreated | None:
    return _map_conversation_wire_event(sdk_frame(frame), replayed=replayed)


def _wav_bytes(pcm: bytes, sample_rate: int = 24000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def test_known_voice_names_match_sdk() -> None:
    """`KnownOpenAIRealtimeVoiceName` mirrors the SDK's own voice union, so it can't quietly fall behind.

    The setting stays open to any string, so a stale list only costs autocomplete — but a voice OpenAI
    adds should show up here, and this is the only thing that would tell us.
    """
    sdk_voices = {voice for member in get_args(SDKVoice) if get_origin(member) is Literal for voice in get_args(member)}
    assert set(get_args(KnownOpenAIRealtimeVoiceName.__value__)) == sdk_voices


def test_map_transcription_usage() -> None:
    assert rt_openai._map_transcription_usage(None) is None  # pyright: ignore[reportPrivateUsage]
    # A zero duration is nothing billed, so there is no usage to report at all.
    assert (
        rt_openai._map_transcription_usage(  # pyright: ignore[reportPrivateUsage]
            UsageTranscriptTextUsageDuration(type='duration', seconds=0)
        )
        is None
    )
    # ...but any real duration rounds up to a visible second rather than down to free.
    assert rt_openai._map_transcription_usage(  # pyright: ignore[reportPrivateUsage]
        UsageTranscriptTextUsageDuration(type='duration', seconds=0.5)
    ) == RequestUsage(details={'input_transcription_seconds': 1})
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
    # A protocol clone (xAI/Azure) may omit the `type` discriminator, which `_validate_usage_shape`
    # tolerates; the SDK's lenient `.construct` then builds the tokens variant with `type=None`. It must
    # be read as tokens rather than falling through to the duration branch, whose `.seconds` this variant
    # lacks — previously an `AttributeError` that escaped the recoverable path and tore the session down.
    assert rt_openai._map_transcription_usage(  # pyright: ignore[reportPrivateUsage]
        UsageTranscriptTextUsageTokens.construct(total_tokens=12)
    ) == RequestUsage(details={'input_transcription_tokens': 12})
    # A type-less payload carrying only a duration is a graceful no-op, not a crash.
    assert rt_openai._map_transcription_usage(UsageTranscriptTextUsageTokens.construct(seconds=1.5)) is None  # pyright: ignore[reportPrivateUsage]


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
    with pytest.raises(ValueError):
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
    # The gateway accepts the `/v1`-less realtime path (like the OpenAI SDK's own `<base>/realtime`), so
    # the realtime URL is derived straight from the provider base URL, without inserting a `/v1` segment.
    monkeypatch.setenv('PYDANTIC_AI_GATEWAY_API_KEY', 'gw-key')
    monkeypatch.setenv('PYDANTIC_AI_GATEWAY_BASE_URL', 'https://gateway.pydantic.dev/proxy')
    string_model = OpenAIRealtimeModel('gpt-realtime', provider='gateway/openai')
    instance_model = OpenAIRealtimeModel(
        'gpt-realtime',
        provider=gateway_provider('openai', api_key='gw-key', base_url='https://gateway.pydantic.dev/proxy'),
    )
    plain_model = OpenAIRealtimeModel('gpt-realtime', provider=OpenAIProvider(api_key='k'))

    gateway_url = 'wss://gateway.pydantic.dev/proxy/openai/realtime?model=gpt-realtime'
    assert string_model._realtime_url() == gateway_url  # pyright: ignore[reportPrivateUsage]
    assert instance_model._realtime_url() == gateway_url  # pyright: ignore[reportPrivateUsage]
    # The plain OpenAI base URL already carries its own `/v1`.
    assert plain_model._realtime_url() == 'wss://api.openai.com/v1/realtime?model=gpt-realtime'  # pyright: ignore[reportPrivateUsage]


def test_map_audio_delta() -> None:
    payload = base64.b64encode(b'\x01\x02').decode('ascii')
    for event_type in ('response.output_audio.delta', 'response.audio.delta'):
        event = map_event({'type': event_type, 'delta': payload, 'item_id': 'item-a'})
        assert event == AudioDelta(data=b'\x01\x02', item_id='item-a')


def test_map_audio_delta_non_string_delta() -> None:
    with pytest.raises(ValueError):
        map_event({'type': 'response.output_audio.delta', 'delta': 123})


def test_map_transcript_delta_and_done() -> None:
    for event_type in ('response.output_audio_transcript.delta', 'response.audio_transcript.delta'):
        assert map_event({'type': event_type, 'delta': 'hel', 'item_id': 'item-a'}) == OutputTranscript(
            text='hel', is_final=False, item_id='item-a'
        )
    for event_type in ('response.output_audio_transcript.done', 'response.audio_transcript.done'):
        assert map_event({'type': event_type, 'transcript': 'hello', 'item_id': 'item-a'}) == OutputTranscript(
            text='hello', is_final=True, item_id='item-a'
        )


def test_map_text_output_delta_and_done() -> None:
    # `output_text=True` distinguishes plain text output from an audio transcript, so the session
    # persists it as a `TextPart` rather than a `SpeechPart`.
    assert map_event({'type': 'response.output_text.delta', 'delta': 'hel'}) == OutputTranscript(
        text='hel', is_final=False, output_text=True
    )
    assert map_event({'type': 'response.output_text.done', 'text': 'hello'}) == OutputTranscript(
        text='hello', is_final=True, output_text=True
    )


def test_map_transcript_missing_field_defaults_to_empty() -> None:
    assert map_event({'type': 'response.output_audio_transcript.delta'}) == OutputTranscript(text='', is_final=False)


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
    assert map_event(_response_done({'id': 'resp-1', 'status': 'completed', 'output': []})) == ResponseDone(
        interrupted=False,
        provider_response_id='resp-1',
        finish_reason='stop',
        provider_details={'status': 'completed'},
    )


def test_map_response_done_cancelled() -> None:
    # A cancelled response is a barge-in, not an error: `interrupted=True` (→ `state='interrupted'`)
    # carries the meaning and `finish_reason` is left unset, matching a classic cancelled stream.
    assert map_event(_response_done({'id': 'resp-2', 'status': 'cancelled'})) == ResponseDone(
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
    assert map_event(_response_done(response)) == ResponseDone(
        interrupted=False,
        provider_response_id='resp-incomplete',
        finish_reason=finish_reason,
        provider_details={'status': 'incomplete', 'finish_reason': reason},
    )


def test_map_response_done_function_call_only_is_skipped() -> None:
    assert (
        map_event(_response_done({'status': 'completed', 'output': [{'type': 'function_call', 'name': 'x'}]})) is None
    )


@pytest.mark.parametrize('status', ['cancelled', 'incomplete', 'failed'])
def test_map_response_done_terminal_function_call_only_is_completed(status: str) -> None:
    event = map_event(_response_done({'status': status, 'output': [{'type': 'function_call', 'name': 'x'}]}))
    assert isinstance(event, ResponseDone)
    assert event.provider_details == {'status': status}


def test_map_response_done_mixed_output_is_turn_complete() -> None:
    data = _response_done({'status': 'completed', 'output': [{'type': 'function_call'}, {'type': 'message'}]})
    assert map_event(data) == ResponseDone(
        interrupted=False, finish_reason='stop', provider_details={'status': 'completed'}
    )


def test_map_response_done_without_response_object() -> None:
    with pytest.raises(ValueError):
        map_event({'type': 'response.done'})


def test_map_response_done_failed_and_unknown_incomplete_reason() -> None:
    assert map_event(_response_done({'status': 'failed'})) == ResponseDone(
        interrupted=False, finish_reason='error', provider_details={'status': 'failed'}
    )
    with pytest.raises(ValueError):
        map_event(_response_done({'status': 'incomplete', 'status_details': {'reason': 'network'}}))


def test_map_conversation_item_without_identifiers_is_ignored() -> None:
    assert map_conversation_event({'type': 'conversation.item.created', 'item': {}}) is None
    with pytest.raises(ValueError):
        map_conversation_event({'type': 'conversation.item.created'})


@pytest.mark.parametrize(
    'frame',
    [
        {'type': 'conversation.created', 'conversation': 'bad'},
        {'type': 'conversation.item.created', 'item': 'bad'},
    ],
)
def test_map_conversation_event_rejects_malformed_nested_object(frame: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        map_conversation_event(frame)
    with pytest.raises(ValueError):
        map_conversation_event({'type': 'conversation.created'})


# The conversation mapper is shared OpenAI-protocol surface, but only xAI opts into it (OpenAI itself
# doesn't support resumption, so its mapper deliberately leaves these frames unmapped). The next three
# tests therefore drive it directly rather than through a cassette: the frames appear only in xAI
# recordings, and behaviour this mapper defines deserves provider-neutral coverage that survives
# whichever providers happen to consume it.
def test_map_conversation_created_returns_the_conversation_id() -> None:
    assert map_conversation_event({'type': 'conversation.created', 'conversation': {'id': 'conv-1'}}) == (
        ConversationCreated('conv-1')
    )
    # A handshake that names no conversation carries nothing to resume against, so it maps to nothing.
    assert map_conversation_event({'type': 'conversation.created', 'conversation': {}}) is None


@pytest.mark.parametrize('event_type', ['conversation.item.added', 'conversation.item.created'])
def test_map_conversation_item_lifecycle_events_carry_identifiers(event_type: str) -> None:
    """Both item lifecycle spellings map alike, and a function call also surfaces its `call_id`."""
    assert map_conversation_event({'type': event_type, 'item': {'id': 'item-1'}}) == ConversationItemCreated(
        item_id='item-1', tool_call_id=None, replayed=False
    )
    assert map_conversation_event(
        {'type': event_type, 'item': {'id': 'item-2', 'type': 'function_call', 'call_id': 'call-1'}}
    ) == ConversationItemCreated(item_id='item-2', tool_call_id='call-1', replayed=False)
    # `replayed=True` is set only by the reconnect handshake's burst capture, and marks the item as
    # already-seen history rather than a live event.
    assert map_conversation_event({'type': event_type, 'item': {'id': 'item-3'}}, replayed=True) == (
        ConversationItemCreated(item_id='item-3', tool_call_id=None, replayed=True)
    )


def test_map_conversation_event_ignores_unrelated_event_types() -> None:
    assert map_conversation_event({'type': 'response.done'}) is None


def test_map_error_event_with_message() -> None:
    assert map_event({'type': 'error', 'error': {'message': 'bad'}}) == RealtimeSessionErrorEvent(message='bad')


def test_map_error_event_without_message_serializes_payload() -> None:
    assert map_event({'type': 'error', 'error': {'code': 'x'}}) == RealtimeSessionErrorEvent(
        message='{"message":"","type":"","code":"x"}', code='x'
    )


def test_map_error_event_non_dict_payload() -> None:
    with pytest.raises(ValueError):
        map_event({'type': 'error', 'error': 'plain'})


def test_handshake_error_message_falls_back_to_repr() -> None:
    # `map_event` always hands `_error_message` a parsed `RealtimeError`, but `RealtimeHandshakeError`
    # takes whatever the handshake produced. A payload that is neither a `RealtimeError` nor a dict with
    # a string `message` still has to yield *something* readable rather than blowing up on access.
    assert str(RealtimeHandshakeError({'code': 'x'})) == snapshot("{'code': 'x'}")
    assert str(RealtimeHandshakeError(None)) == snapshot('None')


def test_map_error_event_with_type_and_code_is_recoverable() -> None:
    event = map_event({'type': 'error', 'error': {'message': 'bad', 'type': 'invalid_request_error', 'code': 'c1'}})
    assert event == RealtimeSessionErrorEvent(message='bad', type='invalid_request_error', code='c1', recoverable=True)


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
        details={'input_text_tokens': 20, 'input_image_tokens': 5, 'output_text_tokens': 10, 'audio_tokens': 40},
    )


def test_map_usage_matches_standard_openai_normalization() -> None:
    """Realtime and Chat Completions normalize every shared usage concept identically."""
    realtime_usage = rt_openai._map_usage(  # pyright: ignore[reportPrivateUsage]
        RealtimeResponseUsage.construct(
            input_tokens=100,
            output_tokens=50,
            input_token_details={'audio_tokens': 80, 'cached_tokens': 30},
            output_token_details={'audio_tokens': 40, 'reasoning_tokens': 7},
        )
    )
    standard_response = ChatCompletion(
        id='response-id',
        choices=[],
        created=0,
        model='gpt-realtime',
        object='chat.completion',
        usage=CompletionUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details=PromptTokensDetails(audio_tokens=80, cached_tokens=30),
            completion_tokens_details=CompletionTokensDetails(audio_tokens=40, reasoning_tokens=7),
        ),
    )
    standard_usage = _map_standard_usage(standard_response, 'openai', 'https://api.openai.com/v1', 'gpt-realtime')

    assert realtime_usage == standard_usage


def test_map_usage_minimal_and_missing() -> None:
    sdk_usage = RealtimeResponseUsage.construct(input_tokens=7)
    assert rt_openai._map_usage(sdk_usage) == RequestUsage(input_tokens=7)  # pyright: ignore[reportPrivateUsage]
    assert rt_openai._map_usage(None) is None  # pyright: ignore[reportPrivateUsage]


def test_map_speech_started() -> None:
    assert map_event({'type': 'input_audio_buffer.speech_started'}) == RealtimeInputSpeechStartEvent()


def test_map_speech_stopped() -> None:
    assert map_event({'type': 'input_audio_buffer.speech_stopped'}) == RealtimeInputSpeechEndEvent()


def test_map_unhandled_event_returns_none() -> None:
    # Frames we don't surface as user-facing events (lifecycle acks like `session.created`, and
    # `rate_limits.updated`, which has no `RealtimeEvent` representation) fall through to `None`.
    assert map_event({'type': 'session.created'}) is None
    assert map_event({'type': 'rate_limits.updated', 'rate_limits': [{'name': 'requests', 'limit': 100}]}) is None


@pytest.mark.parametrize(
    ('frame', 'expected'),
    [
        (
            {'type': 'response.output_audio.delta', 'delta': 'AQI=', 'item_id': 'a'},
            AudioDelta(b'\x01\x02', item_id='a'),
        ),
        ({'type': 'response.audio.delta', 'delta': 'AQI=', 'item_id': 'a'}, AudioDelta(b'\x01\x02', item_id='a')),
        (
            {'type': 'response.output_audio_transcript.delta', 'delta': 'hel', 'item_id': 'a'},
            OutputTranscript('hel', is_final=False, item_id='a'),
        ),
        (
            {'type': 'response.audio_transcript.delta', 'delta': 'hel', 'item_id': 'a'},
            OutputTranscript('hel', is_final=False, item_id='a'),
        ),
        (
            {'type': 'response.output_audio_transcript.done', 'transcript': 'hello', 'item_id': 'a'},
            OutputTranscript('hello', is_final=True, item_id='a'),
        ),
        (
            {'type': 'response.audio_transcript.done', 'transcript': 'hello', 'item_id': 'a'},
            OutputTranscript('hello', is_final=True, item_id='a'),
        ),
        (
            {'type': 'response.output_text.delta', 'delta': 'hel'},
            OutputTranscript('hel', is_final=False, output_text=True),
        ),
        (
            {'type': 'response.output_text.done', 'text': 'hello'},
            OutputTranscript('hello', is_final=True, output_text=True),
        ),
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
            ToolCall('call-1', tool_name='weather', args='{}', response_usage_follows=True),
        ),
        ({'type': 'input_audio_buffer.speech_started'}, RealtimeInputSpeechStartEvent()),
        ({'type': 'input_audio_buffer.speech_stopped'}, RealtimeInputSpeechEndEvent()),
        (
            {'type': 'response.done', 'response': {'id': 'r', 'status': 'completed', 'output': []}},
            ResponseDone(
                interrupted=False,
                provider_response_id='r',
                finish_reason='stop',
                provider_details={'status': 'completed'},
            ),
        ),
        ({'type': 'error', 'error': {'message': 'bad'}}, RealtimeSessionErrorEvent('bad')),
        (
            {'type': 'conversation.item.input_audio_transcription.failed', 'error': {'message': 'bad'}},
            RealtimeInputTranscriptionErrorEvent(message='bad', content_index=0),
        ),
        (
            {
                'type': 'conversation.item.input_audio_transcription.failed',
                'error': {'message': 'bad', 'type': 'transcription_error', 'code': 'audio_unintelligible'},
                'item_id': 'u',
                'content_index': 2,
            },
            RealtimeInputTranscriptionErrorEvent(
                message='bad',
                type='transcription_error',
                code='audio_unintelligible',
                item_id='u',
                content_index=2,
            ),
        ),
        (
            # Azure's `DeploymentNotFound` message names only the affected item, so the remedy is appended.
            {
                'type': 'conversation.item.input_audio_transcription.failed',
                'error': {'message': 'x', 'type': 'server_error', 'code': 'DeploymentNotFound'},
            },
            RealtimeInputTranscriptionErrorEvent(
                message=(
                    'x The transcription model is not deployed on this Azure OpenAI resource. Deploy one and '
                    'set `input_transcription_model` to its deployment name, or set it to `None` to disable '
                    'transcription.'
                ),
                type='server_error',
                code='DeploymentNotFound',
                content_index=0,
            ),
        ),
        (
            # A `DeploymentNotFound` with no message of its own still carries the remedy, without a leading space.
            {
                'type': 'conversation.item.input_audio_transcription.failed',
                'error': {'code': 'DeploymentNotFound'},
            },
            RealtimeInputTranscriptionErrorEvent(
                message=(
                    'The transcription model is not deployed on this Azure OpenAI resource. Deploy one and '
                    'set `input_transcription_model` to its deployment name, or set it to `None` to disable '
                    'transcription.'
                ),
                code='DeploymentNotFound',
                content_index=0,
            ),
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
    def _normalize_frame(frame: Any) -> Any:
        if not isinstance(frame, str):
            return frame
        try:
            data = json.loads(frame)
        except json.JSONDecodeError:
            return frame
        return json.dumps(sdk_frame(cast('dict[str, Any]', data))) if isinstance(data, dict) else frame

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
async def test_expect_event_hands_skipped_frames_to_the_callback() -> None:
    """Frames arriving before the awaited one are skipped, and offered to `on_unexpected` when given.

    Driven directly rather than through a cassette: no OpenAI handshake passes a callback (only xAI's
    resumption burst capture does), so this shared handshake reader's hand-off is otherwise unpinned.
    """
    ws = FakeWebSocket(
        [
            json.dumps({'type': 'rate_limits.updated'}),
            json.dumps({'type': 'conversation.item.created', 'item': {'id': 'item-1'}}),
            _updated(),
        ]
    )
    skipped: list[dict[str, Any]] = []

    awaited = await expect_event(ws, 'session.updated', timeout=5, on_unexpected=skipped.append)  # type: ignore[arg-type]

    assert awaited == {'type': 'session.updated'}
    assert [frame['type'] for frame in skipped] == ['rate_limits.updated', 'conversation.item.created']


@pytest.mark.anyio
async def test_connect_handshake_and_session_config(monkeypatch: pytest.MonkeyPatch) -> None:
    transcript = json.dumps({'type': 'response.audio_transcript.done', 'transcript': 'hi'})
    ws = FakeWebSocket([_created(), _updated(), transcript])
    fake_connect = FakeConnect(ws)
    monkeypatch.setattr(rt_openai.websockets, 'connect', fake_connect)

    model = OpenAIRealtimeModel(
        'gpt-realtime',
        provider=OpenAIProvider(api_key='k'),
        settings=rt_openai.OpenAIRealtimeModelSettings(openai_voice='alloy'),
    )
    tools = [ToolDefinition(name='get_weather', description='Weather', parameters_json_schema={'type': 'object'})]

    async with _connect(model, 'Be nice', tools=tools) as conn:
        events = await collect_codec_events(conn)

    assert events == [OutputTranscript(text='hi', is_final=True)]
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
async def test_connect_webrtc_sideband_handshake_and_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    updated = json.dumps({'type': 'session.updated', 'session': {}})
    ws = FakeWebSocket([updated])
    fake_connect = FakeConnect(ws)
    monkeypatch.setattr(rt_openai.websockets, 'connect', fake_connect)
    model = OpenAIRealtimeModel('gpt-realtime', provider=OpenAIProvider(api_key='k'))
    history = [
        ModelRequest(parts=[UserPromptPart(content='My favorite color is teal.')]),
        ModelResponse(parts=[TextPart(content='Got it, teal.')]),
    ]

    async with model.connect_webrtc(
        WebRTCSession(provider_name='openai', session_id='rtc_seed'),
        messages=history,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    ) as conn:
        assert await collect_codec_events(conn, sideband=True) == []

    assert fake_connect.url == 'wss://api.openai.com/v1/realtime?call_id=rtc_seed'
    assert conn.model_name is None
    sent = [json.loads(frame) for frame in ws.sent]
    assert sent[0]['type'] == 'session.update'
    assert [frame['item']['role'] for frame in sent[1:]] == ['user', 'assistant']


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


async def test_connect_resolves_workload_identity_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """The handshake exchanges a `workload_identity` credential for a token, not the SDK placeholder.

    A `workload_identity` client leaves `client.api_key` set to a placeholder string and swaps in a real
    token inside its HTTP request path, which the raw WebSocket handshake bypasses — so a regression
    sends the placeholder as the bearer credential and the handshake is rejected.
    """
    client = AsyncOpenAI(
        workload_identity=WorkloadIdentity(
            identity_provider_id='idp-1',
            service_account_id='sa-1',
            provider={'token_type': 'jwt', 'get_token': lambda: 'subject-token'},
        )
    )
    assert client.api_key == 'workload-identity-auth'  # the placeholder, not a usable credential

    workload_identity_auth = client._workload_identity_auth  # pyright: ignore[reportPrivateUsage]
    assert workload_identity_auth is not None
    monkeypatch.setattr(workload_identity_auth, 'get_token', lambda: 'exchanged-token')

    ws = FakeWebSocket([_created(), _updated()])
    fake_connect = FakeConnect(ws)
    monkeypatch.setattr(rt_openai.websockets, 'connect', fake_connect)

    model = OpenAIRealtimeModel('gpt-realtime', provider=OpenAIProvider(openai_client=client))
    async with _connect(model, 'hi') as conn:
        _ = [e async for e in conn]

    assert fake_connect.headers is not None
    assert fake_connect.headers['Authorization'] == 'Bearer exchanged-token'


def test_session_config_server_vad_params() -> None:
    model = OpenAIRealtimeModel(
        'gpt-realtime',
        settings=rt_openai.OpenAIRealtimeModelSettings(
            openai_turn_detection={
                'type': 'server_vad',
                'threshold': 0.7,
                'prefix_padding_ms': 200,
                'silence_duration_ms': 400,
                'create_response': False,
                'interrupt_response': False,
                'idle_timeout_ms': 5000,
            },
        ),
    )
    config = model._session_config('hi', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]
    assert config['audio']['input']['turn_detection'] == {
        'type': 'server_vad',
        'create_response': False,
        'interrupt_response': False,
        'threshold': 0.7,
        'prefix_padding_ms': 200,
        'silence_duration_ms': 400,
        'idle_timeout_ms': 5000,
    }


def test_session_config_uses_profile_sample_rates() -> None:
    model = OpenAIRealtimeModel(
        'gpt-realtime', profile=RealtimeModelProfile(audio_input_sample_rate=16000, audio_output_sample_rate=32000)
    )

    config = model._session_config('', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]

    assert config['audio']['input']['format']['rate'] == 16000
    assert config['audio']['output']['format']['rate'] == 32000


def test_server_vad_from_turn_detection_mapping() -> None:
    # All three cross-provider knobs map through; sensitivity resolves via the threshold table.
    assert server_vad_from_turn_detection(
        {'sensitivity': 'high', 'prefix_padding_ms': 100, 'silence_duration_ms': 300}
    ) == {'type': 'server_vad', 'threshold': 0.3, 'prefix_padding_ms': 100, 'silence_duration_ms': 300}
    # Without knobs the provider defaults stay in charge: only the discriminator is sent.
    assert server_vad_from_turn_detection({}) == {'type': 'server_vad'}


def test_session_config_truncation_modes() -> None:
    # A plain mode passes through as-is; a retention ratio maps to the retention_ratio truncation shape.
    auto = OpenAIRealtimeModel(
        'gpt-realtime', settings=rt_openai.OpenAIRealtimeModelSettings(openai_truncation='disabled')
    )
    assert auto._session_config('hi', None, model_settings=None)['truncation'] == 'disabled'  # pyright: ignore[reportPrivateUsage]

    ratio = OpenAIRealtimeModel(
        'gpt-realtime',
        settings=rt_openai.OpenAIRealtimeModelSettings(
            openai_truncation={'type': 'retention_ratio', 'retention_ratio': 0.8}
        ),
    )
    assert ratio._session_config('hi', None, model_settings=None)['truncation'] == {  # pyright: ignore[reportPrivateUsage]
        'type': 'retention_ratio',
        'retention_ratio': 0.8,
    }
    # Absent by default so the wire stays byte-identical for sessions that don't set it.
    assert 'truncation' not in OpenAIRealtimeModel('gpt-realtime')._session_config('hi', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]


def test_session_config_thinking_maps_to_reasoning_on_reasoning_models() -> None:
    # `thinking` maps to reasoning effort on the `gpt-realtime-2*` reasoning models (live-verified
    # that these accept `reasoning.effort` while the GA `gpt-realtime` rejects it).
    def reasoning(thinking: ThinkingLevel) -> object:
        model = OpenAIRealtimeModel(
            'gpt-realtime-2.1', settings=rt_openai.OpenAIRealtimeModelSettings(thinking=thinking)
        )
        return model._session_config('hi', None, model_settings=None).get('reasoning')  # pyright: ignore[reportPrivateUsage]

    assert reasoning('low') == {'effort': 'low'}
    assert reasoning('high') == {'effort': 'high'}
    assert reasoning(True) == {'effort': 'medium'}
    # `thinking=False` maps to effort `'none'`, which the realtime `reasoning.effort` doesn't accept,
    # so it's omitted (a reasoning model falls back to its default rather than erroring).
    assert reasoning(False) is None


def test_session_config_thinking_on_non_reasoning_model_is_ignored() -> None:
    # The GA `gpt-realtime` isn't a reasoning model, so `thinking` is silently dropped, matching
    # unsupported generic settings on classic model adapters.
    model = OpenAIRealtimeModel('gpt-realtime', settings=rt_openai.OpenAIRealtimeModelSettings(thinking='high'))
    config = model._session_config('hi', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]
    assert 'reasoning' not in config


def test_session_config_openai_turn_detection_overrides_base() -> None:
    model = OpenAIRealtimeModel(
        'gpt-realtime',
        settings=rt_openai.OpenAIRealtimeModelSettings(
            turn_detection={'sensitivity': 'low'},
            openai_turn_detection={'type': 'semantic_vad', 'eagerness': 'high'},
        ),
    )
    config = model._session_config('hi', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]
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
    settings = rt_openai.OpenAIRealtimeModelSettings(turn_detection={'sensitivity': sensitivity})
    config = OpenAIRealtimeModel('gpt-realtime', settings=settings)._session_config(  # pyright: ignore[reportPrivateUsage]
        'hi', None, model_settings=None
    )
    assert config['audio']['input']['turn_detection']['threshold'] == threshold


def test_session_config_manual_turn_detection_is_null() -> None:
    """`turn_detection=False` disables VAD (push-to-talk), sent as an explicit null."""
    model = OpenAIRealtimeModel('gpt-realtime', settings=rt_openai.OpenAIRealtimeModelSettings(turn_detection=False))
    config = model._session_config('hi', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]
    assert config['audio']['input']['turn_detection'] is None


def test_session_config_turn_detection_true_matches_default() -> None:
    """`turn_detection=True` enables server VAD at the provider defaults — identical to an absent setting."""
    enabled = OpenAIRealtimeModel('gpt-realtime', settings=rt_openai.OpenAIRealtimeModelSettings(turn_detection=True))
    default = OpenAIRealtimeModel('gpt-realtime')
    assert (
        enabled._session_config('hi', None, model_settings=None)['audio']['input']['turn_detection']  # pyright: ignore[reportPrivateUsage]
        == default._session_config('hi', None, model_settings=None)['audio']['input']['turn_detection']  # pyright: ignore[reportPrivateUsage]
    )


def test_session_config_noise_reduction_and_speed_and_modalities() -> None:
    model = OpenAIRealtimeModel(
        'gpt-realtime',
        settings=rt_openai.OpenAIRealtimeModelSettings(
            openai_input_noise_reduction='near_field', openai_output_speed=1.25, output_modality='text'
        ),
    )
    config = model._session_config('hi', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]
    assert config['audio']['input']['noise_reduction'] == {'type': 'near_field'}
    assert config['audio']['output']['speed'] == 1.25
    assert config['output_modalities'] == ['text']


def test_session_config_forwards_parallel_tool_calls_and_tool_choice() -> None:
    settings = rt_openai.OpenAIRealtimeModelSettings(parallel_tool_calls=True, tool_choice='required')
    model = OpenAIRealtimeModel('gpt-realtime', settings=settings)
    assert model.settings == settings
    tools = [ToolDefinition(name='get_weather', parameters_json_schema={'type': 'object'})]
    config = model._session_config('hi', tools, model_settings=settings)  # pyright: ignore[reportPrivateUsage]
    assert config['parallel_tool_calls'] is True
    assert config['tool_choice'] == 'required'


def test_session_config_merges_model_defaults_and_connection_overrides() -> None:
    model = OpenAIRealtimeModel(
        'gpt-realtime', settings=rt_openai.OpenAIRealtimeModelSettings(openai_voice='alloy', max_tokens=128)
    )
    config = model._session_config(  # pyright: ignore[reportPrivateUsage]
        'hi', None, model_settings=rt_openai.OpenAIRealtimeModelSettings(openai_voice='echo')
    )

    assert config['audio']['output']['voice'] == 'echo'
    assert config['max_output_tokens'] == 128


def test_session_config_forwards_custom_voice_id() -> None:
    model = OpenAIRealtimeModel(
        'gpt-realtime', settings=rt_openai.OpenAIRealtimeModelSettings(openai_voice=VoiceID(id='voice_custom'))
    )
    config = model._session_config('hi', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]

    assert config['audio']['output']['voice'] == {'id': 'voice_custom'}


def test_session_config_tool_choice_single_function() -> None:
    model = OpenAIRealtimeModel('gpt-realtime')
    tools = [ToolDefinition(name=name, parameters_json_schema={'type': 'object'}) for name in ('get_weather', 'other')]
    config = model._session_config(  # pyright: ignore[reportPrivateUsage]
        'hi', tools, model_settings=rt_openai.OpenAIRealtimeModelSettings(tool_choice=['get_weather'])
    )
    assert config['tool_choice'] == {'type': 'function', 'name': 'get_weather'}
    assert [tool['name'] for tool in config['tools']] == ['get_weather']


def test_session_config_tool_choice_multi_tool_restricts_advertised_tools() -> None:
    model = OpenAIRealtimeModel('gpt-realtime')
    tools = [ToolDefinition(name=name, parameters_json_schema={'type': 'object'}) for name in ('a', 'b', 'excluded')]
    config = model._session_config(  # pyright: ignore[reportPrivateUsage]
        'hi', tools, model_settings=rt_openai.OpenAIRealtimeModelSettings(tool_choice=['a', 'b'])
    )
    assert config['tool_choice'] == 'required'
    assert [tool['name'] for tool in config['tools']] == ['a', 'b']


def test_session_config_tool_choice_tool_or_output_restricts_advertised_tools() -> None:
    model = OpenAIRealtimeModel('gpt-realtime')
    tools = [ToolDefinition(name=name, parameters_json_schema={'type': 'object'}) for name in ('a', 'excluded')]
    settings = rt_openai.OpenAIRealtimeModelSettings(tool_choice=ToolOrOutput(function_tools=['a']))
    config = model._session_config('hi', tools, model_settings=settings)  # pyright: ignore[reportPrivateUsage]
    assert config['tool_choice'] == 'auto'
    assert [tool['name'] for tool in config['tools']] == ['a']


def test_session_config_tool_choice_none_advertises_no_tools() -> None:
    tools = [ToolDefinition(name='unsafe', parameters_json_schema={'type': 'object'})]
    config = OpenAIRealtimeModel('gpt-realtime')._session_config(  # pyright: ignore[reportPrivateUsage]
        'hi', tools, model_settings=rt_openai.OpenAIRealtimeModelSettings(tool_choice='none')
    )
    assert config['tool_choice'] == 'none'
    assert 'tools' not in config


@pytest.mark.anyio
async def test_connect_skips_unrelated_events_during_handshake(monkeypatch: pytest.MonkeyPatch) -> None:
    rate_limits = json.dumps({'type': 'rate_limits.updated'})
    ws = FakeWebSocket([rate_limits, _created(), _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    model = OpenAIRealtimeModel('gpt-realtime')
    async with _connect(model, 'x') as conn:
        assert await collect_codec_events(conn) == []


@pytest.mark.anyio
async def test_connect_surfaces_handshake_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A rejected session config (here an unsupported voice) arrives as an `error` event over the open
    # WebSocket — no HTTP status — so it surfaces as a `ModelAPIError` carrying the provider's message,
    # like a non-status provider error from a regular request, rather than a raw protocol error.
    error = json.dumps(
        {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': "Invalid value: 'Puck'."}}
    )
    ws = FakeWebSocket([_created(), error])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    model = OpenAIRealtimeModel('gpt-realtime')
    with pytest.raises(ModelAPIError, match=re.escape("Invalid value: 'Puck'.")) as exc_info:
        async with _connect(model, 'x'):
            pass  # pragma: no cover
    assert exc_info.value.model_name == 'gpt-realtime'
    assert not isinstance(exc_info.value, ModelHTTPError)  # no HTTP status on a WebSocket error event


@pytest.mark.anyio
async def test_connect_surfaces_http_upgrade_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A rejected WebSocket upgrade (bad key → 401, unknown model → 404) carries a real HTTP status, so
    # it surfaces as `ModelHTTPError`, exactly like a regular request would.
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response

    class _RejectingConnect:
        def __call__(self, url: str, *, additional_headers: dict[str, str] | None = None) -> Any:
            return self

        async def __aenter__(self) -> Any:
            raise InvalidStatus(Response(404, 'Not Found', Headers(), body=b'unknown model'))

        async def __aexit__(self, *exc: object) -> bool:  # pragma: no cover
            return False

    monkeypatch.setattr(rt_openai.websockets, 'connect', _RejectingConnect())
    model = OpenAIRealtimeModel('gpt-realtime')
    with pytest.raises(ModelHTTPError) as exc_info:
        async with _connect(model, 'x'):
            pass  # pragma: no cover
    assert exc_info.value.status_code == 404
    assert exc_info.value.model_name == 'gpt-realtime'
    assert exc_info.value.body == 'unknown model'


@pytest.mark.anyio
async def test_connect_surfaces_handshake_connection_close(monkeypatch: pytest.MonkeyPatch) -> None:
    # The server accepts the upgrade but then closes the socket during the handshake (e.g. a gateway
    # rejecting an unknown model) instead of sending an `error` event. Without mapping, the session would
    # die silently; it should surface as a `ModelAPIError` (a WebSocket close carries no HTTP status).
    from websockets.exceptions import ConnectionClosedError
    from websockets.frames import Close

    class _ClosingWebSocket(FakeWebSocket):
        async def recv(self) -> Any:
            raise ConnectionClosedError(Close(1008, 'unknown model'), None)

    ws = _ClosingWebSocket([])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    model = OpenAIRealtimeModel('gpt-realtime')
    with pytest.raises(ModelAPIError, match='WebSocket error during realtime handshake') as exc_info:
        async with _connect(model, 'x'):
            pass  # pragma: no cover
    assert exc_info.value.model_name == 'gpt-realtime'
    assert not isinstance(exc_info.value, ModelHTTPError)  # a close code is not an HTTP status


@pytest.mark.anyio
@pytest.mark.parametrize(
    ('frame', 'expected'),
    [
        pytest.param(b'\x00binary', 'expected a text frame, got bytes', id='binary'),
        pytest.param('{not json', 'received a malformed frame: 1 validation error', id='invalid-json'),
        pytest.param('[1, 2]', 'received a malformed frame: 1 validation error', id='not-an-object'),
    ],
)
async def test_connect_surfaces_malformed_handshake_frame(
    frame: Any, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A frame we can't read during the handshake is a protocol failure, not a Python-level bug in the
    # caller's code: it surfaces as `ModelAPIError` rather than a bare `TypeError`/`ValueError`.
    ws = FakeWebSocket([frame])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    model = OpenAIRealtimeModel('gpt-realtime')
    with pytest.raises(ModelAPIError, match=re.escape(expected)) as exc_info:
        async with _connect(model, 'x'):
            pass  # pragma: no cover
    assert exc_info.value.model_name == 'gpt-realtime'


class HangingWebSocket(FakeWebSocket):
    """A websocket whose `recv` never returns, to exercise the handshake timeout."""

    async def recv(self) -> Any:
        await asyncio.Event().wait()


@pytest.mark.anyio
async def test_connect_handshake_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = HangingWebSocket([])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    model = OpenAIRealtimeModel('gpt-realtime', settings=rt_openai.OpenAIRealtimeModelSettings(handshake_timeout=0.02))
    with pytest.raises(ModelAPIError, match=re.escape("timed out waiting for a 'session.created' event")):
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

        async def __aexit__(self, *exc: object) -> bool:  # pragma: no cover
            return False

    monkeypatch.setattr(rt_openai.websockets, 'connect', _FailingConnect())
    model = OpenAIRealtimeModel('gpt-realtime')
    with pytest.raises(ModelAPIError, match='Could not reach the realtime API: refused'):
        async with _connect(model, 'x'):
            pass  # pragma: no cover


@pytest.mark.anyio
async def test_connection_iter_skips_non_string_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = json.dumps({'type': 'response.output_audio.delta', 'delta': base64.b64encode(b'\x09').decode('ascii')})
    ws = FakeWebSocket([_created(), _updated(), b'\x00binary', audio])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    model = OpenAIRealtimeModel('gpt-realtime')
    async with _connect(model, 'x') as conn:
        events = await collect_codec_events(conn)
    assert events == [AudioDelta(data=b'\x09')]


@pytest.mark.anyio
async def test_connection_iter_recovers_from_malformed_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    # A malformed frame (invalid JSON, a valid-JSON-but-non-object frame, then a bad base64 audio
    # payload) surfaces as a recoverable RealtimeSessionErrorEvent and the session keeps streaming rather than
    # tearing down. The non-object case guards against `json.loads` returning a list/str/number, which
    # would otherwise raise `AttributeError` from a later `.get()` and escape the recoverable path.
    bad_json = 'not json'
    non_object = json.dumps(['not', 'an', 'object'])
    bad_audio = json.dumps({'type': 'response.output_audio.delta', 'delta': '!!!!'})
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
    ]
    # The same for a final input transcript, except the transcript itself survives: it is already
    # derived by the time the usage payload is validated, so a malformed one costs the usage event
    # rather than the user's words.
    malformed_transcription_frames = [
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
                'usage': {'type': 'duration', 'seconds': float('inf')},
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
    for malformed in (*malformed_nested_frames, *malformed_transcription_frames):
        frames.extend((malformed, good))
    ws = FakeWebSocket([_created(), _updated(), *frames])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    model = OpenAIRealtimeModel('gpt-realtime')
    async with _connect(model, 'x') as conn:
        events = await collect_codec_events(conn)
    assert [type(e).__name__ for e in events] == [
        'RealtimeSessionErrorEvent',
        'RealtimeSessionErrorEvent',
        'RealtimeSessionErrorEvent',
        *['RealtimeSessionErrorEvent', 'AudioDelta'] * len(malformed_nested_frames),
        *['InputTranscript', 'RealtimeSessionErrorEvent', 'AudioDelta'] * 3,
        # `pydantic_core.from_json` rejects the non-standard `Infinity` token before event mapping.
        'RealtimeSessionErrorEvent',
        'AudioDelta',
        *['InputTranscript', 'RealtimeSessionErrorEvent', 'AudioDelta'],
    ]
    errors = [event for event in events if isinstance(event, RealtimeSessionErrorEvent)]
    assert len(errors) == 3 + len(malformed_nested_frames) + len(malformed_transcription_frames)
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

    monkeypatch.setattr('pydantic_ai.realtime._utils.download_item', download_image)
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

    monkeypatch.setattr('pydantic_ai.realtime._utils.download_item', download_image)
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
async def test_replay_items_strips_media_and_keeps_tagged_text() -> None:
    history = [
        ModelRequest(parts=[UserPromptPart(content=[TextContent('earlier question')])]),
        ModelResponse(
            parts=[
                ToolCallPart(tool_name='inspect', args={}, tool_call_id='call-image'),
                FilePart(content=BinaryContent(data=b'file', media_type='application/pdf')),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='inspect',
                    tool_call_id='call-image',
                    content=['done', BinaryContent(data=b'image', media_type='image/png')],
                )
            ]
        ),
    ]

    assert await replay_items(history, profile=RealtimeModelProfile(), provider_name='openai') == [
        {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'earlier question'}]},
        {'type': 'function_call', 'name': 'inspect', 'call_id': 'call-image', 'arguments': '{}'},
        {'type': 'function_call_output', 'call_id': 'call-image', 'output': 'done'},
    ]


@pytest.mark.anyio
async def test_replay_items_keeps_failed_multimodal_tool_return_wrapped_once() -> None:
    history = [
        ModelResponse(parts=[ToolCallPart(tool_name='inspect', args={}, tool_call_id='call-image')]),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='inspect',
                    tool_call_id='call-image',
                    content=['bad', BinaryContent(data=b'image', media_type='image/png')],
                    outcome='failed',
                )
            ]
        ),
    ]

    items = await replay_items(history, profile=RealtimeModelProfile(), provider_name='openai')
    assert items[-1] == {
        'type': 'function_call_output',
        'call_id': 'call-image',
        'output': '{"error":"bad"}',
    }


@pytest.mark.anyio
async def test_seed_call_ids_remain_unique_when_short_id_matches_long_id_hash() -> None:
    long_id = 'long-tool-call-id-that-needs-protocol-shortening'
    colliding_short_id = hashlib.sha256(long_id.encode()).hexdigest()[:32]
    history = [
        ModelResponse(
            parts=[
                ToolCallPart(tool_name='short', args={}, tool_call_id=colliding_short_id),
                ToolCallPart(tool_name='long', args={}, tool_call_id=long_id),
            ]
        )
    ]

    items = await replay_items(history, profile=RealtimeModelProfile(), provider_name='openai')
    assert items[0]['call_id'] == colliding_short_id
    assert items[1]['call_id'] != colliding_short_id


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

    with pytest.raises(UserError, match='cannot be sent to openai in a realtime session'):
        async with _connect(OpenAIRealtimeModel('gpt-realtime'), 'x', messages=history):
            pass  # pragma: no cover


@pytest.mark.anyio
async def test_connect_rejects_image_url_returning_non_image(monkeypatch: pytest.MonkeyPatch) -> None:
    async def download_document(*args: Any, **kwargs: Any) -> Any:
        return {'data': b'not-image', 'data_type': 'application/pdf'}

    monkeypatch.setattr('pydantic_ai.realtime._utils.download_item', download_document)
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
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))
    async with _connect(
        OpenAIRealtimeModel('gpt-realtime'),
        'x',
        messages=[
            ModelRequest(parts=[SpeechPart(speaker='user')]),
            ModelResponse(parts=[SpeechPart(speaker='assistant')]),
        ],
    ):
        pass
    assert not any(json.loads(item).get('type') == 'conversation.item.create' for item in ws.sent)

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
    await conn.send(BinaryAudio(data=b'\x01\x02', media_type='audio/pcm'))
    event = json.loads(ws.sent[0])
    assert event['type'] == 'input_audio_buffer.append'
    assert base64.b64decode(event['audio']) == b'\x01\x02'


@pytest.mark.anyio
async def test_connection_send_audio_rejects_non_pcm_media_type() -> None:
    ws = FakeWebSocket([])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    with pytest.raises(UserError, match='require raw PCM audio'):
        await conn.send(BinaryAudio(data=b'RIFF', media_type='audio/wav'))
    assert ws.sent == []


@pytest.mark.anyio
async def test_connection_send_text() -> None:
    ws = FakeWebSocket([])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    await conn.send('hello')
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
async def test_connection_send_tool_result_unsupported_media_raises_with_nothing_sent() -> None:
    """The follow-up user message is built before any frame goes out, so media the wire can't carry
    (here a PDF) raises with the tool result unsent — never a silent degrade or a half-sent round."""
    ws = FakeWebSocket([])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    with pytest.raises(UserError, match='cannot be sent to openai in a realtime session'):
        await conn.send(
            ToolResult(
                tool_call_id='call_1',
                output='See file doc.pdf.',
                content=['This is file doc.pdf:', BinaryContent(data=b'pdf', media_type='application/pdf')],
            )
        )
    assert ws.sent == []


@pytest.mark.anyio
async def test_connection_send_image() -> None:
    ws = FakeWebSocket([])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    await conn.send(BinaryImage(data=b'\x01\x02', media_type='image/png'))
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
    with pytest.raises(UserError, match='OpenAI Realtime does not support object input'):
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
    assert conn._response_active is True  # pyright: ignore[reportPrivateUsage]
    assert conn._pending_response is True  # pyright: ignore[reportPrivateUsage]


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

    events = await collect_codec_events(conn)
    assert events == [
        ResponseDone(
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
    # arrives. A's usage is still accounted, but its `ResponseDone` must be suppressed — otherwise the
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

    events = await collect_codec_events(conn)
    # A's usage is recorded, B keeps streaming, and no `ResponseDone` fired for the superseded A.
    assert [type(event).__name__ for event in events] == ['SessionUsage', 'AudioDelta']
    assert isinstance(events[0], SessionUsage) and events[0].provider_response_id == 'A'
    assert events[1] == AudioDelta(data=b'\x02', item_id='b-item')
    assert not any(isinstance(event, ResponseDone) for event in events)


@pytest.mark.anyio
async def test_late_cancelled_done_resets_cancel_for_the_next_response() -> None:
    # A barge-in cancels response A; B is created before A's late `response.done` lands. That done
    # doesn't match the active response, so `_clear_active_response` must not run — but the settled
    # cancel still has to release the `_cancel_sent` guard, or the user could never interrupt B.
    created_b = json.dumps({'type': 'response.created', 'response': {'id': 'B'}})
    late_a_done = json.dumps(
        {'type': 'response.done', 'response': {'id': 'A', 'status': 'cancelled', 'usage': {'input_tokens': 1}}}
    )
    ws = FakeWebSocket([created_b, late_a_done])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    conn._response_active = True  # pyright: ignore[reportPrivateUsage]
    conn._active_response_id = 'A'  # pyright: ignore[reportPrivateUsage]
    await conn.send(CancelResponse())  # cancel A (barge-in)
    await collect_codec_events(conn)

    await conn.send(CancelResponse())  # interrupt B — must not be swallowed by A's stale cancel flag
    assert [json.loads(frame)['type'] for frame in ws.sent] == ['response.cancel', 'response.cancel']


@pytest.mark.anyio
async def test_cancel_before_response_created_still_suppresses_stragglers() -> None:
    # `send` and socket iteration run in separate tasks, so an immediate interrupt can race the
    # server's `response.created`: the cancel then targets a response with no server-assigned id yet.
    # The later `response.created` names it, and its trailing deltas must still be dropped.
    created_a = json.dumps({'type': 'response.created', 'response': {'id': 'A'}})
    a_audio = json.dumps(
        {
            'type': 'response.output_audio.delta',
            'response_id': 'A',
            'item_id': 'a-item',
            'delta': base64.b64encode(b'\x01').decode('ascii'),
        }
    )
    a_done = json.dumps(
        {'type': 'response.done', 'response': {'id': 'A', 'status': 'cancelled', 'usage': {'input_tokens': 1}}}
    )
    ws = FakeWebSocket([created_a, a_audio, a_done])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    await conn.send(CreateResponse())  # response requested; no `response.created` seen yet
    await conn.send(CancelResponse())  # immediate barge-in before the server names the response

    events = await collect_codec_events(conn)
    assert not any(isinstance(event, AudioDelta) for event in events)


@pytest.mark.anyio
async def test_malformed_usage_on_response_done_still_releases_the_response() -> None:
    # A `response.done` whose usage payload fails validation is surfaced as a recoverable frame error
    # — but it was still the terminal for its response, so the state must settle first: the active
    # response is released and a deferred `response.create` replays, instead of every later request
    # queueing behind a response that already ended.
    done = json.dumps({'type': 'response.done', 'response': {'id': 'A', 'status': 'completed', 'usage': 'bogus'}})
    ws = FakeWebSocket([done])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    conn._response_active = True  # pyright: ignore[reportPrivateUsage]
    conn._active_response_id = 'A'  # pyright: ignore[reportPrivateUsage]
    conn._pending_response = True  # pyright: ignore[reportPrivateUsage]

    events = await collect_codec_events(conn)
    assert any(isinstance(event, RealtimeSessionErrorEvent) and event.recoverable for event in events)
    assert conn._response_active is True  # pyright: ignore[reportPrivateUsage]  # the replayed request
    assert conn._pending_response is False  # pyright: ignore[reportPrivateUsage]
    assert '{"type":"response.create"}' in ws.sent


class _ResetWebSocket(FakeWebSocket):
    """A websocket whose iteration raises a socket-level `OSError` rather than `ConnectionClosed`."""

    async def __aiter__(self) -> AsyncIterator[Any]:
        raise OSError('connection reset by peer')
        yield  # pragma: no cover  (makes this an async generator)


@pytest.mark.anyio
async def test_socket_oserror_is_reported_like_a_drop() -> None:
    # `OSError` is in `transport_errors` for exactly this: a reset escaping `websockets` iteration is
    # the link failing, and must take the same error/reconnect path as `ConnectionClosed` instead of
    # escaping the stream and bypassing the reconnect policy.
    conn = OpenAIRealtimeConnection(_ResetWebSocket([]))  # type: ignore[arg-type]
    events = [event async for event in conn]
    assert len(events) == 1
    error = events[0]
    assert isinstance(error, RealtimeSessionErrorEvent)
    assert error.recoverable is False
    assert 'connection reset by peer' in error.message


@pytest.mark.anyio
async def test_response_done_without_response_object_is_recoverable() -> None:
    # The terminal frame still releases response state, but its missing payload must not finalize a turn.
    ws = FakeWebSocket([json.dumps({'type': 'response.done'})])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    conn._response_active = True  # pyright: ignore[reportPrivateUsage]
    conn._pending_response = True  # pyright: ignore[reportPrivateUsage]
    events = await collect_codec_events(conn)
    assert len(events) == 1
    assert isinstance(events[0], RealtimeSessionErrorEvent)
    assert 'validation errors for union[ResponseDoneEvent,ProtocolResponseDoneEvent]' in events[0].message
    assert ws.sent == ['{"type":"response.create"}']
    assert conn._response_active is True  # pyright: ignore[reportPrivateUsage]
    assert conn._pending_response is False  # pyright: ignore[reportPrivateUsage]


@pytest.mark.anyio
async def test_response_done_without_response_object_sends_nothing_when_none_was_queued() -> None:
    # The same malformed terminal with no deferred request: state is still released, but there is no
    # `response.create` to replay, so nothing goes out.
    ws = FakeWebSocket([json.dumps({'type': 'response.done'})])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    conn._response_active = True  # pyright: ignore[reportPrivateUsage]
    events = await collect_codec_events(conn)
    assert len(events) == 1
    assert isinstance(events[0], RealtimeSessionErrorEvent)
    assert 'validation errors for union[ResponseDoneEvent,ProtocolResponseDoneEvent]' in events[0].message
    assert ws.sent == []
    assert conn._pending_response is False  # pyright: ignore[reportPrivateUsage]


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
    events = await collect_codec_events(conn)
    assert events == [
        InputTranscript(text='hi', is_final=True, item_id='u1'),
        SessionUsage(
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
@pytest.mark.parametrize('usage', [None, {}], ids=['absent', 'empty'])
async def test_transcription_completed_without_usage_emits_only_the_transcript(usage: dict[str, Any] | None) -> None:
    """A final transcript with nothing to report costs no usage event — the counterpart of the case above.

    OpenAI only reports transcription usage when the session asks for it, so a well-formed frame can
    legitimately carry no `usage` at all, or an empty one. Driven directly rather than through a
    cassette because our OpenAI recordings all enable transcription usage; the frames that omit it
    reach us only from protocol clones.
    """
    frame: dict[str, Any] = {
        'type': 'conversation.item.input_audio_transcription.completed',
        'item_id': 'u1',
        'transcript': 'hi',
    }
    if usage is not None:
        frame['usage'] = usage
    conn = OpenAIRealtimeConnection(FakeWebSocket([json.dumps(frame)]))  # type: ignore[arg-type]

    assert await collect_codec_events(conn) == [InputTranscript(text='hi', is_final=True, item_id='u1')]


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
    events = await collect_codec_events(conn)
    assert events == [
        SessionUsage(
            usage=RequestUsage(input_tokens=3, output_tokens=2),
            provider_response_id='resp-1',
            finish_reason='stop',
        ),
        ResponseDone(
            interrupted=False,
            provider_response_id='resp-1',
            finish_reason='stop',
            provider_details={'status': 'completed'},
        ),
    ]


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
    events = await collect_codec_events(conn)
    # function-call-only → no ResponseDone, but usage is still surfaced
    assert events == [
        SessionUsage(
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
    session = RealtimeSession(
        connection, model=FakeRealtimeModel(connection, system='openai'), tool_manager=make_tool_manager(runner)
    )
    async with session:
        _ = await collect_session_events(session)

    messages = session.all_messages()
    assert len(messages) == 3
    tool_response, tool_result, answer = messages
    assert isinstance(tool_response, ModelResponse)
    assert tool_response.parts == [ToolCallPart(tool_name='get_weather', args='{}', tool_call_id='call-1')]
    assert tool_response.usage == RequestUsage()
    assert isinstance(tool_result, ModelRequest)
    assert isinstance(tool_result.parts[0], ToolReturnPart)
    assert isinstance(answer, ModelResponse)
    assert answer.parts == [SpeechPart(speaker='assistant', transcript='Sunny')]
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
        model=FakeRealtimeModel(connection, model_name='gpt-realtime', system='openai'),
        tool_manager=make_tool_manager(),
    )
    async with session:
        _ = await collect_session_events(session)

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
    assert (speech.id, speech.provider_name) == (None, None)


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
    session = RealtimeSession(
        connection, model=FakeRealtimeModel(connection, system='openai'), tool_manager=make_tool_manager()
    )

    async with session:
        _ = await collect_session_events(session)

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
    assert isinstance(error, RealtimeSessionErrorEvent)
    assert error.recoverable is False


class _ExpiredWebSocket(FakeWebSocket):
    """Closed normally by the server, with the code and reason OpenAI sends at its duration cap."""

    close_code = 1001
    close_reason = 'Your session hit the maximum duration of 60 minutes.'


@pytest.mark.anyio
async def test_clean_close_is_reported_as_a_fatal_error() -> None:
    """A *normal* close ends the stream with an error carrying the server's own explanation.

    `websockets` ends iteration silently on a 1000/1001 close and raises only on an abnormal one, so
    without this a hangup is indistinguishable from a conversation that simply finished. It isn't:
    held against the live API, an idle `gpt-realtime` session is closed by the server after exactly
    60 minutes with `1001 Your session hit the maximum duration of 60 minutes.`.
    """
    conn = OpenAIRealtimeConnection(_ExpiredWebSocket([]))  # type: ignore[arg-type]
    assert [event async for event in conn] == [
        RealtimeSessionErrorEvent(
            message=(
                'OpenAI Realtime connection closed: received 1001 Your session hit the maximum duration of 60 minutes.'
            ),
            recoverable=False,
        )
    ]


@pytest.mark.anyio
async def test_clean_close_reconnects_when_a_policy_is_configured() -> None:
    """Hitting the session cap is exactly what a reconnect policy is for, so it re-dials and resumes."""
    transcript = json.dumps({'type': 'response.audio_transcript.done', 'transcript': 'still here'})
    replacements = iter([FakeWebSocket([transcript])])

    async def dial() -> Any:
        try:
            return next(replacements)
        except StopIteration:
            raise OSError('server is down')

    conn = OpenAIRealtimeConnection(
        _ExpiredWebSocket([]),  # type: ignore[arg-type]
        dial=dial,
        reconnect={'base_delay': 0.0, 'max_attempts': 1},
    )
    events = await collect_codec_events(conn)
    assert events == [
        RealtimeSessionReconnectEvent(state_restored=False),
        OutputTranscript(text='still here', is_final=True),
    ]


@pytest.mark.anyio
async def test_reconnect_budget_bounds_a_flapping_server() -> None:
    # `max_attempts` bounds the retries for one drop and resets whenever a dial succeeds, so a server
    # that accepts a connection and immediately drops it would otherwise be reconnected forever. The
    # session-wide budget is what makes that terminate.
    policy = {'max_attempts': 1, 'max_reconnects': 2, 'base_delay': 0, 'jitter': False}
    dials = 0

    async def dial() -> Any:
        nonlocal dials
        dials += 1
        return FakeWebSocket([])

    conn = rt_openai.OpenAIRealtimeConnection(FakeWebSocket([]), dial=dial, reconnect=policy)  # type: ignore[arg-type]
    assert await conn._try_reconnect() is True  # pyright: ignore[reportPrivateUsage]
    assert await conn._try_reconnect() is True  # pyright: ignore[reportPrivateUsage]
    # Budget spent: further drops are terminal rather than re-dialed, and we stop dialing entirely.
    assert await conn._try_reconnect() is False  # pyright: ignore[reportPrivateUsage]
    assert dials == 2


async def test_reconnects_on_drop_and_resumes() -> None:
    transcript = json.dumps({'type': 'response.audio_transcript.done', 'transcript': 'hi'})
    replacements = iter([FakeWebSocket([transcript])])

    async def dial() -> Any:
        try:
            return next(replacements)
        except StopIteration:
            # Once the replacement has said its piece and hung up, the server stays down, so the
            # stream terminates instead of re-dialing until the session's reconnect budget runs out.
            raise OSError('server is down')

    # The initial connection drops; reconnect re-dials and resumes streaming.
    conn = OpenAIRealtimeConnection(
        DroppingWebSocket([]),  # type: ignore[arg-type]
        dial=dial,
        reconnect={'base_delay': 0.0, 'max_attempts': 1},
    )
    events = await collect_codec_events(conn)
    assert events == [RealtimeSessionReconnectEvent(state_restored=False), OutputTranscript(text='hi', is_final=True)]


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
        self.headers: list[dict[str, str]] = []

    def __call__(self, url: str, *, additional_headers: dict[str, str] | None = None) -> Any:
        self.headers.append(dict(additional_headers or {}))
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


@pytest.mark.anyio
async def test_connect_reconnect_closes_previous_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    # A reconnect through `connect()`'s own dial must close the dropped connection before opening the
    # next, and teardown closes the current one — so sockets don't accumulate across drops.
    transcript = json.dumps({'type': 'response.audio_transcript.done', 'transcript': 'hi'})
    dropped = _DropAfterHandshake([_created(), _updated()])
    good = FakeWebSocket([_created(), _updated(), transcript])
    connect = _RecordingConnect([dropped, good])
    monkeypatch.setattr(rt_openai.websockets, 'connect', connect)

    model = OpenAIRealtimeModel('gpt-realtime', settings={'reconnect': {'base_delay': 0.0, 'max_attempts': 1}})
    async with _connect(model, 'x') as conn:
        events = await collect_codec_events(conn)

    assert events == [RealtimeSessionReconnectEvent(state_restored=False), OutputTranscript(text='hi', is_final=True)]
    assert connect.closed == [dropped, good]


@pytest.mark.anyio
async def test_connect_webrtc_reconnect_closes_previous_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    updated = json.dumps({'type': 'session.updated', 'session': {'model': 'gpt-realtime'}})
    transcript = json.dumps({'type': 'response.audio_transcript.done', 'transcript': 'hi'})
    dropped = _DropAfterHandshake([updated])
    good = FakeWebSocket([updated, transcript])
    connect = _RecordingConnect([dropped, good])
    monkeypatch.setattr(rt_openai.websockets, 'connect', connect)
    model = OpenAIRealtimeModel(
        'gpt-realtime',
        provider=OpenAIProvider(api_key='k'),
        settings={'reconnect': {'base_delay': 0.0, 'max_attempts': 1}},
    )

    async with model.connect_webrtc(
        WebRTCSession(provider_name='openai', session_id='rtc_reconnect'),
        messages=[],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    ) as conn:
        events = await collect_codec_events(conn, sideband=True)

    assert events == [RealtimeSessionReconnectEvent(state_restored=False), OutputTranscript(text='hi', is_final=True)]
    assert connect.closed == [dropped, good]


@pytest.mark.anyio
async def test_reconnect_policy_follows_model_settings_layering(monkeypatch: pytest.MonkeyPatch) -> None:
    """`reconnect` layers like any other model setting: the model-level default applies to every
    session, and a per-session policy overrides it."""
    default_policy: rt_openai.ReconnectPolicy = {'max_attempts': 1}
    session_policy: rt_openai.ReconnectPolicy = {'max_attempts': 2}
    model = OpenAIRealtimeModel('gpt-realtime', settings={'reconnect': default_policy})

    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(FakeWebSocket([_created(), _updated()])))
    async with _connect(model, 'x') as conn:
        assert conn._reconnect is default_policy  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(FakeWebSocket([_created(), _updated()])))
    async with _connect(model, 'x', model_settings={'reconnect': session_policy}) as conn:
        assert conn._reconnect is session_policy  # pyright: ignore[reportPrivateUsage]


@pytest.mark.anyio
async def test_reconnect_updates_server_reported_model(monkeypatch: pytest.MonkeyPatch) -> None:
    initial_created = json.dumps({'type': 'session.created', 'session': {'model': 'initial-model'}})
    reconnected_created = json.dumps({'type': 'session.created', 'session': {'model': 'replacement-model'}})
    dropped = _DropAfterHandshake([initial_created, _updated()])
    good = FakeWebSocket([reconnected_created, _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', _RecordingConnect([dropped, good]))
    model = OpenAIRealtimeModel('requested-model', settings={'reconnect': {'base_delay': 0.0, 'max_attempts': 1}})

    async with _connect(model, 'x') as conn:
        await collect_codec_events(conn)
        assert conn.model_name == 'replacement-model'


def test_output_text_events_keep_item_id() -> None:
    assert map_event({'type': 'response.output_text.delta', 'delta': 'hi', 'item_id': 'item-1'}) == (
        OutputTranscript(text='hi', is_final=False, item_id='item-1', output_text=True)
    )
    assert map_event({'type': 'response.output_text.done', 'text': 'hi', 'item_id': 'item-1'}) == (
        OutputTranscript(text='hi', is_final=True, item_id='item-1', output_text=True)
    )


@pytest.mark.anyio
async def test_reconnect_refreshes_async_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    key_calls = 0

    async def provide_key() -> str:
        nonlocal key_calls
        key_calls += 1
        return 'sk-initial' if key_calls == 1 else 'sk-refreshed'

    dropped = _DropAfterHandshake([_created(), _updated()])
    good = FakeWebSocket([_created(), _updated()])
    connect = _RecordingConnect([dropped, good])
    monkeypatch.setattr(rt_openai.websockets, 'connect', connect)
    model = OpenAIRealtimeModel(
        'gpt-realtime',
        provider=OpenAIProvider(openai_client=AsyncOpenAI(api_key=provide_key)),
        settings={'reconnect': {'base_delay': 0.0, 'max_attempts': 1}},
    )

    async with _connect(model, 'x') as conn:
        await collect_codec_events(conn)

    assert connect.headers[:2] == [
        {'Authorization': 'Bearer sk-initial'},
        {'Authorization': 'Bearer sk-refreshed'},
    ]


@pytest.mark.anyio
async def test_reconnect_gives_up_after_max_attempts() -> None:
    async def dial() -> Any:
        raise OSError('still down')  # an expected dial failure (network unreachable)

    conn = OpenAIRealtimeConnection(
        DroppingWebSocket([]),  # type: ignore[arg-type]
        dial=dial,
        reconnect={'max_attempts': 2, 'base_delay': 0.0, 'jitter': False},
    )
    events = [e async for e in conn]
    assert len(events) == 1
    error = events[0]
    assert isinstance(error, RealtimeSessionErrorEvent)
    assert error.recoverable is False
    assert 'reconnect failed' in error.message


@pytest.mark.anyio
async def test_reconnect_handshake_failure_consumes_an_attempt() -> None:
    # A re-dial rejected with an `error` frame, answered with an unparsable frame, or that times out
    # raises `RealtimeHandshakeError` from `expect_event`. `map_connect_errors` only wraps the *initial*
    # dial, so without this the exception would escape the reconnect loop instead of consuming an
    # attempt and eventually surfacing as a non-recoverable session error.
    dials = 0

    async def dial() -> Any:
        nonlocal dials
        dials += 1
        raise RealtimeHandshakeError({'message': 'session expired'})

    conn = OpenAIRealtimeConnection(
        DroppingWebSocket([]),  # type: ignore[arg-type]
        dial=dial,
        reconnect={'max_attempts': 2, 'base_delay': 0.0, 'jitter': False},
    )
    events = [e async for e in conn]
    assert dials == 2
    assert len(events) == 1
    error = events[0]
    assert isinstance(error, RealtimeSessionErrorEvent)
    assert error.recoverable is False
    assert 'reconnect failed' in error.message


@pytest.mark.anyio
async def test_reconnect_replays_a_deferred_response_request() -> None:
    # A `response.create` deferred behind an active response (e.g. a tool result that landed
    # mid-answer) is released by that response's `response.done` — which never arrives when the socket
    # dies first. The fresh socket has just replayed the call, so the request goes out on it rather
    # than being dropped, which would leave the session waiting for a turn that can never start.
    replacement = FakeWebSocket([])
    replacements = iter([replacement])

    async def dial() -> Any:
        try:
            return next(replacements)
        except StopIteration:
            # The replacement said its piece and hung up; the server stays down so the stream ends.
            raise OSError('server is down')

    conn = OpenAIRealtimeConnection(
        DroppingWebSocket([]),  # type: ignore[arg-type]
        dial=dial,
        reconnect={'base_delay': 0.0, 'max_attempts': 1},
    )
    conn._response_active = True  # pyright: ignore[reportPrivateUsage]
    conn._pending_response = True  # pyright: ignore[reportPrivateUsage]

    events = [e async for e in conn]
    assert [type(e).__name__ for e in events] == ['RealtimeSessionReconnectEvent', 'RealtimeSessionErrorEvent']
    assert replacement.sent == ['{"type":"response.create"}']
    assert conn._pending_response is False  # pyright: ignore[reportPrivateUsage]
    assert conn._response_active is True  # pyright: ignore[reportPrivateUsage]


def test_openai_connection_does_not_restore_in_flight_state_on_reconnect() -> None:
    # OpenAI reconnects by replaying finalized history only, so the session settles the in-flight turn.
    conn = OpenAIRealtimeConnection(FakeWebSocket([]))  # type: ignore[arg-type]
    assert conn.reconnect_restores_in_flight_state is False


@pytest.mark.anyio
async def test_reconnect_re_solicits_a_response_that_never_started() -> None:
    # A `response.create` sent when idle goes out immediately (not deferred), so `_pending_response` is
    # False while the connection waits for its `response.created`. If the socket drops in that window the
    # answer the caller is waiting on is neither deferred nor streaming — without re-asking, the re-dial
    # replays the finalized call but the turn never starts. `_response_active` without `_response_started`
    # marks exactly that response, so it is re-solicited on the fresh socket.
    replacement = FakeWebSocket([])
    replacements = iter([replacement])

    async def dial() -> Any:
        try:
            return next(replacements)
        except StopIteration:
            raise OSError('server is down')

    conn = OpenAIRealtimeConnection(
        DroppingWebSocket([]),  # type: ignore[arg-type]
        dial=dial,
        reconnect={'base_delay': 0.0, 'max_attempts': 1},
    )
    conn._response_active = True  # pyright: ignore[reportPrivateUsage]
    conn._response_started = False  # pyright: ignore[reportPrivateUsage]

    events = [e async for e in conn]
    assert [type(e).__name__ for e in events] == ['RealtimeSessionReconnectEvent', 'RealtimeSessionErrorEvent']
    assert replacement.sent == ['{"type":"response.create"}']


@pytest.mark.anyio
async def test_reconnect_does_not_re_solicit_a_cancelled_response() -> None:
    # A response the caller cancelled (barge-in) before its `response.created` arrived is `_response_active`
    # without `_response_started`, but `_cancel_sent` marks it stopped. Re-asking would resurrect a
    # response the user explicitly interrupted, producing unwanted output after the barge-in, so it is
    # not replayed on the fresh socket.
    replacement = FakeWebSocket([])
    replacements = iter([replacement])

    async def dial() -> Any:
        try:
            return next(replacements)
        except StopIteration:
            raise OSError('server is down')

    conn = OpenAIRealtimeConnection(
        DroppingWebSocket([]),  # type: ignore[arg-type]
        dial=dial,
        reconnect={'base_delay': 0.0, 'max_attempts': 1},
    )
    conn._response_active = True  # pyright: ignore[reportPrivateUsage]
    conn._response_started = False  # pyright: ignore[reportPrivateUsage]
    conn._cancel_sent = True  # pyright: ignore[reportPrivateUsage]

    events = [e async for e in conn]
    assert [type(e).__name__ for e in events] == ['RealtimeSessionReconnectEvent', 'RealtimeSessionErrorEvent']
    assert not any(json.loads(frame).get('type') == 'response.create' for frame in replacement.sent)


@pytest.mark.anyio
async def test_reconnect_does_not_re_solicit_a_response_that_was_streaming() -> None:
    # A response already confirmed by its `response.created` (`_response_started`) had output the session
    # settles as interrupted; re-asking for it would make the model answer a turn it was cut off mid-way
    # through, contradicting the state-lost contract of staying quiet until the next input. So the fresh
    # socket carries the replayed call but no `response.create`.
    replacement = FakeWebSocket([])
    replacements = iter([replacement])

    async def dial() -> Any:
        try:
            return next(replacements)
        except StopIteration:
            raise OSError('server is down')

    conn = OpenAIRealtimeConnection(
        DroppingWebSocket([]),  # type: ignore[arg-type]
        dial=dial,
        reconnect={'base_delay': 0.0, 'max_attempts': 1},
    )
    conn._response_active = True  # pyright: ignore[reportPrivateUsage]
    conn._response_started = True  # pyright: ignore[reportPrivateUsage]

    events = [e async for e in conn]
    assert [type(e).__name__ for e in events] == ['RealtimeSessionReconnectEvent', 'RealtimeSessionErrorEvent']
    assert not any(json.loads(frame).get('type') == 'response.create' for frame in replacement.sent)


@pytest.mark.anyio
async def test_reconnect_replay_failure_consumes_an_attempt() -> None:
    # The replayed `response.create` goes out on a socket that has only just come up, which can drop
    # before the frame reaches it. That has to look like any other failed reconnect — consuming an
    # attempt and ending in a non-recoverable session error — rather than escaping the reconnect loop,
    # which runs outside `__aiter__`'s transport guard.
    class _SendFailsWebSocket(FakeWebSocket):
        async def send(self, data: str) -> None:
            raise ConnectionResetError('connection reset by peer')

    dials = 0

    async def dial() -> Any:
        nonlocal dials
        dials += 1
        return _SendFailsWebSocket([])

    conn = OpenAIRealtimeConnection(
        DroppingWebSocket([]),  # type: ignore[arg-type]
        dial=dial,
        reconnect={'max_attempts': 2, 'base_delay': 0.0, 'jitter': False},
    )
    conn._response_active = True  # pyright: ignore[reportPrivateUsage]
    conn._pending_response = True  # pyright: ignore[reportPrivateUsage]

    events = [e async for e in conn]
    assert dials == 2
    # Still queued, so a later attempt would replay it rather than silently losing the turn.
    assert conn._pending_response is True  # pyright: ignore[reportPrivateUsage]
    assert len(events) == 1
    error = events[0]
    assert isinstance(error, RealtimeSessionErrorEvent)
    assert error.recoverable is False
    assert 'reconnect failed' in error.message


@pytest.mark.anyio
async def test_response_done_settles_a_response_whose_id_was_never_announced() -> None:
    # Between `response.create` and its `response.created`, the active response has no id — and a
    # protocol clone that omits the id from `response.created` never leaves that window. A terminal
    # naming a response we can't prove is stale must still settle it; otherwise `_response_active`
    # stays set for the life of the connection and the model never speaks again.
    frame = json.dumps({'type': 'response.done', 'response': {'id': 'resp_1', 'status': 'completed', 'output': []}})
    ws = FakeWebSocket([frame])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    conn._response_active = True  # pyright: ignore[reportPrivateUsage]
    assert conn._active_response_id is None  # pyright: ignore[reportPrivateUsage]

    await collect_codec_events(conn)
    assert conn._response_active is False  # pyright: ignore[reportPrivateUsage]
    await conn._request_response()  # pyright: ignore[reportPrivateUsage]
    assert ws.sent == ['{"type":"response.create"}']


@pytest.mark.anyio
async def test_malformed_response_done_still_releases_the_response() -> None:
    # A `response.done` whose `response` payload is the wrong shape fails to map, which surfaces as a
    # recoverable frame error. The frame is still the only terminal that response will ever get, so the
    # bookkeeping runs first: otherwise `_response_active` stays set for the life of the connection and
    # every later request to speak queues behind a response that already ended.
    ws = FakeWebSocket([json.dumps({'type': 'response.done', 'response': 'bad'})])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    conn._response_active = True  # pyright: ignore[reportPrivateUsage]
    conn._active_response_id = 'resp_1'  # pyright: ignore[reportPrivateUsage]

    events = await collect_codec_events(conn)
    assert [type(e).__name__ for e in events] == ['RealtimeSessionErrorEvent']
    assert conn._response_active is False  # pyright: ignore[reportPrivateUsage]
    # The session can speak again, rather than only ever deferring.
    await conn._request_response()  # pyright: ignore[reportPrivateUsage]
    assert ws.sent == ['{"type":"response.create"}']


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
        reconnect={'base_delay': 0.0},
    )
    with pytest.raises(RuntimeError, match='boom'):
        _ = [e async for e in conn]


def _audio_delta(item_id: str, content_index: int | None = None, *, audio_bytes: int = 1) -> str:
    data: dict[str, Any] = {
        'type': 'response.output_audio.delta',
        'item_id': item_id,
        'delta': base64.b64encode(b'\x01' * audio_bytes).decode('ascii'),
    }
    if content_index is not None:
        data['content_index'] = content_index
    return json.dumps(data)


@pytest.mark.anyio
async def test_truncate_uses_item_tracked_from_audio_delta() -> None:
    ws = FakeWebSocket([_audio_delta('item_7', content_index=2, audio_bytes=480)])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    _ = [e async for e in conn]  # consume the delta → captures the current output item
    await conn.send(TruncateOutput(audio_end_ms=1200))
    assert json.loads(ws.sent[0]) == {
        'type': 'conversation.item.truncate',
        'item_id': 'item_7',
        'content_index': 2,
        'audio_end_ms': 10,
    }


@pytest.mark.anyio
async def test_truncate_in_bounds_audio_end_passes_through() -> None:
    ws = FakeWebSocket([_audio_delta('item_7', audio_bytes=960)])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    _ = [e async for e in conn]
    await conn.send(TruncateOutput(audio_end_ms=10))
    assert json.loads(ws.sent[0])['audio_end_ms'] == 10


@pytest.mark.anyio
async def test_truncate_accumulates_generated_audio_deltas() -> None:
    ws = FakeWebSocket([_audio_delta('item_7', audio_bytes=240), _audio_delta('item_7', audio_bytes=240)])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    _ = [e async for e in conn]
    await conn.send(TruncateOutput(audio_end_ms=20))
    assert json.loads(ws.sent[0])['audio_end_ms'] == 10


@pytest.mark.anyio
async def test_truncate_resets_generated_audio_between_items() -> None:
    ws = FakeWebSocket([_audio_delta('item_1', audio_bytes=480), _audio_delta('item_2', audio_bytes=240)])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    _ = [e async for e in conn]
    await conn.send(TruncateOutput(audio_end_ms=20))
    assert json.loads(ws.sent[0]) == {
        'type': 'conversation.item.truncate',
        'item_id': 'item_2',
        'content_index': 0,
        'audio_end_ms': 5,
    }


@pytest.mark.anyio
async def test_truncate_resets_generated_audio_between_responses() -> None:
    done = json.dumps({'type': 'response.done', 'response': {'status': 'completed', 'output': []}})
    ws = FakeWebSocket([_audio_delta('item_7', audio_bytes=480), done, _audio_delta('item_7', audio_bytes=240)])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    _ = [e async for e in conn]
    await conn.send(TruncateOutput(audio_end_ms=20))
    assert json.loads(ws.sent[0])['audio_end_ms'] == 5


@pytest.mark.anyio
async def test_truncate_sideband_connection_does_not_clamp() -> None:
    ws = FakeWebSocket([_audio_delta('item_7')])
    conn = OpenAIRealtimeConnection(ws, observes_output_audio=False)  # type: ignore[arg-type]
    _ = [e async for e in conn]
    await conn.send(TruncateOutput(audio_end_ms=1200))
    assert json.loads(ws.sent[0])['audio_end_ms'] == 1200


def _playback(event_type: str, response_id: str = 'resp_1') -> str:
    return json.dumps({'type': event_type, 'response_id': response_id})


def _content_part_added(item_id: str, part_type: str = 'audio', content_index: int = 0) -> str:
    return json.dumps(
        {
            'type': 'response.content_part.added',
            'item_id': item_id,
            'content_index': content_index,
            'part': {'type': part_type, 'transcript': ''},
        }
    )


@pytest.mark.anyio
async def test_sideband_tracks_audio_part_and_playback_boundaries() -> None:
    ws = FakeWebSocket(
        [
            _content_part_added('item_sideband', content_index=2),
            _content_part_added('ignored', part_type='text'),
            _playback('output_audio_buffer.started'),
        ]
    )
    conn = OpenAIRealtimeConnection(ws, observes_output_audio=False)  # type: ignore[arg-type]
    assert await collect_codec_events(conn, sideband=True) == [RealtimeOutputSpeechStartEvent()]
    # A barge-in truncation during playback names the tracked audio item and index.
    await conn.send(TruncateOutput(audio_end_ms=1200))
    assert json.loads(ws.sent[0]) == {
        'type': 'conversation.item.truncate',
        'item_id': 'item_sideband',
        'content_index': 2,
        'audio_end_ms': 1200,
    }


@pytest.mark.anyio
async def test_sideband_keeps_item_playing_past_response_done() -> None:
    """On a sideband, `response.done` must not retire the output item while its audio still plays.

    The provider keeps streaming buffered audio to the browser after generation finishes, so a
    barge-in `interrupt(played_ms=...)` in that tail still has to name the playing item; it is
    retired when the provider reports playback over (`output_audio_buffer.stopped`/`.cleared`).
    """
    ws = FakeWebSocket([_content_part_added('item_tail'), _playback('output_audio_buffer.started')])
    conn = OpenAIRealtimeConnection(ws, observes_output_audio=False)  # type: ignore[arg-type]
    assert await collect_codec_events(conn, sideband=True) == [RealtimeOutputSpeechStartEvent()]
    # Generation finished (`response.done` runs this) while playback continues: the item survives...
    conn._clear_active_response()  # pyright: ignore[reportPrivateUsage]
    await conn.send(TruncateOutput(audio_end_ms=800))
    assert json.loads(ws.sent[0])['item_id'] == 'item_tail'


@pytest.mark.anyio
async def test_sideband_barge_in_clear_keeps_item_while_response_active() -> None:
    """`output_audio_buffer.cleared` mid-response (our barge-in clear) must not retire the item.

    The response is still open, so the truncation that follows the barge-in still names it; only an
    end frame arriving *after* `response.done` retires the item.
    """
    ws = FakeWebSocket(
        [
            json.dumps({'type': 'response.created', 'response': {'id': 'resp_active'}}),
            _content_part_added('item_active'),
            _playback('output_audio_buffer.started'),
            _playback('output_audio_buffer.cleared'),
        ]
    )
    conn = OpenAIRealtimeConnection(ws, observes_output_audio=False)  # type: ignore[arg-type]
    assert await collect_codec_events(conn, sideband=True) == [
        RealtimeOutputSpeechStartEvent(),
        RealtimeOutputSpeechEndEvent(),
    ]
    await conn.send(TruncateOutput(audio_end_ms=800))
    assert json.loads(ws.sent[0])['item_id'] == 'item_active'


@pytest.mark.anyio
async def test_sideband_playback_end_retires_output_item() -> None:
    """Once playback ends after the response closed, there is nothing left to truncate."""
    ws = FakeWebSocket(
        [
            _content_part_added('item_done'),
            _playback('output_audio_buffer.started'),
            _playback('output_audio_buffer.stopped'),
        ]
    )
    conn = OpenAIRealtimeConnection(ws, observes_output_audio=False)  # type: ignore[arg-type]
    assert await collect_codec_events(conn, sideband=True) == [
        RealtimeOutputSpeechStartEvent(),
        RealtimeOutputSpeechEndEvent(),
    ]
    await conn.send(TruncateOutput(audio_end_ms=800))
    assert ws.sent == []


@pytest.mark.anyio
async def test_websocket_clear_active_response_retires_output_item() -> None:
    """A connection that observes output audio retires the item on `response.done` as before."""
    ws = FakeWebSocket([_audio_delta('item_ws')])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    _ = await collect_codec_events(conn)
    conn._clear_active_response()  # pyright: ignore[reportPrivateUsage]
    await conn.send(TruncateOutput(audio_end_ms=800))
    assert ws.sent == []


@pytest.mark.anyio
async def test_sideband_cancel_clears_playback_once() -> None:
    ws = FakeWebSocket([_playback('output_audio_buffer.started')])
    conn = OpenAIRealtimeConnection(ws, observes_output_audio=False)  # type: ignore[arg-type]
    assert await collect_codec_events(conn, sideband=True) == [RealtimeOutputSpeechStartEvent()]
    await conn.send(CancelResponse())
    await conn.send(CancelResponse())
    assert [json.loads(frame)['type'] for frame in ws.sent] == ['output_audio_buffer.clear']


@pytest.mark.anyio
async def test_sideband_suppresses_playback_start_for_cancelled_response() -> None:
    """A barge-in cancels a response whose `output_audio_buffer.started` is still in flight.

    Accepting that start boundary would mark playback active and report the model as speaking for a
    response the user already interrupted, so it is dropped like any other cancelled straggler. Its
    matching stop boundary is still processed (it is never a straggler) so no playback state lingers.
    """
    ws = FakeWebSocket(
        [
            _playback('output_audio_buffer.started', response_id='resp-1'),
            _playback('output_audio_buffer.stopped', response_id='resp-1'),
        ]
    )
    conn = OpenAIRealtimeConnection(ws, observes_output_audio=False)  # type: ignore[arg-type]
    conn._response_active = True  # pyright: ignore[reportPrivateUsage]
    conn._active_response_id = 'resp-1'  # pyright: ignore[reportPrivateUsage]
    await conn.send(CancelResponse())  # cancels resp-1 and suppresses its stragglers

    # No `RealtimeOutputSpeechStartEvent` (and thus no paired end event): the start was suppressed.
    assert await collect_codec_events(conn, sideband=True) == []
    assert conn._output_audio_playing is False  # pyright: ignore[reportPrivateUsage]


@pytest.mark.anyio
async def test_websocket_connection_ignores_provider_playback_boundaries() -> None:
    ws = FakeWebSocket([_playback('output_audio_buffer.started'), _playback('output_audio_buffer.stopped')])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    assert await collect_codec_events(conn) == []


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
    # `interrupt(played_ms=...)` before the next turn's first audio delta doesn't truncate a stale item.
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
        self._q.put_nowait(json.dumps(sdk_frame(obj)))

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
async def test_create_response_waits_for_cancelled_response_done() -> None:
    ws = PushWebSocket()
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    task = asyncio.create_task(_drain(conn))

    ws.push({'type': 'response.created'})  # a response is now generating
    await _settle()

    await conn.send(CancelResponse())
    await conn.send(CreateResponse())
    await _settle()
    assert 'response.create' not in ws.sent_types()

    ws.push({'type': 'response.done', 'response': {'status': 'cancelled', 'output': []}})
    await _settle()
    assert ws.sent_types().count('response.create') == 1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.anyio
async def test_deferred_response_dropped_when_active_response_cancelled_by_server() -> None:
    ws = PushWebSocket()
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    task = asyncio.create_task(_drain(conn))

    ws.push({'type': 'response.created'})  # a response is now generating
    await _settle()

    await conn.send(ToolResult(tool_call_id='c1', output='done'))
    await _settle()
    assert 'response.create' not in ws.sent_types()

    # The user barges in and the *server* cancels the active response (no client `interrupt()`): a new
    # user turn is starting, so the deferred response must not replay over it.
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
    await conn.send('new turn')
    ws.push({'type': 'response.done', 'response': {'id': 'resp_old', 'status': 'cancelled', 'output': []}})
    await _settle()
    ws.push({'type': 'response.created', 'response': {'id': 'resp_new'}})
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
        events = await collect_codec_events(conn)
    assert events == [
        ResponseDone(
            interrupted=False,
            finish_reason='stop',
            provider_details={'status': 'completed'},
        )
    ]


async def test_agent_realtime_session_advertises_only_visible_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    # With no deferred tools left to search for, tool search registers no `search_tools` affordance
    # either, so the wire carries exactly the tools the agent declared — pinning that
    # `_open_realtime_session` narrows `function_tools` to `declared_function_tools` before dialing.
    ws = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(rt_openai.websockets, 'connect', FakeConnect(ws))

    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    def visible_tool() -> str:  # pragma: no cover — never called, only advertised
        return 'visible'

    model = OpenAIRealtimeModel('gpt-realtime', provider=OpenAIProvider(api_key='k'))
    async with agent.realtime(model).session():
        pass

    update = json.loads(ws.sent[0])
    assert update['type'] == 'session.update'
    assert [tool['name'] for tool in update['session']['tools']] == ['visible_tool']


async def test_agent_realtime_session_rejects_native_tools() -> None:
    # OpenAI realtime supports no native tools, so a native tool with no local fallback fails up front,
    # before dialing — via the same native ↔ local-tool swap the classic agent-run path applies, so the
    # error points at `local=`.
    agent: Agent[None, str] = Agent()
    with pytest.raises(
        UserError,
        match=r"not supported by this model.*WebSearch\(local='duckduckgo'\)",
    ):
        async with agent.realtime(
            OpenAIRealtimeModel('gpt-realtime'), capabilities=[NativeTool(WebSearchTool())]
        ).session():
            pass  # pragma: no cover


# --- provider resolution & capabilities -------------------------------------------------------


def test_realtime_websocket_url_derivation() -> None:
    # The default OpenAI HTTP base URL maps to the documented realtime WebSocket URL.
    assert realtime_websocket_url('https://api.openai.com/v1/') == 'wss://api.openai.com/v1/realtime'
    # A custom (e.g. self-hosted, non-TLS) base URL keeps its host/path and swaps the scheme.
    assert realtime_websocket_url('http://localhost:8000/v1') == 'ws://localhost:8000/v1/realtime'
    # A base URL with neither scheme is left untouched apart from the appended path.
    assert realtime_websocket_url('localhost:8000/v1') == 'localhost:8000/v1/realtime'
    # The `model` parameter is merged into the query string.
    assert (
        realtime_websocket_url('https://api.openai.com/v1', model='gpt-realtime')
        == 'wss://api.openai.com/v1/realtime?model=gpt-realtime'
    )
    # A base URL carrying its own query keeps it, with `/realtime` landing on the path — not appended
    # after the query into the wrong endpoint — and `model` merged alongside.
    assert (
        realtime_websocket_url('https://host/v1?api-version=x', model='gpt/rt')
        == 'wss://host/v1/realtime?api-version=x&model=gpt%2Frt'
    )
    # A fragment is split off before the path and query are built, and reattached last. Left in place
    # it would swallow both — the handshake would target `/v1` and never send `model`.
    assert (
        realtime_websocket_url('https://host/v1?api-version=x#section', model='gpt-realtime')
        == 'wss://host/v1/realtime?api-version=x&model=gpt-realtime#section'
    )
    assert realtime_websocket_url('https://host/v1#section') == 'wss://host/v1/realtime#section'


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
        profile.get('supports_text_output'),
        profile.get('supports_session_seeding'),
        profile.get('supports_seeding_images'),
        profile.get('supports_seeding_audio'),
    ) == (True, True, True, True, True, True, True, True)
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


def test_user_profile_layer_merges_over_the_provider_and_accepts_a_callable() -> None:
    # The third resolution layer, mirroring `Model.profile`: a partial dict merged over the resolved
    # profile, or a callable that receives it and returns the profile to use.
    merged = OpenAIRealtimeModel(
        'gpt-realtime', profile=RealtimeModelProfile(supports_thinking=True, supports_image_input=False)
    ).profile
    assert merged.get('supports_thinking') is True  # the provider says `False` for GA `gpt-realtime`
    assert merged.get('supports_image_input') is False  # overrides the provider's `True`
    assert merged.get('supports_interruption') is True  # untouched keys still come from the provider

    def only_thinking(resolved: RealtimeModelProfile) -> RealtimeModelProfile:
        assert resolved.get('supports_interruption') is True  # the resolved profile is handed in
        return RealtimeModelProfile(supports_thinking=True)

    replaced = OpenAIRealtimeModel('gpt-realtime', profile=only_thinking).profile
    assert replaced.get('supports_thinking') is True
    # The callable replaces rather than merges, so a claim it drops is gone (and reads as the absent
    # default, `False`, exactly as a provider that never made the claim would).
    assert replaced.get('supports_interruption', False) is False


def test_user_profile_layer_is_still_intersected_with_the_model_class_native_tools() -> None:
    # `profile=` is a layer, not an escape from the final intersection: claiming a native tool the model
    # class doesn't implement can't make it usable, exactly as on a standard `Model`.
    profile = OpenAIRealtimeModel(
        'gpt-realtime', profile=RealtimeModelProfile(supported_native_tools=frozenset({WebSearchTool}))
    ).profile
    assert profile.get('supported_native_tools') == frozenset()


def test_user_profile_corrects_a_thinking_claim_defeated_by_the_model_name() -> None:
    # `supports_thinking` is inferred from the model name, which on Azure is the *deployment* name — a
    # user-chosen string that need not name the model. `profile=` is the documented way out, and the
    # correction has to reach the wire, not just the profile dict.
    provider = OpenAIProvider(api_key='k')
    deployment = OpenAIRealtimeModel('voice-prod', provider=provider, settings={'thinking': 'low'})
    assert deployment.profile.get('supports_thinking') is False
    assert 'reasoning' not in deployment._session_config('', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]

    corrected = OpenAIRealtimeModel(
        'voice-prod',
        provider=provider,
        settings={'thinking': 'low'},
        profile=RealtimeModelProfile(supports_thinking=True),
    )
    assert corrected.profile.get('supports_thinking') is True
    assert corrected._session_config('', None, model_settings=None)['reasoning'] == {'effort': 'low'}  # pyright: ignore[reportPrivateUsage]


class _ConnectSequence:
    """Stand-in for `websockets.connect` that hands out a different socket per dial."""

    def __init__(self, sockets: list[FakeWebSocket]) -> None:
        self._sockets = iter(sockets)

    def __call__(self, url: str, *, additional_headers: dict[str, str] | None = None) -> _ConnectSequence:
        return self

    async def __aenter__(self) -> FakeWebSocket:
        try:
            return next(self._sockets)
        except StopIteration:
            # Out of sockets: fail the dial so the reconnect loop gives up instead of spinning.
            raise OSError('server is down')

    async def __aexit__(self, *exc: object) -> bool:
        return False


@pytest.mark.anyio
async def test_reconnect_replays_the_conversation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A re-dial replays the call, so the model carries on instead of resuming with amnesia.

    The API keeps no state across sessions and OpenAI ends one at 60 minutes, so a long call *expects*
    to renew: without this the model would come back knowing nothing that was said, including the
    `message_history` the caller seeded at connect. A unit test because it is the frames sent on the
    second dial that matter, which a cassette's matcher wouldn't compare.
    """
    transcript = json.dumps({'type': 'response.audio_transcript.done', 'transcript': 'still here'})
    dropped = _ExpiredWebSocket([_created(), _updated()])
    fresh = FakeWebSocket([_created(), _updated(), transcript])
    monkeypatch.setattr(rt_openai.websockets, 'connect', _ConnectSequence([dropped, fresh]))

    model = OpenAIRealtimeModel(
        'gpt-realtime',
        provider=OpenAIProvider(api_key='k'),
        settings={'reconnect': {'base_delay': 0.0, 'max_attempts': 1}},
    )
    # What a session would offer, with every kind of media a call accumulates: retained audio on both
    # sides, and a video frame sent alongside text. All of it is left behind; the words come back.
    frame = BinaryContent(data=b'\x89PNG', media_type='image/png')
    conversation = [
        ModelRequest(
            parts=[
                SpeechPart(
                    speaker='user',
                    transcript='what is the weather in Paris?',
                    audio=BinaryContent(data=b'\x02\x03', media_type='audio/wav'),
                )
            ]
        ),
        ModelResponse(
            parts=[
                SpeechPart(
                    speaker='assistant',
                    transcript='It is sunny in Paris.',
                    audio=BinaryContent(data=b'\x00\x01', media_type='audio/wav'),
                )
            ]
        ),
        ModelRequest(parts=[UserPromptPart(content=['and here?', frame])]),
        # A tool round replays whole, so the model comes back knowing what it already looked up.
        ModelResponse(parts=[ToolCallPart(tool_name='get_weather', args='{"city": "Paris"}', tool_call_id='tc_1')]),
        ModelRequest(parts=[ToolReturnPart(tool_name='get_weather', content='Sunny, 22C', tool_call_id='tc_1')]),
        # Nothing but a frame: no words to replay, so the turn drops out entirely.
        ModelRequest(parts=[UserPromptPart(content=[frame])]),
    ]
    async with _connect(model, 'be brief') as conn:
        conn.set_message_history(lambda: conversation)
        events = await collect_codec_events(conn)

    # The reconnect reports state as restored, because the replay is what restores it.
    assert events[0] == RealtimeSessionReconnectEvent(state_restored=True)
    replayed = [json.loads(frame) for frame in fresh.sent if 'conversation.item.create' in frame]
    assert replayed == snapshot(
        [
            {
                'type': 'conversation.item.create',
                'item': {
                    'type': 'message',
                    'role': 'user',
                    'content': [{'type': 'input_text', 'text': 'what is the weather in Paris?'}],
                },
            },
            {
                'type': 'conversation.item.create',
                'item': {
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'output_text', 'text': 'It is sunny in Paris.'}],
                },
            },
            {
                'type': 'conversation.item.create',
                'item': {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'and here?'}]},
            },
            {
                'type': 'conversation.item.create',
                'item': {
                    'type': 'function_call',
                    'name': 'get_weather',
                    'call_id': 'tc_1',
                    'arguments': '{"city": "Paris"}',
                },
            },
            {
                'type': 'conversation.item.create',
                'item': {'type': 'function_call_output', 'call_id': 'tc_1', 'output': 'Sunny, 22C'},
            },
        ]
    )
    # The first socket only ever received the session config: replay belongs to the *re*-dial.
    assert not [frame for frame in dropped.sent if 'conversation.item.create' in frame]


@pytest.mark.anyio
async def test_reconnect_without_a_session_does_not_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing to replay when no session offered the conversation, so state is honestly not restored."""
    fresh = FakeWebSocket([_created(), _updated()])
    monkeypatch.setattr(
        rt_openai.websockets, 'connect', _ConnectSequence([_ExpiredWebSocket([_created(), _updated()]), fresh])
    )

    model = OpenAIRealtimeModel(
        'gpt-realtime',
        provider=OpenAIProvider(api_key='k'),
        settings={'reconnect': {'base_delay': 0.0, 'max_attempts': 1}},
    )
    async with _connect(model, 'be brief') as conn:
        events = await collect_codec_events(conn)

    assert events[0] == RealtimeSessionReconnectEvent(state_restored=False)
    assert not [frame for frame in fresh.sent if 'conversation.item.create' in frame]
