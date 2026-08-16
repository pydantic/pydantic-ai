"""Shared codec and helpers for the OpenAI Realtime wire protocol.

The OpenAI Realtime API defines a WebSocket wire protocol that other providers clone verbatim — xAI's
Grok Voice realtime API ([`pydantic_ai.realtime.xai`][pydantic_ai.realtime.xai]) is a deliberate copy.
This module holds the provider-agnostic pieces of that protocol so both
[`pydantic_ai.realtime.openai`][pydantic_ai.realtime.openai] and the xAI provider can share them
without reaching into each other's internals: the WebSocket-URL derivation, session seeding, tool-def
conversion, turn-detection/tool-choice config builders, the handshake helper, and the event mapper.

The names below have no leading underscore because they are imported across the `openai`/`xai` provider
modules (they are still private to the package: the module itself is underscore-prefixed). Helpers used
only within this module keep their underscore prefix. The stateful connection and model classes live in
the provider modules, not here.
"""

from __future__ import annotations as _annotations

import asyncio
import base64
import hashlib
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeGuard, TypeVar, get_args
from urllib.parse import quote

import websockets
from openai.types.realtime import (
    ConversationItem,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    ConversationItemInputAudioTranscriptionFailedEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    RealtimeError as RealtimeErrorPayload,
    RealtimeErrorEvent,
    RealtimeResponse,
    RealtimeResponseStatus,
    RealtimeResponseUsage,
    ResponseAudioDeltaEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from pydantic import BaseModel, ConfigDict, TypeAdapter
from pydantic_core import to_json
from typing_extensions import Required, TypedDict, assert_never

from .._utils import generate_tool_call_id
from ..exceptions import ModelHTTPError, UserError
from ..messages import (
    BinaryContent,
    CompactionPart,
    FilePart,
    FinishReason,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponsePart,
    NativeToolCallPart,
    NativeToolReturnPart,
    RealtimeInputSpeechEndEvent,
    RealtimeInputSpeechStartEvent,
    RealtimeInputTranscriptionErrorEvent,
    RealtimeSessionErrorEvent,
    RetryPromptPart,
    SpeechPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
)
from ..models._tool_choice import ResolvedToolChoice
from ..profiles import DEFAULT_THINKING_TAGS
from ..tools import ToolDefinition
from ._utils import seed_pcm_audio, seed_speech_content, seed_user_content
from .codec import (
    AudioDelta,
    InputTranscript,
    OutputTranscript,
    RealtimeCodecEvent,
    ResponseDone,
    ToolCall,
)
from .model import RealtimeError
from .profiles import RealtimeModelProfile
from .settings import TurnDetection

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection


SESSION_CREATED_EVENT = 'session.created'
SESSION_UPDATE_EVENT = 'session.update'
SESSION_UPDATED_EVENT = 'session.updated'
CONVERSATION_ITEM_CREATE_EVENT = 'conversation.item.create'
INPUT_AUDIO_BUFFER_APPEND_EVENT = 'input_audio_buffer.append'
INPUT_AUDIO_BUFFER_COMMIT_EVENT = 'input_audio_buffer.commit'
INPUT_AUDIO_BUFFER_CLEAR_EVENT = 'input_audio_buffer.clear'
RESPONSE_CREATE_EVENT = 'response.create'
RESPONSE_CANCEL_EVENT = 'response.cancel'
CONVERSATION_ITEM_TRUNCATE_EVENT = 'conversation.item.truncate'


class _ReconnectableOpenAIProtocolConnection(Protocol):
    @property
    def message_history(self) -> Callable[[], Sequence[ModelMessage]] | None: ...


_ConnectionT = TypeVar('_ConnectionT', bound=_ReconnectableOpenAIProtocolConnection)


def realtime_websocket_url(base_url: str, *, model: str | None = None, call_id: str | None = None) -> str:
    """Derive the realtime WebSocket URL from a provider's HTTP base URL.

    Swaps the HTTP scheme for the WebSocket one and appends the `realtime` path, so the default
    OpenAI base URL `https://api.openai.com/v1/` yields `wss://api.openai.com/v1/realtime`. The
    path lands *before* any query string the base URL carries, rather than being appended after it
    into the wrong endpoint. A fragment is likewise split off first, so it can't swallow the path
    into the client-side part of the URL. `model`/`call_id` are merged in by `with_realtime_query`.
    """
    url, _, fragment = base_url.partition('#')
    url, _, query = url.partition('?')
    url = url.rstrip('/')
    if url.startswith('https://'):
        url = 'wss://' + url[len('https://') :]
    elif url.startswith('http://'):
        url = 'ws://' + url[len('http://') :]
    url = f'{url}/realtime'
    url = f'{url}?{query}' if query else url
    url = f'{url}#{fragment}' if fragment else url
    return with_realtime_query(url, model=model, call_id=call_id)


def with_realtime_query(websocket_url: str, *, model: str | None = None, call_id: str | None = None) -> str:
    """Merge the session-addressing query parameters into a realtime WebSocket URL.

    Takes the URL as already derived — a provider whose realtime path doesn't follow from its HTTP
    base URL (Azure OpenAI derives it from the resource endpoint) builds that part itself — and
    preserves any query and fragment it already carries. `model` opens a new session; `call_id`
    attaches a control-plane (sideband) connection to a WebRTC call that already exists, which
    carries its own model.

    The fragment is split off first so the parameters land in the query rather than in the
    client-side part of the URL, which the handshake would never send.
    """
    url, _, fragment = websocket_url.partition('#')
    url, _, query = url.partition('?')
    for name, value in (('model', model), ('call_id', call_id)):
        if value is not None:
            param = f'{name}={quote(value, safe="")}'
            query = f'{query}&{param}' if query else param
    url = f'{url}?{query}' if query else url
    return f'{url}#{fragment}' if fragment else url


AUTO_TRANSCRIPTION_MODEL = 'auto'
"""Sentinel `input_transcription_model` value: resolve to the provider's recommended transcription model."""


def resolve_transcription_model(value: str | None, *, default: str) -> str | None:
    """Resolve an `input_transcription_model` setting to a concrete model id, or `None` to disable transcription.

    `'auto'` (the always-supported default) resolves to `default`, the provider's current recommended
    realtime transcription model. Keeping the public default as a stable sentinel — rather than a hardcoded
    model id — lets the concrete model it maps to change over releases without silently altering the
    behavior of apps that pinned a specific id. Any other string is used verbatim; `None` disables
    transcription (no user transcripts; see `audio_retention` to retain the raw audio instead).
    """
    if value == AUTO_TRANSCRIPTION_MODEL:
        return default
    return value


# The OpenAI event names differ between the GA and beta realtime surfaces; accept both.
AUDIO_DELTA_TYPES = frozenset({'response.output_audio.delta', 'response.audio.delta'})
INPUT_TRANSCRIPT_DONE_TYPES = frozenset({'conversation.item.input_audio_transcription.completed'})

OpenAIProtocolEventType = Literal[
    'response.output_audio.delta',
    'response.audio.delta',
    'response.output_audio_transcript.delta',
    'response.audio_transcript.delta',
    'response.output_audio_transcript.done',
    'response.audio_transcript.done',
    'response.output_text.delta',
    'response.output_text.done',
    'conversation.item.input_audio_transcription.delta',
    'conversation.item.input_audio_transcription.completed',
    'conversation.item.input_audio_transcription.failed',
    'response.function_call_arguments.done',
    'input_audio_buffer.speech_started',
    'input_audio_buffer.speech_stopped',
    'response.done',
    'error',
]
_KNOWN_EVENT_TYPES: frozenset[OpenAIProtocolEventType] = frozenset(get_args(OpenAIProtocolEventType))

# `loads_obj`'s contract in one pydantic-core pass: parse the JSON text and require a JSON object.
_JSON_FRAME_ADAPTER: TypeAdapter[dict[str, Any]] = TypeAdapter(dict[str, Any])

_VAD_SENSITIVITY_THRESHOLDS = {'low': 0.7, 'medium': 0.5, 'high': 0.3}


class ServerVAD(TypedDict, total=False):
    """Server-side voice activity detection — the default turn-taking mode.

    The server detects when the user starts and stops speaking and (by default) commits the audio
    and triggers a response automatically. Unset fields fall back to the provider defaults.
    """

    type: Required[Literal['server_vad']]
    """The turn-detection type. Must be `'server_vad'`."""
    threshold: float
    """Activation threshold (0.0-1.0). Higher requires louder audio; better in noisy environments.
    Defaults to the provider default."""
    prefix_padding_ms: int
    """Audio to include before detected speech, in milliseconds. Defaults to the provider default."""
    silence_duration_ms: int
    """Silence required to detect the end of speech, in milliseconds. Defaults to the provider default."""
    create_response: bool
    """Whether to automatically generate a response when the user stops speaking. Defaults to `True`."""
    interrupt_response: bool
    """Whether to interrupt an in-progress response when the user starts speaking. Defaults to `True`."""
    idle_timeout_ms: int
    """If set, auto-trigger a response after this much idle time with no detected speech.
    Defaults to the provider default."""


class SemanticVAD(TypedDict, total=False):
    """Model-based semantic turn detection — uses a model to decide when the user is done speaking."""

    type: Required[Literal['semantic_vad']]
    """The turn-detection type. Must be `'semantic_vad'`."""
    eagerness: Literal['low', 'medium', 'high', 'auto']
    """How eagerly the model responds. Defaults to `'auto'`; `low` waits longer and `high` responds sooner."""
    create_response: bool
    """Whether to automatically generate a response when a turn ends. Defaults to `True`."""
    interrupt_response: bool
    """Whether to interrupt an in-progress response when the user starts speaking. Defaults to `True`."""


class _ProtocolResponseOutputItem(BaseModel):
    """Minimal typed response item used for xAI's SDK-incompatible `audio` output item."""

    model_config = ConfigDict(extra='allow')

    type: str


class _ProtocolInputTranscriptionCompletedEvent(BaseModel):
    """SDK event with xAI's cassette-proven omitted `usage` field."""

    model_config = ConfigDict(extra='allow')

    event_id: str
    item_id: str
    transcript: str
    type: Literal['conversation.item.input_audio_transcription.completed']
    content_index: int
    usage: object | None = None


class _ProtocolAudioDeltaEvent(BaseModel):
    content_index: int
    delta: str
    event_id: str
    item_id: str
    output_index: int
    response_id: str
    type: Literal['response.output_audio.delta', 'response.audio.delta']


class _ProtocolAudioTranscriptDeltaEvent(BaseModel):
    content_index: int
    delta: str
    event_id: str
    item_id: str
    output_index: int
    response_id: str
    type: Literal['response.output_audio_transcript.delta', 'response.audio_transcript.delta']


class _ProtocolAudioTranscriptDoneEvent(BaseModel):
    content_index: int
    event_id: str
    item_id: str
    output_index: int
    response_id: str
    transcript: str
    type: Literal['response.output_audio_transcript.done', 'response.audio_transcript.done']


class ProtocolRealtimeResponse(BaseModel):
    """SDK response with cassette-proven xAI extensions kept typed."""

    model_config = ConfigDict(extra='allow')

    id: str | None = None
    output: list[ConversationItem | _ProtocolResponseOutputItem] | None = None
    status: Literal['completed', 'cancelled', 'failed', 'incomplete', 'in_progress'] | None = None
    status_details: RealtimeResponseStatus | str | None = None
    usage: RealtimeResponseUsage | None = None


class ProtocolResponseDoneEvent(BaseModel):
    """SDK response event with xAI's frame-level usage extension."""

    event_id: str
    response: ProtocolRealtimeResponse
    type: Literal['response.done']
    usage: RealtimeResponseUsage | None = None


class ProtocolResponseCreatedEvent(BaseModel):
    """SDK response-created event with cassette-proven xAI response extensions."""

    event_id: str
    response: ProtocolRealtimeResponse
    type: Literal['response.created']


_InputTranscriptionCompletedEvent = (
    ConversationItemInputAudioTranscriptionCompletedEvent | _ProtocolInputTranscriptionCompletedEvent
)
_AudioDeltaEvent = ResponseAudioDeltaEvent | _ProtocolAudioDeltaEvent
_AudioTranscriptDeltaEvent = ResponseAudioTranscriptDeltaEvent | _ProtocolAudioTranscriptDeltaEvent
_AudioTranscriptDoneEvent = ResponseAudioTranscriptDoneEvent | _ProtocolAudioTranscriptDoneEvent
ResponseDoneEventType = ResponseDoneEvent | ProtocolResponseDoneEvent
ResponseCreatedEventType = ResponseCreatedEvent | ProtocolResponseCreatedEvent

_INPUT_TRANSCRIPTION_COMPLETED_ADAPTER: TypeAdapter[_InputTranscriptionCompletedEvent] = TypeAdapter(
    _InputTranscriptionCompletedEvent
)
_AUDIO_DELTA_ADAPTER: TypeAdapter[_AudioDeltaEvent] = TypeAdapter(_AudioDeltaEvent)
_AUDIO_TRANSCRIPT_DELTA_ADAPTER: TypeAdapter[_AudioTranscriptDeltaEvent] = TypeAdapter(_AudioTranscriptDeltaEvent)
_AUDIO_TRANSCRIPT_DONE_ADAPTER: TypeAdapter[_AudioTranscriptDoneEvent] = TypeAdapter(_AudioTranscriptDoneEvent)
RESPONSE_DONE_EVENT_ADAPTER: TypeAdapter[ResponseDoneEventType] = TypeAdapter(ResponseDoneEventType)
RESPONSE_CREATED_EVENT_ADAPTER: TypeAdapter[ResponseCreatedEventType] = TypeAdapter(ResponseCreatedEventType)

ProtocolResponse = RealtimeResponse | ProtocolRealtimeResponse

# Azure resolves the input-transcription model against the resource's own deployments rather than
# OpenAI's hosted models, so the default fails on every turn until a transcription model is deployed.
# Azure's own message names only the affected item, saying nothing about the cause or the fix, so the
# remedy is appended to it. Keyed on the provider's error code, which no other provider sends.
_MISSING_TRANSCRIPTION_DEPLOYMENT_CODE = 'DeploymentNotFound'
_MISSING_TRANSCRIPTION_DEPLOYMENT_HELP = (
    'The transcription model is not deployed on this Azure OpenAI resource. Deploy one and set '
    '`input_transcription_model` to its deployment name, or set it to `None` to disable transcription.'
)


def tool_def_to_openai(tool: ToolDefinition) -> dict[str, Any]:
    """Convert a [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] to the OpenAI realtime tool format."""
    result: dict[str, Any] = {
        'type': 'function',
        'name': tool.name,
        'parameters': tool.parameters_json_schema,
    }
    if tool.description:
        result['description'] = tool.description
    return result


async def replay_items(
    messages: Sequence[ModelMessage], *, profile: RealtimeModelProfile, provider_name: str
) -> list[dict[str, Any]]:
    """Map a live session's conversation to items that restore it in a freshly dialed session.

    Used when a [reconnect][pydantic_ai.realtime.ReconnectPolicy] gets a session with no server-side
    state, so the model continues the call rather than resuming with amnesia.

    Media is deliberately left behind: what the model needs is the *conversation*, and re-uploading a
    long call's retained audio (or a stale video frame) would cost far more than it restores. Stripping
    it also keeps replay total — a turn whose audio is dropped and has no transcript becomes a
    content-less placeholder, which seeding already skips, where the strict path would rightly refuse it.
    """
    return await seed_items(
        [replayable for message in messages if (replayable := _without_media(message)) is not None],
        # Not `replace`d on the caller's profile: these say what *this* seeding pass will carry, not what
        # the provider supports.
        profile={**profile, 'supports_seeding_audio': False, 'supports_seeding_images': False},
        provider_name=provider_name,
    )


def _without_media(message: ModelMessage) -> ModelMessage | None:
    """Strip audio and images from a message, dropping it entirely when nothing is left to replay."""
    if isinstance(message, ModelRequest):
        request_parts: list[ModelRequestPart] = []
        for part in message.parts:
            if isinstance(part, SpeechPart):
                request_parts.append(replace(part, audio=None))
            elif isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                if text := [item for item in part.content if isinstance(item, (str, TextContent))]:
                    request_parts.append(replace(part, content=text))
            elif isinstance(part, ToolReturnPart) and part.files:
                request_parts.append(replace(part, content=part.model_response_str(wrap_if_error=False)))
            else:
                request_parts.append(part)
        return replace(message, parts=request_parts) if request_parts else None
    response_parts = [
        replace(part, audio=None) if isinstance(part, SpeechPart) else part
        for part in message.parts
        if not isinstance(part, FilePart)
    ]
    return replace(message, parts=response_parts) if response_parts else None


async def seed_items(
    messages: Sequence[ModelMessage], *, profile: RealtimeModelProfile, provider_name: str
) -> list[dict[str, Any]]:
    """Map prior history to OpenAI-protocol `conversation.item.create` items.

    Text, transcripts, images, retained user audio, thinking, and function-tool rounds are replayed in
    part order. Thinking becomes tag-wrapped assistant text; its signature and `provider_details` are
    provider-session-bound and are not replayed. Native-tool parts are metadata about how an answer was
    produced and are skipped while the answer itself is retained. `SystemPromptPart`s are routed through
    the session `instructions` field, and `CachePoint`s are ignored.

    Retained user audio is decoded by the new session's configured input-audio format. Assistant audio
    cannot be inserted by the API, so assistant speech requires a transcript. Any other content that
    cannot be represented faithfully raises [`UserError`][pydantic_ai.exceptions.UserError].
    """
    items: list[dict[str, Any]] = []
    call_ids: dict[str, str] = {}
    seeded_calls: set[str] = set()
    supports_images = profile.get('supports_seeding_images', False)
    supports_audio = profile.get('supports_seeding_audio', False)

    for message in messages:
        if isinstance(message, ModelRequest):
            items.extend(
                await _seed_request_items(
                    message.parts,
                    provider_name=provider_name,
                    supports_images=supports_images,
                    supports_audio=supports_audio,
                    audio_input_sample_rate=profile.get('audio_input_sample_rate', 24000),
                    call_ids=call_ids,
                    seeded_calls=seeded_calls,
                )
            )
        else:
            items.extend(
                _seed_response_items(
                    message.parts,
                    provider_name=provider_name,
                    call_ids=call_ids,
                    seeded_calls=seeded_calls,
                )
            )
    return items


async def _seed_request_items(
    parts: Sequence[ModelRequestPart],
    *,
    provider_name: str,
    supports_images: bool,
    supports_audio: bool,
    audio_input_sample_rate: int,
    call_ids: dict[str, str],
    seeded_calls: set[str],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, (SystemPromptPart, ToolAvailabilityDeltaPart)):
            # System prompts are seeded through session instructions, and tool-availability news
            # from a prior standard run is stale here: the session advertises its own tools.
            continue
        elif isinstance(part, UserPromptPart):
            if content := _user_content_items(
                await seed_user_content(part=part, provider_name=provider_name, supports_images=supports_images)
            ):
                items.append(_message_item('user', content))
        elif isinstance(part, SpeechPart):
            content = seed_speech_content(part=part, provider_name=provider_name, supports_audio=supports_audio)
            if isinstance(content, str):
                if content:
                    items.append(_message_item('user', [_text_content('input_text', content)]))
            else:
                pcm = seed_pcm_audio(
                    audio=content,
                    provider_name=provider_name,
                    sample_rate=audio_input_sample_rate,
                )
                items.append(_message_item('user', [{'type': 'input_audio', 'audio': base64.b64encode(pcm).decode()}]))
        elif isinstance(part, ToolReturnPart):
            _require_seeded_call(part.tool_name, tool_call_id=part.tool_call_id, seeded_calls=seeded_calls)
            output, user_content = part.model_response_str_and_user_content()
            items.append(
                {
                    'type': 'function_call_output',
                    'call_id': _seed_call_id(part.tool_call_id, call_ids),
                    'output': output,
                }
            )
            if user_content and (
                content := _user_content_items(
                    await seed_user_content(
                        part=UserPromptPart(content=user_content),
                        provider_name=provider_name,
                        supports_images=supports_images,
                    )
                )
            ):
                items.append(_message_item('user', content))
        elif isinstance(part, RetryPromptPart):
            output = part.model_response()
            if part.tool_name is None:
                items.append(_message_item('user', [_text_content('input_text', output)]))
            else:
                _require_seeded_call(part.tool_name, tool_call_id=part.tool_call_id, seeded_calls=seeded_calls)
                items.append(
                    {
                        'type': 'function_call_output',
                        'call_id': _seed_call_id(part.tool_call_id, call_ids),
                        'output': output,
                    }
                )
        else:
            assert_never(part)
    return items


def _seed_response_items(
    parts: Sequence[ModelResponsePart],
    *,
    provider_name: str,
    call_ids: dict[str, str],
    seeded_calls: set[str],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, TextPart):
            if part.content:
                items.append(_message_item('assistant', [_text_content('output_text', part.content)]))
        elif isinstance(part, ThinkingPart):
            if part.content:
                start_tag, end_tag = DEFAULT_THINKING_TAGS
                text = '\n'.join([start_tag, part.content, end_tag])
                items.append(_message_item('assistant', [_text_content('output_text', text)]))
        elif isinstance(part, ToolCallPart):
            call_id = _seed_call_id(part.tool_call_id, call_ids)
            seeded_calls.add(part.tool_call_id)
            items.append(
                {
                    'type': 'function_call',
                    'name': part.tool_name,
                    'call_id': call_id,
                    'arguments': part.args_as_json_str(),
                }
            )
        elif isinstance(part, (NativeToolCallPart, NativeToolReturnPart)):
            continue
        elif isinstance(part, SpeechPart):
            # Assistant audio can't be replayed on any provider, so the audio-seeding capability
            # doesn't apply here and the typed result is always a transcript string.
            content = seed_speech_content(part=part, provider_name=provider_name, supports_audio=False)
            if content:
                items.append(_message_item('assistant', [_text_content('output_text', content)]))
        elif isinstance(part, CompactionPart):
            # Provider-session-bound compaction state can't round-trip into another session; classic
            # model adapters skip it when crossing APIs (e.g. Chat Completions), and seeding matches.
            continue
        elif isinstance(part, FilePart):
            raise UserError(
                f'`FilePart` cannot be seeded into {provider_name} realtime history. '
                'Convert it to text or filter it from `message_history` before connecting.'
            )
        else:
            assert_never(part)
    return items


def _seed_call_id(tool_call_id: str, call_ids: dict[str, str]) -> str:
    """Return a stable wire ID no longer than the OpenAI protocol's 32-character limit."""
    if wire_id := call_ids.get(tool_call_id):
        return wire_id
    wire_id = tool_call_id if len(tool_call_id) <= 32 else hashlib.sha256(tool_call_id.encode()).hexdigest()[:32]
    used_ids = set(call_ids.values())
    while wire_id in used_ids:
        wire_id = hashlib.sha256(wire_id.encode()).hexdigest()[:32]
    call_ids[tool_call_id] = wire_id
    return wire_id


def _require_seeded_call(tool_name: str, *, tool_call_id: str, seeded_calls: set[str]) -> None:
    if tool_call_id not in seeded_calls:
        raise UserError(
            f'Cannot seed output for tool {tool_name!r} with call ID {tool_call_id!r}: no preceding '
            '`ToolCallPart` with that ID was included in `message_history`.'
        )


def _user_content_items(content: Sequence[str | BinaryContent]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            if item:
                items.append(_text_content('input_text', item))
        elif item.is_image:
            items.append({'type': 'input_image', 'image_url': item.data_uri})
        else:
            raise UserError(
                f'Expected image content after realtime user-content normalization, got {item.media_type!r}'
            )
    return items


def _text_content(content_type: Literal['input_text', 'output_text'], text: str) -> dict[str, Any]:
    return {'type': content_type, 'text': text}


def _message_item(role: Literal['user', 'assistant'], content: list[dict[str, Any]]) -> dict[str, Any]:
    return {'type': 'message', 'role': role, 'content': content}


async def user_message_item(
    content: str | Sequence[UserContent], *, provider_name: str, supports_images: bool
) -> dict[str, Any] | None:
    """Build a live follow-up user item through the same normalization used for seeded history."""
    normalized = await seed_user_content(
        part=UserPromptPart(content=content), provider_name=provider_name, supports_images=supports_images
    )
    return _message_item('user', items) if (items := _user_content_items(normalized)) else None


def loads_obj(raw: str) -> dict[str, Any]:
    """Parse a JSON text frame into an object, raising `ValueError` if it decodes to a non-object.

    JSON can contain arrays, strings, or numbers; those aren't valid realtime frames, so treat
    them as malformed (a `ValueError`, like a decode error) rather than letting a later `.get()` raise
    `AttributeError` and escape the recoverable-error handling.
    """
    return _JSON_FRAME_ADAPTER.validate_json(raw)


def _is_function_call_only(output: Sequence[ConversationItem | _ProtocolResponseOutputItem] | None) -> bool:
    """Whether a `response.done` output list contains only function calls."""
    return bool(output) and all(item.type == 'function_call' for item in output)


def _response_status_reason(response: ProtocolResponse) -> str | None:
    """Return the raw terminal `status_details.reason`, when present."""
    status_details = response.status_details
    return status_details.reason if isinstance(status_details, RealtimeResponseStatus) else None


def response_finish_reason(response: ProtocolResponse) -> FinishReason | None:
    """Map an OpenAI-protocol response `status`/output to a shared `FinishReason`.

    A `'cancelled'` response is a barge-in (the user interrupted the model), which isn't an error and
    has no dedicated `FinishReason`, so it's left unset — the response's `state='interrupted'` carries
    that meaning, mirroring how a classic cancelled stream leaves `finish_reason` as-is. Incomplete
    responses use `status_details.reason`, matching the classic OpenAI adapter.
    """
    status = response.status
    if status == 'completed':
        return 'tool_call' if _is_function_call_only(response.output) else 'stop'
    if status == 'incomplete':
        reason = _response_status_reason(response)
        if reason == 'max_output_tokens':
            return 'length'
        if reason == 'content_filter':
            return 'content_filter'
    if status == 'failed':
        return 'error'
    return None


def _response_provider_details(response: ProtocolResponse) -> dict[str, Any]:
    """Retain the raw response status and incomplete reason for provider fidelity."""
    details: dict[str, Any] = {'status': response.status}
    if (reason := _response_status_reason(response)) is not None:
        details['finish_reason'] = reason
    return details


def _map_response_done(data: dict[str, Any]) -> RealtimeCodecEvent | None:
    """Map a `response.done` event, returning `None` for function-call-only responses.

    A response whose only output is function calls is an intermediate step: the session executes the
    tools and the model emits a further `response.done` with the actual answer. Surfacing a
    `ResponseDone` here would prematurely signal the end of the turn.
    """
    event = RESPONSE_DONE_EVENT_ADAPTER.validate_python(data)
    response = event.response
    output = response.output
    if response.status == 'completed' and _is_function_call_only(output):
        return None
    status = response.status
    response_id = response.id
    return ResponseDone(
        interrupted=status == 'cancelled',
        provider_response_id=response_id if isinstance(response_id, str) else None,
        finish_reason=response_finish_reason(response),
        provider_details=_response_provider_details(response),
    )


def map_event(data: dict[str, Any]) -> RealtimeCodecEvent | None:
    """Map a raw OpenAI Realtime event to a [`RealtimeCodecEvent`][pydantic_ai.realtime.codec.RealtimeCodecEvent].

    Returns `None` for events that carry no session-relevant content (e.g. `session.created`).
    """
    # Dispatching once on the wire discriminator avoids the SDK union parser misclassifying clone/beta events.
    raw_event_type = data.get('type')
    if not _is_known_event_type(raw_event_type):
        return None
    event_type = raw_event_type

    if event_type in ('response.output_audio.delta', 'response.audio.delta'):
        event = _AUDIO_DELTA_ADAPTER.validate_python(data)
        return AudioDelta(data=base64.b64decode(event.delta, validate=True), item_id=event.item_id or None)

    elif event_type in ('response.output_audio_transcript.delta', 'response.audio_transcript.delta'):
        event = _AUDIO_TRANSCRIPT_DELTA_ADAPTER.validate_python(data)
        return OutputTranscript(text=event.delta or '', is_final=False, item_id=event.item_id or None)

    elif event_type in ('response.output_audio_transcript.done', 'response.audio_transcript.done'):
        event = _AUDIO_TRANSCRIPT_DONE_ADAPTER.validate_python(data)
        return OutputTranscript(text=event.transcript or '', is_final=True, item_id=event.item_id or None)

    elif event_type == 'response.output_text.delta':
        event = ResponseTextDeltaEvent.model_validate(data)
        return OutputTranscript(text=event.delta or '', is_final=False, item_id=event.item_id or None, output_text=True)

    elif event_type == 'response.output_text.done':
        event = ResponseTextDoneEvent.model_validate(data)
        return OutputTranscript(text=event.text or '', is_final=True, item_id=event.item_id or None, output_text=True)

    elif event_type in (
        'conversation.item.input_audio_transcription.delta',
        'conversation.item.input_audio_transcription.completed',
        'conversation.item.input_audio_transcription.failed',
    ):
        return _map_input_transcription_event(data, event_type)

    elif event_type == 'response.function_call_arguments.done':
        event = ResponseFunctionCallArgumentsDoneEvent.model_validate(data)
        return ToolCall(
            # A synthetic id rather than `''`: an empty one is falsy, so a standard run would later
            # generate a *different* id for the call and for its return (both go through
            # `guard_tool_call_id`) and the provider would reject the mismatched pair. The protocol
            # always sends `call_id`, so this only catches a malformed frame.
            tool_call_id=event.call_id or generate_tool_call_id(),
            tool_name=event.name or '',
            args=event.arguments or '{}',
            response_usage_follows=True,
        )

    elif event_type == 'input_audio_buffer.speech_started':
        event = InputAudioBufferSpeechStartedEvent.model_validate(data)
        return RealtimeInputSpeechStartEvent(item_id=event.item_id or None)

    elif event_type == 'input_audio_buffer.speech_stopped':
        # The stopped frame names the input item this speech segment produced, letting the session attach
        # retained input audio to the right user turn even when overlapping turns finalize out of order.
        event = InputAudioBufferSpeechStoppedEvent.model_validate(data)
        return RealtimeInputSpeechEndEvent(item_id=event.item_id or None)

    elif event_type == 'response.done':
        return _map_response_done(data)

    elif event_type == 'error':
        event = RealtimeErrorEvent.model_validate(data)
        error = event.error
        return RealtimeSessionErrorEvent(
            message=_error_message(error),
            type=error.type or None,
            code=error.code or None,
            recoverable=True,  # a protocol `error` keeps the session open; a dropped connection does not
        )
    else:
        assert_never(event_type)


def _is_known_event_type(value: object) -> TypeGuard[OpenAIProtocolEventType]:
    return isinstance(value, str) and value in _KNOWN_EVENT_TYPES


def _map_input_transcription_event(
    data: dict[str, Any], event_type: OpenAIProtocolEventType
) -> InputTranscript | RealtimeInputTranscriptionErrorEvent | None:
    """Map input transcription progress and failure events."""
    if event_type == 'conversation.item.input_audio_transcription.delta':
        event = ConversationItemInputAudioTranscriptionDeltaEvent.model_validate(data)
        return InputTranscript(text=event.delta or '', is_final=False, item_id=event.item_id or None)
    if event_type in INPUT_TRANSCRIPT_DONE_TYPES:
        event = _INPUT_TRANSCRIPTION_COMPLETED_ADAPTER.validate_python(data)
        # OpenAI omits `status`; xAI and Azure may send cumulative `.completed` snapshots whose
        # interim values must not be surfaced as final transcripts.
        status = (event.model_extra or {}).get('status')  # Provider extra used by xAI and Azure.
        if status is not None and status != 'completed':
            return None
        return InputTranscript(text=event.transcript or '', is_final=True, item_id=event.item_id or None)

    event = ConversationItemInputAudioTranscriptionFailedEvent.model_validate(data)
    message = event.error.message or ''
    code = event.error.code or None
    if code == _MISSING_TRANSCRIPTION_DEPLOYMENT_CODE:
        message = f'{message} {_MISSING_TRANSCRIPTION_DEPLOYMENT_HELP}'.strip()
    return RealtimeInputTranscriptionErrorEvent(
        message=message,
        type=event.error.type or None,
        code=code,
        item_id=event.item_id or None,
        content_index=event.content_index,
    )


def _error_message(error: RealtimeErrorPayload | object) -> str:
    """Extract a human-readable message from an OpenAI `error` payload."""
    if isinstance(error, RealtimeErrorPayload):
        return error.message or to_json(error.model_dump(exclude_none=True)).decode()
    return str(error)


class RealtimeHandshakeError(Exception):
    """A failure while establishing an OpenAI-protocol realtime session.

    Covers the server rejecting the session with an `error` event, a frame we couldn't parse, and the
    handshake timing out. Raised by [`expect_event`][pydantic_ai.realtime._openai_protocol.expect_event]
    and mapped to a [`RealtimeError`][pydantic_ai.realtime.RealtimeError] by
    [`map_connect_errors`][pydantic_ai.realtime._openai_protocol.map_connect_errors], so each of them
    surfaces as a typed model exception rather than a bare protocol error or a built-in `TimeoutError`.
    """

    def __init__(self, error: object) -> None:
        self.error = error
        """The server's `error` payload when it rejected the session, otherwise a description of the failure."""
        super().__init__(_error_message(error))


@contextmanager
def map_connect_errors(model_name: str) -> Generator[None]:
    """Map realtime handshake failures to the typed exceptions the regular models raise.

    Wrap the initial dial so anything that stops the session coming up surfaces as a
    [`RealtimeError`][pydantic_ai.realtime.RealtimeError], or a
    [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError] where the failure carries an HTTP status,
    each naming the model — mirroring [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel].
    Reconnects dial outside this manager so the reconnect loop keeps treating a drop as retryable.
    """
    try:
        yield
    except RealtimeHandshakeError as e:
        # A rejected config, a malformed frame, or a timeout: all arrive over the open WebSocket and
        # carry no HTTP status, so they map to `RealtimeError` like a regular non-status provider error.
        raise RealtimeError(model_name=model_name, message=str(e)) from e
    except websockets.InvalidStatus as e:
        # The WebSocket upgrade itself was rejected (bad key → 401, unknown model → 404); this carries a
        # real HTTP status, so it maps to `ModelHTTPError` exactly like a regular request.
        response = e.response
        body = response.body.decode(errors='replace') if response.body else response.reason_phrase
        raise ModelHTTPError(status_code=response.status_code, model_name=model_name, body=body) from e
    except websockets.WebSocketException as e:
        # Any other WebSocket-level handshake failure — the server closed the socket mid-handshake (e.g. a
        # gateway rejecting an unknown model) instead of sending an `error` event, a bad upgrade, etc.
        # These carry no HTTP status, so they map to `RealtimeError` with the underlying detail, ensuring
        # the session surfaces a typed error rather than dying silently.
        raise RealtimeError(model_name=model_name, message=f'WebSocket error during realtime handshake: {e}') from e
    except OSError as e:
        # The connection never came up: DNS failure, refused, reset, or the dial timing out
        # (`TimeoutError` is an `OSError`). No HTTP status exists, so this is a `RealtimeError` too --
        # without this the caller would get a bare built-in from what looks like an ordinary model call.
        raise RealtimeError(model_name=model_name, message=f'Could not reach the realtime API: {e}') from e


@asynccontextmanager
async def connect_openai_protocol(
    *,
    model_name: str,
    messages: Sequence[ModelMessage],
    profile: RealtimeModelProfile,
    provider_name: str,
    session_config: dict[str, Any],
    handshake_timeout: float,
    dial_headers: Callable[[], Awaitable[dict[str, str]]],
    dial_url: Callable[[], str],
    session_model: Callable[[dict[str, Any]], str | None],
    build_connection: Callable[
        [ClientConnection, Callable[[], Awaitable[ClientConnection]], str | None, Callable[[], str | None]],
        _ConnectionT,
    ],
    after_session_created: Callable[[ClientConnection, dict[str, Any]], Awaitable[None]] | None = None,
    on_unexpected_during_update: Callable[[], Callable[[dict[str, Any]], None] | None] | None = None,
    replay_on_redial: bool = False,
) -> AsyncGenerator[_ConnectionT]:
    """Connect an OpenAI-protocol realtime model and own its socket lifecycle.

    OpenAI supplies refreshable `dial_headers` and enables `replay_on_redial`. xAI supplies static
    headers, a `dial_url` that adds its conversation ID, `after_session_created` to establish native
    resumption, and `on_unexpected_during_update` to capture the server's replay burst.
    """
    # Normalize history before opening a socket so unsupported content remains a caller `UserError`.
    seed = await seed_items(messages, profile=profile, provider_name=provider_name)
    cm: AbstractAsyncContextManager[ClientConnection] | None = None
    server_model: str | None = None
    connection: _ConnectionT | None = None

    async def dial() -> ClientConnection:
        nonlocal cm, server_model
        if cm is not None:
            previous, cm = cm, None
            await previous.__aexit__(None, None, None)
        opening = websockets.connect(dial_url(), additional_headers=await dial_headers())
        ws = await opening.__aenter__()
        cm = opening
        created = await expect_event(ws, SESSION_CREATED_EVENT, timeout=handshake_timeout)
        if served_model := session_model(created):
            server_model = served_model
        if after_session_created is not None:
            await after_session_created(ws, created)
        await ws.send(to_json({'type': SESSION_UPDATE_EVENT, 'session': session_config}).decode())
        on_unexpected = on_unexpected_during_update() if on_unexpected_during_update is not None else None
        await expect_event(ws, SESSION_UPDATED_EVENT, timeout=handshake_timeout, on_unexpected=on_unexpected)
        if replay_on_redial and connection is not None and (message_history := connection.message_history) is not None:
            for item in await replay_items(message_history(), profile=profile, provider_name=provider_name):
                await ws.send(to_json({'type': CONVERSATION_ITEM_CREATE_EVENT, 'item': item}).decode())
        return ws

    try:
        # Only the initial dial is mapped; reconnect failures stay retryable for the connection loop.
        with map_connect_errors(model_name):
            ws = await dial()
            for item in seed:
                await ws.send(to_json({'type': CONVERSATION_ITEM_CREATE_EVENT, 'item': item}).decode())
        connection = build_connection(ws, dial, server_model, lambda: server_model)
        yield connection
    finally:
        # Coverage cannot attribute a failed `__aenter__` to the false exit arc.
        if cm is not None:  # pragma: no branch
            await cm.__aexit__(None, None, None)


def server_vad_from_turn_detection(turn_detection: TurnDetection) -> ServerVAD:
    """Map cross-provider turn detection to the OpenAI-compatible server-VAD shape."""
    threshold = _VAD_SENSITIVITY_THRESHOLDS[turn_detection['sensitivity']] if 'sensitivity' in turn_detection else None
    result: ServerVAD = {'type': 'server_vad'}
    if threshold is not None:
        result['threshold'] = threshold
    if (prefix_padding_ms := turn_detection.get('prefix_padding_ms')) is not None:
        result['prefix_padding_ms'] = prefix_padding_ms
    if (silence_duration_ms := turn_detection.get('silence_duration_ms')) is not None:
        result['silence_duration_ms'] = silence_duration_ms
    return result


def resolve_base_turn_detection(base: bool | TurnDetection) -> ServerVAD | None:
    """Resolve a cross-provider `turn_detection` value to a server-VAD config (or `None` to disable).

    `True` (or an absent setting, handled by the caller) uses the provider defaults; `False` disables
    detection (push-to-talk); a [`TurnDetection`][pydantic_ai.realtime.TurnDetection] maps its knobs.
    """
    if base is False:
        return None
    if base is True:
        return {'type': 'server_vad'}
    return server_vad_from_turn_detection(base)


def turn_detection_config(turn_detection: ServerVAD | SemanticVAD | None) -> dict[str, Any] | None:
    """Build the OpenAI `turn_detection` payload, or `None` to disable VAD (manual turn-taking)."""
    if turn_detection is None:
        return None
    if turn_detection['type'] == 'server_vad':
        config: dict[str, Any] = {
            'type': 'server_vad',
            'create_response': turn_detection.get('create_response', True),
            'interrupt_response': turn_detection.get('interrupt_response', True),
        }
        if (threshold := turn_detection.get('threshold')) is not None:
            config['threshold'] = threshold
        if (prefix_padding_ms := turn_detection.get('prefix_padding_ms')) is not None:
            config['prefix_padding_ms'] = prefix_padding_ms
        if (silence_duration_ms := turn_detection.get('silence_duration_ms')) is not None:
            config['silence_duration_ms'] = silence_duration_ms
        if (idle_timeout_ms := turn_detection.get('idle_timeout_ms')) is not None:
            config['idle_timeout_ms'] = idle_timeout_ms
        return config
    return {
        'type': 'semantic_vad',
        'eagerness': turn_detection.get('eagerness', 'auto'),
        'create_response': turn_detection.get('create_response', True),
        'interrupt_response': turn_detection.get('interrupt_response', True),
    }


def tool_choice_config(tool_choice: ResolvedToolChoice) -> str | dict[str, Any]:
    """Map a resolved `tool_choice` to the OpenAI realtime `tool_choice` field.

    Restrictions to a subset of the tools are carried by the advertised tool definitions, which the
    caller has already narrowed, so only the mode is left to send — except for the one restriction
    realtime does express directly, a single named function.
    """
    if isinstance(tool_choice, tuple):
        mode, allowed = tool_choice
        if mode == 'required' and len(allowed) == 1:
            return {'type': 'function', 'name': next(iter(allowed))}
        return mode
    return tool_choice


async def expect_event(
    ws: ClientConnection,
    expected_type: str,
    *,
    timeout: float,
    on_unexpected: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Read events until one of `expected_type` arrives, raising on a server error or timeout.

    Unrelated events received during the handshake (e.g. rate limit notices) are skipped rather than
    treated as a protocol violation, and passed to `on_unexpected` when supplied (xAI uses this to
    capture its resumption replay burst). `timeout` bounds the total wait so `connect()` fails
    predictably instead of hanging if the expected event never arrives.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=max(0.0, deadline - time.monotonic()))
        except asyncio.TimeoutError:
            raise RealtimeHandshakeError(f'timed out waiting for a {expected_type!r} event') from None
        if not isinstance(raw, str):
            raise RealtimeHandshakeError(f'expected a text frame, got {type(raw).__name__}')
        try:
            data = loads_obj(raw)
        except ValueError as e:
            raise RealtimeHandshakeError(f'received a malformed frame: {e}') from e
        event_type = data.get('type')
        if event_type == expected_type:
            return data
        if event_type == 'error':
            raise RealtimeHandshakeError(data.get('error'))
        if on_unexpected is not None:
            on_unexpected(data)
