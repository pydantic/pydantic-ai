"""OpenAI Realtime API provider for speech-to-speech sessions.

Connects to `wss://api.openai.com/v1/realtime` over a WebSocket and maps the OpenAI event
protocol to the shared realtime event types.

Requires the `websockets` and `openai` packages, available via the `realtime` and `openai` optional
groups:

    pip install "pydantic-ai-slim[realtime,openai]"
"""

from __future__ import annotations as _annotations

import base64
import json
import warnings
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import quote

try:
    import websockets
    from openai.types.realtime import (
        ConversationItemInputAudioTranscriptionCompletedEvent,
        RealtimeResponseUsage,
        RealtimeResponseUsageInputTokenDetails,
        RealtimeResponseUsageOutputTokenDetails,
        RealtimeSessionCreateRequest,
        ResponseAudioDeltaEvent,
        ResponseCreatedEvent,
        ResponseDoneEvent,
        SessionCreatedEvent,
    )
    from openai.types.realtime.conversation_item_input_audio_transcription_completed_event import (
        UsageTranscriptTextUsageDuration,
        UsageTranscriptTextUsageTokens,
        UsageTranscriptTextUsageTokensInputTokenDetails,
    )
    from openai.types.realtime.realtime_response_usage_input_token_details import CachedTokensDetails
    from websockets.asyncio.client import ClientConnection
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `websockets` package to use the OpenAI Realtime model, '
        'you can use the `realtime` and `openai` optional groups - `pip install "pydantic-ai-slim[realtime,openai]"`'
    ) from _import_error

if TYPE_CHECKING:
    # Only needed for typing: the provider supplies the concrete client at runtime, so importing the
    # protocol helpers below (e.g. from the xAI realtime provider) doesn't require the `openai` package.
    import httpx
    from openai import AsyncOpenAI
    from openai.types.realtime.realtime_truncation_param import RealtimeTruncationParam

from .._instrumentation import get_instructions
from .._utils import is_str_dict
from ..exceptions import UserError
from ..messages import ModelMessage
from ..models import ModelRequestParameters
from ..profiles.openai import OPENAI_REASONING_EFFORT_MAP
from ..providers import Provider, infer_provider
from ..tools import ToolDefinition
from ..usage import RequestUsage
from ._base import (
    AudioDelta,
    AudioInput,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    CreateResponse,
    ImageInput,
    InputTranscript,
    RealtimeClientSecret,
    RealtimeCodecEvent,
    RealtimeConnection,
    RealtimeInput,
    RealtimeModel,
    RealtimeModelSettings,
    RealtimeProviderSession,
    ReconnectedEvent,
    ReconnectPolicy,
    SessionErrorEvent,
    SessionUsageEvent,
    TextInput,
    ToolResult,
    TruncateOutput,
    WebRTCAnswer,
    inject_trace_context,
    reconnect_with_backoff,
)
from ._openai_protocol import (
    AUDIO_DELTA_TYPES,
    INPUT_TRANSCRIPT_DONE_TYPES,
    SemanticVAD,
    ServerVAD,
    expect_event,
    loads_obj,
    map_event,
    realtime_websocket_url,
    resolve_base_turn_detection,
    resolve_transcription_model,
    response_finish_reason,
    seed_items,
    tool_choice_config,
    tool_def_to_openai,
    turn_detection_config,
    user_message_item,
    validate_response_data,
)
from ._openai_webrtc import answer_webrtc_offer as _answer_webrtc_offer, mint_client_secret as _mint_client_secret

# `input_transcription_model='auto'` resolves to this — OpenAI's recommended realtime transcription model
# ("For the lowest-latency streaming transcription path, use gpt-realtime-whisper"; it's natively streaming
# and designed for realtime sessions, unlike the legacy `whisper-1`). Kept behind the `'auto'` sentinel
# (see `resolve_transcription_model`) so it can be bumped without changing the behavior of apps on `'auto'`.
_AUTO_TRANSCRIPTION_MODEL = 'gpt-realtime-whisper'

__all__ = (
    'OpenAIRealtimeModel',
    'OpenAIRealtimeModelSettings',
    'OpenAIRealtimeConnection',
    'ServerVAD',
    'SemanticVAD',
    'map_event',
)


class OpenAIRealtimeModelSettings(RealtimeModelSettings, total=False):
    """Settings specific to OpenAI realtime models."""

    openai_input_noise_reduction: Literal['near_field', 'far_field']
    """Noise reduction tuned for `near_field` (headset) or `far_field` (laptop/conference) microphones.

    Absent disables it.
    """
    openai_output_speed: float
    """Playback speed multiplier for generated audio (0.25-1.5)."""
    openai_turn_detection: ServerVAD | SemanticVAD
    """OpenAI-specific server or semantic VAD configuration.

    When present, this fully overrides the cross-provider `turn_detection` setting.
    """
    openai_truncation: RealtimeTruncationParam
    """How the session truncates conversation context once it exceeds the model's window.

    `'auto'` (the server default) drops the oldest turns; `'disabled'` keeps everything (and errors
    when the window is full); a `retention_ratio` truncation (`{'type': 'retention_ratio',
    'retention_ratio': 0.8}`) keeps a fixed fraction, holding the prompt-cached prefix stable across
    turns (cached audio is far cheaper). This is the OpenAI SDK's `truncation` shape, forwarded as-is.
    """


def _int(value: Any) -> int:
    """Return `value` if it is an int (but not a bool), otherwise `0`."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _validate_usage_shape(usage: object, *, transcription: bool = False) -> None:
    """Reject malformed nested usage objects before accessing SDK-constructed fields."""
    if usage is None:
        return
    if not is_str_dict(usage):
        raise ValueError('`usage` must be an object')
    detail_keys = ('input_token_details',) if transcription else ('input_token_details', 'output_token_details')
    for key in detail_keys:
        details = usage.get(key)
        if details and not is_str_dict(details):
            raise ValueError(f'`usage.{key}` must be an object')
    input_details = usage.get('input_token_details')
    if is_str_dict(input_details):
        cached_details = input_details.get('cached_tokens_details')
        if cached_details and not is_str_dict(cached_details):
            raise ValueError('`usage.input_token_details.cached_tokens_details` must be an object')
    if transcription:
        # The transcription-usage union (`tokens` | `duration`) is discriminated by `type`. The SDK's
        # lenient `construct` can build the wrong variant for a malformed payload (e.g. a `duration` type
        # with no numeric `seconds`), so validate the raw shape here to keep such frames on the recoverable
        # path rather than crashing on a later `usage.seconds` read.
        usage_type = usage.get('type')
        if usage_type == 'duration':
            seconds = usage.get('seconds')
            if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
                raise ValueError('`usage.seconds` must be a number for a `duration` transcription usage')
        elif usage_type not in ('tokens', None):
            raise ValueError(f'unknown transcription usage type {usage_type!r}')


def _map_usage(usage: RealtimeResponseUsage | None) -> RequestUsage | None:
    """Map a `response.done` `usage` payload to a [`RequestUsage`][pydantic_ai.usage.RequestUsage]."""
    if usage is None or not usage.model_fields_set:
        return None
    inp = usage.input_token_details or None
    out = usage.output_token_details or None
    if inp is not None and not isinstance(inp, RealtimeResponseUsageInputTokenDetails):
        raise ValueError('`usage.input_token_details` must be an object')
    if out is not None and not isinstance(out, RealtimeResponseUsageOutputTokenDetails):
        raise ValueError('`usage.output_token_details` must be an object')
    cached = inp.cached_tokens_details if inp is not None else None
    if cached is not None and not isinstance(cached, CachedTokensDetails):
        raise ValueError('`usage.input_token_details.cached_tokens_details` must be an object')
    details: dict[str, int] = {}
    for key, raw in (
        ('input_text_tokens', inp.text_tokens if inp is not None else None),
        ('input_image_tokens', inp.image_tokens if inp is not None else None),
        ('output_text_tokens', out.text_tokens if out is not None else None),
    ):
        if isinstance(raw, int) and not isinstance(raw, bool):
            details[key] = raw
    # xAI-specific usage (absent on OpenAI): the `grok_tokens` buckets, and second-based audio billing —
    # xAI bills Grok Voice by audio second, so `billable_audio_seconds` is the authoritative cost and is
    # not reconstructable from token counts. Included only when non-zero, so OpenAI's `details` is unchanged.
    for key, raw in (
        ('input_grok_tokens', (inp.model_extra or {}).get('grok_tokens') if inp is not None else None),  # xAI extra
        ('output_grok_tokens', (out.model_extra or {}).get('grok_tokens') if out is not None else None),  # xAI extra
        ('billable_audio_seconds', (usage.model_extra or {}).get('billable_audio_seconds')),  # xAI extra
    ):
        if isinstance(raw, int) and not isinstance(raw, bool) and raw:
            details[key] = raw
    return RequestUsage(
        input_tokens=_int(usage.input_tokens),
        output_tokens=_int(usage.output_tokens),
        input_audio_tokens=_int(inp.audio_tokens if inp is not None else None),
        cache_read_tokens=_int(inp.cached_tokens if inp is not None else None),
        cache_audio_read_tokens=_int(cached.audio_tokens if cached is not None else None),
        output_audio_tokens=_int(out.audio_tokens if out is not None else None),
        details=details,
    )


RealtimeTranscriptionUsage = UsageTranscriptTextUsageTokens | UsageTranscriptTextUsageDuration


def _map_transcription_usage(usage: RealtimeTranscriptionUsage | None) -> RequestUsage | None:
    """Map input-transcription usage into separate [`RequestUsage.details`][pydantic_ai.usage.RequestUsage.details]."""
    if usage is None or not usage.model_fields_set:
        return None
    details: dict[str, int] = {}
    if isinstance(usage, UsageTranscriptTextUsageTokens):
        # Branch on the variant, not `usage.type`: a protocol clone (xAI/Azure) may omit the `type`
        # discriminator, which `_validate_usage_shape` deliberately tolerates, and the SDK's lenient
        # `.construct` then builds this tokens variant with `type=None`. A type-less payload must be read
        # as tokens rather than reaching the duration branch, whose `usage.seconds` this variant lacks —
        # previously an `AttributeError` that escaped the recoverable path and tore the session down.
        token_details = usage.input_token_details or None
        if token_details is not None and not isinstance(token_details, UsageTranscriptTextUsageTokensInputTokenDetails):
            raise ValueError('`usage.input_token_details` must be an object')
        for key, raw in (
            ('input_transcription_tokens', usage.total_tokens),
            ('input_transcription_audio_tokens', token_details.audio_tokens if token_details is not None else None),
            ('input_transcription_text_tokens', token_details.text_tokens if token_details is not None else None),
        ):
            if isinstance(raw, int) and not isinstance(raw, bool):
                details[key] = raw
    elif (seconds := round(usage.seconds)) > 0:
        # `RunUsage.details` values are ints; round the fractional-second duration rather than drop it.
        details['input_transcription_seconds'] = seconds
    return RequestUsage(details=details) if details else None


class OpenAIRealtimeConnection(RealtimeConnection):
    """A live WebSocket connection to the OpenAI Realtime API."""

    _provider_name = 'openai'
    _supports_tool_result_images = True

    def __init__(
        self,
        ws: ClientConnection,
        *,
        dial: Callable[[], Awaitable[ClientConnection]] | None = None,
        reconnect: ReconnectPolicy | None = None,
        input_transcription_enabled: bool = True,
        model_name: str | None = None,
        observes_output_audio: bool = True,
    ) -> None:
        self._ws = ws
        self._model_name = model_name
        # `dial` re-establishes a fully configured connection; with a `reconnect` policy it is used to
        # recover from a dropped WebSocket.
        self._dial = dial
        self._reconnect = reconnect
        self._restores_state_on_reconnect = False
        self._input_transcription_enabled = input_transcription_enabled
        self._observes_output_audio = observes_output_audio
        # The Realtime API rejects `response.create` while a response is already being generated.
        # We track that window and defer requests (e.g. a background tool result that lands while the
        # model is mid-answer) until the active response finishes, so the model still announces it.
        self._response_active = False
        self._active_response_id: str | None = None
        self._pending_response = False
        self._cancel_sent = False
        # Id of a response we cancelled (barge-in): the server keeps streaming a few straggler deltas
        # before its `response.done`, and mapping them would surface speech the user already interrupted.
        # Drop frames carrying this id until its `response.done` closes it. `None` when nothing is being
        # cancelled, or when the cancelled response had no id (defensive/compat path — then not suppressed).
        self._cancelled_response_id: str | None = None
        # The current output audio item, tracked from output-audio deltas so a `TruncateOutput` can
        # name it. These are mutated by `__aiter__` and read by `send` from a separate task; under a
        # single cooperative event loop the plain reads/writes are safe and eventually consistent.
        self._current_item_id: str | None = None
        self._current_content_index = 0
        self._generated_audio_bytes = 0

    @property
    def model_name(self) -> str | None:
        return self._model_name

    @property
    def input_transcription_enabled(self) -> bool:
        return self._input_transcription_enabled

    async def send(self, content: RealtimeInput) -> None:
        """Send content to the OpenAI Realtime API.

        Accepts `AudioInput` (PCM16, 24kHz, mono), `TextInput`, `ImageInput`, `ToolResult`, and the
        control verbs `CommitAudio`, `ClearAudio`, `CreateResponse`, `CancelResponse`, and
        `TruncateOutput`.
        """
        if isinstance(content, AudioInput):
            await self._send_event(
                {
                    'type': 'input_audio_buffer.append',
                    'audio': base64.b64encode(content.data).decode('ascii'),
                }
            )
        elif isinstance(content, TextInput):
            await self._send_event(
                {
                    'type': 'conversation.item.create',
                    'item': {
                        'type': 'message',
                        'role': 'user',
                        'content': [{'type': 'input_text', 'text': content.text}],
                    },
                }
            )
            await self._request_response()
        elif isinstance(content, ToolResult):
            await self._send_event(
                {
                    'type': 'conversation.item.create',
                    'item': {
                        'type': 'function_call_output',
                        'call_id': content.tool_call_id,
                        'output': content.output,
                    },
                }
            )
            if content.content and (
                item := await user_message_item(
                    content.content,
                    provider_name=self._provider_name,
                    supports_images=self._supports_tool_result_images,
                )
            ):
                await self._send_event({'type': 'conversation.item.create', 'item': item})
            await self._request_response()
        elif isinstance(content, ImageInput):
            # An image is added as conversation context (like a video frame), not a turn of its own,
            # so it doesn't trigger a response — drive that with audio (VAD) or `CreateResponse`.
            data_uri = f'data:{content.media_type};base64,{base64.b64encode(content.data).decode("ascii")}'
            await self._send_event(
                {
                    'type': 'conversation.item.create',
                    'item': {
                        'type': 'message',
                        'role': 'user',
                        'content': [{'type': 'input_image', 'image_url': data_uri}],
                    },
                }
            )
        elif isinstance(content, CommitAudio):
            await self._send_event({'type': 'input_audio_buffer.commit'})
        elif isinstance(content, ClearAudio):
            await self._send_event({'type': 'input_audio_buffer.clear'})
        elif isinstance(content, CreateResponse):
            await self._request_response()
        elif isinstance(content, CancelResponse):
            # Only cancel when a response is actually active: with server VAD the provider may have
            # already cancelled on the user's barge-in, and a redundant cancel raises a session error.
            if self._response_active and not self._cancel_sent:
                await self._send_event({'type': 'response.cancel'})
                self._cancel_sent = True
                # Suppress the cancelled response's trailing deltas until its `response.done` arrives.
                self._cancelled_response_id = self._active_response_id
                # The cancelled response's output item is gone; forget it so a following
                # `interrupt(audio_end_ms=...)` before the next turn's first audio doesn't truncate a stale
                # item (server-initiated cancels clear it via `response.done`, but a client cancel doesn't).
                self._current_item_id = None
                self._generated_audio_bytes = 0
        elif isinstance(content, TruncateOutput):
            # No current output item (e.g. the model wasn't speaking) → nothing to truncate.
            if self._current_item_id is not None:
                audio_end_ms = content.audio_end_ms
                # A WebRTC sideband connection does not receive output-audio deltas because media flows
                # directly between the browser and provider. Only clamp connections that observe those
                # deltas; otherwise the byte counter stays zero and every barge-in would truncate to zero.
                if self._observes_output_audio:
                    max_audio_end_ms = self._generated_audio_bytes * 1000 // 48_000
                    audio_end_ms = min(audio_end_ms, max_audio_end_ms)
                await self._send_event(
                    {
                        'type': 'conversation.item.truncate',
                        'item_id': self._current_item_id,
                        'content_index': self._current_content_index,
                        'audio_end_ms': audio_end_ms,
                    }
                )
        else:
            raise NotImplementedError(f'OpenAI Realtime does not support {type(content).__name__} input')

    async def _request_response(self) -> None:
        """Ask the model to respond now, or defer until the active response completes."""
        if self._response_active:
            self._pending_response = True
        else:
            self._response_active = True
            self._active_response_id = None
            await self._send_event({'type': 'response.create'})

    async def _send_event(self, event: dict[str, Any]) -> None:
        await self._ws.send(json.dumps(event))

    def _map_event(self, data: dict[str, Any]) -> RealtimeCodecEvent | None:
        """Map a raw provider frame to a codec event.

        A hook so protocol clones (e.g. the xAI Grok Voice provider) can reuse the whole connection while
        overriding only how frames map to events. Defaults to the OpenAI [`map_event`][pydantic_ai.realtime.openai.map_event].
        """
        return map_event(data)

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        while True:
            try:
                async for raw in self._ws:
                    if not isinstance(raw, str):
                        continue
                    try:
                        events = await self._decode_frame(raw)
                    except ValueError as e:
                        # A malformed frame (bad JSON or audio payload) shouldn't tear down the whole
                        # session; surface it as a recoverable error and keep reading.
                        yield SessionErrorEvent(message=f'Failed to parse OpenAI realtime event: {e}', recoverable=True)
                        continue
                    for event in events:
                        yield event
                return  # the upstream iterator ended without dropping
            except websockets.ConnectionClosed as e:
                if self._reconnect is None or self._dial is None:
                    # No reconnect policy: a dropped connection is fatal. Surface it as a
                    # non-recoverable error and end the stream cleanly, rather than raising.
                    yield SessionErrorEvent(message=f'OpenAI realtime connection closed: {e}', recoverable=False)
                    return
                if await self._try_reconnect():
                    yield ReconnectedEvent(state_restored=self._restores_state_on_reconnect)
                    continue
                yield SessionErrorEvent(
                    message=f'OpenAI realtime connection closed; reconnect failed: {e}', recoverable=False
                )
                return

    def _is_cancelled_straggler(self, event_type: str | None, data: dict[str, Any]) -> bool:
        """Whether this frame is a trailing delta from a response cancelled on barge-in (drop it).

        Its own `response.done` is excluded so the response still closes cleanly; frames for any other
        response carry a different `response_id`.
        """
        return (
            self._cancelled_response_id is not None
            and event_type != 'response.done'
            and data.get('response_id') == self._cancelled_response_id
        )

    def _close_cancelled_response(self, response_id: str | None) -> None:
        """Stop suppressing stragglers once a cancelled response's `response.done` arrives."""
        if response_id == self._cancelled_response_id:
            self._cancelled_response_id = None

    async def _decode_frame(self, raw: str) -> list[RealtimeCodecEvent]:
        """Parse one text frame into events, updating tracked response state.

        Raises `ValueError` (incl. `json.JSONDecodeError` / `binascii.Error`) on a malformed payload.
        """
        data = loads_obj(raw)
        event_type = data.get('type')
        # Drop trailing frames from a response we cancelled on barge-in (its audio/transcript deltas,
        # output-item events, etc.); its own `response.done` still passes through below to close the
        # response, emit usage, and clear the suppression.
        if self._is_cancelled_straggler(event_type, data):
            return []
        events: list[RealtimeCodecEvent] = []
        superseded = False
        event = self._map_event(data)
        if event_type == 'response.created':
            response_data = data.get('response')
            if response_data is not None and not is_str_dict(response_data):
                raise ValueError('`response` must be an object')
            created = ResponseCreatedEvent.construct(**data)
            self._response_active = True
            self._active_response_id = created.response.id or None if is_str_dict(response_data) else None
        elif event_type in AUDIO_DELTA_TYPES:
            audio = ResponseAudioDeltaEvent.construct(**data)
            # Track the speaking item so a later `TruncateOutput` can name it.
            if audio.item_id and isinstance(event, AudioDelta):
                item_changed = (audio.item_id, audio.content_index or 0) != (
                    self._current_item_id,
                    self._current_content_index,
                )
                self._current_item_id = audio.item_id
                self._current_content_index = audio.content_index or 0
                if item_changed:
                    self._generated_audio_bytes = len(event.data)
                else:
                    self._generated_audio_bytes += len(event.data)
        elif event_type == 'response.done':
            done_events, superseded = await self._handle_response_done(data)
            events.extend(done_events)
        if event is not None and not (event_type == 'response.done' and superseded):
            events.append(event)
            if isinstance(event, InputTranscript) and event.is_final and event_type in INPUT_TRANSCRIPT_DONE_TYPES:
                _validate_usage_shape(data.get('usage'), transcription=True)
                completed = ConversationItemInputAudioTranscriptionCompletedEvent.construct(**data)
                if (asr := _map_transcription_usage(completed.usage)) is not None:
                    events.append(SessionUsageEvent(usage=asr, response_scoped=False))
        return events

    async def _handle_response_done(self, data: dict[str, Any]) -> tuple[list[RealtimeCodecEvent], bool]:
        """Update response state and emit usage for a `response.done`.

        Returns `(events, superseded)`. `superseded` is `True` when a *different* response is still
        active — a late/cancelled completion arriving after a new turn began — so the caller suppresses
        its user-facing `TurnCompleteEvent` (which would otherwise finalize the current response's output
        under this old boundary). A frame with no `response` object is malformed/empty; `map_event`
        handles it gracefully, so return early here.
        """
        events: list[RealtimeCodecEvent] = []
        response_data = validate_response_data(data)
        if not response_data:
            return events, False
        _validate_usage_shape(response_data.get('usage'))
        done = ResponseDoneEvent.construct(**data)
        response = done.response
        response_id = response.id
        # The cancelled response is now closed; stop suppressing its stragglers (its own usage still emits
        # below). A no-op for any other response.
        self._close_cancelled_response(response_id)
        # OpenAI response events always carry an ID. Keep the ID-less fallback for compatible protocol
        # implementations and defensive unit inputs that predate response tracking.
        matches_active_response = not isinstance(response_id, str) or (
            self._response_active and response_id == self._active_response_id
        )
        # Superseded only when a *different, known* response is active — a late/cancelled completion after
        # a new turn began. When the active id is unknown (`None`, e.g. an id-less `response.created`), this
        # done can't be proven stale, so don't suppress its turn-completion.
        superseded = (
            isinstance(response_id, str)
            and self._active_response_id is not None
            and response_id != self._active_response_id
        )
        was_client_cancel = matches_active_response and self._cancel_sent
        if matches_active_response:
            self._response_active = False
            self._active_response_id = None
            self._cancel_sent = False
            self._current_item_id = None
            self._generated_audio_bytes = 0
        # Emit usage for every response (including intermediate function-call-only ones) so the session
        # accounts for all tokens. Only the active response may replay a pending request; a late completion
        # for a superseded response must not change current state. OpenAI nests usage under
        # `response.usage`; xAI Grok Voice reports the same shape at the top level of the `response.done`
        # frame (its `response.usage` is empty), so fall back to it.
        top_level_usage = (done.model_extra or {}).get('usage')  # xAI frame-level provider extra.
        _validate_usage_shape(top_level_usage)
        usage = _map_usage(response.usage) or _map_usage(
            RealtimeResponseUsage.construct(**top_level_usage) if is_str_dict(top_level_usage) else None
        )
        if usage is not None:
            events.append(
                SessionUsageEvent(
                    usage=usage,
                    provider_response_id=response_id or None,
                    finish_reason=response_finish_reason(response),
                )
            )
        elif matches_active_response and response_finish_reason(response) == 'tool_call':
            events.append(
                SessionUsageEvent(
                    usage=RequestUsage(),
                    provider_response_id=response_id or None,
                    finish_reason='tool_call',
                )
            )
        if matches_active_response and self._pending_response:
            self._pending_response = False
            # A *server*-cancelled response means the user barged in: a new turn is starting, so don't
            # replay the deferred response over it. After a *client* cancel (`interrupt()`), the caller
            # explicitly queued the next response behind the cancel, so send it now that the cancelled
            # response has closed.
            if response.status != 'cancelled' or was_client_cancel:
                self._response_active = True
                self._active_response_id = None
                await self._send_event({'type': 'response.create'})
        return events, superseded

    async def _try_reconnect(self) -> bool:
        """Re-dial with exponential backoff; return whether a new connection was established."""
        assert self._reconnect is not None and self._dial is not None
        return await reconnect_with_backoff(self._reconnect, self._attempt_reconnect)

    async def _attempt_reconnect(self) -> bool:
        assert self._dial is not None
        try:
            self._ws = await self._dial()
        except (websockets.WebSocketException, OSError, TimeoutError):
            # Expected dial/handshake failures: protocol/connection errors, network failures (DNS,
            # refused, reset), and the handshake timeout. A retry may still succeed. Anything else is a
            # bug in `dial()` and propagates rather than masquerading as a failed reconnect.
            return False
        self._response_active = False
        self._active_response_id = None
        self._pending_response = False
        self._cancel_sent = False
        self._current_item_id = None
        self._generated_audio_bytes = 0
        self._cancelled_response_id = None
        return True


@dataclass
class OpenAIRealtimeModel(RealtimeModel):
    """OpenAI Realtime API model.

    Authentication and the base URL come from a
    [`Provider`][pydantic_ai.providers.Provider], mirroring [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel].
    Pass `provider='openai'` (the default) to read `OPENAI_API_KEY` / `OPENAI_BASE_URL` from the
    environment, or an [`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider] instance for a
    custom key or base URL. The realtime transport is opened separately with `websockets`, so the
    provider's `httpx` client is not used for the WebSocket connection. The realtime WebSocket URL is
    derived from the provider's base URL (e.g. `https://api.openai.com/v1/` →
    `wss://api.openai.com/v1/realtime`), so OpenAI-compatible endpoints that expose a realtime API
    work too.

    Args:
        model: The model name, e.g. `gpt-realtime` or `gpt-realtime-2.1-mini`.
        provider: The provider to use for authentication and the base URL. Defaults to `'openai'`.
            Azure OpenAI is not supported (its realtime endpoint uses a different URL and auth scheme).
        reconnect: Optional [`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy] to transparently
            recover from a dropped connection. With no policy, the low-level connection reports a
            non-recoverable session error; `RealtimeSession` raises
            [`RealtimeError`][pydantic_ai.realtime.RealtimeError] from iteration.
    """

    model: str = 'gpt-realtime'
    provider: InitVar[Provider[AsyncOpenAI] | str] = 'openai'
    settings: RealtimeModelSettings | None = field(default=None, kw_only=True)
    reconnect: ReconnectPolicy | None = None
    _provider: Provider[AsyncOpenAI] = field(init=False, repr=False)

    def __post_init__(self, provider: Provider[AsyncOpenAI] | str) -> None:
        if isinstance(provider, str):
            provider = cast('Provider[AsyncOpenAI]', infer_provider(provider))
        if provider.name == 'azure':
            raise UserError(
                'Azure OpenAI is not supported through `OpenAIRealtimeModel`: its realtime endpoint uses a '
                'different URL and authentication scheme. Use `AzureRealtimeModel` (or the '
                '`azure:` prefix) for Azure OpenAI realtime.'
            )
        self._provider = provider

    @property
    def client(self) -> AsyncOpenAI:
        """The underlying [`AsyncOpenAI`](https://github.com/openai/openai-python) client from the provider."""
        return self._provider.client

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def system(self) -> str:
        return self._provider.name

    def _session_config(
        self,
        instructions: str,
        tools: list[ToolDefinition] | None,
        model_settings: OpenAIRealtimeModelSettings | None,
    ) -> dict[str, Any]:
        model_settings = cast('OpenAIRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        if 'openai_turn_detection' in model_settings:
            turn_detection = model_settings['openai_turn_detection']
        elif 'turn_detection' in model_settings:
            turn_detection = resolve_base_turn_detection(model_settings['turn_detection'])
        else:
            turn_detection = ServerVAD()
        # `turn_detection` is always set: a dict enables VAD, `None` (explicit null) disables it.
        audio_input: dict[str, Any] = {
            'format': {'type': 'audio/pcm', 'rate': 24000},
            'turn_detection': turn_detection_config(turn_detection),
        }
        transcription_model = resolve_transcription_model(
            model_settings.get('input_transcription_model', 'auto'), default=_AUTO_TRANSCRIPTION_MODEL
        )
        if transcription_model is not None:
            audio_input['transcription'] = {'model': transcription_model}
        if (noise_reduction := model_settings.get('openai_input_noise_reduction')) is not None:
            audio_input['noise_reduction'] = {'type': noise_reduction}
        audio_output: dict[str, Any] = {'format': {'type': 'audio/pcm', 'rate': 24000}}
        if voice := model_settings.get('voice'):
            audio_output['voice'] = voice
        if (output_speed := model_settings.get('openai_output_speed')) is not None:
            audio_output['speed'] = output_speed
        config: dict[str, Any] = {
            'type': 'realtime',
            'instructions': instructions,
            'output_modalities': [model_settings.get('output_modality', 'audio')],
            'audio': {'input': audio_input, 'output': audio_output},
        }
        if tools:
            config['tools'] = [tool_def_to_openai(t) for t in tools]
        # Note: GA realtime sessions have no `temperature` field, so it is intentionally not forwarded.
        if (max_tokens := model_settings.get('max_tokens')) is not None:
            config['max_output_tokens'] = max_tokens
        if (parallel_tool_calls := model_settings.get('parallel_tool_calls')) is not None:
            config['parallel_tool_calls'] = parallel_tool_calls
        if (tool_choice := tool_choice_config(model_settings.get('tool_choice'))) is not None:
            config['tool_choice'] = tool_choice
        if (truncation := model_settings.get('openai_truncation')) is not None:
            # Already the OpenAI `truncation` wire shape (`'auto'`/`'disabled'`/retention-ratio dict).
            config['truncation'] = truncation
        if (thinking := model_settings.get('thinking')) is not None:
            if self.profile.get('supports_thinking', False):
                # `False` maps to `'none'`, which the realtime `reasoning.effort` doesn't accept — omit
                # it so a reasoning model falls back to its default rather than erroring.
                if (effort := OPENAI_REASONING_EFFORT_MAP[thinking]) != 'none':
                    config['reasoning'] = {'effort': effort}
            else:
                warnings.warn(
                    f'The {self.model!r} realtime model does not support the `thinking` setting '
                    '(only the `gpt-realtime-2*` reasoning models do); ignoring it.',
                    UserWarning,
                )
        return config

    def _realtime_ws_base(self) -> str:
        """The realtime WebSocket URL without a query string, e.g. `wss://api.openai.com/v1/realtime`."""
        return realtime_websocket_url(self._provider.base_url)

    def _realtime_url(self, model_settings: OpenAIRealtimeModelSettings | None = None) -> str:
        del model_settings  # only the Azure Voice Live override varies the URL on settings
        return f'{self._realtime_ws_base()}?model={self.model}'

    def _sideband_url(self, call_id: str) -> str:
        """The control-plane WebSocket URL that attaches to an existing WebRTC call by `call_id`."""
        return f'{self._realtime_ws_base()}?call_id={quote(call_id, safe="")}'

    def _webrtc_http_base(self) -> str:
        """The HTTP base URL for realtime signaling, always ending in `/` (e.g. `https://api.openai.com/v1/`)."""
        base_url = self._provider.base_url
        return base_url if base_url.endswith('/') else f'{base_url}/'

    def _webrtc_calls_url(self) -> str:
        """The `/realtime/calls` signaling endpoint the browser's SDP offer is relayed to."""
        return f'{self._webrtc_http_base()}realtime/calls'

    def _webrtc_client_secrets_url(self) -> str:
        """The `/realtime/client_secrets` endpoint that mints ephemeral browser tokens."""
        return f'{self._webrtc_http_base()}realtime/client_secrets'

    @property
    def _http_client(self) -> httpx.AsyncClient:
        """The provider's configured `httpx` client, reused for realtime WebRTC signaling."""
        return self._provider.client._client  # pyright: ignore[reportPrivateUsage]

    async def _webrtc_headers(self) -> dict[str, str]:
        """Non-auth default headers from the provider client, plus this model's realtime auth header.

        Auth (`Authorization: Bearer` for OpenAI, `api-key` or Entra `Bearer` for Azure) comes from
        `_auth_headers`, replacing whatever the SDK client carries by default, so a single code path
        signs both the OpenAI and Azure requests.
        """
        headers = {
            key: value
            for key, value in self._provider.client.default_headers.items()
            if isinstance(value, str) and key.lower() not in ('accept', 'content-type', 'authorization', 'api-key')
        }
        headers.update(await self._auth_headers())
        return headers

    def _webrtc_session_config(
        self,
        instructions: str | None,
        tools: Sequence[ToolDefinition] | None,
        model_settings: RealtimeModelSettings | None,
    ) -> dict[str, Any]:
        """The `session` object sent to `/realtime/calls` and `/realtime/client_secrets`.

        Unlike the WebSocket handshake, which carries the model in the `?model=` query, the WebRTC
        signaling endpoints read the model from the session body, so it is injected here.
        """
        settings = cast('OpenAIRealtimeModelSettings | None', model_settings)
        return {
            'model': self.model,
            **self._session_config(instructions or '', list(tools) if tools else None, settings),
        }

    async def create_client_secret(
        self,
        *,
        instructions: str | None = None,
        tools: Sequence[ToolDefinition] | None = None,
        model_settings: RealtimeModelSettings | None = None,
        expires_after_seconds: int | None = None,
    ) -> RealtimeClientSecret:
        return await _mint_client_secret(
            http_client=self._http_client,
            client_secrets_url=self._webrtc_client_secrets_url(),
            headers=await self._webrtc_headers(),
            provider_name=self.system,
            session_config=self._webrtc_session_config(instructions, tools, model_settings),
            expires_after_seconds=expires_after_seconds,
        )

    async def answer_webrtc_offer(
        self,
        sdp_offer: str,
        *,
        instructions: str | None = None,
        tools: Sequence[ToolDefinition] | None = None,
        model_settings: RealtimeModelSettings | None = None,
    ) -> WebRTCAnswer:
        return await _answer_webrtc_offer(
            http_client=self._http_client,
            calls_url=self._webrtc_calls_url(),
            headers=await self._webrtc_headers(),
            provider_name=self.system,
            sdp_offer=sdp_offer,
            session_config=self._webrtc_session_config(instructions, tools, model_settings),
        )

    @asynccontextmanager
    async def connect_webrtc(
        self,
        session: RealtimeProviderSession,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncGenerator[OpenAIRealtimeConnection]:
        if session.provider_name != self.system:
            raise UserError(
                f'This WebRTC call was negotiated by provider {session.provider_name!r}, but this realtime '
                f'model connects through {self.system!r}. Answer the offer and attach the sideband with the '
                'same model/provider.'
            )
        url = self._sideband_url(session.session_id)
        headers = await self._auth_headers()
        # Propagate trace context over the handshake (see `connect` for the rationale).
        inject_trace_context(headers)
        settings = cast('OpenAIRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        handshake_timeout = settings.get('handshake_timeout', 30.0)
        instructions = get_instructions(messages) or ''
        session_config = self._session_config(instructions, model_request_parameters.function_tools, settings)
        transcription_enabled = settings.get('input_transcription_model', 'auto') is not None

        cm: AbstractAsyncContextManager[ClientConnection] | None = None
        server_model: str | None = None

        async def dial() -> ClientConnection:
            nonlocal cm, server_model
            if cm is not None:
                previous, cm = cm, None
                await previous.__aexit__(None, None, None)
            opening = websockets.connect(url, additional_headers=headers)
            ws = await opening.__aenter__()
            cm = opening
            # The call already exists (created when the SDP offer was relayed), so the control WebSocket
            # doesn't emit `session.created`: apply the session config immediately and wait for
            # `session.updated`, which also reports the served model.
            await ws.send(json.dumps({'type': 'session.update', 'session': session_config}))
            updated = await expect_event(ws, 'session.updated', timeout=handshake_timeout)
            session = updated.get('session')
            model = session.get('model') if is_str_dict(session) else None
            if isinstance(model, str) and model:
                server_model = model
            return ws

        try:
            ws = await dial()
            # Seed prior conversation once, after the handshake (as the WebSocket path does).
            for item in await seed_items(messages, profile=self.profile, provider_name=self.system):
                await ws.send(json.dumps({'type': 'conversation.item.create', 'item': item}))
            yield OpenAIRealtimeConnection(
                ws,
                dial=dial,
                reconnect=self.reconnect,
                input_transcription_enabled=transcription_enabled,
                model_name=server_model,
            )
        finally:
            if cm is not None:  # pragma: no branch
                await cm.__aexit__(None, None, None)

    async def _auth_headers(self, model_settings: OpenAIRealtimeModelSettings | None = None) -> dict[str, str]:
        # `model_settings` lets a provider vary auth by session (e.g. Azure Voice Live uses a different
        # resource key); OpenAI's auth doesn't depend on it.
        del model_settings
        # `AsyncOpenAI` accepts an async `api_key` provider, in which case `client.api_key` is empty
        # until resolved. The raw WebSocket handshake bypasses the SDK's request path (which resolves
        # it per request), so resolve it here via the SDK's own refresh — a no-op returning the static
        # key when no provider is configured, so the handshake stays byte-identical in that case.
        api_key = await self._provider.client._refresh_api_key()  # pyright: ignore[reportPrivateUsage]
        return {'Authorization': f'Bearer {api_key}'}

    def _connection_class(self, model_settings: OpenAIRealtimeModelSettings) -> type[OpenAIRealtimeConnection]:
        del model_settings
        return OpenAIRealtimeConnection

    @asynccontextmanager
    async def connect(
        self,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncGenerator[OpenAIRealtimeConnection]:
        settings = cast('OpenAIRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        url = self._realtime_url(settings)
        headers = await self._auth_headers(settings)
        # Propagate trace context over the handshake so a proxy (e.g. the Pydantic AI Gateway) can nest
        # its realtime spans under this session's trace; the raw WebSocket bypasses the provider's
        # `httpx` client, which would otherwise inject it.
        inject_trace_context(headers)
        handshake_timeout = settings.get('handshake_timeout', 30.0)
        instructions = get_instructions(messages) or ''
        session_config = self._session_config(instructions, model_request_parameters.function_tools, settings)
        transcription_enabled = settings.get('input_transcription_model', 'auto') is not None

        # `dial` opens and configures a fresh connection. A reconnect closes the previous connection
        # (including one left half-open by a failed handshake) before opening the next, so sockets
        # don't accumulate; teardown closes whatever is current.
        cm: AbstractAsyncContextManager[ClientConnection] | None = None

        # The model the server reports actually serving, from the `session.created` handshake; it can
        # differ from the requested id (see `RealtimeConnection.model_name`).
        server_model: str | None = None

        async def dial() -> ClientConnection:
            nonlocal cm, server_model
            if cm is not None:
                previous, cm = cm, None
                await previous.__aexit__(None, None, None)
            opening = websockets.connect(url, additional_headers=headers)
            ws = await opening.__aenter__()
            cm = opening
            created = await expect_event(ws, 'session.created', timeout=handshake_timeout)
            session = SessionCreatedEvent.construct(**created).session
            model = session.model if isinstance(session, RealtimeSessionCreateRequest) else None
            if isinstance(model, str) and model:
                server_model = model
            await ws.send(json.dumps({'type': 'session.update', 'session': session_config}))
            await expect_event(ws, 'session.updated', timeout=handshake_timeout)
            return ws

        try:
            ws = await dial()
            # Seed prior conversation once, after the initial handshake. Reconnects deliberately don't
            # re-seed: server state is lost on drop and a `ReconnectedEvent` starts a fresh turn.
            for item in await seed_items(messages, profile=self.profile, provider_name=self.system):
                await ws.send(json.dumps({'type': 'conversation.item.create', 'item': item}))
            yield self._connection_class(settings)(
                ws,
                dial=dial,
                reconnect=self.reconnect,
                input_transcription_enabled=transcription_enabled,
                model_name=server_model,
            )
        finally:
            # Coverage cannot attribute a failed `__aenter__` to the false exit arc; the behavior is
            # exercised by `test_connect_open_failure_propagates_without_teardown`.
            if cm is not None:  # pragma: no branch
                await cm.__aexit__(None, None, None)
