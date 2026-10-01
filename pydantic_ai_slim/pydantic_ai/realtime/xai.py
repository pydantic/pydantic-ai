"""xAI Grok Voice realtime API provider for speech-to-speech sessions.

Connects to `wss://api.x.ai/v1/realtime` over a WebSocket. xAI's realtime API is a deliberate clone of
the OpenAI Realtime protocol, so this provider reuses the OpenAI codec from
[`pydantic_ai.realtime.openai`][pydantic_ai.realtime.openai] — event mapping, session seeding, tool
conversion, server-VAD config, and the WebSocket connection itself — and diverges only where xAI does:

- the `session.update` shape (`voice`/`turn_detection` sit at the session top level, not nested under
  `audio` as on OpenAI's GA surface);
- input audio transcription, delivered as cumulative
  `conversation.item.input_audio_transcription.updated` snapshots plus a final `.completed`, rather
  than OpenAI's incremental `.delta` events (see [`map_event`][pydantic_ai.realtime.xai.map_event]);
- native conversation resumption when a reconnect policy is configured: the provider-assigned
  `conversation.id` is reused and its replay burst is suppressed from local history;
- no output truncation at barge-in: xAI's `conversation.item.truncate` only lands once the response
  has ended, and unreliably even then, so
  [`RealtimeModelProfile.supports_output_truncation`][pydantic_ai.realtime.RealtimeModelProfile.supports_output_truncation]
  is `False` while cancellation-based interruption still works;
- no text output — the API has no response-modality control and always speaks — so
  [`RealtimeModelProfile.supports_text_output`][pydantic_ai.realtime.RealtimeModelProfile.supports_text_output]
  is `False` and `output_modality='text'` raises rather than silently coming back as audio.

Requires the `websockets` package (the `realtime` optional group), `xai-sdk` (the `xai` group, for
[`XaiProvider`][pydantic_ai.providers.xai.XaiProvider]), and `openai` (the `openai` group, whose SDK
supplies the event types the shared OpenAI codec is built on):

    pip install "pydantic-ai-slim[xai-realtime]"
"""

from __future__ import annotations as _annotations

from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import quote

try:
    import websockets as websockets
    from openai.types.realtime import (
        ConversationCreatedEvent,
        ConversationItem,
        ConversationItemAdded,
        ConversationItemCreatedEvent,
        RealtimeConversationItemFunctionCall,
        RealtimeConversationItemFunctionCallOutput,
        RealtimeResponseUsage,
        ResponseFunctionCallArgumentsDoneEvent,
    )
    from pydantic import BaseModel, ConfigDict, TypeAdapter
    from websockets.asyncio.client import ClientConnection
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `websockets` and `openai` packages to use the xAI Grok Voice realtime model '
        '(`openai` supplies the event types of the OpenAI codec this provider reuses), you can use the '
        '`realtime`, `xai`, and `openai` optional groups - '
        '`pip install "pydantic-ai-slim[xai-realtime]"`'
    ) from _import_error

from .._instrumentation import get_instructions
from ..exceptions import UserError
from ..messages import BinaryAudio, ModelMessage, RealtimeSessionReconnectEvent
from ..models import ModelRequestParameters
from ..providers import Provider, infer_provider
from ..tools import ToolDefinition
from ..usage import RequestUsage
from ._lifecycle import InputId, LifecycleEvent
from ._openai_protocol import (
    INPUT_AUDIO_BUFFER_APPEND_EVENT,
    INPUT_AUDIO_BUFFER_CLEAR_EVENT,
    INPUT_AUDIO_BUFFER_COMMIT_EVENT,
    RealtimeHandshakeError,
    client_event_id,
    config_interrupts_response_on_speech,
    connect_openai_protocol,
    expect_event,
    map_event as _map_openai_event,
    realtime_websocket_url,
    resolve_base_turn_detection,
    resolve_transcription_model,
    tool_choice_config,
    tool_def_to_openai,
    turn_detection_config,
)
from ._utils import inject_trace_context, resolve_advertised_tools
from .codec import (
    CancelResponse,
    ClearAudio,
    CommitAudio,
    ConversationCreated,
    ConversationItemCreated,
    CreateResponse,
    InputTranscript,
    RealtimeCodecEvent,
    RealtimeInput,
    ToolCall,
    TruncateOutput,
)
from .model import RealtimeModel
from .openai import OpenAIRealtimeConnection, ServerVAD, _DecodedFrame  # pyright: ignore[reportPrivateUsage]
from .profiles import RealtimeModelProfileSpec
from .settings import RealtimeModelSettings, ReconnectPolicy

if TYPE_CHECKING:
    from xai_sdk import AsyncClient

    from ..providers.xai import XaiProvider

# `input_transcription_model='auto'` resolves to this — xAI's realtime transcription model. Kept behind
# the `'auto'` sentinel (see `resolve_transcription_model`) so it can change without altering the behavior
# of apps on `'auto'`.
_AUTO_TRANSCRIPTION_MODEL = 'grok-transcribe'
_CONVERSATION_CREATED_EVENT = 'conversation.created'

LatestXaiRealtimeModelNames = Literal['grok-voice-latest', 'grok-voice-think-fast-2.0']
XaiRealtimeModelName = str | LatestXaiRealtimeModelNames

LatestXaiRealtimeTranscriptionModelNames = Literal['grok-transcribe']
XaiRealtimeTranscriptionModelName = str | LatestXaiRealtimeTranscriptionModelNames


class _ProtocolConversationItem(BaseModel):
    """Minimal typed item for xAI conversation lifecycle frames."""

    model_config = ConfigDict(extra='allow')

    type: str
    id: str | None = None
    call_id: str | None = None


class _ProtocolConversationItemAdded(BaseModel):
    event_id: str
    item: ConversationItem | _ProtocolConversationItem
    type: Literal['conversation.item.added']


_ConversationItemAddedEvent = ConversationItemAdded | _ProtocolConversationItemAdded
_CONVERSATION_ITEM_ADDED_ADAPTER: TypeAdapter[_ConversationItemAddedEvent] = TypeAdapter(_ConversationItemAddedEvent)


def map_conversation_event(
    data: dict[str, Any], *, replayed: bool | None = None
) -> ConversationCreated | ConversationItemCreated | None:
    """Map xAI's conversation handshake and item lifecycle events to codec control events."""
    event_type = data.get('type')
    if event_type == _CONVERSATION_CREATED_EVENT:
        event = ConversationCreatedEvent.model_validate(data)
        conversation_id = event.conversation.id
        return ConversationCreated(conversation_id) if conversation_id else None
    if event_type == 'conversation.item.added':
        event = _CONVERSATION_ITEM_ADDED_ADAPTER.validate_python(data)
    elif event_type == 'conversation.item.created':
        event = ConversationItemCreatedEvent.model_validate(data)
    else:
        return None
    item_id = event.item.id or data.get('item_id')
    tool_call_id = (
        event.item.call_id
        if isinstance(event.item, (RealtimeConversationItemFunctionCall, RealtimeConversationItemFunctionCallOutput))
        else data.get('call_id')
    )
    item_id = item_id if isinstance(item_id, str) and item_id else None
    tool_call_id = tool_call_id if isinstance(tool_call_id, str) and tool_call_id else None
    if item_id is not None or tool_call_id is not None:
        return ConversationItemCreated(item_id=item_id, tool_call_id=tool_call_id, replayed=bool(replayed))
    return None


class _InputAudioTranscriptionUpdatedEvent(BaseModel):
    """Typed xAI-only cumulative input-transcription event."""

    type: Literal['conversation.item.input_audio_transcription.updated']
    transcript: str | None = None
    item_id: str | None = None


class _XaiSession(BaseModel):
    """Fields read from xAI's SDK-incompatible session-created payload."""

    model: str | None = None


class _XaiSessionCreatedEvent(BaseModel):
    type: Literal['session.created']
    event_id: str
    session: _XaiSession


__all__ = (
    'XaiRealtimeModel',
    'XaiRealtimeModelSettings',
    'XaiRealtimeConnection',
    'map_event',
)


class XaiRealtimeModelSettings(RealtimeModelSettings, total=False):
    """Settings specific to xAI realtime models.

    Grok Voice always produces audio, so its profile reports
    [`supports_text_output=False`][pydantic_ai.realtime.RealtimeModelProfile.supports_text_output] and
    the inherited `output_modality='text'` is rejected up front rather than quietly ignored.
    """

    xai_voice: str
    """Voice used for audio output, e.g. `eve`, or a custom voice ID."""

    xai_turn_detection: ServerVAD
    """xAI-specific server-VAD configuration, for an exact threshold, prefix padding, or silence duration.

    When present, this fully overrides the cross-provider `turn_detection` setting. xAI echoes
    `create_response: False` back but still responds when the user stops speaking; set
    `turn_detection=False` for push-to-talk instead.
    """


def map_event(data: dict[str, Any]) -> RealtimeCodecEvent | None:
    """Map a raw xAI Grok Voice realtime event to a [`RealtimeCodecEvent`][pydantic_ai.realtime.codec.RealtimeCodecEvent].

    xAI clones the OpenAI Realtime protocol, so most events map identically via the OpenAI codec.
    The first exception is input audio transcription: xAI emits cumulative
    `conversation.item.input_audio_transcription.updated` snapshots (which may retroactively *correct*
    earlier text — `'Hello?'` becomes `'Hello, my name is'`) plus cumulative `.completed` snapshots,
    rather than OpenAI's incremental `.delta`. The partials are surfaced as cumulative
    [`InputTranscript`][pydantic_ai.realtime.codec.InputTranscript]s so a live transcript can render
    the user's words as they are spoken; the session adopts each snapshot wholesale, appending when it
    merely extends and replacing when xAI revises itself. The shared codec still drops interim
    `.completed` snapshots.
    The other exception is xAI's conversation lifecycle events, which are surfaced as codec control
    events so the connection can capture `conversation.id` and the session can suppress resume replay.
    """
    event_type = data.get('type')
    if event_type == 'conversation.item.input_audio_transcription.updated':
        event = _InputAudioTranscriptionUpdatedEvent.model_validate(data)
        return InputTranscript(
            text=event.transcript or '',
            cumulative=True,
            item_id=event.item_id,
        )
    if event_type in ('conversation.created', 'conversation.item.added', 'conversation.item.created'):
        return map_conversation_event(data)
    event = _map_openai_event(data)
    if isinstance(event, ToolCall):
        item_id = ResponseFunctionCallArgumentsDoneEvent.model_validate(data).item_id
        if item_id:
            event = replace(event, item_id=item_id)
    elif isinstance(event, InputTranscript):
        # xAI's final `.completed` is a whole snapshot like its `.updated` partials, so it too must
        # replace the accumulated text. Read as an increment it would be *appended* to the snapshots it
        # supersedes, and a revised turn would end up saying everything twice.
        event = replace(event, cumulative=True)
    return event


class XaiRealtimeConnection(OpenAIRealtimeConnection):
    """A live WebSocket connection to the xAI Grok Voice realtime API.

    Reuses [`OpenAIRealtimeConnection`][pydantic_ai.realtime.openai.OpenAIRealtimeConnection] for the
    shared wire protocol, while mapping xAI's cumulative input transcription and conversation lifecycle
    events and emitting the resumption replay controls captured during reconnect handshakes.
    """

    _provider_name = 'xai'
    _provider_label = 'xAI Grok Voice'
    _supports_tool_result_images = False

    def __init__(
        self,
        ws: ClientConnection,
        *,
        dial: Callable[[], Awaitable[ClientConnection]] | None = None,
        reconnect: ReconnectPolicy | None = None,
        input_transcription_enabled: bool = True,
        interrupts_response_on_speech: bool = False,
        model_name: str | None = None,
        model_name_getter: Callable[[], str | None] | None = None,
        conversation_id: str | None = None,
        replayed_items: list[ConversationItemCreated] | None = None,
        manual_turns: bool = False,
    ) -> None:
        super().__init__(
            ws,
            dial=dial,
            reconnect=reconnect,
            input_transcription_enabled=input_transcription_enabled,
            interrupts_response_on_speech=interrupts_response_on_speech,
            model_name=model_name,
            model_name_getter=model_name_getter,
        )
        self._restores_state_on_reconnect = True
        self._conversation_id = conversation_id
        self._replayed_items = replayed_items if replayed_items is not None else []
        # xAI reports `billable_audio_seconds` as the session's running total, not the response's own
        # share (live: three turns of 0.71s, 0.71s and 0.87s report 1, 2 and 3), so each response is
        # credited the increase since the last report. Kept on the connection, which outlives a resumed
        # session, as does xAI's total. The total restarts only with a new conversation.
        self._billed_audio_seconds = 0
        self._billed_conversation_id = conversation_id
        # With turn detection off, xAI answers speech as soon as it's committed. So that a reply comes only
        # when one is asked for, as on every other provider, the commit is held back until then and sent in
        # place of `response.create`. A request sent with the commit would be answered before the speech
        # joins the conversation (checked live). Nothing below applies with turn detection on.
        self._manual_turns = manual_turns
        self._commit_held = False
        self._audio_commit_listener: Callable[[], None] | None = None
        # Whether the listener heard about the held commit already, which a reconnect sends again.
        self._commit_announced = False
        # Whether xAI heard speech in the audio it has since the last commit. It doesn't answer a commit of
        # silence at all (checked live), so without speech the commit goes out with a `response.create`.
        self._speech_detected = False
        # Audio kept back from xAI, with how many of its frames a held commit covers: audio sent after a held
        # commit, until the commit goes out, and audio sent during a reply, until it ends. Speech reaching
        # xAI during a reply stops the reply without ending it (checked live).
        self._held_audio: list[dict[str, Any]] = []
        self._held_audio_committed = 0
        # Audio sent to xAI since the last commit went out, which a dropped connection takes with it: sent
        # again after a reconnect when a held commit covers it.
        self._sent_audio: list[dict[str, Any]] = []
        # After answering committed audio, xAI silently drops a `response.create` until other input arrives
        # (checked live): no response and no error, so the caller would wait for that reply forever. Text,
        # an image, a tool result or `input_audio_buffer.clear` makes it answer again, and so does speech
        # still in the buffer, which the request commits. So such a request clears the empty buffer first,
        # and is answered again, as on every other provider. Buffered silence isn't told apart from speech,
        # so a request after it still goes out on its own and is dropped.
        self._audio_is_latest_input = False
        self._audio_uncommitted = False

    async def send(self, content: RealtimeInput) -> None:
        if self._manual_turns and not isinstance(
            content, (BinaryAudio, CommitAudio, ClearAudio, CreateResponse, CancelResponse, TruncateOutput)
        ):
            # Before sending, since a text turn or tool result asks for its response as it goes out.
            self._audio_is_latest_input = False
        await super().send(content)

    async def _send_event(self, event: dict[str, Any]) -> None:
        if not self._manual_turns:
            await super()._send_event(event)
            return
        event_type = event['type']
        if event_type == INPUT_AUDIO_BUFFER_APPEND_EVENT:
            if self._commit_held or self._held_audio or self._response_active:
                self._held_audio.append(event)
            else:
                await self._send_audio(event)
            self._audio_is_latest_input = True
        elif event_type == INPUT_AUDIO_BUFFER_COMMIT_EVENT:
            if self._commit_held or self._audio_uncommitted or self._held_audio:
                # A commit while one is held extends the same user turn.
                self._commit_held = True
                self._held_audio_committed = len(self._held_audio)
                self._audio_uncommitted = False
            else:
                # Nothing to commit, which xAI ignores, as it always has.
                await super()._send_event(event)
        elif event_type == INPUT_AUDIO_BUFFER_CLEAR_EVENT:
            # Audio no commit covers is discarded where it is: kept back here, or in xAI's buffer. A clear
            # with nothing in that buffer isn't sent, since right after a commit it would cancel the reply.
            del self._held_audio[self._held_audio_committed :]
            if self._audio_uncommitted:
                await super()._send_event(event)
                self._sent_audio.clear()
                self._audio_is_latest_input = self._audio_uncommitted = self._speech_detected = False
        else:
            await super()._send_event(event)

    async def _send_audio(self, frame: dict[str, Any]) -> None:
        # Marked first, so a clear made while the frame is on its way knows there's audio to discard. Audio
        # kept back lands after anything sent meanwhile, so it's xAI's latest input once it goes out.
        self._audio_uncommitted = self._audio_is_latest_input = True
        self._sent_audio.append(frame)
        await super()._send_event(frame)

    async def _send_held_commit(self, input_indexes: Sequence[int], answers: Sequence[InputId]) -> None:
        """Send the held commit, and the audio it covers, with the request for a response."""
        # Marked first, so audio sent meanwhile is kept back until the reply ends rather than joining the turn.
        self._response_active = True
        if not self._commit_announced:
            self._commit_announced = True
            self._announce_commit()
        # Each frame leaves the held audio only as it goes out, and the commit stays held until it's sent, so
        # a connection that drops midway still has the whole turn to send again after the reconnect.
        while self._held_audio_committed:
            self._held_audio_committed -= 1
            frame = self._held_audio.pop(0)
            self._sent_audio.append(frame)
            await super()._send_event(frame)
        speech_detected, self._speech_detected = self._speech_detected, False
        if not speech_detected:
            await super()._send_event({'type': INPUT_AUDIO_BUFFER_COMMIT_EVENT})
            self._release_commit()
            await super()._create_response(input_indexes, answers)
            return
        # What `_create_response` records for a `response.create`, so the reply the commit starts is taken
        # as the answer to `input_indexes`, and an error echoing the id refuses them.
        # A commit can't carry the request's metadata, so the reply isn't tagged with what it answers.
        self._active_response_id = None
        self._response_request_inputs = tuple(input_indexes)
        self._response_request_answers = tuple(answers)
        event_id = client_event_id('response', input_indexes)
        self._lifecycle.request_sent(event_id, self._response_request_answers, tagged=False)
        await super()._send_event({'type': INPUT_AUDIO_BUFFER_COMMIT_EVENT, 'event_id': event_id})
        self._release_commit()

    def _announce_commit(self) -> None:
        """Tell the listener audio is being committed, before anything goes out that xAI could answer first."""
        if self._audio_commit_listener is not None:
            self._audio_commit_listener()

    def _release_commit(self) -> None:
        self._commit_held = self._commit_announced = False
        self._sent_audio.clear()
        # The commit lands after anything sent while it was held, so the audio is xAI's latest input.
        self._audio_is_latest_input = True

    async def _send_held_audio(self) -> None:
        """Send the audio kept back behind a reply that has ended, until a new commit is held or a reply starts."""
        while self._held_audio and not self._commit_held and not self._response_active:
            await self._send_audio(self._held_audio.pop(0))

    @property
    def _defers_audio_commit(self) -> bool:
        return self._manual_turns

    def _set_audio_commit_listener(self, listener: Callable[[], None]) -> None:
        self._audio_commit_listener = listener

    @property
    def _response_request_dropped(self) -> bool:
        """Whether xAI would drop a `response.create` sent now, with no reply on its way to answer it."""
        return self._audio_is_latest_input and not (self._audio_uncommitted or self._commit_held or self._held_audio)

    async def _create_response(self, input_indexes: Sequence[int], answers: Sequence[InputId]) -> None:
        if not self._manual_turns:
            await super()._create_response(input_indexes, answers)
            return
        if self._commit_held:
            await self._send_held_commit(input_indexes, answers)
            return
        # Audio kept back behind the reply that just ended goes first, so the request can commit it.
        await self._send_held_audio()
        if self._response_request_dropped:
            # The buffer is empty, so the clear discards nothing, and it can't cancel a commit's reply: none
            # is on its way when a request goes out.
            await super()._send_event({'type': INPUT_AUDIO_BUFFER_CLEAR_EVENT})
            self._audio_is_latest_input = False
        # Audio still in the buffer is committed by the request, and answered by its response.
        if self._audio_uncommitted:
            self._announce_commit()
        self._audio_uncommitted = self._speech_detected = False
        self._sent_audio.clear()
        await super()._create_response(input_indexes, answers)

    async def _attempt_reconnect(self) -> bool:
        # The new socket's buffer is empty. A held commit still covers the audio that was in it, so that
        # audio goes out again with the commit; audio no commit covers is lost with the socket, as before.
        if self._commit_held:
            self._held_audio[:0] = self._sent_audio
            self._held_audio_committed += len(self._sent_audio)
        self._sent_audio.clear()
        # Whether a commit's reply is still pending on the resumed conversation isn't known, and a clear
        # would cancel it, so a request goes out on its own, as before.
        self._audio_is_latest_input = self._audio_uncommitted = self._speech_detected = False
        return await super()._attempt_reconnect()

    async def _decode_frame(self, raw: str) -> _DecodedFrame:
        events = await super()._decode_frame(raw)
        if self._held_audio and not self._commit_held and not self._response_active:
            await self._send_held_audio()
        return events

    def _map_response_usage(self, usage: RealtimeResponseUsage | None) -> RequestUsage | None:
        mapped = super()._map_response_usage(usage)
        if mapped is None:
            return None
        assert usage is not None
        inp = usage.input_token_details or None
        out = usage.output_token_details or None
        # xAI bills Grok Voice by audio second, so this provider-owned bucket cannot be reconstructed
        # from the OpenAI-protocol token counts.
        for key, raw in (
            ('input_grok_tokens', (inp.model_extra or {}).get('grok_tokens') if inp is not None else None),
            ('output_grok_tokens', (out.model_extra or {}).get('grok_tokens') if out is not None else None),
            ('billable_audio_seconds', self._billed_audio_seconds_increase(usage)),
        ):
            if isinstance(raw, int) and not isinstance(raw, bool) and raw:
                mapped.details[key] = raw
        # Also reported under the name pricing knows it by. Grok Voice has *no* token prices at all — it
        # bills per audio hour — so without this the token counts price to a confident `Decimal('0')`
        # rather than to nothing: `cost_limit` never trips and no unavailable-cost warning is emitted.
        mapped.audio_seconds = mapped.details.get('billable_audio_seconds', 0)
        return mapped

    def _billed_audio_seconds_increase(self, usage: RealtimeResponseUsage) -> int | None:
        """This response's share of xAI's running `billable_audio_seconds` total.

        The session sums each response's usage, so passing the running total through would count every
        earlier second again on each turn. xAI keeps counting across a resumed connection and starts
        afresh only in a new conversation, so that is the one boundary the baseline resets on; a lower
        total within the same conversation is not new usage, and adds nothing.
        """
        total = (usage.model_extra or {}).get('billable_audio_seconds')
        if not isinstance(total, int) or isinstance(total, bool):
            return None
        if self._conversation_id != self._billed_conversation_id:
            self._billed_conversation_id = self._conversation_id
            self._billed_audio_seconds = 0
        increase = max(total - self._billed_audio_seconds, 0)
        self._billed_audio_seconds = max(total, self._billed_audio_seconds)
        return increase

    def set_message_history(self, message_history: Callable[[], Sequence[ModelMessage]]) -> None:
        """Ignored: xAI restores the conversation itself, so replaying it would say everything twice."""

    @property
    def reconnect_restores_in_flight_state(self) -> bool:
        # xAI resumes the conversation server-side, so the in-flight turn is not the session's to settle
        # (unlike the OpenAI base, which reconnects by replaying finalized history only).
        return True

    @property
    def conversation_id(self) -> str | None:
        """The xAI conversation ID used for native session resumption."""
        return self._conversation_id

    @conversation_id.setter
    def conversation_id(self, conversation_id: str | None) -> None:
        self._conversation_id = conversation_id

    def _map_event(self, data: dict[str, Any]) -> RealtimeCodecEvent | None:
        if self._manual_turns:
            event_type = data.get('type')
            if event_type == 'input_audio_buffer.speech_started':
                self._speech_detected = True
            elif event_type in ('input_audio_buffer.committed', 'response.created'):
                # Speech xAI reports only once a commit or request is on its way belongs to the audio that
                # took with it.
                self._speech_detected = False
        return map_event(data)

    async def _all_events(self) -> AsyncIterator[tuple[RealtimeCodecEvent | LifecycleEvent, bool]]:
        async for event, stale in super()._all_events():
            yield event, stale
            if isinstance(event, RealtimeSessionReconnectEvent):
                replayed_items = self._replayed_items[:]
                self._replayed_items.clear()
                for replayed_item in replayed_items:
                    yield replayed_item, False


@dataclass(init=False)
class XaiRealtimeModel(RealtimeModel):
    """xAI Grok Voice realtime API model.

    Pass `provider='xai'` (the default, which reads `XAI_API_KEY`) or an
    [`XaiProvider`][pydantic_ai.providers.xai.XaiProvider] constructed with `api_key=`. A custom
    `api_host` is not supported, and a provider constructed only with `xai_client=` cannot be used
    because the WebSocket connection needs access to the API key. The realtime WebSocket URL is
    `wss://api.x.ai/v1/realtime`.

    Args:
        model: The model name, e.g. `grok-voice-latest` (which tracks the current model) or a
            pinned version like `grok-voice-think-fast-1.0`. The `model` query parameter is required by
            the server, which otherwise falls back to a default silently.
        provider: The provider to use for authentication and the base URL. Defaults to `'xai'`.
        settings: [Model settings][pydantic_ai.realtime.RealtimeModelSettings] used as defaults for
            realtime sessions. A [`reconnect`][pydantic_ai.realtime.RealtimeModelSettings.reconnect]
            policy enables xAI's native session resumption: prior turns are restored when reconnecting
            within xAI's resumption window (30 minutes of inactivity).
        profile: Optional override for the [realtime model profile][pydantic_ai.realtime.RealtimeModelProfile],
            merged over the provider's — a partial dict, or a callable taking the resolved profile and
            returning the one to use. Mirrors `profile=` on a standard
            [`Model`][pydantic_ai.models.Model], and is the escape hatch when a model name doesn't
            identify the model (e.g. an Azure deployment named something other than its model).
    """

    model: XaiRealtimeModelName
    _: KW_ONLY
    settings: RealtimeModelSettings | None = None
    _provider: XaiProvider = field(init=False, repr=False)
    _api_key: str = field(init=False, repr=False)

    # Written out rather than generated because `profile` has to be an init argument while
    # `RealtimeModel.profile` stays the *resolved* profile, exactly as on a standard `Model` — a
    # dataclass field of that name would shadow the property.
    def __init__(
        self,
        model: XaiRealtimeModelName,
        *,
        provider: Provider[AsyncClient] | str = 'xai',
        settings: RealtimeModelSettings | None = None,
        profile: RealtimeModelProfileSpec | None = None,
    ) -> None:
        super().__init__(settings=settings, profile=profile)
        self.model = model
        from ..providers.xai import XaiProvider

        if isinstance(provider, str):
            provider = infer_provider(provider)
        if not isinstance(provider, XaiProvider):
            # Reading the xAI-specific `api_key`/`api_host` off a foreign provider below would fail with
            # an `AttributeError` naming a field the user never heard of, instead of the real mistake.
            raise UserError(f"`XaiRealtimeModel` requires an `XaiProvider` or `provider='xai'`; got {provider.name!r}.")
        api_key = provider.api_key
        if not api_key:
            raise UserError(
                'The xAI realtime provider needs an API key for the WebSocket connection, but the '
                '`XaiProvider` was built from a pre-configured `xai_client` whose key is not exposed. '
                'Pass `provider=XaiProvider(api_key=...)` (or set `XAI_API_KEY`) instead.'
            )
        if provider.api_host is not None:
            # The realtime WebSocket URL is derived from `base_url` (the canonical xAI host), not the
            # gRPC channel target set by `api_host`. Rather than silently connect to the canonical host
            # with the key while the user expects their custom host, fail loudly.
            raise UserError(
                'The xAI realtime provider does not support a custom `api_host`: the realtime WebSocket '
                'connects to the canonical xAI realtime endpoint, not the gRPC channel target that '
                '`api_host` sets. Remove `api_host` from the `XaiProvider` to use realtime.'
            )
        self._provider = provider
        self._api_key = api_key

    @property
    def model_name(self) -> XaiRealtimeModelName:
        return self.model

    @property
    def system(self) -> str:
        return 'xai'

    def _session_config(
        self,
        instructions: str,
        tools: list[ToolDefinition] | None,
        *,
        model_settings: XaiRealtimeModelSettings | None,
    ) -> dict[str, Any]:
        model_settings = cast('XaiRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        # xAI puts `voice` and `turn_detection` at the session top level, unlike OpenAI's GA surface which
        # nests them under `audio`. `turn_detection` is always set: a dict enables VAD, `None` disables it.
        audio_input: dict[str, Any] = {
            'format': {'type': 'audio/pcm', 'rate': self.profile.get('audio_input_sample_rate', 24000)}
        }
        transcription_model = resolve_transcription_model(
            model_settings.get('input_transcription_model', 'auto'), default=_AUTO_TRANSCRIPTION_MODEL
        )
        if transcription_model is not None:
            audio_input['transcription'] = {'model': transcription_model}
        if 'xai_turn_detection' in model_settings:
            turn_detection = model_settings['xai_turn_detection']
        elif 'turn_detection' in model_settings:
            turn_detection = resolve_base_turn_detection(model_settings['turn_detection'])
        else:
            turn_detection: ServerVAD | None = {'type': 'server_vad'}
        config: dict[str, Any] = {
            'instructions': instructions,
            'turn_detection': turn_detection_config(turn_detection),
            'audio': {
                'input': audio_input,
                'output': {
                    'format': {'type': 'audio/pcm', 'rate': self.profile.get('audio_output_sample_rate', 24000)}
                },
            },
        }
        if voice := model_settings.get('xai_voice'):
            config['voice'] = voice
        advertised_tools, tool_choice = resolve_advertised_tools(tools, model_settings.get('tool_choice'))
        if advertised_tools:
            config['tools'] = [tool_def_to_openai(t) for t in advertised_tools]
        if (max_tokens := model_settings.get('max_tokens')) is not None:
            config['max_output_tokens'] = max_tokens
        if (parallel_tool_calls := model_settings.get('parallel_tool_calls')) is not None:
            config['parallel_tool_calls'] = parallel_tool_calls
        if tool_choice is not None:
            config['tool_choice'] = tool_choice_config(tool_choice)
        if (thinking := model_settings.get('thinking')) is not None and self.profile.get('supports_thinking', False):
            # Grok Voice exposes only enabled-at-high and disabled, so every enabled unified effort
            # maps to its sole enabled value.
            config['reasoning'] = {'effort': 'high' if thinking is not False else 'none'}
        if model_settings.get('reconnect') is not None:
            config['resumption'] = {'enabled': True}
        return config

    @asynccontextmanager
    async def connect(
        self,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncGenerator[XaiRealtimeConnection]:
        # The `model` query parameter is required: without it the server silently falls back to a default.
        url = realtime_websocket_url(self._provider.base_url, model=self.model)
        headers = {'Authorization': f'Bearer {self._api_key}'}
        # Propagate trace context over the handshake (see the OpenAI provider for the rationale).
        inject_trace_context(headers)
        settings = cast('XaiRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        reconnect = settings.get('reconnect')
        handshake_timeout = settings.get('handshake_timeout', 30.0)
        instructions = get_instructions(messages, model_request_parameters) or ''
        session_config = self._session_config(
            instructions=instructions, tools=model_request_parameters.function_tools, model_settings=settings
        )
        transcription_enabled = settings.get('input_transcription_model', 'auto') is not None
        conversation_id: str | None = None
        replayed_items: list[ConversationItemCreated] = []
        connection: XaiRealtimeConnection | None = None

        async def dial_headers() -> dict[str, str]:
            return headers

        def dial_url() -> str:
            resume_id = connection.conversation_id if connection is not None else None
            dial_url = f'{url}&conversation_id={quote(resume_id, safe="")}' if resume_id else url
            return dial_url

        def session_model(created: dict[str, Any]) -> str | None:
            return _XaiSessionCreatedEvent.model_validate(created).session.model

        async def after_session_created(ws: ClientConnection, _: dict[str, Any]) -> None:
            nonlocal conversation_id
            if reconnect is not None:
                conversation = map_conversation_event(
                    await expect_event(ws, _CONVERSATION_CREATED_EVENT, timeout=handshake_timeout)
                )
                if not isinstance(conversation, ConversationCreated):
                    raise RealtimeHandshakeError(
                        '`conversation.created` did not include a `conversation.id`, so the session '
                        'cannot be resumed after a drop'
                    )
                conversation_id = conversation.conversation_id
                if connection is not None:
                    connection.conversation_id = conversation_id

        def on_unexpected_during_update() -> Callable[[dict[str, Any]], None] | None:
            if connection is None:
                return None

            def capture_replayed_item(data: dict[str, Any]) -> None:
                event = map_conversation_event(data, replayed=True)
                if isinstance(event, ConversationItemCreated):
                    replayed_items.append(event)

            return capture_replayed_item

        def build_connection(
            ws: ClientConnection,
            dial: Callable[[], Awaitable[ClientConnection]],
            server_model: str | None,
            model_name_getter: Callable[[], str | None],
        ) -> XaiRealtimeConnection:
            nonlocal connection
            connection = XaiRealtimeConnection(
                ws,
                dial=dial,
                reconnect=reconnect,
                input_transcription_enabled=transcription_enabled,
                interrupts_response_on_speech=config_interrupts_response_on_speech(session_config),
                model_name=server_model,
                model_name_getter=model_name_getter,
                conversation_id=conversation_id,
                replayed_items=replayed_items,
                manual_turns=session_config['turn_detection'] is None,
            )
            return connection

        async with connect_openai_protocol(
            model_name=self.model,
            messages=messages,
            profile=self.profile,
            provider_name=self.system,
            session_config=session_config,
            handshake_timeout=handshake_timeout,
            dial_headers=dial_headers,
            dial_url=dial_url,
            session_model=session_model,
            build_connection=build_connection,
            after_session_created=after_session_created,
            on_unexpected_during_update=on_unexpected_during_update,
        ) as connected:
            yield connected
