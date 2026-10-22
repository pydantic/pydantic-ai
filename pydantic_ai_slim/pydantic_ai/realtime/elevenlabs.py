"""ElevenLabs realtime API provider for Eleven Realtime speech-to-speech sessions.

Connects to `wss://api.elevenlabs.io/v1/convai/realtime` over a WebSocket. Eleven Realtime speaks the OpenAI
Realtime protocol, so [`ElevenLabsRealtimeModel`][pydantic_ai.realtime.elevenlabs.ElevenLabsRealtimeModel] is an
[`OpenAIRealtimeModel`][pydantic_ai.realtime.openai.OpenAIRealtimeModel] pointed at ElevenLabs: it reuses the
OpenAI codec (event mapping, session seeding, tool conversion, reconnect-by-replay, and the WebSocket connection
itself), authenticated with the key and base URL of the shared
[`ElevenLabsProvider`][pydantic_ai.providers.elevenlabs.ElevenLabsProvider], and diverges only where ElevenLabs
does:

- authentication uses the `xi-api-key` header, and the model is chosen with `session.model` in `session.update`
  (the server ignores OpenAI's `?model=` query parameter);
- turn detection is semantic VAD only: the server rejects `server_vad`, so the cross-provider
  [`turn_detection`][pydantic_ai.realtime.RealtimeModelSettings.turn_detection] setting maps onto semantic VAD;
- input transcription is always on, even with `input_transcription_model=None`, and takes no model, only a
  language (see
  [`elevenlabs_language`][pydantic_ai.realtime.elevenlabs.ElevenLabsRealtimeModelSettings.elevenlabs_language]);
- voices are ElevenLabs voices, set with
  [`elevenlabs_voice`][pydantic_ai.realtime.elevenlabs.ElevenLabsRealtimeModelSettings.elevenlabs_voice];
- `response.done` carries no usage, so sessions report no token counts;
- `conversation.item.delete` is not implemented, so `openai_prune_tool_items` is ignored;
- images are dropped from user messages, and seeded history can only carry text;
- text output (`output_modalities: ['text']`) ignores audio input altogether, so `output_modality='text'` is
  unsupported;
- the server ends a session after `max_duration_seconds` (600 by default; see
  [`elevenlabs_max_duration_seconds`][pydantic_ai.realtime.elevenlabs.ElevenLabsRealtimeModelSettings.elevenlabs_max_duration_seconds])
  or after 60 seconds without a client message. With a
  [`reconnect`][pydantic_ai.realtime.RealtimeModelSettings.reconnect] policy, either close is re-dialed and the
  local message history replayed into a new session.

WebRTC is not supported, as ElevenLabs serves no `/realtime/calls` endpoint, but
[`create_client_secret`][pydantic_ai.realtime.openai.OpenAIRealtimeModel.create_client_secret] mints the
ephemeral credential a browser uses to open the WebSocket directly.

For a hosted ElevenLabs agent, use
[`ElevenLabsAgentsRealtimeModel`][pydantic_ai.realtime.elevenlabs_agents.ElevenLabsAgentsRealtimeModel] instead.

Requires the `websockets` package (the `realtime` optional group) and `openai` (the `openai` group, whose
SDK supplies the event types the shared OpenAI codec is built on):

    pip install "pydantic-ai-slim[elevenlabs-realtime]"
"""

from __future__ import annotations as _annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence, Set as AbstractSet
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NoReturn, cast

try:
    import websockets as websockets
    from openai.types.realtime import InputAudioBufferSpeechStoppedEvent
    from websockets.asyncio.client import ClientConnection

    from ..providers.openai import OpenAIProvider
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `websockets` and `openai` packages to use the ElevenLabs realtime model '
        '(`openai` supplies the event types of the OpenAI codec this provider reuses), you can use the '
        '`elevenlabs-realtime` optional group - `pip install "pydantic-ai-slim[elevenlabs-realtime]"`'
    ) from _import_error

from .._http import AsyncHTTPClient
from .._utils import cancel_and_drain
from ..exceptions import UserError
from ..messages import ModelMessage, RealtimeSessionErrorEvent
from ..models import ModelRequestParameters
from ..profiles.elevenlabs import elevenlabs_realtime_model_profile
from ..providers import Provider, infer_provider
from ..providers.elevenlabs import ElevenLabsProvider
from ..tools import ToolDefinition
from ._elevenlabs_audio_tags import AudioTagStripper, strip_audio_tags
from ._lifecycle import (
    AnswersRevised,
    InputId,
    LifecycleEvent,
    ResponseEnded,
    ResponseStarted,
    ToolCallsAbandoned,
)
from ._openai_lifecycle import OpenAILifecycle
from ._openai_protocol import (
    AUTO_TRANSCRIPTION_MODEL,
    CONVERSATION_ITEM_ADDED_EVENT_ADAPTER,
    CONVERSATION_ITEM_CREATE_EVENT,
    ProtocolResponse,
    config_has_manual_turns,
    config_interrupts_response_on_speech,
    loads_obj,
    turn_detection_config,
    with_realtime_query,
)
from .codec import OutputTranscript, RealtimeCodecEvent, ToolResult
from .model import RealtimeModel, WebRTCAnswer
from .openai import (
    OpenAIRealtimeConnection,
    OpenAIRealtimeModel,
    OpenAIRealtimeModelSettings,
    _DecodedFrame,  # pyright: ignore[reportPrivateUsage]
)
from .profiles import DEFAULT_AUDIO_SAMPLE_RATE, RealtimeModelProfile, RealtimeModelProfileSpec
from .settings import RealtimeModelSettings, ReconnectPolicy

if TYPE_CHECKING:
    from .model import RealtimeProviderSession

__all__ = (
    'ElevenLabsRealtimeModel',
    'ElevenLabsRealtimeModelName',
    'ElevenLabsRealtimeModelSettings',
    'ElevenLabsRealtimeConnection',
    'LatestElevenLabsRealtimeModelNames',
)

_OPENAI_BASE_PATH = 'v1/convai'
"""Path under `ElevenLabsProvider.base_url` that the OpenAI-protocol base URL points at.

The realtime WebSocket is that URL plus `realtime`, with the HTTP scheme swapped for the WebSocket one, and the
client-secret endpoint is that URL plus `realtime/client_secrets`, as on OpenAI:
`https://api.elevenlabs.io` → `https://api.elevenlabs.io/v1/convai` → `wss://api.elevenlabs.io/v1/convai/realtime`.
A data-residency `base_url` maps onto its own realtime host the same way."""

_KEEP_ALIVE_FRAME = '{"type":"input_audio_buffer.append","audio":""}'
"""An empty audio append: a client message that adds no audio, which keeps the session open (verified live)."""
_RESPONSE_CREATED_EVENT = 'response.created'
_ITEM_ADDED_EVENT = 'conversation.item.added'

LatestElevenLabsRealtimeModelNames = Literal['eleven_realtime_v1_mini', 'eleven_realtime_v1_max']
"""The latest Eleven Realtime model names."""

ElevenLabsRealtimeModelName = str | LatestElevenLabsRealtimeModelNames
"""Possible Eleven Realtime model names.

`eleven_realtime_v1_mini` is optimized for price and speed; `eleven_realtime_v1_max` reasons before it responds,
for complex instructions and multi-step tool use, and needs a `max_tokens` of at least 1024."""


class ElevenLabsRealtimeModelSettings(OpenAIRealtimeModelSettings, total=False):
    """Settings specific to Eleven Realtime models.

    This inherits every [`OpenAIRealtimeModelSettings`][pydantic_ai.realtime.openai.OpenAIRealtimeModelSettings]
    field, since the session is configured with OpenAI's `session.update` shape. `openai_turn_detection` is sent
    as on OpenAI, except that ElevenLabs only supports semantic VAD: a `ServerVAD` value raises a
    [`UserError`][pydantic_ai.exceptions.UserError].

    The remaining inherited fields are **silently ignored**, because ElevenLabs accepts and then drops them:
    `openai_voice` (ElevenLabs voices are set with `elevenlabs_voice`), `openai_input_noise_reduction`,
    `openai_output_speed`, `openai_truncation`, and `openai_prune_tool_items` and `openai_keep_tool_items` (the
    server doesn't implement `conversation.item.delete`). So are the shared `parallel_tool_calls` and `thinking`
    settings. Input audio is always transcribed: `input_transcription_model` takes `'auto'` (the default) or
    `None`, which still leaves transcripts on, and any other value raises a `UserError`.
    """

    elevenlabs_voice: str
    """Voice used for audio output: an ElevenLabs voice ID (e.g. `'JBFqnCBsd6RMkjVDRZzb'`) or the name of a
    default voice. The server's default is `'brian'`."""

    elevenlabs_language: str
    """Language code of the conversation (e.g. `'en'`, `'es'`, `'pt-br'`), used for transcription, speech, and the
    model's reply language. The server's default is English."""

    elevenlabs_strip_audio_tags: bool
    """Whether to remove expressive audio tags such as `[cheerful]` from the model's transcripts. Defaults to `True`.

    Eleven Realtime writes tags into its reply to direct the delivery, and they are part of the transcript, though
    they aren't spoken. They are removed from the transcript, streamed and in history alike, and the tags a response's
    transcript held are kept, in order and without brackets, in its `provider_details['audio_tags']`. Only lowercase
    words in square brackets count as tags, so citations (`[1]`), Markdown links, and placeholders like `[Your Name]`
    are kept. Set this to `False` to keep the transcript exactly as ElevenLabs sends it, tags included.
    """

    elevenlabs_max_duration_seconds: int
    """Maximum wall-clock duration of a session, in seconds, from 60 to 7200. The server's default is 600.

    The server closes the WebSocket once it is exceeded. With a
    [`reconnect`][pydantic_ai.realtime.RealtimeModelSettings.reconnect] policy, the session carries on over a new
    connection, with the local message history replayed into it."""


class _ElevenLabsLifecycle(OpenAILifecycle):
    """The OpenAI lifecycle, adjusted for two things ElevenLabs does differently.

    OpenAI's VAD commits a turn as soon as it reports `speech_stopped`. ElevenLabs' semantic VAD can instead hear
    the user carry on after a pause, and report `speech_started` again for the same item, so a turn is only over
    once `input_audio_buffer.committed` says so.

    And ElevenLabs answers the requests for the results of parallel tool calls with one reply (see
    `response_created`).
    """

    def __init__(self, *, server_vad: bool) -> None:
        super().__init__(server_vad=server_vad)
        self._tool_results: set[InputId] = set()
        """The inputs that carried our tool outputs."""

    def tool_output_sent(self, call_id: str, input_id: InputId) -> None:
        super().tool_output_sent(call_id, input_id)
        self._tool_results.add(input_id)

    def response_created(
        self,
        response: ProtocolResponse,
        *,
        outstanding: tuple[str, tuple[InputId, ...]] | None,
    ) -> list[LifecycleEvent]:
        """A `response.created`, which also answers our earlier requests ElevenLabs neither started nor refused.

        ElevenLabs makes parallel tool calls as back-to-back responses of one call each, and holds back a
        `response.create` sent while any of them is still without its result until they all have one, answering all
        the requests it held with one reply (checked live). So the request that went out with the first result gets
        neither a response nor an error of its own. A response of ours that starts therefore answers the requests for
        tool results still outstanding whose results are in the conversation.
        """
        events = super().response_created(response, outstanding=outstanding)
        started = events[0] if events else None
        if not isinstance(started, ResponseStarted) or not started.answers:
            # A response of the provider's own, which answers none of ours.
            return events
        coalesced = [
            event_id for event_id, (answers, _) in self._requests.items() if self._answered_with_round(answers)
        ]
        answers = [input_id for event_id in coalesced for input_id in self._requests.pop(event_id)[0]]
        if not (answered := self._settle(answers)):
            return events
        return [replace(started, answers=(*answered, *started.answers)), *events[1:]]

    def answers_held_request(self, answers: Sequence[InputId]) -> tuple[InputId, ...] | None:
        """Which inputs of a request held back behind a response under way the response of ours that just started answers.

        A request for tool results is answered as an outstanding one is (see `response_created`), once the results are
        in the conversation: the inputs are settled then, and the request is not to be sent, as it would ask for a
        second reply. Otherwise this is `None`, and the request still goes out.
        """
        return self._settle(answers) if self._answered_with_round(answers) else None

    def _answered_with_round(self, answers: Sequence[InputId]) -> bool:
        """Whether a request is for tool results that are all in the conversation, which ElevenLabs answers together."""
        return self._tool_results.issuperset(answers) and not self._unacknowledged(answers)

    def speech_stopped(self, data: dict[str, Any]) -> list[LifecycleEvent]:
        # Its commit, when it comes, is still VAD's own, not the acknowledgement of one of ours. (ElevenLabs also
        # reports a stop for an item it never reported starting after a commit of ours, which is no turn at all.)
        if (item_id := InputAudioBufferSpeechStoppedEvent.model_validate(data).item_id) and self._server_vad:
            self._vad_stopped.add(item_id)
        return []


class ElevenLabsRealtimeConnection(OpenAIRealtimeConnection):
    """A live WebSocket connection to the ElevenLabs realtime API.

    Reuses [`OpenAIRealtimeConnection`][pydantic_ai.realtime.openai.OpenAIRealtimeConnection] for the shared
    wire protocol, including reconnecting by replaying the local message history.
    """

    _provider_name = 'elevenlabs'
    _provider_label = 'ElevenLabs Realtime'
    # Images are dropped from conversation items, so there is nowhere for a tool result's images to go.
    _supports_tool_result_images = False
    _keep_alive_after: ClassVar[float | None] = 45.0
    """Seconds without sending anything after which the connection sends a keep-alive frame, or `None` for never.

    ElevenLabs closes a session that hasn't received a client message for 60 seconds (`1002`), and doesn't count
    pings."""

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
        observes_output_audio: bool = True,
        audio_output_sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
        prune_tool_items: bool = False,
        keep_tool_items: AbstractSet[str] = frozenset(),
        manual_turns: bool = False,
        strip_audio_tags: bool = True,
    ) -> None:
        # ElevenLabs transcribes every spoken turn, so its transcripts arrive (and belong in history) even when
        # `input_transcription_model=None` asked for none.
        del input_transcription_enabled
        super().__init__(
            ws,
            dial=dial,
            reconnect=reconnect,
            interrupts_response_on_speech=interrupts_response_on_speech,
            model_name=model_name,
            model_name_getter=model_name_getter,
            observes_output_audio=observes_output_audio,
            audio_output_sample_rate=audio_output_sample_rate,
            prune_tool_items=prune_tool_items,
            keep_tool_items=keep_tool_items,
            manual_turns=manual_turns,
        )
        # Semantic VAD commits the speech it hears on its own, unless turn detection is off.
        self._lifecycle: _ElevenLabsLifecycle = _ElevenLabsLifecycle(server_vad=not manual_turns)
        # See `ElevenLabsRealtimeModelSettings.elevenlabs_strip_audio_tags`.
        self._strip_audio_tags = strip_audio_tags
        # The streamed transcript of each output item under way, with its audio tags removed, and the tags each
        # response's transcripts held (see `_strip_transcript_tags`).
        self._tag_strippers: dict[str | None, AudioTagStripper] = {}
        self._audio_tags: dict[str | None, list[str]] = {}
        # When this connection last sent the server anything, on the event loop's clock (see `_keep_alive`).
        self._last_sent = 0.0
        # Tool calls ElevenLabs gave up on when it started a reply to the user before their results were in (see
        # `_abandon_outstanding_tool_calls`).
        self._abandoned_tool_calls: set[str] = set()
        # Responses with tool calls that a user message joined the conversation after: the next response to start can
        # be the reply to it, which abandons their calls (see `_abandon_outstanding_tool_calls`).
        self._interjected: set[str] = set()

    @property
    def _prunes_tool_calls(self) -> bool:
        # ElevenLabs answers `conversation.item.delete` with a `not_implemented` error: `openai_prune_tool_items`
        # is one of the settings it ignores.
        return False

    @asynccontextmanager
    async def _session_scope(self) -> AsyncGenerator[None]:
        # Keeps the connection open while idle for as long as it lasts (see `_keep_alive`).
        if self._keep_alive_after is None:
            yield
            return
        keep_alive = asyncio.create_task(self._keep_alive(self._keep_alive_after), name='elevenlabs-keep-alive')
        try:
            yield
        finally:
            await cancel_and_drain(keep_alive)

    async def _keep_alive(self, after: float) -> None:
        """Keep an idle session open: ElevenLabs closes one that hasn't heard from the client for 60 seconds.

        Pings don't count, so after `after` seconds without sending anything, this sends an empty
        `input_audio_buffer.append`, which adds no audio. It stops once the socket is closed for good.
        """
        loop = asyncio.get_running_loop()
        # The handshake just went out: the clock starts now.
        self._last_sent = loop.time()
        while True:
            idle = loop.time() - self._last_sent
            if idle >= after:
                try:
                    await self._ws.send(_KEEP_ALIVE_FRAME)
                except self.transport_errors:
                    if self._reconnect is None or self._dial is None:
                        # Closed with no reconnect to come: there is nothing left to keep open.
                        return
                    # A reconnect replaces the socket, and the clock restarts with it.
                self._last_sent = loop.time()
                idle = 0.0
            await asyncio.sleep(after - idle)

    async def _send_event(self, event: dict[str, Any]) -> None:
        await super()._send_event(event)
        self._last_sent = asyncio.get_running_loop().time()

    async def _decode_frame(self, raw: str) -> _DecodedFrame:
        abandoned: list[str] = []
        if _RESPONSE_CREATED_EVENT in raw and loads_obj(raw).get('type') == _RESPONSE_CREATED_EVENT:
            abandoned = self._abandon_outstanding_tool_calls()
        decoded = await super()._decode_frame(raw)
        if self._pending_response:
            self._answer_held_request(decoded)
        if _ITEM_ADDED_EVENT in raw and (data := loads_obj(raw)).get('type') == _ITEM_ADDED_EVENT:
            item = CONVERSATION_ITEM_ADDED_EVENT_ADAPTER.validate_python(data).item
            if item.type == 'message' and item.role == 'user':
                # Typed or spoken: a response that starts from now on can be the reply to it.
                self._interjected.update(self._tool_call_batches)
        if self._strip_audio_tags:
            decoded.codec = self._strip_transcript_tags(decoded.codec)
            decoded.after = self._with_audio_tags(decoded.after)
        if abandoned:
            # Their results get no reply, so the session waits for none: the exchange can end with the new response.
            decoded.before.append(ToolCallsAbandoned(tool_call_ids=tuple(abandoned)))
            decoded.codec.append(
                RealtimeSessionErrorEvent(
                    message=(
                        'ElevenLabs Realtime started a new response before the results of tool calls '
                        f'{", ".join(repr(call_id) for call_id in abandoned)} were sent, and abandoned those calls: '
                        'their results will not reach the model.'
                    ),
                    type='tool_calls_abandoned',
                    recoverable=True,
                )
            )
        return decoded

    def _answer_held_request(self, decoded: _DecodedFrame) -> None:
        """Let a response of ours that the frame starts answer the request held back behind the one before, too.

        Once a response's tool results are all out, the connection asks for a response, unless one is under way, as when
        the next of ElevenLabs' back-to-back parallel calls was. The request is then held back until that response
        is over, and the response of ours that starts meanwhile, for an earlier result, answers it too, if the results
        are in the conversation by then. Sent after it, the request would ask for a second reply.
        """
        started = next((event for event in decoded.after if isinstance(event, ResponseStarted) and event.answers), None)
        if started is None:
            return
        claimed = self._lifecycle.answers_held_request(self._deferred_response_answers)
        if claimed is None:
            return
        self._take_deferred_response_inputs()
        decoded.after.append(AnswersRevised(response_id=started.response_id, claimed=claimed))

    def _abandon_outstanding_tool_calls(self) -> list[str]:
        """Give up on the tool calls of finished responses still waiting for results, as ElevenLabs just did.

        Once the reply to something the user said after a response started does, a `function_call_output` for the
        response's calls never reaches the model, and the `response.create` that follows it goes unanswered, so the
        results already sent for those calls are answered by nothing, and those still to come are not asked to be.

        A response that starts with nothing new from the user is no such reply: it carries on the generation of the
        one before. That is how ElevenLabs makes parallel tool calls, as back-to-back responses of one call each, and
        the results of all of them do reach the model (checked live).
        """
        self._interjected &= self._tool_call_batches.keys()
        abandoned: list[str] = []
        for response_id, batch in list(self._tool_call_batches.items()):
            if batch.done and batch.unanswered and response_id in self._interjected:
                self._interjected.discard(response_id)
                del self._tool_call_batches[response_id]
                self._lifecycle.requests_dropped(batch.inputs)
                abandoned.extend(sorted(batch.unanswered))
        self._abandoned_tool_calls.update(abandoned)
        return abandoned

    async def _send_tool_result(self, content: ToolResult, *, input_index: int) -> None:
        if content.tool_call_id not in self._abandoned_tool_calls:
            await super()._send_tool_result(content, input_index=input_index)
            return
        self._abandoned_tool_calls.discard(content.tool_call_id)
        self._tool_call_responses.pop(content.tool_call_id, None)
        # Still added to the conversation, which keeps it complete, but no response is asked for: none would come.
        self._lifecycle.tool_output_sent(content.tool_call_id, input_index)
        await self._send_event(
            {
                'type': CONVERSATION_ITEM_CREATE_EVENT,
                'item': {'type': 'function_call_output', 'call_id': content.tool_call_id, 'output': content.output},
            }
        )
        self._lifecycle.requests_dropped([input_index])

    def _strip_transcript_tags(self, events: list[RealtimeCodecEvent]) -> list[RealtimeCodecEvent]:
        """Remove expressive audio tags from output transcripts, collecting each response's (see `_with_audio_tags`)."""
        stripped: list[RealtimeCodecEvent] = []
        for event in events:
            if isinstance(event, OutputTranscript):
                key = event.item_id or event.response_id
                if event.is_final:
                    # The whole text, stripped at once, which is what the stripped pieces streamed before it add up to.
                    self._tag_strippers.pop(key, None)
                    text, tags = strip_audio_tags(event.text)
                    self._audio_tags.setdefault(event.response_id, []).extend(tags)
                else:
                    text = self._tag_strippers.setdefault(key, AudioTagStripper()).feed(event.text)
                    if not text:
                        continue
                event = replace(event, text=text)
            stripped.append(event)
        return stripped

    def _with_audio_tags(self, events: list[LifecycleEvent]) -> list[LifecycleEvent]:
        """Record the tags a response's transcripts held on the `ResponseEnded` that files it in history."""
        return [
            replace(event, provider_details={**(event.provider_details or {}), 'audio_tags': tags})
            if isinstance(event, ResponseEnded) and (tags := self._audio_tags.pop(event.response_id, None))
            else event
            for event in events
        ]


class _ElevenLabsOpenAIProvider(OpenAIProvider):
    """The OpenAI SDK pointed at ElevenLabs' OpenAI-compatible API, with an `ElevenLabsProvider`'s key and HTTP client.

    Named `'elevenlabs'` and resolving ElevenLabs' realtime profile, so the behavior
    [`OpenAIRealtimeModel`][pydantic_ai.realtime.openai.OpenAIRealtimeModel] derives from its provider
    (`system`, `profile`, the WebSocket and client-secret URLs) reports ElevenLabs rather than OpenAI.
    """

    def __init__(self, elevenlabs_provider: ElevenLabsProvider) -> None:
        super().__init__(
            base_url=f'{elevenlabs_provider.base_url}/{_OPENAI_BASE_PATH}',
            api_key=elevenlabs_provider.api_key,
            http_client=elevenlabs_provider.client,
        )

    @property
    def name(self) -> str:
        return 'elevenlabs'

    @staticmethod
    def realtime_model_profile(model_name: str) -> RealtimeModelProfile:
        return elevenlabs_realtime_model_profile(model_name)


def _turn_detection_config(settings: ElevenLabsRealtimeModelSettings) -> dict[str, Any] | None:
    """The `turn_detection` payload: semantic VAD, the only kind ElevenLabs takes, or `None` for push-to-talk."""
    if 'openai_turn_detection' in settings:
        turn_detection = settings['openai_turn_detection']
        if turn_detection['type'] == 'server_vad':
            raise UserError(
                'Eleven Realtime only supports semantic VAD: set `openai_turn_detection` to '
                "`{'type': 'semantic_vad'}`, or use the cross-provider `turn_detection` setting."
            )
        vad = turn_detection_config(turn_detection)
        assert vad is not None
        if vad.get('eagerness') == 'auto':
            # ElevenLabs takes `low`, `medium` or `high`; leaving it out keeps the server default, `medium`.
            del vad['eagerness']
        return vad
    base = settings.get('turn_detection', True)
    if base is False:
        return None
    # Spelled out, as the shared OpenAI config does, so the session knows the server cancels a response on speech.
    config: dict[str, Any] = {'type': 'semantic_vad', 'create_response': True, 'interrupt_response': True}
    if base is not True:
        # Semantic VAD's `eagerness` takes the same values as the cross-provider `sensitivity`.
        if (sensitivity := base.get('sensitivity')) is not None:
            config['eagerness'] = sensitivity
        for key in ('prefix_padding_ms', 'silence_duration_ms'):
            if (value := base.get(key)) is not None:
                config[key] = value
    return config


@dataclass(init=False)
class ElevenLabsRealtimeModel(OpenAIRealtimeModel):
    """ElevenLabs realtime API model, for the Eleven Realtime speech-to-speech models.

    Pass `provider='elevenlabs'` (the default, which reads `ELEVENLABS_API_KEY`) or an
    [`ElevenLabsProvider`][pydantic_ai.providers.elevenlabs.ElevenLabsProvider] constructed with `api_key=`
    or a data-residency `base_url=`. The session is configured and driven through the OpenAI Realtime
    protocol over a WebSocket; WebRTC is not supported.

    Args:
        model: The model name, `eleven_realtime_v1_mini` (optimized for price and speed) or
            `eleven_realtime_v1_max` (reasons before it responds).
        provider: The provider to use for authentication and the base URL. Defaults to `'elevenlabs'`.
        settings: [Model settings][pydantic_ai.realtime.elevenlabs.ElevenLabsRealtimeModelSettings] used as
            defaults for realtime sessions. With a [`reconnect`][pydantic_ai.realtime.RealtimeModelSettings.reconnect]
            policy, a dropped connection is re-dialed and the local message history replayed into it.
        profile: Optional override for the [realtime model profile][pydantic_ai.realtime.RealtimeModelProfile],
            merged over the provider's — a partial dict, or a callable taking the resolved profile and
            returning the one to use.
    """

    _connection_type: ClassVar[type[OpenAIRealtimeConnection]] = ElevenLabsRealtimeConnection
    _elevenlabs_provider: ElevenLabsProvider = field(init=False, repr=False)

    def __init__(
        self,
        model: ElevenLabsRealtimeModelName,
        *,
        provider: Provider[AsyncHTTPClient] | str = 'elevenlabs',
        settings: RealtimeModelSettings | None = None,
        profile: RealtimeModelProfileSpec | None = None,
    ) -> None:
        if isinstance(provider, str):
            provider = infer_provider(provider)
        if not isinstance(provider, ElevenLabsProvider):
            # Reading `api_key` off a foreign provider would fail with an `AttributeError` instead of the real mistake.
            raise UserError(
                "`ElevenLabsRealtimeModel` requires an `ElevenLabsProvider` or `provider='elevenlabs'`; "
                f'got {provider.name!r}.'
            )
        self._elevenlabs_provider = provider
        super().__init__(model, provider=_ElevenLabsOpenAIProvider(provider), settings=settings, profile=profile)

    @property
    def base_url(self) -> str:
        return self._elevenlabs_provider.base_url

    def _realtime_url(self, model_settings: OpenAIRealtimeModelSettings | None = None) -> str:
        # The model is chosen by `session.model` in `session.update`: the server ignores a `?model=` query.
        del model_settings
        return with_realtime_query(self._realtime_ws_base())

    async def _auth_headers(self, model_settings: OpenAIRealtimeModelSettings | None = None) -> dict[str, str]:
        # The API key goes in `xi-api-key`; `Authorization: Bearer` only takes a client secret.
        del model_settings
        return {'xi-api-key': self._elevenlabs_provider.api_key}

    def _build_connection(
        self,
        ws: ClientConnection,
        *,
        dial: Callable[[], Awaitable[ClientConnection]],
        settings: OpenAIRealtimeModelSettings,
        session_config: dict[str, Any],
        server_model: str | None,
        model_name_getter: Callable[[], str | None],
    ) -> ElevenLabsRealtimeConnection:
        return ElevenLabsRealtimeConnection(
            ws,
            dial=dial,
            reconnect=settings.get('reconnect'),
            interrupts_response_on_speech=config_interrupts_response_on_speech(session_config),
            model_name=server_model,
            model_name_getter=model_name_getter,
            audio_output_sample_rate=self.audio_output_sample_rate,
            prune_tool_items=settings.get('openai_prune_tool_items', False),
            keep_tool_items=settings.get('openai_keep_tool_items', frozenset()),
            manual_turns=config_has_manual_turns(session_config),
            strip_audio_tags=cast(ElevenLabsRealtimeModelSettings, settings).get('elevenlabs_strip_audio_tags', True),
        )

    async def _webrtc_headers(self) -> dict[str, str]:
        # Only the API key: the OpenAI SDK client's other default headers (`OpenAI-Organization`, say) are OpenAI's.
        return await self._auth_headers()

    def _session_model_name(self, created: dict[str, Any], model_settings: OpenAIRealtimeModelSettings) -> str | None:
        # `session.created` always reports the default model. The requested one is set by the `session.update`
        # that follows, which the server rejects, failing the handshake, if it names an unknown model.
        del created, model_settings
        return self.model

    def _session_config(
        self,
        instructions: str,
        tools: list[ToolDefinition] | None,
        *,
        model_settings: OpenAIRealtimeModelSettings | None,
    ) -> dict[str, Any]:
        settings = cast('ElevenLabsRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        transcription_model = settings.get('input_transcription_model', AUTO_TRANSCRIPTION_MODEL)
        if transcription_model not in (AUTO_TRANSCRIPTION_MODEL, None):
            raise UserError(
                'Eleven Realtime always transcribes input audio and takes no transcription model: leave '
                "`input_transcription_model` at `'auto'`, and set the language with `elevenlabs_language`; "
                f'got {transcription_model!r}.'
            )
        config = super()._session_config(instructions, tools, model_settings=settings)
        config['model'] = self.model
        audio_input: dict[str, Any] = config['audio']['input']
        audio_input['turn_detection'] = _turn_detection_config(settings)
        # Transcription takes no model, only the conversation's language, which also sets the reply language.
        audio_input.pop('transcription', None)
        audio_input.pop('noise_reduction', None)
        if (language := settings.get('elevenlabs_language')) is not None:
            audio_input['transcription'] = {'language': language}
        audio_output: dict[str, Any] = config['audio']['output']
        # Output audio is always 24 kHz, and its `format` takes no `rate`.
        audio_output['format'] = {'type': 'audio/pcm'}
        # `openai_voice` names an OpenAI voice, which ElevenLabs doesn't have; see `ElevenLabsRealtimeModelSettings`.
        audio_output.pop('voice', None)
        audio_output.pop('speed', None)
        if voice := settings.get('elevenlabs_voice'):
            audio_output['voice'] = voice
        for ignored in ('parallel_tool_calls', 'truncation'):
            config.pop(ignored, None)
        if (max_duration := settings.get('elevenlabs_max_duration_seconds')) is not None:
            config['max_duration_seconds'] = max_duration
        return config

    # WebRTC: ElevenLabs serves OpenAI's client-secret endpoint, which `create_client_secret` inherits, but not
    # `/realtime/calls`.

    def connect_webrtc(
        self,
        session: RealtimeProviderSession,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> NoReturn:
        self._raise_unsupported_webrtc('connect_webrtc')

    async def answer_webrtc_offer(
        self,
        sdp_offer: str,
        *,
        instructions: str | None = None,
        tools: Sequence[ToolDefinition] | None = None,
        model_settings: RealtimeModelSettings | None = None,
        message_history: Sequence[ModelMessage] | None = None,
    ) -> WebRTCAnswer:
        self._raise_unsupported_webrtc('answer_webrtc_offer')

    def _check_hang_up(self, session: RealtimeProviderSession) -> None:
        RealtimeModel._check_hang_up(self, session)
