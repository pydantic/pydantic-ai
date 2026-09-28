"""OpenAI GPT-Live realtime support.

GPT-Live is a separate protocol from the [OpenAI Realtime API][pydantic_ai.realtime.openai], not a
model served by it, so none of its event mapping is shared: what it borrows from
`_openai_protocol.py` is the parts that are about the *provider* rather than the protocol (deriving
the WebSocket URL, resolving authentication, and mapping handshake failures). The differences that
shape the adapter:

- **Audio drives the session.** Live has no user-turn event for text: text arrives as *context*
  through `session.commentary.append` (speakable) and `session.thinking.append` (silent), each
  capped at 500 tokens. Both are placed on the session's audio timeline, which only advances while
  audio flows — so a session whose microphone isn't streaming silently defers everything sent to it.
- **There is no turn terminal.** Live has no `response.done` equivalent and no transcript-done
  event; transcripts arrive as timeline fragments. The connection synthesizes
  [`ResponseDone`][pydantic_ai.realtime.codec.ResponseDone] once the model has been quiet for
  `openai_live_turn_silence_ms` and no delegated work is outstanding, and the profile reports
  `synthesizes_turn_boundary=True` so consumers know the boundary is inferred.
- **Work is delegated, not tool-called.** The Live model hands a task either to your application
  (`client` delegation) or to a Responses backend it drives itself (`responses` delegation). Only
  the latter produces typed function calls, so it is what this adapter configures: the agent's tools
  are advertised to the backend, its calls arrive nested inside `response.event`, and their results
  go back as Responses input items. See
  [the Live docs](https://ai.pydantic.dev/realtime/openai-live/) for the split.
- **Usage has two meters.** Live reports its own audio as a cumulative duration in seconds and no
  tokens at all; the backend it delegates to reports ordinary Responses token usage, which is where
  most of a call's token cost is.
"""

from __future__ import annotations as _annotations

import array
import asyncio
import base64
import sys
import time
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, ClassVar, Literal, cast

from pydantic import TypeAdapter, ValidationError
from pydantic_core import to_json
from typing_extensions import TypedDict

# The delegated backend is an ordinary Responses call, so its usage is mapped by the same code that
# maps a direct one — including the cache and reasoning breakdowns genai-prices reads.
from .. import _utils
from .._genai_prices import best_effort_price
from .._instrumentation import get_instructions
from .._run_context import get_current_run_context
from ..exceptions import UserError
from ..messages import (
    BinaryImage,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponsePart,
    RealtimeSessionErrorEvent,
    RetryPromptPart,
    SpeechPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from ..models import Model, ModelRequestParameters, infer_model, parse_model_id
from ..models.openai import _map_usage as map_openai_usage  # pyright: ignore[reportPrivateUsage]
from ..providers import Provider, infer_provider
from ..providers.gateway import normalize_gateway_provider
from ..tools import ToolDefinition
from ..usage import RequestUsage
from ._openai_protocol import (
    expect_event,
    map_connect_errors,
    openai_websocket_auth_headers,
    realtime_websocket_url,
    tool_choice_config,
)
from ._utils import inject_trace_context, resolve_advertised_tools
from .codec import (
    AudioDelta,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    CreateResponse,
    InputTranscript,
    OutputTranscript,
    RealtimeCodecEvent,
    RealtimeConnection,
    RealtimeInput,
    ResponseDone,
    SessionUsage,
    TextContext,
    ToolCall,
    ToolResult,
    TruncateOutput,
)
from .model import RealtimeModel
from .profiles import RealtimeModelProfileSpec
from .settings import RealtimeModelSettings

try:
    import tiktoken
    import websockets
    from openai import AsyncOpenAI
    from openai.types.live import (
        DelegationCreatedEvent,
        ErrorEvent,
        InputTranscriptDeltaEvent,
        OutputAudioDeltaEvent,
        OutputTranscriptDeltaEvent,
        ResponseEvent,
        ServerEvent,
        SessionClosedEvent,
        SessionUsageUpdatedEvent,
    )
    from openai.types.responses import (
        Response,
        ResponseCompletedEvent,
        ResponseErrorEvent,
        ResponseFailedEvent,
        ResponseFunctionToolCall,
        ResponseIncompleteEvent,
        ResponseOutputItemDoneEvent,
        ResponseStreamEvent,
    )
    from websockets.asyncio.client import ClientConnection
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install `openai>=3.12` and `websockets` to use the OpenAI GPT-Live model, '
        'you can use the `openai-realtime` optional group — `pip install "pydantic-ai-slim[openai-realtime]"`'
    ) from _import_error

__all__ = (
    'OpenAILiveModel',
    'OpenAILiveConnection',
    'OpenAILiveModelSettings',
    'OpenAILiveResponsesDelegation',
    'LatestOpenAILiveModelNames',
    'OpenAILiveModelName',
)

LatestOpenAILiveModelNames = Literal['gpt-live-1']
"""Latest OpenAI GPT-Live model names."""

OpenAILiveModelName = str | LatestOpenAILiveModelNames
"""Possible OpenAI GPT-Live model names.

Since OpenAI supports a variety of date-stamped models, we explicitly list the latest models but
allow any name in the type hints.
"""

DEFAULT_TURN_SILENCE_MS = 2_000
"""How long the model must stay quiet before a turn is considered complete.

Matches the `assistant_silence_ms` default of OpenAI's own `TranscriptGrouper`, which solves the same
problem for display transcripts.
"""

AUTO_BACKEND_MODEL = 'gpt-6-sol'
"""What a backend model of `'auto'` resolves to: the Responses model Pydantic AI currently recommends.

Used when nothing names one: not `openai_live_delegation`, not the model name, and not the agent. It
moves with new OpenAI models, so pin a backend explicitly when its behavior needs to stay put.
"""

_LIVE_WEBSOCKET_PATH = 'live/sessions'
_SESSION_STARTED_EVENT = 'session.started'
#: The loudest 16-bit sample an output frame can hold and still count as the idle track (about
#: -54 dBFS). Live's idle track is not always exact zeros: every session opens with a frame or two of
#: dither peaking around 20, and treating that as speech would open a reply nobody is giving.
_VOICE_FLOOR = 64

#: How much quiet audio after the model's voice is still forwarded as a pause, in milliseconds of audio.
#: Live's track never stops, so past this the quiet is treated as the gap between chunks every other
#: provider leaves: otherwise each reply would end with the whole turn-silence wait of silence, and a
#: tool call's quiet wait would arrive as a spoken part with nothing in it.
_MAX_FORWARDED_PAUSE_MS = 500

#: The most tokens Live accepts in one `session.commentary.append` or `session.thinking.append`.
_CONTEXT_TOKEN_LIMIT = 500
#: The tokenizer Live counts that limit in. Checked live: 500 `o200k_base` tokens are accepted, 501 refused.
_CONTEXT_ENCODING = 'o200k_base'

#: The Responses model kinds whose `Model.system` is `'openai'`: the only agent models that can be a backend.
_OPENAI_MODEL_KINDS = frozenset({'openai', 'openai-chat', 'openai-responses'})

#: How Live's top-level errors that end a delegated handoff begin (`Responses handoff incomplete.`, recorded
#: live). The frame names no delegation, so this is what says delegated work ended.
_HANDOFF_FAILURE_PREFIX = 'Responses handoff'

_server_event_adapter: TypeAdapter[ServerEvent] = TypeAdapter(ServerEvent)


class _LiveErrorDetails(TypedDict):
    message: str
    code: str | None


class _LiveErrorFrame(TypedDict):
    type: Literal['error']
    error: _LiveErrorDetails


# OpenAI's guide says to expect `error` frames whose `code` is null, but the SDK's `Error.code` is a
# required `str`, so those fail `ServerEvent` validation. This narrower shape still parses them.
_live_error_adapter: TypeAdapter[_LiveErrorFrame] = TypeAdapter(_LiveErrorFrame)

#: Why a session can end without anyone asking. The others, `close_requested` and `remote_hangup`, are
#: an ordinary end of the call.
_ABNORMAL_CLOSE_REASONS = frozenset({'expired', 'content', 'connection_lost'})

#: The PCM16 sample rates Live accepts. (It also takes 8 kHz G.711, which isn't PCM16.)
_LIVE_PCM_RATES = frozenset({16000, 24000})
_response_stream_event_adapter: TypeAdapter[ResponseStreamEvent] = TypeAdapter(ResponseStreamEvent)
#: Nested events the codec acts on. One of these that doesn't parse is a malformed frame, reported as
#: recoverable; any other nested type that doesn't parse is one this SDK doesn't know yet, and ignored.
_ACTED_ON_DELEGATED_RESPONSE_EVENTS = frozenset(
    {'response.completed', 'response.failed', 'response.incomplete', 'response.output_item.done'}
)


class OpenAILiveResponsesDelegation(TypedDict, total=False):
    """Settings for the Responses backend a Live session delegates work to.

    The Live model runs the conversation; the backend model does the reasoning and calls the agent's
    tools. OpenAI's prompting guide asks for the two prompts to stay separate, and Pydantic AI keeps
    them separate by construction: the agent's instructions become the *backend* prompt, because they
    describe the work, while `live_instructions` describes how to speak.
    """

    model: str
    """The Responses model that handles delegated work.

    When unset, the backend is, first match wins: the model named after a `+` in the Live model name
    (`'gpt-live-1+gpt-5.6-sol'`); the agent's own model, when it is an OpenAI model reached the same way
    as the Live model (directly, or through the same gateway route); then `'auto'`, which resolves to
    [`AUTO_BACKEND_MODEL`][pydantic_ai.realtime.openai_live.AUTO_BACKEND_MODEL]. There is always a
    backend, so nothing raises for want of one.
    """
    instructions: str
    """Extra backend instructions, appended after the agent's own instructions."""
    max_output_tokens: int
    """Maximum output tokens per delegated response."""
    parallel_tool_calls: bool
    """Whether the backend may request several tool calls in one response."""
    reasoning_effort: Literal['none', 'minimal', 'low', 'medium', 'high', 'xhigh']
    """Reasoning effort for the backend model."""
    verbosity: Literal['low', 'medium', 'high']
    """How much detail the backend generates. Does not affect the Live model's spoken delivery."""
    service_tier: Literal['auto', 'default', 'flex', 'priority']
    """Service tier for delegated Responses requests."""


class OpenAILiveModelSettings(RealtimeModelSettings, total=False):
    """Settings for OpenAI GPT-Live sessions.

    See [`RealtimeModelSettings`][pydantic_ai.realtime.RealtimeModelSettings] for the shared settings.
    Live has no turn-detection, truncation, token-limit, or temperature controls: it owns turn-taking
    entirely, and voice, audio format and instructions are immutable once the session has started.
    """

    openai_voice: str
    """The voice used for Live speech, e.g. `marin` (the provider default). Immutable after startup."""

    openai_live_instructions: str
    """Instructions for the Live model's *spoken* behavior: voice, pacing, interruptions, and when to
    delegate.

    The agent's own instructions describe the work and become the backend prompt, so this is where
    conversational style belongs. Defaults to a short prompt that tells the model to delegate
    anything it can't answer from the conversation itself."""

    openai_live_delegation: OpenAILiveResponsesDelegation
    """Configuration for the Responses backend the Live session delegates work to."""

    openai_live_turn_silence_ms: int
    """How long the model must stay quiet, in milliseconds, before the session reports the turn
    complete. Defaults to 2000.

    Live has no end-of-turn frame, so this threshold *is* the turn boundary. Lower it for snappier
    turn-taking at the risk of ending a turn during a dramatic pause; raise it when replies contain
    long silences."""

    openai_live_store: bool
    """Whether OpenAI stores the session so it can later be forked or downloaded. Defaults to `False`."""


DEFAULT_LIVE_INSTRUCTIONS = (
    'You are a voice assistant. Keep replies short and conversational. '
    'When the user asks for something you cannot answer from this conversation alone, delegate the '
    'task and tell them you are looking it up.'
)


def tool_def_to_live(tool: ToolDefinition) -> dict[str, Any]:
    """Convert a [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] to a Live backend function tool."""
    result: dict[str, Any] = {
        'type': 'function',
        'name': tool.name,
        'parameters': tool.parameters_json_schema,
    }
    if tool.description:
        result['description'] = tool.description
    if tool.strict is not None:
        result['strict'] = tool.strict
    return result


_SeedRole = Literal['user', 'assistant']


def _seed_item(role: _SeedRole, text: str) -> dict[str, Any] | None:
    """One Live history item, or `None` when there is no text worth seeding."""
    if not (text := text.strip()):
        return None
    content_type = 'output_text' if role == 'assistant' else 'input_text'
    return {'role': role, 'content': [{'type': content_type, 'text': text}]}


def _seed_request_part(part: ModelRequestPart, *, provider_name: str) -> tuple[_SeedRole, str] | None:
    """The role and text a request part seeds as, or `None` when it carries nothing replayable.

    `SystemPromptPart`s are routed through `instructions` instead, exactly as on the Realtime
    protocol. A tool result becomes user-level text because Live has nowhere to put a function part in
    seeded history, and not a developer note: its content comes from whatever the tool read, which
    must not be replayed with more authority than it had as a tool result.
    """
    if isinstance(part, UserPromptPart):
        return 'user', _prompt_text(part, provider_name=provider_name)
    if isinstance(part, SpeechPart):
        return 'user', part.transcript or ''
    if isinstance(part, ToolReturnPart):
        return 'user', f'Result of `{part.tool_name}`: {part.model_response_str()}'
    if isinstance(part, RetryPromptPart):
        # Without this the `ToolCallPart` before it seeds as a call with no outcome, and the backend
        # reads a round that failed as one that succeeded.
        attempt = f'`{part.tool_name}` failed' if part.tool_name else 'The previous attempt failed'
        return 'user', f'{attempt}: {part.model_response()}'
    return None


def _seed_response_part(part: ModelResponsePart) -> tuple[_SeedRole, str] | None:
    """The role and text a response part seeds as, or `None` when it carries nothing replayable.

    Thinking is bound to the session that produced it: its text is replayable, its signature is not.
    """
    if isinstance(part, TextPart):
        return 'assistant', part.content
    if isinstance(part, SpeechPart):
        return 'assistant', part.transcript or ''
    if isinstance(part, ThinkingPart):
        return 'assistant', part.content
    if isinstance(part, ToolCallPart):
        return 'assistant', f'Called `{part.tool_name}` with {part.args_as_json_str()}.'
    return None


def seed_input_items(messages: Sequence[ModelMessage], *, provider_name: str) -> list[dict[str, Any]]:
    """Map prior history to Live's startup `input` list.

    Live seeds from text only: user and assistant messages with one text part each. Tool
    rounds are rendered as readable text — as Gemini Live does for the same reason — because the
    protocol has no place to put function parts in seeded history. Audio, images, and other media
    cannot be seeded at all, and the profile says so, which is what makes the session reject them
    before we get here.
    """
    items: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            seeded_parts = [_seed_request_part(part, provider_name=provider_name) for part in message.parts]
        else:
            seeded_parts = [_seed_response_part(part) for part in message.parts]
        for seeded in seeded_parts:
            if seeded is not None and (item := _seed_item(*seeded)) is not None:
                items.append(item)
    return items


def _prompt_text(part: UserPromptPart, *, provider_name: str) -> str:
    """Extract the seedable text of a user prompt, refusing media Live cannot carry."""
    if isinstance(part.content, str):
        return part.content
    texts: list[str] = []
    for item in part.content:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, TextContent):
            texts.append(item.content)
        else:
            raise UserError(
                f'{provider_name} GPT-Live sessions can only be seeded with text: '
                f'{type(item).__name__} cannot be replayed into a Live session. '
                'Strip non-text content from `message_history`, or summarize it as text.'
            )
    return '\n'.join(texts)


@dataclass
class _Delegation:
    """A unit of work the Live model handed to the Responses backend."""

    id: str
    pending_tool_calls: set[str] = field(default_factory=set[str])
    response_in_flight: bool = True
    """Whether a backend response is running: the one that opened the delegation, or a continuation.

    Only that response's terminal event says every call it will make has been asked for. With parallel
    tool calls a result can come back before the next call does, so answering every call seen so far
    is not yet a reason to continue.
    """
    continuation_due: bool = False
    """Every call seen so far is answered, but the response that asked for them is still running.

    The continuation waits for its terminal, then goes out from the receive loop.
    """


class OpenAILiveConnection(RealtimeConnection):
    """A live WebSocket connection to the OpenAI GPT-Live API.

    Translates Live's session-timeline events into the shared codec vocabulary, including the two
    boundaries Live itself never sends: the end of the user's turn and the end of the model's reply.

    Live streams output audio as a continuous telephony-style track — ten 100 ms frames a second for
    as long as the session is open, digitally silent when the model isn't speaking. So a frame
    arriving means nothing about whether anyone is talking, and *voice* (a transcript fragment or a
    non-silent audio frame) is what drives the turn clock.
    """

    transport_errors: ClassVar[tuple[type[Exception], ...]] = (websockets.WebSocketException, OSError)

    def __init__(
        self,
        ws: ClientConnection,
        *,
        model_name: str | None = None,
        backend_model: str | None = None,
        turn_silence_ms: int = DEFAULT_TURN_SILENCE_MS,
        audio_rate: int = 24000,
        provider_name: str = 'openai',
        provider_url: str = '',
    ) -> None:
        self._ws = ws
        self._model_name = model_name
        self._backend_model = backend_model
        self._provider_name = provider_name
        self._provider_url = provider_url
        self._turn_silence = turn_silence_ms / 1000
        self._audio_bytes_per_ms = audio_rate * 2 / 1000
        self._recv_task: asyncio.Task[str | bytes] | None = None
        self._closed = False
        self._response_open = False
        self._input_open = False
        self._last_voice = 0.0
        # Quiet audio since the model's voice last sounded in this reply, measured by its length rather
        # than when it arrived, or `None` when there is no reply for a pause to continue.
        self._pause_ms: float | None = None
        # The last transcript fragment in each direction, for the space Live leaves out between segments.
        self._last_fragment: dict[Literal['input', 'output'], str] = {'input': '', 'output': ''}
        # Where on the session timeline the user's transcript last ended.
        self._last_input_end_ms: int | None = None
        self._delegations: dict[str, _Delegation] = {}
        self._call_delegations: dict[str, str] = {}
        # Calls a delegation asked for before its backend gave up. The session still runs them and
        # sends their results; those must go nowhere, not restart a response the backend abandoned.
        self._abandoned_calls: set[str] = set()
        # Delegations whose results all came back before the response that asked for them ended. Their
        # continuations go out from the receive loop, which is where that terminal is seen.
        self._continuations_due: list[_Delegation] = []
        self._reported_seconds = 0.0

    @property
    def model_name(self) -> str | None:
        return self._model_name

    @property
    def input_transcription_enabled(self) -> bool:
        """Live always transcribes both directions; there is no way to turn it off."""
        return True

    @property
    def reconnect_restores_in_flight_state(self) -> bool:
        """A redialed Live session starts empty: its history is re-seeded, not resumed."""
        return False

    # --- sending ----------------------------------------------------------------------------------

    async def send(self, content: RealtimeInput) -> None:
        if isinstance(content, str):
            # A soliciting text turn. Live has no user-message event, so this goes in as speakable
            # context: the model relays or answers it rather than hearing it as the user's own words.
            await _check_context_length(content)
            await self._send_event({'type': 'session.commentary.append', 'delegation_id': None, 'content': content})
            return
        if isinstance(content, TextContext):
            # Context that `thinking` carries without prompting speech itself. The model may still
            # choose to mention it.
            await _check_context_length(content.text)
            await self._send_event({'type': 'session.thinking.append', 'delegation_id': None, 'content': content.text})
            return
        if isinstance(content, ToolResult):
            await self._send_tool_result(content)
            return
        if isinstance(content, CreateResponse):
            # Reaches the connection only right after an image: the profile's
            # `image_input_requires_response` is what lets the session send one, and `create_response()`
            # itself still needs manual turn control. Running the backend on the queued image is the
            # response: Live opens a delegation for it and speaks the result.
            await self._send_event({'type': 'response.create'})
            return
        if isinstance(content, (CommitAudio, ClearAudio, CancelResponse, TruncateOutput)):
            raise UserError(
                'OpenAI GPT-Live drives turn-taking itself: manual turn control, cancellation, and '
                'output truncation are not available.'
            )
        if isinstance(content, BinaryImage):
            # The voice model sees no images; the delegated backend does, as ordinary Responses input.
            await self._send_event(
                {
                    'type': 'response.item.create',
                    'item': {
                        'type': 'message',
                        'role': 'user',
                        'content': [{'type': 'input_image', 'image_url': content.data_uri}],
                    },
                }
            )
            return
        await self._send_event({'type': 'session.input_audio.append', 'audio': _b64(content.data)})

    async def _send_tool_result(self, result: ToolResult) -> None:
        """Return a tool result to the delegated Responses backend and let it continue."""
        follow_up = _tool_result_follow_up(result)
        delegation_id = self._call_delegations.pop(result.tool_call_id, None)
        if result.tool_call_id in self._abandoned_calls:
            # The backend that asked for this call gave up before it was answered. Sending the output
            # would attach it to nothing, and `response.create` would start a turn nobody asked for.
            self._abandoned_calls.discard(result.tool_call_id)
            return
        delegation = self._delegations.get(delegation_id) if delegation_id is not None else None
        if delegation is not None:
            delegation.pending_tool_calls.discard(result.tool_call_id)
        await self._send_event(
            {
                'type': 'response.item.create',
                'item': {'type': 'function_call_output', 'call_id': result.tool_call_id, 'output': result.output},
            }
        )
        if follow_up is not None:
            await self._send_event({'type': 'response.item.create', 'item': follow_up})
        if delegation is None:
            # A call we can't correlate to a delegation has no response we can wait on, so continue now.
            await self._send_event({'type': 'response.create'})
            return
        if delegation.pending_tool_calls:
            # With `parallel_tool_calls` the backend resumes from all of its calls' outputs together.
            return
        if delegation.response_in_flight:
            # The response that asked for these calls may still ask for more: continue at its terminal.
            delegation.continuation_due = True
            return
        await self._continue(delegation)

    async def _send_due_continuations(self) -> None:
        """Continue each delegation whose asking response just ended with every result already in."""
        while self._continuations_due:
            await self._continue(self._continuations_due.pop(0))

    async def _continue(self, delegation: _Delegation) -> None:
        """Resume a delegated response that has every tool result it asked for.

        A delegated response waiting on tool results does not resume on its own. The continuation is a
        response in flight like the first, which keeps the turn clock suspended until it lands.
        """
        delegation.response_in_flight = True
        delegation.continuation_due = False
        await self._send_event({'type': 'response.create'})

    async def _send_event(self, event: dict[str, Any]) -> None:
        await self._ws.send(to_json(event).decode())

    async def aclose(self) -> None:
        """Cancel the read in flight so closing the socket doesn't strand its exception."""
        self._closed = True
        task = self._recv_task
        self._cancel_read()
        if task is not None:
            with suppress(asyncio.CancelledError, websockets.WebSocketException):
                await task

    # --- receiving --------------------------------------------------------------------------------

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        # One read is always in flight: the next one starts before this frame is handled, so nothing
        # arrives while the consumer is busy and no frame is dropped between iterations.
        pending = self._start_read()
        while True:
            # Never cancel the pending `recv()`: a cancelled read can drop the frame it already holds,
            # so the turn clock is a timeout on the wait rather than on the read.
            done, _ = await asyncio.wait({pending}, timeout=self._silence_timeout())
            if self._closed:
                # `aclose()` cancelled the read while we were waiting on it.
                return
            if done:
                finished, pending = pending, self._start_read()
                try:
                    raw = finished.result()
                # A cancelled read is not caught here: `aclose()` sets `_closed` before cancelling,
                # so that path returns above rather than reaching this call.
                except websockets.ConnectionClosedOK:
                    # The read started above can no longer complete, and nothing will await it.
                    self._cancel_read()
                    # A graceful close ends whatever was in flight. Live never says a turn is over,
                    # so without this the last reply would be settled as interrupted even though the
                    # model had finished speaking and the session closed normally.
                    for event in self._settle_open_turns():
                        yield event
                    return
                for event in self._map_frame(raw):
                    yield event
                await self._send_due_continuations()
            # Checked after every frame, not just when the socket goes quiet: the idle audio track
            # keeps frames arriving, so a wait that returns is no evidence that anyone spoke.
            for event in self._expire_quiet_turn():
                yield event

    def _start_read(self) -> asyncio.Task[str | bytes]:
        """Begin the next read, remembering it so it can be cancelled on the way out."""
        self._recv_task = asyncio.create_task(_recv(self._ws))
        return self._recv_task

    def _cancel_read(self) -> None:
        """Cancel the read in flight, if any, so it never completes unobserved."""
        if (task := self._recv_task) is not None:
            self._recv_task = None
            task.cancel()

    def _silence_timeout(self) -> float | None:
        """How long until the current turn could end, or `None` when nothing is pending.

        Delegated work suspends the clock: the model goes quiet while the backend thinks, and ending
        the turn there would finalize a reply that is still coming.
        """
        if self._delegations or not (self._response_open or self._input_open):
            return None
        return max(0.0, self._last_voice + self._turn_silence - _now())

    def _expire_quiet_turn(self) -> list[RealtimeCodecEvent]:
        """Close whichever turn is open once the session has been quiet long enough."""
        if (timeout := self._silence_timeout()) is None or timeout > 0:
            return []
        return self._settle_open_turns()

    def _settle_open_turns(self, *, interrupted: bool = False) -> list[RealtimeCodecEvent]:
        """Finalize whichever turns are open, in either direction."""
        events = self._close_input_turn()
        if self._response_open:
            self._response_open = False
            self._pause_ms = None
            self._last_fragment['output'] = ''
            events.append(ResponseDone(interrupted=interrupted))
        return events

    def _close_input_turn(self) -> list[RealtimeCodecEvent]:
        """Finalize the user's turn, which Live also never marks as done."""
        if not self._input_open:
            return []
        self._input_open = False
        self._last_fragment['input'] = ''
        return [InputTranscript('', is_final=True)]

    def _heard_voice(self) -> None:
        self._last_voice = _now()

    def _open_response(self) -> list[RealtimeCodecEvent]:
        """Note that the model is speaking, closing the user turn it is replying to."""
        self._heard_voice()
        events = self._close_input_turn()
        self._response_open = True
        return events

    def _map_frame(self, raw: str | bytes) -> list[RealtimeCodecEvent]:
        try:
            event = _server_event_adapter.validate_json(raw)
        except ValidationError:
            try:
                error = _live_error_adapter.validate_json(raw)['error']
            except ValidationError:
                # An event type this version of the SDK doesn't know is not a reason to end the session.
                return []
            return self._map_error(error['message'], code=error['code'])
        try:
            return self._map_event(event)
        except ValueError as e:
            # A well-formed event with a payload we can't decode (bad base64 audio, say) costs that
            # frame, not the call: report it as recoverable and keep reading, as the Realtime
            # connection does.
            return [RealtimeSessionErrorEvent(message=f'Failed to parse OpenAI GPT-Live event: {e}', recoverable=True)]

    def _map_event(self, event: ServerEvent) -> list[RealtimeCodecEvent]:
        """Translate one Live server event, ignoring the ones the session has no vocabulary for.

        Dispatch is on the parsed SDK types rather than the `type` string so each branch narrows to
        the payload it reads. The events not handled here are the SIP transport notices, the sideband
        audio reflections, and the acknowledgements of our own commands, none of which change what a
        session has said or heard.
        """
        if isinstance(event, OutputAudioDeltaEvent):
            return self._map_output_audio(_b64decode(event.delta))
        if isinstance(event, OutputTranscriptDeltaEvent):
            return [*self._open_response(), OutputTranscript(self._fragment('output', event.delta))]
        if isinstance(event, InputTranscriptDeltaEvent):
            continues_closed_turn = (
                self._response_open
                and not self._input_open
                and self._last_input_end_ms is not None
                and event.start_ms <= self._last_input_end_ms
            )
            self._last_input_end_ms = event.end_ms
            if continues_closed_turn and not any(char.isalnum() for char in event.delta):
                # The tail of the user turn the reply already closed, picking up on the timeline where
                # that turn's transcript ended: Live transcribes the user's closing punctuation after the
                # model has started answering over it. That turn is already recorded, so there is
                # nothing to add it to, and opening a new one for a lone `'.'` would record it as a
                # request of its own.
                return []
            self._input_open = True
            self._heard_voice()
            return [InputTranscript(self._fragment('input', event.delta))]
        if isinstance(event, DelegationCreatedEvent):
            return self._map_delegation(event)
        if isinstance(event, ResponseEvent):
            return self._map_response_event(cast('dict[str, Any]', event.event), delegation_id=event.delegation_id)
        if isinstance(event, SessionUsageUpdatedEvent):
            context_window = event.context_window
            return self._map_usage(
                event.usage.seconds, context_window_used=context_window.usage_ratio if context_window else None
            )
        if isinstance(event, SessionClosedEvent):
            return self._map_session_closed(event)
        if isinstance(event, ErrorEvent):
            return self._map_error(event.error.message, code=event.error.code)
        return []

    def _fragment(self, direction: Literal['input', 'output'], delta: str) -> str:
        """One transcript fragment, with the space Live leaves out where a new segment starts.

        Fragments normally carry their own leading space, but the first one of a new segment doesn't,
        so a sentence that ends one segment runs into the next (`'up.'` then `"It's"`). Only a sentence
        end followed by a capital is taken as that boundary, which a word can't be split across.
        """
        previous, self._last_fragment[direction] = self._last_fragment[direction], delta
        if previous[-1:] in ('.', '!', '?') and delta[:1].isupper():
            return f' {delta}'
        return delta

    def _map_error(self, message: str, *, code: str | None) -> list[RealtimeCodecEvent]:
        """Report a top-level `error` frame, and settle whatever delegated work it ended.

        Live reports some backend failures only this way, with no nested terminal for the delegation:
        a backend that runs out of output tokens mid-handoff arrives as `Responses handoff incomplete.`,
        and a backend model that can't be used as one saying the model does not exist. The frame
        names no delegation, so a handoff failure is taken to have ended every backend response still in
        flight: leaving one open would suspend the turn clock for the rest of the session. Any other
        error is only reported, since nothing ties it to delegated work that may still complete.
        """
        if self._backend_model is not None and f'`{self._backend_model}`' in message:
            # Every delegation would fail the same way, so the session can't do the work it was set up
            # for: end it with a reason rather than let each request fail on its own.
            return [
                RealtimeSessionErrorEvent(
                    message=(
                        f'The OpenAI GPT-Live session cannot delegate to its backend model '
                        f'{self._backend_model!r}: {message} Name a Responses model this account can use '
                        "in `openai_live_delegation={'model': ...}`."
                    ),
                    code='live_backend_model_unavailable',
                    recoverable=False,
                )
            ]
        in_flight = [delegation for delegation in self._delegations.values() if delegation.response_in_flight]
        if not in_flight or not message.startswith(_HANDOFF_FAILURE_PREFIX):
            return [RealtimeSessionErrorEvent(message=message, code=code)]
        events: list[RealtimeCodecEvent] = []
        for delegation in in_flight:
            events.extend(self._response_ended_without_usage())
            self._settle_delegation(delegation, gave_up=True)
        events.append(
            RealtimeSessionErrorEvent(
                message=f'The delegated OpenAI Responses backend did not finish ({code}: {message}).',
                code='live_delegation_failed',
            )
        )
        return events

    def _map_session_closed(self, event: SessionClosedEvent) -> list[RealtimeCodecEvent]:
        """Record the final usage, and say so when the session ended without anyone asking.

        The WebSocket close that follows is clean either way, so without this a reply cut off by the
        safety filter or the duration limit would be settled as though the model had finished it.
        """
        events = self._map_usage(event.usage.seconds)
        if event.reason not in _ABNORMAL_CLOSE_REASONS:
            return events
        events.extend(self._settle_open_turns(interrupted=True))
        events.append(
            RealtimeSessionErrorEvent(
                message=f'The OpenAI GPT-Live session ended: {event.reason}.',
                code=f'live_session_{event.reason}',
                recoverable=False,
            )
        )
        return events

    def _map_output_audio(self, pcm: bytes) -> list[RealtimeCodecEvent]:
        """Forward model audio, ignoring the idle track between replies.

        Only a frame louder than the idle floor is evidence that the model is speaking. Quieter frames
        shortly after it are still forwarded, which keeps a pause mid-sentence from arriving as a gap in
        playback — but not while the user is talking, because an assistant frame marks the end of the
        user's turn, and the idle track between their words would cut one utterance into many. A longer
        quiet stretch is not forwarded: it is most likely the end of the reply, or a wait on delegated
        work, and forwarding it would end every reply with seconds of silence.
        """
        if _is_voiced(pcm):
            self._pause_ms = 0.0
            return [*self._open_response(), AudioDelta(data=pcm)]
        if self._pause_ms is None:
            return []
        # A chunk that crosses the allowance is cut at it, on a sample boundary, so the pause forwarded
        # doesn't depend on how Live happened to chunk the track.
        allowed = max(0, int((_MAX_FORWARDED_PAUSE_MS - self._pause_ms) * self._audio_bytes_per_ms) // 2 * 2)
        self._pause_ms += len(pcm) / self._audio_bytes_per_ms
        if not self._response_open or self._input_open or not allowed:
            return []
        return [AudioDelta(data=pcm[:allowed])]

    def _map_delegation(self, event: DelegationCreatedEvent) -> list[RealtimeCodecEvent]:
        delegation = event.delegation
        if delegation.target == 'client':
            # This adapter configures Responses delegation; say so rather than stalling silently
            # while the model waits for an answer that will never come.
            return [
                RealtimeSessionErrorEvent(
                    message=(
                        'The Live session delegated work to the client, but Pydantic AI configures '
                        'Responses delegation. The request will go unanswered.'
                    ),
                    code='live_client_delegation',
                )
            ]
        self._delegations[delegation.id] = _Delegation(id=delegation.id)
        return self._open_response()

    def _map_response_event(self, nested: dict[str, Any], *, delegation_id: str | None) -> list[RealtimeCodecEvent]:
        """Map one nested Responses streaming event from the delegated backend."""
        try:
            event = _response_stream_event_adapter.validate_python(nested)
        except ValidationError:
            if nested.get('type') in _ACTED_ON_DELEGATED_RESPONSE_EVENTS:
                raise  # a malformed frame, which `_map_frame` reports as recoverable
            return []  # a nested event type this version of the SDK doesn't know
        if isinstance(event, ResponseErrorEvent):
            # Recoverable: the Live session carries on, and a failed response still sends its terminal.
            return [
                RealtimeSessionErrorEvent(
                    message=f'The delegated OpenAI Responses backend reported an error: {event.message}',
                    code=event.code,
                )
            ]
        delegation = self._delegations.get(delegation_id) if delegation_id is not None else None
        if isinstance(event, (ResponseCompletedEvent, ResponseFailedEvent, ResponseIncompleteEvent)):
            events: list[RealtimeCodecEvent] = self._map_backend_usage(event.response)
            if delegation is not None:
                if not events:
                    events = self._response_ended_without_usage()
                self._settle_delegation(delegation, gave_up=not isinstance(event, ResponseCompletedEvent))
            if not isinstance(event, ResponseCompletedEvent):
                events.append(_delegation_stopped(event))
            return events
        if not isinstance(event, ResponseOutputItemDoneEvent) or not isinstance(event.item, ResponseFunctionToolCall):
            return []
        call = event.item
        if delegation is not None:
            delegation.pending_tool_calls.add(call.call_id)
            self._call_delegations[call.call_id] = delegation.id
        events = self._open_response()
        # The call ends the spoken part before it: a quiet stretch after it is a wait, not a pause.
        self._pause_ms = None
        return [
            *events,
            ToolCall(
                call.call_id,
                tool_name=call.name,
                args=call.arguments or '{}',
                # The backend response that asked for the call reports its tokens on its terminal event,
                # after the call. Without this the call's `ModelResponse` is finalized empty and those
                # tokens land on the spoken reply that follows. A call we can't correlate to a
                # delegation has no terminal we will see, so nothing follows it.
                response_usage_follows=delegation is not None,
            ),
        ]

    @staticmethod
    def _response_ended_without_usage() -> list[RealtimeCodecEvent]:
        """The empty usage report a backend response that ended without one still makes.

        Every backend response is reported exactly once, usage or not: it is a request the backend made
        (see `responses_are_requests` on the profile), and the calls it asked for are reported as
        `response_usage_follows`, so their `ModelResponse` stays open until it arrives.
        """
        return [SessionUsage(RequestUsage())]

    def _settle_delegation(self, delegation: _Delegation, *, gave_up: bool) -> None:
        """Account for one finished backend response, closing the delegation once it owes nothing.

        A delegation suspends the turn clock, so leaving a dead one in the map would keep the session
        from ever reporting another turn boundary: a backend that fails or stops short has to close
        it just as a completed one does.
        """
        delegation.response_in_flight = False
        if gave_up:
            # Nothing further is coming for this delegation — no continuation, and no answer to any
            # call it had asked for — so it must not hold the clock open waiting for one.
            self._abandoned_calls.update(delegation.pending_tool_calls)
            delegation.pending_tool_calls.clear()
            delegation.continuation_due = False
        if delegation.pending_tool_calls:
            return  # the last result to come back continues it
        if delegation.continuation_due:
            self._continuations_due.append(delegation)
            return
        self._delegations.pop(delegation.id, None)
        self._heard_voice()

    def _map_backend_usage(self, response: Response) -> list[RealtimeCodecEvent]:
        """Accumulate the delegated backend's token usage, priced as the backend.

        Live meters its own audio by the second and reports no tokens for it, but the Responses
        backend it delegates to is billed per token like any other model — and that is where most of
        a call's token cost is. The nested lifecycle snapshot carries the same `usage` object a direct
        Responses call would, so it is mapped by the same code, keeping cache and reasoning
        breakdowns (and genai-prices' view of them) identical either way.

        The price is resolved here, against the backend's own model, because this is the only place
        that model name is known. The `ModelResponse` these tokens land on is the Live model's spoken
        turn and carries Live's name, so anything that prices it from `model_name` downstream would
        charge one model's tokens at another's rate. A cost that is already set is never recalculated
        there, so resolving it now is also what stops that from happening.
        """
        mapped = map_openai_usage(response, self._provider_name, self._provider_url, response.model)
        if not mapped.has_values():
            return []
        if (
            price := best_effort_price(
                mapped,
                model_name=response.model,
                provider_api_url=self._provider_url,
                provider_name=self._provider_name,
            )
        ) is not None:
            mapped.cost = price.total_price
        # Response-scoped, so the backend's request is what a per-request input-token limit is
        # measured against: it is the only thing in a Live session that spends input tokens.
        # The response carries Live's name, so it records the backend model as well: without it the
        # tokens on `usage` could not be re-priced later, or even attributed to the model that spent them.
        # Its id goes next to it rather than into `provider_response_id`, which is the id of the response
        # the Live model spoke, and Live assigns none.
        return [
            SessionUsage(
                mapped, provider_details={'delegated_model': response.model, 'delegated_response_id': response.id}
            )
        ]

    def _map_usage(
        self, cumulative_seconds: float, *, context_window_used: float | None = None
    ) -> list[RealtimeCodecEvent]:
        """Emit the audio seconds billed since the last report, priced as the Live model.

        Live reports a running total and says explicitly not to sum successive values, while
        `RunUsage` accumulates what it is given — so the connection does the subtraction. The context
        window fraction is a snapshot, so it is passed on as is, even when no more seconds were billed.

        The price is resolved here because this usage belongs to no `ModelResponse`, and so is never
        priced at a response boundary: without it, the session's cost would leave out the call itself.
        """
        increment = cumulative_seconds - self._reported_seconds
        if increment <= 0:
            if context_window_used is None:
                return []
            return [SessionUsage(RequestUsage(), response_scoped=False, context_window_used=context_window_used)]
        self._reported_seconds = cumulative_seconds
        usage = RequestUsage(audio_seconds=increment)
        if (
            price := best_effort_price(
                usage,
                model_name=self._model_name,
                provider_api_url=self._provider_url,
                provider_name=self._provider_name,
            )
        ) is not None:
            usage.cost = price.total_price
        # Live bills the session, not a response: this belongs to the run, not to any one `ModelResponse`.
        return [SessionUsage(usage, response_scoped=False, context_window_used=context_window_used)]


def _now() -> float:
    """The turn clock. Monotonic rather than the loop's own, so it needs no running loop."""
    return time.monotonic()


async def _recv(ws: ClientConnection) -> str | bytes:
    return await ws.recv()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _tool_result_follow_up(result: ToolResult) -> dict[str, Any] | None:
    """The backend input message carrying a `ToolReturn`'s text `content`, sent after its output.

    Validated before anything is sent, so a result that can't be carried in full fails with nothing on
    the wire rather than reaching the backend without the material that explains it.
    """
    if not result.content:
        return None
    texts: list[str] = []
    for item in result.content:
        if not isinstance(item, str):
            raise UserError(
                'OpenAI GPT-Live does not support media in tool results yet, so the `content` of a '
                '`ToolReturn` can only be text. Put what the model needs in text or in the return value.'
            )
        texts.append(item)
    return {
        'type': 'message',
        'role': 'user',
        'content': [{'type': 'input_text', 'text': text} for text in texts],
    }


async def _check_context_length(text: str) -> None:
    """Refuse context text over Live's cap before it is sent, rather than after Live rejects it.

    A rejected append arrives as a recoverable error well after `send()` has returned and the text has
    been recorded as sent, so it is checked here, in the tokenizer Live counts with. A token spans at
    least one byte, so text that short can't be over and skips loading it.
    """
    if len(text.encode()) <= _CONTEXT_TOKEN_LIMIT:
        return
    # Loading an encoding can download it on first use, so it stays off the event loop. tiktoken caches
    # the loaded encoding under a lock, so concurrent first sends load it once.
    encoding = await _utils.run_in_executor(tiktoken.get_encoding, _CONTEXT_ENCODING)
    # Special-token text such as `<|endoftext|>` is counted as the one token it is, as Live counts it
    # (checked live), rather than refused as tiktoken does by default.
    if (tokens := len(encoding.encode(text, allowed_special='all'))) > _CONTEXT_TOKEN_LIMIT:
        raise UserError(
            f'OpenAI GPT-Live accepts at most {_CONTEXT_TOKEN_LIMIT} tokens of text per `send()`, and this is '
            f'{tokens}. Send it in shorter pieces, or give longer material to the agent as a tool result.'
        )


def _delegation_stopped(event: ResponseFailedEvent | ResponseIncompleteEvent) -> RealtimeSessionErrorEvent:
    """Report a delegated backend that failed or stopped short.

    The turn itself still ends — Live keeps talking and the silence clock closes it — so without this
    the caller would see an ordinary turn boundary and no sign that the delegated work never finished.
    Recoverable, because the session is fine: only this one piece of delegated work was lost.
    """
    response = event.response
    if response.error is not None:
        reason = f'{response.error.code}: {response.error.message}'
    elif response.incomplete_details is not None:
        reason = f'incomplete: {response.incomplete_details.reason}'
    else:
        reason = event.type
    return RealtimeSessionErrorEvent(
        message=f'The delegated OpenAI Responses backend did not finish ({reason}).',
        code='live_delegation_failed' if isinstance(event, ResponseFailedEvent) else 'live_delegation_incomplete',
    )


def _is_voiced(pcm: bytes) -> bool:
    """Whether a little-endian PCM16 frame is louder than the idle track."""
    samples = array.array('h', pcm[: len(pcm) - len(pcm) % 2])
    if sys.byteorder == 'big':  # pragma: no cover
        samples.byteswap()
    return any(abs(sample) > _VOICE_FLOOR for sample in samples)


def _resolve_openai_model(model_id: str) -> Model | None:
    """Resolve an agent's model name as its run would, when it names an OpenAI model at all.

    Anything else can't be a backend, so it isn't resolved: that would build another provider's client
    (and need its package and key) only to find it doesn't qualify. A name that can't be resolved (its
    key isn't set, say) isn't one this session can use either.
    """
    provider_name, _ = parse_model_id(model_id)
    if provider_name is None or normalize_gateway_provider(provider_name) not in _OPENAI_MODEL_KINDS:
        return None
    try:
        return infer_model(model_id)
    except UserError:
        return None


def _b64decode(data: str) -> bytes:
    return base64.b64decode(data)


@dataclass(init=False)
class OpenAILiveModel(RealtimeModel):
    """OpenAI GPT-Live model.

    GPT-Live runs the spoken conversation and delegates the thinking to a Responses backend that
    Pydantic AI configures with the agent's instructions and tools, so tools, dependencies,
    validation, and message history stay on the Pydantic AI side. Reach it through
    [`Agent.realtime`][pydantic_ai.agent.Agent.realtime] as `'openai:gpt-live-1'`, or construct it
    directly for model-level configuration.

    Live differs from the [Realtime API][pydantic_ai.realtime.openai.OpenAIRealtimeModel] in ways that
    change what a session can do — no text input, an inferred turn boundary, no manual turn control or
    interruption, and duration-based usage. See
    [the Live docs](https://ai.pydantic.dev/realtime/openai-live/).

    Args:
        model: The model name, e.g. `gpt-live-1`.
        provider: The provider to use for authentication and the base URL. Defaults to `'openai'`.
        settings: [Model settings][pydantic_ai.realtime.RealtimeModelSettings] used as defaults for
            realtime sessions.
        profile: Optional override for the [realtime model profile][pydantic_ai.realtime.RealtimeModelProfile].
    """

    model: OpenAILiveModelName
    _: KW_ONLY
    settings: RealtimeModelSettings | None = None
    _provider: Provider[AsyncOpenAI] = field(init=False, repr=False)

    def __init__(
        self,
        model: OpenAILiveModelName,
        *,
        provider: Provider[AsyncOpenAI] | str = 'openai',
        settings: RealtimeModelSettings | None = None,
        profile: RealtimeModelProfileSpec | None = None,
    ) -> None:
        super().__init__(settings=settings, profile=profile)
        # `'gpt-live-1+gpt-5.6-sol'` names the voice model and the backend it delegates to.
        live_model, _, backend_model = model.partition('+')
        self.model = live_model
        self._backend_model = backend_model or None
        if isinstance(provider, str):
            provider = cast('Provider[AsyncOpenAI]', infer_provider(provider))
        if provider.name == 'azure':
            raise UserError(
                'Azure OpenAI does not serve GPT-Live. Use `AzureRealtimeModel` (or the `azure:` prefix) '
                'for Azure OpenAI realtime models.'
            )
        self._provider = provider

    @property
    def client(self) -> AsyncOpenAI:
        """The underlying [`AsyncOpenAI`](https://github.com/openai/openai-python) client from the provider."""
        return self._provider.client

    def _agent_model_name(self) -> str | None:
        """The agent's own model, as the backend to delegate to, when it is reached the same way as this one.

        The backend runs the agent's instructions and calls its tools, so it is the agent doing its work,
        and an agent built on an OpenAI model has already said which model that should be. It has to be
        an OpenAI model at this model's base URL, so both go to OpenAI directly or both through the same
        gateway route; anything else can't serve as this session's backend.

        Read from the current run context: a session opened with
        [`Agent.realtime`][pydantic_ai.agent.Agent.realtime] is an agent run, so its run context is
        current while connecting, as it is around a standard run's model request. A connection
        opened any other way has no agent to consult. An agent whose model is still a name (with
        `defer_model_check=True`) is resolved the way its own run would resolve it.
        """
        run_context = get_current_run_context()
        agent_model = run_context.agent.model if run_context is not None and run_context.agent is not None else None
        if isinstance(agent_model, str):
            agent_model = _resolve_openai_model(agent_model)
        if isinstance(agent_model, Model) and agent_model.system == 'openai' and agent_model.base_url == self.base_url:
            return agent_model.model_name
        return None

    def _resolve_backend_model(self, settings: OpenAILiveModelSettings) -> str:
        """The Responses model this session delegates to, from the first source that names one."""
        delegation_settings = settings.get('openai_live_delegation', OpenAILiveResponsesDelegation())
        backend_model = delegation_settings.get('model') or self._backend_model or self._agent_model_name() or 'auto'
        return AUTO_BACKEND_MODEL if backend_model == 'auto' else backend_model

    @property
    def model_name(self) -> OpenAILiveModelName:
        return self.model

    @property
    def system(self) -> str:
        return self._provider.name

    def _session_config(
        self,
        *,
        instructions: str,
        tools: list[ToolDefinition] | None,
        messages: Sequence[ModelMessage],
        settings: OpenAILiveModelSettings,
    ) -> dict[str, Any]:
        delegation_settings = settings.get('openai_live_delegation', OpenAILiveResponsesDelegation())
        backend_instructions = '\n\n'.join(
            text for text in (instructions, delegation_settings.get('instructions')) if text
        )
        responses: dict[str, Any] = {'model': self._resolve_backend_model(settings)}
        if backend_instructions:
            responses['instructions'] = backend_instructions
        advertised_tools, tool_choice = resolve_advertised_tools(tools, settings.get('tool_choice'))
        if tool_choice == 'required' or (isinstance(tool_choice, tuple) and tool_choice[0] == 'required'):
            # The backend applies the choice to every response of a delegation, including the one meant
            # to answer after the tools have run, so a forced call never lets a delegation finish: it
            # looped on tool calls (seven in ten seconds, live) until a limit ended the session.
            raise UserError(
                f'OpenAI GPT-Live cannot force a tool call, so `tool_choice={settings.get("tool_choice")!r}` cannot '
                'be set: the backend would apply it to every response, including the one after the tool '
                "results, and never answer. Use `'auto'`, `'none'`, or `ToolOrOutput` to limit the tools instead."
            )
        if advertised_tools:
            responses['tools'] = [tool_def_to_live(tool) for tool in advertised_tools]
        if tool_choice is not None:
            # The backend takes the same forms as the Realtime API. A restriction is applied by trimming
            # the advertised tools above, leaving its mode to send.
            responses['tool_choice'] = tool_choice_config(tool_choice)
        for setting, key in (
            ('max_output_tokens', 'max_output_tokens'),
            ('parallel_tool_calls', 'parallel_tool_calls'),
            ('service_tier', 'service_tier'),
        ):
            if (value := delegation_settings.get(setting)) is not None:
                responses[key] = value
        if (effort := delegation_settings.get('reasoning_effort')) is not None:
            responses['reasoning'] = {'effort': effort}
        if (verbosity := delegation_settings.get('verbosity')) is not None:
            responses['text'] = {'verbosity': verbosity}

        config: dict[str, Any] = {
            'model': self.model,
            'instructions': settings.get('openai_live_instructions', DEFAULT_LIVE_INSTRUCTIONS),
            'audio': {'format': {'type': 'audio/pcm', 'rate': self._audio_rate()}},
            'delegation': {'type': 'responses', 'responses': responses},
        }
        if voice := settings.get('openai_voice'):
            config['audio']['output'] = {'voice': voice}
        if settings.get('openai_live_store'):
            config['store'] = True
        if seed := seed_input_items(messages, provider_name=self.system):
            config['input'] = seed
        return config

    def _audio_rate(self) -> int:
        """The PCM16 sample rate to run the session at, from the profile's audio rates.

        Live takes one audio format for both directions, at 16 or 24 kHz, while the profile has a rate
        for each; set both through `profile=` to choose 16 kHz. Checked before connecting, since a
        mismatch would have the session resample one direction to a rate Live isn't using.
        """
        input_rate = self.profile.get('audio_input_sample_rate', 24000)
        output_rate = self.profile.get('audio_output_sample_rate', 24000)
        if input_rate != output_rate or input_rate not in _LIVE_PCM_RATES:
            raise UserError(
                'OpenAI GPT-Live uses one PCM16 audio format for both directions, at 16000 or 24000 Hz, so '
                '`audio_input_sample_rate` and `audio_output_sample_rate` must be equal and one of those; '
                f'got {input_rate} and {output_rate}.'
            )
        return input_rate

    def _reject_unsupported(self, settings: OpenAILiveModelSettings) -> None:
        """Refuse settings Live cannot honor, rather than silently ignoring a stated requirement."""
        for setting, feature in (
            ('turn_detection', 'turn detection: Live owns turn-taking and exposes no VAD configuration'),
            ('max_tokens', 'a token limit on the spoken conversation'),
            ('input_transcription_model', 'choosing a transcription model'),
        ):
            if settings.get(setting) is not None:
                raise UserError(f'OpenAI GPT-Live does not support {feature}, so `{setting}` cannot be set.')

    def _live_url(self) -> str:
        return realtime_websocket_url(self._provider.base_url, path=_LIVE_WEBSOCKET_PATH)

    @asynccontextmanager
    async def connect(
        self,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncGenerator[OpenAILiveConnection]:
        settings = cast('OpenAILiveModelSettings', self._merge_model_settings(model_settings) or {})
        self._reject_unsupported(settings)
        handshake_timeout = settings.get('handshake_timeout', 30.0)
        instructions = get_instructions(messages, model_request_parameters) or ''
        session_config = self._session_config(
            instructions=instructions,
            tools=model_request_parameters.function_tools,
            messages=messages,
            settings=settings,
        )

        cm: AbstractAsyncContextManager[ClientConnection] | None = None
        connection: OpenAILiveConnection | None = None
        try:
            with map_connect_errors(self.model):
                # The raw WebSocket bypasses the provider's `httpx` client, so the handshake carries
                # freshly resolved authentication and the current trace context itself.
                headers = await openai_websocket_auth_headers(self.client)
                inject_trace_context(headers)
                opening = websockets.connect(self._live_url(), additional_headers=headers)
                ws = await opening.__aenter__()
                cm = opening
                await ws.send(to_json({'type': 'session.start', 'session': session_config}).decode())
                started = await expect_event(ws, _SESSION_STARTED_EVENT, timeout=handshake_timeout)
            connection = OpenAILiveConnection(
                ws,
                model_name=started.get('session', {}).get('model'),
                backend_model=session_config['delegation']['responses']['model'],
                audio_rate=session_config['audio']['format']['rate'],
                turn_silence_ms=settings.get('openai_live_turn_silence_ms', DEFAULT_TURN_SILENCE_MS),
                provider_name=self.system,
                provider_url=self._provider.base_url,
            )
            yield connection
        finally:
            if connection is not None:
                await connection.aclose()
            if cm is not None:  # pragma: no branch
                await cm.__aexit__(None, None, None)
