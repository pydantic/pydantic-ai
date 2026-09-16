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

import asyncio
import base64
import time
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, ClassVar, Literal, cast

from pydantic import TypeAdapter, ValidationError
from pydantic_core import to_json
from typing_extensions import TypedDict

from .._instrumentation import get_instructions
from ..exceptions import UserError
from ..messages import (
    ModelMessage,
    ModelRequest,
    RealtimeSessionErrorEvent,
    SpeechPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from ..models import ModelRequestParameters

# The delegated backend is an ordinary Responses call, so its usage is mapped by the same code that
# maps a direct one — including the cache and reasoning breakdowns genai-prices reads.
from ..models.openai import _map_usage as map_openai_usage  # pyright: ignore[reportPrivateUsage]
from ..providers import Provider, infer_provider
from ..tools import ToolDefinition
from ..usage import RequestUsage
from ._openai_protocol import (
    RealtimeHandshakeError,
    loads_obj,
    map_connect_errors,
    openai_websocket_auth_headers,
    realtime_websocket_url,
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
    import websockets
    from openai import AsyncOpenAI
    from openai.types.live import ServerEvent
    from openai.types.responses import Response
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

DEFAULT_BACKEND_MODEL = 'gpt-5.6-sol'
"""The Responses model the Live session delegates to when `openai_live_delegation` doesn't name one."""

_LIVE_WEBSOCKET_PATH = 'live/sessions'
_SESSION_STARTED_EVENT = 'session.started'

_server_event_adapter: TypeAdapter[ServerEvent] = TypeAdapter(ServerEvent)
_responses_adapter: TypeAdapter[Response] = TypeAdapter(Response)


class OpenAILiveResponsesDelegation(TypedDict, total=False):
    """Settings for the Responses backend a Live session delegates work to.

    The Live model runs the conversation; the backend model does the reasoning and calls the agent's
    tools. OpenAI's prompting guide asks for the two prompts to stay separate, and Pydantic AI keeps
    them separate by construction: the agent's instructions become the *backend* prompt, because they
    describe the work, while `live_instructions` describes how to speak.
    """

    model: str
    """The Responses model that handles delegated work. Defaults to `gpt-5.6-sol`."""
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


_SeedRole = Literal['developer', 'user', 'assistant']


def _seed_item(role: _SeedRole, text: str) -> dict[str, Any] | None:
    """One Live history item, or `None` when there is no text worth seeding."""
    if not (text := text.strip()):
        return None
    content_type = 'output_text' if role == 'assistant' else 'input_text'
    return {'role': role, 'content': [{'type': content_type, 'text': text}]}


def _seed_request_part(part: Any, *, provider_name: str) -> tuple[_SeedRole, str] | None:
    """The role and text a request part seeds as, or `None` when it carries nothing replayable.

    `SystemPromptPart`s are routed through `instructions` instead, exactly as on the Realtime
    protocol, and a tool result becomes a developer note because Live has nowhere to put a function
    part in seeded history.
    """
    if isinstance(part, UserPromptPart):
        return 'user', _prompt_text(part, provider_name=provider_name)
    if isinstance(part, SpeechPart):
        return 'user', part.transcript or ''
    if isinstance(part, ToolReturnPart):
        return 'developer', f'Result of `{part.tool_name}`: {part.model_response_str()}'
    return None


def _seed_response_part(part: Any) -> tuple[_SeedRole, str] | None:
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

    Live seeds from text only: developer, user, and assistant messages with one text part each. Tool
    rounds are rendered as readable text — as Gemini Live does for the same reason — because the
    protocol has no place to put function parts in seeded history. Audio, images, and other media
    cannot be seeded at all, and the profile says so, which is what makes the session reject them
    before we get here.
    """
    items: list[dict[str, Any]] = []
    for message in messages:
        is_request = isinstance(message, ModelRequest)
        for part in message.parts:
            seeded = _seed_request_part(part, provider_name=provider_name) if is_request else _seed_response_part(part)
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
    pending_tool_calls: set[str] = field(default_factory=set)


class OpenAILiveConnection(RealtimeConnection):
    """A live WebSocket connection to the OpenAI GPT-Live API.

    Translates Live's session-timeline events into the shared codec vocabulary, including the two
    boundaries Live itself never sends: the end of the user's turn and the end of the model's reply.

    Live streams output audio as a continuous telephony-style track — ten 100 ms frames a second for
    as long as the session is open, digitally silent when the model isn't speaking. So a frame
    arriving means nothing about whether anyone is talking, and *voice* (a transcript fragment or a
    non-silent audio frame) is what drives the turn clock.
    """

    transport_errors: ClassVar[tuple[type[Exception], ...]] = (websockets.WebSocketException,)

    def __init__(
        self,
        ws: ClientConnection,
        *,
        model_name: str | None = None,
        turn_silence_ms: int = DEFAULT_TURN_SILENCE_MS,
        provider_name: str = 'openai',
        provider_url: str = '',
    ) -> None:
        self._ws = ws
        self._model_name = model_name
        self._provider_name = provider_name
        self._provider_url = provider_url
        self._turn_silence = turn_silence_ms / 1000
        self._recv_task: asyncio.Task[str | bytes] | None = None
        self._response_open = False
        self._input_open = False
        self._last_voice = 0.0
        self._delegations: dict[str, _Delegation] = {}
        self._call_delegations: dict[str, str] = {}
        self._reported_seconds = 0

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
            await self._send_event({'type': 'session.commentary.append', 'delegation_id': None, 'content': content})
            return
        if isinstance(content, TextContext):
            # Context that must not itself prompt speech — exactly what `thinking` is for.
            await self._send_event({'type': 'session.thinking.append', 'delegation_id': None, 'content': content.text})
            return
        if isinstance(content, ToolResult):
            await self._send_tool_result(content)
            return
        if isinstance(content, (CommitAudio, ClearAudio, CreateResponse, CancelResponse, TruncateOutput)):
            raise UserError(
                'OpenAI GPT-Live drives turn-taking itself: manual turn control, cancellation, and '
                'output truncation are not available.'
            )
        if content.is_image:
            raise UserError('OpenAI GPT-Live does not accept image input.')
        await self._send_event({'type': 'session.input_audio.append', 'audio': _b64(content.data)})

    async def _send_tool_result(self, result: ToolResult) -> None:
        """Return a tool result to the delegated Responses backend and let it continue."""
        delegation_id = self._call_delegations.pop(result.tool_call_id, None)
        if delegation_id is not None and (delegation := self._delegations.get(delegation_id)) is not None:
            delegation.pending_tool_calls.discard(result.tool_call_id)
        await self._send_event(
            {
                'type': 'response.item.create',
                'item': {'type': 'function_call_output', 'call_id': result.tool_call_id, 'output': result.output},
            }
        )
        # A delegated response waiting on tool results does not resume on its own.
        await self._send_event({'type': 'response.create'})

    async def _send_event(self, event: dict[str, Any]) -> None:
        await self._ws.send(to_json(event).decode())

    async def aclose(self) -> None:
        """Cancel the read in flight so closing the socket doesn't strand its exception."""
        if (task := self._recv_task) is not None:
            self._recv_task = None
            task.cancel()
            with suppress(asyncio.CancelledError, websockets.WebSocketException):
                await task

    # --- receiving --------------------------------------------------------------------------------

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        # One read is always in flight: the next one starts before this frame is handled, so nothing
        # arrives while the consumer is busy and no frame is dropped between iterations.
        self._recv_task = asyncio.create_task(_recv(self._ws))
        while True:
            # Never cancel the pending `recv()`: a cancelled read can drop the frame it already holds,
            # so the turn clock is a timeout on the wait rather than on the read.
            done, _ = await asyncio.wait({self._recv_task}, timeout=self._silence_timeout())
            if (task := self._recv_task) is None:
                # `aclose()` cancelled the read while we were waiting on it.
                return
            if done:
                self._recv_task = asyncio.create_task(_recv(self._ws))
                try:
                    raw = task.result()
                # A cancelled read is not caught here: `aclose()` clears `_recv_task` before
                # cancelling, so that path returns above rather than reaching this call.
                except websockets.ConnectionClosedOK:
                    # A graceful close ends whatever was in flight. Live never says a turn is over,
                    # so without this the last reply would be settled as interrupted even though the
                    # model had finished speaking and the session closed normally.
                    for event in self._settle_open_turns():
                        yield event
                    return
                for event in self._map_frame(raw):
                    yield event
            # Checked after every frame, not just when the socket goes quiet: the idle audio track
            # keeps frames arriving, so a wait that returns is no evidence that anyone spoke.
            for event in self._expire_quiet_turn():
                yield event

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

    def _settle_open_turns(self) -> list[RealtimeCodecEvent]:
        """Finalize whichever turns are open, in either direction."""
        events = self._close_input_turn()
        if self._response_open:
            self._response_open = False
            events.append(ResponseDone())
        return events

    def _close_input_turn(self) -> list[RealtimeCodecEvent]:
        """Finalize the user's turn, which Live also never marks as done."""
        if not self._input_open:
            return []
        self._input_open = False
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
            # An event type this version of the SDK doesn't know is not a reason to end the session.
            return []
        return self._map_event(event)

    def _map_event(self, event: ServerEvent) -> list[RealtimeCodecEvent]:
        kind = event.type
        if kind == 'session.output_audio.delta':
            return self._map_output_audio(_b64decode(event.delta))
        if kind == 'session.output_transcript.delta':
            return [*self._open_response(), OutputTranscript(event.delta)]
        if kind == 'session.input_transcript.delta':
            self._input_open = True
            self._heard_voice()
            return [InputTranscript(event.delta)]
        if kind == 'session.delegation.created':
            return self._map_delegation(event)
        if kind == 'response.event':
            return self._map_response_event(event.event, delegation_id=event.delegation_id)
        if kind in ('session.usage.updated', 'session.closed'):
            return self._map_usage(event.usage.seconds)
        if kind == 'error':
            return [RealtimeSessionErrorEvent(message=event.error.message, code=event.error.code)]
        return []

    def _map_output_audio(self, pcm: bytes) -> list[RealtimeCodecEvent]:
        """Forward model audio, ignoring the idle track between replies.

        `strip` tests for *digital* silence rather than measuring loudness, so this drops only frames
        the model filled with zeros. Frames inside a reply are forwarded whether or not they are
        silent, which keeps a pause mid-sentence from arriving as a gap in playback.
        """
        if pcm.strip(b'\x00'):
            return [*self._open_response(), AudioDelta(data=pcm)]
        if self._response_open:
            return [AudioDelta(data=pcm)]
        return []

    def _map_delegation(self, event: Any) -> list[RealtimeCodecEvent]:
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
        nested_type = nested.get('type')
        delegation = self._delegations.get(delegation_id) if delegation_id is not None else None
        if nested_type == 'response.completed':
            # The backend finished. If it asked for tools, the call is still outstanding and another
            # response follows once we answer, so the delegation stays open until that one lands.
            if delegation is not None and not delegation.pending_tool_calls:
                del self._delegations[delegation.id]
                self._heard_voice()
            return self._map_backend_usage(nested.get('response'))
        if nested_type != 'response.output_item.done':
            return []
        item = nested.get('item')
        if not isinstance(item, dict) or item.get('type') != 'function_call':
            return []
        call_id = cast('str', item['call_id'])
        if delegation is not None:
            delegation.pending_tool_calls.add(call_id)
            self._call_delegations[call_id] = delegation.id
        return [
            *self._open_response(),
            ToolCall(
                call_id,
                tool_name=cast('str', item['name']),
                args=cast('str', item.get('arguments') or '{}'),
                # Live reports no per-response token usage, so nothing follows the call.
                response_usage_follows=False,
            ),
        ]

    def _map_backend_usage(self, response: Any) -> list[RealtimeCodecEvent]:
        """Accumulate the delegated backend's token usage.

        Live meters its own audio by the second and reports no tokens for it, but the Responses
        backend it delegates to is billed per token like any other model — and that is where most of
        a call's token cost is. The nested lifecycle snapshot carries the same `usage` object a direct
        Responses call would, so it is mapped by the same code, keeping cache and reasoning
        breakdowns (and genai-prices' view of them) identical either way.
        """
        if not isinstance(response, dict):  # pragma: no cover
            return []
        try:
            parsed = _responses_adapter.validate_python(response)
        except ValidationError:  # pragma: no cover
            return []
        mapped = map_openai_usage(parsed, self._provider_name, self._provider_url, parsed.model)
        if not mapped.has_values():
            return []  # pragma: no cover
        return [SessionUsage(mapped)]

    def _map_usage(self, cumulative_seconds: float) -> list[RealtimeCodecEvent]:
        """Emit the *increment* since the last report.

        Live reports a running total and says explicitly not to sum successive values, while
        `RunUsage` accumulates what it is given — so the connection does the subtraction. Seconds are
        rounded because `RequestUsage.details` holds integers; diffing the rounded totals keeps the
        running figure accurate rather than compounding a per-event rounding error.
        """
        total = round(cumulative_seconds)
        increment, self._reported_seconds = total - self._reported_seconds, max(total, self._reported_seconds)
        if increment <= 0:
            return []
        return [
            SessionUsage(
                RequestUsage(details={'billable_audio_seconds': increment}),
                # Live bills the session, not a response: this belongs to the run, not to any one
                # `ModelResponse`.
                response_scoped=False,
            )
        ]


def _now() -> float:
    """The turn clock. Monotonic rather than the loop's own, so it needs no running loop."""
    return time.monotonic()


async def _recv(ws: ClientConnection) -> str | bytes:
    return await ws.recv()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


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
        self.model = model
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
        responses: dict[str, Any] = {'model': delegation_settings.get('model', DEFAULT_BACKEND_MODEL)}
        if backend_instructions:
            responses['instructions'] = backend_instructions
        advertised_tools, tool_choice = resolve_advertised_tools(tools, settings.get('tool_choice'))
        if advertised_tools:
            responses['tools'] = [tool_def_to_live(tool) for tool in advertised_tools]
        if tool_choice is not None and not isinstance(tool_choice, tuple):
            # An allow-list is applied by trimming the advertised tools above; Live takes only the
            # declarative modes on the wire.
            responses['tool_choice'] = tool_choice
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
            'audio': {'format': {'type': 'audio/pcm', 'rate': self.profile.get('audio_input_sample_rate', 24000)}},
            'delegation': {'type': 'responses', 'responses': responses},
        }
        if voice := settings.get('openai_voice'):
            config['audio']['output'] = {'voice': voice}
        if settings.get('openai_live_store'):
            config['store'] = True
        if seed := seed_input_items(messages, provider_name=self.system):
            config['input'] = seed
        return config

    def _reject_unsupported(self, settings: OpenAILiveModelSettings) -> None:
        """Refuse settings Live cannot honor, rather than silently ignoring a stated requirement."""
        for setting, feature in (
            ('turn_detection', 'turn detection: Live owns turn-taking and exposes no VAD configuration'),
            ('max_tokens', 'a token limit on the spoken conversation'),
            ('input_transcription_model', 'choosing a transcription model'),
        ):
            if setting in settings:
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
                started = await _expect_started(ws, timeout=handshake_timeout)
            connection = OpenAILiveConnection(
                ws,
                model_name=started.get('session', {}).get('model'),
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


async def _expect_started(ws: ClientConnection, *, timeout: float) -> dict[str, Any]:
    """Read frames until the session starts, mapping a rejected configuration to a handshake error."""
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=max(0.0, deadline - asyncio.get_running_loop().time()))
        except asyncio.TimeoutError:
            raise RealtimeHandshakeError(f'timed out waiting for a {_SESSION_STARTED_EVENT!r} event') from None
        if not isinstance(raw, str):  # pragma: no cover
            raise RealtimeHandshakeError(f'expected a text frame, got {type(raw).__name__}')
        data = loads_obj(raw)
        if data.get('type') == _SESSION_STARTED_EVENT:
            return data
        if data.get('type') == 'error':
            raise RealtimeHandshakeError(data.get('error'))
