"""Abstractions for realtime multimodal models.

Providers like OpenAI Realtime, Gemini Live, and Amazon Nova Sonic offer bidirectional
streaming APIs over a persistent connection (WebSocket or HTTP/2) rather than the
request-response pattern of the standard [`Model`][pydantic_ai.models.Model] interface.

This module defines the provider-facing ABCs ([`RealtimeModel`][pydantic_ai.realtime.RealtimeModel],
[`RealtimeConnection`][pydantic_ai.realtime.RealtimeConnection]) and the event/input types
exchanged over a realtime session.
"""

from __future__ import annotations as _annotations

import asyncio
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Literal

from typing_extensions import TypeAliasType

from ..messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelResponsePart,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    UserPromptPart,
)
from ..native_tools import AbstractNativeTool
from ..settings import ModelSettings
from ..tools import ToolDefinition
from ..usage import RequestUsage

AudioRetention = TypeAliasType('AudioRetention', Literal['transcript_only', 'input', 'output', 'both'])
"""How much audio a [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] retains in its history.

- `'transcript_only'` (default): keep only transcripts; drop all audio bytes.
- `'input'`: also retain the user's spoken audio.
- `'output'`: also retain the model's spoken audio.
- `'both'`: retain both sides' audio.

Retained audio is stored on the [`SpeechPart`][pydantic_ai.messages.SpeechPart]'s
`audio` as raw PCM [`BinaryContent`][pydantic_ai.messages.BinaryContent]. Alignment between retained
audio and its transcript is approximate.
"""


# Input content types (fed into the connection via `send`).


@dataclass
class AudioInput:
    """A chunk of audio data to stream to the model."""

    data: bytes
    """Raw PCM audio bytes. The expected sample rate is provider-specific."""


@dataclass
class ImageInput:
    """An image frame to send to the model (e.g. for Gemini Live video).

    Not every provider accepts image input; sending this to one that doesn't raises
    `NotImplementedError`.
    """

    data: bytes
    """Raw image bytes."""
    mime_type: str = 'image/jpeg'
    """The image media type."""


@dataclass
class TextInput:
    """A text message to send to the model as a complete turn.

    Useful for text-only conversations or for injecting text context alongside audio.
    """

    text: str
    """The message text."""


@dataclass
class ToolResult:
    """The result of a tool call, to send back to the model."""

    tool_call_id: str
    """Identifier of the `ToolCall` this result answers."""
    output: str
    """The tool's output, rendered as a string."""


@dataclass
class CommitAudio:
    """Commit the buffered input audio as a user turn (manual turn-taking / push-to-talk).

    Only needed when automatic voice activity detection is disabled; with server-side VAD the
    provider commits audio and triggers a response automatically.
    """


@dataclass
class ClearAudio:
    """Discard any buffered, uncommitted input audio."""


@dataclass
class CreateResponse:
    """Ask the model to generate a response now (manual turn-taking, after `CommitAudio`)."""


@dataclass
class CancelResponse:
    """Cancel the model's in-progress response (maps to the provider's response-cancel)."""


@dataclass
class TruncateOutput:
    """Truncate the model's current audio output at `audio_end_ms`.

    After a barge-in the user only heard part of the model's audio. Truncating tells the provider how
    much was actually played, so its stored transcript matches and the conversation context stays
    consistent. The provider resolves which output item to truncate from its own state.
    """

    audio_end_ms: int
    """Milliseconds of the current output audio that were actually played before the interruption."""


RealtimeInput = TypeAliasType(
    'RealtimeInput',
    'AudioInput | ImageInput | TextInput | ToolResult | CommitAudio | ClearAudio | CreateResponse | CancelResponse | TruncateOutput',
)
"""Union of content types accepted by [`RealtimeConnection.send`][pydantic_ai.realtime.RealtimeConnection.send]."""


# Connection-level events (yielded by `RealtimeConnection.__aiter__`).


@dataclass
class AudioDelta:
    """A chunk of audio output from the model."""

    data: bytes
    """Raw PCM audio bytes. The sample rate is provider-specific."""


@dataclass
class Transcript:
    """A transcription of the model's audio output (partial or final)."""

    text: str
    """Transcript text. A partial event carries the incremental delta; a final event the full turn."""
    is_final: bool = False
    """Whether this is the final transcript for the turn."""


@dataclass
class InputTranscript:
    """A transcription of the user's audio input (partial or final)."""

    text: str
    """Transcript text."""
    is_final: bool = False
    """Whether this is the final transcript for the user's turn."""


@dataclass
class ToolCall:
    """The model is requesting a tool call."""

    tool_call_id: str
    """Provider-assigned identifier for this call."""
    tool_name: str
    """Name of the tool to invoke."""
    args: str
    """Raw JSON-encoded arguments. May be an empty string if the model sent no arguments."""


@dataclass
class TurnComplete:
    """The model finished (or was interrupted during) its turn."""

    interrupted: bool = False
    """Whether the turn ended because it was cancelled (e.g. the user barged in)."""

    event_kind: Literal['turn_complete'] = 'turn_complete'
    """Event type identifier, used as a discriminator."""


@dataclass
class SpeechStarted:
    """The provider detected that the user started speaking.

    Useful for barge-in: stop playing any buffered model audio when this arrives, since the model's
    in-progress turn is being interrupted.
    """

    event_kind: Literal['speech_started'] = 'speech_started'
    """Event type identifier, used as a discriminator."""


@dataclass
class SpeechStopped:
    """The provider detected that the user stopped speaking.

    Useful as a 'processing' indicator: the user's turn has ended and the model is about to respond.
    """

    event_kind: Literal['speech_stopped'] = 'speech_stopped'
    """Event type identifier, used as a discriminator."""


@dataclass
class SessionUsage:
    """Token usage reported by the provider for a completed model response."""

    usage: RequestUsage
    """Normalized token usage for the response, ready to accumulate into a `RunUsage`."""

    event_kind: Literal['session_usage'] = 'session_usage'
    """Event type identifier, used as a discriminator."""


@dataclass
class RateLimit:
    """A single provider rate-limit entry."""

    name: str
    """Which limit this is, e.g. `requests` or `tokens`."""
    limit: int | None = None
    """The maximum allowed value, if reported."""
    remaining: int | None = None
    """The remaining value before the limit is reached, if reported."""
    reset_seconds: float | None = None
    """Seconds until the limit resets, if reported."""


@dataclass
class RateLimits:
    """An updated rate-limit snapshot, typically emitted at the start of each response."""

    limits: list[RateLimit]
    """The reported rate limits."""

    event_kind: Literal['rate_limits'] = 'rate_limits'
    """Event type identifier, used as a discriminator."""


@dataclass
class Reconnected:
    """The connection dropped and was automatically re-established.

    Session configuration (instructions, tools, voice, ...) is restored, but server-side conversation
    state (the audio buffer and prior turns) is not — treat it as the start of a fresh turn.
    """

    event_kind: Literal['reconnected'] = 'reconnected'
    """Event type identifier, used as a discriminator."""


@dataclass
class SessionError:
    """A provider-reported error occurred in the session."""

    message: str
    """Human-readable error message."""
    type: str | None = None
    """Provider error category, e.g. `invalid_request_error` or `server_error`."""
    code: str | None = None
    """Provider error code, if any."""
    recoverable: bool = True
    """Whether the session can continue. A protocol `error` is recoverable; a dropped connection is not."""

    event_kind: Literal['session_error'] = 'session_error'
    """Event type identifier, used as a discriminator."""


@dataclass
class WebSource:
    """A web page the model used to ground its response."""

    url: str
    """URL of the source page."""
    title: str | None = None
    """Title of the page, if the provider reported one."""


@dataclass
class Sources:
    """Web sources the model grounded its response on (search results or fetched URLs).

    Emitted by providers that report grounding metadata — e.g. Gemini Live when the agent uses
    [`WebSearch`][pydantic_ai.capabilities.WebSearch] (Grounding with Google Search) or
    [`WebFetch`][pydantic_ai.capabilities.WebFetch] (URL context). Use it to surface citations in a UI.
    """

    sources: list[WebSource]
    """The web pages the response drew on."""
    queries: list[str] = field(default_factory=list[str])
    """Search queries the model issued, if the provider reported them."""

    event_kind: Literal['sources'] = 'sources'
    """Event type identifier, used as a discriminator."""


@dataclass
class Grounding:
    """Native tool call/return parts reconstructed from a grounded turn's provider metadata.

    The history-facing companion to [`Sources`][pydantic_ai.realtime.Sources]: where `Sources` carries a
    flattened citation list for a UI, this carries the exact
    [`NativeToolCallPart`][pydantic_ai.messages.NativeToolCallPart] /
    [`NativeToolReturnPart`][pydantic_ai.messages.NativeToolReturnPart] pair(s) a classic
    [`Model`][pydantic_ai.models.Model] request produces for the same grounding — the flattened `Sources`
    drops fields (a source's `domain`, a fetch's retrieval status) and merges search and URL-context
    results, so it can't reproduce those parts faithfully. The session folds these into the turn's
    assistant [`ModelResponse`][pydantic_ai.messages.ModelResponse] rather than yielding them, so a
    grounded voice turn's [`all_messages`][pydantic_ai.realtime.RealtimeSession.all_messages] matches a
    classic [`Agent.run`][pydantic_ai.agent.AbstractAgent.run] turn.
    """

    parts: list[ModelResponsePart]
    """The native tool call/return parts to fold into the assistant response's history."""

    event_kind: Literal['grounding'] = 'grounding'
    """Event type identifier, used as a discriminator."""


RealtimeEvent = TypeAliasType(
    'RealtimeEvent',
    AudioDelta
    | Transcript
    | InputTranscript
    | ToolCall
    | TurnComplete
    | SpeechStarted
    | SpeechStopped
    | SessionUsage
    | RateLimits
    | Reconnected
    | Sources
    | Grounding
    | SessionError,
)
"""Union of the low-level codec events yielded by [`RealtimeConnection`][pydantic_ai.realtime.RealtimeConnection].

This is the provider-facing vocabulary: providers translate their wire protocol into these events, and
[`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] translates them again into the shared
[`RealtimeSessionEvent`][pydantic_ai.realtime.RealtimeSessionEvent] vocabulary while building
[`ModelMessage`][pydantic_ai.messages.ModelMessage] history.
"""


# Session-level events (yielded by `RealtimeSession.__aiter__`).
#
# A session translates the low-level codec events into the shared message/part event vocabulary from
# `pydantic_ai.messages`: `AudioDelta`/`Transcript`/`InputTranscript` become `PartStartEvent` /
# `PartDeltaEvent` / `PartEndEvent` for `SpeechPart`s, and `ToolCall` becomes a
# `ToolCallPart` part (start/end) plus `FunctionToolCallEvent` / `FunctionToolResultEvent` around its
# execution. The remaining control-plane events pass through unchanged.


RealtimeSessionEvent = TypeAliasType(
    'RealtimeSessionEvent',
    PartStartEvent
    | PartDeltaEvent
    | PartEndEvent
    | FunctionToolCallEvent
    | FunctionToolResultEvent
    | TurnComplete
    | SpeechStarted
    | SpeechStopped
    | SessionUsage
    | RateLimits
    | Reconnected
    | Sources
    | SessionError,
)
"""Union of events yielded by [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession].

Content is streamed as the shared [`PartStartEvent`][pydantic_ai.messages.PartStartEvent] /
[`PartDeltaEvent`][pydantic_ai.messages.PartDeltaEvent] / [`PartEndEvent`][pydantic_ai.messages.PartEndEvent]
events (carrying [`SpeechPart`][pydantic_ai.messages.SpeechPart]s and
[`ToolCallPart`][pydantic_ai.messages.ToolCallPart]s), tool execution as
[`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent] /
[`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent], and the rest as realtime
control-plane events.
"""


@dataclass(frozen=True)
class RealtimeCapabilities:
    """What a [`RealtimeModel`][pydantic_ai.realtime.RealtimeModel] supports.

    A [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] queries these flags to reject
    unsupported operations with a clear error *before* sending them, rather than letting the provider
    fail mid-session. Read them via [`RealtimeModel.capabilities`][pydantic_ai.realtime.RealtimeModel.capabilities];
    each flag maps to the session methods a provider may not support.
    """

    image_input: bool = False
    """Whether the model accepts image/video frames via [`send_image`][pydantic_ai.realtime.RealtimeSession.send_image]."""
    manual_turn_control: bool = False
    """Whether the model supports manual turn-taking — [`commit_audio`][pydantic_ai.realtime.RealtimeSession.commit_audio],
    [`clear_audio`][pydantic_ai.realtime.RealtimeSession.clear_audio], and
    [`create_response`][pydantic_ai.realtime.RealtimeSession.create_response] (push-to-talk). When `False` the
    provider only supports automatic voice activity detection."""
    interruption: bool = False
    """Whether the model supports server-side interruption — cancelling the model's in-progress response
    via [`interrupt`][pydantic_ai.realtime.RealtimeSession.interrupt]."""
    output_truncation: bool = False
    """Whether the model can truncate its in-progress audio output to the point the user actually heard,
    via [`truncate_output`][pydantic_ai.realtime.RealtimeSession.truncate_output] and the `audio_end_ms`
    argument of [`interrupt`][pydantic_ai.realtime.RealtimeSession.interrupt].

    Distinct from [`interruption`][pydantic_ai.realtime.RealtimeCapabilities.interruption]: a provider may
    support cancelling a response (barge-in) without supporting output truncation. OpenAI supports both;
    xAI Grok Voice supports cancellation but not truncation."""
    session_seeding: bool = False
    """Whether the model can seed a session with prior conversation (`message_history`)."""


class RealtimeConnection(ABC):
    """A live connection to a realtime model.

    Providers implement this to handle protocol-specific framing (WebSocket frames,
    HTTP/2 messages, etc.). Content is fed in via [`send`][pydantic_ai.realtime.RealtimeConnection.send]
    and events are consumed by iterating the connection.
    """

    @abstractmethod
    async def send(self, content: RealtimeInput) -> None:
        """Feed content into the session.

        The accepted input types depend on the provider: OpenAI accepts `AudioInput`,
        `TextInput`, and `ToolResult`, while Gemini Live additionally accepts `ImageInput`.
        Providers raise `NotImplementedError` for input types they don't support.
        """
        raise NotImplementedError

    @abstractmethod
    def __aiter__(self) -> AsyncIterator[RealtimeEvent]:
        """Iterate over events received from the model."""
        raise NotImplementedError


class RealtimeModel(ABC):
    """Abstract base class for realtime model providers.

    Unlike [`Model`][pydantic_ai.models.Model], which is request-response, a realtime model
    opens a persistent bidirectional connection for streaming content in and out.
    """

    @abstractmethod
    def connect(
        self,
        *,
        instructions: str,
        tools: list[ToolDefinition] | None = None,
        native_tools: list[AbstractNativeTool] | None = None,
        model_settings: ModelSettings | None = None,
        messages: Sequence[ModelMessage] | None = None,
    ) -> AbstractAsyncContextManager[RealtimeConnection]:
        """Open a connection to the realtime model.

        Args:
            instructions: System instructions for the session.
            tools: Tool definitions the model may invoke.
            native_tools: Provider-native tools (e.g. [`WebSearchTool`][pydantic_ai.native_tools.WebSearchTool])
                the model runs server-side. Providers raise `UserError` for ones they don't support.
            model_settings: Optional provider-specific settings.
            messages: Optional prior conversation to seed the session with, projected to the provider's
                initial conversation items. Only text/transcript content is seeded (v1): audio is not
                replayed. Providers degrade gracefully, dropping content they can't project.

        Returns:
            An async context manager yielding a [`RealtimeConnection`][pydantic_ai.realtime.RealtimeConnection].
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model name, e.g. `gpt-realtime`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def capabilities(self) -> RealtimeCapabilities:
        """The operations this model supports, so a session can reject unsupported ones up front."""
        raise NotImplementedError


@dataclass
class ReconnectPolicy:
    """How to recover when a realtime connection drops mid-session.

    On a dropped connection the session is re-dialed and its configuration (instructions, tools,
    voice, ...) re-applied, emitting a [`Reconnected`][pydantic_ai.realtime.Reconnected] event. What
    server-side state survives depends on the provider: OpenAI Realtime starts a fresh turn (the audio
    buffer and prior turns are lost), while Gemini Live restores conversation state when
    `enable_session_resumption=True` (a prerequisite for reconnecting there).
    """

    max_attempts: int = 3
    """Number of re-dial attempts before giving up with a non-recoverable `SessionError`."""
    base_delay: float = 0.5
    """Base backoff delay in seconds; doubles each attempt up to `max_delay`."""
    max_delay: float = 30.0
    """Maximum backoff delay in seconds."""
    jitter: bool = True
    """Whether to apply random jitter to each backoff delay to avoid thundering herds."""


async def reconnect_with_backoff(policy: ReconnectPolicy, attempt: Callable[[], Awaitable[bool]]) -> bool:
    """Retry `attempt` with exponential backoff (and optional jitter) until it succeeds or attempts run out.

    `attempt` performs one provider-specific re-dial and returns whether it succeeded.
    """
    for i in range(policy.max_attempts):
        delay = min(policy.max_delay, policy.base_delay * (2**i))
        if policy.jitter:
            delay *= 0.5 + random.random() * 0.5
        await asyncio.sleep(delay)
        if await attempt():
            return True
    return False


def user_prompt_text(part: UserPromptPart) -> str:
    """Extract the plain text from a `UserPromptPart` (dropping multimodal content for text seeding)."""
    if isinstance(part.content, str):
        return part.content
    return ''.join(item for item in part.content if isinstance(item, str))
