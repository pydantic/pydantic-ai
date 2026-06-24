"""Abstractions for realtime multimodal models.

Providers like OpenAI Realtime, Gemini Live, and Amazon Nova Sonic offer bidirectional
streaming APIs over a persistent connection (WebSocket or HTTP/2) rather than the
request-response pattern of the standard [`Model`][pydantic_ai.models.Model] interface.

This module defines the provider-facing ABCs ([`RealtimeModel`][pydantic_ai.realtime.RealtimeModel],
[`RealtimeConnection`][pydantic_ai.realtime.RealtimeConnection]) and the event/input types
exchanged over a realtime session.
"""

from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass

from typing_extensions import TypeAliasType

from ..settings import ModelSettings
from ..tools import ToolDefinition
from ..usage import RequestUsage

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


@dataclass
class SpeechStarted:
    """The provider detected that the user started speaking.

    Useful for barge-in: stop playing any buffered model audio when this arrives, since the model's
    in-progress turn is being interrupted.
    """


@dataclass
class SpeechStopped:
    """The provider detected that the user stopped speaking.

    Useful as a 'processing' indicator: the user's turn has ended and the model is about to respond.
    """


@dataclass
class Usage:
    """Token usage reported by the provider for a completed model response."""

    usage: RequestUsage
    """Normalized token usage for the response, ready to accumulate into a `RunUsage`."""


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


@dataclass
class Reconnected:
    """The connection dropped and was automatically re-established.

    Session configuration (instructions, tools, voice, ...) is restored, but server-side conversation
    state (the audio buffer and prior turns) is not — treat it as the start of a fresh turn.
    """


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


RealtimeEvent = TypeAliasType(
    'RealtimeEvent',
    'AudioDelta | Transcript | InputTranscript | ToolCall | TurnComplete | SpeechStarted | SpeechStopped | Usage | RateLimits | Reconnected | SessionError',
)
"""Union of events yielded by [`RealtimeConnection`][pydantic_ai.realtime.RealtimeConnection]."""


# Session-level events (yielded by `RealtimeSession.__aiter__`).
#
# A session replaces raw `ToolCall` events with `ToolCallStarted` / `ToolCallCompleted`
# bookends around its own tool execution; everything else passes through unchanged.


@dataclass
class ToolCallStarted:
    """The session has begun executing a tool call."""

    tool_name: str
    """Name of the tool being executed."""
    tool_call_id: str
    """Identifier of the originating `ToolCall`."""


@dataclass
class ToolCallCompleted:
    """The session has finished executing a tool call and sent the result to the model."""

    tool_name: str
    """Name of the tool that was executed."""
    tool_call_id: str
    """Identifier of the originating `ToolCall`."""
    result: str
    """The tool's output that was sent back to the model."""


RealtimeSessionEvent = TypeAliasType(
    'RealtimeSessionEvent',
    'AudioDelta | Transcript | InputTranscript | ToolCallStarted | ToolCallCompleted | TurnComplete | SpeechStarted | SpeechStopped | Usage | RateLimits | Reconnected | SessionError',
)
"""Union of events yielded by [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession]."""


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
        model_settings: ModelSettings | None = None,
    ) -> AbstractAsyncContextManager[RealtimeConnection]:
        """Open a connection to the realtime model.

        Args:
            instructions: System instructions for the session.
            tools: Tool definitions the model may invoke.
            model_settings: Optional provider-specific settings.

        Returns:
            An async context manager yielding a [`RealtimeConnection`][pydantic_ai.realtime.RealtimeConnection].
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model name, e.g. `gpt-realtime`."""
        raise NotImplementedError
