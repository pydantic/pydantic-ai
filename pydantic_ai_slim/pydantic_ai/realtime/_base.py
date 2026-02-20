"""Abstractions for realtime multimodal models.

Providers like OpenAI Realtime, Gemini Live, and Nova Sonic offer bidirectional
streaming APIs. These use persistent connections (WebSocket or HTTP/2) rather
than the request-response pattern of the standard `Model` interface.

This module defines the provider-facing ABCs (`RealtimeModel`, `RealtimeConnection`)
and all event/input types for a realtime session.
"""

from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass

from typing_extensions import TypeAliasType, TypeVar

from ..settings import ModelSettings
from ..tools import ToolDefinition

# ---------------------------------------------------------------------------
# Input content types (fed into the connection via send())
# ---------------------------------------------------------------------------


@dataclass
class AudioInput:
    """A chunk of audio data to send to the model."""

    data: bytes


@dataclass
class ImageInput:
    """An image frame to send to the model (e.g. for Gemini Live video).

    Not all providers support image input. Sending this to a provider that
    doesn't will raise `NotImplementedError` at runtime (and be caught by
    type checkers for providers that don't include `ImageInput` in their
    type parameter).
    """

    data: bytes
    mime_type: str = 'image/jpeg'


@dataclass
class ToolResult:
    """The result of a tool call to send back to the model."""

    tool_call_id: str
    output: str


RealtimeInput = TypeAliasType(
    'RealtimeInput',
    AudioInput | ImageInput | ToolResult,
)

# ---------------------------------------------------------------------------
# Connection-level events (yielded by RealtimeConnection.__aiter__)
# ---------------------------------------------------------------------------


@dataclass
class AudioDelta:
    """A chunk of audio data from the model."""

    data: bytes


@dataclass
class Transcript:
    """Model output transcription (partial or final)."""

    text: str
    is_final: bool = False


@dataclass
class InputTranscript:
    """Transcription of the user's speech input (partial or final)."""

    text: str
    is_final: bool = False


@dataclass
class ToolCall:
    """The model is requesting a tool call."""

    tool_call_id: str
    tool_name: str
    args: str  # JSON string


@dataclass
class TurnComplete:
    """The model finished its turn."""

    interrupted: bool = False


@dataclass
class SessionError:
    """An error occurred in the realtime session."""

    message: str


RealtimeEvent = TypeAliasType(
    'RealtimeEvent',
    AudioDelta | Transcript | InputTranscript | ToolCall | TurnComplete | SessionError,
)

# ---------------------------------------------------------------------------
# Session-level events (yielded by RealtimeSession.__aiter__)
# ---------------------------------------------------------------------------


@dataclass
class ToolCallStarted:
    """Emitted when the agent begins executing a tool call."""

    tool_name: str
    tool_call_id: str


@dataclass
class ToolCallCompleted:
    """Emitted when the agent finishes executing a tool call."""

    tool_name: str
    tool_call_id: str
    result: str


RealtimeSessionEvent = TypeAliasType(
    'RealtimeSessionEvent',
    AudioDelta | Transcript | InputTranscript | ToolCallStarted | ToolCallCompleted | TurnComplete | SessionError,
)


# ---------------------------------------------------------------------------
# Provider ABCs
# ---------------------------------------------------------------------------

InputT = TypeVar('InputT', bound=RealtimeInput, default=RealtimeInput)


class RealtimeConnection(ABC):
    """A live connection to a realtime model.

    Providers implement this to handle protocol-specific framing (WebSocket
    messages, HTTP/2 frames, etc.). Content is fed in via `send()` and events
    are yielded via `__aiter__` - similar to a sans-IO protocol handler.
    """

    @abstractmethod
    async def send(self, content: RealtimeInput) -> None:
        """Feed content into the realtime session.

        The concrete input types accepted depend on the provider. For example,
        OpenAI accepts `AudioInput` and `ToolResult`, while Gemini Live also
        accepts `ImageInput`.
        """
        raise NotImplementedError

    @abstractmethod
    def __aiter__(self) -> AsyncIterator[RealtimeEvent]:
        """Iterate over events from the model."""
        raise NotImplementedError


class RealtimeModel(ABC):
    """Abstract base for realtime model providers.

    Unlike the standard `Model` ABC which uses request-response, this opens a
    persistent bidirectional connection for streaming content in and out.
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
            An async context manager yielding a `RealtimeConnection`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model name, e.g. ``'gpt-4o-realtime'``."""
        raise NotImplementedError
