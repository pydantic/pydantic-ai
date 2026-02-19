"""Abstractions for realtime speech-to-speech models.

Providers like OpenAI Realtime, Gemini Live, and Nova Sonic offer bidirectional
streaming audio APIs. These use persistent connections (WebSocket or HTTP/2) rather
than the request-response pattern of the standard `Model` interface.

This module defines the provider-facing ABCs (`RealtimeModel`, `RealtimeConnection`)
and all event types yielded during a realtime session.
"""

from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field

from typing_extensions import TypeAliasType

from ..settings import ModelSettings
from ..tools import ToolDefinition

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
# Session-level events (yielded by VoiceSession.__aiter__)
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


VoiceSessionEvent = TypeAliasType(
    'VoiceSessionEvent',
    AudioDelta | Transcript | InputTranscript | ToolCallStarted | ToolCallCompleted | TurnComplete | SessionError,
)


# ---------------------------------------------------------------------------
# Provider ABCs
# ---------------------------------------------------------------------------


class RealtimeConnection(ABC):
    """A live connection to a realtime model.

    Providers implement this to handle protocol-specific framing (WebSocket
    messages, HTTP/2 frames, etc.).
    """

    @abstractmethod
    async def send_audio(self, data: bytes) -> None:
        """Send an audio chunk to the model.

        The expected format (sample rate, encoding) is provider-specific.
        See each provider's documentation for details.
        """
        raise NotImplementedError

    @abstractmethod
    async def send_tool_result(self, tool_call_id: str, output: str) -> None:
        """Send the result of a tool call back to the model."""
        raise NotImplementedError

    @abstractmethod
    def __aiter__(self) -> AsyncIterator[RealtimeEvent]:
        """Iterate over events from the model."""
        raise NotImplementedError


class RealtimeModel(ABC):
    """Abstract base for realtime speech-to-speech model providers.

    Unlike the standard `Model` ABC which uses request-response, this opens a
    persistent bidirectional connection for streaming audio in and out.
    """

    @abstractmethod
    def connect(
        self,
        *,
        instructions: str,
        tools: list[ToolDefinition] = field(default_factory=list),
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
