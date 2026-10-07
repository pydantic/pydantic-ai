"""A revocable run handle over a persistent realtime driver."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Sequence
from copy import deepcopy
from typing import Literal, overload

from ..conversation import Conversation
from ..exceptions import UserError
from ..messages import ModelMessage, SpeechPart
from ..run import AgentRunResult
from ..usage import RunUsage
from ._session import RealtimeEvent, RealtimeSession, TranscriptUpdate
from .codec import RealtimeSessionInput
from .profiles import RealtimeModelProfile

__all__ = ('RealtimeRun',)


class RealtimeRun:
    """One execution on a persistent realtime connection.

    Sending, streaming and turn control have the same semantics as `RealtimeSession`. The handle
    is revoked when its context exits: keeping it cannot submit work into a later run. Results and
    message snapshots remain available. Closing or cancelling a run aborts its connection rather
    than pretending outstanding provider work was safely drained.
    """

    def __init__(self, session: RealtimeSession) -> None:
        self._session = session
        self._state = session._run  # pyright: ignore[reportPrivateUsage]
        self._conversation: Conversation | None = None

    def _capture(self) -> None:
        self._conversation = deepcopy(self._session.conversation)

    def _active(self) -> RealtimeSession:
        if self.closed or self._session._run is not self._state:  # pyright: ignore[reportPrivateUsage]
            raise UserError('This realtime run has ended.')
        return self._session

    @property
    def closed(self) -> bool:
        """Whether this run has ended, independently of the underlying connection."""
        return self._state.finished or self._conversation is not None or self._session.closed

    @property
    def result(self) -> AgentRunResult[str] | None:
        """The final result after the run context exits, otherwise `None`."""
        return self._state.result

    @property
    def conversation(self) -> Conversation:
        """A detached snapshot of the conversation at this run's boundary."""
        return deepcopy(self._conversation if self._conversation is not None else self._session.conversation)

    @property
    def usage(self) -> RunUsage:
        """Cumulative session usage through this run, frozen on exit."""
        return self.conversation.usage

    def all_messages(self) -> list[ModelMessage]:
        """Conversation messages through this run, including its history."""
        return self.conversation.messages

    def new_messages(self) -> list[ModelMessage]:
        """Messages produced by this run only."""
        return self.all_messages()[self._state.new_message_index :]

    @property
    def profile(self) -> RealtimeModelProfile:
        """Capabilities of the attached model."""
        return self._session.profile

    @property
    def audio_input_sample_rate(self) -> int:
        """Expected input PCM sample rate in Hz."""
        return self._session.audio_input_sample_rate

    @property
    def audio_output_sample_rate(self) -> int:
        """Output PCM sample rate in Hz."""
        return self._session.audio_output_sample_rate

    @property
    def context_window_used(self) -> float | None:
        """Latest provider context-window measurement while this run is active."""
        return self._active().context_window_used

    async def send(
        self, content: RealtimeSessionInput | Sequence[RealtimeSessionInput], *, respond: bool | None = None
    ) -> None:
        """Send text, images or audio, as on `RealtimeSession.send`."""
        await self._active().send(content, respond=respond)

    async def send_audio(self, data: bytes | AsyncIterable[bytes]) -> None:
        """Stream PCM audio into this run."""
        await self._active().send_audio(data)

    async def commit_audio(self) -> None:
        """Commit the current input audio turn."""
        await self._active().commit_audio()

    async def clear_audio(self) -> None:
        """Clear uncommitted input audio."""
        await self._active().clear_audio()

    async def create_response(self) -> None:
        """Request a response to the current conversation."""
        await self._active().create_response()

    async def wait_for_reply(self) -> None:
        """Wait for generation, not playback or run finalization."""
        await self._active().wait_for_reply()

    async def wait_for_playback(self) -> None:
        """Wait for subscribed audio playback to catch up."""
        await self._active().wait_for_playback()

    @overload
    async def interrupt(self, *, played_ms: int | None = None) -> None: ...

    @overload
    async def interrupt(self, *, played_bytes: int) -> bool: ...

    async def interrupt(self, *, played_ms: int | None = None, played_bytes: int | None = None) -> bool | None:
        """Interrupt generation and truncate unheard audio."""
        session = self._active()
        if played_bytes is not None:
            return await session.interrupt(played_bytes=played_bytes)
        return await session.interrupt(played_ms=played_ms)

    def stream_audio(self) -> AsyncIterator[bytes]:
        """Subscribe to this run's model audio."""
        return self._active().stream_audio()

    @overload
    def stream_transcripts(self, *, delta: Literal[False] = False) -> AsyncIterator[SpeechPart]: ...

    @overload
    def stream_transcripts(self, *, delta: Literal[True]) -> AsyncIterator[TranscriptUpdate]: ...

    def stream_transcripts(self, *, delta: bool = False) -> AsyncIterator[SpeechPart | TranscriptUpdate]:
        """Subscribe to complete transcripts or incremental updates."""
        session = self._active()
        return session.stream_transcripts(delta=True) if delta else session.stream_transcripts()

    def __aiter__(self) -> AsyncIterator[RealtimeEvent]:
        return self._active().__aiter__()

    async def close(self) -> None:
        """Abort this run and its connection; normal context exit instead drains the run."""
        await self._active().close()

    async def hang_up(self) -> None:
        """End the provider call, including a WebRTC media connection."""
        await self._active().hang_up()
