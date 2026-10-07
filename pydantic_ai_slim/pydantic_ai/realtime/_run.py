"""A revocable run handle over a persistent realtime driver."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar, overload

import anyio

from .._utils import aclose_all, cancel_and_drain
from ..conversation import Conversation
from ..exceptions import UserError
from ..messages import ModelMessage, SpeechPart
from ..run import AgentRunResult
from ..usage import RunUsage
from ._session import RealtimeEvent, RealtimeSession, TranscriptUpdate
from .codec import RealtimeSessionInput
from .profiles import RealtimeModelProfile

__all__ = ('RealtimeRun',)

_Item = TypeVar('_Item')


@dataclass
class _AudioInput:
    stream: _RunStream[bytes]
    scope: anyio.CancelScope
    finished: anyio.Event = field(default_factory=anyio.Event)


class _RunStream(AsyncIterator[_Item], Generic[_Item]):
    """Pull a wrapper in one owned task, including its same-task async-generator cleanup."""

    def __init__(self, owner: RealtimeRun, source: AsyncIterator[_Item]) -> None:
        self.owner = owner
        self.source = source
        self.task: asyncio.Task[None] | None = None
        self.requested = asyncio.Event()
        self.ready = asyncio.Event()
        self.item: _Item | None = None
        self.error: BaseException | None = None
        self.ended = False
        self.reading = False
        self.pulling = False
        self.closing = False
        self.close_finished = asyncio.Event()

    def __aiter__(self) -> AsyncIterator[_Item]:
        return self

    async def __anext__(self) -> _Item:
        if self.task is None:
            self.owner._active()  # pyright: ignore[reportPrivateUsage]
            self.task = asyncio.create_task(self._pump(), name='realtime-run-stream')
        if self.reading:
            raise UserError('This realtime stream is already being read.')
        self.reading = True
        try:
            if not self.ended:
                # A cancelled reader leaves its pull in flight. Reuse that pull (or its buffered
                # result) rather than letting a second request overwrite the unconsumed item.
                if not self.pulling:
                    self.pulling = True
                    self.requested.set()
                await self.ready.wait()
                self.ready.clear()
                self.pulling = False
            if self.error is not None:
                error, self.error = self.error, None
                raise error
            if self.ended:
                raise StopAsyncIteration
            item, self.item = self.item, None
            assert item is not None
            return item
        finally:
            self.reading = False

    async def _pump(self) -> None:
        try:
            with anyio.CancelScope() as cleanup_scope:
                try:
                    while True:
                        await self.requested.wait()
                        self.requested.clear()
                        self.item = await anext(self.source)
                        self.ready.set()
                finally:
                    # Enter this scope before the wrapper: wrappers may hold their own scopes
                    # across yields, and closing them must preserve the task's LIFO scope stack.
                    cleanup_scope.shield = True
                    await aclose_all((self.source,))
        except (StopAsyncIteration, asyncio.CancelledError):
            pass
        except BaseException as exc:
            self.error = exc
        finally:
            self.ended = True
            self.ready.set()

    async def aclose(self) -> None:
        if self.closing:
            await self.close_finished.wait()
            return
        self.closing = True
        try:
            if self.task is not None:
                await cancel_and_drain(self.task)
            else:
                await aclose_all((self.source,))
            if self.error is not None:
                error, self.error = self.error, None
                raise error
        finally:
            self.ended = True
            self.ready.set()
            self.close_finished.set()


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
        self._streams: list[AsyncIterator[object]] = []
        self._audio_inputs: list[_AudioInput] = []
        self._inputs_stopping = False

    async def _stop_inputs(self, *, abort: bool = False) -> None:
        self._inputs_stopping = True
        inputs = self._audio_inputs.copy()
        if abort:
            for input in inputs:
                input.scope.cancel()
        try:
            await aclose_all(input.stream for input in inputs)
        finally:
            for input in inputs:
                await input.finished.wait()

    async def _close_streams(self) -> None:
        with anyio.CancelScope(shield=True):
            await aclose_all(self._streams)

    def _stream(self, source: AsyncIterator[_Item]) -> AsyncIterator[_Item]:
        stream = _RunStream(self, source)
        self._streams.append(stream)
        return stream

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
        session = self._active()
        if isinstance(data, bytes):
            await session.send_audio(data)
            return
        if self._inputs_stopping:
            raise UserError('This realtime run is no longer accepting audio streams.')
        with anyio.CancelScope() as scope:
            input = _AudioInput(_RunStream(self, aiter(data)), scope)
            self._audio_inputs.append(input)
            try:
                async for chunk in input.stream:
                    if self._inputs_stopping:
                        break
                    # Normal exit waits for a wire send; abort cancels it and closes the connection.
                    await self._active().send_audio(chunk)
            finally:
                try:
                    with anyio.CancelScope(shield=True):
                        await input.stream.aclose()
                finally:
                    self._audio_inputs.remove(input)
                    input.finished.set()

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
        return self._stream(self._active().stream_audio())

    @overload
    def stream_transcripts(self, *, delta: Literal[False] = False) -> AsyncIterator[SpeechPart]: ...

    @overload
    def stream_transcripts(self, *, delta: Literal[True]) -> AsyncIterator[TranscriptUpdate]: ...

    def stream_transcripts(self, *, delta: bool = False) -> AsyncIterator[SpeechPart | TranscriptUpdate]:
        """Subscribe to complete transcripts or incremental updates."""
        session = self._active()
        return self._stream(session.stream_transcripts(delta=True) if delta else session.stream_transcripts())

    def __aiter__(self) -> AsyncIterator[RealtimeEvent]:
        return self._stream(self._active().__aiter__())

    async def close(self) -> None:
        """Abort this run and its connection; normal context exit instead drains the run."""
        await self._active().close()

    async def hang_up(self) -> None:
        """End the provider call, including a WebRTC media connection."""
        await self._active().hang_up()
