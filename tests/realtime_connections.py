"""A base for scripted realtime test connections that yield codec events alone.

Every `RealtimeConnection` yields frames, with the lifecycle events that say which response, spoken turn, and
input each codec event is about. A scripted connection names none of those, so an `InferredLifecycle` works
them out, as it does for a protocol that identifies nothing (Gemini Live, OpenAI GPT-Live).
"""

from __future__ import annotations as _annotations

from abc import abstractmethod
from collections.abc import AsyncIterator, Sequence
from functools import cached_property

from pydantic_ai.realtime.codec import RealtimeCodecEvent, RealtimeConnection, RealtimeInput
from pydantic_ai.realtime.lifecycle import InferredLifecycle, LifecycleEvent, RealtimeFrame, StaleEvent


class InferredConnection(RealtimeConnection):
    """A scripted connection: subclasses take inputs in `send_input()` and yield codec events from `codec_events()`."""

    @cached_property
    def lifecycle(self) -> InferredLifecycle:
        return InferredLifecycle(transcribes=self.input_transcription_enabled)

    async def send(self, content: RealtimeInput) -> Sequence[LifecycleEvent]:
        input_id = self.lifecycle.input_sent(content)
        try:
            await self.send_input(content)
        except BaseException:
            self.lifecycle.input_failed(input_id)
            raise
        return self.lifecycle.take_pending()

    @abstractmethod
    async def send_input(self, content: RealtimeInput) -> None:
        raise NotImplementedError

    def frames(self) -> AsyncIterator[RealtimeFrame]:
        return self.lifecycle.frames(self.codec_events())

    @abstractmethod
    def codec_events(self) -> AsyncIterator[RealtimeCodecEvent]:
        raise NotImplementedError


async def lifecycle_events(connection: RealtimeConnection) -> AsyncIterator[RealtimeCodecEvent | LifecycleEvent]:
    """A connection's codec and lifecycle events, in order, without the stale ones: what the session applies."""
    async for frame in connection.frames():
        for event in frame:
            if not isinstance(event, StaleEvent):
                yield event
