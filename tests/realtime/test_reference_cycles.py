"""A closed realtime session releases its run state by reference counting, not cyclic GC."""

from __future__ import annotations

import asyncio
import gc
import weakref
from collections.abc import AsyncGenerator, AsyncIterator, Generator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.realtime import RealtimeModel, RealtimeModelSettings
from pydantic_ai.realtime.codec import (
    RealtimeCodecEvent,
    RealtimeConnection,
    RealtimeInput,
    ResponseDone,
    ToolCall,
)
from pydantic_ai.realtime.lifecycle import InferredLifecycle, LifecycleEvent, RealtimeFrame


class ScriptedConnection(RealtimeConnection):
    """Replays a fixed list of codec events, then ends."""

    def __init__(self, events: list[ToolCall | ResponseDone]) -> None:
        self.events = events
        self.lifecycle = InferredLifecycle(transcribes=False)

    async def send(self, content: RealtimeInput) -> Sequence[LifecycleEvent] | None:
        with self.lifecycle.sending(content):
            pass
        return self.lifecycle.take_pending()

    async def codec_events(self) -> AsyncIterator[RealtimeCodecEvent]:
        for event in self.events:
            yield event

    def frames(self) -> AsyncIterator[RealtimeFrame]:
        return self.lifecycle.frames(self.codec_events())


class ScriptedRealtimeModel(RealtimeModel):
    def __init__(self, events: list[ToolCall | ResponseDone]) -> None:
        super().__init__()
        self.events = events

    @property
    def model_name(self) -> str:
        return 'scripted'

    @property
    def system(self) -> str:
        return 'scripted'

    @asynccontextmanager
    async def connect(
        self,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncGenerator[ScriptedConnection]:
        yield ScriptedConnection(self.events)


@dataclass
class Deps:
    pass


def make_agent() -> Agent[Deps, str]:
    agent = Agent(deps_type=Deps, name='agent')

    @agent.tool
    def lookup(ctx: RunContext[Deps]) -> str:
        return 'result'

    return agent


@pytest.fixture
def no_gc() -> Generator[None]:
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


@pytest.mark.usefixtures('no_gc')
@pytest.mark.parametrize('turns', [0, 1, 2])
async def test_closed_session_releases_deps(turns: int) -> None:
    events: list[ToolCall | ResponseDone] = []
    for i in range(turns):
        events += [ToolCall(f'call-{i}', tool_name='lookup', args='{}'), ResponseDone()]

    agent = make_agent()
    deps = Deps()
    deps_ref = weakref.ref(deps)

    async with agent.realtime(ScriptedRealtimeModel(events), deps=deps).session() as session:
        async for _ in session:
            pass

    del deps, session
    # Let the finished tasks' done callbacks run (one loop pass suffices; a few spare ones keep this
    # robust across Python versions), without giving the cyclic collector a chance to help.
    for _ in range(10):
        await asyncio.sleep(0)

    assert deps_ref() is None
