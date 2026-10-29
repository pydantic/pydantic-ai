"""A closed realtime session releases its run state by reference counting, not cyclic GC."""

from __future__ import annotations

import asyncio
import gc
import weakref
from collections.abc import AsyncIterator, Generator
from contextvars import ContextVar
from dataclasses import dataclass

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.realtime.codec import RealtimeCodecEvent, ResponseDone, SessionUsage, ToolCall

from .test_session import FakeRealtimeConnection, FakeRealtimeModel


@dataclass
class Deps:
    pass


def make_agent() -> Agent[Deps, str]:
    agent = Agent(deps_type=Deps, name='agent')

    # Async, so the call stays on the loop: a sync tool runs in a worker thread, which keeps its call's
    # arguments (and so the run context) until the thread is next scheduled, racing the assertion.
    @agent.tool
    async def lookup(ctx: RunContext[Deps]) -> str:
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
    events: list[RealtimeCodecEvent] = []
    for i in range(turns):
        events += [ToolCall(f'call-{i}', tool_name='lookup', args='{}'), ResponseDone()]

    agent = make_agent()
    deps = Deps()
    deps_ref = weakref.ref(deps)

    async with agent.realtime(FakeRealtimeModel(FakeRealtimeConnection(events)), deps=deps).session() as session:
        async for _ in session:
            pass

    del deps, session
    # Let the finished tasks' done callbacks run (one loop pass suffices; a few spare ones keep this
    # robust across Python versions), without giving the cyclic collector a chance to help.
    for _ in range(10):
        await asyncio.sleep(0)

    assert deps_ref() is None


_caller_value: ContextVar[str | None] = ContextVar('caller_value', default=None)


class _RecordsEndContext(FakeRealtimeConnection):
    """Records what the caller's context variable holds while the closing session ends the provider session."""

    seen: str | None = None

    async def _end_session(self) -> AsyncIterator[SessionUsage]:
        self.seen = _caller_value.get()
        return
        yield


async def test_teardown_keeps_the_callers_context() -> None:
    """Only the run context is cut from the teardown: its end-of-session work still sees the caller's context."""
    connection = _RecordsEndContext([])
    token = _caller_value.set('caller')
    try:
        async with make_agent().realtime(FakeRealtimeModel(connection), deps=Deps()).session() as session:
            async for _ in session:
                pass
    finally:
        _caller_value.reset(token)

    assert connection.seen == 'caller'
