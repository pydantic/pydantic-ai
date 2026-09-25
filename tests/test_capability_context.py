from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.anyio


async def test_capability_contexts_isolate_assignments_and_share_run_state() -> None:
    contexts: list[RunContext[list[str]]] = []

    @dataclass
    class Recorder(AbstractCapability[list[str]]):
        async def before_run(self, ctx: RunContext[list[str]]) -> None:
            assert ctx.capability_active is True
            assert ctx.retry == 0
            ctx.retry = 99
            contexts.append(ctx)
            assert self.id is not None
            ctx.deps.append(self.id)

        async def wrap_run(
            self, ctx: RunContext[list[str]], *, handler: Callable[[], Awaitable[AgentRunResult[Any]]]
        ) -> AgentRunResult[Any]:
            assert ctx.retry == 0
            ctx.retry = 100
            contexts.append(ctx)
            result = await handler()
            assert ctx.retry == 100
            assert ctx.capability_active is True
            return result

    agent = Agent(
        TestModel(custom_output_text='ok'),
        deps_type=list[str],
        capabilities=[Recorder(id='first'), Recorder(id='second')],
    )

    @agent.tool
    async def probe(ctx: RunContext[list[str]]) -> str:
        assert ctx.retry == 0
        assert ctx.capability_active is None
        assert all(ctx.deps is cap_ctx.deps and ctx.usage is cap_ctx.usage for cap_ctx in contexts)
        ctx.deps.append('tool')
        return 'ok'

    for _ in range(2):
        contexts.clear()
        deps: list[str] = []
        result = await agent.run('hello', deps=deps)
        assert result.output == 'ok'
        assert deps == ['first', 'second', 'tool']
        assert len({id(ctx) for ctx in contexts}) == 4
        assert [ctx.retry for ctx in contexts] == [100, 100, 99, 99]


async def test_capability_context_subclass_is_reconstructed() -> None:
    @dataclass
    class CheckedContext(RunContext[None]):
        active_at_init: bool | None = field(init=False)

        def __post_init__(self) -> None:
            self.active_at_init = self.capability_active

    contexts: list[RunContext[None]] = []

    class Recorder(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            contexts.append(ctx)

    ctx = CheckedContext(deps=None, model=TestModel(), usage=RunUsage())
    await CombinedCapability([Recorder()]).before_run(ctx)
    (cap_ctx,) = contexts
    assert isinstance(cap_ctx, CheckedContext)
    assert cap_ctx is not ctx
    assert cap_ctx.active_at_init is True
    assert cap_ctx.usage is ctx.usage
    assert ctx.active_at_init is None
    assert ctx.capability_active is None
