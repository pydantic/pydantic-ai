"""Tests for `RunContext.usage_limits`: the run's enforced limits exposed to tools and capabilities.

These use `TestModel` rather than VCR because the behavior under test is purely about how the
run threads its `UsageLimits` into the `RunContext` that tools and capability hooks receive; no
real provider response is involved.
"""

from __future__ import annotations

from typing import Any

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai._run_context import RunContext as RunContextClass
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage, UsageLimits

pytestmark = pytest.mark.anyio


def _tool_recording_agent(seen: list[UsageLimits | None]) -> Agent[Any]:
    agent = Agent(TestModel(call_tools=['record_limits']))

    @agent.tool
    def record_limits(ctx: RunContext[Any]) -> str:
        seen.append(ctx.usage_limits)
        return 'done'

    return agent


class _RecordLimits(AbstractCapability[Any]):
    def __init__(self, seen: list[UsageLimits | None]):
        self.seen = seen

    async def before_run(self, ctx: RunContext[Any]) -> None:
        self.seen.append(ctx.usage_limits)


def test_run_context_usage_limits_defaults_to_none():
    """A bare/synthetic `RunContext` has no limits: the field defaults to `None`."""
    ctx = RunContextClass(deps=None, model=TestModel(), usage=RunUsage())
    assert ctx.usage_limits is None


def test_usage_limits_visible_in_tool_run_sync():
    """A tool sees the exact `UsageLimits` the run was started with via `run_sync`."""
    seen: list[UsageLimits | None] = []
    agent = _tool_recording_agent(seen)
    limits = UsageLimits(request_limit=10, total_tokens_limit=5000)

    agent.run_sync('hello', usage_limits=limits)

    assert seen == [limits]


async def test_usage_limits_visible_in_tool_run_and_stream():
    """A tool sees the run's `UsageLimits` via both `run` and `run_stream`."""
    seen: list[UsageLimits | None] = []
    agent = _tool_recording_agent(seen)
    limits = UsageLimits(request_limit=10, total_tokens_limit=5000)

    await agent.run('hello', usage_limits=limits)
    async with agent.run_stream('hello', usage_limits=limits) as result:
        await result.get_output()

    assert seen == [limits, limits]


def test_usage_limits_visible_in_capability_hook_run_sync():
    """A capability's `before_run` hook reads the run's limits without being configured with them."""
    seen: list[UsageLimits | None] = []
    limits = UsageLimits(request_limit=3)
    agent = Agent(TestModel(), capabilities=[_RecordLimits(seen)])

    agent.run_sync('hello', usage_limits=limits)

    assert seen == [limits]


async def test_usage_limits_visible_in_capability_hook_run_and_stream():
    """A capability's `before_run` hook reads the run's limits via `run` and `run_stream`."""
    seen: list[UsageLimits | None] = []
    limits = UsageLimits(request_limit=3)
    agent = Agent(TestModel(), capabilities=[_RecordLimits(seen)])

    await agent.run('hello', usage_limits=limits)
    async with agent.run_stream('hello', usage_limits=limits) as result:
        await result.get_output()

    assert seen == [limits, limits]


def test_usage_limits_defaults_to_effective_limits_during_run():
    """Without an explicit `usage_limits=`, tools see the run's effective default limits, not `None`."""
    seen: list[UsageLimits | None] = []
    agent = _tool_recording_agent(seen)

    agent.run_sync('hello')

    assert seen == [UsageLimits()]
