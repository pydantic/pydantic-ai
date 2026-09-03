"""Dynamic-toolset durable-wrapping tests for `AWSLambdaDurability`.

A `DynamicToolset` resolves its inner toolset through a user factory that may do I/O. Left
unwrapped it would resolve inline and re-run when the execution resumes, so both the resolution
and the tool call are checkpointed.
"""

from __future__ import annotations

import pytest

pytest.importorskip('aws_durable_execution_sdk_python')

from typing import Any

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset  # pyright: ignore[reportPrivateUsage]

from pydantic_ai_harness.aws_lambda import AWSLambdaDurability, run_durable

from .conftest import FakeDurableContext


def double_then_done() -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answered = any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)
        if not answered:
            return ModelResponse(parts=[ToolCallPart(tool_name='double', args={'value': 21})])
        return ModelResponse(parts=[TextPart(content='doubled')])

    return FunctionModel(fn, model_name='fn')


class Resolver:
    """Records how often the dynamic toolset factory runs."""

    def __init__(self, tool_name: str = 'double') -> None:
        self.tool_name = tool_name
        self.resolutions = 0
        self.calls: list[int] = []

    def __call__(self, ctx: RunContext[object]) -> FunctionToolset[object]:
        self.resolutions += 1
        toolset = FunctionToolset[object]()

        def double(value: int) -> int:
            self.calls.append(value)
            return value * 2

        toolset.add_function(double, name=self.tool_name)
        return toolset


def build(resolver: Resolver) -> Agent[Any, Any]:
    dynamic = DynamicToolset[object](resolver, id='tools')
    return Agent(double_then_done(), name='d', toolsets=[dynamic], capabilities=[AWSLambdaDurability()])


class TestDynamicToolset:
    def test_resolution_and_call_are_checkpointed(self) -> None:
        resolver = Resolver()
        agent = build(resolver)
        ctx = FakeDurableContext()

        result = run_durable(lambda: agent.run('double 21'), context=ctx)

        assert result.output == 'doubled'
        assert resolver.calls == [21]
        assert 'd__dynamic_toolset__tools.get_tools' in ctx.step_names
        assert 'd__dynamic_toolset__tools.call_tool:double' in ctx.step_names

    def test_resume_reruns_neither_the_factory_nor_the_tool(self) -> None:
        resolver = Resolver()
        agent = build(resolver)

        first = FakeDurableContext()
        run_durable(lambda: agent.run('double 21'), context=first)
        resolutions, calls = resolver.resolutions, list(resolver.calls)
        assert calls == [21]

        resumed = FakeDurableContext(journal=first.operations)
        result = run_durable(lambda: agent.run('double 21'), context=resumed)

        assert result.output == 'doubled'
        assert resumed.invoked == []
        assert resolver.resolutions == resolutions
        assert resolver.calls == calls

    def test_transparent_outside_a_durable_handler(self) -> None:
        resolver = Resolver()
        agent = build(resolver)

        result = agent.run_sync('double 21')

        assert result.output == 'doubled'
        assert resolver.calls == [21]

    def test_a_dynamic_toolset_needs_an_id(self) -> None:
        dynamic = DynamicToolset[object](Resolver())
        with pytest.raises(UserError, match='unique `id`'):
            Agent(double_then_done(), name='d', toolsets=[dynamic], capabilities=[AWSLambdaDurability()])


class TestMultipleDynamicToolsets:
    def test_two_dynamic_toolsets_resolve_concurrently(self) -> None:
        """Two dynamic toolsets are resolved by `CombinedToolset` concurrently, so they issue
        sibling step requests that must queue rather than be rejected as nested."""
        first, second = Resolver(), Resolver(tool_name='double2')
        agent = Agent(
            double_then_done(),
            name='d',
            toolsets=[DynamicToolset[object](first, id='ta'), DynamicToolset[object](second, id='tb')],
            capabilities=[AWSLambdaDurability()],
        )
        ctx = FakeDurableContext()

        result = run_durable(lambda: agent.run('double 21'), context=ctx)

        assert result.output == 'doubled'
        assert 'd__dynamic_toolset__ta.get_tools' in ctx.step_names
        assert 'd__dynamic_toolset__tb.get_tools' in ctx.step_names


class TestEnqueueGuard:
    def test_enqueue_from_the_toolset_factory_raises(self) -> None:
        """Resolution is checkpointed, so on replay the recorded tool set is served and the factory
        never runs again -- anything it enqueued the first time round would be lost. The factory is
        user code, which makes this the discovery path most likely to try."""

        def resolver(ctx: RunContext[object]) -> FunctionToolset[object]:
            ctx.enqueue('later')
            return FunctionToolset[object]()  # pragma: no cover - the enqueue above always raises

        agent = Agent(
            double_then_done(),
            name='d',
            toolsets=[DynamicToolset[object](resolver, id='tools')],
            capabilities=[AWSLambdaDurability()],
        )
        ctx = FakeDurableContext()

        with pytest.raises(UserError, match='enqueue'):
            run_durable(lambda: agent.run('double 21'), context=ctx)
