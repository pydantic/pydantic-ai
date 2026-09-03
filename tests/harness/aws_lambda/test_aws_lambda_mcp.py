"""MCP durable-wrapping tests for `AWSLambdaDurability`.

A lightweight `FakeMCPToolset` stands in for a real server: it is a genuine `MCPToolset`
subclass, so the capability's `isinstance` wrapping and `tool_for_tool_def` rebuild apply, but
its wire methods return in-memory results, which keeps the test off Docker and the network.
"""

from __future__ import annotations

import pytest

pytest.importorskip('aws_durable_execution_sdk_python')
pytest.importorskip('pydantic_ai.mcp')

from typing import Any

import anyio
from pydantic_ai import Agent, ToolsetTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    InstructionPart,
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext, ToolDefinition

from pydantic_ai_harness.aws_lambda import AWSLambdaDurability, run_durable

from .conftest import FakeDurableContext

_ADD_SCHEMA = {
    'type': 'object',
    'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
    'required': ['a', 'b'],
}


class FakeMCPToolset(MCPToolset[object]):
    """In-memory `MCPToolset` whose I/O methods return canned results.

    Bypasses `MCPToolset.__init__` (which would build a real transport) and sets only the
    attributes the durable wrapper and the run touch.
    """

    def __init__(
        self,
        *,
        id: str,
        instructions: str | None = None,
        include_instructions: bool = True,
        tool_metadata: dict[str, object] | None = None,
        tool_name: str = 'add',
    ) -> None:
        self._id = id
        self._tool_name = tool_name
        self.max_retries = None
        self.cache_tools = True
        self.include_instructions = include_instructions
        self.include_return_schema = None
        self._instructions_text = instructions
        self._tool_metadata = tool_metadata
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.enter_count = 0
        self._session_depth = 0
        self.implicit_sessions = 0

    async def __aenter__(self) -> FakeMCPToolset:
        self.enter_count += 1
        self._session_depth += 1
        return self

    async def __aexit__(self, *args: object) -> None:
        self._session_depth -= 1

    async def _require_session(self) -> None:
        """Model a real server: I/O needs an active session, and a call without one opens its
        own implicit session for the duration of the call, as `MCPToolset` does."""
        if self._session_depth == 0:
            self.implicit_sessions += 1
            await self.__aenter__()
            await self.__aexit__(None, None, None)

    async def get_tools(self, ctx: RunContext[object]) -> dict[str, ToolsetTool[object]]:
        await self._require_session()
        tool_def = ToolDefinition(
            name=self._tool_name,
            description='Add two integers.',
            parameters_json_schema=_ADD_SCHEMA,
            metadata=self._tool_metadata,
        )
        return {self._tool_name: self.tool_for_tool_def(tool_def, ctx=ctx)}

    async def get_instructions(self, ctx: RunContext[object]) -> InstructionPart | None:
        await self._require_session()
        if not self.include_instructions or self._instructions_text is None:
            return None
        return InstructionPart(content=self._instructions_text)

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[object], tool: ToolsetTool[object]
    ) -> int:
        await self._require_session()
        self.tool_calls.append((name, dict(tool_args)))
        return int(tool_args['a']) + int(tool_args['b'])


def add_then_done() -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answered = any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)
        if not answered:
            return ModelResponse(parts=[ToolCallPart(tool_name='add', args={'a': 2, 'b': 3})])
        return ModelResponse(parts=[TextPart(content='summed')])

    return FunctionModel(fn, model_name='fn')


def add_then_add2_then_done() -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answers = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if not answers:
            return ModelResponse(parts=[ToolCallPart(tool_name='add', args={'a': 2, 'b': 3})])
        if len(answers) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='add2', args={'a': 5, 'b': 8})])
        return ModelResponse(parts=[TextPart(content='summed')])

    return FunctionModel(fn, model_name='fn')


def build(server: FakeMCPToolset) -> Agent[Any, Any]:
    return Agent(add_then_done(), name='calc', toolsets=[server], capabilities=[AWSLambdaDurability()])


class TestMcpCheckpointing:
    def test_get_tools_instructions_and_call_tool_are_checkpointed(self) -> None:
        server = FakeMCPToolset(id='calc', instructions='Use the calculator.')
        agent = build(server)
        ctx = FakeDurableContext()

        result = run_durable(lambda: agent.run('add 2 and 3'), context=ctx)

        assert result.output == 'summed'
        assert server.tool_calls == [('add', {'a': 2, 'b': 3})]
        assert 'calc__mcp_server__calc.get_tools' in ctx.step_names
        assert 'calc__mcp_server__calc.get_instructions' in ctx.step_names
        assert 'calc__mcp_server__calc.call_tool' in ctx.step_names

    def test_resume_does_not_reach_the_server(self) -> None:
        server = FakeMCPToolset(id='calc', instructions='Use the calculator.')
        agent = build(server)

        first = FakeDurableContext()
        run_durable(lambda: agent.run('add 2 and 3'), context=first)
        assert len(server.tool_calls) == 1

        resumed = FakeDurableContext(journal=first.operations)
        result = run_durable(lambda: agent.run('add 2 and 3'), context=resumed)

        assert result.output == 'summed'
        assert resumed.invoked == []
        assert len(server.tool_calls) == 1

    def test_instructions_are_omitted_when_disabled(self) -> None:
        server = FakeMCPToolset(id='calc', instructions='Use it.', include_instructions=False)
        agent = build(server)
        ctx = FakeDurableContext()

        run_durable(lambda: agent.run('add 2 and 3'), context=ctx)

        assert 'calc__mcp_server__calc.get_instructions' not in ctx.step_names

    def test_the_run_keeps_one_session(self) -> None:
        """`lifecycle='enter-always'` keeps a single session for the run rather than an implicit
        session per call, matching what a plain non-durable run does."""
        server = FakeMCPToolset(id='calc', instructions='Use the calculator.')
        agent = build(server)
        ctx = FakeDurableContext()

        run_durable(lambda: agent.run('add 2 and 3'), context=ctx)

        assert server.implicit_sessions == 0
        assert server.enter_count >= 1

    def test_opting_an_mcp_tool_out_is_rejected(self) -> None:
        server = FakeMCPToolset(id='calc', instructions='Use it.', tool_metadata={'aws_lambda': False})
        agent = build(server)
        ctx = FakeDurableContext()

        with pytest.raises(UserError, match='cannot run outside a durable step'):
            run_durable(lambda: agent.run('add 2 and 3'), context=ctx)

    def test_transparent_outside_a_durable_handler(self) -> None:
        server = FakeMCPToolset(id='calc', instructions='Use the calculator.')
        agent = build(server)

        result = agent.run_sync('add 2 and 3')

        assert result.output == 'summed'
        assert server.tool_calls == [('add', {'a': 2, 'b': 3})]


class TestMultipleServers:
    """`CombinedToolset` lists toolsets concurrently, so two servers issue sibling step requests.

    Sibling steps must queue and run one at a time, not be rejected as if they were nested.
    """

    def test_two_mcp_servers_are_listed_and_called(self) -> None:
        first = FakeMCPToolset(id='s1', instructions='One.')
        second = FakeMCPToolset(id='s2', instructions='Two.', tool_name='add2')
        agent = Agent(
            add_then_add2_then_done(), name='calc', toolsets=[first, second], capabilities=[AWSLambdaDurability()]
        )
        ctx = FakeDurableContext()

        result = run_durable(lambda: agent.run('add 2 and 3'), context=ctx)

        assert result.output == 'summed'
        assert 'calc__mcp_server__s1.get_tools' in ctx.step_names
        assert 'calc__mcp_server__s2.get_tools' in ctx.step_names
        assert first.tool_calls == [('add', {'a': 2, 'b': 3})]
        assert second.tool_calls == [('add2', {'a': 5, 'b': 8})]

    def test_two_servers_replay_from_checkpoints(self) -> None:
        def build_two() -> tuple[FakeMCPToolset, FakeMCPToolset, Agent[Any, Any]]:
            a = FakeMCPToolset(id='s1', instructions='One.')
            b = FakeMCPToolset(id='s2', instructions='Two.', tool_name='add2')
            return a, b, Agent(add_then_done(), name='calc', toolsets=[a, b], capabilities=[AWSLambdaDurability()])

        _, _, agent = build_two()
        first = FakeDurableContext()
        run_durable(lambda: agent.run('add 2 and 3'), context=first)

        # A replay must line up with the recorded operations, which only holds if sibling steps
        # were queued in a stable order.
        resumed = FakeDurableContext(journal=first.operations)
        result = run_durable(lambda: agent.run('add 2 and 3'), context=resumed)

        assert result.output == 'summed'
        assert resumed.invoked == []


class TestFakeServerFidelity:
    def test_the_fake_opens_an_implicit_session_when_not_entered(self) -> None:
        """Without this the `implicit_sessions == 0` assertion above would be vacuous."""
        server = FakeMCPToolset(id='calc')

        async def call_without_entering() -> None:
            await server._require_session()  # pyright: ignore[reportPrivateUsage]

        anyio.run(call_without_entering)

        assert server.implicit_sessions == 1

    def test_a_server_without_instructions_contributes_none(self) -> None:
        server = FakeMCPToolset(id='calc', instructions=None)
        agent = build(server)
        ctx = FakeDurableContext()

        result = run_durable(lambda: agent.run('add 2 and 3'), context=ctx)

        assert result.output == 'summed'
        assert 'calc__mcp_server__calc.get_instructions' in ctx.step_names


class TestEnqueueGuard:
    """Discovery is checkpointed too, so a message enqueued while resolving tools or instructions
    is dropped on replay just as silently as one enqueued from a tool call."""

    def test_enqueue_while_listing_tools_raises(self) -> None:
        class EnqueueingServer(FakeMCPToolset):
            async def get_tools(self, ctx: RunContext[object]) -> dict[str, ToolsetTool[object]]:
                ctx.enqueue('later')
                return await super().get_tools(ctx)  # pragma: no cover - the enqueue always raises

        agent = build(EnqueueingServer(id='calc', instructions='Use the calculator.'))
        ctx = FakeDurableContext()

        with pytest.raises(UserError, match='enqueue'):
            run_durable(lambda: agent.run('add 2 and 3'), context=ctx)

    def test_enqueue_while_fetching_instructions_raises(self) -> None:
        class EnqueueingServer(FakeMCPToolset):
            async def get_instructions(self, ctx: RunContext[object]) -> InstructionPart | None:
                ctx.enqueue('later')
                return await super().get_instructions(ctx)  # pragma: no cover - the enqueue always raises

        agent = build(EnqueueingServer(id='calc', instructions='Use the calculator.'))
        ctx = FakeDurableContext()

        with pytest.raises(UserError, match='enqueue'):
            run_durable(lambda: agent.run('add 2 and 3'), context=ctx)
