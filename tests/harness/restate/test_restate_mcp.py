"""MCP durable-wrapping tests for `RestateDurability`.

The MCP extra may be absent, so the module `importorskip`s it. A lightweight `FakeMCPToolset`
stands in for a real server: it is a genuine `MCPToolset` subclass (so the capability's
`isinstance` wrapping and `tool_for_tool_def` rebuild apply) whose wire methods return in-memory
results, which keeps the test off Docker and the network.
"""

from __future__ import annotations

import pytest

pytest.importorskip('restate')
pytest.importorskip('pydantic_ai.mcp')

from typing import Any

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
from pydantic_ai_harness.restate import RestateDurability

from .conftest import FakeRestateContext, restate_context

pytestmark = pytest.mark.anyio

_ADD_SCHEMA = {
    'type': 'object',
    'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
    'required': ['a', 'b'],
}


class FakeMCPToolset(MCPToolset[object]):
    """In-memory `MCPToolset` whose I/O methods return canned results.

    Bypasses `MCPToolset.__init__` (which would build a real transport) and sets only the attributes
    the durable wrapper and the run touch.
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
        """Model a real server: I/O needs an active session, and a call without one opens its own
        implicit session for the duration of the call, as `MCPToolset` does."""
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
        if not self.include_instructions or self._instructions_text is None:
            return None
        return InstructionPart(content=self._instructions_text)

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[object], tool: ToolsetTool[object]
    ) -> int:
        await self._require_session()
        self.tool_calls.append((name, dict(tool_args)))
        return int(tool_args['a']) + int(tool_args['b'])


def _add_then_done_model() -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answered = any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)
        if not answered:
            return ModelResponse(parts=[ToolCallPart(tool_name='add', args={'a': 2, 'b': 3})])
        return ModelResponse(parts=[TextPart(content='summed')])

    return FunctionModel(fn, model_name='fn')


class TestMcpCheckpointing:
    async def test_get_tools_get_instructions_and_call_tool_journaled(self) -> None:
        server = FakeMCPToolset(id='calc', instructions='Use the calculator.')
        agent = Agent(_add_then_done_model(), name='calc', toolsets=[server], capabilities=[RestateDurability()])

        ctx = FakeRestateContext()
        with restate_context(ctx):
            result = await agent.run('add 2 and 3')

        assert result.output == 'summed'
        assert server.tool_calls == [('add', {'a': 2, 'b': 3})]
        assert 'calc__mcp_server__calc.get_tools' in ctx.step_names
        assert 'calc__mcp_server__calc.get_instructions' in ctx.step_names
        assert 'calc__mcp_server__calc.call_tool' in ctx.step_names

    async def test_replay_does_not_rehit_server(self) -> None:
        server = FakeMCPToolset(id='calc', instructions='Use the calculator.')
        agent = Agent(_add_then_done_model(), name='calc', toolsets=[server], capabilities=[RestateDurability()])

        ctx = FakeRestateContext()
        with restate_context(ctx):
            first = await agent.run('add 2 and 3')

        replay = ctx.replay()
        with restate_context(replay):
            second = await agent.run('add 2 and 3')

        assert first.output == second.output == 'summed'
        assert server.tool_calls == [('add', {'a': 2, 'b': 3})]
        assert replay.invoked == []

    async def test_instructions_omitted_when_disabled(self) -> None:
        server = FakeMCPToolset(id='calc', instructions='Use it.', include_instructions=False)
        agent = Agent(_add_then_done_model(), name='calc', toolsets=[server], capabilities=[RestateDurability()])

        ctx = FakeRestateContext()
        with restate_context(ctx):
            await agent.run('add 2 and 3')

        assert 'calc__mcp_server__calc.get_instructions' not in ctx.step_names


class TestMultipleServers:
    """`CombinedToolset` lists toolsets concurrently, so two servers issue sibling step requests.

    Sibling steps must each claim their own journal slot, not be mis-rejected as if nested.
    """

    async def test_two_mcp_servers_are_listed_and_replay(self) -> None:
        first = FakeMCPToolset(id='s1', instructions='One.')
        second = FakeMCPToolset(id='s2', instructions='Two.', tool_name='add2')
        agent = Agent(_add_then_done_model(), name='calc', toolsets=[first, second], capabilities=[RestateDurability()])

        ctx = FakeRestateContext()
        with restate_context(ctx):
            result = await agent.run('add 2 and 3')

        assert result.output == 'summed'
        assert 'calc__mcp_server__s1.get_tools' in ctx.step_names
        assert 'calc__mcp_server__s2.get_tools' in ctx.step_names

        # A replay must line up with the recorded journal, which only holds if sibling steps
        # claimed their slots in a stable order.
        replay = ctx.replay()
        with restate_context(replay):
            replayed = await agent.run('add 2 and 3')

        assert replayed.output == 'summed'
        assert replay.invoked == []


class TestMcpSessionLifecycle:
    async def test_wrapper_holds_one_session_no_implicit_per_call(self) -> None:
        server = FakeMCPToolset(id='calc', instructions='Use the calculator.')
        agent = Agent(_add_then_done_model(), name='calc', toolsets=[server], capabilities=[RestateDurability()])

        ctx = FakeRestateContext()
        with restate_context(ctx):
            result = await agent.run('add 2 and 3')

        assert result.output == 'summed'
        assert server.implicit_sessions == 0
        assert server.enter_count >= 1

    async def test_transparent_run_also_reuses_one_session(self) -> None:
        server = FakeMCPToolset(id='calc', instructions='Use the calculator.')
        agent = Agent(_add_then_done_model(), name='calc', toolsets=[server], capabilities=[RestateDurability()])

        result = await agent.run('add 2 and 3')

        assert result.output == 'summed'
        assert server.implicit_sessions == 0


class TestFakeServerModelsImplicitSessions:
    async def test_io_without_an_open_session_opens_an_implicit_one(self) -> None:
        # Proves the fake models a real `MCPToolset`: I/O with no session open opens a transient
        # implicit one, so the `implicit_sessions == 0` assertions above are meaningful.
        server = FakeMCPToolset(id='calc')
        await server._require_session()
        assert server.implicit_sessions == 1

        async with server:
            await server._require_session()
        assert server.implicit_sessions == 1


class TestMcpInlineOptOutForbidden:
    async def test_metadata_false_on_mcp_tool_raises(self) -> None:
        server = FakeMCPToolset(id='calc', tool_metadata={'restate': False})
        agent = Agent(_add_then_done_model(), name='calc', toolsets=[server], capabilities=[RestateDurability()])

        ctx = FakeRestateContext()
        with restate_context(ctx):
            with pytest.raises(UserError, match='cannot run outside a durable step'):
                await agent.run('add 2 and 3')

    async def test_non_empty_dict_config_on_mcp_tool_raises(self) -> None:
        server = FakeMCPToolset(id='calc', tool_metadata={'restate': {'retries': 3}})
        agent = Agent(_add_then_done_model(), name='calc', toolsets=[server], capabilities=[RestateDurability()])

        ctx = FakeRestateContext()
        with restate_context(ctx):
            with pytest.raises(UserError, match='take no per-tool options'):
                await agent.run('add 2 and 3')
