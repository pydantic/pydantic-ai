from __future__ import annotations

from typing import Any

import pytest

from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.experimental.mcp_server import mcp_server_from_toolsets

try:
    from fastmcp.client import Client
except ImportError:  # pragma: no cover
    pytest.skip('fastmcp not installed', allow_module_level=True)


def _math_toolset() -> FunctionToolset[dict[str, str]]:
    toolset = FunctionToolset[dict[str, str]]()

    @toolset.tool
    def double(ctx: RunContext[dict[str, str]], x: int) -> str:
        return ctx.deps['prefix'] + str(x * 2)

    return toolset


class TestMcpServer:
    async def test_tool_listed_with_schema(self):
        server = await mcp_server_from_toolsets(toolsets=[_math_toolset()], name='math')

        async with Client(server) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}

        assert 'double' in tools
        assert tools['double'].inputSchema == {
            'properties': {'x': {'type': 'integer'}},
            'required': ['x'],
            'type': 'object',
            'additionalProperties': False,
        }

    async def test_tool_call_returns_result(self):
        server = await mcp_server_from_toolsets(toolsets=[_math_toolset()], name='math', deps={'prefix': 'result: '})

        async with Client(server) as client:
            result = await client.call_tool('double', {'x': 21})

        assert result.content[0].text == 'result: 42'
        assert not result.is_error

    async def test_failing_tool_is_an_error_result(self):
        toolset = FunctionToolset()

        @toolset.tool
        def boom(ctx: RunContext[None]) -> str:
            raise ValueError('kaboom')

        server = await mcp_server_from_toolsets(toolsets=[toolset], name='failing')

        async with Client(server) as client:
            result = await client.call_tool('boom', {}, raise_on_error=False)

        assert result.is_error
        assert 'kaboom' in result.content[0].text

    async def test_deferred_and_non_function_tools_are_skipped(self):
        toolset = FunctionToolset()

        @toolset.tool(requires_approval=True)
        def dangerous(ctx: RunContext[None]) -> str:
            return 'nope'  # pragma: no cover

        @toolset.tool
        def plain(ctx: RunContext[None]) -> str:
            return 'fine'

        server = await mcp_server_from_toolsets(toolsets=[toolset], name='mixed')

        async with Client(server) as client:
            names = [tool.name for tool in await client.list_tools()]

        assert names == ['plain']

    async def test_model_placeholder_is_benign_metadata(self):
        """Without `model=`, ctx.model is a placeholder whose metadata is harmless and clearly named."""
        toolset = FunctionToolset()

        @toolset.tool
        def probe(ctx: RunContext[Any]) -> str:
            return ctx.model.system

        server = await mcp_server_from_toolsets(toolsets=[toolset], name='no-model')

        async with Client(server) as client:
            result = await client.call_tool('probe', {})

        assert not result.is_error
        assert result.content[0].text == 'mcp-server'
