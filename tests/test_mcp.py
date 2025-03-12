"""Tests for the MCP (Model Context Protocol) server implementation."""

import pytest
from dirty_equals import IsInstance
from inline_snapshot import snapshot

from pydantic_ai.agent import Agent
from pydantic_ai.mcp import MCPServer
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart

from .conftest import IsDatetime, try_import

with try_import() as imports_successful:
    from mcp.types import CallToolResult, TextContent


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='mcp not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


async def test_stdio_server():
    async with MCPServer.stdio('python', ['-m', 'tests.mcp_server']) as server:
        tools = await server.list_tools()
        assert len(tools) == 1
        assert tools[0].name == 'celsius_to_fahrenheit'
        assert tools[0].description.startswith('Convert Celsius to Fahrenheit.')

        # Test calling the temperature conversion tool
        result = await server.call_tool('celsius_to_fahrenheit', {'celsius': 0})
        assert result.content == snapshot([TextContent(type='text', text='32.0')])


def test_sse_server():
    sse_server = MCPServer.sse(url='http://localhost:8000/sse')
    assert sse_server.url == 'http://localhost:8000/sse'


async def test_agent_with_stdio_server(allow_model_requests: None):
    async with MCPServer.stdio('python', ['-m', 'tests.mcp_server']) as server:
        agent = Agent('openai:gpt-4o', mcp_servers=[server])
        result = await agent.run('What is 0 degrees Celsius in Fahrenheit?')
        assert result.data == snapshot('0 degrees Celsius is 32.0 degrees Fahrenheit.')
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='What is 0 degrees Celsius in Fahrenheit?', timestamp=IsDatetime())]
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='celsius_to_fahrenheit',
                            args='{"celsius":0}',
                            tool_call_id='call_UNesABTXfwIkYdh3HzXWw2wD',
                        )
                    ],
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='celsius_to_fahrenheit',
                            content=IsInstance(CallToolResult),
                            tool_call_id='call_UNesABTXfwIkYdh3HzXWw2wD',
                            timestamp=IsDatetime(),
                        )
                    ]
                ),
                ModelResponse(
                    parts=[TextPart(content='0 degrees Celsius is 32.0 degrees Fahrenheit.')],
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                ),
            ]
        )
