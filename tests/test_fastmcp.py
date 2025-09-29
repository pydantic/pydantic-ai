"""Tests for the FastMCP Toolset implementation."""

from __future__ import annotations

import base64
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from fastmcp.client import PythonStdioTransport, SSETransport
from fastmcp.client.transports import ClientTransport
from inline_snapshot import snapshot

from pydantic_ai._run_context import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import BinaryContent
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from .conftest import try_import

with try_import() as imports_successful:
    from fastmcp.client import Client
    from fastmcp.client.transports import (
        FastMCPTransport,
        MCPConfigTransport,
        NodeStdioTransport,
        StdioTransport,
        StreamableHttpTransport,
    )
    from fastmcp.exceptions import ToolError
    from fastmcp.server.server import FastMCP
    from mcp.types import (
        AudioContent,
        ImageContent,
        TextContent,
    )

    # Import the content mapping functions for testing
    from pydantic_ai.toolsets.fastmcp import (
        FastMCPToolset,
    )


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='fastmcp not installed'),
    pytest.mark.anyio,
]


def get_client_from_toolset(toolset: FastMCPToolset[None]) -> Client[FastMCPTransport]:
    """Get the client from the toolset."""
    return toolset._client  # type: ignore


def get_transport_from_toolset(toolset: FastMCPToolset[None]) -> ClientTransport:
    """Get the transport from the toolset."""
    return get_client_from_toolset(toolset).transport


@pytest.fixture
async def fastmcp_server() -> FastMCP:
    """Create a real in-memory FastMCP server for testing."""
    server = FastMCP('test_server')

    @server.tool()
    async def test_tool(param1: str, param2: int = 0) -> str:
        """A test tool that returns a formatted string."""
        return f'param1={param1}, param2={param2}'

    @server.tool()
    async def another_tool(value: float) -> dict[str, Any]:
        """Another test tool that returns structured data."""
        return {'result': 'success', 'value': value, 'doubled': value * 2}

    @server.tool()
    async def error_tool() -> str:
        """A tool that can fail for testing error handling."""
        raise ValueError('This is a test error')

    @server.tool()
    async def binary_tool() -> ImageContent:
        """A tool that returns binary content."""
        fake_image_data = b'fake_image_data'
        encoded_data = base64.b64encode(fake_image_data).decode('utf-8')
        return ImageContent(type='image', data=encoded_data, mimeType='image/png')

    @server.tool()
    async def audio_tool() -> AudioContent:
        """A tool that returns audio content."""
        fake_audio_data = b'fake_audio_data'
        encoded_data = base64.b64encode(fake_audio_data).decode('utf-8')
        return AudioContent(type='audio', data=encoded_data, mimeType='audio/mpeg')

    @server.tool()
    async def text_tool(message: str) -> str:
        """A tool that returns text content."""
        return f'Echo: {message}'

    @server.tool()
    async def text_list_tool(message: str) -> list[TextContent]:
        """A tool that returns text content without a return annotation."""
        return [
            TextContent(type='text', text=f'Echo: {message}'),
            TextContent(type='text', text=f'Echo: {message} again'),
        ]

    @server.tool()
    async def text_tool_wo_return_annotation(message: str):
        """A tool that returns text content."""
        return f'Echo: {message}'

    @server.tool()
    async def json_tool(data: dict[str, Any]) -> str:
        """A tool that returns JSON data."""
        import json

        return json.dumps({'received': data, 'processed': True})

    return server


@pytest.fixture
async def fastmcp_client(fastmcp_server: FastMCP) -> Client[FastMCPTransport]:
    """Create a real FastMCP client connected to the test server."""
    return Client(transport=fastmcp_server)


@pytest.fixture
def run_context() -> RunContext[None]:
    """Create a run context for testing."""
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )


class TestFastMCPToolsetInitialization:
    """Test FastMCP Toolset initialization and basic functionality."""

    async def test_init_with_client(self, fastmcp_client: Client[FastMCPTransport]):
        """Test initialization with a FastMCP client."""
        toolset = FastMCPToolset(mcp=fastmcp_client)

        # Test that the client is accessible via the property
        assert toolset.id is None

    async def test_init_with_id(self, fastmcp_client: Client[FastMCPTransport]):
        """Test initialization with an id."""
        toolset = FastMCPToolset(fastmcp_client, _id='test_id')

        # Test that the client is accessible via the property
        assert toolset.id == 'test_id'

    async def test_init_with_custom_retries_and_error_behavior(self, fastmcp_client: Client[FastMCPTransport]):
        """Test initialization with custom retries and error behavior."""
        toolset = FastMCPToolset(fastmcp_client, max_retries=5, tool_error_behavior='model_retry')

        # Test that the toolset was created successfully
        assert toolset._client is fastmcp_client  # type: ignore

    async def test_id_property(self, fastmcp_client: Client[FastMCPTransport]):
        """Test that the id property returns None."""
        toolset = FastMCPToolset(fastmcp_client)
        assert toolset.id is None


class TestFastMCPToolsetContextManagement:
    """Test FastMCP Toolset context management."""

    async def test_context_manager_single_enter_exit(
        self, fastmcp_client: Client[FastMCPTransport], run_context: RunContext[None]
    ):
        """Test single enter/exit cycle."""
        toolset = FastMCPToolset(fastmcp_client)

        async with toolset:
            # Test that we can get tools when the context is active
            tools = await toolset.get_tools(run_context)
            assert len(tools) > 0
            assert 'test_tool' in tools

        # After exit, the toolset should still be usable but the client connection is closed

    async def test_context_manager_no_enter(
        self, fastmcp_client: Client[FastMCPTransport], run_context: RunContext[None]
    ):
        """Test no enter/exit cycle."""
        toolset = FastMCPToolset(fastmcp_client)

        # Test that we can get tools when the context is not active
        tools = await toolset.get_tools(run_context)
        assert len(tools) > 0
        assert 'test_tool' in tools

    async def test_context_manager_nested_enter_exit(
        self, fastmcp_client: Client[FastMCPTransport], run_context: RunContext[None]
    ):
        """Test nested enter/exit cycles."""
        toolset = FastMCPToolset(fastmcp_client)

        async with toolset:
            tools1 = await toolset.get_tools(run_context)
            async with toolset:
                tools2 = await toolset.get_tools(run_context)
                assert tools1 == tools2
            # Should still work after inner context exits
            tools3 = await toolset.get_tools(run_context)
            assert tools1 == tools3


class TestFastMCPToolsetToolDiscovery:
    """Test FastMCP Toolset tool discovery functionality."""

    async def test_get_tools(
        self,
        fastmcp_client: Client[FastMCPTransport],
        run_context: RunContext[None],
    ):
        """Test getting tools from the FastMCP client."""
        toolset = FastMCPToolset(fastmcp_client)

        async with toolset:
            tools = await toolset.get_tools(run_context)

            # Should have all the tools we defined in the server
            expected_tools = {
                'test_tool',
                'another_tool',
                'audio_tool',
                'error_tool',
                'binary_tool',
                'text_tool',
                'text_list_tool',
                'text_tool_wo_return_annotation',
                'json_tool',
            }
            assert set(tools.keys()) == expected_tools

            # Check tool definitions
            test_tool = tools['test_tool']
            assert test_tool.tool_def.name == 'test_tool'
            assert test_tool.tool_def.description is not None
            assert 'test tool that returns a formatted string' in test_tool.tool_def.description
            assert test_tool.max_retries == 2
            assert test_tool.toolset is toolset

            # Check that the tool has proper schema
            schema = test_tool.tool_def.parameters_json_schema
            assert schema['type'] == 'object'
            assert 'param1' in schema['properties']
            assert 'param2' in schema['properties']

    async def test_get_tools_with_empty_server(self, run_context: RunContext[None]):
        """Test getting tools from an empty FastMCP server."""
        empty_server = FastMCP('empty_server')
        empty_client = Client(transport=empty_server)
        toolset = FastMCPToolset(empty_client)

        async with toolset:
            tools = await toolset.get_tools(run_context)
            assert len(tools) == 0


class TestFastMCPToolsetToolCalling:
    """Test FastMCP Toolset tool calling functionality."""

    @pytest.fixture
    async def fastmcp_toolset(self, fastmcp_client: Client[FastMCPTransport]) -> FastMCPToolset[None]:
        """Create a FastMCP Toolset."""
        return FastMCPToolset(fastmcp_client)

    async def test_call_tool_success(
        self,
        fastmcp_toolset: FastMCPToolset[None],
        run_context: RunContext[None],
    ):
        """Test successful tool call."""
        async with fastmcp_toolset:
            tools = await fastmcp_toolset.get_tools(run_context)
            test_tool = tools['test_tool']

            result = await fastmcp_toolset.call_tool(
                name='test_tool', tool_args={'param1': 'hello', 'param2': 42}, ctx=run_context, tool=test_tool
            )

            assert result == {'result': 'param1=hello, param2=42'}

    async def test_call_tool_with_structured_content(
        self,
        fastmcp_toolset: FastMCPToolset[None],
        run_context: RunContext[None],
    ):
        """Test tool call with structured content."""
        async with fastmcp_toolset:
            tools = await fastmcp_toolset.get_tools(run_context)
            another_tool = tools['another_tool']

            result = await fastmcp_toolset.call_tool(
                name='another_tool', tool_args={'value': 3.14}, ctx=run_context, tool=another_tool
            )

            assert result == {'result': 'success', 'value': 3.14, 'doubled': 6.28}

    async def test_call_tool_with_binary_content(
        self,
        fastmcp_toolset: FastMCPToolset[None],
        run_context: RunContext[None],
    ):
        """Test tool call that returns binary content."""
        async with fastmcp_toolset:
            tools = await fastmcp_toolset.get_tools(run_context)
            binary_tool = tools['binary_tool']

            result = await fastmcp_toolset.call_tool(
                name='binary_tool', tool_args={}, ctx=run_context, tool=binary_tool
            )

            assert result == snapshot(
                BinaryContent(data=b'fake_image_data', media_type='image/png', identifier='427d68')
            )

    async def test_call_tool_with_audio_content(
        self,
        fastmcp_toolset: FastMCPToolset[None],
        run_context: RunContext[None],
    ):
        """Test tool call that returns audio content."""
        async with fastmcp_toolset:
            tools = await fastmcp_toolset.get_tools(run_context)
            audio_tool = tools['audio_tool']

            result = await fastmcp_toolset.call_tool(name='audio_tool', tool_args={}, ctx=run_context, tool=audio_tool)

            assert result == snapshot(
                BinaryContent(data=b'fake_audio_data', media_type='audio/mpeg', identifier='f1220f')
            )

    async def test_call_tool_with_text_content(
        self,
        fastmcp_toolset: FastMCPToolset[None],
        run_context: RunContext[None],
    ):
        """Test tool call that returns text content."""
        async with fastmcp_toolset:
            tools = await fastmcp_toolset.get_tools(run_context)
            text_tool = tools['text_tool']

            result = await fastmcp_toolset.call_tool(
                name='text_tool', tool_args={'message': 'Hello World'}, ctx=run_context, tool=text_tool
            )

            assert result == snapshot({'result': 'Echo: Hello World'})

            text_list_tool = tools['text_list_tool']

            result = await fastmcp_toolset.call_tool(
                name='text_list_tool', tool_args={'message': 'Hello World'}, ctx=run_context, tool=text_list_tool
            )

            assert result == snapshot(['Echo: Hello World', 'Echo: Hello World again'])

    async def test_call_tool_with_unknown_text_content(
        self,
        fastmcp_toolset: FastMCPToolset[None],
        run_context: RunContext[None],
    ):
        """Test tool call that returns text content."""
        async with fastmcp_toolset:
            tools = await fastmcp_toolset.get_tools(run_context)
            text_tool = tools['text_tool_wo_return_annotation']

            result = await fastmcp_toolset.call_tool(
                name='text_tool_wo_return_annotation',
                tool_args={'message': 'Hello World'},
                ctx=run_context,
                tool=text_tool,
            )

            assert result == snapshot('Echo: Hello World')

    async def test_call_tool_with_json_content(
        self,
        fastmcp_toolset: FastMCPToolset[None],
        run_context: RunContext[None],
    ):
        """Test tool call that returns JSON content."""
        async with fastmcp_toolset:
            tools = await fastmcp_toolset.get_tools(run_context)
            json_tool = tools['json_tool']

            result = await fastmcp_toolset.call_tool(
                name='json_tool', tool_args={'data': {'key': 'value'}}, ctx=run_context, tool=json_tool
            )

            # Should parse the JSON string into a dict
            assert result == snapshot({'result': '{"received": {"key": "value"}, "processed": true}'})

    async def test_call_tool_with_error_behavior_raise(
        self,
        fastmcp_toolset: FastMCPToolset[None],
        run_context: RunContext[None],
    ):
        """Test tool call with error behavior set to raise."""
        async with fastmcp_toolset:
            tools = await fastmcp_toolset.get_tools(run_context)
            error_tool = tools['error_tool']

            with pytest.raises(ToolError, match='This is a test error'):
                await fastmcp_toolset.call_tool(name='error_tool', tool_args={}, ctx=run_context, tool=error_tool)

    async def test_call_tool_with_error_behavior_model_retry(
        self,
        fastmcp_client: Client[FastMCPTransport],
        run_context: RunContext[None],
    ):
        """Test tool call with error behavior set to model retry."""
        toolset = FastMCPToolset(fastmcp_client, tool_error_behavior='model_retry')

        async with toolset:
            tools = await toolset.get_tools(run_context)
            error_tool = tools['error_tool']

            with pytest.raises(ModelRetry, match='This is a test error'):
                await toolset.call_tool('error_tool', {}, run_context, error_tool)


class TestFastMCPToolsetFactoryMethods:
    """Test FastMCP Toolset factory methods."""

    async def test_python_stdio(self, run_context: RunContext[None]):
        """Test creating toolset from FastMCP server."""
        server_script = """
from fastmcp import FastMCP

server = FastMCP('test_server')

@server.tool()
async def test_tool(param1: str, param2: int = 0) -> str:
    return f'param1={param1}, param2={param2}'

server.run()"""
        with TemporaryDirectory() as temp_dir:
            server_py = Path(temp_dir) / 'server.py'
            server_py.write_text(server_script)
            toolset = FastMCPToolset(mcp=server_py)

            assert isinstance(toolset, FastMCPToolset)
            assert toolset.id is None

            async with toolset:
                tools = await toolset.get_tools(run_context)
                assert 'test_tool' in tools

    async def test_transports(self):
        """Test creating toolset from different transports."""
        toolset = FastMCPToolset(mcp='http://localhost:8000/mcp')
        assert isinstance(get_transport_from_toolset(toolset), StreamableHttpTransport)

        toolset = FastMCPToolset(mcp='http://localhost:8000/sse')
        assert isinstance(get_transport_from_toolset(toolset), SSETransport)

        toolset = FastMCPToolset(mcp=StdioTransport(command='python', args=['-c', 'print("test")']))
        assert isinstance(get_transport_from_toolset(toolset), StdioTransport)

        with TemporaryDirectory() as temp_dir:
            server_py: Path = Path(temp_dir) / 'server.py'
            server_py.write_text(data='')
            toolset = FastMCPToolset(mcp=server_py)
            assert isinstance(get_transport_from_toolset(toolset), PythonStdioTransport)
            toolset = FastMCPToolset(mcp=str(server_py))
            assert isinstance(get_transport_from_toolset(toolset), PythonStdioTransport)

            server_js: Path = Path(temp_dir) / 'server.js'
            server_js.write_text(data='')
            toolset = FastMCPToolset(mcp=server_js)
            assert isinstance(get_transport_from_toolset(toolset), NodeStdioTransport)
            toolset = FastMCPToolset(mcp=str(server_js))
            assert isinstance(get_transport_from_toolset(toolset), NodeStdioTransport)

        toolset = FastMCPToolset(
            mcp={'mcpServers': {'test_server': {'command': 'python', 'args': ['-c', 'print("test")']}}}
        )
        assert isinstance(get_transport_from_toolset(toolset), MCPConfigTransport)

    #     async def test_sse_transport(self, run_context: RunContext[None]):
    #         """Test creating toolset from stdio transport."""
    #         server_script = """
    # from fastmcp import FastMCP

    # server = FastMCP('test_server')

    # @server.tool()
    # async def test_tool(param1: str, param2: int = 0) -> str:
    #     return f'param1={param1}, param2={param2}'

    # server.run(transport='sse')"""
    #         with TemporaryDirectory() as temp_dir:
    #             server_py = Path(temp_dir) / 'server.py'
    #             server_py.write_text(server_script)

    #             with subprocess.Popen(['python', server_py], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True):
    #                 toolset = FastMCPToolset(mcp='http://localhost:8000/sse')
    #                 async with toolset:
    #                     tools = await toolset.get_tools(
    #                         RunContext(deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0)
    #                     )
    #                     assert 'test_tool' in tools

    #     async def test_streamable_http_transport(self, run_context: RunContext[None]):
    #         """Test creating toolset from stdio transport."""

    #         toolset = FastMCPToolset(mcp='http://localhost:8001/mcp')

    #         assert isinstance(get_client_from_toolset(toolset).transport, StreamableHttpTransport)
    #         server_script = """
    # from fastmcp import FastMCP
    # import asyncio

    # server = FastMCP('test_server')

    # @server.tool()
    # async def test_tool(param1: str, param2: int = 0) -> str:
    #     return f'param1={param1}, param2={param2}'

    # asyncio.run(server.run_streamable_http_async(host='localhost', port=8001))
    # """
    #         with TemporaryDirectory() as temp_dir:
    #             server_py = Path(temp_dir) / 'server.py'
    #             server_py.write_text(server_script)

    #             with subprocess.Popen(['python', server_py], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True):
    #                 toolset = FastMCPToolset(mcp='http://localhost:8001/mcp')
    #                 async with toolset:
    #                     tools = await toolset.get_tools(run_context)
    #                     assert 'test_tool' in tools

    async def test_in_memory_transport(self, run_context: RunContext[None]):
        """Test creating toolset from stdio transport."""
        fastmcp_server = FastMCP('test_server')

        @fastmcp_server.tool()
        def test_tool(param1: str, param2: int = 0) -> str:
            return f'param1={param1}, param2={param2}'

        toolset = FastMCPToolset(mcp=fastmcp_server)
        async with toolset:
            tools = await toolset.get_tools(run_context)
            assert 'test_tool' in tools

    async def test_from_mcp_config_dict(self):
        """Test creating toolset from MCP config dictionary."""

        config_dict = {'mcpServers': {'test_server': {'command': 'python', 'args': ['-c', 'print("test")']}}}

        toolset = FastMCPToolset(mcp=config_dict)
        client = toolset._client  # type: ignore
        assert isinstance(client.transport, MCPConfigTransport)
