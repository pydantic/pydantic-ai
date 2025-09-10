"""Tests for MCP tool metadata support and filtering."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage

from .conftest import try_import

with try_import() as imports_successful:
    from mcp import types as mcp_types

    from pydantic_ai.mcp import MCPServer

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='mcp not installed'),
    pytest.mark.anyio,
]


class MockMCPServer(MCPServer):
    """Mock MCP server for testing metadata functionality."""

    def __init__(self, mock_tools: list[mcp_types.Tool], **kwargs: Any):
        super().__init__(**kwargs)
        self._mock_tools = mock_tools

    @asynccontextmanager
    async def client_streams(self) -> AsyncIterator[tuple[Any, Any]]:
        """Not used in these tests."""
        raise NotImplementedError('Mock server does not implement streams')
        # This is unreachable but needed for type checking
        yield None, None  # pragma: no cover

    async def list_tools(self) -> list[mcp_types.Tool]:
        """Return mock tools with metadata."""
        return self._mock_tools

    async def direct_call_tool(self, name: str, args: dict[str, Any], metadata: dict[str, Any] | None = None):
        """Mock tool call - not used in metadata tests."""
        return f'Called {name} with {args}'


async def test_tool_metadata_extraction():
    """Test that MCP tool metadata is properly extracted into ToolDefinition."""
    # Create mock tools with different metadata
    mock_tools = [
        mcp_types.Tool(
            name='simple_tool',
            description='A simple tool without metadata',
            inputSchema={'type': 'object', 'properties': {}},
            _meta=None,
        ),
        mcp_types.Tool(
            name='complex_tool',
            description='A complex tool with metadata',
            inputSchema={'type': 'object', 'properties': {'query': {'type': 'string'}}},
            _meta={'complexity': 'high', 'category': 'analysis'},
        ),
        mcp_types.Tool(
            name='auth_tool',
            description='Tool with authentication metadata',
            inputSchema={'type': 'object', 'properties': {}},
            _meta={'complexity': 'low', '_fastmcp': {'tags': ['unauthenticated', 'public']}},
        ),
    ]

    server = MockMCPServer(mock_tools)
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())

    tools = await server.get_tools(ctx)

    # Check that we have all tools
    assert len(tools) == 3
    assert 'simple_tool' in tools
    assert 'complex_tool' in tools
    assert 'auth_tool' in tools

    # Check metadata extraction
    simple_tool = tools['simple_tool'].tool_def
    assert simple_tool.name == 'simple_tool'
    assert simple_tool.metadata is None

    complex_tool = tools['complex_tool'].tool_def
    assert complex_tool.name == 'complex_tool'
    assert complex_tool.metadata == {'complexity': 'high', 'category': 'analysis'}

    auth_tool = tools['auth_tool'].tool_def
    assert auth_tool.name == 'auth_tool'
    assert auth_tool.metadata == {'complexity': 'low', '_fastmcp': {'tags': ['unauthenticated', 'public']}}


async def test_tool_filtering_by_complexity():
    """Test filtering tools by complexity metadata."""
    mock_tools = [
        mcp_types.Tool(
            name='simple_tool',
            description='Simple tool',
            inputSchema={'type': 'object'},
            _meta={'complexity': 'low'},
        ),
        mcp_types.Tool(
            name='complex_tool',
            description='Complex tool',
            inputSchema={'type': 'object'},
            _meta={'complexity': 'high'},
        ),
        mcp_types.Tool(
            name='no_meta_tool',
            description='Tool without metadata',
            inputSchema={'type': 'object'},
            _meta=None,
        ),
    ]

    server = MockMCPServer(mock_tools)

    # Filter to exclude high complexity tools
    def complexity_filter(ctx: RunContext, tool_def: ToolDefinition) -> bool:
        if tool_def.metadata is None:
            return True  # Include tools without metadata
        return tool_def.metadata.get('complexity') != 'high'

    filtered_server = server.filtered(complexity_filter)
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())

    filtered_tools = await filtered_server.get_tools(ctx)

    # Should only have simple_tool and no_meta_tool
    assert len(filtered_tools) == 2
    assert 'simple_tool' in filtered_tools
    assert 'no_meta_tool' in filtered_tools
    assert 'complex_tool' not in filtered_tools


async def test_tool_filtering_by_tags():
    """Test filtering tools by FastMCP tags."""
    mock_tools = [
        mcp_types.Tool(
            name='public_tool',
            description='Public tool',
            inputSchema={'type': 'object'},
            _meta={'_fastmcp': {'tags': ['unauthenticated', 'public']}},
        ),
        mcp_types.Tool(
            name='private_tool',
            description='Private tool',
            inputSchema={'type': 'object'},
            _meta={'_fastmcp': {'tags': ['authenticated', 'private']}},
        ),
        mcp_types.Tool(
            name='no_tags_tool',
            description='Tool without tags',
            inputSchema={'type': 'object'},
            _meta={'some_other': 'metadata'},
        ),
    ]

    server = MockMCPServer(mock_tools)

    # Filter to only include unauthenticated tools
    def unauthenticated_filter(ctx: RunContext, tool_def: ToolDefinition) -> bool:
        if tool_def.metadata is None:
            return False
        fastmcp_meta = tool_def.metadata.get('_fastmcp', {})
        tags = fastmcp_meta.get('tags', [])
        return 'unauthenticated' in tags

    filtered_server = server.filtered(unauthenticated_filter)
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())

    filtered_tools = await filtered_server.get_tools(ctx)

    # Should only have public_tool
    assert len(filtered_tools) == 1
    assert 'public_tool' in filtered_tools
    assert 'private_tool' not in filtered_tools
    assert 'no_tags_tool' not in filtered_tools


async def test_tool_prefix_with_metadata():
    """Test that tool prefixing works correctly with metadata."""
    mock_tools = [
        mcp_types.Tool(
            name='original_tool',
            description='Tool with metadata',
            inputSchema={'type': 'object'},
            _meta={'category': 'test'},
        ),
    ]

    server = MockMCPServer(mock_tools, tool_prefix='prefix')
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())

    tools = await server.get_tools(ctx)

    # Tool should be prefixed but metadata preserved
    assert len(tools) == 1
    assert 'prefix_original_tool' in tools

    tool_def = tools['prefix_original_tool'].tool_def
    assert tool_def.name == 'prefix_original_tool'
    assert tool_def.metadata == {'category': 'test'}


async def test_combined_filtering():
    """Test combining multiple filter criteria."""
    mock_tools = [
        mcp_types.Tool(
            name='simple_public',
            description='Simple public tool',
            inputSchema={'type': 'object'},
            _meta={'complexity': 'low', '_fastmcp': {'tags': ['unauthenticated']}},
        ),
        mcp_types.Tool(
            name='complex_public',
            description='Complex public tool',
            inputSchema={'type': 'object'},
            _meta={'complexity': 'high', '_fastmcp': {'tags': ['unauthenticated']}},
        ),
        mcp_types.Tool(
            name='simple_private',
            description='Simple private tool',
            inputSchema={'type': 'object'},
            _meta={'complexity': 'low', '_fastmcp': {'tags': ['authenticated']}},
        ),
    ]

    server = MockMCPServer(mock_tools)

    # Filter for simple AND unauthenticated tools
    def combined_filter(ctx: RunContext, tool_def: ToolDefinition) -> bool:
        if tool_def.metadata is None:
            return False

        # Check complexity
        if tool_def.metadata.get('complexity') != 'low':
            return False

        # Check tags
        fastmcp_meta = tool_def.metadata.get('_fastmcp', {})
        tags = fastmcp_meta.get('tags', [])
        if 'unauthenticated' not in tags:
            return False

        return True

    filtered_server = server.filtered(combined_filter)
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())

    filtered_tools = await filtered_server.get_tools(ctx)

    # Should only have simple_public
    assert len(filtered_tools) == 1
    assert 'simple_public' in filtered_tools


async def test_empty_metadata():
    """Test handling of empty metadata."""
    mock_tools = [
        mcp_types.Tool(
            name='empty_meta_tool',
            description='Tool with empty metadata dict',
            inputSchema={'type': 'object'},
            _meta={},
        ),
    ]

    server = MockMCPServer(mock_tools)
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())

    tools = await server.get_tools(ctx)

    assert len(tools) == 1
    tool_def = tools['empty_meta_tool'].tool_def
    assert tool_def.metadata == {}
