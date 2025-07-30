"""Tests for MCP elicitation callback functionality."""

from __future__ import annotations

from typing import Any

import pytest

from .conftest import try_import

with try_import() as imports_successful:
    from mcp import types as mcp_types
    from mcp.client.session import ClientSession
    from mcp.shared.context import RequestContext

    from pydantic_ai.mcp import MCPServerSSE, MCPServerStdio

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='mcp not installed'),
    pytest.mark.anyio,
]


class TestMCPElicitationCallback:
    """Test MCP elicitation callback functionality."""

    async def test_elicitation_callback_default_values(self):
        """Test that default values for elicitation are correct."""
        server = MCPServerStdio(command='python', args=['-c', 'print("test")'])

        assert server.allow_elicitation is True
        assert server.elicitation_callback is None

    async def test_elicitation_callback_stdio_server(self):
        """Test elicitation callback can be set on MCPServerStdio."""

        async def mock_callback(
            context: RequestContext[ClientSession, Any], params: mcp_types.ElicitRequestParams
        ) -> mcp_types.ElicitResult | mcp_types.ErrorData:
            return mcp_types.ElicitResult(action='accept', content={'result': 'test'})

        server = MCPServerStdio(command='python', args=['-c', 'print("test")'], elicitation_callback=mock_callback)

        assert server.elicitation_callback is mock_callback
        assert server.allow_elicitation is True

    async def test_elicitation_callback_sse_server(self):
        """Test elicitation callback can be set on MCPServerSSE."""

        async def mock_callback(
            context: RequestContext[ClientSession, Any], params: mcp_types.ElicitRequestParams
        ) -> mcp_types.ElicitResult | mcp_types.ErrorData:
            return mcp_types.ElicitResult(action='decline', content={})

        server = MCPServerSSE(url='http://localhost:3001/sse', elicitation_callback=mock_callback)

        assert server.elicitation_callback is mock_callback
        assert server.allow_elicitation is True

    async def test_allow_elicitation_false(self):
        """Test that allow_elicitation can be disabled."""

        async def mock_callback(
            context: RequestContext[ClientSession, Any], params: mcp_types.ElicitRequestParams
        ) -> mcp_types.ElicitResult | mcp_types.ErrorData:
            return mcp_types.ElicitResult(action='accept', content={'result': 'test'})

        server = MCPServerStdio(
            command='python', args=['-c', 'print("test")'], elicitation_callback=mock_callback, allow_elicitation=False
        )

        assert server.elicitation_callback is mock_callback
        assert server.allow_elicitation is False

    async def test_elicitation_callback_conditional_logic(self):
        """Test the conditional logic for when elicitation callback should be used."""

        async def mock_callback(
            context: RequestContext[ClientSession, Any], params: mcp_types.ElicitRequestParams
        ) -> mcp_types.ElicitResult | mcp_types.ErrorData:
            return mcp_types.ElicitResult(action='accept', content={'result': 'test'})

        # Test: callback provided + elicitation enabled = callback should be used
        server1 = MCPServerStdio(
            command='python', args=['-c', 'print("test")'], elicitation_callback=mock_callback, allow_elicitation=True
        )
        expected1 = server1.elicitation_callback if server1.allow_elicitation else None
        assert expected1 is mock_callback

        # Test: callback provided + elicitation disabled = callback should be None
        server2 = MCPServerStdio(
            command='python', args=['-c', 'print("test")'], elicitation_callback=mock_callback, allow_elicitation=False
        )
        expected2 = server2.elicitation_callback if server2.allow_elicitation else None
        assert expected2 is None

        # Test: no callback provided + elicitation enabled = None
        server3 = MCPServerStdio(command='python', args=['-c', 'print("test")'], allow_elicitation=True)
        expected3 = server3.elicitation_callback if server3.allow_elicitation else None
        assert expected3 is None

    async def test_elicitation_callback_signature(self):
        """Test that elicitation callback has the correct signature."""

        async def valid_callback(
            context: RequestContext[ClientSession, Any], params: mcp_types.ElicitRequestParams
        ) -> mcp_types.ElicitResult | mcp_types.ErrorData:
            return mcp_types.ElicitResult(action='accept', content={'result': 'test'})

        # This should not raise any type errors
        server = MCPServerStdio(command='python', args=['-c', 'print("test")'], elicitation_callback=valid_callback)

        assert server.elicitation_callback is valid_callback

    async def test_elicitation_callback_return_types(self):
        """Test different valid return types for elicitation callbacks."""

        # Test returning ElicitResult
        async def accept_callback(
            context: RequestContext[ClientSession, Any], params: mcp_types.ElicitRequestParams
        ) -> mcp_types.ElicitResult:
            return mcp_types.ElicitResult(action='accept', content={'result': 'accepted'})

        # Test returning ErrorData
        async def error_callback(
            context: RequestContext[ClientSession, Any], params: mcp_types.ElicitRequestParams
        ) -> mcp_types.ErrorData:
            return mcp_types.ErrorData(code=mcp_types.INTERNAL_ERROR, message='Callback error')

        server1 = MCPServerStdio(command='python', args=['-c', 'print("test")'], elicitation_callback=accept_callback)

        server2 = MCPServerStdio(command='python', args=['-c', 'print("test")'], elicitation_callback=error_callback)

        assert server1.elicitation_callback is accept_callback
        assert server2.elicitation_callback is error_callback

    async def test_multiple_servers_independent_callbacks(self):
        """Test that multiple servers can have different elicitation callbacks."""

        async def callback1(
            context: RequestContext[ClientSession, Any], params: mcp_types.ElicitRequestParams
        ) -> mcp_types.ElicitResult:
            return mcp_types.ElicitResult(action='accept', content={'server': '1'})

        async def callback2(
            context: RequestContext[ClientSession, Any], params: mcp_types.ElicitRequestParams
        ) -> mcp_types.ElicitResult:
            return mcp_types.ElicitResult(action='accept', content={'server': '2'})

        server1 = MCPServerStdio(command='python', args=['-c', 'print("test1")'], elicitation_callback=callback1)

        server2 = MCPServerStdio(command='python', args=['-c', 'print("test2")'], elicitation_callback=callback2)

        assert server1.elicitation_callback is callback1
        assert server2.elicitation_callback is callback2
        assert server1.elicitation_callback is not server2.elicitation_callback

    async def test_elicitation_inheritance_http_servers(self):
        """Test that HTTP-based servers also support elicitation callbacks."""

        async def http_callback(
            context: RequestContext[ClientSession, Any], params: mcp_types.ElicitRequestParams
        ) -> mcp_types.ElicitResult:
            return mcp_types.ElicitResult(action='decline', content={'reason': 'HTTP test'})

        sse_server = MCPServerSSE(
            url='http://localhost:3001/sse', elicitation_callback=http_callback, allow_elicitation=True
        )

        assert sse_server.elicitation_callback is http_callback
        assert sse_server.allow_elicitation is True

    async def test_parameters_order_consistency(self):
        """Test that parameter order is consistent across all server types."""

        async def callback(
            context: RequestContext[ClientSession, Any], params: mcp_types.ElicitRequestParams
        ) -> mcp_types.ElicitResult:
            return mcp_types.ElicitResult(action='accept', content={})

        # Test that all these constructors work with the same parameter order
        stdio_server = MCPServerStdio(
            'python',
            args=['-c', 'print("test")'],
            tool_prefix='test',
            timeout=10.0,
            allow_sampling=True,
            allow_elicitation=True,
            elicitation_callback=callback,
        )

        sse_server = MCPServerSSE(
            url='http://localhost:3001/sse',
            tool_prefix='test',
            timeout=10.0,
            allow_sampling=True,
            allow_elicitation=True,
            elicitation_callback=callback,
        )

        assert stdio_server.elicitation_callback is callback
        assert sse_server.elicitation_callback is callback
        assert stdio_server.allow_elicitation is True
        assert sse_server.allow_elicitation is True

    async def test_elicitation_with_other_callbacks(self):
        """Test that elicitation callback works alongside other callbacks like sampling."""
        from pydantic_ai.models.test import TestModel

        async def elicit_callback(
            context: RequestContext[ClientSession, Any], params: mcp_types.ElicitRequestParams
        ) -> mcp_types.ElicitResult:
            return mcp_types.ElicitResult(action='accept', content={'elicited': True})

        server = MCPServerStdio(
            command='python',
            args=['-c', 'print("test")'],
            elicitation_callback=elicit_callback,
            allow_elicitation=True,
            sampling_model=TestModel(),  # For sampling callback
            allow_sampling=True,
        )

        assert server.elicitation_callback is elicit_callback
        assert server.allow_elicitation is True
        assert server.sampling_model is not None
        assert server.allow_sampling is True
