"""Tests for MCP elicitation callback functionality."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from .conftest import try_import

with try_import() as imports_successful:
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
        mock_callback = AsyncMock()
        server = MCPServerStdio(command='python', args=['-c', 'print("test")'], elicitation_callback=mock_callback)

        assert server.elicitation_callback is mock_callback
        assert server.allow_elicitation is True

    async def test_elicitation_callback_sse_server(self):
        """Test elicitation callback can be set on MCPServerSSE."""
        mock_callback = AsyncMock()
        server = MCPServerSSE(url='http://localhost:3001/sse', elicitation_callback=mock_callback)

        assert server.elicitation_callback is mock_callback
        assert server.allow_elicitation is True

    async def test_allow_elicitation_false(self):
        """Test that allow_elicitation can be disabled."""
        mock_callback = AsyncMock()
        server = MCPServerStdio(
            command='python', args=['-c', 'print("test")'], elicitation_callback=mock_callback, allow_elicitation=False
        )

        assert server.elicitation_callback is mock_callback
        assert server.allow_elicitation is False

    async def test_elicitation_callback_conditional_logic(self):
        """Test the conditional logic for when elicitation callback should be used."""
        mock_callback = AsyncMock()

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
        valid_callback = AsyncMock()

        # This should not raise any type errors
        server = MCPServerStdio(command='python', args=['-c', 'print("test")'], elicitation_callback=valid_callback)

        assert server.elicitation_callback is valid_callback

    async def test_elicitation_callback_return_types(self):
        """Test different valid return types for elicitation callbacks."""
        accept_callback = AsyncMock()
        error_callback = AsyncMock()

        server1 = MCPServerStdio(command='python', args=['-c', 'print("test")'], elicitation_callback=accept_callback)
        server2 = MCPServerStdio(command='python', args=['-c', 'print("test")'], elicitation_callback=error_callback)

        assert server1.elicitation_callback is accept_callback
        assert server2.elicitation_callback is error_callback

    async def test_multiple_servers_independent_callbacks(self):
        """Test that multiple servers can have different elicitation callbacks."""
        callback1 = AsyncMock()
        callback2 = AsyncMock()

        server1 = MCPServerStdio(command='python', args=['-c', 'print("test1")'], elicitation_callback=callback1)
        server2 = MCPServerStdio(command='python', args=['-c', 'print("test2")'], elicitation_callback=callback2)

        assert server1.elicitation_callback is callback1
        assert server2.elicitation_callback is callback2
        assert server1.elicitation_callback is not server2.elicitation_callback

    async def test_elicitation_inheritance_http_servers(self):
        """Test that HTTP-based servers also support elicitation callbacks."""
        http_callback = AsyncMock()

        sse_server = MCPServerSSE(
            url='http://localhost:3001/sse', elicitation_callback=http_callback, allow_elicitation=True
        )

        assert sse_server.elicitation_callback is http_callback
        assert sse_server.allow_elicitation is True

    async def test_parameters_order_consistency(self):
        """Test that parameter order is consistent across all server types."""
        callback = AsyncMock()

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

        elicit_callback = AsyncMock()

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

    async def test_elicitation_callback_runtime_integration(self):
        """Test that elicitation callback is passed to ClientSession during server startup."""
        mock_callback = AsyncMock()

        # Test: elicitation enabled - callback should be passed to ClientSession
        server_enabled = MCPServerStdio(
            command='python',
            args=['-c', 'print("test")'],
            elicitation_callback=mock_callback,
            allow_elicitation=True,
        )

        captured_elicitation_callback: Any = None

        # Create a mock ClientSession class that captures the elicitation_callback parameter
        class MockClientSession:
            def __init__(self, **kwargs: Any) -> None:
                nonlocal captured_elicitation_callback
                captured_elicitation_callback = kwargs.get('elicitation_callback')

            async def __aenter__(self) -> MockClientSession:
                return self

            async def __aexit__(self, *args: Any) -> None:
                pass

            async def initialize(self) -> None:
                pass

        # Create mock streams that support async context manager protocol
        mock_read_stream = AsyncMock()
        mock_write_stream = AsyncMock()
        mock_read_stream.__aenter__ = AsyncMock(return_value=mock_read_stream)
        mock_read_stream.__aexit__ = AsyncMock(return_value=None)
        mock_write_stream.__aenter__ = AsyncMock(return_value=mock_write_stream)
        mock_write_stream.__aexit__ = AsyncMock(return_value=None)

        # Create a mock async context manager for client_streams
        class MockClientStreams:
            async def __aenter__(self) -> tuple[AsyncMock, AsyncMock]:
                return mock_read_stream, mock_write_stream

            async def __aexit__(self, *args: Any) -> None:
                pass

        # Patch ClientSession in the mcp module where it's used
        with patch('pydantic_ai.mcp.ClientSession', MockClientSession):
            with patch.object(server_enabled, 'client_streams', return_value=MockClientStreams()):
                async with server_enabled:
                    pass

        # Verify that elicitation_callback was passed to ClientSession
        assert captured_elicitation_callback is mock_callback

        # Test: elicitation disabled - callback should be None in ClientSession
        server_disabled = MCPServerStdio(
            command='python',
            args=['-c', 'print("test")'],
            elicitation_callback=mock_callback,
            allow_elicitation=False,
        )

        captured_elicitation_callback_disabled: Any = None

        class MockClientSessionDisabled:
            def __init__(self, **kwargs: Any) -> None:
                nonlocal captured_elicitation_callback_disabled
                captured_elicitation_callback_disabled = kwargs.get('elicitation_callback')

            async def __aenter__(self) -> MockClientSessionDisabled:
                return self

            async def __aexit__(self, *args: Any) -> None:
                pass

            async def initialize(self) -> None:
                pass

        # Create mock streams for the disabled test as well
        mock_read_stream_disabled = AsyncMock()
        mock_write_stream_disabled = AsyncMock()
        mock_read_stream_disabled.__aenter__ = AsyncMock(return_value=mock_read_stream_disabled)
        mock_read_stream_disabled.__aexit__ = AsyncMock(return_value=None)
        mock_write_stream_disabled.__aenter__ = AsyncMock(return_value=mock_write_stream_disabled)
        mock_write_stream_disabled.__aexit__ = AsyncMock(return_value=None)

        class MockClientStreamsDisabled:
            async def __aenter__(self) -> tuple[AsyncMock, AsyncMock]:
                return mock_read_stream_disabled, mock_write_stream_disabled

            async def __aexit__(self, *args: Any) -> None:
                pass

        with patch('pydantic_ai.mcp.ClientSession', MockClientSessionDisabled):
            with patch.object(server_disabled, 'client_streams', return_value=MockClientStreamsDisabled()):
                async with server_disabled:
                    pass

        # Verify that elicitation_callback was None when allow_elicitation=False
        assert captured_elicitation_callback_disabled is None
