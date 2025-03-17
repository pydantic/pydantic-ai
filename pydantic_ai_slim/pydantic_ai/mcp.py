"""This module implements the MCP server interface between the agent and the LLM.

See <https://docs.cursor.com/context/model-context-protocol> for more information.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from types import TracebackType
from typing import Any

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.types import JSONRPCMessage
from typing_extensions import Self

from pydantic_ai.tools import ToolDefinition

try:
    from mcp.client.session import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.types import CallToolResult
except ImportError as _import_error:
    raise ImportError(
        'Please install the `mcp` package to use the MCP server, '
        "you can use the `mcp` optional group — `pip install 'pydantic-ai-slim[mcp]'`"
    ) from _import_error

__all__ = ('MCPServer', 'MCPServerStdio', 'MCPServerSSE')


class MCPServer(ABC):
    """Base class for MCP servers that can be used to run a command or connect to an SSE server.

    See <https://modelcontextprotocol.io/introduction> for more information.
    """

    is_running: bool = False

    _client: ClientSession
    _read_stream: MemoryObjectReceiveStream[JSONRPCMessage | Exception]
    _write_stream: MemoryObjectSendStream[JSONRPCMessage]
    _exit_stack: AsyncExitStack

    @abstractmethod
    @asynccontextmanager
    async def client_streams(
        self,
    ) -> AsyncIterator[
        tuple[MemoryObjectReceiveStream[JSONRPCMessage | Exception], MemoryObjectSendStream[JSONRPCMessage]]
    ]:
        """Create the streams for the MCP server."""
        raise NotImplementedError('MCP Server subclasses must implement this method.')
        yield

    async def list_tools(self) -> list[ToolDefinition]:
        """Retrieve tools that are currently active on the server.

        Note:
        - We don't cache tools as they might change.
        - We also don't subscribe to the server to avoid complexity.
        """
        tools = await self._client.list_tools()
        return [
            ToolDefinition(
                name=tool.name,
                description=tool.description or '',
                parameters_json_schema=tool.inputSchema,
            )
            for tool in tools.tools
        ]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> CallToolResult:
        """Call a tool on the server.

        Args:
            tool_name: The name of the tool to call.
            arguments: The arguments to pass to the tool.

        Returns:
            The result of the tool call.
        """
        return await self._client.call_tool(tool_name, arguments)

    async def __aenter__(self) -> Self:
        self._exit_stack = AsyncExitStack()

        self._read_stream, self._write_stream = await self._exit_stack.enter_async_context(self.client_streams())
        client = ClientSession(read_stream=self._read_stream, write_stream=self._write_stream)
        self._client = await self._exit_stack.enter_async_context(client)

        await self._client.initialize()
        self.is_running = True
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> bool | None:
        await self._exit_stack.aclose()
        self.is_running = False


@dataclass
class MCPServerStdio(MCPServer):
    """An MCP server that runs a subprocess.

    This class implements the stdio transport from the MCP specification.
    See <https://modelcontextprotocol.io/docs/concepts/transports#standard-input%2Foutput-stdio> for more information.

    Example:
    ```python {py="3.10"}
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPServerStdio

    server = MCPServerStdio('python', ['-m', 'pydantic_ai_examples.mcp_server'])
    agent = Agent('openai:gpt-4o', mcp_servers=[server])
    ```
    """

    command: str
    """The command to run."""

    args: Sequence[str]
    """The arguments to pass to the command."""

    env: dict[str, str] | None = None
    """The environment variables the CLI server will have access to."""

    @asynccontextmanager
    async def client_streams(
        self,
    ) -> AsyncIterator[
        tuple[MemoryObjectReceiveStream[JSONRPCMessage | Exception], MemoryObjectSendStream[JSONRPCMessage]]
    ]:
        server = StdioServerParameters(command=self.command, args=list(self.args), env=self.env)
        async with stdio_client(server=server) as (read_stream, write_stream):
            yield read_stream, write_stream


@dataclass
class MCPServerSSE(MCPServer):
    """An MCP server that connects to a remote server.

    This class implements the SSE transport from the MCP specification.
    See <https://modelcontextprotocol.io/docs/concepts/transports#server-sent-events-sse> for more information.
    """

    url: str
    """The URL of the remote server."""

    @asynccontextmanager
    async def client_streams(
        self,
    ) -> AsyncIterator[
        tuple[MemoryObjectReceiveStream[JSONRPCMessage | Exception], MemoryObjectSendStream[JSONRPCMessage]]
    ]:  # pragma: no cover
        async with sse_client(url=self.url) as (read_stream, write_stream):
            yield read_stream, write_stream
