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
        'you can use the `mcp` optional group — `pip install "pydantic-ai-slim[mcp]"`'
    ) from _import_error

__all__ = 'MCPServer', 'MCPServerStdio', 'MCPServerHTTP'


class MCPServer(ABC):
    """Base class for attaching agents to MCP servers.

    See <https://modelcontextprotocol.io> for more information.
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
    """Runs an MCP server in a subprocess and communicates with it over stdin/stdout.

    This class implements the stdio transport from the MCP specification.
    See <https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#stdio> for more information.

    !!! note
        Using this class as an async context manager will start the server as a subprocess when entering the context,
        and stop it when exiting the context.

    Example:
    ```python {py="3.10"}
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPServerStdio

    server = MCPServerStdio(  # (1)!
        'deno',
        args=[
            'run',
            '-N',
            '-R=node_modules',
            '-W=node_modules',
            '--node-modules-dir=auto',
            'jsr:@pydantic/mcp-run-python',
            'stdio',
        ]
    )
    agent = Agent('openai:gpt-4o', mcp_servers=[server])

    async def main():
        async with agent.run_mcp_servers():  # (2)!
            ...
    ```

    1. See [MCP Run Python](../mcp/run-python.md) for more information.
    2. This will start the server as a subprocess and connect to it.
    """

    command: str
    """The command to run."""

    args: Sequence[str]
    """The arguments to pass to the command."""

    env: dict[str, str] | None = None
    """The environment variables the CLI server will have access to.

    By default the subprocess will not inherit any environment variables from the parent process.
    If you want to inherit the environment variables from the parent process, use `env=os.environ`.
    """

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
class MCPServerHTTP(MCPServer):
    """An MCP server that connects over streamable HTTP connections.

    This class implements the SSE transport from the MCP specification.
    See <https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#http-with-sse> for more information.

    The name "HTTP" is used since this implemented will be adapted in future to use the new
    [Streamable HTTP](https://github.com/modelcontextprotocol/specification/pull/206) currently in development.

    !!! note
        Using this class as an async context manager will create a new pool of HTTP connections to connect
        to a server which should already be running.

    Example:
    ```python {py="3.10"}
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPServerHTTP

    server = MCPServerHTTP('http://localhost:3001/sse')  # (1)!
    agent = Agent('openai:gpt-4o', mcp_servers=[server])

    async def main():
        async with agent.run_mcp_servers():  # (2)!
            ...
    ```

    1. E.g. you might be connecting to a server run with [`mcp-run-python`](../mcp/run-python.md).
    2. This will connect to a server running on `localhost:3001`.
    """

    url: str
    """The URL of the SSE endpoint on the MCP server.

    For example for a server running locally, this might be `http://localhost:3001/sse`.
    """

    headers: dict[str, Any] | None = None
    """Optional HTTP headers to be sent with each request to the SSE endpoint.

    These headers will be passed directly to the underlying `httpx.AsyncClient`.
    Useful for authentication, custom headers, or other HTTP-specific configurations.
    """

    timeout: float = 5
    """Initial connection timeout in seconds for establishing the SSE connection.

    This timeout applies to the initial connection setup and handshake.
    If the connection cannot be established within this time, the operation will fail.
    """

    sse_read_timeout: float = 60 * 5
    """Maximum time in seconds to wait for new SSE messages before timing out.

    This timeout applies to the long-lived SSE connection after it's established.
    If no new messages are received within this time, the connection will be considered stale
    and may be closed. Defaults to 5 minutes (300 seconds).
    """

    @asynccontextmanager
    async def client_streams(
        self,
    ) -> AsyncIterator[
        tuple[MemoryObjectReceiveStream[JSONRPCMessage | Exception], MemoryObjectSendStream[JSONRPCMessage]]
    ]:  # pragma: no cover
        async with sse_client(
            url=self.url, headers=self.headers, timeout=self.timeout, sse_read_timeout=self.sse_read_timeout
        ) as (read_stream, write_stream):
            yield read_stream, write_stream
