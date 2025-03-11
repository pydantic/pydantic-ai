"""This module implements the MCP server interface between the agent and the LLM.

See https://docs.cursor.com/context/model-context-protocol for more information.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Any, NotRequired, Union, cast

from typing_extensions import TypedDict, TypeIs

from pydantic_ai.tools import ToolDefinition

try:
    from mcp.client.session import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.types import CallToolResult
except ImportError as _import_error:
    raise ImportError(
        'Please install `mcp` to use the MCP server, '
        "you can use the `mcp` optional group — `pip install 'pydantic-ai-slim[mcp]'`"
    ) from _import_error

__all__ = ('MCPServer', 'StdioMCPServerConfig', 'SseMCPServerConfig')


class MCPServer:
    """A server that runs a command and streams the output to the client."""

    def __init__(self, config: MCPServerConfig):
        self._config = config
        self.exit_stack = AsyncExitStack()

    @property
    def tools(self) -> dict[str, ToolDefinition]:
        """The tools that the server has."""
        return self._tools

    async def list_tools(self) -> list[ToolDefinition]:
        """This is the only method that needs to be implemented by the server."""
        tools = await self._client.list_tools()
        self._tools = {
            tool.name: ToolDefinition(
                name=tool.name,
                description=tool.description or '',
                parameters_json_schema=tool.inputSchema,
            )
            for tool in tools.tools
        }
        return list(self.tools.values())

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> CallToolResult:
        """This is the only method that needs to be implemented by the server."""
        return await self._client.call_tool(tool_name, arguments)  # type: ignore

    async def __aenter__(self) -> MCPServer:
        if self.is_stdio(self._config):
            read_stream, write_stream = await self.exit_stack.enter_async_context(
                stdio_client(
                    server=StdioServerParameters(
                        command=self._config['command'],
                        args=list(self._config['args']),
                        env=self._config.get('env', {}),
                    )
                )
            )
        else:
            read_stream, write_stream = await self.exit_stack.enter_async_context(sse_client(url=self._config['url']))

        client_session = ClientSession(read_stream=read_stream, write_stream=write_stream)
        self._client = cast(ClientSession, await self.exit_stack.enter_async_context(client_session))
        await self._client.initialize()
        return self

    async def __aexit__(
        self, exc_type: type[Exception] | None, exc_value: Exception | None, traceback: TracebackType | None
    ) -> None:
        await self.exit_stack.aclose()

    @abstractmethod
    def is_stdio(self, config: MCPServerConfig) -> TypeIs[StdioMCPServerConfig]:
        return config.get('url') is None


class StdioMCPServerConfig(TypedDict):
    """The configuration for a StdioMCPServer."""

    command: str
    """The command to run."""
    args: Sequence[str]
    """The arguments to pass to the command."""
    env: NotRequired[dict[str, str]]
    """The environment variables the CLI server will have access to."""


class SseMCPServerConfig(TypedDict):
    """The configuration for a SseMCPServer."""

    url: str
    """The URL of the SSE server."""


MCPServerConfig = Union[StdioMCPServerConfig, SseMCPServerConfig]
