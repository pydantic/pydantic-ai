"""This module implements the MCP server interface between the agent and the LLM.

See https://docs.cursor.com/context/model-context-protocol for more information.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from types import TracebackType
from typing import Any, cast

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

__all__ = ('MCPServer',)


@dataclass
class MCPServer:
    """The MCP server that can be used to run a command or connect to an SSE server.

    See https://modelcontextprotocol.io/introduction for more information.
    """

    url: str | None = None
    command: str | None = None
    args: Sequence[str] = ()
    env: dict[str, str] | None = None

    @classmethod
    def stdio(cls, command: str, args: Sequence[str] = (), env: dict[str, str] | None = None) -> MCPServer:
        """Create a new MCP server using the Standard Input/Output (stdio) protocol.

        See https://modelcontextprotocol.io/docs/concepts/transports#standard-input%2Foutput-stdio for more information.

        Example:
            For illustration, here's how you can use the MCP server to run an agent:

            ```python
            import asyncio

            from pydantic_ai.agent import Agent
            from pydantic_ai.mcp import MCPServer


            async def main():
                async with MCPServer.stdio('python', ['-m', 'pydantic_ai.mcp']) as server:
                    agent = Agent('openai:gpt-4o', mcp_servers=[server])
                    result = await agent.run('Can you convert 30 degrees celsius to fahrenheit?')
                    print(result)


            asyncio.run(main())
            ```

        Args:
            command: The command to run.
            args: The arguments to pass to the command.
            env: The environment variables the CLI server will have access to.
        """
        return cls(command=command, args=args, env=env)

    @classmethod
    def sse(cls, url: str) -> MCPServer:
        """Create a new MCP server using the Server-Sent Events (SSE) protocol.

        See https://modelcontextprotocol.io/docs/concepts/transports#server-sent-events-sse for more information.

        Args:
            url: The URL of the SSE server.
        """
        return cls(url=url)

    def __post_init__(self):
        self.exit_stack = AsyncExitStack()

    async def list_tools(self) -> list[ToolDefinition]:
        """Retrieve the tools that the server has.

        Note:
        - We don't cache it because the tools might change.
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
        """Call a tool on the server."""
        return await self._client.call_tool(tool_name, arguments)  # type: ignore

    async def __aenter__(self) -> MCPServer:
        if self.url is None:
            assert self.command is not None and self.args is not None
            read_stream, write_stream = await self.exit_stack.enter_async_context(
                stdio_client(
                    server=StdioServerParameters(
                        command=self.command,
                        args=list(self.args),
                        env=self.env,
                    )
                )
            )
        else:
            read_stream, write_stream = await self.exit_stack.enter_async_context(sse_client(url=self.url))

        client_session = ClientSession(read_stream=read_stream, write_stream=write_stream)
        self._client = cast(ClientSession, await self.exit_stack.enter_async_context(client_session))
        await self._client.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[Exception] | None,
        exc_value: Exception | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.exit_stack.aclose()


# This is a simple example of how to use the MCP server.
# It's not used in the agent, but it's a good example of how to use the MCP server.
# You can run it directly with `python -m pydantic_ai.mcp`.
if __name__ == '__main__':
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP('PydanticAI MCP Server')

    @mcp.tool()
    async def celsius_to_fahrenheit(celsius: float) -> float:
        """Convert Celsius to Fahrenheit.

        Args:
            celsius: Temperature in Celsius

        Returns:
            Temperature in Fahrenheit
        """
        return (celsius * 9 / 5) + 32

    mcp.run()
