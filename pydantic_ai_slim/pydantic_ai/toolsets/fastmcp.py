from __future__ import annotations

import base64
from asyncio import Lock
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import Self

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.mcp import TOOL_SCHEMA_VALIDATOR, messages
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.abstract import ToolsetTool

try:
    from fastmcp.client import Client
    from fastmcp.client.transports import FastMCPTransport, MCPConfigTransport
    from fastmcp.exceptions import ToolError
    from fastmcp.mcp_config import MCPConfig
    from fastmcp.server.server import FastMCP
    from mcp.types import (
        AudioContent,
        ContentBlock,
        ImageContent,
        TextContent,
        Tool as MCPTool,
    )

except ImportError as _import_error:
    raise ImportError(
        'Please install the `fastmcp` package to use the FastMCP server, '
        'you can use the `fastmcp` optional group — `pip install "pydantic-ai-slim[fastmcp]"`'
    ) from _import_error


if TYPE_CHECKING:
    from fastmcp import FastMCP
    from fastmcp.client.client import CallToolResult
    from fastmcp.client.transports import FastMCPTransport
    from fastmcp.mcp_config import MCPServerTypes


FastMCPToolResult = messages.BinaryContent | dict[str, Any] | str | None

ToolErrorBehavior = Literal['model_retry', 'error']


class FastMCPToolset(AbstractToolset[AgentDepsT]):
    """A toolset that uses a FastMCP client as the underlying toolset."""

    tool_error_behavior: Literal['model_retry', 'error']
    fastmcp_client: Client[Any]

    max_retries: int

    _id: str | None

    _enter_lock: Lock
    _running_count: int
    _exit_stack: AsyncExitStack | None

    def __init__(
        self,
        fastmcp_client: Client[Any],
        *,
        max_retries: int = 2,
        tool_error_behavior: ToolErrorBehavior | None = None,
        id: str | None = None,
    ):
        """Build a new FastMCPToolset.

        Args:
            fastmcp_client: The FastMCP client to use.
            max_retries: The maximum number of retries for each tool during a run.
            tool_error_behavior: The behavior to take when a tool error occurs.
            id: An optional unique ID for the toolset. A toolset needs to have an ID in order to be used in a durable execution environment like Temporal,
                in which case the ID will be used to identify the toolset's activities within the workflow.
        """
        self.max_retries = max_retries
        self.fastmcp_client = fastmcp_client
        self._enter_lock = Lock()
        self._running_count = 0
        self._id = id

        self.tool_error_behavior = tool_error_behavior or 'error'

        super().__init__()

    @property
    def id(self) -> str | None:
        return self._id

    async def __aenter__(self) -> Self:
        async with self._enter_lock:
            if self._running_count == 0 and self.fastmcp_client:
                self._exit_stack = AsyncExitStack()
                await self._exit_stack.enter_async_context(self.fastmcp_client)

            self._running_count += 1

        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        async with self._enter_lock:
            self._running_count -= 1
            if self._running_count == 0 and self._exit_stack:
                await self._exit_stack.aclose()
                self._exit_stack = None

        return None

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        async with self:
            mcp_tools: list[MCPTool] = await self.fastmcp_client.list_tools()

            return {
                tool.name: _convert_mcp_tool_to_toolset_tool(toolset=self, mcp_tool=tool, retries=self.max_retries)
                for tool in mcp_tools
            }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        async with self:
            try:
                call_tool_result: CallToolResult = await self.fastmcp_client.call_tool(name=name, arguments=tool_args)
            except ToolError as e:
                if self.tool_error_behavior == 'model_retry':
                    raise ModelRetry(message=str(e)) from e
                else:
                    raise e

        # If any of the results are not text content, let's map them to Pydantic AI binary message parts
        if any(not isinstance(part, TextContent) for part in call_tool_result.content):
            return _map_fastmcp_tool_results(parts=call_tool_result.content)

        # Otherwise, if we have structured content, return that
        if call_tool_result.structured_content:
            return call_tool_result.structured_content

        return _map_fastmcp_tool_results(parts=call_tool_result.content)

    @classmethod
    def from_fastmcp_server(
        cls,
        fastmcp_server: FastMCP[Any],
        *,
        tool_error_behavior: ToolErrorBehavior | None = None,
        tool_retries: int = 2,
    ) -> Self:
        """Build a FastMCPToolset from a FastMCP server.

        Example:
        ```python
        from fastmcp import FastMCP

        from pydantic_ai.toolsets.fastmcp import FastMCPToolset

        fastmcp_server = FastMCP('my_server')
        @fastmcp_server.tool()
        async def my_tool(a: int, b: int) -> int:
            return a + b

        FastMCPToolset.from_fastmcp_server(fastmcp_server=fastmcp_server)
        ```
        """
        transport = FastMCPTransport(fastmcp_server)
        fastmcp_client: Client[FastMCPTransport] = Client[FastMCPTransport](transport=transport)
        return cls(fastmcp_client=fastmcp_client, max_retries=tool_retries, tool_error_behavior=tool_error_behavior)

    @classmethod
    def from_mcp_server(
        cls,
        name: str,
        mcp_server: MCPServerTypes | dict[str, Any],
        *,
        tool_error_behavior: ToolErrorBehavior | None = None,
        tool_retries: int = 2,
    ) -> Self:
        """Build a FastMCPToolset from an individual MCP server configuration.

        Example:
        ```python
        from pydantic_ai.toolsets.fastmcp import FastMCPToolset

        time_mcp_server = {
            'command': 'uv',
            'args': ['run', 'mcp-run-python', 'stdio'],
        }

        FastMCPToolset.from_mcp_server(name='time_server', mcp_server=time_mcp_server)
        ```
        """
        mcp_config: MCPConfig = MCPConfig.from_dict(config={name: mcp_server})

        return cls.from_mcp_config(
            mcp_config=mcp_config, tool_error_behavior=tool_error_behavior, max_retries=tool_retries
        )

    @classmethod
    def from_mcp_config(
        cls,
        mcp_config: MCPConfig | dict[str, Any],
        *,
        tool_error_behavior: ToolErrorBehavior | None = None,
        max_retries: int = 2,
    ) -> Self:
        """Build a FastMCPToolset from an MCP json-derived / dictionary configuration object.

        Example:
        ```python
        from pydantic_ai.toolsets.fastmcp import FastMCPToolset

        mcp_config = {
            'mcpServers': {
                'first_server': {
                    'command': 'uv',
                    'args': ['run', 'mcp-run-python', 'stdio'],
                },
                'second_server': {
                    'command': 'uv',
                    'args': ['run', 'mcp-run-python', 'stdio'],
                }
            }
        }

        FastMCPToolset.from_mcp_config(mcp_config)
        ```
        """
        transport: MCPConfigTransport = MCPConfigTransport(config=mcp_config)
        fastmcp_client: Client[MCPConfigTransport] = Client[MCPConfigTransport](transport=transport)
        return cls(fastmcp_client=fastmcp_client, max_retries=max_retries, tool_error_behavior=tool_error_behavior)


def _convert_mcp_tool_to_toolset_tool(
    toolset: FastMCPToolset[AgentDepsT],
    mcp_tool: MCPTool,
    retries: int,
) -> ToolsetTool[AgentDepsT]:
    """Convert an MCP tool to a toolset tool."""
    return ToolsetTool[AgentDepsT](
        tool_def=ToolDefinition(
            name=mcp_tool.name,
            description=mcp_tool.description,
            parameters_json_schema=mcp_tool.inputSchema,
        ),
        toolset=toolset,
        max_retries=retries,
        args_validator=TOOL_SCHEMA_VALIDATOR,
    )


def _map_fastmcp_tool_results(parts: list[ContentBlock]) -> list[FastMCPToolResult] | FastMCPToolResult:
    """Map FastMCP tool results to toolset tool results."""
    mapped_results = [_map_fastmcp_tool_result(part) for part in parts]

    if len(mapped_results) == 1:
        return mapped_results[0]

    return mapped_results


def _map_fastmcp_tool_result(part: ContentBlock) -> FastMCPToolResult:
    if isinstance(part, TextContent):
        return part.text

    if isinstance(part, ImageContent | AudioContent):
        return messages.BinaryContent(data=base64.b64decode(part.data), media_type=part.mimeType)

    msg = f'Unsupported/Unknown content block type: {type(part)}'  # pragma: no cover
    raise ValueError(msg)  # pragma: no cover
