from __future__ import annotations

import base64
from asyncio import Lock
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fastmcp.client.transports import ClientTransport
from fastmcp.mcp_config import MCPConfig
from fastmcp.server import FastMCP
from mcp.server.fastmcp import FastMCP as FastMCP1Server
from pydantic import AnyUrl
from typing_extensions import Self

from pydantic_ai import messages
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.abstract import ToolsetTool

try:
    from fastmcp.client import Client
    from fastmcp.exceptions import ToolError
    from mcp.types import (
        AudioContent,
        ContentBlock,
        ImageContent,
        TextContent,
        Tool as MCPTool,
    )

    from pydantic_ai.mcp import TOOL_SCHEMA_VALIDATOR

except ImportError as _import_error:
    raise ImportError(
        'Please install the `fastmcp` package to use the FastMCP server, '
        'you can use the `fastmcp` optional group — `pip install "pydantic-ai-slim[fastmcp]"`'
    ) from _import_error


if TYPE_CHECKING:
    from fastmcp.client.client import CallToolResult


FastMCPToolResult = messages.BinaryContent | dict[str, Any] | str | None

ToolErrorBehavior = Literal['model_retry', 'error']


@dataclass
class FastMCPToolset(AbstractToolset[AgentDepsT]):
    """A FastMCP Toolset that uses the FastMCP Client to call tools from a local or remote MCP Server."""

    mcp: Client[Any] | ClientTransport | FastMCP | FastMCP1Server | AnyUrl | Path | MCPConfig | dict[str, Any] | str
    """The FastMCP transport to use. This can be a local or remote MCP Server configuration or a FastMCP Client."""

    tool_error_behavior: Literal['model_retry', 'error'] = field(default='error')
    """The behavior to take when a tool error occurs."""

    max_retries: int = field(default=2)
    """The maximum number of retries to attempt if a tool call fails."""

    _id: str | None = field(default=None)

    def __post_init__(self):
        self._enter_lock: Lock = Lock()
        self._running_count: int = 0
        self._exit_stack: AsyncExitStack | None = None

        self._client: Client[Any]

        if isinstance(self.mcp, Client):
            self._client = self.mcp
        else:
            self._client = Client[Any](transport=self.mcp)

    @property
    def id(self) -> str | None:
        return self._id

    async def __aenter__(self) -> Self:
        async with self._enter_lock:
            if self._running_count == 0 and self._client:
                self._exit_stack = AsyncExitStack()
                await self._exit_stack.enter_async_context(self._client)

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
            mcp_tools: list[MCPTool] = await self._client.list_tools()

            return {
                tool.name: _convert_mcp_tool_to_toolset_tool(toolset=self, mcp_tool=tool, retries=self.max_retries)
                for tool in mcp_tools
            }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        async with self:
            try:
                call_tool_result: CallToolResult = await self._client.call_tool(name=name, arguments=tool_args)
            except ToolError as e:
                if self.tool_error_behavior == 'model_retry':
                    raise ModelRetry(message=str(e)) from e
                else:
                    raise e

        # If we have structured content, return that
        if call_tool_result.structured_content:
            return call_tool_result.structured_content

        # Otherwise, return the content
        return _map_fastmcp_tool_results(parts=call_tool_result.content)


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
            metadata={
                'meta': mcp_tool.meta,
                'annotations': mcp_tool.annotations.model_dump() if mcp_tool.annotations else None,
                'output_schema': mcp_tool.outputSchema or None,
            },
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
