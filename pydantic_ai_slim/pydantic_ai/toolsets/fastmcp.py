from __future__ import annotations

import base64
from asyncio import Lock
from contextlib import AsyncExitStack
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import AnyUrl
from typing_extensions import Self, assert_never

from pydantic_ai import messages
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.abstract import ToolsetTool

try:
    from fastmcp.client import Client
    from fastmcp.client.transports import ClientTransport
    from fastmcp.exceptions import ToolError
    from fastmcp.mcp_config import MCPConfig
    from fastmcp.server import FastMCP
    from mcp.server.fastmcp import FastMCP as FastMCP1Server
    from mcp.types import (
        AudioContent,
        BlobResourceContents,
        ContentBlock,
        EmbeddedResource,
        ImageContent,
        ResourceLink,
        TextContent,
        TextResourceContents,
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

UNKNOWN_BINARY_MEDIA_TYPE = 'application/octet-stream'


@dataclass(init=False)
class FastMCPToolset(AbstractToolset[AgentDepsT]):
    """A FastMCP Toolset that uses the FastMCP Client to call tools from a local or remote MCP Server.

    The Toolset can accept a FastMCP Client, a FastMCP Transport, or any other object which a FastMCP Transport can be created from.

    See https://gofastmcp.com/clients/transports for a full list of transports available.
    """

    client: Client[Any]
    """The FastMCP client to use."""

    _: KW_ONLY

    tool_error_behavior: Literal['model_retry', 'error']
    """The behavior to take when a tool error occurs."""

    max_retries: int
    """The maximum number of retries to attempt if a tool call fails."""

    _id: str | None

    def __init__(
        self,
        client: Client[Any]
        | ClientTransport
        | FastMCP
        | FastMCP1Server
        | AnyUrl
        | Path
        | MCPConfig
        | dict[str, Any]
        | str,
        *,
        max_retries: int = 1,
        tool_error_behavior: Literal['model_retry', 'error'] = 'model_retry',
        id: str | None = None,
    ) -> None:
        if isinstance(client, Client):
            self.client = client
        else:
            self.client = Client[Any](transport=client)

        self._id = id
        self.max_retries = max_retries
        self.tool_error_behavior = tool_error_behavior

        self._enter_lock: Lock = Lock()
        self._running_count: int = 0
        self._exit_stack: AsyncExitStack | None = None

    @property
    def id(self) -> str | None:
        return self._id

    async def __aenter__(self) -> Self:
        async with self._enter_lock:
            if self._running_count == 0:
                self._exit_stack = AsyncExitStack()
                await self._exit_stack.enter_async_context(self.client)

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
            return {
                mcp_tool.name: self.tool_for_tool_def(
                    ToolDefinition(
                        name=mcp_tool.name,
                        description=mcp_tool.description,
                        parameters_json_schema=mcp_tool.inputSchema,
                        metadata={
                            'meta': mcp_tool.meta,
                            'annotations': mcp_tool.annotations.model_dump() if mcp_tool.annotations else None,
                            'output_schema': mcp_tool.outputSchema or None,
                        },
                    )
                )
                for mcp_tool in await self.client.list_tools()
            }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        async with self:
            try:
                call_tool_result: CallToolResult = await self.client.call_tool(name=name, arguments=tool_args)
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

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        return ToolsetTool[AgentDepsT](
            tool_def=tool_def,
            toolset=self,
            max_retries=self.max_retries,
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
    elif isinstance(part, ImageContent | AudioContent):
        return messages.BinaryContent(data=base64.b64decode(part.data), media_type=part.mimeType)
    elif isinstance(part, EmbeddedResource):
        if isinstance(part.resource, BlobResourceContents):
            return messages.BinaryContent(
                data=base64.b64decode(part.resource.blob),
                media_type=part.resource.mimeType or UNKNOWN_BINARY_MEDIA_TYPE,
            )
        elif isinstance(part.resource, TextResourceContents):
            return part.resource.text
        else:
            assert_never(part.resource)
    elif isinstance(part, ResourceLink):
        # ResourceLink is not yet supported by the FastMCP toolset as reading resources is not yet supported.
        raise NotImplementedError(
            'ResourceLink is not supported by the FastMCP toolset as reading resources is not yet supported.'
        )
    else:
        assert_never(part)
