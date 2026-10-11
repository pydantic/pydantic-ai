"""Serve Pydantic AI toolsets over the Model Context Protocol.

Experimental prototype for https://github.com/pydantic/pydantic-ai/issues/9191: the reverse of
[`MCPToolset`][pydantic_ai.mcp.MCPToolset], exposing existing toolsets to MCP clients such as Claude Code
without rewriting them as FastMCP tools.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import PrivateAttr

from pydantic_ai.models import AbstractModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

if TYPE_CHECKING:
    from fastmcp import FastMCP
    from fastmcp.tools import Tool, ToolResult

    from pydantic_ai.toolsets import AbstractToolset, AgentToolset, ToolDefinition, ToolsetTool
else:
    try:
        from fastmcp import FastMCP
        from fastmcp.tools import Tool, ToolResult
    except ImportError as _import_error:
        raise ImportError(
            'Install MCP server support with: uv add "pydantic-ai-harness[mcp-server]"'
        ) from _import_error


__all__ = ('mcp_server_from_toolsets', 'run_mcp_toolset_server')


class _NoModel(AbstractModel):
    """Model placeholder for direct MCP tool calls, which never make a model request.

    Its name and system are benign metadata; making an actual model request fails with a clear error
    instead of silently talking to a real model.
    """

    @property
    def model_name(self) -> str:
        return 'mcp-server'

    @property
    def system(self) -> str:
        return 'mcp-server'

    async def request(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise RuntimeError('Direct MCP tool calls have no model; pass model= to the server to support such tools.')

    async def request_stream(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise RuntimeError('Direct MCP tool calls have no model; pass model= to the server to support such tools.')


class _ToolsetToolAdapter(Tool):
    """A FastMCP tool whose definition and execution are delegated to a Pydantic AI toolset."""

    _toolset: AbstractToolset[Any] = PrivateAttr()
    _ctx: RunContext[Any] = PrivateAttr()
    _tool: ToolsetTool[Any] = PrivateAttr()
    _structured: bool = PrivateAttr(default=False)

    @classmethod
    def from_toolset_tool(
        cls,
        toolset: AbstractToolset[Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
        tool_def: ToolDefinition,
    ) -> _ToolsetToolAdapter:
        # MCP structured output must be an object, so only advertise a return schema for object-shaped
        # results; everything else is returned as text.
        output_schema = (
            tool_def.return_schema
            if tool_def.return_schema is not None and tool_def.return_schema.get('type') == 'object'
            else None
        )
        adapter = cls(
            name=tool_def.name,
            description=tool_def.description or None,
            parameters=tool_def.parameters_json_schema,
            output_schema=output_schema,
        )
        adapter._toolset = toolset
        adapter._ctx = ctx
        adapter._tool = tool
        adapter._structured = output_schema is not None
        return adapter

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            result = await self._toolset.call_tool(self.name, arguments, self._ctx, self._tool)
        except Exception as e:
            return ToolResult(content=f'Tool {self.name} failed: {e!r}', is_error=True)
        text = result if isinstance(result, str) else repr(result)
        if self._structured and isinstance(result, dict):
            return ToolResult(content=text, structured_content=result)
        return ToolResult(content=text)


async def mcp_server_from_toolsets(
    *,
    toolsets: Sequence[AgentToolset[Any]],
    name: str,
    deps: Any = None,
    model: AbstractModel | None = None,
) -> FastMCP:
    """Build a FastMCP server exposing every executable tool of `toolsets` to MCP clients.

    Experimental: the API and the RunContext contract below may change without a deprecation period.

    The tools are resolved once at server construction with a synthetic RunContext carrying `deps` and, when
    given, `model`; each MCP call executes against that same context. Only plain function tools are exposed:
    tools that require approval or are deferred for loading need a session contract this prototype does not
    build yet, and are skipped. Dynamic toolsets are resolved at construction, not per call.

    Args:
        toolsets: The toolsets (or toolset factories) whose tools to serve.
        name: The MCP server name clients see.
        deps: Deps passed to tools via `ctx.deps`.
        model: Model exposed to tools via `ctx.model`; tools that use it require this.
    """
    from pydantic_ai.toolsets import CombinedToolset

    toolset: AbstractToolset[Any] = toolsets[0] if len(toolsets) == 1 else CombinedToolset[Any](list(toolsets))
    ctx = RunContext(deps=deps, model=model or _NoModel(), usage=RunUsage())
    tools = await toolset.get_tools(ctx)

    server = FastMCP(name)
    for tool_name, tool in tools.items():
        tool_def = tool.tool_def
        if tool_def.kind != 'function' or tool_def.defer_loading:
            continue
        server.add_tool(_ToolsetToolAdapter.from_toolset_tool(toolset, ctx, tool_def=tool_def, tool=tool))
    return server


def run_mcp_toolset_server(
    *,
    toolsets: Sequence[AgentToolset[Any]],
    name: str,
    deps: Any = None,
    model: AbstractModel | None = None,
    transport: Literal['stdio', 'http', 'sse'] = 'stdio',
) -> None:
    """Build a toolset server with `mcp_server_from_toolsets` and run it until the client disconnects.

    Experimental, see `mcp_server_from_toolsets`.
    """
    import anyio

    server = anyio.run(mcp_server_from_toolsets, toolsets, name=name, deps=deps, model=model)
    server.run(transport=transport)
