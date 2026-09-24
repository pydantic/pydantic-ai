from typing import TYPE_CHECKING

from fastmcp.server import Context, FastMCP

if TYPE_CHECKING:
    from fastmcp.utilities.tasks import TaskConfig
else:
    try:
        from fastmcp.utilities.tasks import TaskConfig
    except ImportError:
        # FastMCP 3 keeps `TaskConfig` in the server package.
        from fastmcp.server.tasks import TaskConfig

mcp: FastMCP[None] = FastMCP('Pydantic AI MCP Task Server')

try:
    from fastmcp_tasks import TasksExtension
except ImportError:
    # FastMCP 3 has task support built in.
    pass
else:
    mcp.add_extension(TasksExtension())


@mcp.tool(task=TaskConfig(mode='required'))
async def required_task_tool() -> str:
    return 'required_completed'


@mcp.tool(task=TaskConfig(mode='optional'))
async def optional_task_tool(ctx: Context) -> str:
    return 'optional_task' if ctx.is_background_task else 'optional_sync'


if __name__ == '__main__':
    mcp.run()
