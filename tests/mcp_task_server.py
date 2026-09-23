from typing import TYPE_CHECKING, Any

from fastmcp.server import Context, FastMCP

try:
    from fastmcp.server.tasks import TaskConfig
except ImportError:
    # FastMCP 4 moved `TaskConfig`.
    from fastmcp.utilities.tasks import TaskConfig

# `fastmcp_tasks` is never installed in the typecheck environment, so pyright only gets a
# declaration. It needs a value rather than a bare annotation: this module registers the extension
# at import, and an annotation alone leaves the name unbound at that use.
if TYPE_CHECKING:
    TasksExtension: Any = None
else:
    try:
        from fastmcp_tasks import TasksExtension
    except ImportError:
        TasksExtension = None

mcp: FastMCP[None] = FastMCP('Pydantic AI MCP Task Server')


@mcp.tool(task=TaskConfig(mode='required'))
async def required_task_tool() -> str:
    return 'required_completed'


@mcp.tool(task=TaskConfig(mode='optional'))
async def optional_task_tool(ctx: Context) -> str:
    return 'optional_task' if ctx.is_background_task else 'optional_sync'


# FastMCP 3 serves task-augmented tools itself (SEP-1686). FastMCP 4 hands them to the tasks
# extension (SEP-2663) and refuses to start a server that declares them without it registered.
if TasksExtension is not None:
    getattr(mcp, 'add_extension')(TasksExtension())


if __name__ == '__main__':
    mcp.run()
