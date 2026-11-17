from typing import TYPE_CHECKING, Any

from fastmcp.server import Context, FastMCP

from pydantic_ai._mcp_compat import MCP_SDK_V2

if TYPE_CHECKING:
    from fastmcp.utilities.tasks import TaskConfig
else:
    try:
        from fastmcp.utilities.tasks import TaskConfig
    except ImportError:  # FastMCP 3 keeps `TaskConfig` in its server package.
        from fastmcp.server.tasks import TaskConfig

# `fastmcp_tasks` is FastMCP 4's tasks extension, which the FastMCP 3 CI job doesn't install, so
# pyright gets a declaration that allows `None`. It needs a value rather than a bare annotation:
# this module registers the extension at import, and an annotation alone leaves the name unbound.
if TYPE_CHECKING:
    TasksExtension: Any = None
else:
    try:
        from fastmcp_tasks import TasksExtension
    except ImportError:
        TasksExtension = None

# FastMCP 4 moved task execution into the optional `fastmcp-tasks` package, which only the
# `mcp-tasks` extra installs, and refuses to start a server declaring task tools without it. Tests
# that run this server skip on such installs; FastMCP 3 serves task tools itself.
TASK_SERVER_UNAVAILABLE = MCP_SDK_V2 and TasksExtension is None

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
    mcp.add_extension(TasksExtension())


if __name__ == '__main__':
    mcp.run()
