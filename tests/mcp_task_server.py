from fastmcp.server import Context, FastMCP

try:
    from fastmcp.server.tasks import TaskConfig
except ImportError:
    # FastMCP 4 moved `TaskConfig`.
    from fastmcp.utilities.tasks import TaskConfig

mcp: FastMCP[None] = FastMCP('Pydantic AI MCP Task Server')


@mcp.tool(task=TaskConfig(mode='required'))
async def required_task_tool() -> str:
    return 'required_completed'


@mcp.tool(task=TaskConfig(mode='optional'))
async def optional_task_tool(ctx: Context) -> str:
    return 'optional_task' if ctx.is_background_task else 'optional_sync'


try:
    from fastmcp_tasks import TasksExtension
except ImportError:
    # FastMCP 3 serves task-augmented tools itself (SEP-1686). FastMCP 4 hands them to the tasks
    # extension (SEP-2663) and refuses to start a server that declares them without it registered.
    pass
else:
    mcp.add_extension(TasksExtension())


if __name__ == '__main__':
    mcp.run()
