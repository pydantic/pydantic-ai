# Prepare Tools

[`PrepareTools`][pydantic_ai.capabilities.PrepareTools] and [`PrepareOutputTools`][pydantic_ai.capabilities.PrepareOutputTools] wrap a [`ToolsPrepareFunc`][pydantic_ai.tools.ToolsPrepareFunc] as a [capability](overview.md), for filtering or modifying [tool definitions](../tools.md) per step. `PrepareTools` handles function tools; `PrepareOutputTools` handles [output tools][pydantic_ai.output.ToolOutput].

```python {title="prepare_tools_native.py"}
from pydantic_ai import Agent, RunContext, ToolDefinition
from pydantic_ai.capabilities import PrepareTools


async def hide_dangerous(ctx: RunContext, tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
    return [td for td in tool_defs if not td.name.startswith('delete_')]


agent = Agent('openai:gpt-5.2', capabilities=[PrepareTools(hide_dangerous)])


@agent.tool_plain
def delete_file(path: str) -> str:
    """Delete a file."""
    return f'deleted {path}'


@agent.tool_plain
def read_file(path: str) -> str:
    """Read a file."""
    return f'contents of {path}'


result = agent.run_sync('hello')
# The model only sees `read_file`, not `delete_file`
```

For more complex tool preparation logic, see [Tool preparation](custom.md#tool-preparation) under lifecycle hooks.
