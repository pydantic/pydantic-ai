from __future__ import annotations

import inspect
from dataclasses import dataclass

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.tools import ToolDefinition, ToolsPrepareFunc

from .abstract import AbstractCapability


@dataclass
class PrepareTools(AbstractCapability[AgentDepsT]):
    """Capability that filters or modifies function tool definitions using a callable.

    Wraps a [`ToolsPrepareFunc`][pydantic_ai.tools.ToolsPrepareFunc] as a capability —
    sugar for the agent-level `prepare_tools=` constructor argument, which injects this
    capability automatically. Filters/modifies **function** tools only; for output tools
    use [`PrepareOutputTools`][pydantic_ai.capabilities.PrepareOutputTools].

    ```python
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.capabilities import PrepareTools
    from pydantic_ai.tools import ToolDefinition


    async def hide_admin_tools(
        ctx: RunContext[None], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition] | None:
        return [td for td in tool_defs if not td.name.startswith('admin_')]


    agent = Agent('openai:gpt-5', capabilities=[PrepareTools(hide_admin_tools)])
    ```
    """

    prepare_func: ToolsPrepareFunc[AgentDepsT]

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # Not spec-serializable (takes a callable)

    async def prepare_tools(self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        return await _call_prepare_func(self.prepare_func, ctx, tool_defs)


@dataclass
class PrepareOutputTools(AbstractCapability[AgentDepsT]):
    """Capability that filters or modifies output tool definitions using a callable.

    Mirrors [`PrepareTools`][pydantic_ai.capabilities.PrepareTools] for
    [output tools][pydantic_ai.output.ToolOutput]. `ctx.retry`/`ctx.max_retries` reflect
    the **output** retry budget (`max_result_retries`), matching the output hook lifecycle.

    ```python
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.capabilities import PrepareOutputTools
    from pydantic_ai.output import ToolOutput
    from pydantic_ai.tools import ToolDefinition


    async def only_after_first_step(
        ctx: RunContext[None], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition] | None:
        return tool_defs if ctx.run_step > 0 else []


    agent = Agent(
        'openai:gpt-5',
        output_type=ToolOutput(str),
        capabilities=[PrepareOutputTools(only_after_first_step)],
    )
    ```
    """

    prepare_func: ToolsPrepareFunc[AgentDepsT]

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # Not spec-serializable (takes a callable)

    async def prepare_output_tools(
        self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        return await _call_prepare_func(self.prepare_func, ctx, tool_defs)


async def _call_prepare_func(
    prepare_func: ToolsPrepareFunc[AgentDepsT],
    ctx: RunContext[AgentDepsT],
    tool_defs: list[ToolDefinition],
) -> list[ToolDefinition]:
    # Just sync/async + `None` normalization — `PreparedToolset.get_tools` validates that
    # the result didn't add or rename tools when these capabilities' hooks dispatch through it.
    result = prepare_func(ctx, tool_defs)
    if inspect.isawaitable(result):
        result = await result
    return list(result or [])
