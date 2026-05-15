from __future__ import annotations

import inspect
from dataclasses import dataclass

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.tools import ToolDefinition, ToolSelector, ToolsPrepareFunc, matches_tool_selector

from .abstract import AbstractCapability


@dataclass
class PrepareTools(AbstractCapability[AgentDepsT]):
    """Capability that filters or modifies function tool definitions using a callable.

    Wraps a [`ToolsPrepareFunc`][pydantic_ai.tools.ToolsPrepareFunc] as a capability.
    Filters/modifies **function** tools only; for output tools use
    [`PrepareOutputTools`][pydantic_ai.capabilities.PrepareOutputTools].

    The `tools` parameter scopes which tools the prepare function sees.
    Non-matching tools pass through unchanged.

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
    tools: ToolSelector[AgentDepsT] = 'all'

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # Not spec-serializable (takes a callable)

    async def prepare_tools(self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        return await _apply_selector_and_prepare(self.tools, self.prepare_func, ctx, tool_defs)


@dataclass
class PrepareOutputTools(AbstractCapability[AgentDepsT]):
    """Capability that filters or modifies output tool definitions using a callable.

    Mirrors [`PrepareTools`][pydantic_ai.capabilities.PrepareTools] for
    [output tools][pydantic_ai.output.ToolOutput]. `ctx.retry`/`ctx.max_retries` reflect
    the **output** retry budget (`max_output_retries`), matching the output hook lifecycle.

    The `tools` parameter scopes which output tools the prepare function sees.
    Non-matching tools pass through unchanged.

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
    tools: ToolSelector[AgentDepsT] = 'all'

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # Not spec-serializable (takes a callable)

    async def prepare_output_tools(
        self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        return await _apply_selector_and_prepare(self.tools, self.prepare_func, ctx, tool_defs)


async def _apply_selector_and_prepare(
    selector: ToolSelector[AgentDepsT],
    prepare_func: ToolsPrepareFunc[AgentDepsT],
    ctx: RunContext[AgentDepsT],
    tool_defs: list[ToolDefinition],
) -> list[ToolDefinition]:
    """Run `prepare_func` over only tools matching `selector`; pass others through unchanged.

    Preserves original tool order: non-matching tools stay at their original positions;
    matching tools are replaced with the prepared output (or dropped if the prepare
    function filters them out).
    """
    if selector == 'all':
        return await _call_prepare_func(prepare_func, ctx, tool_defs)

    matching: list[ToolDefinition] = []
    matching_names: set[str] = set()
    for td in tool_defs:
        if await matches_tool_selector(selector, ctx, td):
            matching.append(td)
            matching_names.add(td.name)

    prepared = await _call_prepare_func(prepare_func, ctx, matching)
    prepared_by_name = {td.name: td for td in prepared}

    result: list[ToolDefinition] = []
    for td in tool_defs:
        if td.name in matching_names:
            if td.name in prepared_by_name:
                result.append(prepared_by_name[td.name])
            # else: prepare_func dropped this tool
        else:
            result.append(td)
    return result


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
