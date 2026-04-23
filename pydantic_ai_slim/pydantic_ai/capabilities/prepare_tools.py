from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.tools import AgentDepsT, ToolsPrepareFunc
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.toolsets.prepared import PreparedToolset

from .abstract import AbstractCapability


@dataclass
class PrepareTools(AbstractCapability[AgentDepsT]):
    """Capability that filters or modifies tool definitions using a callable.

    Wraps a [`ToolsPrepareFunc`][pydantic_ai.tools.ToolsPrepareFunc] as a capability,
    allowing it to be composed with other capabilities via the capability system.

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

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        return PreparedToolset(toolset, self.prepare_func)
