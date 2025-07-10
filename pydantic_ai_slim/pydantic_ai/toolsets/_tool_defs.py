from __future__ import annotations

from dataclasses import dataclass

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from . import AbstractToolset
from .wrapper import WrapperToolset


@dataclass(init=False)
class ToolDefsToolset(WrapperToolset[AgentDepsT]):
    """A toolset that caches specific tool definitions."""

    _tool_defs: list[ToolDefinition]

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        tool_defs: list[ToolDefinition],
    ):
        super().__init__(wrapped)
        self._tool_defs = tool_defs

    async def _rewrap_for_run(
        self, wrapped: AbstractToolset[AgentDepsT], ctx: RunContext[AgentDepsT]
    ) -> WrapperToolset[AgentDepsT]:
        return ToolDefsToolset(wrapped, self._tool_defs)

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        return self._tool_defs
