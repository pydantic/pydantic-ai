from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition, ToolsPrepareFunc
from .abstract import AbstractToolset
from .prepared import PreparedToolset


@dataclass(init=False)
class SetMetadataToolset(PreparedToolset[AgentDepsT]):
    """A toolset that merges metadata key-value pairs onto all its tools.

    See [toolset docs](../toolsets.md) for more information.
    """

    prepare_func: ToolsPrepareFunc[AgentDepsT] = field(init=False, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])

    def __init__(self, wrapped: AbstractToolset[AgentDepsT], metadata: dict[str, Any]) -> None:
        self.metadata = metadata

        async def _set_metadata(ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            return [replace(td, metadata={**(td.metadata or {}), **self.metadata}) for td in tool_defs]

        super().__init__(wrapped=wrapped, prepare_func=_set_metadata)
