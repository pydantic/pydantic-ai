"""Internal ToolSearch capability that wraps tools with ToolSearchToolset."""

from __future__ import annotations

from dataclasses import dataclass

from .._run_context import AgentDepsT
from ..toolsets import AbstractToolset
from ..toolsets._tool_search import ToolSearchToolset
from .abstract import AbstractCapability, CapabilityOrdering


@dataclass
class ToolSearch(AbstractCapability[AgentDepsT]):
    """Internal capability that wraps tools with ToolSearchToolset for deferred tool discovery.

    Auto-injected when not explicitly provided by the user. Short-circuits
    when no deferred tools exist, so there is zero overhead for agents
    without deferred loading.

    Internal for now — will be exported publicly once we add
    user-facing configuration options.
    """

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # not spec-constructible (internal)

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        return ToolSearchToolset(wrapped=toolset)
