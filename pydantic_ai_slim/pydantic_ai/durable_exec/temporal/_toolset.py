from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Literal

from temporalio.workflow import ActivityConfig

from pydantic_ai.mcp import MCPServer
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.toolsets.wrapper import WrapperToolset

from ._run_context import TemporalRunContext


class TemporalWrapperToolset(WrapperToolset[AgentDepsT], ABC):
    @property
    def id(self) -> str:
        # An error is raised in `TemporalAgent` if no `id` is set.
        assert self.wrapped.id is not None
        return self.wrapped.id

    @property
    @abstractmethod
    def temporal_activities(self) -> list[Callable[..., Any]]:
        raise NotImplementedError

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        # Temporalized toolsets cannot be swapped out after the fact.
        return self


def temporalize_toolset(
    toolset: AbstractToolset[AgentDepsT],
    activity_name_prefix: str,
    activity_config: ActivityConfig,
    tool_activity_config: dict[str, ActivityConfig | Literal[False]],
    deps_type: type[AgentDepsT],
    run_context_type: type[TemporalRunContext[AgentDepsT]] = TemporalRunContext[AgentDepsT],
) -> AbstractToolset[AgentDepsT]:
    """Temporalize a toolset.

    Args:
        toolset: The toolset to temporalize.
        activity_name_prefix: Prefix for Temporal activity names.
        activity_config: The Temporal activity config to use.
        tool_activity_config: The Temporal activity config to use for specific tools identified by tool name.
        deps_type: The type of agent's dependencies object. It needs to be serializable using Pydantic's `TypeAdapter`.
        run_context_type: The `TemporalRunContext` (sub)class that's used to serialize and deserialize the run context.
    """
    if isinstance(toolset, FunctionToolset):
        from ._function_toolset import TemporalFunctionToolset

        return TemporalFunctionToolset(
            toolset,
            activity_name_prefix=activity_name_prefix,
            activity_config=activity_config,
            tool_activity_config=tool_activity_config,
            deps_type=deps_type,
            run_context_type=run_context_type,
        )
    elif isinstance(toolset, MCPServer):
        from ._mcp_server import TemporalMCPServer

        return TemporalMCPServer(
            toolset,
            activity_name_prefix=activity_name_prefix,
            activity_config=activity_config,
            tool_activity_config=tool_activity_config,
            deps_type=deps_type,
            run_context_type=run_context_type,
        )
    else:
        return toolset
