from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Literal

from temporalio.workflow import ActivityConfig

from pydantic_ai.ext.temporal._run_context import TemporalRunContext
from pydantic_ai.mcp import MCPServer
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.toolsets.wrapper import WrapperToolset


class TemporalWrapperToolset(WrapperToolset[Any], ABC):
    @property
    @abstractmethod
    def temporal_activities(self) -> list[Callable[..., Any]]:
        raise NotImplementedError


def temporalize_toolset(
    toolset: AbstractToolset[Any],
    activity_config: ActivityConfig = {},
    tool_activity_config: dict[str, ActivityConfig | Literal[False]] = {},
    run_context_type: type[TemporalRunContext] = TemporalRunContext,
) -> AbstractToolset[Any]:
    """Temporalize a toolset.

    Args:
        toolset: The toolset to temporalize.
        activity_config: The Temporal activity config to use.
        tool_activity_config: The Temporal activity config to use for specific tools identified by tool name.
        run_context_type: The type of run context to use to serialize and deserialize the run context.
    """
    if isinstance(toolset, FunctionToolset):
        from ._function_toolset import TemporalFunctionToolset

        return TemporalFunctionToolset(toolset, activity_config, tool_activity_config, run_context_type)
    elif isinstance(toolset, MCPServer):
        from ._mcp_server import TemporalMCPServer

        return TemporalMCPServer(toolset, activity_config, tool_activity_config, run_context_type)
    else:
        return toolset
