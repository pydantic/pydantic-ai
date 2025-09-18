from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from hatchet_sdk import Hatchet
from hatchet_sdk.runnables.workflow import Standalone

from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.toolsets.wrapper import WrapperToolset

from ._run_context import HatchetRunContext
from ._utils import TaskConfig


class HatchetWrapperToolset(WrapperToolset[AgentDepsT], ABC):
    @property
    def id(self) -> str:
        assert self.wrapped.id is not None
        return self.wrapped.id

    @property
    @abstractmethod
    def hatchet_tasks(self) -> list[Standalone[Any, Any]]:
        """Return the list of Hatchet tasks for this toolset."""
        raise NotImplementedError

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        return self


def hatchetize_toolset(
    toolset: AbstractToolset[AgentDepsT],
    hatchet: Hatchet,
    task_name_prefix: str,
    task_config: TaskConfig,
    deps_type: type[AgentDepsT],
    run_context_type: type[HatchetRunContext[AgentDepsT]] = HatchetRunContext[AgentDepsT],
) -> AbstractToolset[AgentDepsT]:
    """Hatchetize a toolset.

    Args:
        toolset: The toolset to hatchetize.
        hatchet: The Hatchet instance to use for creating tasks.
        task_name_prefix: Prefix for Hatchet task names.
        task_config: The Hatchet task config to use.
        deps_type: The type of agent's dependencies object. It needs to be serializable using Pydantic's `TypeAdapter`.
        run_context_type: The `HatchetRunContext` (sub)class that's used to serialize and deserialize the run context.
    """
    if isinstance(toolset, FunctionToolset):
        from ._function_toolset import HatchetFunctionToolset

        return HatchetFunctionToolset(
            toolset,
            hatchet=hatchet,
            task_name_prefix=task_name_prefix,
            task_config=task_config,
        )

    try:
        from pydantic_ai.mcp import MCPServer

        from ._mcp_server import HatchetMCPServer
    except ImportError:
        pass
    else:
        if isinstance(toolset, MCPServer):
            return HatchetMCPServer(
                toolset,
                hatchet=hatchet,
                task_name_prefix=task_name_prefix,
                task_config=task_config,
                deps_type=deps_type,
                run_context_type=run_context_type,
            )

    return toolset
