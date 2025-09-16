from __future__ import annotations

from hatchet_sdk import Hatchet

from pydantic_ai.agent import AbstractAgent, WrapperAgent
from pydantic_ai.exceptions import UserError
from pydantic_ai.output import OutputDataT
from pydantic_ai.tools import (
    AgentDepsT,
)
from pydantic_ai.toolsets import AbstractToolset

from ._model import HatchetModel
from ._utils import TaskConfig


class HatchetAgent(WrapperAgent[AgentDepsT, OutputDataT]):
    def __init__(
        self,
        wrapped: AbstractAgent[AgentDepsT, OutputDataT],
        hatchet: Hatchet,
        *,
        name: str | None = None,
        mcp_task_config: TaskConfig | None = None,
        model_task_config: TaskConfig | None = None,
    ):
        """Wrap an agent to enable it with Hatchet durable tasks, by automatically offloading model requests, tool calls, and MCP server communication to Hatchet tasks.

        After wrapping, the original agent can still be used as normal outside of the Hatchet workflow.

        Args:
            wrapped: The agent to wrap.
            hatchet: The Hatchet instance to use for creating tasks.
        """
        super().__init__(wrapped)

        self._name = name or wrapped.name
        self._hatchet = hatchet

        if not self._name:
            raise UserError(
                "An agent needs to have a unique `name` in order to be used with DBOS. The name will be used to identify the agent's workflows and steps."
            )

        self._model = HatchetModel(
            wrapped.model,
            task_name_prefix=self._name,
            task_config=model_task_config or TaskConfig(),
            hatchet=self._hatchet,
        )
        hatchet_agent_name = self._name

        def hatchetify_toolset(toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            # Replace MCPServer with DBOSMCPServer
            try:
                from pydantic_ai.mcp import MCPServer

                from ._mcp_server import HatchetMCPServer
            except ImportError:
                pass
            else:
                if isinstance(toolset, MCPServer):
                    return HatchetMCPServer[AgentDepsT](
                        wrapped=toolset,
                        hatchet=hatchet,
                        task_name_prefix=hatchet_agent_name,
                        task_config=mcp_task_config or TaskConfig(),
                    )

            return toolset

        self._toolsets = [toolset.visit_and_replace(hatchetify_toolset) for toolset in wrapped.toolsets]
