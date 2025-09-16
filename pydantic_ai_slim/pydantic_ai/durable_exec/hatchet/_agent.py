from __future__ import annotations

from hatchet_sdk import Hatchet

from pydantic_ai.agent import AbstractAgent, WrapperAgent
from pydantic_ai.output import OutputDataT
from pydantic_ai.tools import (
    AgentDepsT,
)


class HatchetAgent(WrapperAgent[AgentDepsT, OutputDataT]):
    def __init__(
        self,
        wrapped: AbstractAgent[AgentDepsT, OutputDataT],
        hatchet: Hatchet,
    ):
        """Wrap an agent to enable it with Hatchet durable tasks, by automatically offloading model requests, tool calls, and MCP server communication to Hatchet tasks.

        After wrapping, the original agent can still be used as normal outside of the Hatchet workflow.

        Args:
            wrapped: The agent to wrap.
            hatchet: The Hatchet instance to use for creating tasks.
        """
        super().__init__(wrapped)

        self.hatchet = hatchet
