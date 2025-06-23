"""Create a Pydantic AI agent and AG-UI adapter."""

from __future__ import annotations

from dataclasses import dataclass
from types import NoneType
from typing import Generic

from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.ag_ui import Adapter
from pydantic_ai.result import OutputDataT
from pydantic_ai.tools import AgentDepsT


@dataclass(init=False, repr=False)
class AGUIAgent(Generic[AgentDepsT, OutputDataT]):
    """Pydantic AI agent with AG-UI adapter."""

    agent: Agent[AgentDepsT, str]
    adapter: Adapter[AgentDepsT, str]
    instructions: str | None

    def __init__(
        self, deps_type: type[AgentDepsT] = NoneType, instructions: str | None = None
    ) -> None:
        """Initialize the API agent with AG-UI adapter.

        Args:
            deps_type: Type annotation for the agent dependencies.
            instructions: Optional instructions for the agent.
        """
        # Ensure environment variables are loaded.
        load_dotenv()

        self.agent = Agent(
            'openai:gpt-4o-mini',
            output_type=str,
            instructions=instructions,
            deps_type=deps_type,
        )
        self.adapter = self.agent.to_ag_ui()
