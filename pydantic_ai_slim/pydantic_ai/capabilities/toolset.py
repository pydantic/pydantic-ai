from dataclasses import dataclass

from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from .abstract import AbstractCapability


@dataclass
class Toolset(AbstractCapability[AgentDepsT]):
    """A capability that provides a toolset."""

    toolset: AgentToolset[AgentDepsT]

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # Not spec-serializable (takes a callable)

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        return self.toolset
