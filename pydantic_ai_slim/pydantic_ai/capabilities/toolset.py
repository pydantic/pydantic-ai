from dataclasses import dataclass

from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset

from .abstract import AbstractCapability


@dataclass
class Toolset(AbstractCapability[AgentDepsT]):
    """A capability that provides a toolset."""

    toolset: AbstractToolset[AgentDepsT]

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
        return self.toolset
