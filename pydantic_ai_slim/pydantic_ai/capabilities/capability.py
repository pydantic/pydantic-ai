from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai._instructions import AgentInstructions
from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.toolsets import AgentToolset


@dataclass
class Capability(AbstractCapability[AgentDepsT]):
    """Convenience capability for bundling instructions and a toolset without subclassing.

    Use this when you just need to attach static instructions, a toolset, or a description
    to an agent. For dynamic behavior or lifecycle hooks, subclass
    [`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability] directly.
    """

    description: str | None = field(default=None, kw_only=True)
    """Human-readable description of what this capability provides."""

    instructions: AgentInstructions[AgentDepsT] | None = field(default=None, kw_only=True)
    """Instructions to include in the system prompt."""

    toolset: AgentToolset[AgentDepsT] | None = field(default=None, kw_only=True)
    """Toolset to register with the agent."""

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        return self.instructions

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        return self.toolset

    def get_description(self) -> str | None:
        return self.description
