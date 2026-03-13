from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai import _instructions
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.tools import AgentDepsT


@dataclass
class Instructions(AbstractCapability[AgentDepsT]):
    """A capability that provides static or dynamic instructions."""

    instructions: _instructions.Instructions[AgentDepsT]

    def get_instructions(self) -> _instructions.Instructions[AgentDepsT] | None:
        return self.instructions

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> Instructions[Any]:
        """Create from spec. Accepts a string instruction (since __init__ also takes callables)."""
        return cls(*args, **kwargs)
