from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai import _instructions
from pydantic_ai._template import TemplateStr
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.tools import AgentDepsT


@dataclass
class Instructions(AbstractCapability[AgentDepsT]):
    """A capability that provides static or dynamic instructions."""

    instructions: _instructions.AgentInstructions[AgentDepsT]

    def get_instructions(self) -> _instructions.AgentInstructions[AgentDepsT] | None:
        return self.instructions

    @classmethod
    def from_spec(cls, instructions: TemplateStr[AgentDepsT] | str = '') -> Instructions[Any]:
        """Create from spec.

        Narrower than the full `Instructions` type alias (which also accepts callables
        and sequences) because specs only support literal strings and template strings.
        """
        return cls(instructions=instructions)
