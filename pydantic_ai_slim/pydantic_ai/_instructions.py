from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic

from pydantic_ai._template import TemplateStr
from pydantic_ai.tools import AgentDepsT

from . import _system_prompt

InstructionContent = TemplateStr[AgentDepsT] | str | _system_prompt.SystemPromptFunc[AgentDepsT]


@dataclass
class Instruction(Generic[AgentDepsT]):
    content: InstructionContent[AgentDepsT]
    defer_loading: bool | None = None
    capability_id: str | None = None


InstructionInput = InstructionContent[AgentDepsT] | Instruction[AgentDepsT]
AgentInstructions = InstructionInput[AgentDepsT] | Sequence[InstructionInput[AgentDepsT]] | None


# We changed a lot of stuff here to make Instructions capture the defer_loading and capability_id metadata which is important for us to retain this information for later


def normalize_instructions(
    instructions: AgentInstructions[AgentDepsT],
) -> list[Instruction[AgentDepsT]]:
    if instructions is None:
        return []

    if isinstance(instructions, Instruction):
        return [instructions]

    if isinstance(instructions, str) or callable(instructions):
        return [Instruction(content=instructions)]

    return [
        instruction if isinstance(instruction, Instruction) else Instruction(content=instruction)
        for instruction in instructions
    ]
