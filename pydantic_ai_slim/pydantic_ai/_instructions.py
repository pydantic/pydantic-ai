from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai._template import TemplateStr

from . import _system_prompt

AgentInstructions = (
    TemplateStr[AgentDepsT]
    | str
    | _system_prompt.SystemPromptFunc[AgentDepsT]
    | Sequence[TemplateStr[AgentDepsT] | str | _system_prompt.SystemPromptFunc[AgentDepsT]]
    | None
)

PreparedInstruction = str | _system_prompt.SystemPromptRunner[AgentDepsT]


def normalize_instructions(
    instructions: AgentInstructions[AgentDepsT],
) -> list[str | _system_prompt.SystemPromptFunc[AgentDepsT]]:
    if instructions is None:
        return []
    # Note: TemplateStr is callable (__call__) so it's handled by the callable branch
    if isinstance(instructions, str) or callable(instructions):
        return [instructions]
    return list(instructions)


def prepare_instructions(
    instructions: AgentInstructions[AgentDepsT],
) -> list[PreparedInstruction[AgentDepsT]]:
    prepared: list[PreparedInstruction[AgentDepsT]] = []
    for instruction in normalize_instructions(instructions):
        if isinstance(instruction, str):
            prepared.append(instruction)
        else:
            # TemplateStr instances land here too: they are callable with a
            # RunContext parameter, so SystemPromptRunner handles them like
            # any other system prompt function.
            prepared.append(_system_prompt.SystemPromptRunner[AgentDepsT](instruction))
    return prepared


async def resolve_instructions(
    instructions: AgentInstructions[AgentDepsT],
    run_context: RunContext[AgentDepsT],
) -> list[str]:
    parts: list[str] = []
    for instruction in prepare_instructions(instructions):
        if isinstance(instruction, str):
            parts.append(instruction)
        else:
            resolved = await instruction.run(run_context)
            if resolved is not None:
                parts.append(resolved)
    return parts
