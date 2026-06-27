from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai._template import TemplateStr
from pydantic_ai.messages import InstructionPart

from . import _system_prompt
from .tools import SystemPromptFunc

AgentInstructions = (
    TemplateStr[AgentDepsT]
    | str
    | SystemPromptFunc[AgentDepsT]
    | Sequence[TemplateStr[AgentDepsT] | str | SystemPromptFunc[AgentDepsT]]
    | None
)


PreparedInstruction = str | _system_prompt.SystemPromptRunner[AgentDepsT]


def normalize_instructions(
    instructions: AgentInstructions[AgentDepsT],
) -> list[str | SystemPromptFunc[AgentDepsT]]:
    if instructions is None:
        return []
    # Note: TemplateStr is callable (__call__) so it's handled by the callable branch
    if isinstance(instructions, str) or callable(instructions):
        return [instructions]
    return list(instructions)


def prepare_instructions(
    instructions: AgentInstructions[AgentDepsT],
) -> list[PreparedInstruction[AgentDepsT]]:
    """Resolve raw instructions into their prepared form (`PreparedInstruction`s).

    Sits between `normalize_instructions` (which flattens the input into a list) and
    `resolve_instructions` (which runs the prepared items against a `RunContext`): static
    strings pass through unchanged, while functions and `TemplateStr`s are wrapped in a
    `SystemPromptRunner` so they can be invoked later. `None` (and other empty inputs) are
    valid and yield an empty list.
    """
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


def normalize_toolset_instructions(
    result: str | InstructionPart | Sequence[str | InstructionPart] | None,
) -> list[InstructionPart]:
    """Normalize a toolset `get_instructions` result into non-empty `InstructionPart`s.

    A toolset may return a single `str` or `InstructionPart`, a sequence of either, or `None`.
    Plain strings are treated as dynamic (they come from an external/changeable source) and
    whitespace-only content is dropped. Shared by `_agent_graph._get_instructions` and the
    deferred-capability loader's owned-toolset instruction collection so the two stay in sync.
    """
    if not result:
        return []
    items = [result] if isinstance(result, (str, InstructionPart)) else result
    parts: list[InstructionPart] = []
    for item in items:
        part = item if isinstance(item, InstructionPart) else InstructionPart(content=item, dynamic=True)
        if part.content.strip():
            parts.append(part)
    return parts


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
