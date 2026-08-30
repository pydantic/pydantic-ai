from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, replace
from typing import Generic

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai._utils import dataclasses_no_defaults_repr
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import InstructionId, InstructionPart, InstructionSource
from pydantic_ai.template import TemplateStr

from . import _system_prompt
from .tools import SystemPromptFunc

AgentInstruction = TemplateStr[AgentDepsT] | str | InstructionPart | SystemPromptFunc[AgentDepsT]
"""One instruction: literal text, a function computing it, or an `InstructionPart` declaring both the
text and how it should be treated — its [`name`][pydantic_ai.messages.InstructionPart.name] (qualified
into an id against whatever source contributes it) and whether it counts as
[`dynamic`][pydantic_ai.messages.InstructionPart.dynamic] for prompt caching."""

AgentInstructions = AgentInstruction[AgentDepsT] | Sequence[AgentInstruction[AgentDepsT]] | None


def validate_instruction_id_segment(id: str, *, kind: str) -> None:
    """Reject values that cannot be represented unambiguously in an instruction id."""
    if ':' in id:
        raise UserError(f'{kind} {id!r} cannot contain a colon because `:` is reserved as an instruction ID delimiter.')


def validate_instruction_name(name: str) -> None:
    """Reject names an author cannot declare on an instruction part.

    A name is one segment of the id built around it, so it must not be able to spell a key by itself.
    `'agent'` is the only one it could reach: every other key is namespaced, and a colon is rejected
    above.
    """
    validate_instruction_id_segment(name, kind='Instruction name')
    if name == 'agent':
        raise UserError(
            "Instruction name 'agent' is reserved for the agent's own instructions; choose a different name."
        )


@dataclass(frozen=True, repr=False)
class SourcedInstruction(Generic[AgentDepsT]):
    """A lazy instruction recipe with the name and key its content should be addressed by."""

    instruction: AgentInstruction[AgentDepsT]

    _: KW_ONLY

    name: str | None = None
    id: InstructionId | None = None
    dynamic: bool = False

    __repr__ = dataclasses_no_defaults_repr


def sourced_instruction(
    instruction: AgentInstruction[AgentDepsT], source: InstructionSource | None
) -> SourcedInstruction[AgentDepsT]:
    """Attribute one instruction recipe to the source that authored it.

    The single place a declared name meets its source, so every author applies the same rule: with a
    source the name is qualified into an [`InstructionId`][pydantic_ai.messages.InstructionId] beneath
    it, and without one there is no key to qualify against, so the name stays a name and the part
    stays unaddressable.

    A caller passes `None` for a recipe its source does not speak for -- a callable the agent was
    built with, or instructions belonging to a single run rather than to the agent.
    """
    name = instruction.name if isinstance(instruction, InstructionPart) else None
    if name is not None:
        validate_instruction_name(name)
    return SourcedInstruction(
        instruction,
        name=name,
        id=InstructionId(source, name=name) if source is not None else None,
        dynamic=not isinstance(instruction, (str, InstructionPart)),
    )


async def resolve_sourced_instructions(
    instructions: Sequence[SourcedInstruction[AgentDepsT]], run_context: RunContext[AgentDepsT]
) -> list[InstructionPart]:
    """Resolve authored instructions into the parts sent to the model.

    Literal strings with the same source key form one addressable part. An
    [`InstructionPart`][pydantic_ai.messages.InstructionPart] always remains independent so its
    cache treatment applies only to its own text, while callable instructions are resolved lazily
    against the current `RunContext`.
    """
    parts: list[InstructionPart] = []
    group: list[InstructionPart] = []
    pending_parts: list[InstructionPart] = []
    group_key: InstructionId | None = None

    def flush_group() -> None:
        if content := InstructionPart.join(group):
            parts.append(InstructionPart(content=content, id=group[0].id))
        group.clear()
        parts.extend(pending_parts)
        pending_parts.clear()

    for sourced in instructions:
        instruction = sourced.instruction
        if isinstance(instruction, InstructionPart):
            if not (content := instruction.content.strip()):
                continue
            flush_group()
            group_key = None
            parts.append(replace(instruction, content=content, id=sourced.id))
        elif isinstance(instruction, str):
            if not (content := instruction.strip()):
                continue
            if group and (sourced.id is None or group_key != sourced.id):
                flush_group()
            group_key = sourced.id
            group.append(InstructionPart(content=content, id=sourced.id))
        else:
            if content := await _system_prompt.SystemPromptRunner[AgentDepsT](instruction).run(run_context):
                part = InstructionPart(content=content, name=sourced.name, id=sourced.id, dynamic=sourced.dynamic)
                if group:
                    pending_parts.append(part)
                else:
                    parts.append(part)
    flush_group()
    return parts


def normalize_instructions(
    instructions: AgentInstructions[AgentDepsT],
) -> list[AgentInstruction[AgentDepsT]]:
    if instructions is None:
        return []
    # Note: TemplateStr is callable (__call__) so it's handled by the callable branch
    if isinstance(instructions, (str, InstructionPart)) or callable(instructions):
        return [instructions]
    return list(instructions)


def normalize_toolset_instruction_parts(
    result: str | InstructionPart | Sequence[str | InstructionPart] | None,
) -> list[InstructionPart]:
    """Normalize a toolset `get_instructions` result into non-empty parts, ids untouched.

    A toolset may return a single `str` or `InstructionPart`, a sequence of either, or `None`. Plain
    strings are treated as dynamic (they come from an external/changeable source) and whitespace-only
    content is dropped. Ids are left exactly as the author wrote them, so whoever interprets them can
    still tell a key issued below from a segment declared here.
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
