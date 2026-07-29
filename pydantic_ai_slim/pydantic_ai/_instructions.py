from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Generic

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.messages import InstructionPart
from pydantic_ai.template import TemplateStr

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


AGENT_INSTRUCTION_ID = 'agent'
"""The [`InstructionPart.id`][pydantic_ai.messages.InstructionPart.id] source key for the agent itself.

Bare rather than namespaced: the agent is the one source there can only be one of, and every other
source key is prefixed, so none of them can claim it.
"""


def toolset_instruction_id(toolset_id: str) -> str:
    """The [`InstructionPart.id`][pydantic_ai.messages.InstructionPart.id] source key for a toolset."""
    return f'toolset:{toolset_id}'


def capability_instruction_id(capability_id: str) -> str:
    """The [`InstructionPart.id`][pydantic_ai.messages.InstructionPart.id] source key for a capability."""
    return f'capability:{capability_id}'


def declared_instruction_id(source_id: str, declared_id: str) -> str:
    """The [`InstructionPart.id`][pydantic_ai.messages.InstructionPart.id] for one declared block of a source.

    One rule, applied uniformly: a source key addresses everything that source contributes, and
    appending a segment addresses a single block the author declared within it — `'agent:local_time'`,
    `'capability:budget:note'`. A declared segment can therefore never collide with a source key.
    """
    return f'{source_id}:{declared_id}'


@dataclass(frozen=True)
class SourcedInstruction(Generic[AgentDepsT]):
    """An agent-level instruction along with the `InstructionPart.id` its content should be addressed by."""

    instruction: str | SystemPromptFunc[AgentDepsT]
    id: str | None = None


@dataclass(frozen=True)
class DeclaredInstruction(Generic[AgentDepsT]):
    """An instruction with the id its author declared for it, relative to the source that holds it.

    Sources whose own key isn't known until collection time (a capability's, which depends on its
    `id`) store this and combine the two halves then.
    """

    instruction: str | SystemPromptFunc[AgentDepsT]
    declared_id: str | None = None


@dataclass(frozen=True)
class SourcedInstructionRunner(Generic[AgentDepsT]):
    """A prepared instruction function along with the `InstructionPart.id` its output should be addressed by."""

    runner: _system_prompt.SystemPromptRunner[AgentDepsT]
    id: str | None = None


def source_instructions(
    instructions: Sequence[str | SystemPromptFunc[AgentDepsT]], id: str | None
) -> list[SourcedInstruction[AgentDepsT]]:
    """Attribute every one of `instructions` to the source identified by `id`.

    Literal and computed instructions alike: addressing a source's id means addressing everything
    it tells the model, the same way a toolset's id covers every block it returns.
    """
    return [SourcedInstruction(instruction, id) for instruction in instructions]


def source_declared_instructions(
    instructions: Sequence[DeclaredInstruction[AgentDepsT]], source_id: str | None
) -> list[SourcedInstruction[AgentDepsT]]:
    """Resolve each declared id against its source key, falling back to the source key itself.

    Without a source key there is nothing for a declared id to hang off, so those blocks stay
    unidentified rather than claiming a top-level key of their own.
    """
    if source_id is None:
        return [SourcedInstruction(declared.instruction) for declared in instructions]
    return [
        SourcedInstruction(
            declared.instruction,
            declared_instruction_id(source_id, declared.declared_id) if declared.declared_id else source_id,
        )
        for declared in instructions
    ]


def source_agent_instructions(
    instructions: Sequence[str | SystemPromptFunc[AgentDepsT]],
) -> list[SourcedInstruction[AgentDepsT]]:
    """Attribute the agent's own instructions to `AGENT_INSTRUCTION_ID`, literals only.

    `'agent'` names the base prompt, and taking that over must not silently swallow an
    `@agent.instructions` function that injects the date or the user's name — mixing the two is
    routine on the agent itself. Such a function opts in separately via `@agent.instructions(id=...)`.
    """
    return [
        SourcedInstruction(instruction, AGENT_INSTRUCTION_ID if isinstance(instruction, str) else None)
        for instruction in instructions
    ]


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
    toolset_id: str | None = None,
) -> list[InstructionPart]:
    """Normalize a toolset `get_instructions` result into non-empty `InstructionPart`s.

    A toolset may return a single `str` or `InstructionPart`, a sequence of either, or `None`.
    Plain strings are treated as dynamic (they come from an external/changeable source) and
    whitespace-only content is dropped. Shared by `_agent_graph._get_instructions` and the
    deferred-capability loader's owned-toolset instruction collection so the two stay in sync.

    When `toolset_id` is provided, parts that don't already have an
    [`id`][pydantic_ai.messages.InstructionPart.id] are stamped with the contributing toolset's,
    so consumers can address them. Composition points pass the id of the toolset they're calling:
    an id already set by a toolset closer to the source always wins.
    """
    if not result:
        return []
    items = [result] if isinstance(result, (str, InstructionPart)) else result
    parts: list[InstructionPart] = []
    for item in items:
        part = item if isinstance(item, InstructionPart) else InstructionPart(content=item, dynamic=True)
        if not part.content.strip():
            continue
        if part.id is None and toolset_id is not None:
            part = replace(part, id=toolset_instruction_id(toolset_id))
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
