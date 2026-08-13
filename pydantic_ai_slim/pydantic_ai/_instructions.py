from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, replace
from typing import Generic

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai._utils import dataclasses_no_defaults_repr
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import InstructionPart
from pydantic_ai.template import TemplateStr

from . import _system_prompt
from .tools import SystemPromptFunc

AgentInstruction = TemplateStr[AgentDepsT] | str | InstructionPart | SystemPromptFunc[AgentDepsT]
"""One instruction: literal text, a function computing it, or an `InstructionPart` declaring both the
text and how it should be treated — its [`id`][pydantic_ai.messages.InstructionPart.id] (resolved
against the key of whatever source contributes it) and whether it counts as
[`dynamic`][pydantic_ai.messages.InstructionPart.dynamic] for prompt caching."""

AgentInstructions = AgentInstruction[AgentDepsT] | Sequence[AgentInstruction[AgentDepsT]] | None


AGENT_INSTRUCTION_ID = 'agent'
"""The [`InstructionPart.id`][pydantic_ai.messages.InstructionPart.id] source key for the agent itself.

Bare rather than namespaced: the agent is the one source there can only be one of, and every other
source key is prefixed, so none of them can claim it.
"""


def validate_instruction_id_segment(id: str, *, kind: str) -> None:
    """Reject values that cannot be represented unambiguously in an instruction id."""
    if ':' in id:
        raise UserError(f'{kind} {id!r} cannot contain a colon because `:` is reserved as an instruction ID delimiter.')


TOOLSET_INSTRUCTION_NAMESPACE = 'toolset'
"""The segment every toolset's source key starts with.

The one namespace `normalize_toolset_instructions` can see, which is what lets it tell a key it
already issued from a raw value an author wrote.
"""


def toolset_instruction_id(toolset_id: str) -> str:
    """The [`InstructionPart.id`][pydantic_ai.messages.InstructionPart.id] source key for a toolset."""
    validate_instruction_id_segment(toolset_id, kind='Toolset id')
    return f'{TOOLSET_INSTRUCTION_NAMESPACE}:{toolset_id}'


def capability_instruction_id(capability_id: str) -> str:
    """The [`InstructionPart.id`][pydantic_ai.messages.InstructionPart.id] source key for a capability."""
    validate_instruction_id_segment(capability_id, kind='Capability id')
    return f'capability:{capability_id}'


def declared_instruction_id(source_id: str, declared_id: str) -> str:
    """The [`InstructionPart.id`][pydantic_ai.messages.InstructionPart.id] for one declared block of a source.

    One rule, applied uniformly: a source key addresses everything that source contributes, and
    appending a segment addresses a single block the author declared within it — `'agent:local_time'`,
    `'capability:budget:note'`. A declared segment can therefore never collide with a source key.
    """
    validate_instruction_id_segment(declared_id, kind='Declared instruction id')
    return f'{source_id}:{declared_id}'


def resolve_declared_id(source_id: str | None, declared_id: str | None) -> str | None:
    """Resolve one block's declared id against the key of the source that contributed it.

    The single place the rule lives, so every source applies it the same way: an author writing a
    block declares a bare name for it (`'limits'`) and the framework qualifies that against the
    source key it belongs to (`'toolset:weather:limits'`). Authors don't repeat their own identity,
    and can't accidentally claim a top-level key — including one that already means something, like
    the agent's own `'agent'`.

    Without a source key there is nothing for a declared segment to hang off, so the block stays
    unidentified rather than claiming a top-level key of its own. An id that already carries the
    source key is passed through, so writing the qualified form yields what the author meant rather
    than `'toolset:weather:toolset:weather:limits'`.
    """
    if source_id is None:
        return None
    if declared_id is None:
        return source_id
    if declared_id == source_id or declared_id.startswith(f'{source_id}:'):
        return declared_id
    return declared_instruction_id(source_id, declared_id)


@dataclass(frozen=True, repr=False)
class SourcedInstruction(Generic[AgentDepsT]):
    """A lazy instruction recipe with the `InstructionPart.id` its content should be addressed by."""

    instruction: AgentInstruction[AgentDepsT]

    _: KW_ONLY

    id: str | None = None
    dynamic: bool = False

    __repr__ = dataclasses_no_defaults_repr


async def resolve_sourced_instructions(
    instructions: Sequence[SourcedInstruction[AgentDepsT]], run_context: RunContext[AgentDepsT]
) -> list[InstructionPart]:
    """Resolve authored instructions into the parts sent to the model.

    Literal strings with the same source key form one addressable block. An
    [`InstructionPart`][pydantic_ai.messages.InstructionPart] always remains independent so its
    cache treatment applies only to its own text, while callable instructions are resolved lazily
    against the current `RunContext`.
    """
    parts: list[InstructionPart] = []
    group: list[InstructionPart] = []
    group_key: str | None = None

    def flush_group() -> None:
        if content := InstructionPart.join(group):
            parts.append(InstructionPart(content=content, id=group[0].id))
        group.clear()

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
            flush_group()
            group_key = None
            if content := await _system_prompt.SystemPromptRunner[AgentDepsT](instruction).run(run_context):
                parts.append(InstructionPart(content=content, id=sourced.id, dynamic=sourced.dynamic))
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


def normalize_toolset_instructions(
    result: str | InstructionPart | Sequence[str | InstructionPart] | None,
    toolset_id: str | None = None,
) -> list[InstructionPart]:
    """Normalize a toolset `get_instructions` result into non-empty `InstructionPart`s.

    A toolset may return a single `str` or `InstructionPart`, a sequence of either, or `None`.
    Plain strings are treated as dynamic (they come from an external/changeable source) and
    whitespace-only content is dropped. Shared by `_agent_graph._get_instructions` and the
    deferred-capability loader's owned-toolset instruction collection so the two stay in sync.

    When `toolset_id` is provided, each part's [`id`][pydantic_ai.messages.InstructionPart.id] is
    resolved against the toolset's key by `resolve_declared_id`, so a part without one is addressed
    by `'toolset:<id>'` and a part declaring `'limits'` by `'toolset:<id>:limits'`. Composition
    points pass the id of the toolset they're calling, and only the point that reaches the authoring
    toolset has one to pass — a wrapper reports no `id`, so nothing resolves an id twice.

    Without one, an id the framework already issued passes straight through (that's an outer layer
    seeing a part an inner one resolved), while anything else is dropped: an author writing on a
    toolset with no `id` has no source key to hang a declared segment off, and letting the raw value
    stand would let it claim a key belonging to somebody else — `'agent'`, say.
    """
    if not result:
        return []
    items = [result] if isinstance(result, (str, InstructionPart)) else result
    source_id = toolset_instruction_id(toolset_id) if toolset_id is not None else None
    parts: list[InstructionPart] = []
    for item in items:
        part = item if isinstance(item, InstructionPart) else InstructionPart(content=item, dynamic=True)
        if not part.content.strip():
            continue
        if source_id is not None:
            resolved_id = resolve_declared_id(source_id, part.id)
        elif part.id is not None and not part.id.startswith(f'{TOOLSET_INSTRUCTION_NAMESPACE}:'):
            resolved_id = None
        else:
            resolved_id = part.id
        if resolved_id != part.id:
            part = replace(part, id=resolved_id)
        parts.append(part)
    return parts
