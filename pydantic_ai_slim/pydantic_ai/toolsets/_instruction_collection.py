from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Generic

from pydantic_ai._instructions import validate_instruction_id_segment, validate_instruction_name
from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import InstructionId, InstructionPart, ToolsetInstructionSource

if TYPE_CHECKING:
    from .abstract import AbstractToolset


@dataclass(frozen=True)
class InstructionContribution(Generic[AgentDepsT]):
    """One source's instruction parts, with the key they are addressed by."""

    source: AbstractToolset[AgentDepsT]
    source_id: ToolsetInstructionSource | None
    parts: tuple[InstructionPart, ...]


def make_contribution(
    source: AbstractToolset[AgentDepsT], part: InstructionPart
) -> InstructionContribution[AgentDepsT]:
    """Attribute one surviving part to the source that authored it, minting that source's key.

    Takes a part rather than a `get_instructions` result because minting validates the source's
    `id`, and a source that says nothing to the model must never be the reason that fails. Callers
    normalize first and never reach here with nothing, so there is no empty case to mint for.
    """
    source_id = None
    resolved_part = part
    if part.name is not None:
        validate_instruction_name(part.name)
    key = source._instruction_key  # pyright: ignore[reportPrivateUsage]
    if key is not None:
        validate_instruction_id_segment(key, kind='Toolset id')
        source_id = ToolsetInstructionSource(key)
        if part.id is None or part.id.source != source_id:
            # A part claiming a key that isn't this source's is re-keyed beneath the source that
            # actually contributed it, keeping the name its author declared relative to itself.
            resolved_part = replace(part, id=InstructionId(source_id, name=part.name))
    elif part.id is not None:
        resolved_part = replace(part, id=None)
    return InstructionContribution(source=source, source_id=source_id, parts=(resolved_part,))


def framework_appended_instruction_ids(toolset: AbstractToolset[AgentDepsT]) -> set[InstructionId]:
    """The ids of blocks under `toolset` that append by the framework's choice rather than their author's."""
    ids: set[InstructionId] = set()
    toolset.apply(lambda leaf: ids.update(leaf._framework_appended_instruction_ids()))  # pyright: ignore[reportPrivateUsage]
    return ids


async def collect_toolset_instructions(
    toolset: AbstractToolset[AgentDepsT], ctx: RunContext[AgentDepsT]
) -> list[InstructionPart]:
    """Collect instruction parts once at the authoring toolset boundary."""
    contributions = await toolset._collect_instruction_contributions(ctx)  # pyright: ignore[reportPrivateUsage]
    return flatten_instruction_contributions(contributions)


def flatten_instruction_contributions(
    contributions: list[InstructionContribution[AgentDepsT]],
) -> list[InstructionPart]:
    """Flatten collected contributions, rejecting two sources that would claim one key.

    Separate from the collection itself so a toolset that has already gathered its children can
    apply the rule without re-entering the walk that produced them.
    """
    owners: dict[ToolsetInstructionSource, AbstractToolset[AgentDepsT]] = {}
    parts: list[InstructionPart] = []
    for contribution in contributions:
        if contribution.source_id is not None:
            source_id = contribution.source_id
            if (owner := owners.setdefault(source_id, contribution.source)) is not contribution.source:
                if owner.id is None or contribution.source.id is None:
                    raise UserError(
                        f'Two toolsets contribute instructions keyed {str(source_id)!r}, at least one of them '
                        'without an `id`, such as two `MCPToolset`s for servers on one host, which key their '
                        "instructions by the server's hostname. Pass a distinct `id` to each."
                    )
                raise UserError(
                    f'Two toolsets have the same `id` {source_id.id!r} and both contribute '
                    f'instructions, so {str(source_id)!r} would address parts from each. '
                    'Toolset `id`s must be unique among all toolsets registered with the same agent.'
                )
        parts.extend(contribution.parts)
    return parts
