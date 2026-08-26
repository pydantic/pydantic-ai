from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic

from pydantic_ai._instructions import normalize_toolset_instruction_parts, qualify_toolset_instruction_parts
from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import InstructionPart

if TYPE_CHECKING:
    from .abstract import AbstractToolset


@dataclass(frozen=True)
class InstructionContribution(Generic[AgentDepsT]):
    """One source's instruction blocks, with the key they are addressed by."""

    source: AbstractToolset[AgentDepsT]
    source_id: str | None
    parts: tuple[InstructionPart, ...]


def make_contribution(
    source: AbstractToolset[AgentDepsT],
    result: str | InstructionPart | Sequence[str | InstructionPart] | None,
) -> list[InstructionContribution[AgentDepsT]]:
    """Build one source's contribution, minting its key only once a block has survived.

    The only way a contribution is made, so the ordering holds everywhere: normalize and strip
    first, and return nothing at all when there is nothing left. Minting validates the toolset's
    `id`, and a source that says nothing to the model must never be the reason that fails.
    """
    parts = normalize_toolset_instruction_parts(result)
    if not parts:
        return []
    source_id, parts = qualify_toolset_instruction_parts(parts, source.id)
    return [InstructionContribution(source=source, source_id=source_id, parts=tuple(parts))]


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
    owners: dict[str, AbstractToolset[AgentDepsT]] = {}
    parts: list[InstructionPart] = []
    for contribution in contributions:
        if contribution.source_id is not None:
            source_id = contribution.source_id
            if owners.setdefault(source_id, contribution.source) is not contribution.source:
                raise UserError(
                    f'Two toolsets have the same `id` {source_id.partition(":")[2]!r} and both contribute '
                    f'instructions, so {source_id!r} would address blocks from each. '
                    'Toolset `id`s must be unique among all toolsets registered with the same agent.'
                )
        parts.extend(contribution.parts)
    return parts
