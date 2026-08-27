from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic

from pydantic_ai._instructions import qualify_toolset_instruction_parts
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
    source: AbstractToolset[AgentDepsT], part: InstructionPart
) -> InstructionContribution[AgentDepsT]:
    """Attribute one surviving block to the source that authored it, minting that source's key.

    Takes a block rather than a `get_instructions` result because minting validates the source's
    `id`, and a source that says nothing to the model must never be the reason that fails. Callers
    normalize first and never reach here with nothing, so there is no empty case to mint for.
    """
    source_id, parts = qualify_toolset_instruction_parts([part], source.id)
    return InstructionContribution(source=source, source_id=source_id, parts=tuple(parts))


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
