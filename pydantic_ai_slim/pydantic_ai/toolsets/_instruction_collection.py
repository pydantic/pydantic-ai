from __future__ import annotations

from pydantic_ai._instructions import instruction_source_key
from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import InstructionPart

from .abstract import AbstractToolset


async def collect_toolset_instructions(
    toolset: AbstractToolset[AgentDepsT], ctx: RunContext[AgentDepsT]
) -> list[InstructionPart]:
    """Collect instruction parts once at the authoring toolset boundary."""
    contributions = await toolset._collect_instruction_contributions(ctx)  # pyright: ignore[reportPrivateUsage]
    return flatten_instruction_contributions(contributions)


def flatten_instruction_contributions(
    contributions: list[tuple[AbstractToolset[AgentDepsT], list[InstructionPart]]],
) -> list[InstructionPart]:
    """Flatten collected contributions, rejecting two sources that would claim one key.

    Separate from the collection itself so a toolset that has already gathered its children can
    apply the rule without re-entering the walk that produced them.
    """
    owners: dict[str, AbstractToolset[AgentDepsT]] = {}
    parts: list[InstructionPart] = []
    for source, source_parts in contributions:
        for part in source_parts:
            if part.id is None:
                continue
            # Ownership is claimed over the source key, not the whole id: two toolsets sharing an
            # `id` make `toolset:<id>` ambiguous even where each declares a different segment under
            # it, so comparing full ids would let exactly that pair through.
            source_key = instruction_source_key(part.id)
            if owners.setdefault(source_key, source) is not source:
                raise UserError(
                    f'Two toolsets have the same `id` {source_key.partition(":")[2]!r} and both contribute '
                    f'instructions, so {source_key!r} would address blocks from each. '
                    'Toolset `id`s must be unique among all toolsets registered with the same agent.'
                )
        parts.extend(source_parts)
    return parts
