from __future__ import annotations

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
    sources_by_id: dict[str, AbstractToolset[AgentDepsT]] = {}
    parts: list[InstructionPart] = []
    for source, source_parts in contributions:
        for part in source_parts:
            if part.id is not None and (existing := sources_by_id.setdefault(part.id, source)) is not source:
                raise UserError(
                    f'Two toolsets have the same `id` {existing.id!r} and both contribute instructions, '
                    f'so {part.id!r} would address blocks from each. '
                    'Toolset `id`s must be unique among all toolsets registered with the same agent.'
                )
        parts.extend(source_parts)
    return parts
