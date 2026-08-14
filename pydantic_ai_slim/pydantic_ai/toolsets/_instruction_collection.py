from __future__ import annotations

from pydantic_ai._instructions import normalize_toolset_instructions
from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai._utils import gather
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import InstructionPart

from .abstract import AbstractToolset
from .combined import CombinedToolset
from .wrapper import WrapperToolset


async def collect_toolset_instructions(
    toolset: AbstractToolset[AgentDepsT], ctx: RunContext[AgentDepsT]
) -> list[InstructionPart]:
    """Collect instruction parts once at the authoring toolset boundary."""
    contributions = await _collect_toolset_instructions(toolset, ctx)
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


async def _collect_toolset_instructions(
    toolset: AbstractToolset[AgentDepsT], ctx: RunContext[AgentDepsT]
) -> list[tuple[AbstractToolset[AgentDepsT], list[InstructionPart]]]:
    if isinstance(toolset, WrapperToolset) and type(toolset).get_instructions is WrapperToolset.get_instructions:
        return await _collect_toolset_instructions(toolset.wrapped, ctx)
    if isinstance(toolset, CombinedToolset) and type(toolset).get_instructions is CombinedToolset.get_instructions:
        child_contributions = await gather(*(_collect_toolset_instructions(child, ctx) for child in toolset.toolsets))
        return [contribution for contributions in child_contributions for contribution in contributions]

    result = await toolset.get_instructions(ctx)
    source = toolset
    while isinstance(source, WrapperToolset):
        source = source.wrapped
    return [(source, normalize_toolset_instructions(result, toolset.id))]
