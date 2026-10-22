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
    derived_key: bool = False
    """Whether the framework chose the key rather than an `id` the author set on the contributing toolset.

    An `MCPToolset` without an `id` keys its server instructions by its URL's hostname, and an id-less
    `DynamicToolset` passes its built `MCPToolset`'s key along. Two such contributions that land on one
    key fall back to unkeyed, rewriting instructions instead of colliding.
    """


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
    return InstructionContribution(
        source=source,
        source_id=source_id,
        parts=(resolved_part,),
        derived_key=source_id is not None and source.id is None,
    )


def framework_appended_instruction_ids(toolset: AbstractToolset[AgentDepsT]) -> set[InstructionId]:
    """The ids of blocks under `toolset` that append by the framework's choice rather than their author's.

    Keyed the way collection keys them: a toolset that passes its instruction children along leaves
    their keys as they are, and one that speaks for what's below it without passing it along, such as
    a `DynamicToolset` with an `id`, keys everything its leaves contribute under its own.
    """
    ids: set[InstructionId] = set()

    def visit(node: AbstractToolset[AgentDepsT]) -> None:
        if children := node._instruction_children():  # pyright: ignore[reportPrivateUsage]
            for child in children:
                visit(child)
            return
        leaf_ids: set[InstructionId] = set()
        node.apply(lambda leaf: leaf_ids.update(leaf._framework_appended_instruction_ids()))  # pyright: ignore[reportPrivateUsage]
        if (source := node._instruction_source()) is not None:  # pyright: ignore[reportPrivateUsage]
            ids.update(InstructionId(source, name=leaf_id.name) for leaf_id in leaf_ids)

    visit(toolset)
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
    """Flatten collected contributions, rejecting two sources that would claim one key by their `id`s.

    A contribution whose key the framework derived, such as an id-less `MCPToolset`'s server host,
    doesn't make that a collision: when another source shares its key, its parts go out unkeyed and
    rewriting, as they did before keys were derived, so only explicitly chosen `id`s can collide.

    Separate from the collection itself so a toolset that has already gathered its children can
    apply the rule without re-entering the walk that produced them.
    """
    explicit_owners: dict[ToolsetInstructionSource, AbstractToolset[AgentDepsT]] = {}
    sources_by_key: dict[ToolsetInstructionSource, set[int]] = {}
    for contribution in contributions:
        if (source_id := contribution.source_id) is None:
            continue
        sources_by_key.setdefault(source_id, set()).add(id(contribution.source))
        if contribution.derived_key:
            continue
        if explicit_owners.setdefault(source_id, contribution.source) is not contribution.source:
            raise UserError(
                f'Two toolsets have the same `id` {source_id.id!r} and both contribute '
                f'instructions, so {str(source_id)!r} would address parts from each. '
                'Toolset `id`s must be unique among all toolsets registered with the same agent.'
            )
    parts: list[InstructionPart] = []
    for contribution in contributions:
        if (
            contribution.derived_key
            and contribution.source_id is not None
            and len(sources_by_key[contribution.source_id]) > 1
        ):
            parts.extend(replace(part, id=None, on_change='rewrite') for part in contribution.parts)
        else:
            parts.extend(contribution.parts)
    return parts
