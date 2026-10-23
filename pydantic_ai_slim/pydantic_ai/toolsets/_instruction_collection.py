from __future__ import annotations

from contextvars import ContextVar
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
    framework_appended: bool = False
    """Whether these parts use `on_change='append'` by the framework's choice rather than their author's.

    Set where such a part is made, by `mark_framework_appended`, and carried with the part through every toolset that passes it along or keys it anew, so it never
    has to be inferred from an id another block could share.
    """


_FRAMEWORK_APPENDED_PARTS: ContextVar[dict[int, InstructionPart] | None] = ContextVar(
    '_FRAMEWORK_APPENDED_PARTS', default=None
)
"""The part objects marked framework-appended during the collection in progress, by identity."""


def mark_framework_appended(part: InstructionPart) -> InstructionPart:
    """Record that `part` uses `on_change='append'` by the framework's choice, for the collection in progress.

    A toolset that appends its blocks by default, rather than because its author asked, returns its
    parts through this, so where appending isn't supported they fall back to rewriting without the
    warning an author's own opt-in gets. Tracked by object identity, so a block another toolset builds,
    or that ends up under the same id, is never mistaken for it.
    """
    if (marked := _FRAMEWORK_APPENDED_PARTS.get()) is not None:
        marked[id(part)] = part
    return part


def is_framework_appended(part: InstructionPart) -> bool:
    """Whether this part object was marked by `mark_framework_appended` during the collection in progress."""
    marked = _FRAMEWORK_APPENDED_PARTS.get()
    return part.on_change == 'append' and marked is not None and marked.get(id(part)) is part


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
        framework_appended=is_framework_appended(part),
    )


async def collect_toolset_instructions(
    toolset: AbstractToolset[AgentDepsT], ctx: RunContext[AgentDepsT]
) -> list[InstructionPart]:
    """Collect instruction parts once at the authoring toolset boundary."""
    parts, _ = await collect_toolset_instructions_with_framework_parts(toolset, ctx)
    return parts


async def collect_toolset_instructions_with_framework_parts(
    toolset: AbstractToolset[AgentDepsT], ctx: RunContext[AgentDepsT]
) -> tuple[list[InstructionPart], list[InstructionPart]]:
    """Collect instruction parts, and which of them append by the framework's choice rather than their author's.

    The second list holds the very part objects in the first that are framework-appended, so a consumer
    tells them apart by identity rather than by an id an author's block could share.
    """
    token = _FRAMEWORK_APPENDED_PARTS.set({})
    try:
        contributions = await toolset._collect_instruction_contributions(ctx)  # pyright: ignore[reportPrivateUsage]
        flattened = flatten_marked_contributions(contributions)
    finally:
        _FRAMEWORK_APPENDED_PARTS.reset(token)
    return [part for part, _ in flattened], [part for part, framework in flattened if framework]


def flatten_instruction_contributions(
    contributions: list[InstructionContribution[AgentDepsT]],
) -> list[InstructionPart]:
    """Flatten collected contributions, rejecting two sources that would claim one key by their `id`s.

    A contribution whose key the framework derived, such as an id-less `MCPToolset`'s server host,
    doesn't make that a collision: when another source shares its key, its parts go out unkeyed and
    rewriting, as they did before keys were derived, so only explicitly chosen `id`s can collide.

    Separate from the collection itself so a toolset that has already gathered its children can
    apply the rule without re-entering the walk that produced them. The framework-appended parts it
    returns stay marked, so a toolset that keys them anew still knows them for what they are.
    """
    return [part for part, _ in flatten_marked_contributions(contributions)]


def flatten_marked_contributions(
    contributions: list[InstructionContribution[AgentDepsT]],
) -> list[tuple[InstructionPart, bool]]:
    """Flatten contributions as `flatten_instruction_contributions` does, pairing each part with whether it's framework-appended."""
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
    parts: list[tuple[InstructionPart, bool]] = []
    for contribution in contributions:
        if (
            contribution.derived_key
            and contribution.source_id is not None
            and len(sources_by_key[contribution.source_id]) > 1
        ):
            # Rewriting, an empty block says nothing, so it's dropped like any other empty rewrite-mode part.
            parts.extend(
                (replace(part, id=None, on_change='rewrite'), False)
                for part in contribution.parts
                if part.content.strip()
            )
            continue
        framework = contribution.framework_appended
        for part in contribution.parts:
            parts.append((mark_framework_appended(part) if framework else part, framework))
    return parts
