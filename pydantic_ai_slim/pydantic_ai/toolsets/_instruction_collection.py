from __future__ import annotations

import warnings
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Generic

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


@dataclass(frozen=True)
class RelayedPart:
    """A part a child contributed while an authoring container's `get_instructions` ran."""

    source: AbstractToolset[Any]
    source_id: ToolsetInstructionSource | None
    part: InstructionPart
    """The part object itself, held so its identity can't be reused by another object meanwhile."""
    snapshot: InstructionPart
    """A copy of the part as the child contributed it, so a part edited in place no longer counts as relayed."""


@dataclass
class Relays:
    """What an authoring container's declared instruction children contributed while its `get_instructions` ran."""

    author: AbstractToolset[Any]
    parts: dict[int, RelayedPart] = field(default_factory=dict[int, RelayedPart])
    """The relayed parts, by part identity."""
    _tree: set[int] | None = None

    def covers(self, toolset: AbstractToolset[Any]) -> bool:
        """Whether `toolset` is the author or below it through `_instruction_children`.

        A toolset the author reaches some other way, like the one a `DynamicToolset` builds, isn't a child
        the author passes along: what it returns is the author's own text.
        """
        if self._tree is None:
            tree: set[int] = set()
            pending: list[AbstractToolset[Any]] = [self.author]
            while pending:
                node = pending.pop()
                tree.add(id(node))
                pending.extend(node._instruction_children())  # pyright: ignore[reportPrivateUsage]
            self._tree = tree
        return id(toolset) in self._tree


RELAYS: ContextVar[Relays | None] = ContextVar('RELAYS', default=None)
"""The relays of the authoring container whose `get_instructions` is running."""


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
    if source.id is not None:
        validate_instruction_id_segment(source.id, kind='Toolset id')
        source_id = ToolsetInstructionSource(source.id)
        if part.id is None or part.id.source != source_id:
            # A part claiming a key that isn't this source's is re-keyed beneath the source that
            # actually contributed it, keeping the name its author declared relative to itself.
            resolved_part = replace(part, id=InstructionId(source_id, name=part.name))
    elif part.id is not None:
        resolved_part = replace(part, id=None)
    return InstructionContribution(source=source, source_id=source_id, parts=(resolved_part,))


def framework_appended_toolset_sources(toolset: AbstractToolset[AgentDepsT]) -> set[ToolsetInstructionSource]:
    """The instruction sources of the toolsets under `toolset` whose blocks append by the framework's choice."""
    sources: set[ToolsetInstructionSource] = set()

    def visit(leaf: AbstractToolset[AgentDepsT]) -> None:
        if leaf._appends_instructions_by_default and leaf.id is not None:  # pyright: ignore[reportPrivateUsage]
            sources.add(ToolsetInstructionSource(leaf.id))

    toolset.apply(visit)
    return sources


async def collect_toolset_instructions(
    toolset: AbstractToolset[AgentDepsT], ctx: RunContext[AgentDepsT]
) -> list[InstructionPart]:
    """Collect instruction parts once at the authoring toolset boundary."""
    contributions = await toolset._collect_instruction_contributions(ctx)  # pyright: ignore[reportPrivateUsage]
    return flatten_instruction_contributions(contributions)


def flatten_instruction_contributions(
    contributions: list[InstructionContribution[AgentDepsT]],
) -> list[InstructionPart]:
    """Flatten collected contributions, rejecting two sources that would claim one key by `id`s they were given.

    A toolset whose `id` is derived, as an `MCPToolset` without one takes its server's hostname,
    doesn't collide: when another source shares its key, its parts go out unkeyed and rewriting, as
    they did before the `id` was derived, with one warning suggesting a distinct `id`.

    Separate from the collection itself so a toolset that has already gathered its children can
    apply the rule without re-entering the walk that produced them.
    """
    explicit_owners: dict[ToolsetInstructionSource, AbstractToolset[AgentDepsT]] = {}
    sources_by_key: dict[ToolsetInstructionSource, set[int]] = {}
    for contribution in contributions:
        if (source_id := contribution.source_id) is None:
            continue
        sources_by_key.setdefault(source_id, set()).add(id(contribution.source))
        if _has_derived_id(contribution.source):
            continue
        if explicit_owners.setdefault(source_id, contribution.source) is not contribution.source:
            raise UserError(
                f'Two toolsets have the same `id` {source_id.id!r} and both contribute '
                f'instructions, so {str(source_id)!r} would address parts from each. '
                'Toolset `id`s must be unique among all toolsets registered with the same agent.'
            )
    parts: list[InstructionPart] = []
    for contribution in contributions:
        source_id = contribution.source_id
        if source_id is None or len(sources_by_key[source_id]) == 1 or not _has_derived_id(contribution.source):
            parts.extend(contribution.parts)
            continue
        _warn_derived_id_collision(source_id)
        # Rewriting, an empty block says nothing, so it's dropped like any other empty rewrite-mode part.
        replacements = [
            replace(part, id=None, on_change='rewrite') for part in contribution.parts if part.content.strip()
        ]
        parts.extend(replacements)
        # Flattened inside an authoring container's `get_instructions`, the replacements are still the child's
        # parts, not text the container wrote, so they stay attributed to the child.
        if (relays := RELAYS.get()) is not None and relays.covers(contribution.source):
            relays.parts.update(
                (id(part), RelayedPart(contribution.source, source_id, part, replace(part))) for part in replacements
            )
    return parts


def _has_derived_id(toolset: AbstractToolset[AgentDepsT]) -> bool:
    """Whether the toolset's `id` is one it derived, such as an `MCPToolset`'s server host, rather than one it was given."""
    return toolset.id is not None and toolset._explicit_id is None  # pyright: ignore[reportPrivateUsage]


_warned_derived_id_collisions: set[str] = set()
"""The keys a derived-`id` collision was already reported for, so it's reported once rather than on every request."""


def _warn_derived_id_collision(source_id: ToolsetInstructionSource) -> None:
    key = str(source_id)
    if key in _warned_derived_id_collisions:
        return
    _warned_derived_id_collisions.add(key)
    warnings.warn(
        f'Toolsets for more than one MCP server derive the `id` {source_id.id!r} from their server URL, so '
        "the instructions of those without an explicit `id` aren't keyed, and a change to them rewrites the "
        'instruction prefix instead of being appended. Pass a distinct `id=` to each `MCPToolset` to append them.',
        UserWarning,
        stacklevel=2,
    )
