"""Choose which Render-supported toolsets register tasks before binding starts."""

from __future__ import annotations

from collections.abc import Awaitable, Sequence
from typing import Any, NoReturn, Protocol, TypeAlias, runtime_checkable

from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset, FunctionToolset
from pydantic_ai_harness.subagents import SubAgents
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits

from ._compat import CapabilityOwnedToolset

__all__ = ('prepare_capability_toolset_ids',)

SupportedLeafToolset: TypeAlias = 'AbstractToolset[Any]'


@runtime_checkable
class _MCPToolsetShape(Protocol):
    """Identify MCP leaves without importing the optional MCP client package."""

    def list_resources(self) -> Awaitable[object]: ...


def prepare_capability_toolset_ids(toolsets: Sequence[AbstractToolset[Any]]) -> frozenset[int]:
    """Return unnamed capability-owned leaves that must stay inline.

    Pydantic AI exposes no public setter for a toolset ID. A capability-built leaf that
    arrives unnamed therefore remains inline rather than being assigned private state.
    User-owned leaves can be named at construction and are rejected before registration
    when they are not.
    """
    nodes = _walk(toolsets)
    leaves = [node for node in nodes if _is_supported_leaf(node)]
    owners = _capability_owners(nodes)
    _reject_duplicate_ids(leaves)

    inline: set[int] = set()
    for toolset in leaves:
        # Delegation and overflow readers use live run state owned by their capability.
        if isinstance(owners.get(id(toolset)), SubAgents | ToolOutputLimits):
            inline.add(id(toolset))
            continue
        if toolset.id is not None:
            continue
        if id(toolset) in owners:
            inline.add(id(toolset))
        else:
            _reject_unnameable(toolset)
    return frozenset(inline)


def _reject_unnameable(toolset: SupportedLeafToolset) -> NoReturn:
    """Refuse an unnamed leaf the caller can name through its public constructor."""
    raise UserError(f'{type(toolset).__name__} needs a unique `id` to register tasks with Render Workflows.')


def _capability_owners(nodes: Sequence[AbstractToolset[Any]]) -> dict[int, AbstractCapability[Any]]:
    owners: dict[int, AbstractCapability[Any]] = {}

    for node in nodes:
        if not isinstance(node, CapabilityOwnedToolset):
            continue

        for leaf in _walk((node.wrapped,)):
            if _is_supported_leaf(leaf):
                # Nested capability wrappers are visited after their parents,
                # so the closest capability becomes the owner.
                owners[id(leaf)] = node.capability

    return owners


def _is_supported_leaf(toolset: AbstractToolset[Any]) -> bool:
    return isinstance(toolset, FunctionToolset | DynamicToolset | _MCPToolsetShape)


def _reject_duplicate_ids(leaves: Sequence[SupportedLeafToolset]) -> None:
    """Refuse an `id` two distinct toolsets already carry, rather than renaming either.

    A derived name never takes an `id` a toolset already holds, so a clash between two
    toolsets that were both named is a genuine collision and stays Pydantic AI's answer,
    only reached before this app registers anything.
    """
    seen: dict[str, SupportedLeafToolset] = {}
    for toolset in leaves:
        toolset_id = toolset.id
        if toolset_id is None:
            continue
        existing = seen.get(toolset_id)
        if existing is not None and existing is not toolset:
            raise UserError(
                f'Two toolsets have the same `id` {toolset_id!r}. Toolset `id`s must be unique among all '
                'toolsets registered with the same agent.'
            )
        seen[toolset_id] = toolset


def _walk(toolsets: Sequence[AbstractToolset[Any]]) -> list[AbstractToolset[Any]]:
    """Return every node visited by Pydantic AI's toolset traversal."""
    nodes: list[AbstractToolset[Any]] = []

    for root in toolsets:
        root.apply(nodes.append)

    return nodes
