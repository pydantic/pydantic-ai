"""Internal helpers for collecting capability-contributed toolsets."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets._capability_owned import CapabilityOwnedToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset

from .abstract import AbstractCapability


def get_capability_toolsets(
    capabilities: Sequence[AbstractCapability[AgentDepsT]],
) -> list[CapabilityOwnedToolset[AgentDepsT]]:
    """Collect normalized capability contributions in capability order."""
    return [toolset for capability in capabilities if (toolset := get_capability_toolset(capability)) is not None]


def get_capability_toolset(
    capability: AbstractCapability[AgentDepsT],
) -> CapabilityOwnedToolset[AgentDepsT] | None:
    """Normalize one capability contribution and retain its owning capability."""
    toolset = capability.get_toolset()
    if toolset is None:
        return None
    if not isinstance(toolset, AbstractToolset):
        toolset = DynamicToolset[AgentDepsT](toolset_func=toolset)
    return CapabilityOwnedToolset(
        wrapped=toolset,  # pyright: ignore[reportUnknownArgumentType]
        capability=capability,
    )
