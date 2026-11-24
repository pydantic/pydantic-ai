"""Private helper for finding a capability's per-run copy from the toolset it shares."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic_ai.capabilities import AbstractCapability, WrapperCapability
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AbstractToolset

CapabilityT = TypeVar('CapabilityT', bound=AbstractCapability[Any])


def find_run_capability(
    ctx: RunContext[Any], capability: CapabilityT, toolset: AbstractToolset[Any]
) -> CapabilityT | None:
    """The capability of `capability`'s type in `ctx.root_capability` whose toolset is `toolset`.

    A capability whose `for_run` copy shares the toolset it was constructed with uses this to reach
    the copy, and the state `for_run` resolved for the run. A durable worker's tree holds the
    construction-time capability instead. `None` when the context has no capability tree or the tree
    doesn't provide `toolset`.
    """
    found: CapabilityT | None = None

    def select(candidate: AbstractCapability[Any]) -> None:
        nonlocal found
        # A wrapper such as `prefix_tools()` may be visited in place of the capability it wraps.
        while isinstance(candidate, WrapperCapability):
            candidate = candidate.wrapped
        if isinstance(candidate, type(capability)) and candidate.get_toolset() is toolset:
            found = candidate

    if ctx.root_capability is not None:
        ctx.root_capability.apply(select)
    return found
