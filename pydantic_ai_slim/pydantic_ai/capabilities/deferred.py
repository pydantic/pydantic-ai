from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_ai._instructions import AgentInstructions
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets._deferred_capability import DeferredCapabilityToolset

from .abstract import AbstractCapability, CapabilityOrdering
from .wrapper import WrapperCapability


@dataclass
class DeferredCapability(WrapperCapability[AgentDepsT]):
    """A wrapper that suppresses a capability's contributions until explicitly loaded.

    When a capability has ``defer_loading=True``, the agent automatically wraps it
    in a ``DeferredCapability``. Until the model calls ``load_capability(id)``, all
    ``get_*`` methods return empty values (``None`` / ``[]``), hiding the capability's
    instructions, tools, and settings from the model. The capability's ``id`` and
    ``description`` remain visible for catalog rendering so the model can discover
    and request it.

    Loaded state is derived from message history (the ``ToolReturn`` recorded by
    ``load_capability``), so the same capability instance can be safely shared
    across agents and runs without state leaking.

    Hooks are always forwarded regardless of loaded state, allowing the wrapped
    capability to observe lifecycle events and decide how to act based on
    whether it has been loaded.
    """

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        return None


@dataclass
class DeferredLoadingCapability(AbstractCapability[AgentDepsT]):
    """Framework capability that provides the ``load_capability`` tool for deferred capabilities.

    Auto-injected by the agent when any capability has ``defer_loading=True``.
    Provides a :class:`~pydantic_ai.toolsets._deferred_capability.DeferredCapabilityToolset`
    that exposes a single ``load_capability(id)`` tool. The tool description includes a
    catalog of all unloaded capabilities so the model knows what's available.
    """

    deferred_capabilities: Sequence[DeferredCapability[AgentDepsT]]
    """The deferred capabilities to expose in the `load_capability` catalog."""

    def get_ordering(self) -> CapabilityOrdering | None:
        return CapabilityOrdering(position='outermost')

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        return DeferredCapabilityToolset(wrapped=toolset, deferred_capabilities=self.deferred_capabilities)
