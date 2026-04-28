from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic_ai._instructions import AgentInstructions
from pydantic_ai.tools import AgentBuiltinTool, AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, AgentToolset
from pydantic_ai.toolsets._deferred_capability import DeferredCapabilityToolset

from .abstract import AbstractCapability, CapabilityOrdering
from .wrapper import WrapperCapability

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AgentModelSettings


@dataclass
class DeferredCapability(WrapperCapability[AgentDepsT]):
    """A wrapper that suppresses a capability's contributions until explicitly loaded.

    When a capability has ``defer_loading=True``, the agent automatically wraps it
    in a ``DeferredCapability``. Before :meth:`load` is called, all ``get_*`` methods
    return empty values (``None`` / ``[]``), hiding the capability's instructions,
    tools, and settings from the model. The capability's ``id`` and ``description``
    remain visible for catalog rendering so the model can discover and request it
    via ``load_capability(id)``.

    Hooks are always forwarded regardless of loaded state, allowing the wrapped
    capability to observe lifecycle events and decide how to act based on
    whether it has been loaded.
    """

    _loaded: bool = field(default=False, init=False, repr=False)

    def load(self) -> None:
        self._loaded = True

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        if not self._loaded:
            return None
        return self.wrapped.get_instructions()

    def get_model_settings(self) -> AgentModelSettings[AgentDepsT] | None:
        if not self._loaded:
            return None
        return self.wrapped.get_model_settings()

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        if not self._loaded:
            return None
        return self.wrapped.get_toolset()

    def get_builtin_tools(self) -> Sequence[AgentBuiltinTool[AgentDepsT]]:
        if not self._loaded:
            return []
        return self.wrapped.get_builtin_tools()

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        if not self._loaded:
            return None
        return self.wrapped.get_wrapper_toolset(toolset)


@dataclass
class DeferredLoadingCapability(AbstractCapability[AgentDepsT]):
    """Framework capability that provides the ``load_capability`` tool for deferred capabilities.

    Auto-injected by the agent when any capability has ``defer_loading=True``.
    Provides a :class:`~pydantic_ai.toolsets._deferred_capability.DeferredCapabilityToolset`
    that exposes a single ``load_capability(id)`` tool. The tool description includes a
    catalog of all unloaded capabilities so the model knows what's available.
    """

    deferred_capabilities: Sequence[DeferredCapability[AgentDepsT]]

    def get_ordering(self) -> CapabilityOrdering | None:
        # It makes sense for this to be outermost although I am skeptical how this will work with tool search something to understand for me
        return CapabilityOrdering(position='outermost')

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        # This toolset will provide us with the tools to use deferred capabilities
        return DeferredCapabilityToolset(wrapped=toolset, deferred_capabilities=self.deferred_capabilities)
