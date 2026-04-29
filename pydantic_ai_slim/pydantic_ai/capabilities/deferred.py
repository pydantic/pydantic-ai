from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from pydantic_ai._instructions import AgentInstructions
from pydantic_ai._run_context import RunContext
from pydantic_ai.tools import AgentBuiltinTool, AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, AgentToolset
from pydantic_ai.toolsets._deferred_capability import DeferredCapabilityToolset, parse_loaded_capabilities
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset

from .abstract import AbstractCapability, CapabilityOrdering
from .wrapper import WrapperCapability

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AgentModelSettings


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

    _loaded: bool = field(default=False, init=False, repr=False)

    @property
    def loaded(self) -> bool:
        return self._loaded

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        result = await super().for_run(ctx)
        # Each sibling DeferredCapability re-scans the same messages here; deduping would
        # require threading a shared cache through for_run/RunContext, which isn't worth
        # the API surface for a run-start cost on a small list.
        loaded_capabilities = parse_loaded_capabilities(ctx.messages)
        if isinstance(result, DeferredCapability):
            result._loaded = result.wrapped.id in loaded_capabilities
        return result

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        if not self._loaded:
            return None
        return self.wrapped.get_instructions()

    def get_model_settings(self) -> AgentModelSettings[AgentDepsT] | None:
        if not self._loaded:
            return None
        return self.wrapped.get_model_settings()

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        toolset = self.wrapped.get_toolset()
        if toolset is None:
            return None
        if isinstance(toolset, AbstractToolset):
            wrapped = cast(AbstractToolset[AgentDepsT], toolset)
        else:
            wrapped = DynamicToolset[AgentDepsT](toolset_func=toolset)
        return _GatedToolset(wrapped=wrapped, deferred=self)

    def get_builtin_tools(self) -> Sequence[AgentBuiltinTool[AgentDepsT]]:
        if not self._loaded:
            return []
        return self.wrapped.get_builtin_tools()

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        if not self._loaded:
            return None
        return self.wrapped.get_wrapper_toolset(toolset)


@dataclass
class _GatedToolset(WrapperToolset[AgentDepsT]):
    """Toolset that lives in the chain from setup but returns empty tools until loaded."""

    deferred: DeferredCapability[AgentDepsT]

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        if self.deferred.wrapped.id not in parse_loaded_capabilities(ctx.messages):
            return {}
        return await self.wrapped.get_tools(ctx)

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)


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
        return CapabilityOrdering(position='outermost')

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        return DeferredCapabilityToolset(wrapped=toolset, deferred_capabilities=self.deferred_capabilities)
