from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

from pydantic_ai._utils import install_deprecated_kwarg_alias
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.exceptions import UserError
from pydantic_ai.native_tools import AbstractNativeTool
from pydantic_ai.tools import AgentDepsT, AgentNativeTool, RunContext, Tool, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.toolsets.prepared import PreparedToolset

from .abstract import AbstractCapability


@dataclass(init=False)
class NativeOrLocalTool(AbstractCapability[AgentDepsT]):
    """Capability that pairs a provider-native tool with a local fallback.

    When the model supports the native tool, the local fallback is removed.
    When the model doesn't support the native tool, it is removed and the local tool stays.

    Can be used directly:

    ```python {test="skip" lint="skip"}
    from pydantic_ai.capabilities import NativeOrLocalTool

    cap = NativeOrLocalTool(native=WebSearchTool(), local=my_search_func)
    ```

    Or subclassed to set defaults by overriding `_default_native`, `_default_local`,
    and `_requires_native`.
    The built-in [`WebSearch`][pydantic_ai.capabilities.WebSearch],
    [`WebFetch`][pydantic_ai.capabilities.WebFetch], and
    [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration] capabilities
    are all subclasses.
    """

    native: AgentNativeTool[AgentDepsT] | bool = True
    """Configure the provider-native tool.

    - `True` (default): use the default native tool configuration (subclasses only).
    - `False`: disable the native tool; always use the local tool.
    - An `AbstractNativeTool` instance: use this specific configuration.
    - A callable (`NativeToolFunc`): dynamically create the native tool per-run via `RunContext`.
    """

    local: str | Tool[AgentDepsT] | Callable[..., Any] | AbstractToolset[AgentDepsT] | bool | None = None
    """Configure the local fallback tool.

    - `None` (default): auto-detect a local fallback via `_default_local`.
    - `True`: opt in to the default local fallback (resolved via `_resolve_local_strategy`).
    - `False`: disable the local fallback; only use the native tool.
    - A named strategy (e.g. `'duckduckgo'`): resolved via `_resolve_local_strategy` in subclasses.
    - A `Tool` or `AbstractToolset` instance: use this specific local tool.
    - A bare callable: automatically wrapped in a `Tool`.
    """

    def __init__(
        self,
        *,
        native: AgentNativeTool[AgentDepsT] | bool = True,
        local: str | Tool[AgentDepsT] | Callable[..., Any] | AbstractToolset[AgentDepsT] | bool | None = None,
    ) -> None:
        self.native = native
        self.local = local
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.native is False and self.local is False:
            raise UserError(f'{type(self).__name__}: both `native` and `local` cannot be False')

        # Resolve native=True → default instance (subclass hook)
        if self.native is True:
            default = self._default_native()
            if default is None:
                raise UserError(
                    f'{type(self).__name__}: native=True requires a subclass that overrides '
                    f'`_default_native()`, or pass an `AbstractNativeTool` instance directly'
                )
            self.native = default

        # Resolve local: None → default, True/str → named strategy, callable → Tool
        if self.local is None:
            self.local = self._default_local()
        elif self.local is True or isinstance(self.local, str):
            self.local = self._resolve_local_strategy(self.local)
        elif self.local is False:
            pass
        elif callable(self.local) and not isinstance(self.local, (Tool, AbstractToolset)):
            self.local = Tool(self.local)

        # Catch contradictory config: native disabled but constraint fields require it
        if self.native is False and self._requires_native():
            raise UserError(f'{type(self).__name__}: constraint fields require the native tool, but native=False')

    # --- Subclass hooks (not abstract — direct use is supported) ---

    def _default_native(self) -> AbstractNativeTool | None:
        """Create the default native tool instance.

        Override in subclasses. Returns None by default (direct use requires
        passing an explicit `AbstractNativeTool` instance as `native`).
        """
        return None

    def _native_unique_id(self) -> str:
        """The unique_id used for `unless_native` on local tool definitions.

        By default, derived from the native tool's `unique_id` property.
        Override in subclasses for custom behavior.
        """
        native = self.native
        if isinstance(native, AbstractNativeTool):
            return native.unique_id
        raise UserError(
            f'{type(self).__name__}: cannot derive native unique_id — override `_native_unique_id()` in your subclass'
        )

    def _default_local(self) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT] | None:
        """Auto-detect a local fallback. Override in subclasses that have one."""
        return None

    def _resolve_local_strategy(self, name: str | bool) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT]:
        """Resolve a named local strategy (e.g. `'duckduckgo'`) or `local=True` to a concrete tool.

        Override in subclasses that expose named strategies. The default implementation raises
        `UserError`.
        """
        raise UserError(
            f'{type(self).__name__}: `local={name!r}` is not supported. '
            'Pass a `Tool`, `AbstractToolset`, or callable directly.'
        )

    def _requires_native(self) -> bool:
        """Return True if capability-level constraint fields require the native tool.

        When True, the local fallback is suppressed. If the model doesn't support
        the native tool, `UserError` is raised — preventing silent constraint violation.

        Override in subclasses that expose native-only constraint fields
        (e.g. `allowed_domains`, `blocked_domains`).
        """
        return False

    # --- Shared logic ---

    def get_native_tools(self) -> Sequence[AgentNativeTool[AgentDepsT]]:
        if self.native is False:
            return []
        # After __post_init__, native=True is resolved to an AbstractNativeTool instance
        assert not isinstance(self.native, bool)
        return [self.native]

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
        local = self.local
        if local is None or local is False or self._requires_native():
            return None

        # local is Tool | AbstractToolset after __post_init__ resolution
        toolset: AbstractToolset[AgentDepsT] = (
            cast(AbstractToolset[AgentDepsT], local)
            if isinstance(local, AbstractToolset)
            else FunctionToolset([cast(Tool[AgentDepsT], local)])
        )

        if self.native is not False:
            uid = self._native_unique_id()

            async def _add_unless_native(
                ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                return [replace(d, unless_native=uid) for d in tool_defs]

            return PreparedToolset(wrapped=toolset, prepare_func=_add_unless_native)
        return toolset

    def __getattr__(self, name: str) -> Any:
        # Deprecated alias for read access to the renamed `builtin=` field.
        if name == 'builtin':
            warnings.warn(
                f'`{type(self).__name__}.builtin` is deprecated, use `.native` instead.',
                PydanticAIDeprecationWarning,
                stacklevel=2,
            )
            return self.native
        raise AttributeError(name)


install_deprecated_kwarg_alias(NativeOrLocalTool, old='builtin', new='native')
