from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentBuiltinTool, AgentDepsT, RunContext, Tool, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.toolsets.prepared import PreparedToolset

from .abstract import AbstractCapability


@dataclass
class BuiltinOrLocalTool(AbstractCapability[AgentDepsT]):
    """Capability that pairs a provider builtin tool with a local fallback.

    When the model supports the builtin natively, the local fallback is removed.
    When the model doesn't support the builtin, it is removed and the local tool stays.

    Can be used directly:

    ```python {test="skip" lint="skip"}
    from pydantic_ai.capabilities import BuiltinOrLocalTool

    cap = BuiltinOrLocalTool(builtin=WebSearchTool(), local=my_search_func)
    ```

    Or subclassed to set defaults by overriding `_default_builtin`, `_default_local`,
    and `_requires_builtin`.
    The built-in [`WebSearch`][pydantic_ai.capabilities.WebSearch],
    [`WebFetch`][pydantic_ai.capabilities.WebFetch], and
    [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration] capabilities
    are all subclasses.
    """

    builtin: AgentBuiltinTool[AgentDepsT] | bool = True
    """Configure the provider builtin tool.

    - `True` (default): use the default builtin tool configuration (subclasses only).
    - `False`: disable the builtin; always use the local tool.
    - An `AbstractBuiltinTool` instance: use this specific configuration.
    - A callable (`BuiltinToolFunc`): dynamically create the builtin per-run via `RunContext`.
    """

    local: Tool[AgentDepsT] | Callable[..., Any] | AbstractToolset[AgentDepsT] | Literal[False] | None = None
    """Configure the local fallback tool.

    - `None` (default): auto-detect a local fallback via `_default_local`.
    - `False`: disable the local fallback; only use the builtin.
    - A `Tool` or `AbstractToolset` instance: use this specific local tool.
    - A bare callable: automatically wrapped in a `Tool`.
    """

    def __post_init__(self) -> None:
        if self.builtin is False and self.local is False:
            raise UserError(f'{type(self).__name__}: both builtin and local cannot be False')

        # Resolve builtin=True → default instance (subclass hook)
        if self.builtin is True:
            default = self._default_builtin()
            if default is None:
                raise UserError(
                    f'{type(self).__name__}: builtin=True requires a subclass that overrides '
                    f'_default_builtin(), or pass an AbstractBuiltinTool instance directly'
                )
            self.builtin = default

        # Resolve local: None → default, callable → Tool
        if self.local is None:
            self.local = self._default_local()
        elif callable(self.local) and not isinstance(self.local, (Tool, AbstractToolset)):
            self.local = Tool(self.local)

        # Catch contradictory config: builtin disabled but constraint fields require it
        if self.builtin is False and self._requires_builtin():
            raise UserError(f'{type(self).__name__}: constraint fields require the builtin tool, but builtin=False')

    # --- Subclass hooks (not abstract — direct use is supported) ---

    def _default_builtin(self) -> AbstractBuiltinTool | None:
        """Create the default builtin tool instance.

        Override in subclasses. Returns None by default (direct use requires
        passing an explicit `AbstractBuiltinTool` instance as `builtin`).
        """
        return None

    def _builtin_unique_id(self) -> str:
        """The unique_id used for `prefer_builtin` on local tool definitions.

        By default, derived from the builtin tool's `unique_id` property.
        Override in subclasses for custom behavior.
        """
        builtin = self.builtin
        if isinstance(builtin, AbstractBuiltinTool):
            return builtin.unique_id
        raise UserError(
            f'{type(self).__name__}: cannot derive builtin_unique_id — override _builtin_unique_id() in your subclass'
        )

    def _default_local(self) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT] | None:
        """Auto-detect a local fallback. Override in subclasses that have one."""
        return None

    def _requires_builtin(self) -> bool:
        """Return True if capability-level constraint fields require the builtin.

        When True, the local fallback is suppressed. If the model doesn't support
        the builtin, `UserError` is raised — preventing silent constraint violation.

        Override in subclasses that expose builtin-only constraint fields
        (e.g. `allowed_domains`, `blocked_domains`).
        """
        return False

    # --- Shared logic ---

    def get_builtin_tools(self) -> Sequence[AgentBuiltinTool[AgentDepsT]]:
        if self.builtin is False:
            return []
        # After __post_init__, builtin=True is resolved to an AbstractBuiltinTool instance
        assert not isinstance(self.builtin, bool)
        return [self.builtin]

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
        local = self.local
        if local is None or local is False or self._requires_builtin():
            return None

        # local is Tool | AbstractToolset after __post_init__ resolution
        toolset: AbstractToolset[AgentDepsT] = local if isinstance(local, AbstractToolset) else FunctionToolset([local])  # pyright: ignore[reportUnknownVariableType]

        if self.builtin is not False:
            uid = self._builtin_unique_id()

            async def _add_prefer_builtin(
                ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                return [replace(d, prefer_builtin=uid) for d in tool_defs]

            return PreparedToolset(wrapped=toolset, prepare_func=_add_prefer_builtin)
        return toolset
