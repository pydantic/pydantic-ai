from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, BuiltinToolFunc, RunContext, Tool, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset

from .abstract import AbstractCapability


@dataclass
class BuiltinToolCapability(AbstractCapability[AgentDepsT], ABC):
    """Base class for capabilities that pair a provider builtin tool with a local fallback.

    When the model supports the builtin natively, the local fallback is removed.
    When the model doesn't support the builtin, it is removed and the local tool stays.

    Subclasses must override :meth:`_default_builtin` and :meth:`_builtin_unique_id`.
    They may also override :meth:`_default_local` and :meth:`_requires_builtin`.
    """

    builtin: AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT] | bool = True
    """Configure the provider builtin tool.

    - ``True`` (default): use the default builtin tool configuration.
    - ``False``: disable the builtin; always use the local tool.
    - An ``AbstractBuiltinTool`` instance: use this specific configuration.
    - A callable (``BuiltinToolFunc``): dynamically create the builtin per-run.
    """

    local: Tool[Any] | Callable[..., Any] | AbstractToolset[Any] | Literal[False] | None = None
    """Configure the local fallback tool.

    - ``None`` (default): auto-detect a local fallback via :meth:`_default_local`.
    - ``False``: disable the local fallback; only use the builtin.
    - A ``Tool`` or ``AbstractToolset`` instance: use this specific local tool.
    - A bare callable: automatically wrapped in a ``Tool``.
    """

    def __post_init__(self) -> None:
        if self.builtin is False and self.local is False:
            raise UserError(f'{type(self).__name__}: both builtin and local cannot be False')

        # Resolve builtin=True → default instance
        if self.builtin is True:
            self.builtin = self._default_builtin()

        # Resolve local: None → default, callable → Tool
        if self.local is None:
            self.local = self._default_local()
        elif callable(self.local) and not isinstance(self.local, (Tool, AbstractToolset)):
            self.local = Tool(self.local)

        # Catch contradictory config: builtin disabled but constraint fields require it
        if self.builtin is False and self._requires_builtin():
            raise UserError(f'{type(self).__name__}: constraint fields require the builtin tool, but builtin=False')

    # --- Subclass hooks ---

    @abstractmethod
    def _default_builtin(self) -> AbstractBuiltinTool:
        """Create the default builtin tool instance."""
        ...

    @abstractmethod
    def _builtin_unique_id(self) -> str:
        """The unique_id used for ``prefer_builtin`` on local tool definitions."""
        ...

    def _default_local(self) -> Tool[Any] | AbstractToolset[Any] | None:
        """Auto-detect a local fallback. Override in subclasses that have one."""
        return None

    def _requires_builtin(self) -> bool:
        """Return True if capability-level constraint fields require the builtin.

        When True, the local fallback is suppressed. If the model doesn't support
        the builtin, ``UserError`` is raised — preventing silent constraint violation.

        Override in subclasses that expose builtin-only constraint fields
        (e.g. ``allowed_domains``, ``blocked_domains``).
        """
        return False

    # --- Shared logic ---

    def get_builtin_tools(self) -> Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]]:
        if self.builtin is False:
            return []
        # After __post_init__, builtin is AbstractBuiltinTool | BuiltinToolFunc (not bool True)
        return [self.builtin]  # type: ignore[list-item]

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
        local = self.local
        if local is None or local is False or self._requires_builtin():
            return None

        from pydantic_ai.toolsets.function import FunctionToolset
        from pydantic_ai.toolsets.prepared import PreparedToolset

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
