"""Keep a turn going when an optional MCP server a plugin contributes cannot connect or list its tools.

CLAI's own `mcp` plugin already marks a server that fails to connect as `error`. MCP toolsets that other plugins
contribute (Logfire MCP, Slack, GitHub, ...) would otherwise fail the whole prompt; here they are skipped for
that run, with one warning per server per session.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Self

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool, WrapperToolset


@dataclass
class _SkipOnFailure(WrapperToolset[Any]):
    warn: Callable[[str, BaseException], None] = field(default=lambda label, error: None)
    _failed: bool = field(default=False, init=False)

    async def __aenter__(self) -> Self:
        try:
            await self.wrapped.__aenter__()
        except Exception as error:  # noqa: BLE001 -- an optional server must not fail the turn
            self._failed = True
            self.warn(self.wrapped.label, error)
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        if self._failed:
            self._failed = False
            return None
        return await self.wrapped.__aexit__(*args)

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        if self._failed:
            return {}
        try:
            return await self.wrapped.get_tools(ctx)
        except Exception as error:  # noqa: BLE001 -- an optional server must not fail the turn
            self.warn(self.wrapped.label, error)
            return {}


@dataclass(kw_only=True)
class ResilientMCP(AbstractCapability[Any]):
    """Wrap every plugin-contributed MCP toolset so a failure skips it instead of failing the turn."""

    warn: Callable[[str], None]
    id: str | None = 'clai2_resilient_mcp'
    _warned: set[str] = field(default_factory=set[str], init=False)

    def get_wrapper_toolset(self, toolset: AbstractToolset[Any]) -> AbstractToolset[Any]:
        return toolset.visit_and_replace(self._wrap)

    def _wrap(self, toolset: AbstractToolset[Any]) -> AbstractToolset[Any]:
        if isinstance(toolset, MCPToolset):
            return _SkipOnFailure(wrapped=toolset, warn=self._warn)
        return toolset

    def _warn(self, label: str, error: BaseException) -> None:
        if label in self._warned:
            return
        self._warned.add(label)
        self.warn(f'Skipped MCP server {label} for now: {type(error).__name__}: {error}'[:300])
