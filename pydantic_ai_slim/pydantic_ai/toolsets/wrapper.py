from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from typing_extensions import Self

from .._run_context import AgentDepsT, RunContext
from .abstract import AbstractToolset, ToolsetTool


@dataclass
class WrapperToolset(AbstractToolset[AgentDepsT]):
    """A toolset that wraps another toolset and delegates to it.

    See [toolset docs](../toolsets.md#wrapping-a-toolset) for more information.
    """

    wrapped: AbstractToolset[AgentDepsT]

    @property
    def id(self) -> str | None:
        return None  # pragma: no cover

    @property
    def label(self) -> str:
        return f'{self.__class__.__name__}({self.wrapped.label})'

    async def __aenter__(self) -> Self:
        await self.wrapped.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return await self.wrapped.__aexit__(*args)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        return await self.wrapped.get_tools(ctx)

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)

    async def get_instructions(self, ctx: RunContext[AgentDepsT]) -> str | None:
        """Delegate instructions to the wrapped toolset.

        This explicit delegation ensures type safety and proper propagation of custom
        instructions from wrapped toolsets to the agent's system prompt.
        """
        return await self.wrapped.get_instructions(ctx)

    def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:
        self.wrapped.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        return replace(self, wrapped=self.wrapped.visit_and_replace(visitor))

    def __getattr__(self, name: str) -> Any:
        """Fallback delegation for custom attributes and methods from the wrapped toolset.

        This enables runtime access to custom methods and attributes that may be defined
        on the wrapped toolset but not on this wrapper. Note that static type checkers
        may not recognize these delegated attributes.
        """
        return getattr(self.wrapped, name)
