from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from typing_extensions import Self

from .._instructions import normalize_toolset_instructions
from .._run_context import AgentDepsT, RunContext
from ..messages import InstructionPart
from ._instruction_collection import collect_toolset_instructions
from .abstract import AbstractToolset, ToolsetTool


@dataclass
class WrapperToolset(AbstractToolset[AgentDepsT]):
    """A toolset that wraps another toolset and delegates to it.

    See [toolset docs](../toolsets.md#changing-tool-execution) for more information.
    """

    wrapped: AbstractToolset[AgentDepsT]

    @property
    def id(self) -> str | None:
        return None

    @property
    def label(self) -> str:
        return f'{self.__class__.__name__}({self.wrapped.label})'

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        new_wrapped = await self.wrapped.for_run(ctx)
        if new_wrapped is self.wrapped:
            return self
        return replace(self, wrapped=new_wrapped)

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        new_wrapped = await self.wrapped.for_run_step(ctx)
        if new_wrapped is self.wrapped:
            return self
        return replace(self, wrapped=new_wrapped)

    async def __aenter__(self) -> Self:
        await self.wrapped.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return await self.wrapped.__aexit__(*args)

    async def get_instructions(
        self, ctx: RunContext[AgentDepsT]
    ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        """Collect instructions from the wrapped authoring toolset."""
        return await collect_toolset_instructions(self.wrapped, ctx) or None

    async def _collect_instruction_contributions(
        self, ctx: RunContext[AgentDepsT]
    ) -> list[tuple[AbstractToolset[AgentDepsT], list[InstructionPart]]]:
        if type(self).get_instructions is not WrapperToolset.get_instructions:
            result = await self.get_instructions(ctx)
            source = self.wrapped
            while isinstance(source, WrapperToolset):
                source = source.wrapped
            return [(source, normalize_toolset_instructions(result, self.id, already_keyed=True))]
        return await self.wrapped._collect_instruction_contributions(ctx)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        return await self.wrapped.get_tools(ctx)

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)

    def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:
        self.wrapped.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        return replace(self, wrapped=self.wrapped.visit_and_replace(visitor))
