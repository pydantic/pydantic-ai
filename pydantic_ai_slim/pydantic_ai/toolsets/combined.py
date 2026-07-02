from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from typing import Any

from typing_extensions import Self

from .._run_context import AgentDepsT, RunContext
from .._utils import gather
from ..exceptions import UserError
from ..messages import InstructionPart
from .abstract import AbstractToolset, ToolsetTool


@dataclass(kw_only=True)
class _CombinedToolsetTool(ToolsetTool[AgentDepsT]):
    """A tool definition for a combined toolset tools that keeps track of the source toolset and tool."""

    source_toolset: AbstractToolset[AgentDepsT]
    source_tool: ToolsetTool[AgentDepsT]


@dataclass
class CombinedToolset(AbstractToolset[AgentDepsT]):
    """A toolset that combines multiple toolsets.

    See [toolset docs](../toolsets.md#combining-toolsets) for more information.
    """

    toolsets: Sequence[AbstractToolset[AgentDepsT]]

    _exit_stack: AsyncExitStack | None = field(init=False, default=None)

    @property
    def id(self) -> str | None:
        return None  # pragma: no cover

    @property
    def label(self) -> str:
        return f'{self.__class__.__name__}({", ".join(toolset.label for toolset in self.toolsets)})'  # pragma: no cover

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        new_toolsets = await gather(*(t.for_run(ctx) for t in self.toolsets))
        return replace(self, toolsets=new_toolsets)

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        new_toolsets = await gather(*(t.for_run_step(ctx) for t in self.toolsets))
        if all(new is old for new, old in zip(new_toolsets, self.toolsets)):
            return self
        return replace(self, toolsets=new_toolsets)

    async def __aenter__(self) -> Self:
        async with AsyncExitStack() as exit_stack:
            for toolset in self.toolsets:
                await exit_stack.enter_async_context(toolset)
            self._exit_stack = exit_stack.pop_all()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        toolsets_tools = await gather(*(toolset.get_tools(ctx) for toolset in self.toolsets))
        all_tools: dict[str, ToolsetTool[AgentDepsT]] = {}

        for toolset, tools in zip(self.toolsets, toolsets_tools):
            for name, tool in tools.items():
                tool_toolset = tool.toolset
                if existing_tool := all_tools.get(name):
                    capitalized_toolset_label = tool_toolset.label[0].upper() + tool_toolset.label[1:]
                    raise UserError(
                        f'{capitalized_toolset_label} defines a tool whose name conflicts with existing tool from {existing_tool.toolset.label}: {name!r}. {toolset.tool_name_conflict_hint}'
                    )

                all_tools[name] = _CombinedToolsetTool(
                    toolset=tool_toolset,
                    tool_def=tool.tool_def,
                    max_retries=tool.max_retries,
                    args_validator=tool.args_validator,
                    args_validator_func=tool.args_validator_func,
                    source_toolset=toolset,
                    source_tool=tool,
                )
        return all_tools

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        assert isinstance(tool, _CombinedToolsetTool)
        return await tool.source_toolset.call_tool(name, tool_args, ctx, tool.source_tool)

    def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:
        for toolset in self.toolsets:
            toolset.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        return replace(self, toolsets=[toolset.visit_and_replace(visitor) for toolset in self.toolsets])

    async def get_instructions(self, ctx: RunContext[AgentDepsT]) -> list[str | InstructionPart] | None:
        results = await gather(*(ts.get_instructions(ctx) for ts in self.toolsets))
        parts: list[str | InstructionPart] = []
        for r in results:
            if r is not None:
                if isinstance(r, (str, InstructionPart)):
                    parts.append(r)
                else:
                    parts.extend(r)
        return parts or None
