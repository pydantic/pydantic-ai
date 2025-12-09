from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from .._run_context import AgentDepsT, RunContext
from .abstract import AbstractToolset, ToolsetTool


SEARCH_TOOL_NAME = "search_tool"


@dataclass
class _SearchTool(ToolsetTool[AgentDepsT]):
    """A tool that searches for more relevant tools from a SearchableToolSet"""
    pass


@dataclass
class SearchableToolset(AbstractToolset[AgentDepsT]):
    """A toolset that implements tool search and deferred tool loading."""

    toolset: AbstractToolset[AgentDepsT]
    _active_tool_names: set[str] = {}

    @property
    def id(self) -> str | None:
        return None  # pragma: no cover

    @property
    def label(self) -> str:
        return f'{self.__class__.__name__}({self.toolset.label})'  # pragma: no cover

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:

        all_tools: dict[str, ToolsetTool[AgentDepsT]] = {}
        all_tools[SEARCH_TOOL_NAME] = _SearchTool(
            toolset=self,
            tool_def=None,
            max_retries=1,
            args_validator=None,
        )

        toolset_tools = await self.toolset.get_tools(ctx)
        for tool in toolset_tools:

            # TODO proper error handling
            assert tool.name != SEARCH_TOOL_NAME

            if tool.name in self._active_tool_names:
                all_tools[tool.name] = tool
        return all_tools

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if isinstance(tool, _SearchTool):
            raise Exception("TODO call search tool")
        else:
            return await self.toolset.call_tool(name, tool_args, ctx, tool)

    def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:
        self.toolset.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        return replace(self, toolset=self.toolset.visit_and_replace(visitor))
