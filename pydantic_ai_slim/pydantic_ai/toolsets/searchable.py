import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, TypedDict

from pydantic import TypeAdapter
from typing_extensions import Self

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from .abstract import AbstractToolset, SchemaValidatorProt, ToolsetTool

_SEARCH_TOOL_NAME = 'load_tools'


class _SearchToolArgs(TypedDict):
    regex: str


def _search_tool_def() -> ToolDefinition:
    return ToolDefinition(
        name=_SEARCH_TOOL_NAME,
        description="""Search and load additional tools to make them available to the agent.

DO call this to find and load more tools needed for a task.
NEVER ask the user if you should try loading tools, just try.
""",
        parameters_json_schema={
            'type': 'object',
            'properties': {
                'regex': {
                    'type': 'string',
                    'description': 'Regex pattern to search for relevant tools',
                }
            },
            'required': ['regex'],
        },
    )


def _search_tool_validator() -> SchemaValidatorProt:
    return TypeAdapter(_SearchToolArgs).validator


@dataclass
class _SearchTool(ToolsetTool[AgentDepsT]):
    """A tool that searches for more relevant tools from a SearchableToolSet."""

    tool_def: ToolDefinition = field(default_factory=_search_tool_def)
    args_validator: SchemaValidatorProt = field(default_factory=_search_tool_validator)


@dataclass
class SearchableToolset(AbstractToolset[AgentDepsT]):
    """A toolset that implements tool search and deferred tool loading."""

    toolset: AbstractToolset[AgentDepsT]
    _active_tool_names: set[str] = field(default_factory=set)

    @property
    def id(self) -> str | None:
        return None  # pragma: no cover

    @property
    def label(self) -> str:
        return f'{self.__class__.__name__}({self.toolset.label})'  # pragma: no cover

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        logging.debug("SearchableToolset.get_tools")
        all_tools: dict[str, ToolsetTool[AgentDepsT]] = {}
        all_tools[_SEARCH_TOOL_NAME] = _SearchTool(
            toolset=self,
            max_retries=1,
        )

        toolset_tools = await self.toolset.get_tools(ctx)
        for tool_name, tool in toolset_tools.items():
            # TODO proper error handling
            assert tool_name != _SEARCH_TOOL_NAME

            if tool_name in self._active_tool_names:
                all_tools[tool_name] = tool

        logging.debug(f"SearchableToolset.get_tools ==> {[t for t in all_tools]}")
        return all_tools

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if isinstance(tool, _SearchTool):
            adapter = TypeAdapter(_SearchToolArgs)
            typed_args = adapter.validate_python(tool_args)
            result = await self.call_search_tool(typed_args, ctx)
            logging.debug(f"SearchableToolset.call_tool({name}, {tool_args}) ==> {result}")
            return result
        else:
            result = await self.toolset.call_tool(name, tool_args, ctx, tool)
            logging.debug(f"SearchableToolset.call_tool({name}, {tool_args}) ==> {result}")
            return result

    async def call_search_tool(self, args: _SearchToolArgs, ctx: RunContext[AgentDepsT]) -> list[str]:
        """Searches for tools matching the query, activates them and returns their names."""
        toolset_tools = await self.toolset.get_tools(ctx)
        matching_tool_names: list[str] = []

        for tool_name, tool in toolset_tools.items():
            rx = re.compile(args['regex'])
            if rx.search(tool.tool_def.name) or rx.search(tool.tool_def.description):
                matching_tool_names.append(tool.tool_def.name)

        self._active_tool_names.update(matching_tool_names)
        return matching_tool_names

    def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:
        self.toolset.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        return replace(self, toolset=self.toolset.visit_and_replace(visitor))

    async def __aenter__(self) -> Self:
        await self.toolset.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return await self.toolset.__aexit__(*args)
