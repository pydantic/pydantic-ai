import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, TypedDict, cast

from pydantic import TypeAdapter
from typing_extensions import Self

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from .abstract import AbstractToolset, SchemaValidatorProt, ToolsetTool

_SEARCH_TOOL_NAME = 'load_tools'


class _SearchToolArgs(TypedDict):
    regex: str


## TODO Check out Tool.from_schema and the Tool constructor that takes a function (as used by FunctionToolset) for easier
## ways to construct a single tool. The function approach is the easiest by far
def _search_tool_def() -> ToolDefinition:
    return ToolDefinition(
        name=_SEARCH_TOOL_NAME,

## TODO Simplify the prompt.
        description="""Search and load additional tools to make them available to the agent.

DO call this to find and load more tools needed for a task.
NEVER ask the user if you should try loading tools, just try.
""",
        parameters_json_schema={
            'type': 'object',
            'properties': {
## TODO Check if pattern is a better name than regex.
                'regex': {
                    'type': 'string',
                    'description': 'Regex pattern to search for relevant tools',
                }
            },
            'required': ['regex'],
        },
    )


## TODO Why is this a function? TypeAdapter is expensive to create.
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

    # TODO Have a look at Wrapper Toolset.
    toolset: AbstractToolset[AgentDepsT]

    _active_tool_names: dict[str, set[str]] = field(default_factory=dict)
    """Tracks activate tool name sets indexed by RunContext.run_id"""

    @property
    def id(self) -> str | None:
        return None  # pragma: no cover

    @property
    def label(self) -> str:
        return f'{self.__class__.__name__}({self.toolset.label})'  # pragma: no cover

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        toolset_tools = await self.toolset.get_tools(ctx)

        # TODO Should not expose search tool if there are no defer loading tools.
        all_tools: dict[str, ToolsetTool[AgentDepsT]] = {}

        all_tools[_SEARCH_TOOL_NAME] = _SearchTool(
            toolset=self,
            max_retries=3,
        )

        for tool_name, tool in toolset_tools.items():
            # TODO proper error handling, checkout ModelRetry exception?
            assert tool_name != _SEARCH_TOOL_NAME
            defer = tool.tool_def.defer_loading
            all_tools[tool_name] = _SearchToolsetToolWrapper(tool, self) if defer else tool

        return all_tools

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if isinstance(tool, _SearchTool):
            adapter = TypeAdapter(_SearchToolArgs)
            # TODO already validated, can cast.
            typed_args = adapter.validate_python(tool_args)
            result = await self.call_search_tool(typed_args, ctx)
            logging.debug(f"SearchableToolset.call_tool({name}, {tool_args}) ==> {result}")
            return result
        else:
            assert isinstance(tool, _SearchToolsetToolWrapper)
            result = await self.toolset.call_tool(name, tool_args, ctx, tool.wrapped)
            logging.debug(f"SearchableToolset.call_tool({name}, {tool_args}) ==> {result}")
            return result

    async def call_search_tool(self, args: _SearchToolArgs, ctx: RunContext[AgentDepsT]) -> list[str]:
        """Searches for tools matching the query, activates them and returns their names."""
        toolset_tools = await self.toolset.get_tools(ctx)
        matching_tool_names: list[str] = []

        rx = re.compile(args['regex'])

        for _, tool in toolset_tools.items():
            if rx.search(tool.tool_def.name) or rx.search(tool.tool_def.description):
                matching_tool_names.append(tool.tool_def.name)

        self._active_tool_names.setdefault(ctx.run_id, set()).update(matching_tool_names)
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

    def is_active(self, tool_def: ToolDefinition, run_id: str) -> bool:
        return tool_def.name in self._active_tool_names.get(run_id, set())


@dataclass(kw_only=True)
class _SearchToolsetToolWrapper(ToolsetTool[AgentDepsT]):
    """A ToolsetTool that tags its ToolDefinition to enable tool_is_active query."""

    wrapped: ToolsetTool[AgentDepsT]
    _tool_def: ToolDefinition
    _searchable_toolset: SearchableToolset

    def __init__(
        self,
        tool: ToolsetTool[AgentDepsT],
        searchable_toolset: SearchableToolset,
    ) -> None:
        self.wrapped = tool
        self._searchable_toolset = searchable_toolset
        self.toolset = tool.toolset
        self._tool_def = tool.tool_def
        self.max_retries = tool.max_retries
        self.args_validator = tool.args_validator

    @property
    def tool_def(self) -> ToolDefinition:
        metadata = (self._tool_def.metadata or {}).copy()
        toolset = self._searchable_toolset
        tool = self._tool_def

        # ModelRequestParameters and its ToolDefinitions need to be serializable to work with Temporal durable
        # execution, so storing a callable is not going to work I'm afraid.

        # Also overriding metadata is not going to work if something underneath is already having it.
        metadata["active"] = lambda run_id: toolset.is_active(tool_def=tool, run_id=run_id)
        return replace(self._tool_def, metadata=metadata)

    @tool_def.setter
    def tool_def(self, value: ToolDefinition) -> None:
        self._tool_def = value


def is_active(tool_def: ToolDefinition, run_id: str) -> bool:
    """Filters out not-yet-active defer_loading tools."""
    if not tool_def.defer_loading:
        return True
    metadata = (tool_def.metadata or {}).copy()
    predicate = metadata.get("active")
    return predicate and predicate(run_id)


def is_search_tool(tool_def: ToolDefinition) -> bool:
    """Check if this tool is a tool implementing search and loading."""
    return tool_def.name == _SEARCH_TOOL_NAME
