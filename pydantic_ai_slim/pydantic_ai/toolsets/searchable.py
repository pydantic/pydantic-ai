import re
from dataclasses import dataclass, field, replace
from typing import Any

from .._run_context import AgentDepsT, RunContext
from ..tools import Tool, ToolDefinition
from .abstract import ToolsetTool
from .wrapper import WrapperToolset

_TOOL_SEARCH_METADATA_KEY = '__tool_search__'
"""ToolDefinition.metadata key to store runtime metadata."""


@dataclass
class SearchableToolset(WrapperToolset[AgentDepsT]):
    """A toolset that implements tool search and deferred tool loading."""

    _active_tool_names: set[str] = field(default_factory=set)
    """Tracks activated tool names."""

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        tools = await self.wrapped.get_tools(ctx)

        # No need to add a search tool if there are no defer_loading tools.
        if not any(t.tool_def.defer_loading for t in tools.values()):
            return tools

        # If any nested SearchableToolset instances have added their own search tools, drop those so that only one
        # search tool is exposed to the model.
        tools = {name: tool for name, tool in tools.items() if not is_search_tool(tool.tool_def)}

        all_tools: dict[str, ToolsetTool[AgentDepsT]] = {}

        for tool_name, tool in tools.items():
            defer = tool.tool_def.defer_loading
            all_tools[tool_name] = _SearchToolsetToolWrapper[AgentDepsT](tool, self) if defer else tool

        search_tool, search_toolset_tool = _search_tool(toolset=self)

        # TODO how to handle this error, or should we automatically disambiguate this name?
        assert search_tool.name not in all_tools
        all_tools[search_tool.name] = search_toolset_tool

        return all_tools

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if is_search_tool(tool.tool_def):
            return await self.search_tools(ctx, tool_args['regex'])
        elif isinstance(tool, _SearchToolsetToolWrapper):
            return await self.wrapped.call_tool(name, tool_args, ctx, tool.wrapped)
        else:
            return await self.wrapped.call_tool(name, tool_args, ctx, tool)

    async def search_tools(self, ctx: RunContext[AgentDepsT], regex: str) -> list[str]:
        """Searches for tools matching the query, activates them and returns their names."""
        toolset_tools = await self.wrapped.get_tools(ctx)
        matching_tool_names: list[str] = []

        rx = re.compile(regex)

        for _, tool in toolset_tools.items():
            if rx.search(tool.tool_def.name) or rx.search(tool.tool_def.description):
                matching_tool_names.append(tool.tool_def.name)

        self._active_tool_names.update(matching_tool_names)
        return matching_tool_names

    def is_active(self, tool_def: ToolDefinition) -> bool:
        return tool_def.name in self._active_tool_names


def _search_tool(toolset: SearchableToolset) -> tuple[Tool, ToolsetTool[AgentDepsT]]:
    async def search(ctx: RunContext[AgentDepsT], regex: str) -> list[str]:
        return await toolset.search_tools(ctx, regex)

    # TODO Check if pattern is a better name than regex.
    # TODO Are examples allowed to be defined somewhere to expose to the model?
    schema = {
        'type': 'object',
        'properties': {
            'regex': {
                'type': 'string',
                'description': 'Regex pattern to search for relevant tools',
            }
        },
        'required': ['regex'],
    }

    desc = 'Search and load additional tools that defer tool loading'
    tool = Tool.from_schema(search, name='load_tools', description=desc, json_schema=schema, takes_ctx=True)

    metadata = _update_metadata(tool.metadata, is_search_tool=True)
    tool = replace(tool, metadata=metadata)

    return tool, ToolsetTool[AgentDepsT](
        toolset=toolset,
        tool_def=tool.tool_def,
        max_retries=tool.max_retries or 3,
        args_validator=tool.function_schema.validator,
    )


@dataclass(kw_only=True)
class _SearchToolsetToolWrapper(ToolsetTool[AgentDepsT]):
    """A ToolsetTool that tags its ToolDefinition to enable is_active query."""

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

    # TODO Mypy: Dataclass attribute may only be overridden by another attribute [misc]
    @property
    def tool_def(self) -> ToolDefinition:
        is_active = self._searchable_toolset.is_active(tool_def=self._tool_def)
        return _update_tool_def_metadata(self._tool_def, is_active=is_active)

    @tool_def.setter
    def tool_def(self, value: ToolDefinition) -> None:
        self._tool_def = value


def is_active(tool_def: ToolDefinition) -> bool:
    """Check if a tool does not need defer_loading or else needs it but has been activated."""
    if not tool_def.defer_loading:
        return True

    metadata = tool_def.metadata or {}
    return bool(metadata.get(_TOOL_SEARCH_METADATA_KEY, {}).get('active'))


def is_search_tool(tool_def: ToolDefinition) -> bool:
    """Check if this tool is a tool implementing search and loading."""
    metadata = tool_def.metadata or {}
    return bool(metadata.get(_TOOL_SEARCH_METADATA_KEY, {}).get('search'))


# TODO add a typed record for tool search metadata perhaps.
def _update_tool_def_metadata(
    tool_def: ToolDefinition,
    is_search_tool: bool = False,
    is_active: bool = False,
) -> ToolDefinition:
    new_metadata: dict[str, Any] | None = None
    new_metadata = _update_metadata(tool_def.metadata, is_search_tool=is_search_tool, is_active=is_active)
    return replace(tool_def, metadata=new_metadata)


def _update_metadata(
    metadata: dict[str, Any] | None = None,
    is_search_tool: bool = False,
    is_active: bool = False,
) -> dict[str, Any] | None:
    updated_metadata = metadata.copy() if metadata else {}
    if is_search_tool or is_active:
        updated_metadata[_TOOL_SEARCH_METADATA_KEY] = {
            'search': is_search_tool,
            'active': is_active,
        }
    elif _TOOL_SEARCH_METADATA_KEY in updated_metadata:
        del updated_metadata[_TOOL_SEARCH_METADATA_KEY]

    # Avoid gratuitously changing None to {}.
    if len(updated_metadata) == 0:
        updated_metadata = metadata

    return updated_metadata
