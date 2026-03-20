from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import Field, TypeAdapter
from typing_extensions import TypedDict

from .._run_context import AgentDepsT, RunContext
from ..exceptions import ModelRetry, UserError
from ..messages import ModelRequest, ToolReturn, ToolReturnPart
from ..tools import ToolDefinition
from .abstract import ToolsetTool
from .wrapper import WrapperToolset

_SEARCH_TOOLS_NAME = 'search_tools'

_DISCOVERED_TOOLS_METADATA_KEY = 'discovered_tools'

_MAX_SEARCH_RESULTS = 10


class _SearchToolArgs(TypedDict):
    keywords: Annotated[
        str,
        Field(
            description=(
                'Space-separated keywords to match against tool names and descriptions.'
                ' Use specific words likely to appear in tool names or descriptions to narrow down relevant tools.'
            )
        ),
    ]


_search_tool_args_ta = TypeAdapter(_SearchToolArgs)


@dataclass(kw_only=True)
class _SearchToolIndex:
    name: str
    name_lower: str
    description: str | None
    description_lower: str | None


@dataclass(kw_only=True)
class _SearchTool(ToolsetTool[AgentDepsT]):
    deferred_tools: dict[str, ToolsetTool[AgentDepsT]]
    search_index: list[_SearchToolIndex]


@dataclass
class ToolSearchToolset(WrapperToolset[AgentDepsT]):
    """A toolset that enables tool discovery for large toolsets.

    This toolset wraps another toolset and provides a `search_tools` tool that allows
    the model to discover tools marked with `defer_loading=True`.

    Tools with `defer_loading=True` are not initially presented to the model.
    Instead, they become available after the model discovers them via the search tool.
    """

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        all_tools = await self.wrapped.get_tools(ctx)

        deferred: dict[str, ToolsetTool[AgentDepsT]] = {}
        visible: dict[str, ToolsetTool[AgentDepsT]] = {}
        for name, tool in all_tools.items():
            if tool.tool_def.defer_loading:
                deferred[name] = tool
            else:
                visible[name] = tool

        if not deferred:
            return all_tools

        if _SEARCH_TOOLS_NAME in all_tools:
            raise UserError(
                f"Tool name '{_SEARCH_TOOLS_NAME}' is reserved for tool search. Rename your tool to avoid conflicts."
            )

        discovered = self._parse_discovered_tools(ctx)

        if discovered.issuperset(deferred):
            return all_tools

        search_index = [
            _SearchToolIndex(
                name=name,
                name_lower=name.lower(),
                description=tool.tool_def.description,
                description_lower=tool.tool_def.description.lower() if tool.tool_def.description else None,
            )
            for name, tool in deferred.items()
        ]

        search_tool_def = ToolDefinition(
            name=_SEARCH_TOOLS_NAME,
            description=(
                'There are additional tools not yet visible to you.'
                ' When you need a capability not provided by your current tools,'
                ' search here by providing specific keywords to discover and activate relevant tools.'
                ' Each keyword is matched independently against tool names and descriptions.'
                ' If no tools are found, they do not exist — do not retry.'
            ),
            parameters_json_schema=_search_tool_args_ta.json_schema(),
        )

        search_tool = _SearchTool(
            toolset=self,
            tool_def=search_tool_def,
            max_retries=1,
            args_validator=_search_tool_args_ta.validator,  # pyright: ignore[reportArgumentType]
            deferred_tools=deferred,
            search_index=search_index,
        )

        result: dict[str, ToolsetTool[AgentDepsT]] = {_SEARCH_TOOLS_NAME: search_tool}
        result.update(visible)
        for name, tool in deferred.items():
            if name in discovered:
                result[name] = tool

        return result

    def _parse_discovered_tools(self, ctx: RunContext[AgentDepsT]) -> set[str]:
        """Parse message history to find tools discovered via search_tools."""
        discovered: set[str] = set()
        for msg in ctx.messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart) and part.tool_name == _SEARCH_TOOLS_NAME:
                        metadata = part.metadata
                        if isinstance(metadata, dict):
                            tool_names = metadata.get(_DISCOVERED_TOOLS_METADATA_KEY)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                            if isinstance(tool_names, list):
                                for item in tool_names:  # pyright: ignore[reportUnknownVariableType]
                                    if isinstance(item, str):
                                        discovered.add(item)
        return discovered

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if name == _SEARCH_TOOLS_NAME and isinstance(tool, _SearchTool):
            return await self._search_tools(tool_args, tool)
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)

    async def _search_tools(self, tool_args: dict[str, Any], search_tool: _SearchTool[AgentDepsT]) -> ToolReturn:
        """Search for tools matching the keywords.

        Splits the keywords into individual terms and matches any term against
        tool names and descriptions. This handles multi-keyword queries
        like "exchange rate" matching a tool named "get_exchange_rate".
        """
        keywords = tool_args['keywords']
        if not keywords:
            raise ModelRetry('Please provide search keywords.')

        terms = keywords.lower().split()

        matches: list[dict[str, str | None]] = []
        for entry in search_tool.search_index:
            searchable = entry.name_lower + (' ' + entry.description_lower if entry.description_lower else '')
            if any(term in searchable for term in terms):
                matches.append({'name': entry.name, 'description': entry.description})
                if len(matches) >= _MAX_SEARCH_RESULTS:
                    break

        tool_names = [match['name'] for match in matches]

        if not matches:
            return ToolReturn(
                return_value='No matching tools found. The tools you need may not be available.',
                metadata={_DISCOVERED_TOOLS_METADATA_KEY: tool_names},
            )

        return ToolReturn(
            return_value=matches,
            metadata={_DISCOVERED_TOOLS_METADATA_KEY: tool_names},
        )
