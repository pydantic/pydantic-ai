from __future__ import annotations

import re
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
_SEARCH_TOKEN_RE = re.compile(r'[a-z0-9]+')


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


# TypeAdapter doesn't support config= for TypedDict, so we fix the title on the generated schema
# to avoid leaking the private class name '_SearchToolArgs' to the model.
_search_tool_args_ta = TypeAdapter(_SearchToolArgs)
_SEARCH_TOOL_SCHEMA = _search_tool_args_ta.json_schema()
_SEARCH_TOOL_SCHEMA['title'] = 'SearchToolArgs'


@dataclass(kw_only=True)
class _SearchIndexEntry:
    name: str
    description: str | None
    search_terms: set[str]


@dataclass(kw_only=True)
class _SearchTool(ToolsetTool[AgentDepsT]):
    search_index: list[_SearchIndexEntry]


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
            _SearchIndexEntry(
                name=name,
                description=tool.tool_def.description,
                search_terms=self._search_terms(name, tool.tool_def.description),
            )
            for name, tool in deferred.items()
            if name not in discovered
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
            parameters_json_schema=_SEARCH_TOOL_SCHEMA,
        )

        search_tool = _SearchTool(
            toolset=self,
            tool_def=search_tool_def,
            max_retries=1,
            args_validator=_search_tool_args_ta.validator,  # pyright: ignore[reportArgumentType]
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
                    if (
                        isinstance(part, ToolReturnPart)
                        and part.tool_name == _SEARCH_TOOLS_NAME
                        and isinstance(metadata := part.metadata, dict)
                        and isinstance(tool_names := metadata.get(_DISCOVERED_TOOLS_METADATA_KEY), list)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                    ):
                        discovered.update(item for item in tool_names if isinstance(item, str))  # pyright: ignore[reportUnknownVariableType]
        return discovered

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if name == _SEARCH_TOOLS_NAME and isinstance(tool, _SearchTool):
            return await self._search_tools(tool_args, tool)
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)

    @staticmethod
    def _search_terms(name: str, description: str | None) -> set[str]:
        search_terms = set(_SEARCH_TOKEN_RE.findall(name.lower()))
        if description:
            search_terms.update(_SEARCH_TOKEN_RE.findall(description.lower()))
        return search_terms

    async def _search_tools(self, tool_args: dict[str, Any], search_tool: _SearchTool[AgentDepsT]) -> ToolReturn:
        """Search for tools ordered by token overlap with the query.

        Tokenizes both the query and each tool's name/description on alphanumeric runs,
        scores each tool by how many query tokens it contains, and returns all positive
        matches ordered from highest to lowest score. This prefers more specific
        matches such as `github_get_me` for "github profile" without matching
        substrings inside words like `comment` for the query `me`.
        """
        keywords = tool_args['keywords']
        if not keywords:
            raise ModelRetry('Please provide search keywords.')

        terms = self._search_terms(keywords, None)
        if not terms:
            raise ModelRetry('Please provide search keywords.')

        scored_matches: list[tuple[int, dict[str, str | None]]] = []
        for entry in search_tool.search_index:
            score = len(terms & entry.search_terms)
            if score == 0:
                continue
            scored_matches.append((score, {'name': entry.name, 'description': entry.description}))

        if not scored_matches:
            return ToolReturn(
                return_value='No matching tools found. The tools you need may not be available.',
                metadata={_DISCOVERED_TOOLS_METADATA_KEY: []},
            )

        scored_matches.sort(key=lambda item: item[0], reverse=True)
        matches = [match for _, match in scored_matches[:_MAX_SEARCH_RESULTS]]
        tool_names = [match['name'] for match in matches]

        return ToolReturn(
            return_value=matches,
            metadata={_DISCOVERED_TOOLS_METADATA_KEY: tool_names},
        )
