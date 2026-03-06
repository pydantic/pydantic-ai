from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_core import SchemaValidator, core_schema

from .._run_context import AgentDepsT, RunContext
from ..exceptions import ModelRetry, UserError
from ..messages import ModelRequest, ToolReturn, ToolReturnPart
from ..tools import ToolDefinition
from .abstract import ToolsetTool
from .wrapper import WrapperToolset

_SEARCH_TOOLS_NAME = 'search_tools'

_DISCOVERED_TOOLS_METADATA_KEY = 'discovered_tools'

_MAX_SEARCH_RESULTS = 10

_SEARCH_TOOLS_VALIDATOR = SchemaValidator(
    schema=core_schema.typed_dict_schema(
        {
            'query': core_schema.typed_dict_field(core_schema.str_schema(), required=True),
        }
    )
)

_SEARCH_TOOL_DEF = ToolDefinition(
    name=_SEARCH_TOOLS_NAME,
    description='Search for available tools by keyword. Returns matching tool names and descriptions.',
    parameters_json_schema={
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': 'The search query to match against tool names and descriptions.',
            }
        },
        'required': ['query'],
        'additionalProperties': False,
    },
)


@dataclass(kw_only=True)
class _SearchToolIndex:
    name: str
    name_lower: str
    description: str | None
    description_lower: str | None


@dataclass(kw_only=True)
class _SearchTool(ToolsetTool[AgentDepsT]):
    lazy_tools: dict[str, ToolsetTool[AgentDepsT]]
    search_index: list[_SearchToolIndex]


@dataclass
class SearchableToolset(WrapperToolset[AgentDepsT]):
    """A toolset that enables tool discovery for large toolsets.

    This toolset wraps another toolset and provides a `search_tools` tool that allows
    the model to discover tools marked with `lazy=True`.

    Tools with `lazy=True` are not initially presented to the model.
    Instead, they become available after the model discovers them via the search tool.
    """

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        all_tools = await self.wrapped.get_tools(ctx)

        lazy: dict[str, ToolsetTool[AgentDepsT]] = {}
        visible: dict[str, ToolsetTool[AgentDepsT]] = {}
        for name, tool in all_tools.items():
            if tool.tool_def.lazy:
                lazy[name] = tool
            else:
                visible[name] = tool

        if not lazy:
            return all_tools

        if _SEARCH_TOOLS_NAME in all_tools:
            raise UserError(
                f"Tool name '{_SEARCH_TOOLS_NAME}' is reserved for tool search. Rename your tool to avoid conflicts."
            )

        discovered = self._parse_discovered_tools(ctx)

        search_index = [
            _SearchToolIndex(
                name=name,
                name_lower=name.lower(),
                description=tool.tool_def.description,
                description_lower=tool.tool_def.description.lower() if tool.tool_def.description else None,
            )
            for name, tool in lazy.items()
        ]

        search_tool = _SearchTool(
            toolset=self,
            tool_def=_SEARCH_TOOL_DEF,
            max_retries=1,
            args_validator=_SEARCH_TOOLS_VALIDATOR,
            lazy_tools=lazy,
            search_index=search_index,
        )

        result: dict[str, ToolsetTool[AgentDepsT]] = {_SEARCH_TOOLS_NAME: search_tool}
        result.update(visible)
        for name, tool in lazy.items():
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
        """Search for tools matching the query."""
        query = tool_args['query']
        if not query:
            raise ModelRetry('Please provide a search query.')

        query_lower = query.lower()

        matches: list[dict[str, str | None]] = []
        for entry in search_tool.search_index:
            name_match = query_lower in entry.name_lower
            desc_match = query_lower in entry.description_lower if entry.description_lower else False

            if name_match or desc_match:
                matches.append({'name': entry.name, 'description': entry.description})
                if len(matches) >= _MAX_SEARCH_RESULTS:
                    break

        if matches:
            message = f"Found {len(matches)} tool(s) matching '{query}'"
        else:
            message = f"No tools found matching '{query}'"

        tool_names = [match['name'] for match in matches]

        return ToolReturn(
            return_value={'message': message, 'tools': matches},
            metadata={_DISCOVERED_TOOLS_METADATA_KEY: tool_names},
        )
