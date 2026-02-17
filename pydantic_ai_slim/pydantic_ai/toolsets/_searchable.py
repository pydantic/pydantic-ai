from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from pydantic_core import SchemaValidator, core_schema

from .._run_context import AgentDepsT, RunContext
from ..exceptions import ModelRetry, UserError
from ..messages import ModelRequest, ToolReturn, ToolReturnPart
from ..tools import ToolDefinition
from .abstract import ToolsetTool
from .wrapper import WrapperToolset

SEARCH_TOOLS_NAME = 'search_tools'

_MAX_SEARCH_RESULTS = 10

_SEARCH_TOOLS_VALIDATOR = SchemaValidator(
    schema=core_schema.typed_dict_schema(
        {
            'query': core_schema.typed_dict_field(core_schema.str_schema(), required=True),
        }
    )
)

_SEARCH_TOOL_DEF = ToolDefinition(
    name=SEARCH_TOOLS_NAME,
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
    },
)


@dataclass(kw_only=True)
class _SearchToolsetTool(ToolsetTool[AgentDepsT]):
    deferred_tools: dict[str, ToolsetTool[AgentDepsT]]


@dataclass
class SearchableToolset(WrapperToolset[AgentDepsT]):
    """A toolset that enables tool discovery for large toolsets.

    This toolset wraps another toolset and provides a `search_tools` tool that allows
    the model to discover tools marked with `defer_loading=True`.

    Tools with `defer_loading=True` are not initially presented to the model.
    Instead, they become available after the model discovers them via the search tool.
    """

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        all_tools = await self.wrapped.get_tools(ctx)

        deferred: dict[str, ToolsetTool[AgentDepsT]] = {}
        non_deferred: dict[str, ToolsetTool[AgentDepsT]] = {}
        for name, tool in all_tools.items():
            if tool.tool_def.defer_loading:
                deferred[name] = tool
            else:
                non_deferred[name] = tool

        if not deferred:
            return all_tools

        if SEARCH_TOOLS_NAME in all_tools:
            raise UserError(
                f"Tool name '{SEARCH_TOOLS_NAME}' is reserved for tool search. Rename your tool to avoid conflicts."
            )

        discovered = self._parse_discovered_tools(ctx)

        search_tool = _SearchToolsetTool(
            toolset=self,
            tool_def=_SEARCH_TOOL_DEF,
            max_retries=1,
            args_validator=_SEARCH_TOOLS_VALIDATOR,
            deferred_tools=deferred,
        )

        result: dict[str, ToolsetTool[AgentDepsT]] = {SEARCH_TOOLS_NAME: search_tool}
        result.update(non_deferred)
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
                    if isinstance(part, ToolReturnPart) and part.tool_name == SEARCH_TOOLS_NAME:
                        metadata = part.metadata
                        if isinstance(metadata, list):
                            for item in cast(list[Any], metadata):
                                if isinstance(item, str):
                                    discovered.add(item)
        return discovered

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if name == SEARCH_TOOLS_NAME and isinstance(tool, _SearchToolsetTool):
            return await self._search_tools(tool_args, tool.deferred_tools)
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)

    async def _search_tools(
        self, tool_args: dict[str, Any], deferred_tools: dict[str, ToolsetTool[AgentDepsT]]
    ) -> ToolReturn:
        """Search for tools matching the query."""
        query = tool_args.get('query', '')
        if not query:
            raise ModelRetry('Please provide a search query.')

        if not deferred_tools:
            return ToolReturn(
                return_value={'message': 'No searchable tools available.', 'tools': []},
                metadata=[],
            )

        query_lower = query.lower()

        matches: list[dict[str, str | None]] = []
        for name, tool in deferred_tools.items():
            tool_def = tool.tool_def
            name_match = query_lower in name.lower()
            desc_match = query_lower in tool_def.description.lower() if tool_def.description else False

            if name_match or desc_match:
                matches.append(
                    {
                        'name': name,
                        'description': tool_def.description,
                    }
                )

        matches = matches[:_MAX_SEARCH_RESULTS]

        if matches:
            message = f"Found {len(matches)} tool(s) matching '{query}'"
        else:
            message = f"No tools found matching '{query}'"

        tool_names = [m['name'] for m in matches]

        return ToolReturn(
            return_value={'message': message, 'tools': matches},
            metadata=tool_names,
        )
