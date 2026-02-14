from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic_core import SchemaValidator, core_schema

from .._run_context import AgentDepsT, RunContext
from ..exceptions import UserError
from ..messages import ModelRequest, ToolReturnPart
from ..tools import ToolDefinition
from .abstract import ToolsetTool
from .wrapper import WrapperToolset

SEARCH_TOOLS_NAME = 'search_tools'

_SEARCH_TOOLS_VALIDATOR = SchemaValidator(
    schema=core_schema.typed_dict_schema(
        {
            'query': core_schema.typed_dict_field(core_schema.str_schema(), required=True),
        }
    )
)


@dataclass
class SearchableToolset(WrapperToolset[AgentDepsT]):
    """A toolset that enables tool discovery for large toolsets.

    This toolset wraps another toolset and provides a `search_tools` tool that allows
    the model to discover tools marked with `defer_loading=True`.

    Tools with `defer_loading=True` are not initially presented to the model.
    Instead, they become available after the model discovers them via the search tool.
    """

    max_results: int = 5
    """Maximum number of tools to return from a search query."""

    _search_tool: ToolsetTool[AgentDepsT] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._search_tool = ToolsetTool(
            toolset=self,
            tool_def=ToolDefinition(
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
            ),
            max_retries=1,
            args_validator=_SEARCH_TOOLS_VALIDATOR,
        )

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        all_tools = await self.wrapped.get_tools(ctx)

        if SEARCH_TOOLS_NAME in all_tools:
            raise UserError(
                f"Tool name '{SEARCH_TOOLS_NAME}' is reserved for tool search. Rename your tool to avoid conflicts."
            )

        deferred: dict[str, ToolsetTool[AgentDepsT]] = {}
        non_deferred: dict[str, ToolsetTool[AgentDepsT]] = {}
        for name, tool in all_tools.items():
            if tool.tool_def.defer_loading:
                deferred[name] = tool
            else:
                non_deferred[name] = tool

        if not deferred:
            return all_tools

        discovered = self._parse_discovered_tools(ctx)

        result: dict[str, ToolsetTool[AgentDepsT]] = {SEARCH_TOOLS_NAME: self._search_tool}
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
                        content = part.model_response_object()
                        if isinstance(content, dict) and 'tools' in content:
                            tools_list = content['tools']
                            if isinstance(tools_list, list):
                                for tool_info in cast(list[dict[str, Any]], tools_list):
                                    if isinstance(tool_info, dict) and 'name' in tool_info:
                                        tool_name = tool_info['name']
                                        if isinstance(tool_name, str):
                                            discovered.add(tool_name)
        return discovered

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if name == SEARCH_TOOLS_NAME:
            return await self._search_tools(tool_args, ctx)
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)

    async def _search_tools(self, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT]) -> dict[str, Any]:
        """Search for tools matching the query."""
        query = tool_args.get('query', '')
        if not query:
            return {'message': 'Please provide a search query.', 'tools': []}

        all_tools = await self.wrapped.get_tools(ctx)

        deferred_tools = [(name, tool) for name, tool in all_tools.items() if tool.tool_def.defer_loading]

        if not deferred_tools:
            return {'message': 'No searchable tools available.', 'tools': []}

        pattern = re.compile(re.escape(query), re.IGNORECASE)

        matches: list[dict[str, str | None]] = []
        for name, tool in deferred_tools:
            tool_def = tool.tool_def
            name_match = pattern.search(name)
            desc_match = pattern.search(tool_def.description) if tool_def.description else None

            if name_match or desc_match:
                matches.append(
                    {
                        'name': name,
                        'description': tool_def.description,
                    }
                )

        matches = matches[: self.max_results]

        if matches:
            message = f"Found {len(matches)} tool(s) matching '{query}'"
        else:
            message = f"No tools found matching '{query}'"

        return {'message': message, 'tools': matches}
