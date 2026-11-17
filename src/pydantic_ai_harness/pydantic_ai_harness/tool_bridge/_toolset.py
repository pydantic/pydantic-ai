"""Toolset that puts selected tools behind three fixed dispatcher tools."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict

from pydantic_ai import ToolDefinition
from pydantic_ai.capabilities import ToolSearchFunc
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import AgentDepsT, RunContext, ToolSelector, matches_tool_selector
from pydantic_ai.toolsets import WrapperToolset
from pydantic_ai.toolsets._tool_search import keywords_search_fn
from pydantic_ai.toolsets.abstract import SchemaValidatorProt, ToolsetTool
from pydantic_ai_harness._code_tools import FRAMEWORK_CONTROL_TOOL_KINDS, reserve_nested_call

SEARCH_TOOL_NAME = 'tool_search'
DESCRIBE_TOOL_NAME = 'tool_describe'
CALL_TOOL_NAME = 'tool_call'


class _SearchArgs(TypedDict):
    query: str


class _DescribeArgs(TypedDict):
    name: str


class _CallArgs(TypedDict):
    name: str
    args_json: str


class _Match(TypedDict):
    name: str
    description: str | None


class _Description(TypedDict):
    name: str
    description: str | None
    parameters_json_schema: dict[str, Any]


def _dispatcher(description: str, args: type[Any]) -> tuple[str, dict[str, Any], SchemaValidatorProt]:
    adapter = TypeAdapter(args)
    return description, adapter.json_schema(), adapter.validator  # pyright: ignore[reportReturnType]


_DISPATCHERS = {
    SEARCH_TOOL_NAME: _dispatcher(
        'Search the tools available through `tool_call` by keywords. '
        'Returns the names and descriptions of the matching tools.',
        _SearchArgs,
    ),
    DESCRIBE_TOOL_NAME: _dispatcher(
        'Get the description and the JSON schema of the arguments of a tool available through `tool_call`.',
        _DescribeArgs,
    ),
    CALL_TOOL_NAME: _dispatcher(
        'Call a tool found with `tool_search`. Pass its name, and its arguments as a JSON object string '
        'matching the schema `tool_describe` returns, for example `{"city": "Paris"}`, or `{}` for none.',
        _CallArgs,
    ),
}


@dataclass(kw_only=True)
class _DispatcherTool(ToolsetTool[AgentDepsT]):
    bridged: dict[str, ToolsetTool[AgentDepsT]] = field(repr=False)
    """This step's bridged tools, so a call resolves against the tools the model could have found."""


@dataclass
class ToolBridgeToolset(WrapperToolset[AgentDepsT]):
    """Implementation toolset for the [`ToolBridge`][pydantic_ai_harness.ToolBridge] capability.

    Replaces the selected tools with `tool_search`, `tool_describe` and `tool_call`, whose
    definitions don't depend on the tools behind them. The other tools pass through unchanged.
    """

    tools: ToolSelector[AgentDepsT] = 'all'
    """Which wrapped tools to bridge; the rest stay native."""

    search: ToolSearchFunc[AgentDepsT] = keywords_search_fn
    """Ranks the bridged tools for a `tool_search` query."""

    max_results: int = 10
    """Maximum number of `tool_search` results."""

    max_retries: int = 3
    """Retries for each dispatcher tool, spent on unknown tool names, invalid arguments and bridged tools' retries."""

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Return the dispatcher tools plus the tools that aren't bridged."""
        native: dict[str, ToolsetTool[AgentDepsT]] = {}
        bridged: dict[str, ToolsetTool[AgentDepsT]] = {}
        for name, tool in (await self.wrapped.get_tools(ctx)).items():
            if await self._bridges(ctx, tool.tool_def):
                # The bridge discloses its tools itself, so tool search must not hold a deferred one back.
                bridged[name] = replace(tool, tool_def=replace(tool.tool_def, defer_loading=False))
            else:
                native[name] = tool
        # `sequential` only decides how calls are scheduled, so changing it leaves the provider's tool list as it was.
        sequential = any(tool.tool_def.sequential for tool in bridged.values())
        for name, (description, schema, validator) in _DISPATCHERS.items():
            if name in native:
                raise UserError(f'Tool name {name!r} is reserved for `ToolBridge`. Rename the tool or bridge it.')
            native[name] = _DispatcherTool(
                toolset=self,
                tool_def=ToolDefinition(
                    name=name,
                    description=description,
                    parameters_json_schema=schema,
                    sequential=sequential and name == CALL_TOOL_NAME,
                ),
                max_retries=self.max_retries,
                args_validator=validator,
                bridged=bridged,
            )
        return native

    async def _bridges(self, ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> bool:
        """Whether a wrapped tool goes behind the bridge.

        Framework control tools, local fallbacks for native tools, and tools that need approval or run
        outside the agent stay native. A hidden tool a capability owns is bridged once it's revealed.
        """
        return (
            tool_def.kind == 'function'
            and tool_def.tool_kind not in FRAMEWORK_CONTROL_TOOL_KINDS
            and not tool_def.unless_native
            and (tool_def.capability_id is None or ctx.is_tool_available(tool_def))
            and await matches_tool_selector(self.tools, ctx, tool_def)
        )

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        """Run a dispatcher tool, or pass a native tool through to the wrapped toolset."""
        if not isinstance(tool, _DispatcherTool):
            return await self.wrapped.call_tool(name, tool_args, ctx, tool)
        if name == SEARCH_TOOL_NAME:
            return await self._search(ctx, tool_args['query'], tool.bridged)
        target = tool.bridged.get(tool_args['name'])
        if target is None:
            raise ModelRetry(
                f'No tool named {tool_args["name"]!r} is available through `tool_call`. Use `tool_search` to find one.'
            )
        if name == DESCRIBE_TOOL_NAME:
            tool_def = target.tool_def
            return _Description(
                name=tool_def.name,
                description=tool_def.description,
                parameters_json_schema=tool_def.parameters_json_schema,
            )
        return await self._call(ctx, target.tool_def.name, tool_args['args_json'], tool.bridged)

    async def _search(
        self, ctx: RunContext[AgentDepsT], query: str, bridged: dict[str, ToolsetTool[AgentDepsT]]
    ) -> list[_Match]:
        names = self.search(ctx, [query], [tool.tool_def for tool in bridged.values()])
        if inspect.isawaitable(names):
            names = await names
        matches = [
            _Match(name=name, description=bridged[name].tool_def.description) for name in names if name in bridged
        ]
        return matches[: self.max_results]

    async def _call(
        self, ctx: RunContext[AgentDepsT], name: str, args_json: str, bridged: dict[str, ToolsetTool[AgentDepsT]]
    ) -> Any:
        """Run a bridged tool through a nested `ToolManager`, so its validation and hooks apply as to a direct call.

        A retry the tool asks for is a retry of `tool_call`, so the tool sees that call's retry count as its own.
        """
        parent = ctx.tool_manager
        if parent is None:
            raise UserError(
                "`ToolBridgeToolset` needs the run's tool manager, so bridged calls run through tool hooks."
            )
        nested_ctx = replace(ctx, retries={**ctx.retries, name: ctx.retry})
        manager = ToolManager(
            toolset=self.wrapped, root_capability=parent.root_capability, ctx=nested_ctx, tools=bridged
        )
        call = ToolCallPart(tool_name=name, args=args_json or '{}')
        reserved = reserve_nested_call(ctx)
        try:
            return await manager.handle_call(
                call, approved=ctx.tool_call_approved, metadata=ctx.tool_call_metadata, wrap_validation_errors=False
            )
        except ValidationError as exc:
            errors = exc.errors(include_url=False, include_context=False)
            raise ModelRetry(f'Invalid arguments for tool {name!r}: {errors}') from exc
        finally:
            ctx.usage.tool_calls -= reserved
