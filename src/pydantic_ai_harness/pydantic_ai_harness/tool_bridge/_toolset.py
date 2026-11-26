"""Toolset that puts selected tools behind three fixed dispatcher tools."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import TypeAdapter
from typing_extensions import TypedDict

from pydantic_ai import ToolDefinition
from pydantic_ai._tool_execution import NestedCallContent
from pydantic_ai.capabilities import ToolSearchFunc, keywords_search
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    ToolFailed,
    ToolFailedError,
    ToolRetryError,
    UserError,
)
from pydantic_ai.messages import ToolCallPart, ToolKind, ToolReturn, ToolReturnPart
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import AgentDepsT, RunContext, ToolSelector, matches_tool_selector
from pydantic_ai.toolsets import WrapperToolset
from pydantic_ai.toolsets.abstract import SchemaValidatorProt, ToolsetTool
from pydantic_ai_harness._code_tools import is_code_execution_tool, reserve_nested_call

SEARCH_TOOL_NAME = 'tool_search'
DESCRIBE_TOOL_NAME = 'tool_describe'
CALL_TOOL_NAME = 'tool_call'


class _SearchArgs(TypedDict):
    queries: list[str]


class _DescribeArgs(TypedDict):
    name: str


class ToolCallArgs(TypedDict):
    """Arguments of the `tool_call` dispatcher."""

    name: str
    """The bridged tool to call."""

    args_json: str
    """The bridged tool's arguments, as a JSON object string."""


TOOL_CALL_KIND = ToolKind('tool_bridge', CALL_TOOL_NAME, args=ToolCallArgs)
"""The kind of `ToolBridge`'s `tool_call` tool, so an approval UI or capability can tell it apart whatever it is named."""


class _Match(TypedDict):
    name: str
    description: str | None


class _Description(TypedDict):
    name: str
    description: str | None
    parameters_json_schema: dict[str, Any]


def _dispatcher(description: str, args: type[object]) -> tuple[str, dict[str, Any], SchemaValidatorProt]:
    adapter = TypeAdapter(args)
    return description, adapter.json_schema(), adapter.validator  # pyright: ignore[reportReturnType]


_DISPATCHERS = {
    SEARCH_TOOL_NAME: _dispatcher(
        'Search the tools available through `tool_call` with one or more queries. '
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
        ToolCallArgs,
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

    search: ToolSearchFunc[AgentDepsT] | None = None
    """Ranks the bridged tools for `tool_search`'s queries; `None` matches the queries' words."""

    max_results: int = 10
    """Maximum number of `tool_search` results."""

    max_retries: int | None = None
    """Retries for each dispatcher tool, spent on unknown tool names, invalid arguments and bridged tools' retries.

    `None` (default) uses the agent's tool retry budget, as `Agent(retries={'tools': N})` sets it.
    """

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Return the dispatcher tools plus the tools that aren't bridged."""
        native: dict[str, ToolsetTool[AgentDepsT]] = {}
        bridged: dict[str, ToolsetTool[AgentDepsT]] = {}
        for name, tool in (await self.wrapped.get_tools(ctx)).items():
            if name in _DISPATCHERS:
                raise UserError(f'Tool name {name!r} is reserved for `ToolBridge`. Rename the tool.')
            if await self._bridges(ctx, tool.tool_def):
                # The bridge discloses its tools itself, so tool search must not hold a deferred one back.
                bridged[name] = replace(tool, tool_def=replace(tool.tool_def, defer_loading=False))
            else:
                native[name] = tool
        # `sequential` only decides how calls are scheduled, so changing it leaves the provider's tool list as it was.
        sequential = any(tool.tool_def.sequential for tool in bridged.values())
        for name, (description, schema, validator) in _DISPATCHERS.items():
            native[name] = _DispatcherTool(
                toolset=self,
                tool_def=ToolDefinition(
                    name=name,
                    description=description,
                    parameters_json_schema=schema,
                    sequential=sequential and name == CALL_TOOL_NAME,
                    tool_kind=TOOL_CALL_KIND if name == CALL_TOOL_NAME else None,
                ),
                max_retries=self.max_retries if self.max_retries is not None else ctx.max_retries,
                args_validator=validator,
                bridged=bridged,
            )
        return native

    async def _bridges(self, ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> bool:
        """Whether a wrapped tool goes behind the bridge.

        Tools that stay native:

        - tools with a `tool_kind`, which capabilities and the run find by kind (framework control tools,
          `read_file`), and tools that step aside for one (`unless_tool_kind`), so the run's filter sees both;
        - local fallbacks for native tools, and tools that need approval or run outside the agent;
        - code-execution tools, whose program would otherwise travel JSON-escaped in `args_json`;
        - deferred tools a capability owns, so core's tool search and capability loading reveal them without
          changing the tool definitions already sent.
        """
        return (
            tool_def.kind == 'function'
            and tool_def.tool_kind is None
            and tool_def.unless_tool_kind is None
            and not tool_def.unless_native
            and not is_code_execution_tool(tool_def)
            and not (tool_def.defer_loading and tool_def.capability_id is not None)
            and await matches_tool_selector(self.tools, ctx, tool_def)
        )

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        """Run a dispatcher tool, or pass a native tool through to the wrapped toolset."""
        if not isinstance(tool, _DispatcherTool):
            return await self.wrapped.call_tool(name, tool_args, ctx, tool)
        if name == SEARCH_TOOL_NAME:
            return await self._search(ctx, tool_args['queries'], tool.bridged)
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
        return await self._call(ctx, target, tool_args['args_json'], tool.bridged)

    async def _search(
        self, ctx: RunContext[AgentDepsT], queries: list[str], bridged: dict[str, ToolsetTool[AgentDepsT]]
    ) -> list[_Match]:
        search = self.search or keywords_search
        names = search(ctx, queries, [tool.tool_def for tool in bridged.values()])
        if inspect.isawaitable(names):
            names = await names
        matches = [
            _Match(name=name, description=bridged[name].tool_def.description) for name in names if name in bridged
        ]
        return matches[: self.max_results]

    async def _call(
        self,
        ctx: RunContext[AgentDepsT],
        target: ToolsetTool[AgentDepsT],
        args_json: str,
        bridged: dict[str, ToolsetTool[AgentDepsT]],
    ) -> Any:
        """Run a bridged tool through a nested `ToolManager`, so its validation and hooks apply as to a direct call.

        The nested manager shares the step's record of failed and successful tools with the run's, so the
        tool's retries are counted under its own name, and its `ctx.retry` and `max_retries` see only its
        own failures. A retry or failure it reports is passed on as `tool_call`'s.

        A deferral the tool raises (`ApprovalRequired`, `CallDeferred`) is only resolved by a
        `HandleDeferredToolCalls` handler, which sees the tool's own name. Otherwise the run fails with a
        `UserError` rather than deferring `tool_call`, whose approval would cover whichever tool the model named.
        """
        parent = ctx.tool_manager
        if parent is None:
            raise UserError(
                "`ToolBridgeToolset` needs the run's tool manager, so bridged calls run through tool hooks."
            )
        name = target.tool_def.name
        manager = ToolManager(
            toolset=self.wrapped,
            root_capability=parent.root_capability,
            ctx=ctx,
            tools=bridged,
            failed_tools=parent.failed_tools,
            succeeded_tools=parent.succeeded_tools,
        )
        assert ctx.tool_call_id is not None, 'the run sets the ID of the `tool_call` call being executed'
        # One `tool_call` runs one bridged call, so they share an ID, as a direct call would keep its own.
        call = ToolCallPart(tool_name=name, args=args_json or '{}', tool_call_id=ctx.tool_call_id)
        reserved = reserve_nested_call(ctx)
        # The `tool_call` running this counts only once it returns, so it is pending too: concurrent
        # `tool_call`s would otherwise all pass the check before any of them is counted.
        # usage-attribution: a temporary pending-call reservation, released below
        ctx.usage.tool_calls += reserved
        reserved *= 2
        try:
            result = await manager.handle_call(call)
        except (ApprovalRequired, CallDeferred) as exc:
            raise UserError(
                f'Tool {name!r} raised `{type(exc).__name__}` behind `ToolBridge`, and no `HandleDeferredToolCalls` '
                'handler resolved it. Keep the tool native with `ToolBridge(tools=...)`, or add a handler.'
            ) from exc
        except ToolRetryError as exc:
            retry = exc.tool_retry
            assert isinstance(retry, ToolReturnPart), 'a retry of a tool call answers it'
            message = retry.model_response_str(wrap_if_error=False)
            if not isinstance(retry.content, str):
                message = f'Invalid arguments for tool {name!r}: {message}'
            raise ModelRetry(message) from exc
        except ToolFailedError as exc:
            raise ToolFailed(exc.tool_failed.model_response_str(wrap_if_error=False)) from exc
        finally:
            ctx.usage.tool_calls -= reserved
        # Pyright strict reads `isinstance` on the `Any` that `handle_call` can return as `ToolReturn[Unknown]`, so
        # both returns below, the `ToolReturn` with no content included, carry an unknown-type ignore.
        capability_id = target.tool_def.capability_id
        owner = None if capability_id is None else ctx.capabilities.get(capability_id)
        lowers = owner is not None and owner._effective_authority != 'operator'  # pyright: ignore[reportPrivateUsage]
        if isinstance(result, ToolReturn) and result.content and lowers:
            # Held to the bridged tool's capability's authority, as if the model had called the tool itself. Content
            # from a tool no capability below `'operator'` contributes stays as the tool returned it.
            content = NestedCallContent([(capability_id, result.content)])
            return replace(result, content=content)  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]
        return result  # pyright: ignore[reportUnknownVariableType]
