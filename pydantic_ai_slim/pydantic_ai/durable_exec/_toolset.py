from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias, cast

from pydantic import Discriminator, Tag
from typing_extensions import Self, assert_never

from pydantic_ai import AbstractToolset, FunctionToolset, ToolsetTool, WrapperToolset
from pydantic_ai._enqueue import PendingMessage
from pydantic_ai._utils import is_str_dict
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.messages import InstructionPart, ToolReturn, ToolReturnContent
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.toolsets.external import TOOL_SCHEMA_VALIDATOR

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPToolset

DurableConfig: TypeAlias = Mapping[str, Any]
ToolConfig: TypeAlias = DurableConfig | Literal[False]
Lifecycle: TypeAlias = Literal['enter-outside-durable', 'enter-always', 'enter-never']
Instructions: TypeAlias = str | InstructionPart | Sequence[str | InstructionPart] | None
CallToolOperation: TypeAlias = Callable[
    [str, dict[str, Any], RunContext[Any], ToolsetTool[Any], DurableConfig], Awaitable[Any]
]
"""Runs one tool call inside the engine's durable unit (activity/step/task)."""
ResolveToolConfig: TypeAlias = Callable[[ToolsetTool[Any] | None, str], ToolConfig]
"""Resolve a tool's per-tool durable config: a config mapping to merge, or `False` to run the tool inline.

Engines that restrict inline execution enforce it here, where the engine's own error
wording is available (e.g. Temporal requires async tools and forbids inline MCP tools).
"""


@dataclass
class DynamicToolInfo:
    """Serializable tool information returned from dynamic tool discovery."""

    tool_def: ToolDefinition
    max_retries: int


@dataclass
class DynamicToolsResult:
    """Serializable result of the dynamic toolset's tool discovery operation.

    Instructions are collected in the same durable unit (and thus the same single resolution and entry of
    the inner toolset) as the tools. For an MCP-backed dynamic toolset this means the server is entered
    once per run step instead of once for tools and again for instructions; the second entry would add a
    redundant `initialize` round-trip whose `notifications/initialized` races teardown.
    """

    tools: dict[str, DynamicToolInfo]
    instructions: Instructions


async def get_dynamic_tools(toolset: AbstractToolset[AgentDepsT], ctx: RunContext[AgentDepsT]) -> DynamicToolsResult:
    """Resolve a dynamic toolset fresh and collect its tools and instructions in a single entry.

    Self-contained on purpose: each durable unit (activity/step/task) re-resolves the toolset
    rather than relying on state left behind by another unit, so replay/recovery in a fresh
    process stays deterministic.
    """
    run_toolset = await toolset.for_run(ctx)
    async with run_toolset:
        run_toolset = await run_toolset.for_run_step(ctx)
        tools = await run_toolset.get_tools(ctx)
        instructions = await run_toolset.get_instructions(ctx)
        return DynamicToolsResult(
            tools={
                name: DynamicToolInfo(tool_def=tool.tool_def, max_retries=tool.max_retries)
                for name, tool in tools.items()
            },
            instructions=instructions,
        )


async def call_dynamic_tool(
    toolset: AbstractToolset[AgentDepsT], name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT]
) -> Any:
    """Resolve a dynamic toolset fresh, re-validate the tool args, and call the tool.

    The args were only parsed (not validated) on the workflow/flow side, where the real tool
    isn't available; validation happens here against the resolved tool's own validator.
    """
    run_toolset = await toolset.for_run(ctx)
    async with run_toolset:
        run_toolset = await run_toolset.for_run_step(ctx)
        tools = await run_toolset.get_tools(ctx)
        tool = tools.get(name)
        if tool is None:  # pragma: no cover
            raise UserError(
                f'Tool {name!r} not found in dynamic toolset {toolset.id!r}. '
                'The dynamic toolset function may have returned a different toolset than expected.'
            )
        args = tool.args_validator.validate_python(tool_args)
        return await run_toolset.call_tool(name, args, ctx, tool)


@dataclass
class _ApprovalRequired:
    metadata: dict[str, Any] | None = None
    kind: Literal['approval_required'] = 'approval_required'


@dataclass
class _CallDeferred:
    metadata: dict[str, Any] | None = None
    kind: Literal['call_deferred'] = 'call_deferred'


@dataclass
class _ModelRetry:
    message: str
    kind: Literal['model_retry'] = 'model_retry'


def _result_discriminator(value: Any) -> str:
    if isinstance(value, ToolReturn) or (is_str_dict(value) and value.get('kind') == 'tool-return'):
        return 'tool-return'
    return 'content'


_ToolReturnResult = Annotated[
    Annotated[ToolReturn, Tag('tool-return')] | Annotated[ToolReturnContent, Tag('content')],
    Discriminator(_result_discriminator),
]


@dataclass
class _ToolReturn:
    """Legacy wire shape retained for decoding in-flight durable executions."""

    result: _ToolReturnResult
    kind: Literal['tool_return'] = 'tool_return'


@dataclass
class _ToolContentResult:
    # Emitted only when a user dict's `kind` collides with `'tool-return'`. Workers predating this
    # variant cannot decode it, but those payloads already failed to round-trip there; ordinary
    # results deliberately retain the legacy `tool_return` shape for rolling upgrades.
    result: ToolReturnContent
    kind: Literal['tool_content_result'] = 'tool_content_result'


CallToolResult = Annotated[
    _ApprovalRequired | _CallDeferred | _ModelRetry | _ToolReturn | _ToolContentResult,
    Discriminator('kind'),
]


async def wrap_tool_call_result(coro: Awaitable[Any]) -> CallToolResult:
    try:
        result = await coro
        if is_str_dict(result) and result.get('kind') == 'tool-return':
            return _ToolContentResult(result=result)
        return _ToolReturn(result=result)
    except ApprovalRequired as exc:
        return _ApprovalRequired(metadata=exc.metadata)
    except CallDeferred as exc:
        return _CallDeferred(metadata=exc.metadata)
    except ModelRetry as exc:
        return _ModelRetry(message=exc.message)


def unwrap_tool_call_result(result: CallToolResult) -> Any:
    if isinstance(result, _ToolReturn | _ToolContentResult):
        return result.result
    if isinstance(result, _ApprovalRequired):
        raise ApprovalRequired(metadata=result.metadata)
    if isinstance(result, _CallDeferred):
        raise CallDeferred(metadata=result.metadata)
    if isinstance(result, _ModelRetry):
        raise ModelRetry(result.message)
    assert_never(result)


class EnqueueGuard(list[PendingMessage]):
    """Replaces `ctx.pending_messages` inside durable-unit-wrapped tools, where enqueueing can't be supported.

    A durable unit's recorded output is replayed on recovery (DBOS) or cache hit (Prefect)
    without re-executing the tool, so messages enqueued inside it would be silently dropped;
    enqueueing raises the engine's explanatory `UserError` instead.
    """

    def __init__(self, message: str):
        super().__init__()
        self._message = message

    def append(self, pending: PendingMessage) -> None:
        raise UserError(self._message)


def unwrap_recorded_tool_call_result(result: Any) -> Any:
    """Unwrap a durably-recorded tool result, passing raw pre-wrapper values through.

    Engines that replay recorded durable-unit outputs (DBOS step recovery, Prefect task
    caches) may hold outputs recorded before the unit wrapped control-flow exceptions as
    values; those recordings are the raw tool result and are returned unchanged.
    """
    if isinstance(result, _ToolReturn | _ToolContentResult | _ApprovalRequired | _CallDeferred | _ModelRetry):
        return unwrap_tool_call_result(result)
    return result


def resolve_tool_durable_config(
    tool: ToolsetTool[Any] | None,
    tool_name: str,
    fallback_config: Mapping[str, ToolConfig],
    *,
    metadata_key: str,
    config_type_label: str,
) -> ToolConfig:
    """Resolve a tool's durable config: tool metadata under `metadata_key` first, then `fallback_config` by name."""
    if tool is not None and tool.tool_def.metadata is not None:
        metadata_config = tool.tool_def.metadata.get(metadata_key)
        if metadata_config is False:
            return False
        if metadata_config is not None:
            if not isinstance(metadata_config, dict):
                raise UserError(
                    f'Tool {tool_name!r} has invalid {metadata_key!r} metadata: expected a dict '
                    f'(`{config_type_label}`) or `False`, got {type(metadata_config).__name__}.'
                )
            return cast('DurableConfig', metadata_config)
    return fallback_config.get(tool_name, {})


class DurableToolsetBase(WrapperToolset[AgentDepsT]):
    """Shared workflow/flow-side scaffolding for the engines' durable toolset wrappers.

    Mirrors [`DurableModel`][pydantic_ai.durable_exec._utils.DurableModel]: everything
    engine-specific lives in the segment callables the engine supplies, each running one
    operation inside the engine's durable unit (activity/step/task).
    """

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        *,
        in_durable_context: Callable[[], bool],
        lifecycle: Lifecycle,
        durable_registrations: list[Any] | None,
        durable_config: Mapping[str, Any] | None = None,
    ):
        super().__init__(wrapped)
        self._in_durable_context = in_durable_context
        self._lifecycle = lifecycle
        self.durable_registrations = durable_registrations or []
        """Opaque engine handles that must be registered with the engine (e.g. Temporal activities)."""
        self.durable_config = durable_config
        """The engine's base per-operation config for this toolset (e.g. a Temporal `ActivityConfig`)."""

    @property
    def id(self) -> str | None:
        return self.wrapped.id

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        if self._lifecycle == 'enter-outside-durable':
            return self
        return await super().for_run(ctx)

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        if self._lifecycle == 'enter-outside-durable':
            return self
        return await super().for_run_step(ctx)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        return self

    async def __aenter__(self) -> Self:
        should_enter = self._lifecycle == 'enter-always' or (
            self._lifecycle == 'enter-outside-durable' and not self._in_durable_context()
        )
        if should_enter:
            await self.wrapped.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        should_exit = self._lifecycle == 'enter-always' or (
            self._lifecycle == 'enter-outside-durable' and not self._in_durable_context()
        )
        if should_exit:
            return await self.wrapped.__aexit__(*args)
        return None


class DurableFunctionToolset(DurableToolsetBase[AgentDepsT]):
    def __init__(
        self,
        wrapped: FunctionToolset[AgentDepsT],
        *,
        in_durable_context: Callable[[], bool],
        call_tool_operation: CallToolOperation,
        resolve_tool_config: ResolveToolConfig,
        lifecycle: Lifecycle,
        durable_registrations: list[Any] | None = None,
        durable_config: Mapping[str, Any] | None = None,
    ):
        super().__init__(
            wrapped,
            in_durable_context=in_durable_context,
            lifecycle=lifecycle,
            durable_registrations=durable_registrations,
            durable_config=durable_config,
        )
        self._call_tool_operation = call_tool_operation
        self._resolve_tool_config = resolve_tool_config

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if not self._in_durable_context():
            return await self.wrapped.call_tool(name, tool_args, ctx, tool)
        config = self._resolve_tool_config(tool, name)
        if config is False:
            return await self.wrapped.call_tool(name, tool_args, ctx, tool)
        return await self._call_tool_operation(name, tool_args, ctx, tool, config)


class DurableDynamicToolset(DurableToolsetBase[AgentDepsT]):
    def __init__(
        self,
        wrapped: DynamicToolset[AgentDepsT],
        *,
        in_durable_context: Callable[[], bool],
        get_tools_operation: Callable[[RunContext[AgentDepsT]], Awaitable[DynamicToolsResult]],
        call_tool_operation: CallToolOperation,
        resolve_tool_config: ResolveToolConfig,
        lifecycle: Lifecycle,
        durable_registrations: list[Any] | None = None,
        durable_config: Mapping[str, Any] | None = None,
    ):
        super().__init__(
            wrapped,
            in_durable_context=in_durable_context,
            lifecycle=lifecycle,
            durable_registrations=durable_registrations,
            durable_config=durable_config,
        )
        self._get_tools_operation = get_tools_operation
        self._call_tool_operation = call_tool_operation
        self._resolve_tool_config = resolve_tool_config
        self._run_instructions: Instructions = None

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        if not self._in_durable_context():
            # Fully transparent outside the durable context: resolve the dynamic toolset
            # and hand the run its resolved form directly, without the durable dispatch.
            # (The wrapped `DynamicToolset` only resolves in `for_run`; delegating the
            # individual methods to the unresolved factory would silently yield no tools.)
            return await self.wrapped.for_run(ctx)
        # Per-run copy isolates `_run_instructions` from the process-shared instance. The
        # shallow copy shares the engine-registered operations; this is only state isolation.
        run_copy = copy.copy(self)
        run_copy._run_instructions = None
        return run_copy

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        # The per-run copy is stable across steps: resolution happens inside the durable
        # units per call, so a `per_run_step=True` factory must not be re-evaluated in
        # workflow/flow code here. (Outside the durable context this wrapper isn't in the
        # run's tree at all — `for_run` above replaced it with the resolved toolset.)
        return self

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        result = await self._get_tools_operation(ctx)
        self._run_instructions = result.instructions
        return {
            name: ToolsetTool(
                toolset=self,
                tool_def=info.tool_def,
                max_retries=info.max_retries,
                # Only parse here; the real tool validates again inside the durable unit.
                args_validator=TOOL_SCHEMA_VALIDATOR,
            )
            for name, info in result.tools.items()
        }

    async def get_instructions(self, ctx: RunContext[AgentDepsT]) -> Instructions:
        # Set by `get_tools`, which the framework runs earlier in each step.
        return self._run_instructions

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        config = self._resolve_tool_config(tool, name)
        if config is False:
            # The wrapped dynamic toolset is only a construction-time factory; the
            # per-run resolved copy used for discovery has already exited. Resolve a
            # fresh copy in flow code for an explicitly inline call.
            return await call_dynamic_tool(self.wrapped, name, tool_args, ctx)
        return await self._call_tool_operation(name, tool_args, ctx, tool, config)


class DurableMCPToolset(DurableToolsetBase[AgentDepsT]):
    def __init__(
        self,
        wrapped: MCPToolset[AgentDepsT],
        *,
        in_durable_context: Callable[[], bool],
        get_tools_operation: Callable[[RunContext[AgentDepsT]], Awaitable[dict[str, ToolDefinition]]] | None,
        get_instructions_operation: Callable[[RunContext[AgentDepsT]], Awaitable[Instructions]] | None,
        call_tool_operation: CallToolOperation,
        resolve_tool_config: ResolveToolConfig,
        lifecycle: Lifecycle,
        durable_registrations: list[Any] | None = None,
        durable_config: Mapping[str, Any] | None = None,
    ):
        super().__init__(
            wrapped,
            in_durable_context=in_durable_context,
            lifecycle=lifecycle,
            durable_registrations=durable_registrations,
            durable_config=durable_config,
        )
        self._mcp_toolset = wrapped
        self._get_tools_operation = get_tools_operation
        self._get_instructions_operation = get_instructions_operation
        self._call_tool_operation = call_tool_operation
        self._resolve_tool_config = resolve_tool_config

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        if not self._in_durable_context() or self._get_tools_operation is None:
            return await self.wrapped.get_tools(ctx)
        cache_key = self.id or ''
        if self._mcp_toolset.cache_tools and (cached := ctx._mcp_tool_defs_cache.get(cache_key)) is not None:  # pyright: ignore[reportPrivateUsage]
            return {name: self._mcp_toolset.tool_for_tool_def(tool_def) for name, tool_def in cached.items()}
        tool_defs = await self._get_tools_operation(ctx)
        if self._mcp_toolset.cache_tools:
            ctx._mcp_tool_defs_cache[cache_key] = tool_defs  # pyright: ignore[reportPrivateUsage]
        return {name: self._mcp_toolset.tool_for_tool_def(tool_def) for name, tool_def in tool_defs.items()}

    async def get_instructions(self, ctx: RunContext[AgentDepsT]) -> Instructions:
        if not self._mcp_toolset.include_instructions:
            return None
        if not self._in_durable_context() or self._get_instructions_operation is None:  # pragma: no cover
            return await self._mcp_toolset.get_instructions(ctx)
        # Always route through the durable unit: deciding based on locally-cached state (e.g.
        # instructions a warm in-process MCP server already holds) would make the durable
        # schedule depend on process warmth and diverge on replay/recovery (#5884).
        return await self._get_instructions_operation(ctx)

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if not self._in_durable_context():
            return await self._mcp_toolset.call_tool(name, tool_args, ctx, tool)
        config = self._resolve_tool_config(tool, name)
        if config is False:  # pragma: no cover — no engine's resolver currently permits inline MCP tools
            return await self._mcp_toolset.call_tool(name, tool_args, ctx, tool)
        return await self._call_tool_operation(name, tool_args, ctx, tool, config)
