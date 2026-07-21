from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias, cast

from pydantic import Discriminator, Tag
from typing_extensions import Self, assert_never

from pydantic_ai import AbstractToolset, FunctionToolset, ToolsetTool, WrapperToolset
from pydantic_ai._utils import is_str_dict
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.messages import InstructionPart, ToolReturn, ToolReturnContent
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

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
        if not self._in_durable_context():  # pragma: no cover
            return await self._mcp_toolset.call_tool(name, tool_args, ctx, tool)
        config = self._resolve_tool_config(tool, name)
        if config is False:  # pragma: no cover — no engine's resolver currently permits inline MCP tools
            return await self._mcp_toolset.call_tool(name, tool_args, ctx, tool)
        return await self._call_tool_operation(name, tool_args, ctx, tool, config)
