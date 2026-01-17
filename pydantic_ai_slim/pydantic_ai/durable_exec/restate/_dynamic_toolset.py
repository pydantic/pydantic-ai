from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from pydantic import ValidationError
from pydantic.errors import PydanticUserError
from typing_extensions import Self

from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import DynamicToolset
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai.toolsets.external import TOOL_SCHEMA_VALIDATOR
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.toolsets.wrapper import WrapperToolset

from ._restate_types import Context, RunOptions, TerminalError
from ._serde import PydanticTypeAdapter
from ._toolset import CONTEXT_RUN_SERDE, RestateContextRunResult

_RESTATE_DYNAMIC_ORIGIN_KEY = '__pydantic_ai_restate_dynamic_origin'


@dataclass
class _ToolInfo:
    """Serializable tool information returned from a durable get-tools step."""

    tool_def: ToolDefinition
    max_retries: int


@dataclass
class RestateDynamicGetToolsContextRunResult:
    """A simple wrapper for tool results to be used with Restate's `ctx.run_typed()`."""

    output: dict[str, _ToolInfo]


DYNAMIC_GET_TOOLS_SERDE = PydanticTypeAdapter(RestateDynamicGetToolsContextRunResult)


class RestateDynamicToolset(WrapperToolset[AgentDepsT]):
    """A durable wrapper for [`DynamicToolset`][pydantic_ai.toolsets.DynamicToolset].

    Restate durability requirement: the dynamic function may do I/O (e.g. return an MCP toolset),
    so both tool discovery and tool calls must happen inside `ctx.run_typed(...)`.
    """

    def __init__(
        self, wrapped: DynamicToolset[AgentDepsT], context: Context, *, disable_auto_wrapping_tools: bool = False
    ):
        super().__init__(wrapped)
        self._wrapped = wrapped
        self._context = context
        self._call_options = RunOptions[RestateContextRunResult](serde=CONTEXT_RUN_SERDE)
        self._disable_auto_wrapping_tools = disable_auto_wrapping_tools

    @property
    def id(self) -> str | None:  # pragma: no cover
        return self.wrapped.id

    async def __aenter__(self) -> Self:
        """No-op: underlying toolset I/O must occur inside `ctx.run_typed()` for durability."""
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        """No-op: underlying toolset I/O must occur inside `ctx.run_typed()` for durability."""
        return None

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        return visitor(self)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        async def get_tools_in_context() -> RestateDynamicGetToolsContextRunResult:
            async with self._wrapped:
                tools = await self._wrapped.get_tools(ctx)
                return RestateDynamicGetToolsContextRunResult(
                    output={
                        name: _ToolInfo(
                            tool_def=self._with_dynamic_origin(tool.tool_def, tool.toolset),
                            max_retries=tool.max_retries,
                        )
                        for name, tool in tools.items()
                    }
                )

        options = RunOptions[RestateDynamicGetToolsContextRunResult](serde=DYNAMIC_GET_TOOLS_SERDE)
        tool_infos = await self._context.run_typed('get dynamic tools', get_tools_in_context, options)
        return {name: self._tool_for_tool_info(tool_info) for name, tool_info in tool_infos.output.items()}

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        # If automatic tool wrapping is disabled, dynamic toolset *function tools* should be executed outside
        # ctx.run_typed so tool implementations can use the Restate context directly.
        if self._disable_auto_wrapping_tools and self._is_function_tool(tool):
            async with self._wrapped:
                tools = await self._wrapped.get_tools(ctx)
                real_tool = tools.get(name)
                if real_tool is None:  # pragma: no cover
                    raise UserError(
                        f'Tool {name!r} not found in dynamic toolset {self._wrapped.label}. '
                        'The dynamic toolset function may have returned a different toolset than expected.'
                    )

                try:
                    args_dict = real_tool.args_validator.validate_python(tool_args, context=ctx.validation_context)
                except ValidationError as e:
                    raise ModelRetry(str(e)) from e

                return await self._wrapped.call_tool(name, args_dict, ctx, real_tool)

        async def call_tool_in_context() -> RestateContextRunResult:
            # Re-instantiate the dynamic toolset inside the durable step so any underlying I/O happens
            # within `ctx.run_typed(...)`, and ensure resources are cleaned up afterwards.
            async with self._wrapped:
                tools = await self._wrapped.get_tools(ctx)
                real_tool = tools.get(name)
                if real_tool is None:
                    raise TerminalError(
                        f'Tool {name!r} not found in dynamic toolset {self._wrapped.label}. '
                        'The dynamic toolset function may have returned a different toolset than expected.'
                    )

                try:
                    args_dict = real_tool.args_validator.validate_python(tool_args, context=ctx.validation_context)
                except ValidationError as e:
                    # Convert validation errors into ModelRetry so the agent can ask the model to retry.
                    return RestateContextRunResult(kind='model_retry', output=None, error=str(e))

                try:
                    output = await self._wrapped.call_tool(name, args_dict, ctx, real_tool)
                    return RestateContextRunResult(kind='output', output=output, error=None)
                except ModelRetry as e:
                    return RestateContextRunResult(kind='model_retry', output=None, error=e.message)
                except CallDeferred as e:
                    return RestateContextRunResult(kind='call_deferred', output=None, metadata=e.metadata)
                except ApprovalRequired as e:
                    return RestateContextRunResult(kind='approval_required', output=None, metadata=e.metadata)
                except (UserError, PydanticUserError) as e:
                    raise TerminalError(str(e)) from e

        res = await self._context.run_typed(f'Calling dynamic tool {name}', call_tool_in_context, self._call_options)

        if res.kind == 'call_deferred':
            raise CallDeferred(metadata=res.metadata)
        elif res.kind == 'approval_required':
            raise ApprovalRequired(metadata=res.metadata)
        elif res.kind == 'model_retry':
            assert res.error is not None
            raise ModelRetry(res.error)
        else:
            assert res.kind == 'output'
            return res.output

    def _tool_for_tool_info(self, tool_info: _ToolInfo) -> ToolsetTool[AgentDepsT]:
        """Create a `ToolsetTool` from a `_ToolInfo` for use outside durable steps.

        Use a permissive schema validator; actual args validation happens inside `call_tool_in_context`.
        """
        return ToolsetTool(
            toolset=self,
            tool_def=tool_info.tool_def,
            max_retries=tool_info.max_retries,
            args_validator=TOOL_SCHEMA_VALIDATOR,
        )

    def _with_dynamic_origin(self, tool_def: ToolDefinition, source_toolset: AbstractToolset[Any]) -> ToolDefinition:
        """Attach an internal marker describing where this tool came from.

        This is used to decide whether tool execution should be wrapped when
        `disable_auto_wrapping_tools=True`.
        """
        origin = 'function' if self._toolset_is_functionlike(source_toolset) else 'io'
        metadata = dict(tool_def.metadata or {})
        metadata[_RESTATE_DYNAMIC_ORIGIN_KEY] = origin
        return replace(tool_def, metadata=metadata)

    def _is_function_tool(self, tool: ToolsetTool[AgentDepsT]) -> bool:
        metadata = tool.tool_def.metadata or {}
        return metadata.get(_RESTATE_DYNAMIC_ORIGIN_KEY) == 'function'

    def _toolset_is_functionlike(self, toolset: AbstractToolset[Any]) -> bool:
        """Return True if the leaf toolset is a FunctionToolset (possibly wrapped)."""
        unwrapped: AbstractToolset[Any] = toolset
        while isinstance(unwrapped, WrapperToolset):
            unwrapped = unwrapped.wrapped
        return isinstance(unwrapped, FunctionToolset)
