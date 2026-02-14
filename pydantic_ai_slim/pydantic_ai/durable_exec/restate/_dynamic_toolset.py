from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

from pydantic import ValidationError
from typing_extensions import Self

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai.toolsets.external import TOOL_SCHEMA_VALIDATOR
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.toolsets.wrapper import WrapperToolset

from ._restate_types import Context, RunOptions, TerminalError
from ._serde import PydanticTypeAdapter
from ._toolset import CONTEXT_RUN_SERDE, RestateContextRunResult, run_tool_call_step


@dataclass
class _ToolInfo:
    """Serializable tool information returned from a durable get-tools step."""

    tool_def: ToolDefinition
    max_retries: int
    origin: Literal['function', 'io']


@dataclass(kw_only=True)
class _RestateDynamicToolsetTool(ToolsetTool[AgentDepsT]):
    origin: Literal['function', 'io']


@dataclass
class RestateDynamicGetToolsContextRunResult:
    """A simple wrapper for tool results to be used with Restate's `ctx.run_typed()`."""

    output: dict[str, _ToolInfo]


DYNAMIC_GET_TOOLS_SERDE = PydanticTypeAdapter(RestateDynamicGetToolsContextRunResult)


class RestateDynamicToolset(WrapperToolset[AgentDepsT]):
    """A durable wrapper for [`DynamicToolset`][pydantic_ai.toolsets._dynamic.DynamicToolset].

    Restate durability requirement: the dynamic function may do I/O (e.g. return an MCP toolset),
    so both tool discovery and tool calls must happen inside `ctx.run_typed(...)`.
    """

    def __init__(
        self, wrapped: DynamicToolset[AgentDepsT], context: Context, *, disable_auto_wrapping_tools: bool = False
    ):
        super().__init__(wrapped)
        self._context = context
        self._call_options = RunOptions[RestateContextRunResult](serde=CONTEXT_RUN_SERDE)
        self._disable_auto_wrapping_tools = disable_auto_wrapping_tools

    @property
    def _dynamic_toolset(self) -> DynamicToolset[AgentDepsT]:
        return cast(DynamicToolset[AgentDepsT], self.wrapped)

    @property
    def id(self) -> str | None:  # pragma: no cover
        return self._dynamic_toolset.id

    async def __aenter__(self) -> Self:
        """No-op: underlying toolset I/O must occur inside `ctx.run_typed()` for durability."""
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        """No-op: underlying toolset I/O must occur inside `ctx.run_typed()` for durability."""
        return None

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        # Restate-wrapped toolsets cannot be swapped out after wrapping.
        return self

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        async def get_tools_in_context() -> RestateDynamicGetToolsContextRunResult:
            async with self._dynamic_toolset:
                tools = await self._dynamic_toolset.get_tools(ctx)
                return RestateDynamicGetToolsContextRunResult(
                    output={
                        name: _ToolInfo(
                            tool_def=tool.tool_def,
                            max_retries=tool.max_retries,
                            origin=self._toolset_origin(tool.toolset),
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
        async def call_tool_action() -> Any:
            # Re-instantiate the dynamic toolset inside the durable step so any underlying I/O happens
            # within `ctx.run_typed(...)`, and ensure resources are cleaned up afterwards.
            async with self._dynamic_toolset:
                tools = await self._dynamic_toolset.get_tools(ctx)
                real_tool = tools.get(name)
                if real_tool is None:
                    raise TerminalError(
                        f'Tool {name!r} not found in dynamic toolset {self._dynamic_toolset.label}. '
                        'The dynamic toolset function may have returned a different toolset than expected.'
                    )

                try:
                    args_dict = real_tool.args_validator.validate_python(tool_args, context=ctx.validation_context)
                except ValidationError as e:
                    # Convert validation errors into ModelRetry so the agent can ask the model to retry.
                    raise ModelRetry(str(e)) from e

                return await self._dynamic_toolset.call_tool(name, args_dict, ctx, real_tool)

        # If automatic tool wrapping is disabled, dynamic toolset *function tools* should be executed outside
        # `ctx.run_typed(...)` so tool implementations can use the Restate context directly.
        # In this mode, dynamic toolset resolution/validation also happens outside the durable step.
        if self._disable_auto_wrapping_tools and self._is_function_tool(tool):
            return await call_tool_action()

        return await run_tool_call_step(
            self._context, f'Calling dynamic tool {name}', call_tool_action, self._call_options
        )

    def _tool_for_tool_info(self, tool_info: _ToolInfo) -> ToolsetTool[AgentDepsT]:
        """Create a tool from `_ToolInfo` for use outside durable steps.

        Use a permissive schema validator; actual args validation happens inside `call_tool_in_context`.
        """
        return _RestateDynamicToolsetTool(
            toolset=self,
            tool_def=tool_info.tool_def,
            max_retries=tool_info.max_retries,
            args_validator=TOOL_SCHEMA_VALIDATOR,
            origin=tool_info.origin,
        )

    def _is_function_tool(self, tool: ToolsetTool[AgentDepsT]) -> bool:
        return isinstance(tool, _RestateDynamicToolsetTool) and tool.origin == 'function'

    def _toolset_origin(self, toolset: AbstractToolset[Any]) -> Literal['function', 'io']:
        return 'function' if self._toolset_is_functionlike(toolset) else 'io'

    def _toolset_is_functionlike(self, toolset: AbstractToolset[Any]) -> bool:
        """Return True if the leaf toolset is a FunctionToolset (possibly wrapped)."""
        unwrapped: AbstractToolset[Any] = toolset
        while isinstance(unwrapped, WrapperToolset):
            unwrapped = unwrapped.wrapped
        return isinstance(unwrapped, FunctionToolset)
