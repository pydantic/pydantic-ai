from __future__ import annotations

from collections.abc import AsyncIterable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from pydantic_ai._instructions import AgentInstructions
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import AgentStreamEvent, ModelResponse, ToolCallPart
from pydantic_ai.tools import (
    AgentBuiltinTool,
    AgentDepsT,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolDefinition,
)
from pydantic_ai.toolsets import AbstractToolset, AgentToolset

from .abstract import (
    AbstractCapability,
    AgentNode,
    NodeResult,
    RawOutput,
    RawToolArgs,
    ValidatedToolArgs,
    WrapModelRequestHandler,
    WrapNodeRunHandler,
    WrapOutputProcessHandler,
    WrapOutputValidateHandler,
    WrapRunHandler,
    WrapToolExecuteHandler,
    WrapToolValidateHandler,
)

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AgentModelSettings
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.output import OutputContext
    from pydantic_ai.run import AgentRunResult


@dataclass
class WrapperCapability(AbstractCapability[AgentDepsT]):
    """A capability that wraps another capability and delegates all methods.

    Analogous to [`WrapperToolset`][pydantic_ai.toolsets.WrapperToolset] for toolsets.
    Subclass and override specific methods to modify behavior while delegating the rest.
    """

    wrapped: AbstractCapability[AgentDepsT]

    def apply(self, visitor: Callable[[AbstractCapability[AgentDepsT]], None]) -> None:
        self.wrapped.apply(visitor)

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    @property
    def has_wrap_node_run(self) -> bool:
        return type(self).wrap_node_run is not WrapperCapability.wrap_node_run or self.wrapped.has_wrap_node_run

    @property
    def has_wrap_run_event_stream(self) -> bool:
        return (
            type(self).wrap_run_event_stream is not WrapperCapability.wrap_run_event_stream
            or self.wrapped.has_wrap_run_event_stream
        )

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        new_wrapped = await self.wrapped.for_run(ctx)
        if new_wrapped is self.wrapped:
            return self
        return replace(self, wrapped=new_wrapped)

    # --- Get methods ---

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        return self.wrapped.get_instructions()

    def get_model_settings(self) -> AgentModelSettings[AgentDepsT] | None:
        return self.wrapped.get_model_settings()

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        return self.wrapped.get_toolset()

    def get_builtin_tools(self) -> Sequence[AgentBuiltinTool[AgentDepsT]]:
        return self.wrapped.get_builtin_tools()

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        return self.wrapped.get_wrapper_toolset(toolset)

    async def prepare_tools(
        self,
        ctx: RunContext[AgentDepsT],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        return await self.wrapped.prepare_tools(ctx, tool_defs)

    async def prepare_output_tools(
        self,
        ctx: RunContext[AgentDepsT],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        return await self.wrapped.prepare_output_tools(ctx, tool_defs)

    # --- Run lifecycle hooks ---

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        await self.wrapped.before_run(ctx)

    async def after_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        return await self.wrapped.after_run(ctx, result=result)

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        return await self.wrapped.wrap_run(ctx, handler=handler)

    async def on_run_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        error: BaseException,
    ) -> AgentRunResult[Any]:
        return await self.wrapped.on_run_error(ctx, error=error)

    # --- Node run lifecycle hooks ---

    async def before_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
    ) -> AgentNode[AgentDepsT]:
        return await self.wrapped.before_node_run(ctx, node=node)

    async def after_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        result: NodeResult[AgentDepsT],
    ) -> NodeResult[AgentDepsT]:
        return await self.wrapped.after_node_run(ctx, node=node, result=result)

    async def wrap_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        handler: WrapNodeRunHandler[AgentDepsT],
    ) -> NodeResult[AgentDepsT]:
        return await self.wrapped.wrap_node_run(ctx, node=node, handler=handler)

    async def on_node_run_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        error: Exception,
    ) -> NodeResult[AgentDepsT]:
        return await self.wrapped.on_node_run_error(ctx, node=node, error=error)

    # --- Event stream hook ---

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        async for event in self.wrapped.wrap_run_event_stream(ctx, stream=stream):
            yield event

    # --- Model request lifecycle hooks ---

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        return await self.wrapped.before_model_request(ctx, request_context)

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        return await self.wrapped.after_model_request(ctx, request_context=request_context, response=response)

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        return await self.wrapped.wrap_model_request(ctx, request_context=request_context, handler=handler)

    async def on_model_request_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        error: Exception,
    ) -> ModelResponse:
        return await self.wrapped.on_model_request_error(ctx, request_context=request_context, error=error)

    # --- Tool validate lifecycle hooks ---

    async def before_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: RawToolArgs,
    ) -> RawToolArgs:
        return await self.wrapped.before_tool_validate(ctx, call=call, tool_def=tool_def, args=args)

    async def after_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
    ) -> ValidatedToolArgs:
        return await self.wrapped.after_tool_validate(ctx, call=call, tool_def=tool_def, args=args)

    async def wrap_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: RawToolArgs,
        handler: WrapToolValidateHandler,
    ) -> ValidatedToolArgs:
        return await self.wrapped.wrap_tool_validate(ctx, call=call, tool_def=tool_def, args=args, handler=handler)

    async def on_tool_validate_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: RawToolArgs,
        error: ValidationError | ModelRetry,
    ) -> ValidatedToolArgs:
        return await self.wrapped.on_tool_validate_error(ctx, call=call, tool_def=tool_def, args=args, error=error)

    # --- Tool execute lifecycle hooks ---

    async def before_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
    ) -> ValidatedToolArgs:
        return await self.wrapped.before_tool_execute(ctx, call=call, tool_def=tool_def, args=args)

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: Any,
    ) -> Any:
        return await self.wrapped.after_tool_execute(ctx, call=call, tool_def=tool_def, args=args, result=result)

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        return await self.wrapped.wrap_tool_execute(ctx, call=call, tool_def=tool_def, args=args, handler=handler)

    async def on_tool_execute_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        error: Exception,
    ) -> Any:
        return await self.wrapped.on_tool_execute_error(ctx, call=call, tool_def=tool_def, args=args, error=error)

    # --- Output validate lifecycle hooks ---

    async def before_output_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: RawOutput,
    ) -> RawOutput:
        return await self.wrapped.before_output_validate(ctx, output_context=output_context, output=output)

    async def after_output_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
    ) -> Any:
        return await self.wrapped.after_output_validate(ctx, output_context=output_context, output=output)

    async def wrap_output_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: RawOutput,
        handler: WrapOutputValidateHandler,
    ) -> Any:
        return await self.wrapped.wrap_output_validate(
            ctx, output_context=output_context, output=output, handler=handler
        )

    async def on_output_validate_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: RawOutput,
        error: ValidationError | ModelRetry,
    ) -> Any:
        return await self.wrapped.on_output_validate_error(
            ctx, output_context=output_context, output=output, error=error
        )

    # --- Output process lifecycle hooks ---

    async def before_output_process(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
    ) -> Any:
        return await self.wrapped.before_output_process(ctx, output_context=output_context, output=output)

    async def after_output_process(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
    ) -> Any:
        return await self.wrapped.after_output_process(ctx, output_context=output_context, output=output)

    async def wrap_output_process(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
        handler: WrapOutputProcessHandler,
    ) -> Any:
        return await self.wrapped.wrap_output_process(
            ctx, output_context=output_context, output=output, handler=handler
        )

    async def on_output_process_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
        error: Exception,
    ) -> Any:
        return await self.wrapped.on_output_process_error(
            ctx, output_context=output_context, output=output, error=error
        )

    async def handle_deferred_tool_calls(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        requests: DeferredToolRequests,
    ) -> DeferredToolResults | None:
        return await self.wrapped.handle_deferred_tool_calls(ctx, requests=requests)
