from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from pydantic_ai import _system_prompt
from pydantic_ai._instructions import AgentInstructions, normalize_instructions
from pydantic_ai._utils import gather
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import AgentStreamEvent, ModelResponse, ToolCallPart
from pydantic_ai.settings import ModelSettings, merge_model_settings
from pydantic_ai.tools import (
    AgentBuiltinTool,
    AgentDepsT,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolDefinition,
)
from pydantic_ai.toolsets import AbstractToolset, AgentToolset, CombinedToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset

from ._ordering import collect_leaves, sort_capabilities
from .abstract import AbstractCapability, RawOutput, WrapOutputProcessHandler, WrapOutputValidateHandler

if TYPE_CHECKING:
    from pydantic_ai import _agent_graph
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.output import OutputContext
    from pydantic_ai.result import FinalResult
    from pydantic_ai.run import AgentRunResult
    from pydantic_graph import End


@dataclass
class CombinedCapability(AbstractCapability[AgentDepsT]):
    """A capability that combines multiple capabilities."""

    capabilities: Sequence[AbstractCapability[AgentDepsT]]

    def __post_init__(self) -> None:
        if any(leaf.get_ordering() is not None for leaf in collect_leaves(self)):
            self.capabilities = sort_capabilities(list(self.capabilities))

    def apply(self, visitor: Callable[[AbstractCapability[AgentDepsT]], None]) -> None:
        for cap in self.capabilities:
            cap.apply(visitor)

    @property
    def has_wrap_node_run(self) -> bool:
        return any(c.has_wrap_node_run for c in self.capabilities)

    @property
    def has_wrap_run_event_stream(self) -> bool:
        return any(c.has_wrap_run_event_stream for c in self.capabilities)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        new_caps = await gather(*(c.for_run(ctx) for c in self.capabilities))
        if all(new is old for new, old in zip(new_caps, self.capabilities)):
            return self
        return replace(self, capabilities=list(new_caps))

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        instructions: list[str | _system_prompt.SystemPromptFunc[AgentDepsT]] = []
        for capability in self.capabilities:
            instructions.extend(normalize_instructions(capability.get_instructions()))
        return instructions or None

    def get_model_settings(self) -> ModelSettings | Callable[[RunContext[AgentDepsT]], ModelSettings] | None:
        # Collect settings in order, preserving each capability's position in the merge chain.
        # Each entry is either a static dict or a dynamic callable.
        settings_chain: list[ModelSettings | Callable[[RunContext[AgentDepsT]], ModelSettings]] = []
        for capability in self.capabilities:
            cap_settings = capability.get_model_settings()
            if cap_settings is not None:
                settings_chain.append(cap_settings)
        if not settings_chain:
            return None
        if all(not callable(s) for s in settings_chain):
            # All static — merge eagerly
            merged: ModelSettings | None = None
            for s in settings_chain:
                merged = merge_model_settings(merged, s)  # type: ignore[arg-type]
            return merged

        def resolve(ctx: RunContext[AgentDepsT]) -> ModelSettings:
            merged: ModelSettings | None = None
            for entry in settings_chain:
                # Mutate ctx.model_settings so each dynamic entry sees the
                # accumulated settings from all prior layers.
                ctx.model_settings = merge_model_settings(ctx.model_settings, merged)
                resolved = entry(ctx) if callable(entry) else entry
                merged = merge_model_settings(merged, resolved)
            # Update ctx.model_settings to include the final entry's contribution
            ctx.model_settings = merge_model_settings(ctx.model_settings, merged)
            return merged if merged is not None else ModelSettings()

        return resolve

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        toolsets: list[AbstractToolset[AgentDepsT]] = []
        for capability in self.capabilities:
            toolset = capability.get_toolset()
            if toolset is None:
                pass
            elif isinstance(toolset, AbstractToolset):
                # Pyright can't narrow Callable type aliases out of unions after isinstance check
                toolsets.append(toolset)  # pyright: ignore[reportUnknownArgumentType]
            else:
                toolsets.append(DynamicToolset[AgentDepsT](toolset_func=toolset))
        return CombinedToolset(toolsets) if toolsets else None

    def get_builtin_tools(self) -> Sequence[AgentBuiltinTool[AgentDepsT]]:
        builtin_tools: list[AgentBuiltinTool[AgentDepsT]] = []
        for capability in self.capabilities:
            builtin_tools.extend(capability.get_builtin_tools() or [])
        return builtin_tools

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        wrapped = toolset
        any_wrapped = False
        for capability in reversed(self.capabilities):
            result = capability.get_wrapper_toolset(wrapped)
            if result is not None:
                wrapped = result
                any_wrapped = True
        return wrapped if any_wrapped else None

    # --- Tool preparation hooks ---

    async def prepare_tools(
        self,
        ctx: RunContext[AgentDepsT],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        for capability in self.capabilities:
            tool_defs = await capability.prepare_tools(ctx, tool_defs)
        return tool_defs

    async def prepare_output_tools(
        self,
        ctx: RunContext[AgentDepsT],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        for capability in self.capabilities:
            tool_defs = await capability.prepare_output_tools(ctx, tool_defs)
        return tool_defs

    # --- Run lifecycle hooks ---

    async def before_run(
        self,
        ctx: RunContext[AgentDepsT],
    ) -> None:
        for capability in self.capabilities:
            await capability.before_run(ctx)

    async def after_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        for capability in reversed(self.capabilities):
            result = await capability.after_run(ctx, result=result)
        return result

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: Callable[[], Awaitable[AgentRunResult[Any]]],
    ) -> AgentRunResult[Any]:
        chain = handler
        for cap in reversed(self.capabilities):
            chain = _make_run_wrap(cap, ctx, chain)
        return await chain()

    async def on_run_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        error: BaseException,
    ) -> AgentRunResult[Any]:
        for capability in reversed(self.capabilities):
            try:
                return await capability.on_run_error(ctx, error=error)
            except BaseException as new_error:
                error = new_error
        raise error

    # --- Node run lifecycle hooks ---

    async def before_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: _agent_graph.AgentNode[AgentDepsT, Any],
    ) -> _agent_graph.AgentNode[AgentDepsT, Any]:
        for capability in self.capabilities:
            node = await capability.before_node_run(ctx, node=node)
        return node

    async def after_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: _agent_graph.AgentNode[AgentDepsT, Any],
        result: _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]],
    ) -> _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]:
        for capability in reversed(self.capabilities):
            result = await capability.after_node_run(ctx, node=node, result=result)
        return result

    async def wrap_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: _agent_graph.AgentNode[AgentDepsT, Any],
        handler: Callable[
            [_agent_graph.AgentNode[AgentDepsT, Any]],
            Awaitable[_agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]],
        ],
    ) -> _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]:
        chain = handler
        for cap in reversed(self.capabilities):
            chain = _make_node_run_wrap(cap, ctx, chain)
        return await chain(node)

    async def on_node_run_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: _agent_graph.AgentNode[AgentDepsT, Any],
        error: Exception,
    ) -> _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]:
        for capability in reversed(self.capabilities):
            try:
                return await capability.on_node_run_error(ctx, node=node, error=error)
            except Exception as new_error:
                error = new_error
        raise error

    # --- Event stream hook ---

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        for cap in reversed(self.capabilities):
            stream = cap.wrap_run_event_stream(ctx, stream=stream)
        async for event in stream:
            yield event

    # --- Model request lifecycle hooks ---

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        for capability in self.capabilities:
            request_context = await capability.before_model_request(ctx, request_context)
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        for capability in reversed(self.capabilities):
            response = await capability.after_model_request(ctx, request_context=request_context, response=response)
        return response

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: Callable[[ModelRequestContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        chain = handler
        for cap in reversed(self.capabilities):
            chain = _make_model_request_wrap(cap, ctx, chain)
        return await chain(request_context)

    async def on_model_request_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        error: Exception,
    ) -> ModelResponse:
        for capability in reversed(self.capabilities):
            try:
                return await capability.on_model_request_error(ctx, request_context=request_context, error=error)
            except Exception as new_error:
                error = new_error
        raise error

    # --- Tool validate lifecycle hooks ---

    async def before_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: str | dict[str, Any],
    ) -> str | dict[str, Any]:
        for capability in self.capabilities:
            args = await capability.before_tool_validate(ctx, call=call, tool_def=tool_def, args=args)
        return args

    async def after_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        for capability in reversed(self.capabilities):
            args = await capability.after_tool_validate(ctx, call=call, tool_def=tool_def, args=args)
        return args

    async def wrap_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: str | dict[str, Any],
        handler: Callable[[str | dict[str, Any]], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        chain = handler
        for cap in reversed(self.capabilities):
            chain = _make_tool_validate_wrap(cap, ctx, call, tool_def, chain)
        return await chain(args)

    async def on_tool_validate_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: str | dict[str, Any],
        error: ValidationError | ModelRetry,
    ) -> dict[str, Any]:
        for capability in reversed(self.capabilities):
            try:
                return await capability.on_tool_validate_error(
                    ctx, call=call, tool_def=tool_def, args=args, error=error
                )
            except (ValidationError, ModelRetry) as new_error:
                error = new_error
            except (
                Exception
            ):  # pragma: no cover — defensive; on_tool_validate_error shouldn't raise non-validation errors
                raise
        raise error

    # --- Tool execute lifecycle hooks ---

    async def before_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        for capability in self.capabilities:
            args = await capability.before_tool_execute(ctx, call=call, tool_def=tool_def, args=args)
        return args

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        for capability in reversed(self.capabilities):
            result = await capability.after_tool_execute(ctx, call=call, tool_def=tool_def, args=args, result=result)
        return result

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> Any:
        chain = handler
        for cap in reversed(self.capabilities):
            chain = _make_tool_execute_wrap(cap, ctx, call, tool_def, chain)
        return await chain(args)

    async def on_tool_execute_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        error: Exception,
    ) -> Any:
        for capability in reversed(self.capabilities):
            try:
                return await capability.on_tool_execute_error(ctx, call=call, tool_def=tool_def, args=args, error=error)
            except Exception as new_error:
                error = new_error
        raise error

    # --- Output validate lifecycle hooks ---

    async def before_output_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: RawOutput,
    ) -> RawOutput:
        for capability in self.capabilities:
            output = await capability.before_output_validate(ctx, output_context=output_context, output=output)
        return output

    async def after_output_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
    ) -> Any:
        for capability in reversed(self.capabilities):
            output = await capability.after_output_validate(ctx, output_context=output_context, output=output)
        return output

    async def wrap_output_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: RawOutput,
        handler: WrapOutputValidateHandler,
    ) -> Any:
        chain = handler
        for cap in reversed(self.capabilities):
            chain = _make_output_validate_wrap(cap, ctx, output_context, chain)
        return await chain(output)

    async def on_output_validate_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: RawOutput,
        error: ValidationError | ModelRetry,
    ) -> Any:
        for capability in reversed(self.capabilities):
            try:
                return await capability.on_output_validate_error(
                    ctx, output_context=output_context, output=output, error=error
                )
            except (ValidationError, ModelRetry) as new_error:
                error = new_error
            except Exception:  # pragma: no cover — defensive
                raise
        raise error

    # --- Output process lifecycle hooks ---

    async def before_output_process(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
    ) -> Any:
        for capability in self.capabilities:
            output = await capability.before_output_process(ctx, output_context=output_context, output=output)
        return output

    async def after_output_process(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
    ) -> Any:
        for capability in reversed(self.capabilities):
            output = await capability.after_output_process(ctx, output_context=output_context, output=output)
        return output

    async def wrap_output_process(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
        handler: WrapOutputProcessHandler,
    ) -> Any:
        chain = handler
        for cap in reversed(self.capabilities):
            chain = _make_output_process_wrap(cap, ctx, output_context, chain)
        return await chain(output)

    async def on_output_process_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
        error: Exception,
    ) -> Any:
        for capability in reversed(self.capabilities):
            try:
                return await capability.on_output_process_error(
                    ctx, output_context=output_context, output=output, error=error
                )
            except Exception as new_error:
                error = new_error
        raise error

    async def handle_deferred_tool_calls(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        requests: DeferredToolRequests,
    ) -> DeferredToolResults | None:
        accumulated = DeferredToolResults()
        remaining = requests
        any_handled = False
        for capability in self.capabilities:
            result = await capability.handle_deferred_tool_calls(ctx, requests=remaining)
            if result is None or not (result.approvals or result.calls):
                continue
            any_handled = True
            accumulated.update(result)
            remaining_or_none = remaining.remaining(result)
            if remaining_or_none is None:
                break
            remaining = remaining_or_none
        return accumulated if any_handled else None


# --- Composition helpers ---
# These create closures that bind the current capability and inner handler,
# building a middleware chain from outermost (first cap) to innermost (last cap).


def _make_run_wrap(
    cap: AbstractCapability[AgentDepsT],
    ctx: RunContext[AgentDepsT],
    inner: Callable[[], Awaitable[AgentRunResult[Any]]],
) -> Callable[[], Awaitable[AgentRunResult[Any]]]:
    async def wrapped() -> AgentRunResult[Any]:
        return await cap.wrap_run(ctx, handler=inner)

    return wrapped


def _make_model_request_wrap(
    cap: AbstractCapability[AgentDepsT],
    ctx: RunContext[AgentDepsT],
    inner: Callable[[ModelRequestContext], Awaitable[ModelResponse]],
) -> Callable[[ModelRequestContext], Awaitable[ModelResponse]]:
    async def wrapped(request_context: ModelRequestContext) -> ModelResponse:
        return await cap.wrap_model_request(
            ctx,
            request_context=request_context,
            handler=inner,
        )

    return wrapped


def _make_tool_validate_wrap(
    cap: AbstractCapability[AgentDepsT],
    ctx: RunContext[AgentDepsT],
    call: ToolCallPart,
    tool_def: ToolDefinition,
    inner: Callable[[str | dict[str, Any]], Awaitable[dict[str, Any]]],
) -> Callable[[str | dict[str, Any]], Awaitable[dict[str, Any]]]:
    async def wrapped(args: str | dict[str, Any]) -> dict[str, Any]:
        return await cap.wrap_tool_validate(ctx, call=call, tool_def=tool_def, args=args, handler=inner)

    return wrapped


def _make_node_run_wrap(
    cap: AbstractCapability[AgentDepsT],
    ctx: RunContext[AgentDepsT],
    inner: Callable[
        [_agent_graph.AgentNode[AgentDepsT, Any]],
        Awaitable[_agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]],
    ],
) -> Callable[
    [_agent_graph.AgentNode[AgentDepsT, Any]],
    Awaitable[_agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]],
]:
    async def wrapped(
        node: _agent_graph.AgentNode[AgentDepsT, Any],
    ) -> _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]:
        return await cap.wrap_node_run(ctx, node=node, handler=inner)

    return wrapped


def _make_tool_execute_wrap(
    cap: AbstractCapability[AgentDepsT],
    ctx: RunContext[AgentDepsT],
    call: ToolCallPart,
    tool_def: ToolDefinition,
    inner: Callable[[dict[str, Any]], Awaitable[Any]],
) -> Callable[[dict[str, Any]], Awaitable[Any]]:
    async def wrapped(args: dict[str, Any]) -> Any:
        return await cap.wrap_tool_execute(ctx, call=call, tool_def=tool_def, args=args, handler=inner)

    return wrapped


def _make_output_validate_wrap(
    cap: AbstractCapability[AgentDepsT],
    ctx: RunContext[AgentDepsT],
    output_context: OutputContext,
    inner: Callable[[RawOutput], Awaitable[Any]],
) -> Callable[[RawOutput], Awaitable[Any]]:
    async def wrapped(output: RawOutput) -> Any:
        return await cap.wrap_output_validate(ctx, output_context=output_context, output=output, handler=inner)

    return wrapped


def _make_output_process_wrap(
    cap: AbstractCapability[AgentDepsT],
    ctx: RunContext[AgentDepsT],
    output_context: OutputContext,
    inner: Callable[[Any], Awaitable[Any]],
) -> Callable[[Any], Awaitable[Any]]:
    async def wrapped(output: Any) -> Any:
        return await cap.wrap_output_process(ctx, output_context=output_context, output=output, handler=inner)

    return wrapped
