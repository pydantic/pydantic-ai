from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from pydantic import ValidationError

from pydantic_ai._instructions import AgentInstructions, normalize_instructions
from pydantic_ai._utils import gather, replace_no_init
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import AgentStreamEvent, ModelResponse, ToolCallPart
from pydantic_ai.settings import ModelSettings, merge_model_settings
from pydantic_ai.tools import (
    AgentDepsT,
    AgentNativeTool,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    SystemPromptFunc,
    ToolDefinition,
)
from pydantic_ai.toolsets import AbstractToolset, AgentToolset, CombinedToolset
from pydantic_ai.toolsets._capability_owned import CapabilityOwnedToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset

from ._lifecycle import SKIP_LIFECYCLE_ENTRY, LifecycleStack, SkipLifecycleEntry
from ._ordering import collect_leaves, is_innermost, sort_capabilities
from .abstract import (
    AbstractCapability,
    AgentModel,
    RawOutput,
    WrapOutputProcessHandler,
    WrapOutputValidateHandler,
)

if TYPE_CHECKING:
    from pydantic_ai import _agent_graph
    from pydantic_ai.agent.abstract import AbstractAgent
    from pydantic_ai.models import KnownModelName, Model, ModelRequestContext, ModelResolutionContext
    from pydantic_ai.output import OutputContext
    from pydantic_ai.result import FinalResult
    from pydantic_ai.run import AgentRunResult
    from pydantic_graph import End

_ValueT = TypeVar('_ValueT')
_ResultT = TypeVar('_ResultT')
_ErrorT = TypeVar('_ErrorT', bound=BaseException)
_DepsT = TypeVar('_DepsT')
_Params = ParamSpec('_Params')


@dataclass
class CombinedCapability(AbstractCapability[AgentDepsT]):
    """A capability that combines multiple capabilities.

    When any child returns a fresh instance from
    [`for_agent`][pydantic_ai.capabilities.AbstractCapability.for_agent] or
    [`for_run`][pydantic_ai.capabilities.AbstractCapability.for_run], the container is rebound
    as a shallow copy holding the new children: subclass state is carried over verbatim and
    `__init__`/`__post_init__` are not re-run. Compute values derived from `capabilities` on
    access (e.g. via a property) rather than caching them at construction, so they can't go
    stale across a rebind.
    """

    capabilities: Sequence[AbstractCapability[AgentDepsT]]

    def __post_init__(self) -> None:
        self.__normalize_capabilities()

    # Name-mangled deliberately: this upholds a base-class invariant on rebinds, so a
    # subclass attribute of the same name must not be able to override it.
    def __normalize_capabilities(self) -> None:
        # Splat any nested `CombinedCapability` so leaves participate as siblings in the
        # outer ordering pass. Without this, a nested `CombinedCapability` whose leaves
        # span both `outermost` and `innermost` tiers would force `_effective_ordering`
        # to merge them into a single position and raise `Conflicting positions`.
        flat: list[AbstractCapability[AgentDepsT]] = []
        for cap in self.capabilities:
            if isinstance(cap, CombinedCapability):
                flat.extend(cap.capabilities)
            else:
                flat.append(cap)
        self.capabilities = flat
        if any(leaf.get_ordering() is not None for leaf in collect_leaves(self)):
            self.capabilities = sort_capabilities(list(self.capabilities))

    def apply(self, visitor: Callable[[AbstractCapability[AgentDepsT]], None]) -> None:
        for cap in self.capabilities:
            cap.apply(visitor)

    @property
    def _has_wrap_node_run(self) -> bool:
        return any(c._has_wrap_node_run for c in self.capabilities)

    @property
    def has_wrap_run_event_stream(self) -> bool:
        return any(c.has_wrap_run_event_stream for c in self.capabilities)

    def for_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> CombinedCapability[AgentDepsT]:
        new_caps = [capability.for_agent(agent) for capability in self.capabilities]
        if all(new is old for new, old in zip(new_caps, self.capabilities)):
            return self
        new_self = replace_no_init(self, capabilities=new_caps)
        new_self.__normalize_capabilities()
        return new_self

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        new_caps = await gather(*(c.for_run(ctx) for c in self.capabilities))
        if all(new is old for new, old in zip(new_caps, self.capabilities)):
            return self
        new_self = replace_no_init(self, capabilities=list(new_caps))
        new_self.__normalize_capabilities()
        return new_self

    def _validate_runtime_capabilities(
        self, ctx: RunContext[AgentDepsT], capabilities: Sequence[AbstractCapability[AgentDepsT]]
    ) -> None:
        for capability in self.capabilities:
            capability._validate_runtime_capabilities(ctx, capabilities)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        instructions: list[str | SystemPromptFunc[AgentDepsT]] = []
        for capability in self.capabilities:
            if capability.defer_loading is True:
                continue
            instructions.extend(normalize_instructions(capability.get_instructions()))

        return instructions or None

    def get_model_settings(self) -> ModelSettings | Callable[[RunContext[AgentDepsT]], ModelSettings] | None:
        # Collect settings in order, preserving each capability's position in the merge chain.
        # Each entry is either a static dict or a dynamic callable.
        settings_chain: list[ModelSettings | Callable[[RunContext[AgentDepsT]], ModelSettings]] = []
        for capability in self.capabilities:
            cap_settings = capability.get_model_settings()

            if cap_settings is None:
                continue

            if capability.defer_loading is True:
                # Request-only settings can be lazy without changing prompt/tool schemas.
                # Keep them in place so loaded capabilities preserve merge order.
                def deferred_settings(
                    ctx: RunContext[AgentDepsT],
                    *,
                    capability: AbstractCapability[AgentDepsT] = capability,
                    cap_settings: ModelSettings | Callable[[RunContext[AgentDepsT]], ModelSettings] = cap_settings,
                ) -> ModelSettings:
                    cap_ctx = _ctx_for_available_cap(capability, ctx)
                    if cap_ctx is None:
                        return ModelSettings()
                    if callable(cap_settings):
                        return cap_settings(cap_ctx)
                    return cap_settings

                settings_chain.append(deferred_settings)
            else:
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

    def get_model(self) -> AgentModel[AgentDepsT] | None:
        model: AgentModel[AgentDepsT] | None = None
        for capability in self.capabilities:
            if capability.defer_loading is not True and (capability_model := capability.get_model()) is not None:
                model = capability_model
        return model

    @property
    def has_resolve_model_id(self) -> bool:
        return any(
            capability.defer_loading is not True and capability.has_resolve_model_id for capability in self.capabilities
        )

    async def resolve_model_id(
        self,
        ctx: ModelResolutionContext[AgentDepsT],
        *,
        model_id: KnownModelName | str,
    ) -> Model | None:
        for capability in self.capabilities:
            if capability.defer_loading is True:
                continue
            if (model := await capability.resolve_model_id(ctx, model_id=model_id)) is not None:
                return model
        return None

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        toolsets: list[AbstractToolset[AgentDepsT]] = []
        for capability in self.capabilities:
            toolset = capability.get_toolset()
            if toolset is None:
                continue
            elif isinstance(toolset, AbstractToolset):
                # Pyright can't narrow Callable type aliases out of unions after isinstance check
                toolsets.append(
                    CapabilityOwnedToolset(
                        wrapped=toolset,  # pyright: ignore[reportUnknownArgumentType]
                        capability=capability,
                    )
                )
            else:
                toolsets.append(
                    CapabilityOwnedToolset(
                        wrapped=DynamicToolset[AgentDepsT](toolset_func=toolset),
                        capability=capability,
                    )
                )
        return CombinedToolset(toolsets) if toolsets else None

    def get_native_tools(self) -> Sequence[AgentNativeTool[AgentDepsT]]:
        native_tools: list[AgentNativeTool[AgentDepsT]] = []
        for capability in self.capabilities:
            cap_native_tools = capability.get_native_tools() or []
            if capability.defer_loading is not True:
                native_tools.extend(cap_native_tools)
                continue

            for native_tool in cap_native_tools:

                def deferred_native_tool(
                    ctx: RunContext[AgentDepsT],
                    *,
                    capability: AbstractCapability[AgentDepsT] = capability,
                    native_tool: AgentNativeTool[AgentDepsT] = native_tool,
                ) -> Any:
                    cap_ctx = _ctx_for_available_cap(capability, ctx)
                    if cap_ctx is None:
                        return None
                    if callable(native_tool):
                        return native_tool(cap_ctx)
                    return native_tool

                native_tools.append(deferred_native_tool)
        return native_tools

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
        return await _forward_capabilities(
            self.capabilities,
            ctx,
            tool_defs,
            _ctx_for_available_cap,
            lambda capability, cap_ctx, value: capability.prepare_tools(cap_ctx, value),
        )

    async def prepare_output_tools(
        self,
        ctx: RunContext[AgentDepsT],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        return await _forward_capabilities(
            self.capabilities,
            ctx,
            tool_defs,
            _ctx_for_available_cap,
            lambda capability, cap_ctx, value: capability.prepare_output_tools(cap_ctx, value),
        )

    # --- Run lifecycle hooks ---

    async def before_run(
        self,
        ctx: RunContext[AgentDepsT],
    ) -> None:
        await _notify_capabilities(
            self.capabilities,
            ctx,
            _ctx_for_available_cap,
            lambda capability, cap_ctx: capability.before_run(cap_ctx),
        )

    async def after_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        return await _reverse_capabilities(
            self.capabilities,
            ctx,
            result,
            _ctx_for_available_cap,
            lambda capability, cap_ctx, value: capability.after_run(cap_ctx, result=value),
        )

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: Callable[[], Awaitable[AgentRunResult[Any]]],
    ) -> AgentRunResult[Any]:
        chain = _wrap_capabilities(
            self.capabilities,
            ctx,
            handler,
            _ctx_for_available_cap,
            _make_run_wrap,
        )
        return await chain()

    async def on_run_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        error: BaseException,
    ) -> AgentRunResult[Any]:
        return await _recover_capabilities(
            self.capabilities,
            ctx,
            error,
            _ctx_for_available_cap,
            lambda capability, cap_ctx, current: capability.on_run_error(cap_ctx, error=current),
            BaseException,
        )

    # --- Node run lifecycle hooks ---

    async def before_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: _agent_graph.AgentNode[AgentDepsT, Any],
    ) -> _agent_graph.AgentNode[AgentDepsT, Any]:
        return await _forward_capabilities(
            self.capabilities,
            ctx,
            node,
            _ctx_for_available_cap,
            lambda capability, cap_ctx, value: capability.before_node_run(cap_ctx, node=value),
        )

    async def after_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: _agent_graph.AgentNode[AgentDepsT, Any],
        result: _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]],
    ) -> _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]:
        return await _reverse_capabilities(
            self.capabilities,
            ctx,
            result,
            _ctx_for_available_cap,
            lambda capability, cap_ctx, value: capability.after_node_run(cap_ctx, node=node, result=value),
        )

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
        chain = _wrap_capabilities(
            self.capabilities,
            ctx,
            handler,
            _ctx_for_available_cap,
            _make_node_run_wrap,
        )
        return await chain(node)

    async def on_node_run_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: _agent_graph.AgentNode[AgentDepsT, Any],
        error: Exception,
    ) -> _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]:
        return await _recover_capabilities(
            self.capabilities,
            ctx,
            error,
            _ctx_for_available_cap,
            lambda capability, cap_ctx, current: capability.on_node_run_error(cap_ctx, node=node, error=current),
            Exception,
        )

    # --- Event stream hook ---

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        stack = LifecycleStack(self.capabilities)

        def wrap(
            capability: AbstractCapability[AgentDepsT],
            source: AsyncIterable[AgentStreamEvent],
        ) -> AsyncIterable[AgentStreamEvent] | None:
            cap_ctx = _ctx_for_available_cap(capability, ctx)
            if cap_ctx is None:
                return None
            return capability.wrap_run_event_stream(cap_ctx, stream=source)

        async with stack.stream(stream, wrap) as wrapped:
            async for event in wrapped:
                yield event

    # --- Model request lifecycle hooks ---

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        return await _forward_capabilities(
            self.capabilities,
            ctx,
            request_context,
            _ctx_for_available_cap,
            lambda capability, cap_ctx, value: capability.before_model_request(cap_ctx, value),
        )

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        return await _reverse_capabilities(
            self.capabilities,
            ctx,
            response,
            _ctx_for_available_cap,
            lambda capability, cap_ctx, value: capability.after_model_request(
                cap_ctx, request_context=request_context, response=value
            ),
        )

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: Callable[[ModelRequestContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        chain = _wrap_capabilities(
            self.capabilities,
            ctx,
            handler,
            _ctx_for_available_cap,
            _make_model_request_wrap,
        )
        return await chain(request_context)

    async def on_model_request_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        error: Exception,
    ) -> ModelResponse:
        return await _recover_capabilities(
            self.capabilities,
            ctx,
            error,
            _ctx_for_available_cap,
            lambda capability, cap_ctx, current: capability.on_model_request_error(
                cap_ctx, request_context=request_context, error=current
            ),
            Exception,
        )

    # --- Tool validate lifecycle hooks ---

    async def before_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: str | dict[str, Any],
    ) -> str | dict[str, Any]:
        return await _forward_capabilities(
            self.capabilities,
            ctx,
            args,
            lambda capability, run_ctx: _ctx_for_tool_hook(capability, run_ctx, tool_def),
            lambda capability, cap_ctx, value: capability.before_tool_validate(
                cap_ctx, call=call, tool_def=tool_def, args=value
            ),
        )

    async def after_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        return await _reverse_capabilities(
            self.capabilities,
            ctx,
            args,
            lambda capability, run_ctx: _ctx_for_tool_hook(capability, run_ctx, tool_def),
            lambda capability, cap_ctx, value: capability.after_tool_validate(
                cap_ctx, call=call, tool_def=tool_def, args=value
            ),
        )

    async def wrap_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: str | dict[str, Any],
        handler: Callable[[str | dict[str, Any]], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        chain = _wrap_capabilities(
            self.capabilities,
            ctx,
            handler,
            lambda capability, run_ctx: _ctx_for_tool_hook(capability, run_ctx, tool_def),
            lambda capability, run_ctx, inner: _make_tool_validate_wrap(capability, run_ctx, call, tool_def, inner),
        )
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
        return await _recover_capabilities(
            self.capabilities,
            ctx,
            error,
            lambda capability, run_ctx: _ctx_for_tool_hook(capability, run_ctx, tool_def),
            lambda capability, cap_ctx, current: capability.on_tool_validate_error(
                cap_ctx, call=call, tool_def=tool_def, args=args, error=current
            ),
            (ValidationError, ModelRetry),
        )

    # --- Tool execute lifecycle hooks ---

    async def before_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        return await _forward_capabilities(
            self.capabilities,
            ctx,
            args,
            lambda capability, run_ctx: _ctx_for_tool_hook(capability, run_ctx, tool_def),
            lambda capability, cap_ctx, value: capability.before_tool_execute(
                cap_ctx, call=call, tool_def=tool_def, args=value
            ),
        )

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        return await _reverse_capabilities(
            self.capabilities,
            ctx,
            result,
            lambda capability, run_ctx: _ctx_for_tool_hook(capability, run_ctx, tool_def),
            lambda capability, cap_ctx, value: capability.after_tool_execute(
                cap_ctx, call=call, tool_def=tool_def, args=args, result=value
            ),
        )

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> Any:
        chain = _wrap_capabilities(
            self.capabilities,
            ctx,
            handler,
            lambda capability, run_ctx: _ctx_for_tool_hook(capability, run_ctx, tool_def),
            lambda capability, run_ctx, inner: _make_tool_execute_wrap(capability, run_ctx, call, tool_def, inner),
        )
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
        return await _recover_capabilities(
            self.capabilities,
            ctx,
            error,
            lambda capability, run_ctx: _ctx_for_tool_hook(capability, run_ctx, tool_def),
            lambda capability, cap_ctx, current: capability.on_tool_execute_error(
                cap_ctx, call=call, tool_def=tool_def, args=args, error=current
            ),
            Exception,
        )

    # --- Output validate lifecycle hooks ---

    async def before_output_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: RawOutput,
    ) -> RawOutput:
        return await _forward_capabilities(
            self.capabilities,
            ctx,
            output,
            _ctx_for_available_cap,
            lambda capability, cap_ctx, value: capability.before_output_validate(
                cap_ctx, output_context=output_context, output=value
            ),
        )

    async def after_output_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
    ) -> Any:
        return await _reverse_capabilities(
            self.capabilities,
            ctx,
            output,
            _ctx_for_available_cap,
            lambda capability, cap_ctx, value: capability.after_output_validate(
                cap_ctx, output_context=output_context, output=value
            ),
        )

    async def wrap_output_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: RawOutput,
        handler: WrapOutputValidateHandler,
    ) -> Any:
        chain = _wrap_capabilities(
            self.capabilities,
            ctx,
            handler,
            _ctx_for_available_cap,
            lambda capability, run_ctx, inner: _make_output_validate_wrap(capability, run_ctx, output_context, inner),
        )
        return await chain(output)

    async def on_output_validate_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: RawOutput,
        error: ValidationError | ModelRetry,
    ) -> Any:
        return await _recover_capabilities(
            self.capabilities,
            ctx,
            error,
            _ctx_for_available_cap,
            lambda capability, cap_ctx, current: capability.on_output_validate_error(
                cap_ctx, output_context=output_context, output=output, error=current
            ),
            (ValidationError, ModelRetry),
        )

    # --- Output process lifecycle hooks ---

    async def before_output_process(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
    ) -> Any:
        return await _forward_capabilities(
            self.capabilities,
            ctx,
            output,
            _ctx_for_available_cap,
            lambda capability, cap_ctx, value: capability.before_output_process(
                cap_ctx, output_context=output_context, output=value
            ),
        )

    async def after_output_process(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
    ) -> Any:
        return await _reverse_capabilities(
            self.capabilities,
            ctx,
            output,
            _ctx_for_available_cap,
            lambda capability, cap_ctx, value: capability.after_output_process(
                cap_ctx, output_context=output_context, output=value
            ),
        )

    async def wrap_output_process(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
        handler: WrapOutputProcessHandler,
    ) -> Any:
        chain = _wrap_capabilities(
            self.capabilities,
            ctx,
            handler,
            _ctx_for_available_cap,
            lambda capability, run_ctx, inner: _make_output_process_wrap(capability, run_ctx, output_context, inner),
        )
        return await chain(output)

    async def on_output_process_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
        error: Exception,
    ) -> Any:
        return await _recover_capabilities(
            self.capabilities,
            ctx,
            error,
            _ctx_for_available_cap,
            lambda capability, cap_ctx, current: capability.on_output_process_error(
                cap_ctx, output_context=output_context, output=output, error=current
            ),
            Exception,
        )

    async def handle_deferred_tool_calls(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        requests: DeferredToolRequests,
    ) -> DeferredToolResults | None:
        stack = LifecycleStack(self.capabilities)

        async def handle(
            capability: AbstractCapability[AgentDepsT], remaining: DeferredToolRequests
        ) -> DeferredToolResults | None:
            cap_ctx = _ctx_for_available_cap(capability, ctx)
            if cap_ctx is None:
                return None
            return await capability.handle_deferred_tool_calls(cap_ctx, requests=remaining)

        return await stack.settle_deferred(requests, handle)


async def _forward_capabilities(
    capabilities: Sequence[AbstractCapability[_DepsT]],
    ctx: RunContext[_DepsT],
    value: _ValueT,
    get_context: Callable[[AbstractCapability[_DepsT], RunContext[_DepsT]], RunContext[_DepsT] | None],
    transform: Callable[[AbstractCapability[_DepsT], RunContext[_DepsT], _ValueT], Awaitable[_ValueT]],
) -> _ValueT:
    stack = LifecycleStack(capabilities)

    async def apply(capability: AbstractCapability[_DepsT], current: _ValueT) -> _ValueT:
        cap_ctx = get_context(capability, ctx)
        if cap_ctx is None:
            return current
        return await transform(capability, cap_ctx, current)

    return await stack.forward(value, apply)


async def _reverse_capabilities(
    capabilities: Sequence[AbstractCapability[_DepsT]],
    ctx: RunContext[_DepsT],
    value: _ValueT,
    get_context: Callable[[AbstractCapability[_DepsT], RunContext[_DepsT]], RunContext[_DepsT] | None],
    transform: Callable[[AbstractCapability[_DepsT], RunContext[_DepsT], _ValueT], Awaitable[_ValueT]],
) -> _ValueT:
    stack = LifecycleStack(capabilities)

    async def apply(capability: AbstractCapability[_DepsT], current: _ValueT) -> _ValueT:
        cap_ctx = get_context(capability, ctx)
        if cap_ctx is None:
            return current
        return await transform(capability, cap_ctx, current)

    return await stack.reverse(value, apply)


async def _notify_capabilities(
    capabilities: Sequence[AbstractCapability[_DepsT]],
    ctx: RunContext[_DepsT],
    get_context: Callable[[AbstractCapability[_DepsT], RunContext[_DepsT]], RunContext[_DepsT] | None],
    notify: Callable[[AbstractCapability[_DepsT], RunContext[_DepsT]], Awaitable[None]],
) -> None:
    stack = LifecycleStack(capabilities)

    async def apply(capability: AbstractCapability[_DepsT]) -> None:
        cap_ctx = get_context(capability, ctx)
        if cap_ctx is not None:
            await notify(capability, cap_ctx)

    await stack.notify(apply)


def _wrap_capabilities(
    capabilities: Sequence[AbstractCapability[_DepsT]],
    ctx: RunContext[_DepsT],
    handler: Callable[_Params, Awaitable[_ResultT]],
    get_context: Callable[[AbstractCapability[_DepsT], RunContext[_DepsT]], RunContext[_DepsT] | None],
    wrap: Callable[
        [
            AbstractCapability[_DepsT],
            RunContext[_DepsT],
            Callable[_Params, Awaitable[_ResultT]],
        ],
        Callable[_Params, Awaitable[_ResultT]],
    ],
) -> Callable[_Params, Awaitable[_ResultT]]:
    stack = LifecycleStack(capabilities)

    def apply(
        capability: AbstractCapability[_DepsT],
        inner: Callable[_Params, Awaitable[_ResultT]],
    ) -> Callable[_Params, Awaitable[_ResultT]]:
        if get_context(capability, ctx) is None:
            return inner
        return wrap(capability, ctx, inner)

    return stack.wrap(handler, apply)


async def _recover_capabilities(
    capabilities: Sequence[AbstractCapability[_DepsT]],
    ctx: RunContext[_DepsT],
    error: _ErrorT,
    get_context: Callable[[AbstractCapability[_DepsT], RunContext[_DepsT]], RunContext[_DepsT] | None],
    recover: Callable[[AbstractCapability[_DepsT], RunContext[_DepsT], _ErrorT], Awaitable[_ResultT]],
    caught: type[_ErrorT] | tuple[type[_ErrorT], ...],
) -> _ResultT:
    stack = LifecycleStack(capabilities)

    async def apply(capability: AbstractCapability[_DepsT], current: _ErrorT) -> _ResultT | SkipLifecycleEntry:
        cap_ctx = get_context(capability, ctx)
        if cap_ctx is None:
            return SKIP_LIFECYCLE_ENTRY
        return await recover(capability, cap_ctx, current)

    return await stack.recover(error, apply, caught)


# --- Composition helpers ---
# These create closures that bind the current capability and inner handler,
# building a middleware chain from outermost (first cap) to innermost (last cap).


def _make_run_wrap(
    cap: AbstractCapability[AgentDepsT],
    ctx: RunContext[AgentDepsT],
    inner: Callable[[], Awaitable[AgentRunResult[Any]]],
) -> Callable[[], Awaitable[AgentRunResult[Any]]]:
    async def wrapped() -> AgentRunResult[Any]:
        return await cap.wrap_run(_ctx_for_cap(cap, ctx), handler=inner)

    return wrapped


def _make_model_request_wrap(
    cap: AbstractCapability[AgentDepsT],
    ctx: RunContext[AgentDepsT],
    inner: Callable[[ModelRequestContext], Awaitable[ModelResponse]],
) -> Callable[[ModelRequestContext], Awaitable[ModelResponse]]:
    async def wrapped(request_context: ModelRequestContext) -> ModelResponse:
        return await cap.wrap_model_request(
            _ctx_for_cap(cap, ctx),
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
        return await cap.wrap_tool_validate(
            _ctx_for_cap(cap, ctx), call=call, tool_def=tool_def, args=args, handler=inner
        )

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
        return await cap.wrap_node_run(_ctx_for_cap(cap, ctx), node=node, handler=inner)

    return wrapped


def _make_tool_execute_wrap(
    cap: AbstractCapability[AgentDepsT],
    ctx: RunContext[AgentDepsT],
    call: ToolCallPart,
    tool_def: ToolDefinition,
    inner: Callable[[dict[str, Any]], Awaitable[Any]],
) -> Callable[[dict[str, Any]], Awaitable[Any]]:
    async def wrapped(args: dict[str, Any]) -> Any:
        return await cap.wrap_tool_execute(
            _ctx_for_cap(cap, ctx), call=call, tool_def=tool_def, args=args, handler=inner
        )

    return wrapped


def _make_output_validate_wrap(
    cap: AbstractCapability[AgentDepsT],
    ctx: RunContext[AgentDepsT],
    output_context: OutputContext,
    inner: Callable[[RawOutput], Awaitable[Any]],
) -> Callable[[RawOutput], Awaitable[Any]]:
    async def wrapped(output: RawOutput) -> Any:
        return await cap.wrap_output_validate(
            _ctx_for_cap(cap, ctx), output_context=output_context, output=output, handler=inner
        )

    return wrapped


def _make_output_process_wrap(
    cap: AbstractCapability[AgentDepsT],
    ctx: RunContext[AgentDepsT],
    output_context: OutputContext,
    inner: Callable[[Any], Awaitable[Any]],
) -> Callable[[Any], Awaitable[Any]]:
    async def wrapped(output: Any) -> Any:
        return await cap.wrap_output_process(
            _ctx_for_cap(cap, ctx), output_context=output_context, output=output, handler=inner
        )

    return wrapped


def bind_capabilities_tier(
    combined: CombinedCapability[AgentDepsT],
    agent: AbstractAgent[AgentDepsT, Any],
    *,
    innermost: bool,
) -> CombinedCapability[AgentDepsT]:
    """Bind one ordering tier of the combined capability to the agent via `for_agent`.

    `Agent.__init__` binds capabilities in two phases: everything outside the `innermost`
    tier first, then — once the toolsets contributed by those capabilities are visible on
    `agent.toolsets` — the `innermost` tier (durability capabilities), whose `for_agent`
    wraps the agent's toolsets and must see all of them.
    """
    new_caps = [c.for_agent(agent) if is_innermost(c) == innermost else c for c in combined.capabilities]
    if all(new is old for new, old in zip(new_caps, combined.capabilities, strict=True)):
        return combined
    return replace(combined, capabilities=new_caps)


def _ctx_for_cap(capability: AbstractCapability[AgentDepsT], ctx: RunContext[AgentDepsT]) -> RunContext[AgentDepsT]:
    return replace(ctx, capability_loaded=_capability_loaded(capability, ctx))


def _ctx_for_available_cap(
    capability: AbstractCapability[AgentDepsT], ctx: RunContext[AgentDepsT]
) -> RunContext[AgentDepsT] | None:
    capability_loaded = _capability_loaded(capability, ctx)
    if capability.defer_loading is True and not capability_loaded:
        return None
    return replace(ctx, capability_loaded=capability_loaded)


def _ctx_for_tool_hook(
    capability: AbstractCapability[AgentDepsT], ctx: RunContext[AgentDepsT], tool_def: ToolDefinition
) -> RunContext[AgentDepsT] | None:
    """The context for a tool validate/execute hook, or `None` when the hook must not run.

    Like `_ctx_for_available_cap`, but a capability's *own* tool always activates its hooks:
    a capability-owned tool stays callable while its capability counts as unloaded — reveal
    evidence in history with no load pair, or a load pair reset at a `CompactionPart`
    boundary — and executing it without its owner's validation/approval hooks would turn
    model-visibility state into a hook bypass. `capability_loaded` on the returned context
    stays truthful, so a hook can still distinguish the two states.
    """
    cap_ctx = _ctx_for_available_cap(capability, ctx)
    if cap_ctx is None and tool_def.capability_id is not None and tool_def.capability_id == capability.id:
        return _ctx_for_cap(capability, ctx)
    return cap_ctx


def _capability_loaded(capability: AbstractCapability[AgentDepsT], ctx: RunContext[AgentDepsT]) -> bool:
    if capability.defer_loading is not True:
        return True

    # Deferred capabilities are required to have an explicit `id` (enforced in
    # `_build_run_capabilities`), which is also the key they're registered under, so we read
    # it directly rather than resolving the instance back to its run-local registry id.
    return capability.id is not None and capability.id in ctx.available_capability_ids
