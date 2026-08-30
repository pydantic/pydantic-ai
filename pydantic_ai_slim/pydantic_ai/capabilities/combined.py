from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

from pydantic import ValidationError

from pydantic_ai._instructions import (
    AgentInstruction,
    AgentInstructions,
    SourcedInstruction,
    normalize_instructions,
    validate_instruction_id_segment,
)
from pydantic_ai._utils import aclose_all, gather, replace_no_init
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import AgentStreamEvent, ModelResponse, ToolCallPart
from pydantic_ai.settings import ModelSettings, merge_model_settings
from pydantic_ai.tools import (
    AgentDepsT,
    AgentNativeTool,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolDefinition,
)
from pydantic_ai.toolsets import AbstractToolset, AgentToolset, CombinedToolset
from pydantic_ai.toolsets._capability_owned import CapabilityOwnedToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset

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


@dataclass
class CombinedCapability(AbstractCapability[AgentDepsT]):
    """A capability that combines multiple capabilities.

    When any child returns a fresh instance from
    [`for_agent`][pydantic_ai.capabilities.AbstractCapability.for_agent] or
    [`for_run`][pydantic_ai.capabilities.AbstractCapability.for_run], the container is rebound
    via `_rebound`: a shallow copy holding the new children, with subclass state carried over
    verbatim and `__init__`/`__post_init__` not re-run. Compute values derived from `capabilities`
    on access (e.g. via a property) rather than caching them at construction, so they can't go
    stale across a rebind. `_instruction_sources` is the one thing that can't be — flattening
    destroys what it records — so it's carried across by `_rebound` instead, and swapping children
    any other way (`replace()`) silently rebuilds it from the flattened list.
    """

    capabilities: Sequence[AbstractCapability[AgentDepsT]]
    # Combined capabilities are flattened for hook/tool ordering, but public instruction overrides
    # belong to the container itself and therefore need a composition view that retains it.
    _instruction_sources: Sequence[AbstractCapability[AgentDepsT]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.id is not None:
            validate_instruction_id_segment(self.id, kind='Capability id')
        instruction_sources: list[AbstractCapability[AgentDepsT]] = []
        for capability in self.capabilities:
            if (
                isinstance(capability, CombinedCapability)
                and type(capability).get_instructions is CombinedCapability.get_instructions
            ):
                # Its own view, not its `capabilities`: those have already been flattened, so any
                # container *it* retained would be splatted back out and its override lost. Every
                # re-composition goes through here (`Agent.iter` re-combines the resolved layers
                # whenever instrumentation or a run capability is present, and `replace()` re-runs
                # `__post_init__`), so taking the flattened list would drop the override mid-run.
                instruction_sources.extend(capability._instruction_sources)
            else:
                instruction_sources.append(capability)
        self._instruction_sources = instruction_sources
        self.__normalize_capabilities()

    def _rebound(self, new_capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> CombinedCapability[AgentDepsT]:
        """A shallow copy holding `new_capabilities`, with the composition view carried across.

        The only supported way to swap a container's children. `replace()` would re-run
        `__post_init__`, which rebuilds `_instruction_sources` from the already-flattened
        `capabilities` — and flattening is what drops a nested container that overrides
        `get_instructions`, so rebuilding is exactly what loses it.
        """
        new_self = replace_no_init(self, capabilities=list(new_capabilities))
        # Keep ordinary sources aligned with their bound replacements while retained combined
        # overrides continue to represent the container that owns the public method.
        replacements = {id(old): new for old, new in zip(self.capabilities, new_capabilities)}

        def rebind(source: AbstractCapability[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
            if (replacement := replacements.get(id(source))) is not None:
                return replacement
            # Anything else is a retained container: `_instruction_sources` holds either direct
            # children, replaced above, or the containers flattening splatted out. Those are not in
            # `capabilities` and so not in `replacements`, and left alone one would keep answering
            # from the children it had before the bind, with its leaves absent from the ordering
            # positions sorting its part last. Its children *are* in `replacements`, having been
            # flattened into the very list that was just rebound, so rebuild it from those.
            assert isinstance(source, CombinedCapability)
            return source._rebound([replacements.get(id(child), child) for child in source.capabilities])

        new_self._instruction_sources = [rebind(source) for source in new_self._instruction_sources]
        new_self.__normalize_capabilities()
        return new_self

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
    def _has_on_node_run_error(self) -> bool:
        return any(c._has_on_node_run_error for c in self.capabilities)

    @property
    def _has_wrap_model_request(self) -> bool:
        return any(c._has_wrap_model_request for c in self.capabilities)

    @property
    def _has_on_model_request_error(self) -> bool:
        return any(c._has_on_model_request_error for c in self.capabilities)

    @property
    def has_wrap_run_event_stream(self) -> bool:
        return any(c.has_wrap_run_event_stream for c in self.capabilities)

    def for_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> CombinedCapability[AgentDepsT]:
        new_caps = [capability.for_agent(agent) for capability in self.capabilities]
        if all(new is old for new, old in zip(new_caps, self.capabilities)):
            return self
        return self._rebound(new_caps)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        new_caps = await gather(*(c.for_run(ctx) for c in self.capabilities))
        if all(new is old for new, old in zip(new_caps, self.capabilities)):
            return self
        return self._rebound(new_caps)

    def _validate_runtime_capabilities(
        self, ctx: RunContext[AgentDepsT], capabilities: Sequence[AbstractCapability[AgentDepsT]]
    ) -> None:
        for capability in self.capabilities:
            capability._validate_runtime_capabilities(ctx, capabilities)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        # The children's contributions, not `_collect_instructions`: that asks whether a subclass
        # took this method over, and answering it by calling this method is a loop.
        instructions: list[AgentInstruction[AgentDepsT]] = [
            sourced.instruction for sourced in self._collect_child_instructions()
        ]
        return instructions or None

    def _ordered_instruction_sources(self) -> list[AbstractCapability[AgentDepsT]]:
        """The composition view, in the order the ordering pass settled the leaves into.

        Instruction parts have always followed `capabilities`, which `__normalize_capabilities`
        flattens *and* sorts, so a capability that asks to be `outermost` contributes its part
        first however late it was registered. `_instruction_sources` is in registration order and
        retains the containers flattening drops, so each source takes the position of its
        earliest-placed leaf. Sorting the sources directly instead would re-run the ordering pass
        over those retained containers and could raise `Conflicting positions` for one whose leaves
        span two tiers -- the very case flattening exists to avoid.

        Computed on access rather than stored, so a rebind can't leave it stale.
        """
        positions = {id(capability): index for index, capability in enumerate(self.capabilities)}

        def position(source: AbstractCapability[AgentDepsT]) -> int:
            # A source with no leaf among `capabilities` (nothing does this today) sorts last
            # rather than first, so an unplaceable part can't displace a placed one.
            return min(
                (positions[id(leaf)] for leaf in collect_leaves(source) if id(leaf) in positions),
                default=len(positions),
            )

        return sorted(self._instruction_sources, key=position)

    def _collect_instructions(self) -> list[SourcedInstruction[AgentDepsT]]:
        relayed = self._collect_child_instructions()
        if type(self).get_instructions is CombinedCapability.get_instructions:
            return relayed
        # `get_instructions` is a public extension point, so a subclass that overrides it decides what
        # this container says. What it hands back is usually its children's own parts, and those stay
        # attributed to whoever authored them; anything else it wrote itself.
        return self._attribute_container_instructions(normalize_instructions(self.get_instructions()), relayed)

    def _collect_child_instructions(self) -> list[SourcedInstruction[AgentDepsT]]:
        """Collect what the children contribute, without asking whether this container overrode them.

        Reached without the override check so a subclass delegating to `super().get_instructions()`
        is answered by its children rather than routed back into its own override.
        """
        instructions: list[SourcedInstruction[AgentDepsT]] = []
        for capability in self._ordered_instruction_sources():
            if capability.defer_loading is not True:
                instructions.extend(capability._collect_instructions())
        return instructions

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
                    cap_ctx = _ctx_for_active_cap(capability, ctx)
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
            # This layering only runs in the classic request pipeline, where `ctx.model_settings`
            # never holds `RealtimeModelSettings` (realtime sessions resolve settings at connect).
            for entry in settings_chain:
                # Mutate ctx.model_settings so each dynamic entry sees the
                # accumulated settings from all prior layers.
                ctx.model_settings = merge_model_settings(cast('ModelSettings | None', ctx.model_settings), merged)
                resolved = entry(ctx) if callable(entry) else entry
                merged = merge_model_settings(merged, resolved)
            # Update ctx.model_settings to include the final entry's contribution
            ctx.model_settings = merge_model_settings(cast('ModelSettings | None', ctx.model_settings), merged)
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
                    cap_ctx = _ctx_for_active_cap(capability, ctx)
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
        for capability in self.capabilities:
            if (cap_ctx := _ctx_for_active_cap(capability, ctx)) is not None:
                tool_defs = await capability.prepare_tools(cap_ctx, tool_defs)
        return tool_defs

    async def prepare_output_tools(
        self,
        ctx: RunContext[AgentDepsT],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        for capability in self.capabilities:
            if (cap_ctx := _ctx_for_active_cap(capability, ctx)) is not None:
                tool_defs = await capability.prepare_output_tools(cap_ctx, tool_defs)
        return tool_defs

    # --- Run lifecycle hooks ---

    async def before_run(
        self,
        ctx: RunContext[AgentDepsT],
    ) -> None:
        for capability in self.capabilities:
            if (cap_ctx := _ctx_for_active_cap(capability, ctx)) is not None:
                await capability.before_run(cap_ctx)

    def _prepare_run_context(self, ctx: RunContext[AgentDepsT]) -> None:
        for capability in self.capabilities:
            capability._prepare_run_context(ctx)

    async def after_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        for capability in reversed(self.capabilities):
            if (cap_ctx := _ctx_for_active_cap(capability, ctx)) is not None:
                result = await capability.after_run(cap_ctx, result=result)
        return result

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: Callable[[], Awaitable[AgentRunResult[Any]]],
    ) -> AgentRunResult[Any]:
        chain = handler
        for capability in reversed(self.capabilities):
            if _ctx_for_active_cap(capability, ctx) is not None:
                chain = _make_run_wrap(capability, ctx, chain)
        return await chain()

    async def on_run_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        error: BaseException,
    ) -> AgentRunResult[Any]:
        for capability in reversed(self.capabilities):
            cap_ctx = _ctx_for_active_cap(capability, ctx)
            if cap_ctx is None:
                continue
            try:
                return await capability.on_run_error(cap_ctx, error=error)
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
            if (cap_ctx := _ctx_for_active_cap(capability, ctx)) is not None:
                node = await capability.before_node_run(cap_ctx, node=node)
        return node

    async def after_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: _agent_graph.AgentNode[AgentDepsT, Any],
        result: _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]],
    ) -> _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]:
        for capability in reversed(self.capabilities):
            if (cap_ctx := _ctx_for_active_cap(capability, ctx)) is not None:
                result = await capability.after_node_run(cap_ctx, node=node, result=result)
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
        for capability in reversed(self.capabilities):
            if capability._has_wrap_node_run and _ctx_for_active_cap(capability, ctx) is not None:
                chain = _make_node_run_wrap(capability, ctx, chain)
        return await chain(node)

    async def on_node_run_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: _agent_graph.AgentNode[AgentDepsT, Any],
        error: Exception,
    ) -> _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]:
        for capability in reversed(self.capabilities):
            if not capability._has_on_node_run_error:
                continue
            cap_ctx = _ctx_for_active_cap(capability, ctx)
            if cap_ctx is None:
                continue
            try:
                return await capability.on_node_run_error(cap_ctx, node=node, error=error)
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
        wrapped_streams = [stream]
        for capability in reversed(self.capabilities):
            if (cap_ctx := _ctx_for_active_cap(capability, ctx)) is not None:
                stream = capability.wrap_run_event_stream(cap_ctx, stream=stream)
                wrapped_streams.append(stream)
        try:
            async for event in stream:
                yield event
        finally:
            await aclose_all(reversed(wrapped_streams))

    # --- Model request lifecycle hooks ---

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        for capability in self.capabilities:
            if (cap_ctx := _ctx_for_active_cap(capability, ctx)) is not None:
                request_context = await capability.before_model_request(cap_ctx, request_context)
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        for capability in reversed(self.capabilities):
            if (cap_ctx := _ctx_for_active_cap(capability, ctx)) is not None:
                response = await capability.after_model_request(
                    cap_ctx, request_context=request_context, response=response
                )
        return response

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: Callable[[ModelRequestContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        chain = handler
        for capability in reversed(self.capabilities):
            if capability._has_wrap_model_request and _ctx_for_active_cap(capability, ctx) is not None:
                chain = _make_model_request_wrap(capability, ctx, chain)
        return await chain(request_context)

    async def on_model_request_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        error: Exception,
    ) -> ModelResponse:
        for capability in reversed(self.capabilities):
            if not capability._has_on_model_request_error:
                continue
            cap_ctx = _ctx_for_active_cap(capability, ctx)
            if cap_ctx is None:
                continue
            try:
                return await capability.on_model_request_error(cap_ctx, request_context=request_context, error=error)
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
            if (cap_ctx := _ctx_for_active_cap(capability, ctx)) is not None:
                args = await capability.before_tool_validate(cap_ctx, call=call, tool_def=tool_def, args=args)
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
            if (cap_ctx := _ctx_for_active_cap(capability, ctx)) is not None:
                args = await capability.after_tool_validate(cap_ctx, call=call, tool_def=tool_def, args=args)
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
        for capability in reversed(self.capabilities):
            if _ctx_for_active_cap(capability, ctx) is not None:
                chain = _make_tool_validate_wrap(capability, ctx, call, tool_def, chain)
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
            cap_ctx = _ctx_for_active_cap(capability, ctx)
            if cap_ctx is None:
                continue
            try:
                return await capability.on_tool_validate_error(
                    cap_ctx, call=call, tool_def=tool_def, args=args, error=error
                )
            except (ValidationError, ModelRetry) as new_error:
                error = new_error
            except Exception:
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
            if (cap_ctx := _ctx_for_active_cap(capability, ctx)) is not None:
                args = await capability.before_tool_execute(cap_ctx, call=call, tool_def=tool_def, args=args)
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
            if (cap_ctx := _ctx_for_active_cap(capability, ctx)) is not None:
                result = await capability.after_tool_execute(
                    cap_ctx, call=call, tool_def=tool_def, args=args, result=result
                )
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
        for capability in reversed(self.capabilities):
            if _ctx_for_active_cap(capability, ctx) is not None:
                chain = _make_tool_execute_wrap(capability, ctx, call, tool_def, chain)
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
            cap_ctx = _ctx_for_active_cap(capability, ctx)
            if cap_ctx is None:
                continue
            try:
                return await capability.on_tool_execute_error(
                    cap_ctx, call=call, tool_def=tool_def, args=args, error=error
                )
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
            if (cap_ctx := _ctx_for_active_cap(capability, ctx)) is not None:
                output = await capability.before_output_validate(cap_ctx, output_context=output_context, output=output)
        return output

    async def after_output_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
    ) -> Any:
        for capability in reversed(self.capabilities):
            if (cap_ctx := _ctx_for_active_cap(capability, ctx)) is not None:
                output = await capability.after_output_validate(cap_ctx, output_context=output_context, output=output)
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
        for capability in reversed(self.capabilities):
            if _ctx_for_active_cap(capability, ctx) is not None:
                chain = _make_output_validate_wrap(capability, ctx, output_context, chain)
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
            cap_ctx = _ctx_for_active_cap(capability, ctx)
            if cap_ctx is None:
                continue
            try:
                return await capability.on_output_validate_error(
                    cap_ctx, output_context=output_context, output=output, error=error
                )
            except (ValidationError, ModelRetry) as new_error:
                error = new_error
            except Exception:  # pragma: no cover
                # Defensive.
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
            if (cap_ctx := _ctx_for_active_cap(capability, ctx)) is not None:
                output = await capability.before_output_process(cap_ctx, output_context=output_context, output=output)
        return output

    async def after_output_process(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
    ) -> Any:
        for capability in reversed(self.capabilities):
            if (cap_ctx := _ctx_for_active_cap(capability, ctx)) is not None:
                output = await capability.after_output_process(cap_ctx, output_context=output_context, output=output)
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
        for capability in reversed(self.capabilities):
            if _ctx_for_active_cap(capability, ctx) is not None:
                chain = _make_output_process_wrap(capability, ctx, output_context, chain)
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
            cap_ctx = _ctx_for_active_cap(capability, ctx)
            if cap_ctx is None:
                continue
            try:
                return await capability.on_output_process_error(
                    cap_ctx, output_context=output_context, output=output, error=error
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
            cap_ctx = _ctx_for_active_cap(capability, ctx)
            if cap_ctx is None:
                continue
            result = await capability.handle_deferred_tool_calls(cap_ctx, requests=remaining)
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
    return combined._rebound(new_caps)  # pyright: ignore[reportPrivateUsage]


def _ctx_for_cap(capability: AbstractCapability[AgentDepsT], ctx: RunContext[AgentDepsT]) -> RunContext[AgentDepsT]:
    return _replace_capability_context(ctx, capability_active=_capability_active(capability, ctx))


def _ctx_for_active_cap(
    capability: AbstractCapability[AgentDepsT], ctx: RunContext[AgentDepsT]
) -> RunContext[AgentDepsT] | None:
    capability_active = _capability_active(capability, ctx)
    if capability.defer_loading is True and not capability_active:
        return None
    return _replace_capability_context(ctx, capability_active=capability_active)


def _replace_capability_context(ctx: RunContext[AgentDepsT], *, capability_active: bool) -> RunContext[AgentDepsT]:
    return replace(ctx, capability_active=capability_active)


def _capability_active(capability: AbstractCapability[AgentDepsT], ctx: RunContext[AgentDepsT]) -> bool:
    """Whether this capability may act on the current step.

    Activity, not loading: an always-on capability is active for the whole run without ever being
    loaded, which is why the deferred branch below is the only one that consults history.
    """
    if capability.defer_loading is not True:
        return True

    # Deferred capabilities are required to have an explicit `id` (enforced in
    # `_build_run_capabilities`), which is also the key they're registered under, so we read
    # it directly rather than resolving the instance back to its run-local registry id.
    return capability.id is not None and capability.id in ctx.active_capability_ids
