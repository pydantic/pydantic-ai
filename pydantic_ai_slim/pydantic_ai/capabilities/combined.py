from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Self, cast

from pydantic import ValidationError
from typing_extensions import TypeIs

from pydantic_ai._enqueue import copy_said, hold_added_messages, hold_feedback, says_the_same, snapshot_messages
from pydantic_ai._history_mirroring import keep_mirroring
from pydantic_ai._instructions import (
    AgentInstruction,
    AgentInstructions,
    SourcedInstruction,
    normalize_instructions,
    validate_instruction_id_segment,
)
from pydantic_ai._utils import aclose_all, gather, replace_no_init
from pydantic_ai.exceptions import ModelRetry, RetryModelRequest, SkipModelRequest, UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    Authority,
    InstructionPart,
    ModelResponse,
    ModelResponsePart,
    ToolCallPart,
)
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
from pydantic_ai.workspaces import Workspace, WorkspaceBackend, WorkspaceRef

from ._on_event import collect_on_event_methods, marked_listens_to
from ._ordering import collect_leaves, is_innermost, sort_capabilities
from .abstract import (
    AbstractCapability,
    AgentModel,
    RawOutput,
    WrapOutputProcessHandler,
    WrapOutputValidateHandler,
    _combination_roots,  # pyright: ignore[reportPrivateUsage]
    _LoweredInstructionSource,  # pyright: ignore[reportPrivateUsage]
    _RunIdInputsFactory,  # pyright: ignore[reportPrivateUsage]
)
from .wrapper import WrapperCapability

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

    A nested `CombinedCapability` is only a grouping: its children are merged into the enclosing
    one. Give it an `id` and `defer_loading=True` to load its children together instead: the model
    loads the group by its `id`, and loading it activates every child at once. Each child still
    keeps its own ordering position among all of the agent's capabilities, as if the group were
    merged.

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
    # The hook dispatch order. `capabilities` is the tree (plain groupings merged in, each deferred
    # group kept as one object) that walking and rebuilding use; here a deferred group's children
    # appear one by one as `_GroupMember`s, sorted by ordering constraints with the rest. Without a
    # deferred group both lists hold the same sorted capabilities, as before groups existed. With one,
    # `capabilities` keeps the given order: no order of whole branches matches the interleaved
    # dispatch order, and rebuilding from a re-sorted one would change how ties sort.
    _dispatch: Sequence[AbstractCapability[AgentDepsT]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.id is not None:
            validate_instruction_id_segment(self.id, kind='Capability id')
        instruction_sources: list[AbstractCapability[AgentDepsT]] = []
        for capability in self.capabilities:
            if (
                isinstance(capability, CombinedCapability)
                and not is_deferred_group(capability)
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

    def _rebound(
        self,
        new_capabilities: Sequence[AbstractCapability[AgentDepsT]],
        replacements: Mapping[int, Sequence[AbstractCapability[AgentDepsT] | None]] | None = None,
    ) -> Self:
        """A shallow copy holding `new_capabilities`, with the composition view carried across.

        The only supported way to swap a container's children. `replace()` would re-run
        `__post_init__`, which rebuilds `_instruction_sources` from the already-flattened
        `capabilities` — and flattening is what drops a nested container that overrides
        `get_instructions`, so rebuilding is exactly what loses it.

        `replacements` maps each old child by `id()` to what replaced it, or to `None` when it was
        removed. One decision per *occurrence*, in the order the children were visited: the same
        object may sit in `capabilities` more than once, and a merge that keeps the first and drops
        the rest has a different answer for each. Keying by `id()` alone would collapse those into
        one, so every occurrence would take the surviving decision and contribute its instructions
        again. It defaults to pairing the old children with the new ones positionally, which only
        holds when every child was replaced one-for-one;
        [`visit_and_replace`][pydantic_ai.capabilities.AbstractCapability.visit_and_replace] can
        drop children, so it passes the mapping explicitly.
        """
        new_self = replace_no_init(self, capabilities=list(new_capabilities))
        # Keep ordinary sources aligned with their bound replacements while retained combined
        # overrides continue to represent the container that owns the public method.
        if replacements is None:
            positional: dict[int, list[AbstractCapability[AgentDepsT] | None]] = {}
            for old, new in zip(self.capabilities, new_capabilities):
                positional.setdefault(id(old), []).append(new)
            replacements = positional
        # Consumed as a queue so repeats of one object take their decisions in turn. The i-th
        # occurrence here is the i-th occurrence there: `_instruction_sources` and `capabilities`
        # are built from the same list, and `sort_capabilities` is stable, so neither reorders one
        # occurrence of an object past another.
        pending = {key: list(decisions) for key, decisions in replacements.items()}
        new_self._instruction_sources = [
            rebound
            for source in new_self._instruction_sources
            if (rebound := self.__rebind_source(source, replacements, pending)) is not None
        ]
        new_self.__normalize_capabilities()
        return new_self

    # Name-mangled for the same reason as `__normalize_capabilities`: it is part of the rebind invariant.
    def __rebind_source(
        self,
        source: AbstractCapability[AgentDepsT],
        replacements: Mapping[int, Sequence[AbstractCapability[AgentDepsT] | None]],
        pending: dict[int, list[AbstractCapability[AgentDepsT] | None]],
    ) -> AbstractCapability[AgentDepsT] | None:
        """What an instruction source of a rebound container becomes, taking its decisions from `pending`."""
        if id(source) in replacements:
            return _take_decision(source, pending)
        # Anything else is a retained container: `_instruction_sources` holds either direct
        # children, replaced above, or the containers flattening splatted out. Those are not in
        # `capabilities` and so not in `replacements`, and left alone one would keep answering
        # from the children it had before the bind, with its leaves absent from the ordering
        # positions sorting its part last. Its children *are* in `replacements`, having been
        # flattened into the very list that was just rebound, so rebuild it from those.
        assert isinstance(source, CombinedCapability)
        rebound_children = [
            child if id(child) not in replacements else _take_decision(child, pending) for child in source.capabilities
        ]
        surviving = [child for child in rebound_children if child is not None]
        # A retained container that lost every child contributes no instructions any more, so
        # it drops out of the composition view rather than lingering as an empty source.
        return source._rebound(surviving, replacements) if surviving else None

    # Name-mangled deliberately: this upholds a base-class invariant on rebinds, so a
    # subclass attribute of the same name must not be able to override it.
    def __normalize_capabilities(self) -> None:
        # Splat any nested plain grouping so leaves participate as siblings in the outer
        # ordering pass. Without this, a nested `CombinedCapability` whose leaves span both
        # `outermost` and `innermost` tiers would force `_effective_ordering` to merge them
        # into a single position and raise `Conflicting positions`.
        branches: list[AbstractCapability[AgentDepsT]] = []
        for cap in self.capabilities:
            branches.extend(_combination_roots(cap))
        self.capabilities = branches
        # A deferred group stays one branch, but its children are dispatched and ordered one by one, as
        # if merged: an `outermost` child must still run outside every other capability of the agent.
        self._dispatch = [
            member
            for branch in branches
            for member in (
                [_GroupMember(child, group=branch) for child in branch.capabilities]
                if is_deferred_group(branch)
                else [branch]
            )
        ]
        if any(leaf.get_ordering() is not None for leaf in collect_leaves(self)):
            self._dispatch = sort_capabilities(list(self._dispatch))
            if not any(map(is_deferred_group, branches)):
                # Without a deferred group the two lists are one, so registration, binding and `for_run`
                # follow the settled order, as before groups existed.
                self.capabilities = list(self._dispatch)

    def apply(self, visitor: Callable[[AbstractCapability[AgentDepsT]], None]) -> None:
        # Like a wrapper, a deferred group is registered itself, ahead of its children.
        if self.defer_loading is True:
            visitor(self)
        for cap in self.capabilities:
            cap.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[AbstractCapability[AgentDepsT]], AbstractCapability[AgentDepsT] | None]
    ) -> AbstractCapability[AgentDepsT] | None:
        """Visit each child and rebuild the container from the survivors.

        A child the visitor removed is reported to `_rebound` so the composition view drops it
        too; see
        [`AbstractCapability.visit_and_replace`][pydantic_ai.capabilities.AbstractCapability.visit_and_replace]
        for the tree-walking contract.
        """
        if self.defer_loading is True and (replacement := visitor(self)) is not self:
            # Replacing or removing it takes its children with it, as for a wrapper.
            return replacement
        new_caps: list[AbstractCapability[AgentDepsT]] = []
        # `_rebound` needs to know which children were removed, not just which survived, so the
        # composition view can drop them too; a positional pairing can't express a removal.
        replacements: dict[int, list[AbstractCapability[AgentDepsT] | None]] = {}
        unchanged = True
        for cap in self.capabilities:
            new_cap = cap.visit_and_replace(visitor)
            replacements.setdefault(id(cap), []).append(new_cap)
            if new_cap is not cap:
                unchanged = False
            if new_cap is not None:
                new_caps.append(new_cap)
        if unchanged:
            return self
        if not new_caps:
            # A container that lost every child contributes nothing, and reporting it as removed is
            # what lets an enclosing wrapper or container drop it in turn.
            return None
        return self._rebound(new_caps, replacements)

    @property
    def _has_wrap_node_run(self) -> bool:
        return any(c._has_wrap_node_run for c in self._dispatch)

    @property
    def _has_on_node_run_error(self) -> bool:
        return any(c._has_on_node_run_error for c in self._dispatch)

    @property
    def _has_wrap_model_request(self) -> bool:
        return any(c._has_wrap_model_request for c in self._dispatch)

    @property
    def _has_on_model_request_error(self) -> bool:
        return any(c._has_on_model_request_error for c in self._dispatch)

    @property
    def _has_prepare_model_request(self) -> bool:
        return any(c._has_prepare_model_request for c in self._dispatch)

    def _handles_model_request_error(self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext) -> bool:
        return any(
            capability._has_on_model_request_error
            and (cap_ctx := _ctx_for_active_cap(capability, ctx)) is not None
            and capability._handles_model_request_error(cap_ctx, request_context)
            for capability in self._dispatch
        )

    @property
    def _can_supply_model(self) -> bool:
        return any(c.defer_loading is not True and c._can_supply_model for c in self._dispatch)

    @property
    def has_wrap_run_event_stream(self) -> bool:
        return any(c.has_wrap_run_event_stream for c in self._dispatch)

    @property
    def has_on_event(self) -> bool:
        return (
            type(self).on_event is not CombinedCapability.on_event
            or bool(collect_on_event_methods(type(self)))
            or any(c.has_on_event for c in self._dispatch)
        )

    def listens_to(self, event: AgentStreamEvent) -> bool:
        return (
            type(self).on_event is not CombinedCapability.on_event
            or marked_listens_to(type(self), event)
            or any(c.listens_to(event) for c in self._dispatch)
        )

    def with_authority(self, authority: Authority) -> Self:
        """Return a copy whose children, and so what each of them enqueues, carry no more than `authority`.

        Composition merges a container's children into the enclosing one and dispatches each on its own, so the ceiling
        is set on every child as well as on the container.
        """
        lowered = super().with_authority(authority)
        return lowered._rebound([child.with_authority(authority) for child in self.capabilities])

    def for_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> CombinedCapability[AgentDepsT]:
        new_caps = [capability._keep_authority(capability.for_agent(agent)) for capability in self.capabilities]
        if all(new is old for new, old in zip(new_caps, self.capabilities)):
            return self
        return self._rebound(new_caps)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        new_caps = await gather(*(c._for_run_keeping_authority(ctx) for c in self.capabilities))
        if all(new is old for new, old in zip(new_caps, self.capabilities)):
            return self
        return self._rebound(new_caps)

    def _validate_runtime_capabilities(
        self, ctx: RunContext[AgentDepsT], capabilities: Sequence[AbstractCapability[AgentDepsT]]
    ) -> None:
        for capability in self._dispatch:
            capability._validate_runtime_capabilities(ctx, capabilities)

    def _validate_override_capabilities(self, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> None:
        for capability in self._dispatch:
            capability._validate_override_capabilities(capabilities)

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
        positions = {id(capability): index for index, capability in enumerate(self._dispatch)}

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
        lowered = {id(source.capability) for source in self._lowered_instruction_sources()}
        for capability in self._ordered_instruction_sources():
            # A capability below operator authority states its instructions in history instead.
            if capability.defer_loading is not True and id(capability) not in lowered:
                instructions.extend(capability._collect_instructions())
        return instructions

    def _lowered_instruction_sources(self) -> list[_LoweredInstructionSource[AgentDepsT]]:
        # Each source is judged on its own: `with_authority` lowers a container's children too. A container whose
        # subclass took `get_instructions` over decides what it says instead of its lowered children, whose toolsets'
        # instructions are still stated; below operator authority, what it says is a block of its own.
        children = [
            source
            for capability in self._ordered_instruction_sources()
            for source in capability._lowered_instruction_sources()
        ]
        if type(self).get_instructions is CombinedCapability.get_instructions:
            return children
        return [*super()._lowered_instruction_sources(), *self._replace_lowered_instructions(children)]

    def get_model_settings(self) -> ModelSettings | Callable[[RunContext[AgentDepsT]], ModelSettings] | None:
        # Collect settings in order, preserving each capability's position in the merge chain.
        # Each entry is either a static dict or a dynamic callable.
        settings_chain: list[ModelSettings | Callable[[RunContext[AgentDepsT]], ModelSettings]] = []
        for capability in self._dispatch:
            cap_settings = capability.get_model_settings()

            if cap_settings is None:
                continue

            settings_chain.append(_settings_entry(capability, cap_settings))

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
        default: AgentModel[AgentDepsT] | None = None
        for capability in self._dispatch:
            if capability.defer_loading is not True and (capability_model := capability.get_model()) is not None:
                if capability._model_is_default():
                    default = capability_model
                else:
                    model = capability_model
        return model if model is not None else default

    @property
    def has_resolve_model_id(self) -> bool:
        return any(
            capability.defer_loading is not True and capability.has_resolve_model_id for capability in self._dispatch
        )

    async def resolve_model_id(
        self,
        ctx: ModelResolutionContext[AgentDepsT],
        *,
        model_id: KnownModelName | str,
    ) -> Model | None:
        for capability in self._dispatch:
            if capability.defer_loading is True:
                continue
            if (model := await capability.resolve_model_id(ctx, model_id=model_id)) is not None:
                return model
        return None

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        toolsets: list[AbstractToolset[AgentDepsT]] = []
        for capability in self._dispatch:
            toolset = capability.get_toolset()
            # A deferred group's member contributes on the group's behalf, so loading the group reveals its tools.
            owner = capability.group if isinstance(capability, _GroupMember) else capability
            if toolset is None:
                continue
            elif isinstance(toolset, AbstractToolset):
                # Pyright can't narrow Callable type aliases out of unions after isinstance check
                toolsets.append(
                    CapabilityOwnedToolset(
                        wrapped=toolset,  # pyright: ignore[reportUnknownArgumentType]
                        capability=owner,
                    )
                )
            else:
                toolsets.append(
                    CapabilityOwnedToolset(
                        wrapped=DynamicToolset[AgentDepsT](toolset_func=toolset),
                        capability=owner,
                    )
                )
        return CombinedToolset(toolsets) if toolsets else None

    def get_native_tools(self) -> Sequence[AgentNativeTool[AgentDepsT]]:
        native_tools: list[AgentNativeTool[AgentDepsT]] = []
        for capability in self._dispatch:
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

    @property
    def _has_get_workspace(self) -> bool:
        return type(self).get_workspace is not CombinedCapability.get_workspace or any(
            capability._has_get_workspace for capability in self._dispatch
        )

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        for capability in self._dispatch:
            if (workspace := capability.get_workspace(ctx, ref=ref)) is not None:
                return workspace
        return None

    def _prepare_workspace(self, ctx: RunContext[AgentDepsT], workspace: Workspace, *, explicit: bool) -> Workspace:
        # Middleware order, like `get_wrapper_toolset`: the last capability wraps first, so its
        # wrapper sits innermost, directly around the selected workspace.
        for capability in reversed(self._dispatch):
            workspace = capability._prepare_workspace(ctx, workspace, explicit=explicit)
        return workspace

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        wrapped = toolset
        any_wrapped = False
        for capability in reversed(self._dispatch):
            result = capability.get_wrapper_toolset(wrapped)
            if result is not None:
                wrapped = result
                any_wrapped = True
        return wrapped if any_wrapped else None

    def get_wrapper_model(self, model: Model) -> Model | None:
        wrapped = model
        for capability in reversed(self._dispatch):
            if capability.defer_loading is not True and (result := capability.get_wrapper_model(wrapped)) is not None:
                wrapped = result
        return wrapped if wrapped is not model else None

    # --- Tool preparation hooks ---

    async def prepare_tools(
        self,
        ctx: RunContext[AgentDepsT],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        for capability in self._dispatch:
            if (cap_ctx := _ctx_for_active_cap(capability, ctx)) is not None:
                tool_defs = await capability.prepare_tools(cap_ctx, tool_defs)
        return tool_defs

    async def prepare_output_tools(
        self,
        ctx: RunContext[AgentDepsT],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        for capability in self._dispatch:
            if (cap_ctx := _ctx_for_active_cap(capability, ctx)) is not None:
                tool_defs = await capability.prepare_output_tools(cap_ctx, tool_defs)
        return tool_defs

    def get_default_run_id(self) -> str | None:
        for capability in reversed(self._dispatch):
            if (run_id := capability.get_default_run_id()) is not None:
                return run_id
        return None

    async def _get_default_run_id_for_inputs(self, get_inputs: _RunIdInputsFactory) -> str | None:
        if type(self).get_default_run_id is not CombinedCapability.get_default_run_id:
            return self.get_default_run_id()
        for capability in reversed(self._dispatch):
            if (run_id := await capability._get_default_run_id_for_inputs(get_inputs)) is not None:
                return run_id
        return None

    def get_default_conversation_id(self, run_id: str) -> str | None:
        for capability in reversed(self._dispatch):
            if (conversation_id := capability.get_default_conversation_id(run_id)) is not None:
                return conversation_id
        return None

    def get_run_clock(self) -> Callable[[], datetime] | None:
        for capability in reversed(self._dispatch):
            if (clock := capability.get_run_clock()) is not None:
                return clock
        return None

    async def get_run_start_time(self, run_id: str) -> datetime | None:
        for capability in reversed(self._dispatch):
            if (start := await capability.get_run_start_time(run_id)) is not None:
                return start
        return None

    # --- Run lifecycle hooks ---

    async def before_run(
        self,
        ctx: RunContext[AgentDepsT],
    ) -> None:
        for capability in self._dispatch:
            if (cap_ctx := _ctx_for_active_cap(capability, ctx)) is not None:
                await capability.before_run(cap_ctx)

    def _prepare_run_context(self, ctx: RunContext[AgentDepsT]) -> None:
        for capability in self._dispatch:
            capability._prepare_run_context(ctx)

    def _prepare_event_dispatch(self, ctx: RunContext[AgentDepsT], event: AgentStreamEvent) -> None:
        for capability in self._dispatch:
            capability._prepare_event_dispatch(ctx, event)

    async def after_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        for capability in reversed(self._dispatch):
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
        for capability in reversed(self._dispatch):
            if _ctx_for_active_cap(capability, ctx) is not None:
                chain = _make_run_wrap(capability, ctx, chain)
        return await chain()

    async def on_run_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        error: BaseException,
    ) -> AgentRunResult[Any]:
        for capability in reversed(self._dispatch):
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
        for capability in self._dispatch:
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
        for capability in reversed(self._dispatch):
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
        for capability in reversed(self._dispatch):
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
        try:
            for capability in reversed(self._dispatch):
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
        finally:
            # The raised error's traceback holds this frame; retaining error would keep ctx.deps in a cycle.
            del error

    # --- Event hooks ---

    async def on_event(self, ctx: RunContext[AgentDepsT], *, event: AgentStreamEvent) -> None:
        for capability in self._dispatch:
            # Ask against the event a capability would actually see: an earlier capability's
            # (deprecated) replacement is what `Hooks.on_event` picks up and filters on, so testing
            # the original here would skip a listener registered for the replacement's type.
            current = ctx._event_stream_replacements.get(id(event), event)  # pyright: ignore[reportPrivateUsage]
            if capability.listens_to(current) and (cap_ctx := _ctx_for_active_cap(capability, ctx)) is not None:
                await capability.on_event(cap_ctx, event=event)
        # A `CombinedCapability` subclass can carry marked listeners of its own; dispatch them
        # after the children's, matching the combination order used by the other hooks.
        await super().on_event(ctx, event=event)

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        wrapped_streams = [stream]
        for capability in reversed(self._dispatch):
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
        for capability in self._dispatch:
            if (cap_ctx := _ctx_for_active_cap(capability, ctx)) is not None:
                messages = request_context.messages
                request_context = await _request_hook_held('before_model_request', capability, cap_ctx, request_context)
                keep_mirroring(messages, request_context)
        return request_context

    async def prepare_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        for capability in self._dispatch:
            if (cap_ctx := _ctx_for_active_cap(capability, ctx)) is not None:
                request_context = await _request_hook_held(
                    'prepare_model_request', capability, cap_ctx, request_context
                )
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        for capability in reversed(self._dispatch):
            if (cap_ctx := _ctx_for_active_cap(capability, ctx)) is not None:
                response = await _after_model_request_held(capability, cap_ctx, request_context, response)
        return response

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: Callable[[ModelRequestContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        chain = handler
        for capability in reversed(self._dispatch):
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
        try:
            for capability in reversed(self._dispatch):
                if not capability._has_on_model_request_error:
                    continue
                cap_ctx = _ctx_for_active_cap(capability, ctx)
                if cap_ctx is None or not capability._handles_model_request_error(cap_ctx, request_context):
                    continue
                said = _what_it_says(error, copy=True)
                try:
                    recovered = await capability.on_model_request_error(
                        cap_ctx, request_context=request_context, error=error
                    )
                except RetryModelRequest:
                    # Control flow, not a replacement error: a capability asking for another attempt has
                    # answered for the whole chain. Handing it to the next capability as `error` would let
                    # an outer one recover from — or re-wrap — the request for another attempt.
                    raise
                except Exception as new_error:
                    _hold_raised_from_error_hook(cap_ctx, new_error, handed=error, said=said)
                    error = new_error
                    continue
                if (held := _held_to_ceiling(cap_ctx)) is not None:
                    raise _writes_model_turn(held, 'return a response from `on_model_request_error`')
                return recovered
            raise error
        finally:
            # The raised error's traceback holds this frame; retaining error would keep ctx.deps in a cycle.
            del error

    # --- Tool validate lifecycle hooks ---

    async def before_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: str | dict[str, Any],
    ) -> str | dict[str, Any]:
        for capability in self._dispatch:
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
        for capability in reversed(self._dispatch):
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
        for capability in reversed(self._dispatch):
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
        for capability in reversed(self._dispatch):
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
        for capability in self._dispatch:
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
        for capability in reversed(self._dispatch):
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
        for capability in reversed(self._dispatch):
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
        try:
            for capability in reversed(self._dispatch):
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
        finally:
            # The raised error's traceback holds this frame; retaining error would keep ctx.deps in a cycle.
            del error

    # --- Output validate lifecycle hooks ---

    async def before_output_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: RawOutput,
    ) -> RawOutput:
        for capability in self._dispatch:
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
        for capability in reversed(self._dispatch):
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
        for capability in reversed(self._dispatch):
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
        for capability in reversed(self._dispatch):
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
        for capability in self._dispatch:
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
        for capability in reversed(self._dispatch):
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
        for capability in reversed(self._dispatch):
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
        for capability in reversed(self._dispatch):
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
        for capability in self._dispatch:
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


def _take_decision(
    capability: AbstractCapability[AgentDepsT],
    pending: dict[int, list[AbstractCapability[AgentDepsT] | None]],
) -> AbstractCapability[AgentDepsT] | None:
    decisions = pending[id(capability)]
    return decisions.pop(0) if len(decisions) > 1 else decisions[0]


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
        cap_ctx = _ctx_for_cap(cap, ctx)
        held = _held_to_ceiling(cap_ctx)
        if held is None:
            return await cap.wrap_model_request(cap_ctx, request_context=request_context, handler=inner)
        handler = _HeldHandler(held, inner, request_context)
        try:
            response = await cap.wrap_model_request(cap_ctx, request_context=request_context, handler=handler)
        except SkipModelRequest as e:
            if handler.handed_on(e):
                raise
            raise _writes_model_turn(held, 'raise `SkipModelRequest` with a response') from e
        except ModelRetry as e:
            if not handler.handed_on(e):
                # The retry keeps the handler's response in history, so it can't have changed it either.
                _check_handler_responses_unchanged(held, handler.returned)
                hold_feedback(e, held)
            raise
        if not any(response is returned for returned, _ in handler.returned):
            raise _writes_model_turn(held, "return a response from `wrap_model_request` other than its handler's")
        _check_handler_responses_unchanged(held, handler.returned)
        return response

    return wrapped


class _HeldHandler:
    """The handler a capability below `'operator'` gets in `wrap_model_request`, recording what it hands back.

    What the handler returned or raised is the model's or an inner capability's, not the capability's own, and what
    the capability added to the request before calling it is held as `before_model_request`'s additions are.
    """

    def __init__(
        self,
        held: tuple[str, Authority],
        inner: Callable[[ModelRequestContext], Awaitable[ModelResponse]],
        request_context: ModelRequestContext,
    ) -> None:
        self._held = held
        self._inner = inner
        self._snapshot = snapshot_messages(request_context.messages)
        self._instructions = copy_said(request_context.model_request_parameters.instruction_parts or [])
        # Each response with a copy of its parts, so a change made to it in place isn't taken for the handler's.
        self.returned: list[tuple[ModelResponse, Sequence[ModelResponsePart]]] = []
        # Each error with what it told the model when raised, so one changed before it's re-raised isn't handed on.
        self._raised: list[tuple[BaseException, object]] = []

    async def __call__(self, request_context: ModelRequestContext) -> ModelResponse:
        hold_added_messages(request_context.messages, self._snapshot, *self._held)
        _check_instructions_unchanged(self._held, self._instructions, request_context)
        try:
            response = await self._inner(request_context)
        except BaseException as e:
            self._raised.append((e, _what_it_says(e, copy=True)))
            raise
        self.returned.append((response, copy_said(response.parts)))
        return response

    def handed_on(self, error: BaseException) -> bool:
        """Whether `error` is one the handler raised, unchanged."""
        return any(error is raised and says_the_same(_what_it_says(error), said) for raised, said in self._raised)


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

    def bind(capability: AbstractCapability[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        if is_deferred_group(capability):
            # Each child of a deferred group binds in its own tier, as it would outside the group.
            return bind_capabilities_tier(capability, agent, innermost=innermost)
        return (
            capability._keep_authority(capability.for_agent(agent))  # pyright: ignore[reportPrivateUsage]
            if is_innermost(capability) == innermost
            else capability
        )

    branches = combined.capabilities
    new_caps = [bind(c) for c in branches]
    if all(new is old for new, old in zip(new_caps, branches, strict=True)):
        return combined
    return combined._rebound(new_caps)  # pyright: ignore[reportPrivateUsage]


def is_deferred_group(capability: AbstractCapability[Any]) -> TypeIs[CombinedCapability[Any]]:
    """Whether `capability` is a deferred group rather than a plain grouping.

    A deferred group is loaded and replaced as a whole, so it is registered itself and an enclosing
    container keeps it as one branch instead of merging its children in. Its children are still
    dispatched one by one, each at its own ordering position (see `_GroupMember`). One without an
    `id` is kept too, so the run rejects the missing `id` rather than silently merging its children
    in, undeferred.
    """
    return isinstance(capability, CombinedCapability) and capability.defer_loading is True


@dataclass
class _GroupMember(WrapperCapability[AgentDepsT]):
    """A child of a deferred group, dispatched at its own ordering position.

    It carries the group's `id` and deferral, so every per-capability gate (hooks, settings,
    native tools) skips it until the group is loaded. The group itself is what's
    registered, so this registers nothing: `apply` goes straight to the child, which also gives
    the ordering pass the child's own ordering.
    """

    group: CombinedCapability[AgentDepsT] = field(kw_only=True, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.id = self.group.id
        self.defer_loading = True

    def apply(self, visitor: Callable[[AbstractCapability[AgentDepsT]], None]) -> None:
        self.wrapped.apply(visitor)


def _settings_entry(
    capability: AbstractCapability[AgentDepsT],
    cap_settings: ModelSettings | Callable[[RunContext[AgentDepsT]], ModelSettings],
) -> ModelSettings | Callable[[RunContext[AgentDepsT]], ModelSettings]:
    """A child's settings as an entry in the merge chain, evaluated against the child's own context."""
    if capability.defer_loading is True:
        # Request-only settings can be lazy without changing prompt/tool schemas.
        # Keep them in place so loaded capabilities preserve merge order.
        def deferred_settings(ctx: RunContext[AgentDepsT]) -> ModelSettings:
            cap_ctx = _ctx_for_active_cap(capability, ctx)
            if cap_ctx is None:
                return ModelSettings()
            if callable(cap_settings):
                return cap_settings(cap_ctx)
            return cap_settings

        return deferred_settings
    if callable(cap_settings):
        # The settings function is the capability's code, held to its authority if it enqueues.
        def attributed_settings(ctx: RunContext[AgentDepsT]) -> ModelSettings:
            return cap_settings(_ctx_for_cap(capability, ctx))

        return attributed_settings
    return cap_settings


def _ctx_for_cap(capability: AbstractCapability[AgentDepsT], ctx: RunContext[AgentDepsT]) -> RunContext[AgentDepsT]:
    return _replace_capability_context(
        ctx, capability=capability, capability_active=_capability_active(capability, ctx)
    )


def _ctx_for_active_cap(
    capability: AbstractCapability[AgentDepsT], ctx: RunContext[AgentDepsT]
) -> RunContext[AgentDepsT] | None:
    capability_active = _capability_active(capability, ctx)
    if capability.defer_loading is True and not capability_active:
        return None
    return _replace_capability_context(ctx, capability=capability, capability_active=capability_active)


def _replace_capability_context(
    ctx: RunContext[AgentDepsT], *, capability: AbstractCapability[AgentDepsT], capability_active: bool
) -> RunContext[AgentDepsT]:
    if isinstance(capability, _GroupMember):
        # A group member is itself to its hooks and events; only its gate is the group's.
        capability = capability.wrapped
    if type(ctx) is RunContext:
        cap_ctx = copy(ctx)
        cap_ctx.capability_active = capability_active
        cap_ctx._capability = capability  # pyright: ignore[reportPrivateUsage]
        return cap_ctx
    # Subclasses can rely on reconstruction, e.g. TemporalRunContext's field availability guards.
    return replace(ctx, capability_active=capability_active, _capability=capability)


def _capability_active(capability: AbstractCapability[AgentDepsT], ctx: RunContext[AgentDepsT]) -> bool:
    """Whether this capability may act on the current step.

    Activity, not loading: an always-on capability is active for the whole run without ever being
    loaded, which is why the deferred branch below is the only one that consults history.

    During tool-call dispatch that history question is answered against the provider that served the
    response, via `_dispatch_active_capability_ids` — the same set `RunContext.is_tool_available`
    authorizes the call from. Reading the narrower `active_capability_ids` here would let a tool the
    gate admits on anchored evidence execute without its owner's `prepare_tools` ever running.
    """
    if capability.defer_loading is not True:
        return True

    # Deferred capabilities are required to have an explicit `id` (enforced in
    # `_build_run_capabilities`), which is also the key they're registered under, so we read
    # it directly rather than resolving the instance back to its run-local registry id.
    return capability.id is not None and ctx._is_deferred_capability_dispatch_active(capability.id)  # pyright: ignore[reportPrivateUsage]


async def _request_hook_held(
    hook: Literal['before_model_request', 'prepare_model_request'],
    capability: AbstractCapability[AgentDepsT],
    cap_ctx: RunContext[AgentDepsT],
    request_context: ModelRequestContext,
) -> ModelRequestContext:
    """`capability`'s `before_model_request` or `prepare_model_request`, held to its authority below `'operator'`.

    It can't write the model's turn with `SkipModelRequest` or change the instructions, its `ModelRetry` feedback is
    its own text, and what it adds to the request's messages is held as if it had enqueued it.
    """
    held = _held_to_ceiling(cap_ctx)
    run_hook = capability.before_model_request if hook == 'before_model_request' else capability.prepare_model_request
    if held is None or getattr(type(capability), hook) is getattr(AbstractCapability, hook):
        return await run_hook(cap_ctx, request_context)
    snapshot = snapshot_messages(request_context.messages)
    instructions = copy_said(request_context.model_request_parameters.instruction_parts or [])
    try:
        request_context = await run_hook(cap_ctx, request_context)
    except SkipModelRequest as e:
        raise _writes_model_turn(held, 'raise `SkipModelRequest` with a response') from e
    except ModelRetry as e:
        # What it added is already in the history the deprecated mirroring appends to, and the retry keeps it.
        hold_added_messages(request_context.messages, snapshot, *held)
        _check_instructions_unchanged(held, instructions, request_context)
        hold_feedback(e, held)
        raise
    hold_added_messages(request_context.messages, snapshot, *held)
    _check_instructions_unchanged(held, instructions, request_context)
    return request_context


def _check_instructions_unchanged(
    held: tuple[str, Authority], instructions: Sequence[InstructionPart], request_context: ModelRequestContext
) -> None:
    """Raise if a capability below `'operator'` added to or changed the request's instructions.

    The model reads them as the operator's, so there's no lower authority to hold them to. Removing them adds nothing.
    """
    if any(
        not any(says_the_same(part, old) for old in instructions)
        for part in request_context.model_request_parameters.instruction_parts or ()
    ):
        capability_id, ceiling = held
        raise UserError(
            f"Capability {capability_id!r} has {ceiling!r} authority, so it cannot add to or change the request's "
            "instructions, which the model reads as the operator's. Only a capability with operator authority can "
            'write instructions.'
        )


async def _after_model_request_held(
    capability: AbstractCapability[AgentDepsT],
    cap_ctx: RunContext[AgentDepsT],
    request_context: ModelRequestContext,
    response: ModelResponse,
) -> ModelResponse:
    """`capability`'s `after_model_request`, held to its authority when that's below `'operator'`.

    It can't replace the response, change its parts in place or skip with a response of its own, and its `ModelRetry`
    feedback is its own text.
    """
    held = _held_to_ceiling(cap_ctx)
    if held is None or type(capability).after_model_request is AbstractCapability.after_model_request:
        return await capability.after_model_request(cap_ctx, request_context=request_context, response=response)
    parts = copy_said(response.parts)
    try:
        returned = await capability.after_model_request(cap_ctx, request_context=request_context, response=response)
    except SkipModelRequest as e:
        raise _writes_model_turn(held, 'raise `SkipModelRequest` with a response') from e
    except ModelRetry as e:
        # The retry keeps the response in history, so it can't have changed it either.
        _check_handler_responses_unchanged(held, [(response, parts)], 'in `after_model_request`')
        hold_feedback(e, held)
        raise
    # A durable hook's return comes back through its engine as a new object, so what the model reads is compared.
    if returned is not response and not says_the_same(returned.parts, parts):
        raise _writes_model_turn(held, 'return a different response from `after_model_request`')
    _check_handler_responses_unchanged(held, [(response, parts)], 'in `after_model_request`')
    return returned


def _what_it_says(error: BaseException, *, copy: bool = False) -> object:
    """What an error tells the model: a skip's response parts or a retry's feedback, or `None`."""
    if isinstance(error, SkipModelRequest):
        return copy_said(error.response.parts) if copy else error.response.parts
    if isinstance(error, ModelRetry):
        return error.message
    return None


def _check_handler_responses_unchanged(
    held: tuple[str, Authority],
    responses: Sequence[tuple[ModelResponse, Sequence[ModelResponsePart]]],
    where: str = 'its handler returned in `wrap_model_request`',
) -> None:
    """Raise if a capability below operator authority changed a response's parts in place since they were copied."""
    if any(not says_the_same(response.parts, parts) for response, parts in responses):
        raise _writes_model_turn(held, f"change the response's parts {where}")


def _hold_raised_from_error_hook(
    cap_ctx: RunContext[Any], raised: Exception, *, handed: Exception, said: object
) -> None:
    """Hold what a capability's `on_model_request_error` raised itself, rather than handed on, to its authority.

    Re-raising the error it was `handed` hands it on, unless the hook changed what it says (`said` before the hook ran).
    """
    if raised is handed and says_the_same(_what_it_says(raised), said):
        return
    if (held := _held_to_ceiling(cap_ctx)) is None:
        return
    if isinstance(raised, SkipModelRequest):
        raise _writes_model_turn(held, 'raise `SkipModelRequest` with a response') from raised
    if isinstance(raised, ModelRetry):
        hold_feedback(raised, held)


def _held_to_ceiling(cap_ctx: RunContext[Any]) -> tuple[str, Authority] | None:
    """The capability running a hook with `cap_ctx` and its authority, when that's below `'operator'`.

    `None` for a capability at `'operator'`, and for an app-facing callback capability such as `Hooks`, whose
    functions are the application's own.
    """
    capability = cap_ctx._capability  # pyright: ignore[reportPrivateUsage]
    if capability is None or capability._emits_app_events:  # pyright: ignore[reportPrivateUsage]
        return None
    ceiling = capability._effective_authority  # pyright: ignore[reportPrivateUsage]
    if ceiling == 'operator':
        return None
    capability_id = cap_ctx._calling_capability_id() or type(capability).__name__  # pyright: ignore[reportPrivateUsage]
    return capability_id, ceiling


def _writes_model_turn(held: tuple[str, Authority], what: str) -> UserError:
    capability_id, ceiling = held
    return UserError(
        f'Capability {capability_id!r} has {ceiling!r} authority, so it cannot {what}, which the model reads as its '
        "own turn. Only a capability with operator authority can write the model's turns."
    )
