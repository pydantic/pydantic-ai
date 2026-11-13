from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.durable_exec._capability_operation import collect_capability_operations
from pydantic_ai.exceptions import UserError
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AbstractToolset, AgentToolset, CombinedToolset
from pydantic_ai.toolsets._deferred_capability_loader import registered_beneath
from pydantic_ai.toolsets._dynamic import DynamicToolset

from .abstract import AbstractCapability, CapabilityOrdering, _combination_roots  # pyright: ignore[reportPrivateUsage]
from .combined import CombinedCapability
from .wrapper import WrapperCapability

CapabilityFunc: TypeAlias = Callable[
    [RunContext[AgentDepsT]],
    AbstractCapability[AgentDepsT] | None | Awaitable[AbstractCapability[AgentDepsT] | None],
]
"""A sync/async function which takes a run context and returns a capability."""


@dataclass
class DynamicCapability(AbstractCapability[AgentDepsT]):
    """A capability that builds another capability dynamically using a function that takes the run context.

    By default, the factory is called once per agent run from
    [`for_run`][pydantic_ai.capabilities.AbstractCapability.for_run]. The returned
    capability's instructions, model settings, native tools, and hooks flow through
    normally; its toolset is exposed through a stable dynamic toolset contributed at
    agent construction time, which reuses the run's resolved capability instance.

    With `per_run_step=True`, the factory is also called again ahead of each run step,
    and may only return [on-demand capabilities](../capabilities/on-demand.md)
    (`defer_loading=True`, or a `CombinedCapability` of them). A capability that first
    appears at a later step joins the catalog as an appended change, and one that stops
    being returned leaves it the same way; see
    [Capabilities that change during a run](../capabilities/on-demand.md#per-step).

    Under durable execution, a stable `id` is required on `DynamicCapability`: it
    names the durable units (activities/steps/tasks) that list and call the
    contributed tools. The factory itself runs in workflow/flow code, which durable
    engines re-execute on replay, recovery, or flow retry, so it must be
    deterministic given the run's dependencies; leave I/O to the toolset it
    returns, whose use is checkpointed inside the durable units. In-process
    engines (DBOS, Prefect) reuse the run's resolved capability inside those
    units; Temporal re-runs the factory inside its activities (the activity
    boundary can't carry the resolved instance).

    Pass a [`CapabilityFunc`][pydantic_ai.capabilities.CapabilityFunc] directly
    to `Agent(capabilities=[...])` or `agent.run(capabilities=[...])` and it
    will be wrapped in a `DynamicCapability` automatically.

    `defer_loading` on the wrapper itself is rejected because `for_run` replaces
    the wrapper with the factory's return value. Set it on the returned
    capability instead.
    For history replay, set a stable `id` on the capability the factory returns
    rather than on the wrapper.
    """

    capability_func: CapabilityFunc[AgentDepsT]
    """The function that takes the run context and returns a capability or `None`."""

    per_run_step: bool = field(default=False, kw_only=True)
    """Whether to call the function again ahead of each run step, like [`DynamicToolset(per_run_step=True)`][pydantic_ai.toolsets.DynamicToolset].

    The function may then only return on-demand capabilities (`defer_loading=True`), which join and
    leave the deferred capability catalog as the run goes. Returning the same instance as on the
    previous step keeps it, and the state and connections it holds.
    """

    def __post_init__(self) -> None:
        # Forwarding this to the returned capability would be ambiguous: the factory
        # may return None, or a capability that deliberately chose its own loading state/id.
        if self.defer_loading is True:
            raise UserError(
                '`defer_loading` is not supported on `DynamicCapability` — '
                'set it on the capability the factory returns instead.'
            )
        # Built eagerly: this single instance's identity is what durability engines register
        # against at agent construction and match at run time, so it must never fork.
        self._toolset = DynamicToolset(toolset_func=self._resolve_toolset, per_run_step=self.per_run_step, id=self.id)

    def get_toolset(self) -> DynamicToolset[AgentDepsT]:
        return self._toolset

    async def _resolve_toolset(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        # Inside a run, `for_run` has already resolved this capability once: reuse that
        # instance so hooks, instructions, and tools all observe the same per-run state,
        # and so the factory keeps its once-per-run contract. Read the registry via
        # `__dict__` because contexts rehydrated across a durable boundary (e.g.
        # `TemporalRunContext` inside an activity) deliberately don't carry it and raise
        # on regular attribute access.
        registry: dict[str, AbstractCapability[AgentDepsT]] = ctx.__dict__.get('capabilities') or {}
        for capability in registry.values():
            if isinstance(capability, PerStepDynamicCapability) and capability.dynamic_toolset is self._toolset:
                return await capability.step_toolset(ctx)
            if isinstance(capability, ResolvedDynamicCapability) and capability.dynamic_toolset is self._toolset:
                return await _evaluate_agent_toolset(capability.wrapped.get_toolset(), ctx)
            if capability is self:
                # `for_run` kept this capability as-is because the factory returned `None`.
                return None
        # No resolved instance to reuse: the toolset is being used standalone, or inside a
        # durable unit (activity) whose deserialized context carries no capability registry —
        # re-resolve the factory there, where its I/O is allowed.
        capability = await self._resolve_capability(ctx)
        if capability is None:
            return None
        if self.per_run_step:
            # Checked as in the run, which a deterministic factory has already passed. Owned by the capabilities it
            # returned, as in the run, so their tools stay hidden until they're loaded.
            _check_per_step_capability(capability)
            capability = CombinedCapability[AgentDepsT]([capability])
        return await _evaluate_agent_toolset(capability.get_toolset(), ctx)

    async def _call_factory(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT] | None:
        capability = self.capability_func(ctx)
        if inspect.isawaitable(capability):
            capability = await capability
        return capability

    async def _resolve_capability(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT] | None:
        return await _resolve_returned(await self._call_factory(ctx), ctx)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        if self.per_run_step:
            capability = PerStepDynamicCapability(
                wrapped=CombinedCapability[AgentDepsT]([]), dynamic_toolset=self.get_toolset(), source=self
            )
            await capability.for_run_step(ctx)
            return capability
        capability = await self._resolve_capability(ctx)
        if capability is None:
            return self
        return ResolvedDynamicCapability(wrapped=capability, dynamic_toolset=self.get_toolset())


@dataclass
class ResolvedDynamicCapability(WrapperCapability[AgentDepsT]):
    """The per-run replacement for a [`DynamicCapability`][pydantic_ai.capabilities.DynamicCapability].

    Delegates to the factory's resolved capability, except that the resolved capability's own
    toolset contribution is replaced by the `DynamicCapability`'s stable dynamic toolset — the
    one registered with any durable execution engine at agent construction time.
    """

    dynamic_toolset: DynamicToolset[AgentDepsT]

    def get_toolset(self) -> DynamicToolset[AgentDepsT]:
        return self.dynamic_toolset

    def get_ordering(self) -> CapabilityOrdering | None:
        # `CombinedCapability.for_run` re-sorts its (replaced) capabilities, so the resolved
        # capability's ordering constraints must survive the wrapper.
        return self.wrapped.get_ordering()


@dataclass
class PerStepDynamicCapability(ResolvedDynamicCapability[AgentDepsT]):
    """The per-run replacement for a [`DynamicCapability`][pydantic_ai.capabilities.DynamicCapability] with `per_run_step=True`.

    Wraps what the factory returned for the current step in a `CombinedCapability`, so hooks, model settings and
    tools reach the on-demand capabilities of the current step, each gated on whether it's loaded. `for_run_step`
    swaps them for the next step's and updates the run's capability registry in place, where the catalog, the
    `load_capability` tool and `RunContext.load_capability` look them up.
    """

    source: DynamicCapability[AgentDepsT] = field(kw_only=True, repr=False)
    _step: tuple[str | None, int] | None = field(default=None, init=False, repr=False)
    _returned: AbstractCapability[AgentDepsT] | None = field(default=None, init=False, repr=False)
    _toolset_of: AbstractCapability[AgentDepsT] | None = field(default=None, init=False, repr=False)
    _step_toolset: AbstractToolset[AgentDepsT] | None = field(default=None, init=False, repr=False)

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> None:
        """Call the factory for the step `ctx` is on, once per step, and register what it returned in `ctx.capabilities`.

        The first call comes from `DynamicCapability.for_run`, before the run builds its registry from the tree.
        """
        step = (ctx.run_id, ctx.run_step)
        if step == self._step:
            return
        first_step = self._step is None
        self._step = step
        returned = await self.source._call_factory(ctx)  # pyright: ignore[reportPrivateUsage]
        if not first_step and returned is self._returned:
            # The same instance as on the previous step: keep it as resolved, with its state and connections.
            return
        self._returned = returned
        resolved = await _resolve_returned(returned, ctx)
        if resolved is not None:
            _check_per_step_capability(resolved)
        previous = registered_beneath(self)
        self.wrapped = CombinedCapability[AgentDepsT]([resolved] if resolved is not None else [])
        if not first_step:
            _register_beneath(ctx.capabilities, self, previous)

    async def step_toolset(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """The toolset of the current step's capabilities, the same instance until they change."""
        if self._toolset_of is not self.wrapped:
            self._toolset_of = self.wrapped
            self._step_toolset = await _evaluate_agent_toolset(self.wrapped.get_toolset(), ctx)
        return self._step_toolset

    def get_model_settings(self) -> Callable[[RunContext[AgentDepsT]], ModelSettings]:
        # Collected once per run, so it reads the current step's capabilities when called rather than now.
        def model_settings(ctx: RunContext[AgentDepsT]) -> ModelSettings:
            settings = self.wrapped.get_model_settings()
            if callable(settings):
                return settings(ctx)
            return settings or ModelSettings()

        return model_settings


def _check_per_step_capability(capability: AbstractCapability[Any]) -> None:
    """Reject what a per-step factory can't return.

    An always-on capability, and contributions the run only collects when it starts: native tools, toolset wrappers
    (an approval or filtering policy would silently not apply to tools that appear later), and durable operations,
    which a durable engine registers when the agent is built.
    """
    for branch in _combination_roots(capability):
        name = f'{type(branch).__name__}(id={branch.id!r})'
        if branch.defer_loading is not True:
            raise UserError(
                f'A `DynamicCapability` with `per_run_step=True` can only return on-demand capabilities, '
                f'but it returned {name} without `defer_loading=True`.'
            )
        if branch.get_native_tools():
            unsupported = 'native tools'
        elif branch.get_wrapper_toolset(CombinedToolset[Any]([])) is not None:
            unsupported = 'a toolset wrapper (`get_wrapper_toolset`)'
        elif any(collect_capability_operations(leaf) for leaf in [branch, *registered_beneath(branch)]):
            unsupported = 'durable operations'
        else:
            continue
        raise UserError(
            f'A `DynamicCapability` with `per_run_step=True` does not support {unsupported} yet, '
            f'but it returned {name} with them.'
        )


def _register_beneath(
    registry: dict[str, AbstractCapability[Any]],
    capability: AbstractCapability[Any],
    previous: list[AbstractCapability[Any]],
) -> None:
    """Replace what `capability` registered beneath itself, `previous`, with what it registers now, in place.

    The new entries take the place of the old ones, right after `capability` itself, so the catalog keeps its order
    when nothing changed.
    """
    # Imported here: the agent module imports this one.
    from pydantic_ai.agent import (
        _synthetic_capability_id,  # pyright: ignore[reportPrivateUsage]
        _validate_capability_ids,  # pyright: ignore[reportPrivateUsage]
    )

    others = {key: cap for key, cap in registry.items() if not any(cap is old for old in previous)}
    current = registered_beneath(capability)
    taken = others.keys() | _validate_capability_ids([*others.values(), *current])
    entries: dict[str, AbstractCapability[Any]] = {}
    for cap in current:
        entries[cap.id or _synthetic_capability_id(type(cap), taken=taken | entries.keys())] = cap
    registry.clear()
    for key, cap in others.items():
        registry[key] = cap
        if cap is capability:
            registry.update(entries)


async def refresh_per_step_capabilities(ctx: RunContext[AgentDepsT]) -> None:
    """Call the factory of each `DynamicCapability(per_run_step=True)` in the run for the step `ctx` is on."""
    for capability in [cap for cap in ctx.capabilities.values() if isinstance(cap, PerStepDynamicCapability)]:
        await capability.for_run_step(ctx)


def has_per_step_capabilities(capabilities: Iterable[AbstractCapability[Any]]) -> bool:
    """Whether any of `capabilities` is a `DynamicCapability(per_run_step=True)` resolved for a run."""
    return any(isinstance(cap, PerStepDynamicCapability) for cap in capabilities)


async def _resolve_returned(
    capability: AbstractCapability[AgentDepsT] | None, ctx: RunContext[AgentDepsT]
) -> AbstractCapability[AgentDepsT] | None:
    """Bind what a capability function returned to the run's agent, and resolve it for the run."""
    if capability is None:
        return None
    assert ctx.agent is not None, 'CapabilityFunc requires an agent run context'
    capability = capability.for_agent(ctx.agent)
    return await capability.for_run(ctx)


async def _evaluate_agent_toolset(
    toolset: AgentToolset[AgentDepsT] | None, ctx: RunContext[AgentDepsT]
) -> AbstractToolset[AgentDepsT] | None:
    """Normalize a capability's toolset contribution: evaluate the toolset-*function* arm with the run context."""
    if toolset is None or isinstance(toolset, AbstractToolset):
        # Pyright can't narrow Callable type aliases out of unions after an isinstance check
        return toolset  # pyright: ignore[reportUnknownVariableType]
    resolved: AbstractToolset[AgentDepsT] | None | Awaitable[AbstractToolset[AgentDepsT] | None] = toolset(ctx)
    if inspect.isawaitable(resolved):
        resolved = await resolved
    return resolved


def wrap_capability_funcs(
    capabilities: Sequence[AbstractCapability[AgentDepsT] | CapabilityFunc[AgentDepsT]] | None,
) -> list[AbstractCapability[AgentDepsT]]:
    """Wrap any [`CapabilityFunc`][pydantic_ai.capabilities.CapabilityFunc] entries in a `DynamicCapability`."""
    if not capabilities:
        return []
    return [cap if isinstance(cap, AbstractCapability) else DynamicCapability(cap) for cap in capabilities]
