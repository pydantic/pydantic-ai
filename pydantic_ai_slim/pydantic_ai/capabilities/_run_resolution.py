from __future__ import annotations

from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Generic

from typing_extensions import TypeIs

from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability


def _is_capability(value: object) -> TypeIs[AbstractCapability[AgentDepsT]]:
    return isinstance(value, AbstractCapability)


@dataclass
class PartialRunCapabilityResolution(Generic[AgentDepsT]):
    child: AbstractCapability[AgentDepsT]
    wrap: Callable[[AbstractCapability[AgentDepsT]], AbstractCapability[AgentDepsT]]


class RunCapabilityResolutions:
    def __init__(self) -> None:
        self.resolved: dict[int, list[object | None]] = {}
        self.capabilities_by_id: dict[int, object] = {}
        self.layers: list[RunCapabilityResolutions] = []

    def reserve(self, capability: AbstractCapability[AgentDepsT]) -> int:
        # Keep every id key's original object alive until reconstruction finishes.
        self.capabilities_by_id[id(capability)] = capability
        occurrences = self.resolved.setdefault(id(capability), [])
        occurrences.append(None)
        return len(occurrences) - 1

    def record(self, capability: AbstractCapability[AgentDepsT], occurrence: int, resolved: object) -> None:
        self.resolved[id(capability)][occurrence] = resolved


_current_resolutions: ContextVar[RunCapabilityResolutions | None] = ContextVar('_current_resolutions', default=None)
# A capability may appear more than once and resolve concurrently. Keep the active occurrence
# in task-local state so a child created mid-resolution can be attached to the right parent.
_current_resolution: ContextVar[tuple[RunCapabilityResolutions, object, int] | None] = ContextVar(
    '_current_resolution', default=None
)
_setup_error_dispatch: ContextVar[object | None] = ContextVar('_setup_error_dispatch', default=None)
setup_cleanup_reconstruction_active: ContextVar[bool] = ContextVar('setup_cleanup_reconstruction_active', default=False)


@contextmanager
def capture_run_capability_resolutions(
    resolutions: RunCapabilityResolutions | None = None,
) -> Generator[RunCapabilityResolutions, None, None]:
    captured = resolutions or RunCapabilityResolutions()
    token: Token[RunCapabilityResolutions | None] = _current_resolutions.set(captured)
    try:
        yield captured
    finally:
        _current_resolutions.reset(token)


@contextmanager
def setup_error_dispatch_scope(ctx: RunContext[Any]) -> Generator[None, None, None]:
    # Capability-specific RunContext copies share this mapping, while nested Agent runs allocate
    # their own. Keying setup dispatch to the mapping keeps the mode attached to this run instead of
    # leaking into an unrelated run started by an awaited error hook.
    run_capabilities = ctx._run_capabilities_by_id  # pyright: ignore[reportPrivateUsage]
    assert run_capabilities is not None
    token: Token[object | None] = _setup_error_dispatch.set(run_capabilities)
    try:
        yield
    finally:
        _setup_error_dispatch.reset(token)


def is_setup_error_dispatching(ctx: RunContext[Any]) -> bool:
    run_capabilities = ctx._run_capabilities_by_id  # pyright: ignore[reportPrivateUsage]
    active_run_capabilities = _setup_error_dispatch.get()
    return active_run_capabilities is not None and run_capabilities is active_run_capabilities


@contextmanager
def reconstructing_setup_cleanup() -> Generator[None, None, None]:
    """Keep resolved instances available for cleanup without reevaluating their ordering."""
    token: Token[bool] = setup_cleanup_reconstruction_active.set(True)
    try:
        yield
    finally:
        setup_cleanup_reconstruction_active.reset(token)


def resolve_capability_for_run(
    capability: AbstractCapability[AgentDepsT], ctx: RunContext[AgentDepsT]
) -> Awaitable[AbstractCapability[AgentDepsT]]:
    resolutions = _current_resolutions.get()
    occurrence = resolutions.reserve(capability) if resolutions is not None else None

    async def resolve() -> AbstractCapability[AgentDepsT]:
        token: Token[tuple[RunCapabilityResolutions, object, int] | None] | None = None
        if resolutions is not None and occurrence is not None:
            token = _current_resolution.set((resolutions, capability, occurrence))
        try:
            resolved = await capability.for_run(ctx)
            if resolutions is not None and occurrence is not None:
                resolutions.record(capability, occurrence, resolved)
            return resolved
        finally:
            if token is not None:
                _current_resolution.reset(token)

    return resolve()


def record_partial_run_capability_resolution(
    capability: AbstractCapability[AgentDepsT],
    resolved: AbstractCapability[AgentDepsT] | PartialRunCapabilityResolution[AgentDepsT],
) -> None:
    """Retain an accessible child when its enclosing `for_run` has not finished yet."""
    active = _current_resolution.get()
    if active is not None:
        resolutions, resolving, occurrence = active
        if resolving is capability:  # pragma: no branch  # Only the active capability records a partial resolution.
            resolutions.record(capability, occurrence, resolved)


def replace_resolved_run_capabilities(
    capability: AbstractCapability[AgentDepsT], resolutions: RunCapabilityResolutions
) -> AbstractCapability[AgentDepsT]:
    occurrences_seen: dict[int, int] = {}

    def is_partial_resolution(value: object) -> TypeIs[PartialRunCapabilityResolution[AgentDepsT]]:
        return isinstance(value, PartialRunCapabilityResolution)

    def replace(cap: AbstractCapability[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        capability_id = id(cap)
        occurrence = occurrences_seen.get(capability_id, 0)
        occurrences_seen[capability_id] = occurrence + 1
        resolved_occurrences = resolutions.resolved.get(capability_id, [])
        resolved = resolved_occurrences[occurrence] if occurrence < len(resolved_occurrences) else None
        if is_partial_resolution(resolved):
            return resolved.wrap(rebuild(resolved.child))
        if _is_capability(resolved):
            return resolved
        return cap

    def rebuild(cap: AbstractCapability[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        # A container subclass can replace itself in `for_run`; its visitor only visits children.
        # Take a completed root record before descending, including roots inside a partial dynamic.
        capability_id = id(cap)
        occurrence = occurrences_seen.get(capability_id, 0)
        resolved_occurrences = resolutions.resolved.get(capability_id, [])
        resolved = resolved_occurrences[occurrence] if occurrence < len(resolved_occurrences) else None
        if _is_capability(resolved):
            occurrences_seen[capability_id] = occurrence + 1
            return resolved
        return cap.visit_and_replace(replace) or cap

    return rebuild(capability)
