from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

from typing_extensions import TypeIs

from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability


def _is_capability(value: object) -> TypeIs[AbstractCapability[AgentDepsT]]:
    return isinstance(value, AbstractCapability)


class _RunCapabilityResolutions:
    def __init__(self) -> None:
        self.resolved: dict[int, object] = {}


_current_resolutions: ContextVar[_RunCapabilityResolutions | None] = ContextVar('_current_resolutions', default=None)
_setup_error_dispatch: ContextVar[object | None] = ContextVar('_setup_error_dispatch', default=None)


@contextmanager
def capture_run_capability_resolutions() -> Generator[_RunCapabilityResolutions, None, None]:
    resolutions = _RunCapabilityResolutions()
    token: Token[_RunCapabilityResolutions | None] = _current_resolutions.set(resolutions)
    try:
        yield resolutions
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


async def resolve_capability_for_run(
    capability: AbstractCapability[AgentDepsT], ctx: RunContext[AgentDepsT]
) -> AbstractCapability[AgentDepsT]:
    resolved = await capability.for_run(ctx)
    if resolutions := _current_resolutions.get():
        resolutions.resolved[id(capability)] = resolved
    return resolved


def replace_resolved_run_capabilities(
    capability: AbstractCapability[AgentDepsT], resolutions: _RunCapabilityResolutions
) -> AbstractCapability[AgentDepsT]:
    def replace(cap: AbstractCapability[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        resolved = resolutions.resolved.get(id(cap))
        if _is_capability(resolved):
            return resolved
        return cap

    return capability.visit_and_replace(replace) or capability
