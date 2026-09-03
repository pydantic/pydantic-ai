from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any, cast

from pydantic_ai.capabilities import AbstractCapability, AgentCapability, WrapperCapability
from pydantic_ai.capabilities._sandbox import active_leaves
from pydantic_ai.exceptions import UserError
from pydantic_ai.sandboxes import SandboxBackend, SandboxRef, UnavailableSandbox


class WorkflowSandboxGuard(UnavailableSandbox):
    """A backend that explains why workflow-side sandbox operations are unsafe."""

    def __init__(self, reason: str, *, sandbox_id: str) -> None:
        super().__init__(reason)
        self.sandbox_id = sandbox_id


def _supplies_sandbox(capability: AbstractCapability[Any]) -> bool:
    if isinstance(capability, WrapperCapability):
        return _supplies_sandbox(capability.wrapped)
    return type(capability).acquire_sandbox is not AbstractCapability.acquire_sandbox and inspect.iscoroutinefunction(
        capability.acquire_sandbox
    )


def contributes_sandbox(capability: AbstractCapability[Any]) -> bool:
    """Whether the capability tree statically declares an async sandbox supplier.

    The deprecated durable-agent wrappers reject sandbox-supplying capabilities up front: an
    async `acquire_sandbox` performs I/O and the wrappers have no way to route it into a durable
    unit. A sync override is allowed because its contract forbids I/O and requires deterministic output.
    A supplier produced at run time by a capability function is not visible here.
    """
    return any(_supplies_sandbox(leaf) for leaf in active_leaves(capability))


def sandbox_contribution_error(*, run_location: str, sandbox_constraint: str) -> str:
    return (
        f'A capability that supplies a sandbox cannot run {run_location}: '
        f'{sandbox_constraint}. Create the sandbox outside the workflow and pass a `SandboxRef` to the run instead.'
    )


def live_sandbox_error(*, run_location: str, sandbox_constraint: str) -> str:
    return (
        f'A live sandbox handle cannot be passed {run_location}: {sandbox_constraint}. '
        'Pass a `SandboxRef` instead and attach a capability whose `get_sandbox` can connect to it.'
    )


def guard_workflow_sandbox(
    sandbox: SandboxBackend | SandboxRef | None,
    capabilities: Sequence[AgentCapability[Any]] | None,
    *,
    static_contributes_sandbox: bool,
    contribution_error: str,
    live_error: str,
) -> SandboxRef | UnavailableSandbox | None:
    if sandbox is None and (
        static_contributes_sandbox
        or any(
            isinstance(capability, AbstractCapability)
            and contributes_sandbox(cast('AbstractCapability[Any]', capability))
            for capability in capabilities or ()
        )
    ):
        raise UserError(contribution_error)
    if sandbox is not None and not isinstance(sandbox, (SandboxRef, UnavailableSandbox)):
        raise UserError(live_error)
    return sandbox
