from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from pydantic_ai.capabilities import AbstractCapability, AgentCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.sandboxes import SandboxBackend, SandboxRef, UnavailableSandbox


def _sandbox_suppliers(capability: AbstractCapability[Any]) -> list[AbstractCapability[Any]]:
    """The capabilities in the tree that declare sandbox provider hooks, in chain order.

    Wrapper forwarding represents the wrapped provider once, and the durability capability's own
    routing does not count as a contribution. The check is static: a provider produced at run time
    by a dynamic capability function is not visible here.
    """
    suppliers: list[AbstractCapability[Any]] = []
    seen: set[int] = set()

    def visit(leaf: AbstractCapability[Any]) -> None:
        if leaf.defer_loading is True or id(leaf) in seen:
            return
        seen.add(id(leaf))
        if leaf.has_sandbox_hooks:
            suppliers.append(leaf)

    capability.apply(visit)
    return suppliers


def contributes_sandbox(capability: AbstractCapability[Any]) -> bool:
    """Whether the capability tree contains a sandbox provider.

    The deprecated durable-agent wrappers reject sandbox-contributing capabilities up front:
    running the supplier's lifecycle hooks would be I/O in workflow code, and the wrappers
    have no way to route it into a durable unit. `TemporalDurability` does; see
    Generic contributed-operation dispatch is deliberately absent from these deprecated wrappers.
    """
    return bool(_sandbox_suppliers(capability))


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
