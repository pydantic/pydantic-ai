from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from pydantic_ai.capabilities import AbstractCapability, AgentCapability, WrapperCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.sandboxes import SandboxBackend, SandboxConnector, SandboxRef, UnavailableSandbox

AgentDepsT = TypeVar('AgentDepsT')


@dataclass
class SandboxConnectorsCapability(AbstractCapability[Any]):
    """Internal capability used by deprecated durable-agent wrappers."""

    connectors: Sequence[SandboxConnector]

    def get_sandbox_connectors(self) -> Sequence[SandboxConnector]:
        return self.connectors


def contributes_sandbox(capability: AbstractCapability[Any]) -> bool:
    """Whether the capability tree contains a `get_sandbox` override.

    Durable integrations reject sandbox-contributing capabilities up front: entering the
    contributed context manager would run I/O in workflow code. The check is static, so a
    contributor only produced at run time by a dynamic capability function cannot be caught
    here — the workflow engine then blocks the I/O itself, less legibly.

    `WrapperCapability.get_sandbox` is a pure forwarder and
    `BaseDurabilityCapability.get_sandbox` only suppresses the framework default inside a
    durable context; neither is a user sandbox contribution. `apply()` also visits wrapped
    capabilities, so a real contributor behind a wrapper is still found. A durability
    subclass that overrides `get_sandbox` itself is still treated as a contributor.
    """
    # Import lazily so loading this lightweight workflow guard does not pull the durability
    # capability's model/toolset import graph into Temporal workflow sandbox validation.
    from ._base import BaseDurabilityCapability

    found = False

    def visit(leaf: AbstractCapability[Any]) -> None:
        nonlocal found
        if found:
            return
        get_sandbox = type(leaf).get_sandbox
        found = get_sandbox not in (
            AbstractCapability.get_sandbox,
            WrapperCapability.get_sandbox,
            BaseDurabilityCapability.get_sandbox,
        )

    capability.apply(visit)
    return found


def sandbox_contribution_error(*, run_location: str, sandbox_constraint: str) -> str:
    return (
        f'A capability that contributes a sandbox (overrides `get_sandbox`) cannot run {run_location}: '
        f'{sandbox_constraint}. Create the sandbox outside the workflow and pass a `SandboxRef` to the run instead.'
    )


def live_sandbox_error(*, run_location: str, sandbox_constraint: str, connector_hint: str) -> str:
    return (
        f'A live sandbox handle cannot be passed {run_location}: {sandbox_constraint}. '
        f'Pass a `SandboxRef` instead and {connector_hint}.'
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
            and contributes_sandbox(cast(AbstractCapability[Any], capability))
            for capability in capabilities or ()
        )
    ):
        raise UserError(contribution_error)
    if sandbox is not None and not isinstance(sandbox, (SandboxRef, UnavailableSandbox)):
        raise UserError(live_error)
    return sandbox


def with_sandbox_connectors(
    capabilities: Sequence[AgentCapability[AgentDepsT]] | None,
    connector_capability: SandboxConnectorsCapability,
) -> Sequence[AgentCapability[AgentDepsT]] | None:
    if not connector_capability.connectors:
        return capabilities
    return [*(capabilities or ()), connector_capability]
