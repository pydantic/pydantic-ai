from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from pydantic_ai.capabilities import AbstractCapability, AgentCapability, WrapperCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.sandboxes import ManagedSandbox, SandboxBackend, SandboxProvider, SandboxRef, UnavailableSandbox

AgentDepsT = TypeVar('AgentDepsT')


@dataclass
class SandboxProvidersCapability(AbstractCapability[Any]):
    """Internal capability used by deprecated durable-agent wrappers."""

    providers: Sequence[SandboxProvider]

    def get_sandbox_providers(self) -> Sequence[SandboxProvider]:
        return self.providers


def sandbox_suppliers(capability: AbstractCapability[Any]) -> list[AbstractCapability[Any]]:
    """The capabilities in the tree that override `get_sandbox`, in resolved-chain order.

    `WrapperCapability.get_sandbox` is a pure forwarder and
    `BaseDurabilityCapability.get_sandbox` only routes or suppresses the framework default
    inside a durable context; neither is a user sandbox contribution. `apply()` visits a
    wrapper's leaves only when it wraps a container, so a contributor behind a wrapper over a
    single leaf is found by recursing explicitly; the identity set keeps the other case from
    counting a leaf twice. A durability subclass that overrides `get_sandbox` itself is still
    treated as a contributor.

    The result is a chain-ordered list rather than a set because the *last* supplier is the one
    that wins sandbox resolution.

    The check is static, so a contributor only produced at run time by a dynamic capability
    function is not visible here.
    """
    # Import lazily so loading this lightweight workflow guard does not pull the durability
    # capability's model/toolset import graph into Temporal workflow sandbox validation.
    from ._base import BaseDurabilityCapability

    suppliers: list[AbstractCapability[Any]] = []
    seen: set[int] = set()

    def visit(leaf: AbstractCapability[Any]) -> None:
        if leaf.defer_loading is True or id(leaf) in seen:
            return
        seen.add(id(leaf))
        get_sandbox = type(leaf).get_sandbox
        if isinstance(leaf, WrapperCapability) and get_sandbox is WrapperCapability.get_sandbox:
            for supplier in sandbox_suppliers(leaf.wrapped):
                seen.add(id(supplier))
                suppliers.append(supplier)
            return
        if get_sandbox not in (AbstractCapability.get_sandbox, BaseDurabilityCapability.get_sandbox):
            suppliers.append(leaf)

    capability.apply(visit)
    return suppliers


def contributes_sandbox(capability: AbstractCapability[Any]) -> bool:
    """Whether the capability tree contains a `get_sandbox` override.

    The deprecated durable-agent wrappers reject sandbox-contributing capabilities up front:
    entering the contributed context manager would run I/O in workflow code, and the wrappers
    have no way to route it into a durable unit. `TemporalDurability` does — see
    [`managed_sandbox_supplier`][pydantic_ai.durable_exec._sandbox.managed_sandbox_supplier].
    """
    return bool(sandbox_suppliers(capability))


def managed_sandbox_supplier(capability: AbstractCapability[Any]) -> ManagedSandbox | None:
    """The [`ManagedSandbox`][pydantic_ai.sandboxes.ManagedSandbox] that wins sandbox resolution, if any.

    Only the *last* supplier is considered, because that is the one whose `get_sandbox` a
    non-durable run would use: a durable engine must route the same sandbox the user would
    otherwise get, or reject, rather than silently pick a losing supplier.
    """
    suppliers = sandbox_suppliers(capability)
    winner = suppliers[-1] if suppliers else None
    return winner if isinstance(winner, ManagedSandbox) else None


def managed_sandbox_unsupported_error(*, engine: str, container: str) -> str:
    return (
        f'`ManagedSandbox` is not supported inside a {engine} {container}: creating and destroying the '
        f'sandbox would be {container} code, which {engine} replays. Temporal runs the sandbox lifecycle '
        'in activities and does support it; on other engines, create the sandbox outside the '
        f'{container} and pass a `SandboxRef` to the run instead.'
    )


def sandbox_contribution_error(*, run_location: str, sandbox_constraint: str) -> str:
    return (
        f'A capability that contributes a sandbox (overrides `get_sandbox`) cannot run {run_location}: '
        f'{sandbox_constraint}. Create the sandbox outside the workflow and pass a `SandboxRef` to the run instead.'
    )


def live_sandbox_error(*, run_location: str, sandbox_constraint: str, provider_hint: str) -> str:
    return (
        f'A live sandbox handle cannot be passed {run_location}: {sandbox_constraint}. '
        f'Pass a `SandboxRef` instead and {provider_hint}.'
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


def with_sandbox_providers(
    capabilities: Sequence[AgentCapability[AgentDepsT]] | None,
    provider_capability: SandboxProvidersCapability,
) -> Sequence[AgentCapability[AgentDepsT]] | None:
    if not provider_capability.providers:
        return capabilities
    return [*(capabilities or ()), provider_capability]
