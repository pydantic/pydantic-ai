from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from pydantic_ai.capabilities import AbstractCapability, AgentCapability, WrapperCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.sandboxes import SandboxBackend, SandboxRef, UnavailableSandbox


def sandbox_suppliers(capability: AbstractCapability[Any]) -> list[AbstractCapability[Any]]:
    """The capabilities in the tree that override `setup_sandbox`, in resolved-chain order.

    Ordered because the *last* supplier is the one that wins sandbox resolution. Wrapper
    forwarding and the durability capability's own routing don't count as contributions. The
    check is static: a contributor produced at run time by a dynamic capability function is
    not visible here.
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
        setup_sandbox = type(leaf).setup_sandbox
        if isinstance(leaf, WrapperCapability) and setup_sandbox is WrapperCapability.setup_sandbox:
            for supplier in sandbox_suppliers(leaf.wrapped):
                seen.add(id(supplier))
                suppliers.append(supplier)
            return
        if setup_sandbox not in (AbstractCapability.setup_sandbox, BaseDurabilityCapability.setup_sandbox):
            suppliers.append(leaf)

    capability.apply(visit)
    return suppliers


def contributes_sandbox(capability: AbstractCapability[Any]) -> bool:
    """Whether the capability tree contains a `setup_sandbox` override.

    The deprecated durable-agent wrappers reject sandbox-contributing capabilities up front:
    running the supplier's lifecycle hooks would be I/O in workflow code, and the wrappers
    have no way to route it into a durable unit. `TemporalDurability` does; see
    [`run_sandbox_supplier`][pydantic_ai.durable_exec._sandbox.run_sandbox_supplier].
    """
    return bool(sandbox_suppliers(capability))


def run_sandbox_supplier(capability: AbstractCapability[Any]) -> AbstractCapability[Any] | None:
    """The capability whose `setup_sandbox` wins sandbox resolution, if any.

    Only the *last* supplier is considered, because that is the one a non-durable run would
    use: a durable engine must route the same sandbox the user would otherwise get, or reject,
    rather than silently pick a losing supplier.
    """
    suppliers = sandbox_suppliers(capability)
    return suppliers[-1] if suppliers else None


def run_owned_sandbox_unsupported_error(*, engine: str, container: str) -> str:
    return (
        f'A capability that supplies a sandbox (overrides `setup_sandbox`) is not supported inside a '
        f'{engine} {container}: creating and destroying the sandbox would be {container} code, which '
        f'{engine} replays. Temporal runs the sandbox lifecycle in durable units and does support it; '
        f'on other engines, create the sandbox outside the {container} and pass a `SandboxRef` to the '
        'run instead.'
    )


def sandbox_contribution_error(*, run_location: str, sandbox_constraint: str) -> str:
    return (
        f'A capability that supplies a sandbox (overrides `setup_sandbox`) cannot run {run_location}: '
        f'{sandbox_constraint}. Create the sandbox outside the workflow and pass a `SandboxRef` to the run instead.'
    )


def live_sandbox_error(*, run_location: str, sandbox_constraint: str) -> str:
    return (
        f'A live sandbox handle cannot be passed {run_location}: {sandbox_constraint}. '
        "Pass a `SandboxRef` instead and attach a capability whose `get_sandbox` can connect to it."
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
