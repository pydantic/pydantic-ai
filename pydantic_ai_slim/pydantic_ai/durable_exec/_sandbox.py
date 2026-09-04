from __future__ import annotations

from pydantic_ai.exceptions import UserError
from pydantic_ai.sandboxes import SandboxBackend, SandboxRef, UnavailableSandbox


def live_sandbox_error(*, run_location: str, sandbox_constraint: str) -> str:
    return (
        f'A live sandbox handle cannot be passed {run_location}: {sandbox_constraint}. '
        'Pass a `SandboxRef` instead and attach a capability whose `get_sandbox` can supply it.'
    )


def guard_workflow_sandbox(
    sandbox: SandboxBackend | SandboxRef | None,
    *,
    live_error: str,
) -> SandboxRef | UnavailableSandbox | None:
    """Reject a sandbox handle that cannot cross the engine's serialization boundary.

    A capability that supplies a sandbox is fine here: `get_sandbox` does no I/O, and the backend
    it returns only reaches the network on its first operation, which happens inside a durable
    unit. Only an already-built handle is rejected, because it cannot be serialized and replayed.
    """
    if sandbox is not None and not isinstance(sandbox, (SandboxRef, UnavailableSandbox)):
        raise UserError(live_error)
    return sandbox
