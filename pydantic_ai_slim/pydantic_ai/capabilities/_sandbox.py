"""Internal sandbox routing: exactly one capability may supply a run's sandbox."""

from pydantic_ai._run_context import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.sandboxes import SandboxBackend, SandboxRef
from pydantic_ai.tools import AgentDepsT

from .abstract import AbstractCapability, leaf_capabilities


def get_run_sandbox(
    capability: AbstractCapability[AgentDepsT], ctx: RunContext[AgentDepsT], ref: SandboxRef | None
) -> SandboxBackend | None:
    """Ask every active capability for the run's sandbox backend; at most one may answer.

    Does no I/O: the backend it returns creates or attaches on its first operation. Deferred
    capabilities are inert, so they never contribute one.
    """
    supplier: AbstractCapability[AgentDepsT] | None = None
    chosen: SandboxBackend | None = None
    for leaf in leaf_capabilities(capability):
        if leaf.defer_loading is True:
            continue
        backend = leaf.get_sandbox(ctx, ref=ref)
        if backend is None:
            continue
        # Raise on the second one rather than collecting them all: one sandbox is the contract,
        # and the first two names say enough to fix the configuration.
        if supplier is not None:
            raise UserError(
                'Exactly one capability may supply the run sandbox; '
                f'{type(supplier).__name__} and {type(leaf).__name__} both did.'
            )
        supplier, chosen = leaf, backend
    return chosen
