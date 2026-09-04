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
    suppliers: list[tuple[AbstractCapability[AgentDepsT], SandboxBackend]] = []
    for leaf in leaf_capabilities(capability):
        if leaf.defer_loading is True:
            continue
        backend = leaf.get_sandbox(ctx, ref=ref)
        if backend is not None:
            suppliers.append((leaf, backend))

    if not suppliers:
        return None
    if len(suppliers) == 1:
        return suppliers[0][1]

    names = ', '.join(type(leaf).__name__ for leaf, _ in suppliers)
    raise UserError(f'Exactly one capability may supply the run sandbox; {names} all did.')
