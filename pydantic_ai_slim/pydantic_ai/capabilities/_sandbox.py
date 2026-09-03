"""Internal sandbox reference connection error handling."""

from pydantic_ai._run_context import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.sandboxes import SandboxBackend, SandboxRef
from pydantic_ai.tools import AgentDepsT

from .abstract import AbstractCapability


def connect_sandbox_ref(
    capability: AbstractCapability[AgentDepsT], ctx: RunContext[AgentDepsT], ref: SandboxRef
) -> SandboxBackend:
    """Connect to `ref` through a capability hierarchy or raise a user-facing error."""
    try:
        backend = capability.get_sandbox(ctx, ref)
    except UserError:
        raise
    except Exception as error:
        raise UserError(f'Failed to connect to sandbox {ref.sandbox_id!r}.') from error
    if backend is not None:
        return backend
    if ref.capability_id is not None:
        raise UserError(
            f'No capability with id {ref.capability_id!r} is attached to this agent to connect sandbox '
            f'{ref.sandbox_id!r}.'
        )
    raise UserError(
        f'No capability can connect to sandbox {ref.sandbox_id!r}: every `get_sandbox` returned `None`. '
        'Attach a capability whose `get_sandbox` recognizes it.'
    )
