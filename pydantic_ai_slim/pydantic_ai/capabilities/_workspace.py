"""Internal workspace routing: exactly one capability may supply a run's workspace."""

from dataclasses import dataclass
from typing import Any

from pydantic_ai._run_context import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef

from .abstract import AbstractCapability, leaf_capabilities


@dataclass(frozen=True)
class WorkspaceSelection:
    """A lazily configured backend and the capability that supplied it."""

    backend: WorkspaceBackend
    supplier: AbstractCapability[Any]


def get_run_workspace(
    capability: AbstractCapability[AgentDepsT], ctx: RunContext[AgentDepsT], ref: WorkspaceRef | None
) -> WorkspaceSelection | None:
    """Ask every active capability for the run's workspace backend; at most one may answer.

    Does no I/O: the backend it returns creates or attaches on its first operation. Deferred
    capabilities are inert, so they never contribute one.
    """
    selection: WorkspaceSelection | None = None
    for leaf in leaf_capabilities(capability):
        if leaf.defer_loading is True:
            continue
        backend = leaf.get_workspace(ctx, ref=ref)
        if backend is None:
            continue
        # Raise on the second one rather than collecting them all: one workspace is the contract,
        # and the first two names say enough to fix the configuration.
        if selection is not None:
            raise UserError(
                'Exactly one capability may supply the run workspace; '
                f'{type(selection.supplier).__name__} and {type(leaf).__name__} both did.'
            )
        selection = WorkspaceSelection(backend, leaf)
    return selection


def workspace_supplier_scope(
    root: AbstractCapability[AgentDepsT], supplier: AbstractCapability[AgentDepsT]
) -> AbstractCapability[AgentDepsT]:
    """Retain the outermost supplier branch without changing a wrapper's policy inputs."""
    return next(
        capability
        for capability in leaf_capabilities(root)
        if any(leaf is supplier for leaf in leaf_capabilities(capability))
    )
