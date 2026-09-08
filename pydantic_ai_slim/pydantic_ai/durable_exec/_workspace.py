from __future__ import annotations

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import UnavailableWorkspace, WorkspaceBackend, WorkspaceRef


def live_workspace_error(*, run_location: str, workspace_constraint: str) -> str:
    return (
        f'A live workspace handle cannot be passed {run_location}: {workspace_constraint}. '
        'Pass a `WorkspaceRef` instead and attach a capability whose `get_workspace` can supply it.'
    )


def guard_workflow_workspace(
    workspace: WorkspaceBackend | WorkspaceRef | None,
    *,
    live_error: str,
    ref_error: str | None = None,
) -> WorkspaceRef | UnavailableWorkspace | None:
    """Reject a workspace argument an older durable wrapper cannot support safely."""
    if workspace is not None and not isinstance(workspace, (WorkspaceRef, UnavailableWorkspace)):
        raise UserError(live_error)
    if isinstance(workspace, WorkspaceRef) and ref_error is not None:
        raise UserError(ref_error)
    return workspace
