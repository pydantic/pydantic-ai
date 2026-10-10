"""Resource cleanup errors and lifecycle observations."""

from dataclasses import dataclass

from pydantic_ai.workspaces import WorkspaceError


class RenderSandboxCleanupError(WorkspaceError):
    """Provider termination could not be confirmed within the cleanup budget."""


@dataclass(frozen=True)
class RenderSandboxEvent:
    """Lifecycle metadata; excludes command arguments, environment and file contents."""

    event: str
    sandbox_id: str | None
    status: str | None = None
    exit_code: int | None = None
