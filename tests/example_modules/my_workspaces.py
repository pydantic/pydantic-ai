"""A fictional third-party workspace library imported by the examples in `docs/workspace.md`.

`DockerWorkspace` is a `pydantic_ai.workspaces.WorkspaceBackend`, but nothing here runs real containers.
"""

from __future__ import annotations as _annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef


@dataclass(frozen=True)
class ContainerResult:
    exit_code: int = 0
    stdout: str = ''
    stderr: str = ''


class DockerWorkspace(WorkspaceBackend):
    def __init__(self, *, workspace_id: str = 'container-0123456789ab'):
        self._ref = WorkspaceRef(provider='fake', id=workspace_id)

    @property
    def ref(self) -> WorkspaceRef:
        return self._ref

    async def run(
        self,
        command: str | Sequence[str],
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ContainerResult:
        return ContainerResult()

    async def working_dir(self) -> str:
        return '/workspace'


class WorkspaceClient:
    """A fictional provider SDK client used by the workspace capability examples."""

    @classmethod
    def from_environment(cls) -> WorkspaceClient:
        return cls()

    async def create(self, *, idempotency_key: str | None = None) -> DockerWorkspace:
        return DockerWorkspace()

    async def connect(self, workspace_id: str) -> DockerWorkspace:
        return DockerWorkspace(workspace_id=workspace_id)

    async def destroy(self, workspace_id: str) -> None:
        pass
