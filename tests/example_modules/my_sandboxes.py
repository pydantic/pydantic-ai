"""A fictional third-party sandbox library imported by the examples in `docs/sandbox.md`.

`DockerSandbox` conforms to the `pydantic_ai.sandboxes.SandboxBackend` protocol structurally (pinned
at the bottom), but nothing here runs real containers.
"""

from __future__ import annotations as _annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai.sandboxes import SandboxBackend


@dataclass(frozen=True)
class ContainerResult:
    exit_code: int = 0
    stdout: str = ''
    stderr: str = ''


class DockerSandbox:
    def __init__(self, *, sandbox_id: str = 'container-0123456789ab'):
        self.sandbox_id = sandbox_id

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


class SandboxClient:
    """A fictional provider SDK client used by the sandbox capability examples."""

    @classmethod
    def from_environment(cls) -> SandboxClient:
        return cls()

    async def create(self, *, idempotency_key: str | None = None) -> DockerSandbox:
        return DockerSandbox()

    async def connect(self, sandbox_id: str) -> DockerSandbox:
        return DockerSandbox(sandbox_id=sandbox_id)

    async def destroy(self, sandbox_id: str) -> None:
        pass


if TYPE_CHECKING:
    # The docs promise that `DockerSandbox` is a valid `SandboxBackend`; hold this module to it.
    _conforms: SandboxBackend = DockerSandbox()
