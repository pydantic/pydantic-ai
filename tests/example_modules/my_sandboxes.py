"""A fictional third-party sandbox library imported by the examples in `docs/sandbox.md`.

`DockerSandbox` conforms to the `pydantic_ai.sandbox.Sandbox` protocol structurally (pinned
at the bottom), but nothing here runs real containers: commands succeed with empty output and
the "filesystem" is an in-memory dict.
"""

from __future__ import annotations as _annotations

import posixpath
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai.sandbox import Sandbox, SandboxProcess


@dataclass(frozen=True)
class ContainerResult:
    exit_code: int = 0
    stdout: str = ''
    stderr: str = ''
    stdout_dropped: int = 0
    stderr_dropped: int = 0


@dataclass(frozen=True)
class ContainerFileEntry:
    name: str
    path: str
    is_dir: bool
    size: int | None


class ContainerFilesystem:
    def __init__(self) -> None:
        self._files: dict[str, bytes] = {}

    async def read_bytes(self, path: str) -> bytes:
        return self._files[path]

    async def write_bytes(self, path: str, data: bytes) -> None:
        self._files[path] = data

    async def read_text(self, path: str, encoding: str = 'utf-8') -> str:
        return self._files[path].decode(encoding)

    async def write_text(self, path: str, content: str, encoding: str = 'utf-8') -> None:
        self._files[path] = content.encode(encoding)

    async def stat(self, path: str) -> ContainerFileEntry:
        return ContainerFileEntry(name=posixpath.basename(path), path=path, is_dir=False, size=len(self._files[path]))

    async def list_dir(self, path: str) -> Sequence[ContainerFileEntry]:
        prefix = path.rstrip('/') + '/'
        return [await self.stat(file) for file in sorted(self._files) if file.startswith(prefix)]

    async def make_dir(self, path: str) -> None:
        pass

    async def remove(self, path: str) -> None:
        del self._files[path]

    async def exists(self, path: str) -> bool:
        return path in self._files


class DockerSandbox:
    provider = 'docker'

    def __init__(self, *, image: str = 'python:3.13', sandbox_id: str = 'container-0123456789ab'):
        self.image = image
        self._sandbox_id = sandbox_id
        self.fs = ContainerFilesystem()

    @property
    def sandbox_id(self) -> str:
        return self._sandbox_id

    async def __aenter__(self) -> DockerSandbox:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        pass

    async def run(
        self,
        command: str | Sequence[str],
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_limit: int | None = None,
    ) -> ContainerResult:
        return ContainerResult()

    async def start(
        self,
        command: str | Sequence[str],
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_limit: int | None = None,
    ) -> SandboxProcess:
        raise NotImplementedError('this docs stand-in does not run background processes')

    async def working_dir(self) -> str:
        return '/workspace'

    async def resolve(self, path: str, *, base: str | None = None) -> str:
        if posixpath.isabs(path):
            return posixpath.normpath(path)
        return posixpath.normpath(posixpath.join(base or '/workspace', path))


def make_docker_sandbox(image: str = 'python:3.13') -> DockerSandbox:
    return DockerSandbox(image=image)


async def open_sandbox(provider: str, sandbox_id: str) -> DockerSandbox:
    assert provider == DockerSandbox.provider
    return DockerSandbox(sandbox_id=sandbox_id)


if TYPE_CHECKING:
    # The docs promise that `DockerSandbox` is a valid `Sandbox`; hold this module to it.
    _conforms: Sandbox = DockerSandbox()
