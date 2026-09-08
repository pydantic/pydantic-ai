from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import anyio

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.workspaces import (
    SupportsFilesystem,
    Workspace,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceRef,
    WorkspaceResult,
)


@dataclass(frozen=True)
class FakeWorkspaceResult:
    exit_code: int = 0
    stdout: str = ''
    stderr: str = ''


@dataclass(frozen=True)
class FakeEntry:
    name: str
    path: str
    is_dir: bool = False
    size: int | None = None


_SED_WINDOW = re.compile(r'^(\d+),(\d+)p;\2q$')


class FakeWorkspace(WorkspaceBackend, SupportsFilesystem):
    """A lazy in-memory backend with the optional native filesystem."""

    def __init__(
        self, name: str, files: dict[str, bytes] | None = None, *, ref: WorkspaceRef | None = None, sed: bool = True
    ) -> None:
        self.name = name
        self._ref = ref
        self._ready = False
        self._lock = anyio.Lock()
        self.create_calls = 0
        self.attach_calls = 0
        self.cleanup_calls: list[str] = []
        self.commands: list[str | Sequence[str]] = []
        self._sed = sed
        self.files = files or {}
        self.reads: list[str] = []

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._ref

    async def ensure_ready(self) -> None:
        async with self._lock:
            if self._ready:
                return
            await anyio.sleep(0)
            if self._ref is None:
                self.create_calls += 1
                self._ref = WorkspaceRef(provider='fake', id=f'fake-{self.name}')
            else:
                self.attach_calls += 1
            self._ready = True

    async def run(
        self,
        command: str | Sequence[str],
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> FakeWorkspaceResult:
        await self.ensure_ready()
        if not isinstance(command, str) and list(command[:2]) == ['sed', '-n']:
            if not self._sed:
                return FakeWorkspaceResult(exit_code=127, stderr='sed: not found')
            expression, path = command[2], command[3]
            match = _SED_WINDOW.match(expression)
            assert match is not None
            if path not in self.files:
                return FakeWorkspaceResult(exit_code=2, stderr=f'sed: {path}: No such file or directory')
            text = self.files[path].decode('utf-8', errors='replace')
            lines = text.split('\n')
            if lines[-1] == '':
                lines.pop()
            start, end = int(match[1]) - 1, int(match[2])
            selected = lines[start:end]
            stdout = '\n'.join(selected)
            if selected and (start + len(selected) < len(lines) or text.endswith('\n')):
                stdout += '\n'
            return FakeWorkspaceResult(stdout=stdout)
        self.commands.append(command)
        return FakeWorkspaceResult(stdout='connected')

    async def working_dir(self) -> str:
        await self.ensure_ready()
        return '/workspace'

    async def read_bytes(self, path: str) -> bytes:
        await self.ensure_ready()
        self.reads.append(path)
        try:
            return self.files[path]
        except KeyError:
            raise FileNotFoundError(path) from None

    async def write_bytes(self, path: str, data: bytes) -> None:
        await self.ensure_ready()
        self.files[path] = data

    async def stat(self, path: str) -> FakeEntry:
        await self.ensure_ready()
        try:
            data = self.files[path]
        except KeyError:
            raise FileNotFoundError(path) from None
        return FakeEntry(name=path.rsplit('/', 1)[-1], path=path, size=len(data))

    async def list_dir(self, path: str) -> Sequence[FakeEntry]:
        await self.ensure_ready()
        return [FakeEntry(name=p.rsplit('/', 1)[-1], path=p, size=len(data)) for p, data in self.files.items()]

    async def make_dir(self, path: str) -> None:
        await self.ensure_ready()

    async def remove(self, path: str) -> None:
        await self.ensure_ready()
        try:
            del self.files[path]
        except KeyError:
            raise FileNotFoundError(path) from None

    async def exists(self, path: str) -> bool:
        await self.ensure_ready()
        return path in self.files

    async def close(self, *, terminate: bool = False) -> None:  # pragma: no cover
        self.cleanup_calls.append(f'close:{terminate}')

    async def release(self) -> None:  # pragma: no cover
        self.cleanup_calls.append('release')


class RecordingWorkspaceBackend(WorkspaceBackend):
    """The three required backend members, with no `SupportsFilesystem`."""

    def __init__(self, workspace_id: str, *, ref: WorkspaceRef | None = None) -> None:
        self._ref = ref or WorkspaceRef(provider='fake', id=workspace_id)
        self.commands: list[str | Sequence[str]] = []
        self.cleanup_calls: list[str] = []

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._ref

    async def run(
        self,
        command: str | Sequence[str],
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> FakeWorkspaceResult:
        self.commands.append(command)
        return FakeWorkspaceResult(stdout='connected')

    async def working_dir(self) -> str:
        return '/workspace'

    async def close(self, *, terminate: bool = False) -> None:  # pragma: no cover
        self.cleanup_calls.append(f'close:{terminate}')


class RunOnlyWorkspaceBackend(WorkspaceBackend):
    """Hide an inner backend's optional methods to exercise the shell portability path."""

    def __init__(self, inner: WorkspaceBackend) -> None:
        self.inner = inner
        self.commands: list[WorkspaceCommand] = []

    @property
    def ref(self) -> WorkspaceRef | None:
        return self.inner.ref

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> WorkspaceResult:
        self.commands.append(command)
        return await self.inner.run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)

    async def working_dir(self) -> str:
        return await self.inner.working_dir()


def ref_workspace(ref: WorkspaceRef, supplier: AbstractCapability[Any] | None = None) -> Workspace:
    del supplier
    return Workspace(RecordingWorkspaceBackend(ref.id, ref=ref))


class ConnectOnlyWorkspaceCapability(AbstractCapability[Any]):
    """Supplies a run-only backend for the requested ref."""

    id = 'connect_only_workspace'

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.backends: list[RecordingWorkspaceBackend] = []

    def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        if ref is None:
            return None
        self.ids.append(ref.id)
        backend = RecordingWorkspaceBackend(ref.id, ref=ref)
        self.backends.append(backend)
        return backend


class WorkspaceCapability(AbstractCapability[Any]):
    id = 'workspace'

    def __init__(self, backend: FakeWorkspace | None = None) -> None:
        self.backend = backend or FakeWorkspace('capability')
        self.refs: list[WorkspaceRef | None] = []

    def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> WorkspaceBackend:
        self.refs.append(ref)
        return self.backend


class DecliningWorkspaceCapability(AbstractCapability[Any]):
    def __init__(self) -> None:
        self.calls = 0

    def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> None:
        self.calls += 1
        return None
