from __future__ import annotations

import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import anyio

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.workspaces import (
    ReadOnlyWorkspace,
    SupportsCommands,
    SupportsFilesystem,
    Workspace,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceRef,
    WorkspaceResult,
    WorkspaceUnavailableError,
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
_SED_REST = re.compile(r'^(\d+),\$p$')


class FakeWorkspace(WorkspaceBackend, SupportsCommands, SupportsFilesystem):
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
        if not isinstance(command, str) and list(command[:2]) == ['head', '-c']:
            count, path = int(command[2]), command[3]
            if path not in self.files:
                return FakeWorkspaceResult(exit_code=1, stderr=f'head: {path}: No such file or directory')
            return FakeWorkspaceResult(stdout=self.files[path][:count].decode('utf-8', errors='replace'))
        if isinstance(command, str) and command.startswith('sed -n '):
            # The facade sends the bounded window as `sed -n '<expr>' <path>` and, when a byte cap
            # is set, pipes that through `head -c <bytes>`.
            if not self._sed:
                return FakeWorkspaceResult(exit_code=127, stderr='sed: not found')
            sed_part, _, head_bytes = command.partition(' | head -c ')
            max_bytes = int(head_bytes) if head_bytes else None
            _, _, expression, path = shlex.split(sed_part)
            window = _SED_WINDOW.match(expression)
            rest = _SED_REST.match(expression)
            match = window or rest
            assert match is not None
            # The window read only runs after the `head` sniff has already found the file, so a path
            # reaching the `sed` pipeline always exists.
            text = self.files[path].decode('utf-8', errors='replace')
            lines = text.split('\n')
            if lines[-1] == '':
                lines.pop()
            start = int(match[1]) - 1
            end = int(window[2]) if window is not None else len(lines)
            selected = lines[start:end]
            stdout = '\n'.join(selected)
            if selected and (start + len(selected) < len(lines) or text.endswith('\n')):
                stdout += '\n'
            encoded = stdout.encode('utf-8')
            if max_bytes is not None:
                encoded = encoded[:max_bytes]
            return FakeWorkspaceResult(stdout=encoded.decode('utf-8', errors='replace'))
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


class FilesystemOnlyWorkspaceBackend(WorkspaceBackend, SupportsFilesystem):
    """Expose a fake workspace's native filesystem without command execution."""

    def __init__(self, inner: FakeWorkspace) -> None:
        self.inner = inner

    @property
    def ref(self) -> WorkspaceRef | None:
        return self.inner.ref

    async def working_dir(self) -> str:
        return await self.inner.working_dir()

    async def read_bytes(self, path: str) -> bytes:
        return await self.inner.read_bytes(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        await self.inner.write_bytes(path, data)

    async def stat(self, path: str) -> FakeEntry:
        return await self.inner.stat(path)

    async def list_dir(self, path: str) -> Sequence[FakeEntry]:
        return await self.inner.list_dir(path)

    async def make_dir(self, path: str) -> None:
        await self.inner.make_dir(path)

    async def remove(self, path: str) -> None:
        await self.inner.remove(path)

    async def exists(self, path: str) -> bool:
        return await self.inner.exists(path)


class RecordingWorkspaceBackend(WorkspaceBackend, SupportsCommands):
    """A command-only backend with no `SupportsFilesystem`."""

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


class _CommandWorkspaceBackend(WorkspaceBackend, SupportsCommands, Protocol):
    pass


class RunOnlyWorkspaceBackend(WorkspaceBackend, SupportsCommands):
    """Hide an inner backend's optional methods to exercise the shell portability path."""

    def __init__(self, inner: _CommandWorkspaceBackend) -> None:
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


class InMemoryProvider:
    """A fake remote provider: environments live here, so a backend built anywhere can reattach by ref.

    A `WorkspaceRef` names an environment held by the provider, not a backend object, the way a real
    provider's does; a backend with no ref creates an environment on first use and takes its ref.
    """

    def __init__(self, name: str = 'fake') -> None:
        self.name = name
        self.environments: dict[str, dict[str, bytes]] = {}
        self.log: list[str] = []

    def reset(self) -> None:
        self.environments.clear()
        self.log.clear()

    def backend(self, ref: WorkspaceRef | None) -> ProviderBackend:
        return ProviderBackend(self, ref)

    def capability(self, *, read_only: bool = False) -> ProviderWorkspaces:
        return ProviderWorkspaces(self, read_only=read_only)


class ProviderBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem):
    def __init__(self, provider: InMemoryProvider, ref: WorkspaceRef | None) -> None:
        self._provider = provider
        self._ref = ref
        self.attached = False

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._ref

    async def _files(self) -> dict[str, bytes]:
        await anyio.sleep(0)
        provider = self._provider
        if self._ref is None:
            env_id = f'env-{len(provider.environments) + 1}'
            provider.environments[env_id] = {}
            provider.log.append(f'create:{env_id}')
            self._ref = WorkspaceRef(provider=provider.name, id=env_id)
        elif self._ref.id not in provider.environments:
            raise WorkspaceUnavailableError(f'environment {self._ref.id!r} does not exist')
        elif not self.attached:
            provider.log.append(f'attach:{self._ref.id}')
        self.attached = True
        return provider.environments[self._ref.id]

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> FakeWorkspaceResult:
        await self._files()
        if isinstance(command, str) != shell:
            raise TypeError('a shell string needs `shell=True`, an argv sequence needs `shell=False`')
        if isinstance(command, str) or command[0] in ('head', 'sed'):
            # No shell utilities: the facade's bounded read falls back to the filesystem.
            return FakeWorkspaceResult(exit_code=127, stderr='not found')
        return FakeWorkspaceResult(stdout=f'ran:{" ".join(command)}')

    async def working_dir(self) -> str:
        await self._files()
        return '/remote'

    async def read_bytes(self, path: str) -> bytes:
        files = await self._files()
        if path not in files:
            raise FileNotFoundError(path)
        return files[path]

    async def write_bytes(self, path: str, data: bytes) -> None:
        (await self._files())[path] = data

    async def stat(self, path: str) -> FakeEntry:
        files = await self._files()
        if path not in files:
            raise FileNotFoundError(path)
        return FakeEntry(name=path.rsplit('/', 1)[-1], path=path, size=len(files[path]))

    async def list_dir(self, path: str) -> Sequence[FakeEntry]:
        files = await self._files()
        return [
            FakeEntry(name=file.rsplit('/', 1)[-1], path=file, size=len(data)) for file, data in sorted(files.items())
        ]

    async def make_dir(self, path: str) -> None:
        await self._files()

    async def remove(self, path: str) -> None:
        files = await self._files()
        if path not in files:
            raise FileNotFoundError(path)
        del files[path]

    async def exists(self, path: str) -> bool:
        return path in await self._files()


class ProviderWorkspaces(AbstractCapability[Any]):
    """Supplies `InMemoryProvider` environments: a fresh one without a ref, the named one with."""

    def __init__(self, provider: InMemoryProvider, *, read_only: bool = False) -> None:
        self.provider = provider
        self.read_only = read_only

    def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        if ref is not None and ref.provider != self.provider.name:
            return None
        backend = self.provider.backend(ref)
        return ReadOnlyWorkspace(Workspace(backend)) if self.read_only else backend
