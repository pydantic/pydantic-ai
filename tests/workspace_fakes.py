from __future__ import annotations

import posixpath
import re
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
    WorkspaceTimeoutError,
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


_ENV_COMMAND = re.compile(r'^printf %s "\$([A-Z0-9_]+)"$')


def _add_parent_directories(directories: set[str], path: str) -> None:
    parent = posixpath.dirname(path)
    while parent not in ('', '/'):
        directories.add(parent)
        parent = posixpath.dirname(parent)
    directories.add('/')


def _write(files: dict[str, bytes], directories: set[str], path: str, data: bytes) -> None:
    if path in directories:
        raise IsADirectoryError(path)
    _add_parent_directories(directories, path)
    files[path] = data


def _stat(files: dict[str, bytes], directories: set[str], path: str) -> FakeEntry:
    if path in files:
        return FakeEntry(name=posixpath.basename(path), path=path, size=len(files[path]))
    if path in directories:
        return FakeEntry(name=posixpath.basename(path), path=path, is_dir=True)
    raise FileNotFoundError(path)


def _list_dir(files: dict[str, bytes], directories: set[str], path: str) -> list[FakeEntry]:
    if path in files:
        raise NotADirectoryError(path)
    if path not in directories:
        raise FileNotFoundError(path)
    entries = [
        FakeEntry(name=posixpath.basename(file), path=file, size=len(data))
        for file, data in files.items()
        if posixpath.dirname(file) == path
    ]
    entries.extend(
        FakeEntry(name=posixpath.basename(directory), path=directory, is_dir=True)
        for directory in directories
        if directory != path and posixpath.dirname(directory) == path
    )
    return sorted(entries, key=lambda entry: entry.name)


def _make_dir(files: dict[str, bytes], directories: set[str], path: str) -> None:
    if path in files:
        raise FileExistsError(path)
    _add_parent_directories(directories, path)
    directories.add(path)


def _remove(files: dict[str, bytes], directories: set[str], path: str) -> None:
    if path in files:
        del files[path]
        return
    if path not in directories:
        raise FileNotFoundError(path)
    prefix = f'{path.rstrip("/")}/'
    for file in [file for file in files if file.startswith(prefix)]:
        del files[file]
    directories.difference_update(
        {directory for directory in directories if directory == path or directory.startswith(prefix)}
    )


def _run_conformance_command(
    command: WorkspaceCommand,
    *,
    shell: bool,
    cwd: str | None,
    env: Mapping[str, str] | None,
    timeout: float | None,
    working_dir: str,
    files: dict[str, bytes],
    directories: set[str],
) -> FakeWorkspaceResult | None:
    if isinstance(command, str) != shell:
        raise TypeError('a shell string needs `shell=True`, an argv sequence needs `shell=False`')
    if cwd is not None and not posixpath.isabs(cwd):
        raise ValueError('cwd must be absolute')
    if not isinstance(command, str) and list(command) == ['sh', '-c', 'sleep 30'] and timeout is not None:
        raise WorkspaceTimeoutError('command timed out')
    if command == 'printf out; printf err >&2; exit 7':
        return FakeWorkspaceResult(exit_code=7, stdout='out', stderr='err')
    if list(command) == ['pydantic-ai-conformance-missing-program']:
        return FakeWorkspaceResult(exit_code=127, stderr=f'{command[0]}: command not found\n')
    if list(command[:3]) == ['sh', '-c', 'pwd -P']:
        return FakeWorkspaceResult(stdout=f'{cwd or working_dir}\n')
    if len(command) == 5 and list(command[:4]) == ['sh', '-c', 'printf "%s" "$1"', 'sh']:
        return FakeWorkspaceResult(stdout=command[4])
    if len(command) == 3 and (match := _ENV_COMMAND.fullmatch(command[2])):
        return FakeWorkspaceResult(stdout=(env or {}).get(match.group(1), ''))
    if len(command) == 5 and command[2].startswith('IFS= read'):
        _write(files, directories, command[4], b'out\n')
        return FakeWorkspaceResult()
    return None


class FakeWorkspace(WorkspaceBackend, SupportsCommands, SupportsFilesystem):
    """A lazy in-memory backend with the optional native filesystem.

    Without a ref, the first operation counts as creating the environment and sets the ref; with
    one, it counts as attaching. The environment is the object itself, so attaching always
    succeeds; `InMemoryProvider` below is the fake for a ref that can be gone.
    """

    def __init__(self, name: str, files: dict[str, bytes] | None = None, *, ref: WorkspaceRef | None = None) -> None:
        self.name = name
        self._ref = ref
        self._ready = False
        self._lock = anyio.Lock()
        self.create_calls = 0
        self.attach_calls = 0
        self.cleanup_calls: list[str] = []
        self.commands: list[str | Sequence[str]] = []
        self.files = files if files is not None else {}
        self.directories = {'/workspace'}
        for path in self.files:
            _add_parent_directories(self.directories, path)
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
        self.commands.append(command)
        return FakeWorkspaceResult(stdout='connected')

    async def working_dir(self) -> str:
        await self.ensure_ready()
        return '/workspace'

    async def read_bytes(self, path: str) -> bytes:
        await self.ensure_ready()
        self.reads.append(path)
        if path in self.directories:
            raise IsADirectoryError(path)
        try:
            return self.files[path]
        except KeyError:
            raise FileNotFoundError(path) from None

    async def write_bytes(self, path: str, data: bytes) -> None:
        await self.ensure_ready()
        _write(self.files, self.directories, path, data)

    async def stat(self, path: str) -> FakeEntry:
        await self.ensure_ready()
        return _stat(self.files, self.directories, path)

    async def list_dir(self, path: str) -> Sequence[FakeEntry]:
        await self.ensure_ready()
        return _list_dir(self.files, self.directories, path)

    async def make_dir(self, path: str) -> None:
        await self.ensure_ready()
        _make_dir(self.files, self.directories, path)

    async def remove(self, path: str) -> None:
        await self.ensure_ready()
        _remove(self.files, self.directories, path)

    async def exists(self, path: str) -> bool:
        await self.ensure_ready()
        return path in self.files or path in self.directories

    async def realpath(self, path: str) -> str:
        # The environment has no symlinks, so resolving a path only normalizes it.
        await self.ensure_ready()
        return posixpath.normpath(path)

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
    """A command-only backend with no `SupportsFilesystem`, bound to an existing environment.

    It takes the ref it is bound to rather than deriving one from a name: a ref names an
    environment that exists, so a backend never invents one ahead of creating anything.
    """

    def __init__(self, ref: WorkspaceRef) -> None:
        self._ref = ref
        self.commands: list[str | Sequence[str]] = []
        self.cleanup_calls: list[str] = []

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
    return Workspace(RecordingWorkspaceBackend(ref))


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
        backend = RecordingWorkspaceBackend(ref)
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
        self.directories: dict[str, set[str]] = {}
        self.log: list[str] = []

    def reset(self) -> None:
        self.environments.clear()
        self.directories.clear()
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
            provider.directories[env_id] = {'/remote'}
            provider.log.append(f'create:{env_id}')
            self._ref = WorkspaceRef(provider=provider.name, id=env_id)
        elif self._ref.id not in provider.environments:
            raise WorkspaceUnavailableError(f'environment {self._ref.id!r} does not exist')
        elif not self.attached:
            provider.log.append(f'attach:{self._ref.id}')
        self.attached = True
        files = provider.environments[self._ref.id]
        directories = provider.directories.setdefault(self._ref.id, {'/remote'})
        for path in files:
            _add_parent_directories(directories, path)
        return files

    def _directories(self) -> set[str]:
        assert self._ref is not None
        return self._provider.directories[self._ref.id]

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> FakeWorkspaceResult:
        files = await self._files()
        conformance_result = _run_conformance_command(
            command,
            shell=shell,
            cwd=cwd,
            env=env,
            timeout=timeout,
            working_dir='/remote',
            files=files,
            directories=self._directories(),
        )
        if conformance_result is not None:
            return conformance_result
        return FakeWorkspaceResult(stdout=f'ran:{" ".join(command)}')

    async def working_dir(self) -> str:
        await self._files()
        return '/remote'

    async def read_bytes(self, path: str) -> bytes:
        files = await self._files()
        if path in self._directories():
            raise IsADirectoryError(path)
        if path not in files:
            raise FileNotFoundError(path)
        return files[path]

    async def write_bytes(self, path: str, data: bytes) -> None:
        files = await self._files()
        _write(files, self._directories(), path, data)

    async def stat(self, path: str) -> FakeEntry:
        files = await self._files()
        return _stat(files, self._directories(), path)

    async def list_dir(self, path: str) -> Sequence[FakeEntry]:
        files = await self._files()
        return _list_dir(files, self._directories(), path)

    async def make_dir(self, path: str) -> None:
        files = await self._files()
        _make_dir(files, self._directories(), path)

    async def remove(self, path: str) -> None:
        files = await self._files()
        _remove(files, self._directories(), path)

    async def exists(self, path: str) -> bool:
        files = await self._files()
        return path in files or path in self._directories()

    async def realpath(self, path: str) -> str:
        # The environment has no symlinks, so resolving a path only normalizes it.
        await self._files()
        return posixpath.normpath(path)


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
