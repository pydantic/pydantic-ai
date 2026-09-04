from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import anyio

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.sandboxes import Sandbox, SandboxBackend, SandboxRef


@dataclass(frozen=True)
class FakeSandboxResult:
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


class FakeFilesystem:
    def __init__(self, backend: FakeSandbox, files: dict[str, bytes] | None = None) -> None:
        self._backend = backend
        self.files = files or {}
        self.reads: list[str] = []

    async def read_bytes(self, path: str) -> bytes:
        await self._backend.ensure_ready()
        self.reads.append(path)
        try:
            return self.files[path]
        except KeyError:
            raise FileNotFoundError(path) from None

    async def write_bytes(self, path: str, data: bytes) -> None:
        await self._backend.ensure_ready()
        self.files[path] = data

    async def stat(self, path: str) -> FakeEntry:
        await self._backend.ensure_ready()
        try:
            data = self.files[path]
        except KeyError:
            raise FileNotFoundError(path) from None
        return FakeEntry(name=path.rsplit('/', 1)[-1], path=path, size=len(data))

    async def list_dir(self, path: str) -> Sequence[FakeEntry]:
        await self._backend.ensure_ready()
        return [FakeEntry(name=p.rsplit('/', 1)[-1], path=p, size=len(data)) for p, data in self.files.items()]

    async def make_dir(self, path: str) -> None:
        await self._backend.ensure_ready()

    async def remove(self, path: str) -> None:
        await self._backend.ensure_ready()
        try:
            del self.files[path]
        except KeyError:
            raise FileNotFoundError(path) from None

    async def exists(self, path: str) -> bool:
        await self._backend.ensure_ready()
        return path in self.files


class FakeSandbox:
    """A lazy in-memory backend with the optional native filesystem."""

    def __init__(
        self, name: str, files: dict[str, bytes] | None = None, *, ref: SandboxRef | None = None, sed: bool = True
    ) -> None:
        self.name = name
        self._ref = ref
        self._ready = False
        self._lock = anyio.Lock()
        self.create_calls = 0
        self.attach_calls = 0
        self.commands: list[str | Sequence[str]] = []
        # Teardown that must never happen. Nothing in Pydantic AI closes or releases a sandbox,
        # and these record it if anything ever does; `pragma: no cover` because staying uncalled
        # is the assertion.
        self.cleanup_calls: list[str] = []
        self._sed = sed
        self.fs = FakeFilesystem(self, files)

    @property
    def ref(self) -> SandboxRef | None:
        return self._ref

    async def ensure_ready(self) -> None:
        async with self._lock:
            if self._ready:
                return
            await anyio.sleep(0)
            if self._ref is None:
                self.create_calls += 1
                self._ref = SandboxRef(sandbox_id=f'fake-{self.name}')
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
    ) -> FakeSandboxResult:
        await self.ensure_ready()
        if not isinstance(command, str) and list(command[:2]) == ['sed', '-n']:
            if not self._sed:
                return FakeSandboxResult(exit_code=127, stderr='sed: not found')
            expression, path = command[2], command[3]
            match = _SED_WINDOW.match(expression)
            assert match is not None
            if path not in self.fs.files:
                return FakeSandboxResult(exit_code=2, stderr=f'sed: {path}: No such file or directory')
            text = self.fs.files[path].decode('utf-8', errors='replace')
            lines = text.split('\n')
            if lines[-1] == '':
                lines.pop()
            start, end = int(match[1]) - 1, int(match[2])
            selected = lines[start:end]
            stdout = '\n'.join(selected)
            if selected and (start + len(selected) < len(lines) or text.endswith('\n')):
                stdout += '\n'
            return FakeSandboxResult(stdout=stdout)
        self.commands.append(command)
        return FakeSandboxResult(stdout='connected')

    async def working_dir(self) -> str:
        await self.ensure_ready()
        return '/workspace'

    async def close(self, *, terminate: bool = False) -> None:  # pragma: no cover
        self.cleanup_calls.append(f'close:{terminate}')

    async def release(self) -> None:  # pragma: no cover
        self.cleanup_calls.append('release')


class RecordingSandboxBackend:
    """The three required backend members, with no `SupportsFilesystem`."""

    def __init__(self, sandbox_id: str, *, ref: SandboxRef | None = None) -> None:
        self._ref = ref or SandboxRef(sandbox_id=sandbox_id)
        self.commands: list[str | Sequence[str]] = []

    @property
    def ref(self) -> SandboxRef | None:
        return self._ref

    async def run(
        self,
        command: str | Sequence[str],
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> FakeSandboxResult:
        self.commands.append(command)
        return FakeSandboxResult(stdout='connected')

    async def working_dir(self) -> str:
        return '/workspace'


def ref_sandbox(ref: SandboxRef) -> Sandbox:
    return Sandbox(RecordingSandboxBackend(ref.sandbox_id, ref=ref))


class ConnectOnlySandboxCapability(AbstractCapability[Any]):
    """Supplies a run-only backend for the requested ref."""

    def __init__(self) -> None:
        self.sandbox_ids: list[str] = []
        self.backends: list[RecordingSandboxBackend] = []

    def get_sandbox(self, ctx: RunContext[Any], *, ref: SandboxRef | None) -> SandboxBackend | None:
        if ref is None:
            return None
        self.sandbox_ids.append(ref.sandbox_id)
        backend = RecordingSandboxBackend(ref.sandbox_id, ref=ref)
        self.backends.append(backend)
        return backend


class SandboxCapability(AbstractCapability[Any]):
    id = 'sandbox'

    def __init__(self, backend: FakeSandbox | None = None) -> None:
        self.backend = backend or FakeSandbox('capability')
        self.refs: list[SandboxRef | None] = []

    def get_sandbox(self, ctx: RunContext[Any], *, ref: SandboxRef | None) -> SandboxBackend:
        self.refs.append(ref)
        return self.backend


class DecliningSandboxCapability(AbstractCapability[Any]):
    def __init__(self) -> None:
        self.calls = 0

    def get_sandbox(self, ctx: RunContext[Any], *, ref: SandboxRef | None) -> None:
        self.calls += 1
        return None
