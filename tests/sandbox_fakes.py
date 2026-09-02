from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.sandboxes import Sandbox, SandboxBackend, SandboxRef


@dataclass(frozen=True)
class FakeSandboxResult:
    exit_code: int = 0
    stdout: str = 'connected'
    stderr: str = ''


@dataclass(frozen=True)
class FakeEntry:
    name: str
    path: str
    is_dir: bool
    size: int | None = None


class FakeFilesystem:
    """An in-memory `SandboxFilesystem` that raises `FileNotFoundError` for missing paths."""

    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self.files: dict[str, bytes] = files or {}
        self.reads: list[str] = []

    def _content(self, path: str) -> bytes:
        try:
            return self.files[path]
        except KeyError:
            raise FileNotFoundError(path) from None

    async def read_bytes(self, path: str) -> bytes:
        self.reads.append(path)
        return self._content(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        self.files[path] = data

    async def stat(self, path: str) -> FakeEntry:
        return FakeEntry(name=path.rsplit('/', 1)[-1], path=path, is_dir=False, size=len(self._content(path)))

    async def list_dir(self, path: str) -> Sequence[FakeEntry]:
        return [await self.stat(p) for p in self.files]

    async def make_dir(self, path: str) -> None:
        pass

    async def remove(self, path: str) -> None:
        self._content(path)
        del self.files[path]

    async def exists(self, path: str) -> bool:
        return path in self.files


# The facade's bounded slice form: print the window, then quit at its last line.
_SED_WINDOW_EXPR = re.compile(r'^(\d+),(\d+)p;\2q$')


class FakeSandbox:
    """A minimal in-memory `SandboxBackend` with a filesystem.

    The `sed` line-window form the `Sandbox` facade emits is served from the same files `fs`
    exposes; `sed=False` models an environment without a usable `sed` (exit 127). Every other
    command is recorded in `commands` and succeeds with empty output.
    """

    provider = 'fake'

    def __init__(self, name: str, files: dict[str, bytes] | None = None, *, sed: bool = True) -> None:
        self.name = name
        self.fs = FakeFilesystem(files)
        self.commands: list[str | Sequence[str]] = []
        self._sed = sed

    @property
    def sandbox_id(self) -> str:
        return f'fake-{self.name}'

    async def run(
        self,
        command: str | Sequence[str],
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> FakeSandboxResult:
        if not isinstance(command, str) and list(command[:2]) == ['sed', '-n']:
            if not self._sed:
                return FakeSandboxResult(exit_code=127, stdout='', stderr='sed: not found')
            expr, path = command[2], command[3]
            window = _SED_WINDOW_EXPR.match(expr)
            assert window is not None, f'FakeSandbox only emulates the line-window sed form, got {expr!r}'
            if path not in self.fs.files:
                return FakeSandboxResult(exit_code=2, stdout='', stderr=f'sed: {path}: No such file or directory')
            # `sed` splits on `\n` only, prints the selected lines, and keeps the absence of a
            # trailing newline on the file's final line.
            text = self.fs.files[path].decode('utf-8', errors='replace')
            lines = text.split('\n')
            if lines[-1] == '':
                lines.pop()
            start, end = int(window[1]) - 1, int(window[2])
            selected = lines[start:end]
            stdout = '\n'.join(selected)
            if selected and (start + len(selected) < len(lines) or text.endswith('\n')):
                stdout += '\n'
            return FakeSandboxResult(exit_code=0, stdout=stdout, stderr='')
        self.commands.append(command)
        return FakeSandboxResult(exit_code=0, stdout='', stderr='')

    async def working_dir(self) -> str:
        return '/workspace'


class RecordingSandboxBackend:
    """The four required backend members and nothing else, recording every command."""

    provider = 'fake'

    def __init__(self, sandbox_id: str) -> None:
        self.sandbox_id = sandbox_id
        self.commands: list[str | Sequence[str]] = []

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
        return FakeSandboxResult()

    async def working_dir(self) -> str:
        return '/workspace'


def ref_sandbox(ref: SandboxRef) -> Sandbox:
    """A not-yet-connected `Sandbox` for `ref`, as a run holds before the first operation.

    For serialization tests only: connecting through it fails, so a test that needs a live
    backend must reconnect through a capability's `get_sandbox`.
    """

    async def never_connects(_ref: SandboxRef) -> SandboxBackend:
        raise AssertionError('the deferred sandbox must be reconnected through a capability')

    return Sandbox._from_ref(ref, never_connects)  # pyright: ignore[reportPrivateUsage]


class ConnectOnlySandboxCapability(AbstractCapability[Any]):
    """Connects to any ref; never provisions anything."""

    def __init__(self) -> None:
        self.sandbox_ids: list[str] = []
        self.backends: list[RecordingSandboxBackend] = []

    async def get_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> SandboxBackend | None:
        self.sandbox_ids.append(ref.sandbox_id)
        backend = RecordingSandboxBackend(ref.sandbox_id)
        self.backends.append(backend)
        return backend


class AcquireOnlySandboxCapability(AbstractCapability[Any]):
    """Provisions per run and reconnects, inheriting the no-op `release_sandbox`.

    Every lifecycle call is appended to `events`, so tests can pin both the counts and the
    order in which acquisition, connection, and release happened.
    """

    id = 'test-sandbox'

    def __init__(self) -> None:
        self.events: list[str] = []
        self.backends: list[RecordingSandboxBackend] = []
        self._created = 0

    async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef:
        self._created += 1
        sandbox_id = f'created-{self._created}'
        self.events.append(f'acquire:{sandbox_id}')
        return SandboxRef(sandbox_id=sandbox_id)

    async def get_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> SandboxBackend | None:
        self.events.append(f'connect:{ref.sandbox_id}')
        backend = RecordingSandboxBackend(ref.sandbox_id)
        self.backends.append(backend)
        return backend


class LifecycleSandboxCapability(AcquireOnlySandboxCapability):
    """An `AcquireOnlySandboxCapability` that also releases its sandbox leases."""

    async def release_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> None:
        self.events.append(f'release:{ref.sandbox_id}')


class DecliningSandboxCapability(AbstractCapability[Any]):
    """A supplier that overrides `acquire_sandbox` but declines every run."""

    def __init__(self) -> None:
        self.acquire_calls = 0

    async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef | None:
        self.acquire_calls += 1
        return None


class FailingReleaseSandboxCapability(AcquireOnlySandboxCapability):
    """A capability whose `release_sandbox` always fails, e.g. because the sandbox is already gone."""

    async def release_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> None:
        self.events.append(f'release-failed:{ref.sandbox_id}')
        raise RuntimeError(f'sandbox {ref.sandbox_id!r} is already gone')
