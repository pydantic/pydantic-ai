"""[E2B](https://e2b.dev) cloud sandboxes as a [sandbox backend][pydantic_ai.sandboxes.SandboxBackend].

Needs the `e2b` optional dependency group (`pip install "pydantic-ai-slim[e2b]"`). Import
[`E2BSandbox`][pydantic_ai.sandboxes.e2b.E2BSandbox] from this module directly:
`pydantic_ai.sandboxes` deliberately doesn't re-export it, so importing the package never
pulls in the E2B SDK.
"""

from __future__ import annotations as _annotations

import shlex
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anyio
from typing_extensions import Self, Unpack

from .protocol import FileEntry, SandboxCommand

try:
    from e2b import (
        ApiParams,
        AsyncCommandHandle,
        AsyncSandbox,
        CommandExitException,
        EntryInfo,
        FileType,
        SandboxException,
    )
except ImportError as _import_error:
    raise ImportError(
        'Please install `e2b` to use the E2B sandbox, '
        'you can use the `e2b` optional group — `pip install "pydantic-ai-slim[e2b]"`'
    ) from _import_error

if TYPE_CHECKING:
    from .protocol import SandboxBackend, SandboxProcess, SupportsFilesystem, SupportsStart

__all__ = ('E2BSandbox',)


@dataclass(frozen=True)
class _E2BResult:
    exit_code: int
    stdout: str
    stderr: str
    stdout_dropped: int = 0
    stderr_dropped: int = 0


class _E2BProcess:
    """A command running inside an E2B sandbox, as returned by `E2BSandbox.start()`."""

    def __init__(self, handle: AsyncCommandHandle, *, timeout: float | None):
        self._handle = handle
        self._timeout = timeout
        self._lock = anyio.Lock()
        self._outcome: _E2BResult | Exception | None = None

    @property
    def pid(self) -> int:
        return self._handle.pid

    async def wait(self) -> _E2BResult:
        # The deadline below fires once, so the first call's verdict is the process's verdict:
        # caching it is what makes repeated and concurrent `wait()`s agree, as the protocol requires.
        async with self._lock:
            if self._outcome is None:
                try:
                    self._outcome = await self._settle()
                except Exception as error:
                    self._outcome = error
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome

    async def _settle(self) -> _E2BResult:
        try:
            with anyio.fail_after(self._timeout):
                result = await self._handle.wait()
        except CommandExitException as exit_error:
            # A non-zero exit is a normal result, not an error; the SDK reports it by raising.
            return _E2BResult(exit_code=exit_error.exit_code, stdout=exit_error.stdout, stderr=exit_error.stderr)
        except TimeoutError as error:
            # E2B's own command deadline only tears the event stream down and leaves the command
            # running in the sandbox, so the kill the timeout contract promises is ours to perform.
            await self._handle.kill()
            raise TimeoutError(f'command timed out after {self._timeout} seconds and was killed') from error
        return _E2BResult(exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr)

    async def kill(self) -> None:
        await self._handle.kill()


class _E2BFilesystem:
    def __init__(self, sandbox: AsyncSandbox):
        self._sandbox = sandbox

    async def read_bytes(self, path: str) -> bytes:
        # `format='bytes'` is the byte-exact read; it yields a `bytearray`, the protocol wants `bytes`.
        return bytes(await self._sandbox.files.read(path, format='bytes'))

    async def write_bytes(self, path: str, data: bytes) -> None:
        # E2B's `write` already creates missing parents and replaces existing contents.
        # The ignore is for an unparameterized `IO` in the SDK's own signature, not for our call.
        await self._sandbox.files.write(path, data)  # pyright: ignore[reportUnknownMemberType]

    async def stat(self, path: str) -> FileEntry:
        return _file_entry(await self._sandbox.files.get_info(path))

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        return [_file_entry(entry) for entry in await self._sandbox.files.list(path)]

    async def make_dir(self, path: str) -> None:
        # Creates parents, and returns `False` rather than raising when the directory
        # already exists: `mkdir -p` semantics, so the return value carries nothing we need.
        await self._sandbox.files.make_dir(path)

    async def remove(self, path: str) -> None:
        await self._sandbox.files.remove(path)

    async def exists(self, path: str) -> bool:
        return await self._sandbox.files.exists(path)


def _file_entry(entry: EntryInfo) -> FileEntry:
    is_dir = entry.type is FileType.DIR
    # A directory's reported size is an implementation detail of the underlying filesystem
    # rather than a content length, so report none for it, like the other built-in backends.
    return FileEntry(name=entry.name, path=entry.path, is_dir=is_dir, size=None if is_dir else entry.size)


def _command_text(command: SandboxCommand, shell: bool) -> str:
    if shell:
        if not isinstance(command, str):
            raise TypeError('an argv sequence cannot be combined with shell=True; pass a single command string')
        return command
    if isinstance(command, str):
        raise TypeError('a string command requires shell=True; pass an argv sequence otherwise')
    # E2B only executes shell strings (every command runs as `/bin/bash -l -c <string>`), so argv
    # form is quoted back into one. `shlex.join` makes each element a single literal word, which
    # is what argv execution means: no element can be split, expanded, or read as an operator.
    return shlex.join(command)


class E2BSandbox:
    """[`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] over an [E2B](https://e2b.dev) cloud sandbox.

    Commands and file operations run inside E2B's Firecracker microVM, so — unlike
    [`LocalSandbox`][pydantic_ai.sandboxes.LocalSandbox] — the host is never exposed. Create one
    with [`create()`][pydantic_ai.sandboxes.e2b.E2BSandbox.create], or attach to an existing
    environment with [`connect()`][pydantic_ai.sandboxes.e2b.E2BSandbox.connect].

    Background processes are supported natively
    ([`SupportsStart`][pydantic_ai.sandboxes.SupportsStart]), but live output
    ([`SupportsStream`][pydantic_ai.sandboxes.SupportsStream]) is not: the E2B async SDK
    delivers output through `on_stdout`/`on_stderr` callbacks that its own event pump awaits, so
    turning them into an async iterator needs a queue that either grows without bound or stalls
    the pump — and a stalled pump never completes `wait()`. Poll the process with
    `wait()` instead, or tee the output to a file inside the sandbox and read it.

    `output_limit=` is not implemented either: E2B always delivers the full output, and dropping
    characters after the fact would misreport what the command produced. Bound it in-command
    instead, e.g. `| tail -c 10000`.

    Deliberately no base class: it conforms to the protocol structurally, like any third-party
    backend would.

    Args:
        sandbox: A live E2B `AsyncSandbox`. Prefer `create()` or `connect()`; pass a sandbox
            here only when you already hold one, and remember that whoever created it owns
            killing it.
    """

    provider = 'e2b'

    def __init__(self, sandbox: AsyncSandbox):
        self.sandbox = sandbox
        """The underlying E2B `AsyncSandbox`, for provider-specific functionality."""
        self.fs = _E2BFilesystem(sandbox)
        self._working_dir: str | None = None

    @property
    def sandbox_id(self) -> str:
        return self.sandbox.sandbox_id

    @classmethod
    @asynccontextmanager
    async def create(
        cls,
        template: str | None = None,
        *,
        timeout: int | None = None,
        envs: Mapping[str, str] | None = None,
        metadata: Mapping[str, str] | None = None,
        **api_params: Unpack[ApiParams],
    ) -> AsyncGenerator[Self]:
        """Provision a fresh E2B sandbox for the duration of the block, then kill it.

        The supplier of a sandbox owns its lifecycle, so this is an async context manager: the
        environment is killed when the block ends, including on failure and on cancellation.
        E2B's own `timeout` remains the backstop for the block that never ends.

        ```python {test="skip"}
        from pydantic_ai.sandboxes.e2b import E2BSandbox


        async def main() -> None:
            async with E2BSandbox.create(timeout=600) as sandbox:
                result = await sandbox.run(['python', '-c', 'print(1 + 1)'])
                print(result.stdout)
        ```

        Args:
            template: E2B sandbox template name or ID; defaults to E2B's `base` template.
            timeout: How long E2B keeps the sandbox alive, in seconds (E2B's default is 300).
            envs: Environment variables set for the whole sandbox.
            metadata: Custom metadata to tag the sandbox with, e.g. the run it belongs to.
            api_params: E2B connection options such as `api_key` (which defaults to the
                `E2B_API_KEY` environment variable) and `domain`.
        """
        sandbox = await AsyncSandbox.create(
            template=template,
            timeout=timeout,
            envs=dict(envs) if envs is not None else None,
            metadata=dict(metadata) if metadata is not None else None,
            **api_params,
        )
        try:
            yield cls(sandbox)
        finally:
            # Shielded because teardown can run inside an already-cancelled scope, where every
            # await re-raises: an unshielded kill would leak a running (and billed) sandbox.
            with anyio.CancelScope(shield=True):
                await sandbox.kill()

    @classmethod
    async def connect(cls, sandbox_id: str, **api_params: Unpack[ApiParams]) -> Self:
        """Attach to an E2B sandbox that already exists, without taking over its lifecycle.

        This is the building block for a
        [`SandboxResolver`][pydantic_ai.sandboxes.SandboxResolver] — a capability's
        [`get_sandbox`][pydantic_ai.capabilities.AbstractCapability.get_sandbox] hook turning a
        [`SandboxRef`][pydantic_ai.sandboxes.SandboxRef] back into a live backend, wherever the
        work actually runs:

        ```python {test="skip"}
        from typing import Any

        from pydantic_ai import RunContext
        from pydantic_ai.capabilities import AbstractCapability
        from pydantic_ai.sandboxes import SandboxBackend, SandboxRef
        from pydantic_ai.sandboxes.e2b import E2BSandbox


        class E2BCapability(AbstractCapability[Any]):
            async def get_sandbox(
                self, ctx: RunContext[Any], ref: SandboxRef
            ) -> SandboxBackend | None:
                if ref.provider != 'e2b':
                    return None
                return await E2BSandbox.connect(ref.sandbox_id)
        ```

        Connecting never creates: `e2b.SandboxNotFoundException` is raised if the environment is
        gone, rather than an empty replacement being silently swapped in.

        Args:
            sandbox_id: The `sandbox_id` of the environment to attach to.
            api_params: E2B connection options such as `api_key` (which defaults to the
                `E2B_API_KEY` environment variable) and `domain`.
        """
        return cls(await AsyncSandbox.connect(sandbox_id, **api_params))

    async def working_dir(self) -> str:
        # E2B exposes no API for the template's default directory, and a command started without
        # `cwd` lands in exactly that directory, so ask the environment itself. It cannot change.
        if self._working_dir is None:
            result = await self.run(['pwd'])
            if result.exit_code != 0:
                raise SandboxException(
                    f'Could not determine the working directory of sandbox {self.sandbox_id!r}: '
                    f'`pwd` exited {result.exit_code}: {result.stderr}'
                )
            self._working_dir = result.stdout.strip()
        return self._working_dir

    async def run(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_limit: int | None = None,
    ) -> _E2BResult:
        process = await self.start(command, shell=shell, cwd=cwd, env=env, timeout=timeout, output_limit=output_limit)
        return await process.wait()

    async def start(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_limit: int | None = None,
    ) -> _E2BProcess:
        if output_limit is not None:
            raise NotImplementedError('E2BSandbox does not bound output; bound it in-command, e.g. `| tail -c`.')
        handle = await self.sandbox.commands.run(
            _command_text(command, shell),
            background=True,
            # `envs` are applied on top of the sandbox's environment, not in place of it.
            envs=dict(env) if env is not None else None,
            # `cwd=None` leaves the command in the sandbox's own default directory.
            cwd=cwd,
            # `0` disables E2B's command deadline (which would otherwise default to 60 seconds).
            # The deadline is the process handle's to own, because E2B's merely drops the event
            # stream while the command keeps running, which the timeout contract forbids.
            timeout=0,
        )
        return _E2BProcess(handle, timeout=timeout)


if TYPE_CHECKING:
    # Pins full structural conformance — signatures included — which `isinstance` cannot check.
    # `__new__` rather than a call, because neither SDK object can be constructed without a
    # live sandbox behind it; this block never runs.
    _sandbox = AsyncSandbox.__new__(AsyncSandbox)
    _handle = AsyncCommandHandle.__new__(AsyncCommandHandle)
    _backend_conforms: SandboxBackend = E2BSandbox(_sandbox)
    _filesystem_backend_conforms: SupportsFilesystem = E2BSandbox(_sandbox)
    _start_conforms: SupportsStart = E2BSandbox(_sandbox)
    _process_conforms: SandboxProcess = _E2BProcess(_handle, timeout=None)
