"""A minimal local implementation of the [sandbox protocol][pydantic_ai.sandbox.Sandbox].

[`LocalSandbox`][pydantic_ai.local_sandbox.LocalSandbox] runs commands as plain host
subprocesses and touches the real filesystem through `pathlib` — it **isolates nothing**.
It exists so the sandbox concept works out of the box for trusted workloads, tests, and
development, and doubles as the reference for implementing the protocol: the whole thing
is one page over `asyncio.subprocess`.

```python
from pydantic_ai import Agent, LocalSandbox, RunContext, UserError

agent = Agent('anthropic:claude-sonnet-5')


@agent.tool
async def execute(ctx: RunContext[None], command: str) -> str:
    sandbox = ctx.sandbox
    if sandbox is None:
        raise UserError('No sandbox is attached to this run.')
    result = await sandbox.run(command, shell=True, timeout=60)
    return result.stdout if result.exit_code == 0 else f'[exit {result.exit_code}] {result.stderr}'


async def main() -> None:
    async with LocalSandbox() as sandbox:  # a temporary directory, removed on exit
        await agent.run('Write fizzbuzz to fizzbuzz.py and run it.', sandbox=sandbox)
```

What it can't do, it says so: `start()` raises `NotImplementedError` (use `run(timeout=...)`
to bound commands), and so does `output_limit=` (bound output in-command, e.g.
`| tail -c 10000`). POSIX only — construction raises `NotImplementedError` elsewhere rather
than shipping a broken kill guarantee. `timeout=` honors the protocol's contract: the whole
process group is killed at the deadline and a `TimeoutError` subclass is raised.
"""

from __future__ import annotations as _annotations

import asyncio
import os
import posixpath
import shutil
import signal
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, NoReturn

from typing_extensions import Self

from .sandbox import SandboxCommand

if TYPE_CHECKING:
    from .sandbox import Sandbox

__all__ = ('LocalSandbox',)


@dataclass(frozen=True)
class _LocalResult:
    exit_code: int
    stdout: str
    stderr: str
    stdout_dropped: int = 0
    stderr_dropped: int = 0


@dataclass(frozen=True)
class _LocalFileEntry:
    name: str
    path: str
    is_dir: bool
    size: int | None


class _LocalFilesystem:
    async def read_bytes(self, path: str) -> bytes:
        return await asyncio.to_thread(Path(path).read_bytes)

    async def write_bytes(self, path: str, data: bytes) -> None:
        def write() -> None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        await asyncio.to_thread(write)

    async def read_text(self, path: str, encoding: str = 'utf-8') -> str:
        return (await self.read_bytes(path)).decode(encoding)

    async def write_text(self, path: str, content: str, encoding: str = 'utf-8') -> None:
        await self.write_bytes(path, content.encode(encoding))

    async def stat(self, path: str) -> _LocalFileEntry:
        def stat() -> _LocalFileEntry:
            target = Path(path)
            size = target.stat().st_size
            is_dir = target.is_dir()
            return _LocalFileEntry(name=target.name, path=path, is_dir=is_dir, size=None if is_dir else size)

        return await asyncio.to_thread(stat)

    async def list_dir(self, path: str) -> Sequence[_LocalFileEntry]:
        def list_entries() -> list[_LocalFileEntry]:
            entries: list[_LocalFileEntry] = []
            for child in sorted(Path(path).iterdir()):
                is_dir = child.is_dir()
                try:
                    # stat, not lstat: a symlinked file reports its target's size, matching `stat()`.
                    size = None if is_dir else child.stat().st_size
                except OSError:
                    # A broken symlink in the directory must not fail the whole listing.
                    size = None
                entries.append(_LocalFileEntry(name=child.name, path=str(child), is_dir=is_dir, size=size))
            return entries

        return await asyncio.to_thread(list_entries)

    async def make_dir(self, path: str) -> None:
        await asyncio.to_thread(lambda: Path(path).mkdir(parents=True, exist_ok=True))

    async def remove(self, path: str) -> None:
        def remove() -> None:
            target = Path(path)
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()  # files and symlinks (even to directories) unlink

        await asyncio.to_thread(remove)

    async def exists(self, path: str) -> bool:
        return await asyncio.to_thread(Path(path).exists)


class LocalSandbox:
    """[`Sandbox`][pydantic_ai.sandbox.Sandbox] over host subprocesses and the host filesystem. Isolates nothing.

    Implements the protocol *structurally* — deliberately no base class, like any
    third-party implementation. Conformance is pinned by the type-checked assignment at
    the bottom of this module.

    Args:
        root: The working directory commands run in and relative paths resolve against.
            Defaults to a fresh temporary directory (created on first use), which is
            removed again when the sandbox is used as an async context manager. A
            caller-supplied `root` is never removed.
    """

    provider = 'local'

    def __init__(self, root: str | Path | None = None):
        if os.name != 'posix':
            raise NotImplementedError(
                'LocalSandbox only supports POSIX platforms: its timeout contract kills the whole '
                'process group. On other platforms, attach a container- or VM-based sandbox instead.'
            )
        self._owns_root = root is None
        self._root = None if root is None else Path(root).absolute()
        self._id = f'local-{uuid.uuid4().hex}'
        self.fs = _LocalFilesystem()

    @property
    def sandbox_id(self) -> str:
        return self._id

    @property
    def _root_path(self) -> Path:
        # The default temp root is created lazily, so a constructed-but-unused
        # sandbox doesn't leak a directory.
        if self._root is None:
            self._root = Path(tempfile.mkdtemp(prefix='pydantic-ai-sandbox-'))
        return self._root

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if self._owns_root and self._root is not None:
            try:
                await asyncio.to_thread(shutil.rmtree, self._root)
            except FileNotFoundError:
                # A command or `fs.remove()` may have deleted the root already; exiting
                # must not raise (it would mask the exception that ended the block).
                pass

    async def working_dir(self) -> str:
        return str(self._root_path)

    async def resolve(self, path: str, *, base: str | None = None) -> str:
        if posixpath.isabs(path):
            return posixpath.normpath(path)
        return posixpath.normpath(posixpath.join(base or str(self._root_path), path))

    async def run(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_limit: int | None = None,
    ) -> _LocalResult:
        if output_limit is not None:
            raise NotImplementedError('LocalSandbox does not bound output; bound it in-command, e.g. `| tail -c`.')
        # `env` overlays the host environment rather than replacing it, so passing one
        # variable doesn't strip PATH from the child.
        merged_env = {**os.environ, **env} if env is not None else None
        if shell:
            if not isinstance(command, str):
                raise TypeError('an argv sequence cannot be combined with shell=True; pass a single command string')
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd or self._root_path,
                env=merged_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Each command leads its own process group, so the timeout kill takes out
                # the whole tree — killing only `sh` would leave its children running.
                start_new_session=True,
            )
        else:
            if isinstance(command, str):
                raise TypeError('a string command requires shell=True; pass an argv sequence otherwise')
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd or self._root_path,
                env=merged_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        communicated = False
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
            communicated = True
        except (TimeoutError, asyncio.TimeoutError) as error:  # asyncio's is distinct on 3.10
            # The contract: a timeout kills the command and raises a TimeoutError subclass.
            raise TimeoutError(f'command timed out after {timeout} seconds and was killed') from error
        finally:
            # One teardown covers every exit path — timeout, cancellation of the awaiting
            # task, any other failure. `returncode` is not enough to tell whether the process
            # group is gone: a shell can exit while a background child keeps the stdout/stderr
            # pipes (and therefore `communicate()`) open. Kill the group whenever communication
            # did not finish, then reap the direct child even if signalling failed.
            if not communicated:
                try:
                    self._kill(process)
                finally:
                    await process.wait()
        assert process.returncode is not None
        return _LocalResult(
            exit_code=process.returncode,
            stdout=stdout.decode('utf-8', errors='replace'),
            stderr=stderr.decode('utf-8', errors='replace'),
        )

    @staticmethod
    def _kill(process: asyncio.subprocess.Process) -> None:
        # `start_new_session=True` made the child the leader of a fresh group we own, and
        # until the child is reaped its pgid cannot be reused — so "already exited" is the
        # only benign failure here. If a hardened host denies `killpg`, kill the direct child
        # as a best-effort fallback but still propagate the error: grandchildren may remain,
        # so the caller must not be told that the whole-group guarantee was satisfied.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            try:
                process.kill()
            finally:
                raise

    async def start(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_limit: int | None = None,
    ) -> NoReturn:
        raise NotImplementedError('LocalSandbox only supports run(); use timeout= to bound commands.')


if TYPE_CHECKING:
    # LocalSandbox satisfies the Sandbox protocol structurally; this assignment makes the
    # type checker verify full conformance — signatures included — which a
    # `@runtime_checkable` isinstance check cannot.
    _conforms: Sandbox = LocalSandbox()
