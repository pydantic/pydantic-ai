"""A controllable fake `modal` SDK for ModalSandbox tests.

Tests never reach real Modal: a fake `modal` module is injected into `sys.modules`
(via the `fake_modal` fixture in `conftest.py`), so the lazy `import modal` inside
the backend returns it. The fake records calls and lets each test decide what
`exec` returns.

Fidelity to the real SDK is the point: signatures are closed, `.aio` suspends, an
exec output reader replays from byte zero on every read, and missing paths raise
Modal's own filesystem exception. Flattering the code under test here would hide
production failures.
"""

from __future__ import annotations

import asyncio
import os
import posixpath
import select
import shutil
import signal
import stat
import subprocess
import time
import types
from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import anyio.lowlevel

StreamData = bytes | str
# A responder maps (argv, timeout) to (stdout, stderr, exit_code).
Responder = Callable[[list[str], 'int | None'], 'tuple[StreamData, StreamData, int]']


def _echo_responder(argv: list[str], timeout: int | None) -> tuple[bytes, bytes, int]:
    return (' '.join(argv) + '\n').encode(), b'', 0


def _stream_bytes(data: StreamData) -> bytes:
    if isinstance(data, bytes):
        return data
    return data.encode()


@dataclass
class ExecCall:
    argv: list[str]
    timeout: int | None
    text: bool
    workdir: str | None = None
    env: dict[str, str | None] | None = None


class _AioCallable:
    """Mimics a synchronicity-wrapped Modal method: callable, plus an `.aio` async twin.

    The backend only calls `.aio`, but exposing both mirrors the real SDK shape.
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - capability only uses `.aio`
        # Modal's callables work sync or async; we mirror both for fidelity, but the
        # capability drives the async `.aio` path exclusively, so this never runs in tests.
        return self._fn(*args, **kwargs)

    async def aio(self, *args: Any, **kwargs: Any) -> Any:
        # A real Modal `.aio` call suspends (it awaits gRPC); yield here so a concurrent
        # batch of tool calls actually interleaves in tests -- otherwise the sync fake would
        # run each call start-to-finish and hide races like a duplicated `pwd` probe.
        await anyio.lowlevel.checkpoint()
        return self._fn(*args, **kwargs)


class _GatedCreate(_AioCallable):
    """A create call whose control-plane response can be held.

    The sandbox exists before the response is held, as with real Modal, where the server
    commits the create before its reply reaches the client.
    """

    def __init__(self, fn: Callable[..., Any], control: FakeModal) -> None:
        super().__init__(fn)
        self._control = control

    async def aio(self, *args: Any, **kwargs: Any) -> Any:
        await anyio.lowlevel.checkpoint()
        if self._control.create_before_gate and self._control.create_gate is not None:
            await self._control.create_gate.wait()
        created = self._fn(*args, **kwargs)
        self._control.create_started = True
        if self._control.create_gate is not None:
            await self._control.create_gate.wait()
        if self._control.create_reply_error is not None:
            raise self._control.create_reply_error
        return created


class _HostExec(_AioCallable):
    async def aio(self, *args: Any, **kwargs: Any) -> Any:
        # subprocess.run blocks; let stop exec and cancellation run while the host command waits.
        return await asyncio.to_thread(self._fn, *args, **kwargs)


class _HangingExec(_AioCallable):
    async def aio(self, *args: Any, **kwargs: Any) -> Any:
        if args[:4] == ('setsid', '-w', 'sh', '-c'):
            await anyio.sleep_forever()
        return await super().aio(*args, **kwargs)


class _HangingAioCall:
    """An `.aio` that never returns, for tests that cancel a pending call."""

    async def aio(self, *args: Any, **kwargs: Any) -> Any:
        await anyio.sleep_forever()


class _DelayedAioCall(_AioCallable):
    """An `.aio` that returns after `delay` seconds, like output still draining after the process exited."""

    def __init__(self, fn: Callable[..., Any], delay: float) -> None:
        super().__init__(fn)
        self._delay = delay

    async def aio(self, *args: Any, **kwargs: Any) -> Any:
        await anyio.sleep(self._delay)
        return self._fn(*args, **kwargs)


class _FakeStream:
    """Mimics the whole-output `.read.aio()` surface used by the backend."""

    def __init__(self, data: bytes, hangs: bool = False, delay: float = 0.0) -> None:
        self._data = data
        self._hangs = hangs
        self._delay = delay
        self.read = (
            _HangingAioCall() if hangs else _DelayedAioCall(self._read, delay) if delay else _AioCallable(self._read)
        )

    async def __aiter__(self) -> AsyncGenerator[bytes, None]:
        if self._hangs:
            await anyio.sleep_forever()
        if self._delay:
            await anyio.sleep(self._delay)
        await anyio.lowlevel.checkpoint()
        yield self._data

    def _read(self) -> bytes:
        return self._data


class _FakeProcess:
    def __init__(
        self,
        stdout: bytes,
        stderr: bytes,
        returncode: int,
        wait_error: Exception | None,
        wait_hangs: bool,
        stdout_hangs: bool = False,
        stdout_delay: float = 0.0,
    ) -> None:
        self.stdout = _FakeStream(stdout, stdout_hangs, stdout_delay)
        self.stderr = _FakeStream(stderr)
        self._returncode = returncode
        self._wait_error = wait_error
        self.returncode: int | None = None
        self.wait: _AioCallable | _HangingAioCall = _HangingAioCall() if wait_hangs else _AioCallable(self._wait)

    def _wait(self) -> int:
        if self._wait_error is not None:
            raise self._wait_error
        self.returncode = self._returncode
        return self._returncode


@dataclass(frozen=True)
class FakeImage:
    """A `modal.Image` built with `debian_slim(...).apt_install(...)`."""

    python_version: str
    apt_packages: tuple[str, ...] = ()

    def apt_install(self, *packages: str) -> FakeImage:
        return FakeImage(self.python_version, self.apt_packages + packages)


class FakeModalError(Exception):
    """Stand-in for `modal.exception.Error`."""


class FakeInvalidError(FakeModalError):
    """Stand-in for `modal.exception.InvalidError`."""


class FakeConflictError(FakeInvalidError):
    """Stand-in for `modal.exception.ConflictError` (first exec on a dead sandbox, or a transient abort)."""


class FakeSandboxFilesystemError(FakeModalError):
    """Stand-in for `modal.exception.SandboxFilesystemError`."""


class FakeSandboxFilesystemNotFoundError(FakeSandboxFilesystemError):
    """Stand-in for `modal.exception.SandboxFilesystemNotFoundError` (a missing file, recoverable)."""


class FakeSandboxFilesystemNotADirectoryError(FakeSandboxFilesystemError):
    """Stand-in for `modal.exception.SandboxFilesystemNotADirectoryError` (a non-directory path component)."""


class FakeSandboxFilesystemIsADirectoryError(FakeSandboxFilesystemError):
    """Stand-in for `modal.exception.SandboxFilesystemIsADirectoryError`."""


# The rest of Modal's exceptions the backend classifies, each a direct subclass of `Error` as in
# Modal 1.5.2 except where the real hierarchy says otherwise.
_EXCEPTION_BASES: dict[str, type[Exception]] = {
    'AlreadyExistsError': FakeModalError,
    'AuthError': FakeModalError,
    'ConnectionError': FakeModalError,
    'ExecutionError': FakeModalError,
    'FilesystemExecutionError': FakeModalError,
    'InternalError': FakeModalError,
    'ImageBuildError': FakeModalError,
    'NotFoundError': FakeModalError,
    'PermissionDeniedError': FakeModalError,
    'RequestSizeError': FakeModalError,
    'ResourceExhaustedError': FakeModalError,
    'SandboxTerminatedError': FakeModalError,
    'SandboxTimeoutError': FakeModalError,
    'ServiceError': FakeModalError,
    'SandboxFilesystemPathAlreadyExistsError': FakeSandboxFilesystemError,
    'SandboxFilesystemPermissionError': FakeSandboxFilesystemError,
}
_EXCEPTIONS: dict[str, type[Exception]] = {
    'Error': FakeModalError,
    'InvalidError': FakeInvalidError,
    'ConflictError': FakeConflictError,
    'SandboxFilesystemError': FakeSandboxFilesystemError,
    'SandboxFilesystemNotFoundError': FakeSandboxFilesystemNotFoundError,
    'SandboxFilesystemNotADirectoryError': FakeSandboxFilesystemNotADirectoryError,
    'SandboxFilesystemIsADirectoryError': FakeSandboxFilesystemIsADirectoryError,
    **{
        name: type(name if name == 'ImageBuildError' else f'Fake{name}', (base,), {})
        for name, base in _EXCEPTION_BASES.items()
    },
}


@dataclass
class FileInfo:
    """Minimal stand-in for `modal.types.FileInfo`, covering what the backend reads."""

    name: str
    _is_dir: bool
    size: int = 0
    symlink_target: str | None = None

    def is_dir(self) -> bool:
        return self._is_dir

    def is_symlink(self) -> bool:
        return self.symlink_target is not None


class _FakeFilesystem:
    """Mirrors `sandbox.filesystem`: an in-memory store the tests can drive and inspect."""

    def __init__(self, sandbox: FakeSandbox) -> None:
        self._sandbox = sandbox
        self.read_bytes = _AioCallable(self._read_bytes)
        self.write_bytes = _AioCallable(self._write_bytes)
        self.list_files = _AioCallable(self._list_files)
        self.stat = _AioCallable(self._stat)
        self.make_directory = _AioCallable(self._make_directory)
        self.remove = _AioCallable(self._remove)

    def _read_bytes(self, remote_path: str) -> bytes:
        self._check(remote_path)
        data = self._sandbox.files.get(remote_path)
        if data is None:
            raise FakeSandboxFilesystemNotFoundError(f'No such file or directory: {remote_path}')
        return data

    def _stat(self, remote_path: str) -> FileInfo:
        self._check(remote_path)
        if remote_path in self._sandbox.directories:
            return FileInfo(posixpath.basename(remote_path), True)
        data = self._sandbox.files.get(remote_path)
        if data is None:
            raise FakeSandboxFilesystemNotFoundError(f'No such file or directory: {remote_path}')
        # Real Modal reports the entry's basename, not the full path.
        return FileInfo(posixpath.basename(remote_path), False, size=len(data))

    def _write_bytes(self, data: bytes, remote_path: str) -> None:
        self._check(remote_path)
        self._sandbox.files[remote_path] = data

    def _list_files(self, remote_path: str) -> list[FileInfo]:
        self._check(remote_path)
        return self._sandbox.listing

    def _make_directory(self, remote_path: str, *, create_parents: bool = True) -> None:
        # Closed keyword signature on purpose, like `sandbox_create`: `create_parents` is the
        # real API's `mkdir -p` switch and defaults to True there too.
        self._check(remote_path)
        self._sandbox.directories.add(remote_path)

    def _remove(self, remote_path: str, *, recursive: bool = False) -> None:
        self._check(remote_path)
        self._sandbox.removals.append((remote_path, recursive))
        self._sandbox.directories.discard(remote_path)
        self._sandbox.files.pop(remote_path, None)

    def _check(self, remote_path: str) -> None:
        # Real Modal's filesystem API only accepts absolute paths; assert it here so a
        # regression that let a relative path through unresolved fails in the fake the way it
        # would in prod, instead of silently keying the in-memory store on a relative path.
        assert posixpath.isabs(remote_path), f'Modal filesystem requires an absolute path, got {remote_path!r}'
        error = self._sandbox.fs_error
        if isinstance(error, FakeSandboxFilesystemError):
            # What the filesystem tool itself reported; Modal raises these as they are.
            raise error
        if error is not None:
            with _exec_errors():
                raise error


@contextmanager
def _exec_errors() -> Generator[None]:
    """Replace an SDK failure the way Modal 1.5.2's `translate_exec_errors` does.

    Modal runs each filesystem operation as an exec and re-raises the exec's failure `from None`
    as a stand-in, keeping the original only in `__context__`.
    """
    unavailable = tuple(_EXCEPTIONS[name] for name in ('NotFoundError', 'ServiceError', 'ConnectionError'))
    try:
        yield
    except unavailable:
        raise _EXCEPTIONS['NotFoundError'](
            'The Sandbox is unavailable. This Sandbox may have already shut down.'
        ) from None
    except FakeModalError:
        raise FakeSandboxFilesystemError('An unexpected error occurred, please contact support@modal.com') from None


@contextmanager
def _host_errors(remote_path: str) -> Generator[None]:
    """Raise Modal's filesystem exceptions for the host errors the real SDK reports as them."""
    try:
        yield
    except FileNotFoundError as e:
        raise FakeSandboxFilesystemNotFoundError(f'No such file or directory: {remote_path}') from e
    except IsADirectoryError as e:
        raise FakeSandboxFilesystemIsADirectoryError(f'Is a directory: {remote_path}') from e
    except NotADirectoryError as e:
        raise FakeSandboxFilesystemNotADirectoryError(f'Not a directory: {remote_path}') from e


class _HostFilesystem:
    """Mirrors `sandbox.filesystem` on the real host filesystem, for the conformance suite.

    The suite checks that commands and filesystem methods see one environment, which the
    in-memory store cannot show; here both act on the same host paths.
    """

    def __init__(self) -> None:
        self.read_bytes = _AioCallable(self._read_bytes)
        self.write_bytes = _AioCallable(self._write_bytes)
        self.list_files = _AioCallable(self._list_files)
        self.stat = _AioCallable(self._stat)
        self.make_directory = _AioCallable(self._make_directory)
        self.remove = _AioCallable(self._remove)

    def _read_bytes(self, remote_path: str) -> bytes:
        with _host_errors(remote_path):
            return Path(remote_path).read_bytes()

    def _write_bytes(self, data: bytes, remote_path: str) -> None:
        with _host_errors(remote_path):
            try:
                Path(remote_path).parent.mkdir(parents=True, exist_ok=True)
            except FileExistsError as e:
                # mkdir on a file parent reports EEXIST on macOS, but traversal is ENOTDIR.
                raise FakeSandboxFilesystemNotADirectoryError(f'Not a directory: {remote_path}') from e
            Path(remote_path).write_bytes(data)

    def _list_files(self, remote_path: str) -> list[FileInfo]:
        with _host_errors(remote_path):
            children = sorted(Path(remote_path).iterdir())
        return [self._info(child) for child in children]

    def _stat(self, remote_path: str) -> FileInfo:
        with _host_errors(remote_path):
            return self._info(Path(remote_path))

    @staticmethod
    def _info(path: Path) -> FileInfo:
        # Like Modal's, an entry describes a symlink itself, not the target it points to.
        info = path.lstat()
        target = os.readlink(path) if path.is_symlink() else None
        return FileInfo(path.name, stat.S_ISDIR(info.st_mode), info.st_size, symlink_target=target)

    def _make_directory(self, remote_path: str, *, create_parents: bool = True) -> None:
        with _host_errors(remote_path):
            Path(remote_path).mkdir(parents=create_parents, exist_ok=True)

    def _remove(self, remote_path: str, *, recursive: bool = False) -> None:
        path = Path(remote_path)
        with _host_errors(remote_path):
            if path.is_dir() and recursive:
                shutil.rmtree(path)
            else:
                path.unlink()


class FakeSandbox:
    def __init__(self, control: FakeModal, object_id: str, name: str | None = None) -> None:
        self._control = control
        self.object_id = object_id
        self.name = name
        self.exec_calls: list[ExecCall] = []
        self.start_scripts: list[str] = []
        self.exec = _HangingExec(self._exec) if control.exec_hangs else _AioCallable(self._exec)
        self.poll = _AioCallable(self._poll)
        # Filesystem state the tests read and write.
        self.files: dict[str, bytes] = {}
        self.directories: set[str] = set()
        self.removals: list[tuple[str, bool]] = []
        self.listing: list[FileInfo] = []
        self.fs_error: Exception | None = None
        self.poll_result: int | None = None
        self.poll_error: Exception | None = None
        self.shutting_down = False
        self.terminate = _AioCallable(self._terminate)
        self.workdir: str | None = None
        self._filesystem: _FakeFilesystem | _HostFilesystem = _FakeFilesystem(self)
        if control.host_root is not None:
            self.exec = _HostExec(self._host_exec)
            self._filesystem = _HostFilesystem()

    @property
    def filesystem(self) -> _FakeFilesystem | _HostFilesystem:
        return self._filesystem

    def _terminate(self) -> None:
        # Like real Modal right after `terminate()`: the sandbox still resolves by id and polls
        # as running while it shuts down, and exec is refused with a `ConflictError`. The
        # in-memory filesystem fails generically, as Modal's does.
        self.shutting_down = True
        self.fs_error = FakeSandboxFilesystemError('An unexpected error occurred, please contact support@modal.com')

    def _host_exec(
        self,
        *args: str,
        timeout: int | None = None,
        workdir: str | None = None,
        env: dict[str, str | None] | None = None,
        text: bool = True,
    ) -> _FakeProcess:
        # Runs the command on the host, rooted at the sandbox's working directory, so the
        # conformance suite sees real exit codes, output, `cwd`, `env`, and deadlines.
        argv = list(args)
        # macOS lacks util-linux setsid. Start the wrapper in its own group so the
        # backend's stop exec can signal the same group it would signal on Modal.
        isolated = argv[:4] == ['setsid', '-w', 'sh', '-c']
        if isolated:
            self.start_scripts.append(argv[4])
            argv = argv[2:]
        self.exec_calls.append(ExecCall(argv=argv, timeout=timeout, text=text, workdir=workdir, env=env))
        if self.shutting_down:
            raise FakeConflictError('Modal Sandbox is shutting down.')
        assert self._control.host_root is not None
        variables = {**os.environ, **{key: value for key, value in (env or {}).items() if value is not None}}
        cwd = workdir or self.workdir or str(self._control.host_root)
        try:
            process = subprocess.Popen(
                argv, cwd=cwd, env=variables, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
            )
            assert process.stdout is not None and process.stderr is not None
            streams = {process.stdout: bytearray(), process.stderr: bytearray()}
            active = list(streams)
            for stream in streams:
                os.set_blocking(stream.fileno(), False)
            deadline = time.monotonic() + timeout if timeout is not None else None
            exited_at: float | None = None
            # Modal reports exit separately from pipe EOF; drain while running to avoid
            # pipe backpressure, then allow a short grace for output already in flight.
            while active:
                now = time.monotonic()
                if process.poll() is not None and exited_at is None:
                    exited_at = now
                if exited_at is not None and now - exited_at >= 0.1:
                    break
                if deadline is not None and now >= deadline and exited_at is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    stdout, stderr = bytes(streams[process.stdout]), bytes(streams[process.stderr])
                    process.stdout.close()
                    process.stderr.close()
                    return _FakeProcess(stdout, stderr, -1, None, False)
                ready = select.select(active, [], [], 0.01)[0]
                for stream in ready:
                    chunk = os.read(stream.fileno(), 65536)
                    if chunk:
                        streams[stream].extend(chunk)
                    else:
                        active.remove(stream)
            code = process.wait()
            stdout, stderr = bytes(streams[process.stdout]), bytes(streams[process.stderr])
            process.stdout.close()
            process.stderr.close()
        except FileNotFoundError:
            # A program that doesn't exist is an ordinary exit code 127, as from `sh`.
            return _FakeProcess(b'', b'', 127, None, False)
        return _FakeProcess(stdout, stderr, code, None, False)

    def _exec(
        self,
        *args: str,
        timeout: int | None = None,
        workdir: str | None = None,
        env: dict[str, str | None] | None = None,
        text: bool = True,
    ) -> _FakeProcess:
        # Closed keyword signature on purpose: real `Sandbox.exec` rejects unknown kwargs,
        # so the fake must too, or a bad kwarg in the backend would only fail in production.
        argv = list(args)
        stopping = argv[:4] == ['sh', '-c', argv[2] if len(argv) > 2 else '', 'modal-stop']
        if argv[:4] == ['setsid', '-w', 'sh', '-c']:
            self.start_scripts.append(argv[4])
            argv = argv[8:]
        self.exec_calls.append(ExecCall(argv=argv, timeout=timeout, text=text, workdir=workdir, env=env))
        if stopping:
            return _FakeProcess(b'', b'', 0, None, False)
        if self.shutting_down:
            raise FakeConflictError('Modal Sandbox is shutting down.')
        if self._control.exec_error is not None:
            raise self._control.exec_error
        # The command wrapper runs argv through sh; answer for the actual program.
        program = argv[4:] if argv[:4] == ['/bin/sh', '-c', 'exec "$@"', 'sh'] else argv
        # In-memory fake stores only regular files; its generic responder does not know FIFOs.
        probing_fifo = program[:2] == ['test', '-p']
        stdout, stderr, code = ('', '', 1) if probing_fifo else self._control.responder(program, timeout)
        return _FakeProcess(
            _stream_bytes(stdout),
            _stream_bytes(stderr),
            code,
            self._control.wait_error,
            self._control.wait_hangs and not probing_fifo,
            self._control.stdout_hangs and not probing_fifo,
            self._control.stdout_delay,
        )

    def _poll(self) -> int | None:
        if self.poll_error is not None:
            raise self.poll_error
        if self.poll_result is not None:
            return self.poll_result
        return None


class FakeModal:
    """Control surface for the injected fake `modal` module."""

    def __init__(self) -> None:
        self.responder: Responder = _echo_responder
        self.sandboxes: list[FakeSandbox] = []
        self.create_kwargs: list[dict[str, object]] = []
        self.app_lookups: list[dict[str, object]] = []
        self.image_tags: list[str] = []
        self.attach_ids: list[str] = []
        self.owned_creates = 0
        self.create_error: Exception | None = None
        self.create_reply_error: Exception | None = None
        self.create_gate: anyio.Event | None = None
        self.create_before_gate = False
        self.create_started = False
        self.attach_error: Exception | None = None
        self.attach_poll_result: int | None = None
        self.exec_error: Exception | None = None
        self.exec_hangs = False
        self.wait_error: Exception | None = None
        self.wait_hangs = False
        self.stdout_hangs = False
        # Seconds stdout takes to drain after the process has exited.
        self.stdout_delay = 0.0
        # When set, sandboxes run commands and file operations on the host under this directory.
        self.host_root: Path | None = None
        self.module = self._build_module()

    def exception(self, name: str) -> type[Exception]:
        """The fake of `modal.exception.<name>`."""
        return _EXCEPTIONS[name]

    def _build_module(self) -> types.ModuleType:
        control = self
        module = types.ModuleType('modal')

        def app_lookup(name: str, *, create_if_missing: bool = False) -> object:
            control.app_lookups.append({'name': name, 'create_if_missing': create_if_missing})
            return object()

        def image_from_registry(tag: str) -> object:
            # Closed signature on purpose, like `sandbox_create` below: signature drift in
            # the backend should fail here, not only in production.
            control.image_tags.append(tag)
            return object()

        def sandbox_create(
            *,
            app: object,
            image: object,
            name: str | None = None,
            timeout: int = 300,
            idle_timeout: int | None = None,
            workdir: str | None = None,
            env: dict[str, str | None] | None = None,
        ) -> FakeSandbox:
            if control.create_error is not None:
                raise control.create_error
            control.create_kwargs.append(
                {
                    'app': app,
                    'image': image,
                    'name': name,
                    'workdir': workdir,
                    'env': env,
                    'timeout': timeout,
                    'idle_timeout': idle_timeout,
                }
            )
            control.owned_creates += 1
            suffix = '' if control.owned_creates == 1 else f'-{control.owned_creates}'
            workspace = FakeSandbox(control, f'sb-owned{suffix}', name=name)
            workspace.workdir = workdir
            control.sandboxes.append(workspace)
            return workspace

        def sandbox_from_id(id: str) -> FakeSandbox:
            control.attach_ids.append(id)
            if control.attach_error is not None:
                raise control.attach_error
            existing = next((s for s in control.sandboxes if s.object_id == id), None)
            if existing is not None:
                return existing
            sandbox = FakeSandbox(control, id)
            sandbox.poll_result = control.attach_poll_result
            control.sandboxes.append(sandbox)
            return sandbox

        def workspace_from_name(app_name: str, name: str) -> FakeSandbox:
            existing = next((sandbox for sandbox in control.sandboxes if sandbox.name == name), None)
            if existing is None:
                raise _EXCEPTIONS['NotFoundError']('name not found')
            return existing

        class App:
            lookup = _AioCallable(app_lookup)

        def image_debian_slim(*, python_version: str) -> FakeImage:
            return FakeImage(python_version)

        class Image:
            from_registry = staticmethod(image_from_registry)
            debian_slim = staticmethod(image_debian_slim)

        class Sandbox:
            create = _GatedCreate(sandbox_create, control)
            from_id = _AioCallable(sandbox_from_id)
            from_name = _AioCallable(workspace_from_name)

        module.App = App  # type: ignore[attr-defined]
        module.Image = Image  # type: ignore[attr-defined]
        module.Sandbox = Sandbox  # type: ignore[attr-defined]
        module.exception = types.SimpleNamespace(**_EXCEPTIONS)  # type: ignore[attr-defined]
        return module


if TYPE_CHECKING:
    import modal.types

    class _FileInfoSurface(Protocol):
        """The `modal.types.FileInfo` members the backend reads.

        Pinned against both the fake and the real SDK type below, so a fake that drifts from
        Modal's own entry shape fails the type check instead of at the next live run.
        """

        @property
        def name(self) -> str: ...

        @property
        def size(self) -> int: ...

        @property
        def symlink_target(self) -> str | None: ...

        def is_dir(self) -> bool: ...

        def is_symlink(self) -> bool: ...

    _fake_file_info_conforms: _FileInfoSurface = FileInfo('name', False)
    _real_file_info_conforms: _FileInfoSurface = modal.types.FileInfo.__new__(modal.types.FileInfo)
