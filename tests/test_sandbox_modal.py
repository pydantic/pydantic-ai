"""Tests for the Modal sandbox backend.

The Modal SDK talks to a remote control plane over gRPC and its own worker command router, so
there is no HTTP boundary a cassette could record. The tests therefore stand fakes in for the two
SDK entry points `ModalSandbox` reaches for — `Sandbox` and `App` — while every type that crosses
the boundary between the SDK and our code (`FileInfo`, `FileType`, and the exceptions) is the real
one, so a change in those shapes fails here instead of in production. `modal.App` is the one
exception: it cannot be constructed without a live workspace behind it, and nothing about its
shape is load-bearing here — the backend only recognizes one and passes it through.
"""

from __future__ import annotations

import asyncio
import posixpath
import re
from collections.abc import Awaitable, Callable, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, cast

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.sandboxes import (
    Sandbox as SandboxFacade,
    SandboxBackend,
    SandboxRef,
    SupportsFilesystem,
    SupportsStart,
    SupportsStream,
)

from .conftest import try_import

with try_import() as imports_successful:
    from modal import Sandbox
    from modal.exception import ExecutionError, SandboxFilesystemNotFoundError, SandboxTerminatedError
    from modal.types import FileInfo, FileType

    from pydantic_ai.sandboxes.modal import ModalSandbox

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='modal not installed'),
]

WORKING_DIR = '/root'

Stream = Literal['stdout', 'stderr']


@dataclass(frozen=True)
class Aio:
    """Stand-in for a synchronicity dual sync/async callable; our code only ever uses `.aio`."""

    aio: Callable[..., Awaitable[Any]]


def _file_info(path: str, *, is_dir: bool, size: int, name: str | None = None) -> FileInfo:
    return FileInfo(
        name=name if name is not None else posixpath.basename(path),
        path=path,
        type=FileType.DIRECTORY if is_dir else FileType.FILE,
        size=size,
        mode=0o755 if is_dir else 0o644,
        permissions='drwxr-xr-x' if is_dir else '-rw-r--r--',
        owner='root',
        group='root',
        modified_time=1786000000.0,
        symlink_target=None,
    )


class FakeFilesystem:
    """Dictionary-backed stand-in for `Sandbox.filesystem`, matching the SDK's semantics."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.dirs: set[str] = {WORKING_DIR}
        self.made_dirs: list[tuple[str, bool]] = []
        self.removals: list[tuple[str, bool]] = []
        self.read_bytes = Aio(self._read_bytes)
        self.write_bytes = Aio(self._write_bytes)
        self.stat = Aio(self._stat)
        self.list_files = Aio(self._list_files)
        self.make_directory = Aio(self._make_directory)
        self.remove = Aio(self._remove)

    def _exists(self, path: str) -> bool:
        return path in self.files or path in self.dirs

    async def _read_bytes(self, remote_path: str) -> bytes:
        if remote_path not in self.files:
            raise SandboxFilesystemNotFoundError(remote_path)
        return self.files[remote_path]

    async def _write_bytes(self, data: bytes, remote_path: str) -> None:
        parent = posixpath.dirname(remote_path)
        while parent and parent != '/':  # the SDK's `write_bytes` creates missing parents
            self.dirs.add(parent)
            parent = posixpath.dirname(parent)
        self.files[remote_path] = data

    async def _stat(self, remote_path: str) -> FileInfo:
        if not self._exists(remote_path):
            raise SandboxFilesystemNotFoundError(remote_path)
        is_dir = remote_path in self.dirs
        return _file_info(remote_path, is_dir=is_dir, size=4096 if is_dir else len(self.files[remote_path]))

    async def _list_files(self, remote_path: str) -> list[FileInfo]:
        if remote_path not in self.dirs:
            raise SandboxFilesystemNotFoundError(remote_path)
        children = [child for child in [*self.files, *self.dirs] if posixpath.dirname(child) == remote_path]
        return [
            # A bare name, not the absolute path: the backend rebuilds the absolute path the
            # protocol promises from the directory it listed, and this keeps that load-bearing.
            _file_info(
                posixpath.basename(child),
                is_dir=child in self.dirs,
                size=4096 if child in self.dirs else len(self.files[child]),
            )
            for child in sorted(children)
        ]

    async def _make_directory(self, remote_path: str, *, create_parents: bool = True) -> None:
        self.made_dirs.append((remote_path, create_parents))
        self.dirs.add(remote_path)

    async def _remove(self, remote_path: str, *, recursive: bool = False) -> None:
        self.removals.append((remote_path, recursive))
        if not self._exists(remote_path):
            raise SandboxFilesystemNotFoundError(remote_path)
        self.dirs.discard(remote_path)
        for target in [
            target for target in self.files if target == remote_path or target.startswith(f'{remote_path}/')
        ]:
            del self.files[target]


class FakeStreamReader:
    """Stand-in for `modal.io_streams.StreamReader`.

    Async-iterable like the SDK's, and `read()` picks up wherever iteration stopped — that
    resumption is what lets `wait()` collect the output an abandoned `stream()` left behind.
    Sharing an `order` list between the two readers makes their chunks arrive in one
    deterministic sequence instead of whichever the event loop happens to pick.
    """

    def __init__(self, chunks: Sequence[str] = (), *, name: Stream = 'stdout', order: list[Stream] | None = None):
        self._chunks = list(chunks)
        self._name = name
        self._order = order
        self.read = Aio(self._read)

    def __aiter__(self) -> FakeStreamReader:
        return self

    async def __anext__(self) -> str:
        if not self._chunks:
            raise StopAsyncIteration
        while self._order and self._order[0] != self._name:
            await asyncio.sleep(0)  # not this stream's turn yet; let the other reader go first
        if self._order:
            self._order.pop(0)
        return self._chunks.pop(0)

    async def _read(self) -> str:
        rest, self._chunks = ''.join(self._chunks), []
        return rest


class FakeContainerProcess:
    """Stand-in for `ContainerProcess`; reports a non-zero exit as a return value, as the SDK does."""

    def __init__(
        self,
        *,
        stdout: FakeStreamReader | str = '',
        stderr: FakeStreamReader | str = '',
        exit_code: int = 0,
        error: Exception | None = None,
    ) -> None:
        self.stdout = FakeStreamReader([stdout] if stdout else []) if isinstance(stdout, str) else stdout
        self.stderr = FakeStreamReader([stderr] if stderr else [], name='stderr') if isinstance(stderr, str) else stderr
        self._exit_code = exit_code
        self._error = error
        self.wait = Aio(self._wait)

    async def _wait(self) -> int:
        if self._error is not None:
            raise self._error
        return self._exit_code


@dataclass(frozen=True)
class ExecCall:
    argv: tuple[str, ...]
    workdir: str | None
    env: dict[str, str] | None
    timeout: int | None


Responder = Callable[[tuple[str, ...]], FakeContainerProcess]


def default_responder(argv: tuple[str, ...]) -> FakeContainerProcess:
    stdout = f'{WORKING_DIR}\n' if argv == ('pwd',) else f'ran:{" ".join(argv)}'
    return FakeContainerProcess(stdout=stdout)


class FakeSandbox:
    def __init__(self, sandbox_id: str = 'sb-1', responder: Responder = default_responder) -> None:
        self.object_id = sandbox_id
        self.filesystem = FakeFilesystem()
        self.calls: list[ExecCall] = []
        self.processes: list[FakeContainerProcess] = []
        self.terminations = 0
        self._responder = responder
        self.exec = Aio(self._exec)
        self.terminate = Aio(self._terminate)

    async def _exec(
        self,
        *argv: str,
        timeout: int | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
        text: bool = True,
    ) -> FakeContainerProcess:
        self.calls.append(ExecCall(argv=argv, workdir=workdir, env=env, timeout=timeout))
        process = self._responder(argv)
        self.processes.append(process)
        return process

    async def _terminate(self) -> None:
        self.terminations += 1


class FakeSandboxApi:
    """Stand-in for the `Sandbox` class itself, recording how it was asked for a sandbox."""

    def __init__(self, responder: Responder = default_responder) -> None:
        self.created: list[dict[str, Any]] = []
        self.connected: list[dict[str, Any]] = []
        self.sandboxes: list[FakeSandbox] = []
        self._responder = responder
        self.create = Aio(self._create)
        self.from_id = Aio(self._from_id)

    def _new(self, sandbox_id: str) -> FakeSandbox:
        sandbox = FakeSandbox(sandbox_id, self._responder)
        self.sandboxes.append(sandbox)
        return sandbox

    async def _create(self, **kwargs: Any) -> FakeSandbox:
        self.created.append(kwargs)
        return self._new(f'sb-{len(self.sandboxes) + 1}')

    async def _from_id(self, sandbox_id: str, **kwargs: Any) -> FakeSandbox:
        self.connected.append({'sandbox_id': sandbox_id, **kwargs})
        return self._new(sandbox_id)


class FakeApp:
    """Stand-in for the `App` class: `create()` both recognizes one and looks one up by name."""

    lookups: ClassVar[list[dict[str, Any]]] = []
    lookup: ClassVar[Aio]

    def __init__(self, name: str) -> None:
        self.name = name


async def _lookup_app(name: str, *, create_if_missing: bool = False, client: Any = None) -> FakeApp:
    FakeApp.lookups.append({'name': name, 'create_if_missing': create_if_missing, 'client': client})
    return FakeApp(name)


FakeApp.lookup = Aio(_lookup_app)


class FakeClock:
    """Replaces the backend's `time`, so a deadline can elapse without the test spending it."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def modal_api(monkeypatch: pytest.MonkeyPatch) -> FakeSandboxApi:
    """Replace the SDK entry points `ModalSandbox.create`/`connect` reach for."""
    api = FakeSandboxApi()
    monkeypatch.setattr('pydantic_ai.sandboxes.modal.Sandbox', api)
    monkeypatch.setattr('pydantic_ai.sandboxes.modal.App', FakeApp)
    monkeypatch.setattr(FakeApp, 'lookups', [])
    return api


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr('pydantic_ai.sandboxes.modal.time', fake)
    return fake


def backend_for(sandbox: FakeSandbox) -> ModalSandbox:
    # The fake implements the `Sandbox` members `ModalSandbox` uses; the cast is what lets a test
    # hold one without a live Modal workspace behind it.
    return ModalSandbox(cast(Sandbox, sandbox))


async def test_backend_conforms_to_the_protocols_it_implements():
    backend = backend_for(FakeSandbox())
    assert isinstance(backend, SandboxBackend)
    assert isinstance(backend, SupportsFilesystem)
    assert isinstance(backend, SupportsStart)
    # Unlike E2B's, Modal's output streams are async-iterable, so its processes are streamable.
    assert isinstance(await backend.start(['true']), SupportsStream)


async def test_create_provisions_a_sandbox_and_terminates_it_on_exit(modal_api: FakeSandboxApi):
    app = FakeApp('held-app')
    async with ModalSandbox.create(
        app=cast(Any, app), timeout=600, workdir='/work', env={'STAGE': 'test'}, cpu=2.0, memory=4096
    ) as backend:
        assert (backend.provider, backend.sandbox_id) == ('modal', 'sb-1')
        assert modal_api.sandboxes[0].terminations == 0
    assert modal_api.created == [
        {
            'app': app,
            'image': None,
            'timeout': 600,
            'workdir': '/work',
            'env': {'STAGE': 'test'},
            'cpu': 2.0,
            'memory': 4096,
            'client': None,
        }
    ]
    assert FakeApp.lookups == []  # an app that is already held is used as-is
    assert modal_api.sandboxes[0].terminations == 1


async def test_create_looks_up_the_default_app_by_name(modal_api: FakeSandboxApi):
    """Every Modal sandbox needs an app, so a name is resolved into one — creating it if needed."""
    async with ModalSandbox.create():
        pass
    assert FakeApp.lookups == [{'name': 'pydantic-ai-sandbox', 'create_if_missing': True, 'client': None}]
    assert modal_api.created[0]['app'].name == 'pydantic-ai-sandbox'


async def test_create_terminates_the_sandbox_when_the_block_raises(modal_api: FakeSandboxApi):
    with pytest.raises(RuntimeError, match='boom'):
        async with ModalSandbox.create():
            raise RuntimeError('boom')
    assert modal_api.sandboxes[0].terminations == 1


async def test_connect_attaches_to_an_existing_sandbox_without_owning_it(modal_api: FakeSandboxApi):
    """`connect` is the resolver building block: a ref round-trips into a live backend, and the
    caller that provisioned the environment keeps the right to destroy it.
    """
    ref = SandboxRef(provider='modal', sandbox_id='sb-existing')
    backend = await ModalSandbox.connect(ref.sandbox_id)

    assert SandboxRef(provider=backend.provider, sandbox_id=backend.sandbox_id) == ref
    assert modal_api.connected == [{'sandbox_id': 'sb-existing', 'client': None}]
    assert modal_api.sandboxes[0].terminations == 0


@pytest.mark.parametrize(
    ('command', 'shell', 'expected'),
    [
        (['echo', 'hello world'], False, ('echo', 'hello world')),
        (['sh', '-c', 'rm -rf /'], False, ('sh', '-c', 'rm -rf /')),
        ('echo $HOME | wc -c', True, ('/bin/sh', '-c', 'echo $HOME | wc -c')),
    ],
    ids=['argv-verbatim', 'argv-stays-separate-words', 'shell-gets-a-shell'],
)
async def test_command_reaches_modal_as_argv(command: str | Sequence[str], shell: bool, expected: tuple[str, ...]):
    """Modal only executes argv, so argv goes straight through and a shell string is handed to a shell."""
    sandbox = FakeSandbox()
    await backend_for(sandbox).run(command, shell=shell)
    assert [call.argv for call in sandbox.calls] == [expected]


@pytest.mark.parametrize(
    ('command', 'shell', 'message'),
    [
        ('echo hello', False, 'a string command requires shell=True'),
        (['echo', 'hello'], True, 'an argv sequence cannot be combined with shell=True'),
    ],
    ids=['string-without-shell', 'argv-with-shell'],
)
async def test_run_rejects_a_command_shape_that_contradicts_shell(
    command: str | Sequence[str], shell: bool, message: str
):
    with pytest.raises(TypeError, match=message):
        await backend_for(FakeSandbox()).run(command, shell=shell)


async def test_run_forwards_cwd_and_env():
    sandbox = FakeSandbox()
    await backend_for(sandbox).run(['true'], cwd='/tmp', env={'TOKEN': 'secret'})
    assert sandbox.calls == [ExecCall(argv=('true',), workdir='/tmp', env={'TOKEN': 'secret'}, timeout=None)]


@pytest.mark.parametrize(
    ('timeout', 'expected'),
    [(None, None), (0.01, 1), (1.5, 2), (30.0, 30)],
    ids=['unbounded', 'sub-second', 'fractional', 'whole'],
)
async def test_the_deadline_reaches_modal_as_whole_seconds(timeout: float | None, expected: int | None):
    """Modal takes whole seconds and reads a missing deadline as unbounded, so a sub-second
    deadline has to round up rather than round away to nothing.
    """
    sandbox = FakeSandbox()
    await backend_for(sandbox).run(['true'], timeout=timeout)
    assert [call.timeout for call in sandbox.calls] == [expected]


async def test_a_non_zero_exit_is_a_result_not_an_exception():
    sandbox = FakeSandbox(
        responder=lambda argv: FakeContainerProcess(stdout='partial', stderr='not found', exit_code=127)
    )
    result = await backend_for(sandbox).run(['missing-binary'])
    assert (result.exit_code, result.stdout, result.stderr) == (127, 'partial', 'not found')
    assert (result.stdout_dropped, result.stderr_dropped) == (0, 0)


@pytest.mark.parametrize('exit_code', [-1, 137], ids=['client-deadline', 'server-sigkill'])
async def test_a_deadline_kill_raises_a_builtin_timeout_error(clock: FakeClock, exit_code: int):
    """Modal's deadline kills the command itself, reporting `-1` when its client-side copy of the
    deadline won the race and the plain SIGKILL exit when the server's kill did.
    """
    sandbox = FakeSandbox(responder=lambda argv: FakeContainerProcess(exit_code=exit_code))
    process = await backend_for(sandbox).start(['sleep', '600'], timeout=0.01)
    clock.now = 60.0  # the deadline window has passed
    with pytest.raises(TimeoutError, match=re.escape('command timed out after 0.01 seconds and was killed')):
        await process.wait()


async def test_a_sigkill_the_deadline_cannot_explain_is_still_a_result(clock: FakeClock):
    """A command can exit 137 on its own account — an OOM kill, a `kill -9` it asked for — so
    that exit is only read as a timeout once the deadline window has actually elapsed.
    """
    sandbox = FakeSandbox(responder=lambda argv: FakeContainerProcess(exit_code=137))
    result = await backend_for(sandbox).run(['self-destruct'], timeout=30)
    assert result.exit_code == 137


async def test_a_deadline_kill_is_not_read_into_a_command_that_had_none():
    sandbox = FakeSandbox(responder=lambda argv: FakeContainerProcess(exit_code=137))
    result = await backend_for(sandbox).run(['self-destruct'])
    assert result.exit_code == 137


async def test_output_limit_is_not_supported():
    with pytest.raises(NotImplementedError, match='does not bound output'):
        await backend_for(FakeSandbox()).run(['true'], output_limit=100)


async def test_an_sdk_failure_surfaces_instead_of_becoming_a_fake_exit_code():
    """An environment that is gone is an infrastructure failure, never `exit_code=1`."""
    gone = SandboxTerminatedError('sandbox sb-1 is no longer running')
    sandbox = FakeSandbox(responder=lambda argv: FakeContainerProcess(error=gone))
    with pytest.raises(SandboxTerminatedError) as exc_info:
        await backend_for(sandbox).run(['true'])
    assert exc_info.value is gone


async def test_a_started_process_reports_no_pid_and_cannot_be_killed():
    """Modal names a command by an exec id of its own and offers no way to kill one, so the
    protocol's escape hatch applies: say so, and name the deadline as the way to bound a command.
    """
    process = await backend_for(FakeSandbox()).start(['sleep', '600'])
    assert process.pid is None
    with pytest.raises(NotImplementedError, match='start it with `timeout=`'):
        await process.kill()


async def test_waiting_twice_reports_the_same_outcome(clock: FakeClock):
    """The protocol requires repeated waits to agree, and after a timeout the command is gone,
    so only the first wait can reach it.
    """
    sandbox = FakeSandbox(responder=lambda argv: FakeContainerProcess(exit_code=-1))
    process = await backend_for(sandbox).start(['sleep', '600'], timeout=0.01)
    with pytest.raises(TimeoutError) as first:
        await process.wait()
    with pytest.raises(TimeoutError) as second:
        await process.wait()
    assert second.value is first.value

    finished = await backend_for(FakeSandbox()).start(['true'])
    assert await finished.wait() == await finished.wait()


async def test_stream_yields_both_streams_in_arrival_order():
    order: list[Stream] = ['stdout', 'stderr', 'stdout']
    process = FakeContainerProcess(
        stdout=FakeStreamReader(['one\n', 'three\n'], name='stdout', order=order),
        stderr=FakeStreamReader(['two\n'], name='stderr', order=order),
    )
    started = await backend_for(FakeSandbox(responder=lambda argv: process)).start(['noisy'])
    assert isinstance(started, SupportsStream)

    chunks = [(chunk.stream, chunk.data) async for chunk in started.stream()]
    assert chunks == [('stdout', 'one\n'), ('stderr', 'two\n'), ('stdout', 'three\n')]


async def test_wait_after_streaming_reports_what_was_streamed():
    """`stream()` consumes the output, so `wait()` has to remember it rather than re-read it."""
    order: list[Stream] = ['stdout', 'stderr']
    process = FakeContainerProcess(
        stdout=FakeStreamReader(['done\n'], name='stdout', order=order),
        stderr=FakeStreamReader(['warned\n'], name='stderr', order=order),
        exit_code=3,
    )
    started = await backend_for(FakeSandbox(responder=lambda argv: process)).start(['noisy'])

    async for _ in started.stream():
        pass
    result = await started.wait()
    assert (result.exit_code, result.stdout, result.stderr) == (3, 'done\n', 'warned\n')


async def test_output_delivered_to_an_abandoned_stream_still_reaches_wait():
    """A consumer that stops iterating leaves an already-delivered chunk behind. Modal cannot be
    asked for it again, so it is part of the command's output and `wait()` still reports it.
    """
    process = FakeContainerProcess(stdout='out', stderr='err')
    started = await backend_for(FakeSandbox(responder=lambda argv: process)).start(['noisy'])

    # Both streams delivered in the same wake-up; only one of the two is consumed here. Closing
    # the iterator explicitly is what a `break` leaves to the garbage collector to do later.
    async with aclosing(started.stream()) as chunks:
        async for _ in chunks:
            break

    result = await started.wait()
    assert (result.stdout, result.stderr) == ('out', 'err')


async def test_working_dir_asks_the_environment_once():
    """Modal exposes no API for a running sandbox's working directory, so the sandbox is asked,
    and the answer — which cannot change — is kept.
    """
    sandbox = FakeSandbox()
    backend = backend_for(sandbox)
    assert await backend.working_dir() == WORKING_DIR
    assert await backend.working_dir() == WORKING_DIR
    assert [call.argv for call in sandbox.calls] == [('pwd',)]


async def test_working_dir_reports_an_environment_that_cannot_answer():
    sandbox = FakeSandbox(responder=lambda argv: FakeContainerProcess(stderr='pwd: not found', exit_code=127))
    with pytest.raises(ExecutionError, match=r"working directory of sandbox 'sb-1'.+`pwd` exited 127"):
        await backend_for(sandbox).working_dir()


@pytest.mark.parametrize(
    'data',
    [b'', b'plain text\n', b'\x00\x01\xff\xfe binary \x80', bytes(range(256))],
    ids=['empty', 'text', 'invalid-utf8', 'every-byte'],
)
async def test_bytes_round_trip_unchanged(data: bytes):
    backend = backend_for(FakeSandbox())
    await backend.fs.write_bytes(f'{WORKING_DIR}/blob.bin', data)
    assert await backend.fs.read_bytes(f'{WORKING_DIR}/blob.bin') == data


async def test_write_creates_missing_parent_directories():
    sandbox = FakeSandbox()
    await backend_for(sandbox).fs.write_bytes(f'{WORKING_DIR}/deep/nested/file.txt', b'hi')
    assert f'{WORKING_DIR}/deep/nested' in sandbox.filesystem.dirs


async def test_stat_reports_a_file_and_a_directory():
    """A directory's reported size is filesystem bookkeeping, not content, so it is dropped."""
    backend = backend_for(FakeSandbox())
    await backend.fs.write_bytes(f'{WORKING_DIR}/notes.txt', b'hello')

    file_entry = await backend.fs.stat(f'{WORKING_DIR}/notes.txt')
    assert (file_entry.name, file_entry.path, file_entry.is_dir, file_entry.size) == (
        'notes.txt',
        f'{WORKING_DIR}/notes.txt',
        False,
        5,
    )
    directory = await backend.fs.stat(WORKING_DIR)
    assert (directory.name, directory.is_dir, directory.size) == ('root', True, None)


async def test_list_dir_reports_children_at_absolute_paths():
    """Modal names a listed entry relative to the directory it listed; the protocol promises an
    absolute path, so the backend rebuilds one.
    """
    backend = backend_for(FakeSandbox())
    await backend.fs.write_bytes(f'{WORKING_DIR}/a.txt', b'a')
    await backend.fs.write_bytes(f'{WORKING_DIR}/sub/b.txt', b'bb')

    entries = await backend.fs.list_dir(WORKING_DIR)
    assert [(entry.path, entry.is_dir, entry.size) for entry in entries] == [
        (f'{WORKING_DIR}/a.txt', False, 1),
        (f'{WORKING_DIR}/sub', True, None),
    ]


async def test_make_dir_is_mkdir_p():
    sandbox = FakeSandbox()
    backend = backend_for(sandbox)
    await backend.fs.make_dir(f'{WORKING_DIR}/work')
    await backend.fs.make_dir(f'{WORKING_DIR}/work')  # an existing directory is not an error
    assert sandbox.filesystem.made_dirs == [(f'{WORKING_DIR}/work', True), (f'{WORKING_DIR}/work', True)]


async def test_remove_deletes_files_and_directories():
    sandbox = FakeSandbox()
    backend = backend_for(sandbox)
    await backend.fs.write_bytes(f'{WORKING_DIR}/tree/leaf.txt', b'x')

    await backend.fs.remove(f'{WORKING_DIR}/tree')
    assert await backend.fs.exists(f'{WORKING_DIR}/tree') is False
    assert await backend.fs.exists(f'{WORKING_DIR}/tree/leaf.txt') is False
    # Recursive, because a directory with contents is a directory the protocol must remove.
    assert sandbox.filesystem.removals == [(f'{WORKING_DIR}/tree', True)]


async def test_exists_answers_for_a_file_that_is_there():
    backend = backend_for(FakeSandbox())
    await backend.fs.write_bytes(f'{WORKING_DIR}/here.txt', b'x')
    assert await backend.fs.exists(f'{WORKING_DIR}/here.txt') is True


@pytest.mark.parametrize('operation', ['read_bytes', 'stat', 'list_dir', 'remove'])
async def test_a_missing_path_raises_the_sdk_error_unchanged(operation: str):
    """File errors are Modal's to describe: they pass through rather than being renamed."""
    filesystem = backend_for(FakeSandbox()).fs
    with pytest.raises(SandboxFilesystemNotFoundError):
        await getattr(filesystem, operation)(f'{WORKING_DIR}/nope')


async def test_the_sandbox_serves_an_agent_run():
    """The end a user sees: commands and files reach the same environment through
    [`RunContext.sandbox`][pydantic_ai.tools.RunContext.sandbox], with relative paths resolved
    against the sandbox's own working directory.
    """
    sandbox = FakeSandbox()
    seen: list[str] = []
    agent: Agent = Agent(TestModel())  # TestModel calls every registered tool, then returns text

    @agent.tool
    async def note(ctx: RunContext[Any]) -> str:
        await ctx.sandbox.write_text('note.txt', 'from the model')
        seen.append(await ctx.sandbox.read_text('note.txt'))
        return 'ok'

    await agent.run('go', sandbox=backend_for(sandbox))

    assert seen == ['from the model']
    assert sandbox.filesystem.files == {f'{WORKING_DIR}/note.txt': b'from the model'}


async def test_the_facade_exposes_the_backend_for_provider_specific_work():
    sandbox = FakeSandbox()
    backend = backend_for(sandbox)
    assert SandboxFacade(backend).backend is backend
    assert backend.sandbox is cast(Sandbox, sandbox)
