"""Tests for the E2B sandbox backend.

The E2B SDK talks to a remote control plane over its own RPC transport, so there is no HTTP
boundary a cassette could record. The tests therefore stand a fake in for `AsyncSandbox` — the
one object `E2BSandbox` holds — while every type that crosses the boundary between the SDK and
our code (`EntryInfo`, `CommandResult`, and the exceptions) is the real one, so a change in
those shapes fails here instead of in production.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, cast

import anyio
import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.sandboxes import (
    Sandbox,
    SandboxBackend,
    SandboxRef,
    SupportsFilesystem,
    SupportsStart,
    SupportsStream,
)

from .conftest import try_import

with try_import() as imports_successful:
    from e2b import (
        AsyncSandbox,
        CommandExitException,
        CommandResult,
        EntryInfo,
        FileNotFoundException,
        FileType,
        SandboxException,
        WriteInfo,
    )

    from pydantic_ai.sandboxes.e2b import E2BSandbox

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='e2b not installed'),
]

WORKING_DIR = '/home/user'


def _entry(path: str, *, is_dir: bool, size: int) -> EntryInfo:
    return EntryInfo(
        name=posixpath.basename(path),
        type=FileType.DIR if is_dir else FileType.FILE,
        path=path,
        size=size,
        mode=0o755 if is_dir else 0o644,
        permissions='drwxr-xr-x' if is_dir else '-rw-r--r--',
        owner='user',
        group='user',
        modified_time=datetime(2026, 8, 17, tzinfo=timezone.utc),
    )


class FakeFiles:
    """Dictionary-backed stand-in for `AsyncSandbox.files`, matching the SDK's semantics."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.dirs: set[str] = {WORKING_DIR}
        self.made_dirs: list[str] = []
        self.removed: list[str] = []

    def _exists(self, path: str) -> bool:
        return path in self.files or path in self.dirs

    async def read(self, path: str, format: Literal['bytes'] = 'bytes') -> bytearray:
        if path not in self.files:
            raise FileNotFoundException(path)
        # The SDK's byte read hands back a `bytearray`, not `bytes`.
        return bytearray(self.files[path])

    async def write(self, path: str, data: bytes) -> WriteInfo:
        parent = posixpath.dirname(path)
        while parent and parent != '/':  # the SDK's `write` creates missing parents
            self.dirs.add(parent)
            parent = posixpath.dirname(parent)
        self.files[path] = data
        return WriteInfo(name=posixpath.basename(path), type=FileType.FILE, path=path)

    async def get_info(self, path: str) -> EntryInfo:
        if not self._exists(path):
            raise FileNotFoundException(path)
        is_dir = path in self.dirs
        return _entry(path, is_dir=is_dir, size=4096 if is_dir else len(self.files[path]))

    async def list(self, path: str) -> list[EntryInfo]:
        if path not in self.dirs:
            raise FileNotFoundException(path)
        children = [child for child in [*self.files, *self.dirs] if posixpath.dirname(child) == path]
        return [
            _entry(child, is_dir=child in self.dirs, size=4096 if child in self.dirs else len(self.files[child]))
            for child in sorted(children)
        ]

    async def make_dir(self, path: str) -> bool:
        self.made_dirs.append(path)
        created = path not in self.dirs
        self.dirs.add(path)
        return created  # the SDK reports `False` for an existing directory rather than raising

    async def remove(self, path: str) -> None:
        self.removed.append(path)
        if not self._exists(path):
            raise FileNotFoundException(path)
        self.dirs.discard(path)
        for target in [target for target in self.files if target == path or target.startswith(f'{path}/')]:
            del self.files[target]

    async def exists(self, path: str) -> bool:
        return self._exists(path)


class FakeCommandHandle:
    """Stand-in for `AsyncCommandHandle`; reports non-zero exits the way the SDK does."""

    def __init__(self, result: CommandResult | None = None, *, error: Exception | None = None) -> None:
        self.pid = 4242
        self.kills = 0
        self._result = result
        self._error = error

    async def wait(self) -> CommandResult:
        if self._error is not None:
            raise self._error
        assert self._result is not None
        if self._result.exit_code != 0:
            raise CommandExitException(
                stderr=self._result.stderr,
                stdout=self._result.stdout,
                exit_code=self._result.exit_code,
                error=self._result.error,
            )
        return self._result

    async def kill(self) -> bool:
        self.kills += 1
        return True


class HangingCommandHandle(FakeCommandHandle):
    """A command that never finishes, so a caller's deadline is the only thing that can end it."""

    def __init__(self) -> None:
        super().__init__(CommandResult(stderr='', stdout='', exit_code=0, error=None))

    async def wait(self) -> CommandResult:
        await anyio.sleep_forever()
        assert False


@dataclass(frozen=True)
class RunCall:
    command: str
    cwd: str | None
    envs: dict[str, str] | None
    timeout: float | None


Responder = Callable[[str], FakeCommandHandle]


def default_responder(command: str) -> FakeCommandHandle:
    stdout = f'{WORKING_DIR}\n' if command == 'pwd' else f'ran:{command}'
    return FakeCommandHandle(CommandResult(stderr='', stdout=stdout, exit_code=0, error=None))


class FakeCommands:
    def __init__(self, responder: Responder) -> None:
        self.calls: list[RunCall] = []
        self.handles: list[FakeCommandHandle] = []
        self._responder = responder

    async def run(
        self,
        cmd: str,
        *,
        background: Literal[True],
        envs: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float | None = 60,
    ) -> FakeCommandHandle:
        self.calls.append(RunCall(command=cmd, cwd=cwd, envs=envs, timeout=timeout))
        handle = self._responder(cmd)
        self.handles.append(handle)
        return handle


class FakeAsyncSandbox:
    def __init__(self, sandbox_id: str = 'sbx-1', responder: Responder = default_responder) -> None:
        self.sandbox_id = sandbox_id
        self.files = FakeFiles()
        self.commands = FakeCommands(responder)
        self.kills = 0

    async def kill(self) -> bool:
        self.kills += 1
        return True


class FakeSandboxApi:
    """Stand-in for the `AsyncSandbox` class itself, recording how it was asked for a sandbox."""

    def __init__(self, responder: Responder = default_responder) -> None:
        self.created: list[dict[str, Any]] = []
        self.connected: list[dict[str, Any]] = []
        self.sandboxes: list[FakeAsyncSandbox] = []
        self._responder = responder

    def _new(self, sandbox_id: str) -> FakeAsyncSandbox:
        sandbox = FakeAsyncSandbox(sandbox_id, self._responder)
        self.sandboxes.append(sandbox)
        return sandbox

    async def create(self, **kwargs: Any) -> FakeAsyncSandbox:
        self.created.append(kwargs)
        return self._new(f'sbx-{len(self.sandboxes) + 1}')

    async def connect(self, sandbox_id: str, **kwargs: Any) -> FakeAsyncSandbox:
        self.connected.append({'sandbox_id': sandbox_id, **kwargs})
        return self._new(sandbox_id)


@pytest.fixture
def sandbox_api(monkeypatch: pytest.MonkeyPatch) -> FakeSandboxApi:
    """Replace the SDK entry point `E2BSandbox.create`/`connect` reach for."""
    api = FakeSandboxApi()
    monkeypatch.setattr('pydantic_ai.sandboxes.e2b.AsyncSandbox', api)
    return api


def backend_for(sandbox: FakeAsyncSandbox) -> E2BSandbox:
    # The fake implements the `AsyncSandbox` members `E2BSandbox` uses; the cast is what lets a
    # test hold one without a live E2B account behind it.
    return E2BSandbox(cast(AsyncSandbox, sandbox))


async def test_backend_conforms_to_the_protocols_it_implements():
    backend = backend_for(FakeAsyncSandbox())
    assert isinstance(backend, SandboxBackend)
    assert isinstance(backend, SupportsFilesystem)
    assert isinstance(backend, SupportsStart)
    # E2B's async SDK reports live output through callbacks, so processes are not streamable.
    assert not isinstance(await backend.start(['true']), SupportsStream)


async def test_create_provisions_a_sandbox_and_kills_it_on_exit(sandbox_api: FakeSandboxApi):
    async with E2BSandbox.create(
        'python-3.13', timeout=600, envs={'STAGE': 'test'}, metadata={'run': 'r1'}, api_key='e2b_key'
    ) as backend:
        assert (backend.provider, backend.sandbox_id) == ('e2b', 'sbx-1')
        assert sandbox_api.sandboxes[0].kills == 0
    assert sandbox_api.created == [
        {
            'template': 'python-3.13',
            'timeout': 600,
            'envs': {'STAGE': 'test'},
            'metadata': {'run': 'r1'},
            'api_key': 'e2b_key',
        }
    ]
    assert sandbox_api.sandboxes[0].kills == 1


async def test_create_defaults_leave_every_choice_to_e2b(sandbox_api: FakeSandboxApi):
    async with E2BSandbox.create():
        pass
    assert sandbox_api.created == [{'template': None, 'timeout': None, 'envs': None, 'metadata': None}]


async def test_create_kills_the_sandbox_when_the_block_raises(sandbox_api: FakeSandboxApi):
    with pytest.raises(RuntimeError, match='boom'):
        async with E2BSandbox.create():
            raise RuntimeError('boom')
    assert sandbox_api.sandboxes[0].kills == 1


async def test_connect_attaches_to_an_existing_sandbox_without_owning_it(sandbox_api: FakeSandboxApi):
    """`connect` is the resolver building block: a ref round-trips into a live backend, and the
    caller that provisioned the environment keeps the right to destroy it.
    """
    ref = SandboxRef(provider='e2b', sandbox_id='sbx-existing')
    backend = await E2BSandbox.connect(ref.sandbox_id, api_key='e2b_key')

    assert SandboxRef(provider=backend.provider, sandbox_id=backend.sandbox_id) == ref
    assert sandbox_api.connected == [{'sandbox_id': 'sbx-existing', 'api_key': 'e2b_key'}]
    assert sandbox_api.sandboxes[0].kills == 0


@pytest.mark.parametrize(
    ('command', 'shell', 'expected'),
    [
        (['echo', 'hello world'], False, "echo 'hello world'"),
        (['sh', '-c', 'rm -rf /'], False, "sh -c 'rm -rf /'"),
        ('echo $HOME | wc -c', True, 'echo $HOME | wc -c'),
    ],
    ids=['argv-quoted', 'argv-stays-one-word', 'shell-verbatim'],
)
async def test_command_reaches_e2b_as_a_shell_string(command: str | Sequence[str], shell: bool, expected: str):
    """E2B only executes shell strings, so argv is quoted back into one word per element."""
    sandbox = FakeAsyncSandbox()
    await backend_for(sandbox).run(command, shell=shell)
    assert [call.command for call in sandbox.commands.calls] == [expected]


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
        await backend_for(FakeAsyncSandbox()).run(command, shell=shell)


async def test_run_forwards_cwd_and_env_and_owns_the_deadline():
    sandbox = FakeAsyncSandbox()
    await backend_for(sandbox).run(['true'], cwd='/tmp', env={'TOKEN': 'secret'}, timeout=30)
    assert sandbox.commands.calls == [
        # `timeout=0` disables E2B's own deadline: it would only drop the event stream, leaving
        # the command running, so `E2BSandbox` applies the caller's 30 seconds itself.
        RunCall(command='true', cwd='/tmp', envs={'TOKEN': 'secret'}, timeout=0)
    ]


async def test_a_non_zero_exit_is_a_result_not_an_exception():
    """The SDK raises `CommandExitException` for a failed command; the protocol calls that a
    normal result, so the exception is unpacked rather than propagated.
    """
    sandbox = FakeAsyncSandbox(
        responder=lambda command: FakeCommandHandle(
            CommandResult(stderr='not found', stdout='partial', exit_code=127, error='exit status 127')
        )
    )
    result = await backend_for(sandbox).run(['missing-binary'])
    assert (result.exit_code, result.stdout, result.stderr) == (127, 'partial', 'not found')
    assert (result.stdout_dropped, result.stderr_dropped) == (0, 0)


async def test_timeout_kills_the_command_before_raising():
    sandbox = FakeAsyncSandbox(responder=lambda command: HangingCommandHandle())
    with pytest.raises(TimeoutError, match=re.escape('command timed out after 0.01 seconds and was killed')):
        await backend_for(sandbox).run(['sleep', '600'], timeout=0.01)
    assert sandbox.commands.handles[0].kills == 1


async def test_output_limit_is_not_supported():
    with pytest.raises(NotImplementedError, match='does not bound output'):
        await backend_for(FakeAsyncSandbox()).run(['true'], output_limit=100)


async def test_an_sdk_failure_surfaces_instead_of_becoming_a_fake_exit_code():
    """An environment that is gone is an infrastructure failure, never `exit_code=1`."""
    gone = SandboxException('sandbox sbx-1 is no longer running')
    sandbox = FakeAsyncSandbox(responder=lambda command: FakeCommandHandle(error=gone))
    with pytest.raises(SandboxException) as exc_info:
        await backend_for(sandbox).run(['true'])
    assert exc_info.value is gone


async def test_start_hands_back_a_live_process():
    sandbox = FakeAsyncSandbox()
    process = await backend_for(sandbox).start(['sleep', '600'])
    assert process.pid == 4242
    await process.kill()
    assert sandbox.commands.handles[0].kills == 1


async def test_waiting_twice_reports_the_same_outcome():
    """The protocol requires repeated waits to agree, and after a timeout the command is gone,
    so only the first wait can reach it.
    """
    sandbox = FakeAsyncSandbox(responder=lambda command: HangingCommandHandle())
    process = await backend_for(sandbox).start(['sleep', '600'], timeout=0.01)
    with pytest.raises(TimeoutError) as first:
        await process.wait()
    with pytest.raises(TimeoutError) as second:
        await process.wait()
    assert second.value is first.value
    assert sandbox.commands.handles[0].kills == 1

    finished = await backend_for(FakeAsyncSandbox()).start(['true'])
    assert await finished.wait() == await finished.wait()


async def test_working_dir_asks_the_environment_once():
    """E2B exposes no API for the template's default directory, so the sandbox is asked, and the
    answer — which cannot change — is kept.
    """
    sandbox = FakeAsyncSandbox()
    backend = backend_for(sandbox)
    assert await backend.working_dir() == WORKING_DIR
    assert await backend.working_dir() == WORKING_DIR
    assert [call.command for call in sandbox.commands.calls] == ['pwd']


async def test_working_dir_reports_an_environment_that_cannot_answer():
    sandbox = FakeAsyncSandbox(
        responder=lambda command: FakeCommandHandle(
            CommandResult(stderr='pwd: not found', stdout='', exit_code=127, error='exit status 127')
        )
    )
    with pytest.raises(SandboxException, match=r"working directory of sandbox 'sbx-1'.+`pwd` exited 127"):
        await backend_for(sandbox).working_dir()


@pytest.mark.parametrize(
    'data',
    [b'', b'plain text\n', b'\x00\x01\xff\xfe binary \x80', bytes(range(256))],
    ids=['empty', 'text', 'invalid-utf8', 'every-byte'],
)
async def test_bytes_round_trip_unchanged(data: bytes):
    backend = backend_for(FakeAsyncSandbox())
    await backend.fs.write_bytes(f'{WORKING_DIR}/blob.bin', data)
    assert await backend.fs.read_bytes(f'{WORKING_DIR}/blob.bin') == data


async def test_write_creates_missing_parent_directories():
    sandbox = FakeAsyncSandbox()
    await backend_for(sandbox).fs.write_bytes(f'{WORKING_DIR}/deep/nested/file.txt', b'hi')
    assert f'{WORKING_DIR}/deep/nested' in sandbox.files.dirs


async def test_stat_reports_a_file_and_a_directory():
    """A directory's reported size is filesystem bookkeeping, not content, so it is dropped."""
    backend = backend_for(FakeAsyncSandbox())
    await backend.fs.write_bytes(f'{WORKING_DIR}/notes.txt', b'hello')

    file_entry = await backend.fs.stat(f'{WORKING_DIR}/notes.txt')
    assert (file_entry.name, file_entry.path, file_entry.is_dir, file_entry.size) == (
        'notes.txt',
        f'{WORKING_DIR}/notes.txt',
        False,
        5,
    )
    directory = await backend.fs.stat(WORKING_DIR)
    assert (directory.name, directory.is_dir, directory.size) == ('user', True, None)


async def test_list_dir_reports_children_only():
    backend = backend_for(FakeAsyncSandbox())
    await backend.fs.write_bytes(f'{WORKING_DIR}/a.txt', b'a')
    await backend.fs.write_bytes(f'{WORKING_DIR}/sub/b.txt', b'bb')

    entries = await backend.fs.list_dir(WORKING_DIR)
    assert [(entry.name, entry.is_dir, entry.size) for entry in entries] == [('a.txt', False, 1), ('sub', True, None)]


async def test_make_dir_is_mkdir_p():
    sandbox = FakeAsyncSandbox()
    backend = backend_for(sandbox)
    await backend.fs.make_dir(f'{WORKING_DIR}/work')
    await backend.fs.make_dir(f'{WORKING_DIR}/work')  # an existing directory is not an error
    assert sandbox.files.made_dirs == [f'{WORKING_DIR}/work', f'{WORKING_DIR}/work']


async def test_remove_deletes_files_and_directories():
    backend = backend_for(FakeAsyncSandbox())
    await backend.fs.write_bytes(f'{WORKING_DIR}/tree/leaf.txt', b'x')

    await backend.fs.remove(f'{WORKING_DIR}/tree')
    assert await backend.fs.exists(f'{WORKING_DIR}/tree') is False
    assert await backend.fs.exists(f'{WORKING_DIR}/tree/leaf.txt') is False


@pytest.mark.parametrize('operation', ['read_bytes', 'stat', 'list_dir', 'remove'])
async def test_a_missing_path_raises_the_sdk_error_unchanged(operation: str):
    """File errors are E2B's to describe: they pass through rather than being renamed."""
    filesystem = backend_for(FakeAsyncSandbox()).fs
    with pytest.raises(FileNotFoundException):
        await getattr(filesystem, operation)(f'{WORKING_DIR}/nope')


async def test_the_sandbox_serves_an_agent_run():
    """The end a user sees: commands and files reach the same environment through
    [`RunContext.sandbox`][pydantic_ai.tools.RunContext.sandbox], with relative paths resolved
    against the sandbox's own working directory.
    """
    sandbox = FakeAsyncSandbox()
    seen: list[str] = []
    agent: Agent = Agent(TestModel())  # TestModel calls every registered tool, then returns text

    @agent.tool
    async def note(ctx: RunContext[Any]) -> str:
        await ctx.sandbox.write_text('note.txt', 'from the model')
        seen.append(await ctx.sandbox.read_text('note.txt'))
        return 'ok'

    await agent.run('go', sandbox=backend_for(sandbox))

    assert seen == ['from the model']
    assert sandbox.files.files == {f'{WORKING_DIR}/note.txt': b'from the model'}


async def test_the_facade_exposes_the_backend_for_provider_specific_work():
    sandbox = FakeAsyncSandbox()
    backend = backend_for(sandbox)
    assert Sandbox(backend).backend is backend
    assert backend.sandbox is cast(AsyncSandbox, sandbox)
