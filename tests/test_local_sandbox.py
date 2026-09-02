"""Tests for the shipped minimal `LocalSandbox` implementation of the sandbox protocol."""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.sandboxes import (
    LocalSandbox,
    Sandbox,
    SandboxBackend,
    SandboxError,
    SandboxTimeoutError,
    SupportsFilesystem,
)

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(os.name != 'posix', reason='LocalSandbox tests drive POSIX shell commands'),
]


_HAS_PROCFS = Path('/proc/self').exists()


def _process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # No procfs (macOS): signalable is the best signal we have.
    if not _HAS_PROCFS:  # pragma: no cover
        return True
    # A killed orphan re-parents to PID 1 and stays a signalable zombie until reaped,
    # which a loaded CI host can delay past this polling window — but a zombie is dead:
    # it can never run again, which is what the kill guarantee promises.
    try:
        state = Path(f'/proc/{pid}/stat').read_text(encoding='ascii').rsplit(')', 1)[1].split()[0]
    # Reaped between the signal check and the procfs read; ESRCH surfaces as
    # `ProcessLookupError` from the read itself.
    except (FileNotFoundError, ProcessLookupError):  # pragma: no cover
        return False
    return state != 'Z'


async def _assert_process_gone(pid: int) -> None:
    for _ in range(200):
        if not _process_running(pid):
            return
        await asyncio.sleep(0.01)
    with suppress(ProcessLookupError):  # pragma: no cover - defensive cleanup before failing
        os.kill(pid, signal.SIGKILL)
    pytest.fail(f'process {pid} survived sandbox cleanup')  # pragma: no cover


def _background_sleep_command(pid_file: Path) -> str:
    return f'sleep 30 & echo $! > {shlex.quote(str(pid_file))}'


async def _wait_for_pid_file(pid_file: Path) -> None:
    for _ in range(200):
        if pid_file.exists() and pid_file.read_text(encoding='ascii').strip():
            return
        await asyncio.sleep(0.01)
    pytest.fail(f'background process did not write its PID to {pid_file}')  # pragma: no cover


def test_non_posix_platforms_are_rejected_at_construction(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(os, 'name', 'nt')
    with pytest.raises(NotImplementedError, match='only supports POSIX'):
        LocalSandbox()


async def test_local_sandbox_conforms_to_the_protocol(tmp_path: Path):
    sandbox = LocalSandbox(tmp_path)
    assert isinstance(sandbox, SandboxBackend)
    assert isinstance(sandbox, SupportsFilesystem)
    typed: SandboxBackend = sandbox  # static conformance, checked because tests are type-checked
    assert typed.sandbox_id.startswith('local-')


@pytest.mark.parametrize('operation', ['root', 'cwd', 'fs'])
async def test_relative_paths_are_rejected(tmp_path: Path, operation: str):
    """A relative path would resolve against the host process's working directory, outside the
    sandbox root, so every entry point rejects it instead of silently escaping."""
    with pytest.raises(ValueError, match='absolute'):
        if operation == 'root':
            LocalSandbox('work')
        elif operation == 'cwd':
            await LocalSandbox(tmp_path).run(['pwd'], cwd='subdir')
        else:
            await LocalSandbox(tmp_path).fs.write_bytes('outside.txt', b'escape')


async def test_run_argv_and_shell(tmp_path: Path):
    sandbox = LocalSandbox(tmp_path)
    result = await sandbox.run(['echo', 'hello'])
    assert (result.exit_code, result.stdout, result.stderr) == (0, 'hello\n', '')
    shell_result = await sandbox.run('echo foo | tr a-z A-Z', shell=True)
    assert shell_result.stdout == 'FOO\n'


async def test_shell_discipline(tmp_path: Path):
    sandbox = LocalSandbox(tmp_path)
    with pytest.raises(TypeError, match='requires shell=True'):
        await sandbox.run('echo hello')
    with pytest.raises(TypeError, match='single command string'):
        await sandbox.run(['echo', 'hello'], shell=True)


async def test_missing_binary_raises(tmp_path: Path):
    """A spawn failure propagates as-is: the argv path execs directly, without a shell."""
    sandbox = LocalSandbox(tmp_path)
    with pytest.raises(FileNotFoundError):
        await sandbox.run([str(tmp_path / 'missing-binary')])


async def test_nonzero_exit_is_a_result(tmp_path: Path):
    sandbox = LocalSandbox(tmp_path)
    result = await sandbox.run('echo oops >&2; exit 3', shell=True)
    assert result.exit_code == 3
    assert result.stderr == 'oops\n'


async def test_timeout_kills_the_whole_process_group_and_raises(tmp_path: Path):
    sandbox = LocalSandbox(tmp_path)
    pid_file = tmp_path / 'pid'
    timeout = 0.2
    with pytest.raises(SandboxTimeoutError, match='was killed') as exc_info:
        # `exec` makes the shell's own PID the sleeping direct child, so the timeout applies to
        # a command that has not completed rather than to a descendant holding a pipe open.
        await sandbox.run(f'echo $$ > {shlex.quote(str(pid_file))}; exec sleep 30', shell=True, timeout=timeout)

    error = exc_info.value
    assert isinstance(error, TimeoutError)
    assert error.timeout == timeout

    await _assert_process_gone(int(pid_file.read_text()))


async def test_output_over_safety_cap_kills_the_process_group(tmp_path: Path):
    sandbox = LocalSandbox(tmp_path)
    pid_file = tmp_path / 'pid'
    with pytest.raises(SandboxError, match=r'10 MiB.*redirect.*file.*read_file'):
        await sandbox.run(
            f"echo $$ > {shlex.quote(str(pid_file))}; exec sh -c 'yes x & yes y >&2 & wait'",
            shell=True,
        )

    await _assert_process_gone(int(pid_file.read_text()))


async def test_background_child_holding_a_pipe_returns_after_the_drain_grace(tmp_path: Path):
    sandbox = LocalSandbox(tmp_path)
    pid_file = tmp_path / 'pid'
    child_pid_file = tmp_path / 'child-pid'
    command = (
        f'echo $$ > {shlex.quote(str(pid_file))}; sleep 30 & echo $! > {shlex.quote(str(child_pid_file))}; echo started'
    )
    started = time.monotonic()
    result = await sandbox.run(command, shell=True, timeout=10)

    assert time.monotonic() - started < 5
    assert (result.exit_code, result.stdout) == (0, 'started\n')
    await _assert_process_gone(int(pid_file.read_text()))
    child_pid = int(child_pid_file.read_text())
    try:
        assert _process_running(child_pid)
    finally:
        with suppress(ProcessLookupError):
            os.kill(child_pid, signal.SIGKILL)
    await _assert_process_gone(child_pid)


async def test_timeout_keeps_output_printed_before_the_deadline(tmp_path: Path):
    sandbox = LocalSandbox(tmp_path)
    with pytest.raises(SandboxTimeoutError) as exc_info:
        await sandbox.run('echo stdout; echo stderr >&2; sleep 30', shell=True, timeout=0.2)

    error = exc_info.value
    assert error.stdout == 'stdout\n'
    assert error.stderr == 'stderr\n'


async def test_stdin_is_devnull(tmp_path: Path):
    sandbox = LocalSandbox(tmp_path)
    result = await sandbox.run([sys.executable, '-c', 'import sys; print("eof" if sys.stdin.read() == "" else "data")'])

    assert result.stdout == 'eof\n'


async def test_cancellation_kills_the_whole_process_group(tmp_path: Path):
    """The kill guarantee is not timeout-only: cancelling the awaiting task (an outer
    `asyncio.wait_for`, a durable runner aborting, a user breaking out of `iter()`) must
    also tear down the process group instead of leaking it."""
    sandbox = LocalSandbox(tmp_path)
    pid_file = tmp_path / 'pid file'
    task = asyncio.create_task(sandbox.run(_background_sleep_command(pid_file), shell=True))
    await _wait_for_pid_file(pid_file)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await _assert_process_gone(int(pid_file.read_text()))


async def test_cancellation_during_spawn_still_kills_the_process_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The child is forked before the spawn coroutine finishes, so a cancellation delivered
    mid-spawn must still tear down the group — asyncio's own transport cleanup kills only the
    direct child, and the shell here has already exited."""
    sandbox = LocalSandbox(tmp_path)
    pid_file = tmp_path / 'pid'
    release = asyncio.Event()
    real_create_subprocess_shell = asyncio.create_subprocess_shell

    async def held_spawn(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        process = await real_create_subprocess_shell(*args, **kwargs)
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, 'create_subprocess_shell', held_spawn)
    task = asyncio.create_task(sandbox.run(_background_sleep_command(pid_file), shell=True))
    await _wait_for_pid_file(pid_file)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    release.set()
    await _assert_process_gone(int(pid_file.read_text()))


async def test_timeout_during_spawn_still_kills_the_process_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sandbox = LocalSandbox(tmp_path)
    pid_file = tmp_path / 'pid'
    release = asyncio.Event()
    real_create_subprocess_shell = asyncio.create_subprocess_shell

    async def held_spawn(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        process = await real_create_subprocess_shell(*args, **kwargs)
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, 'create_subprocess_shell', held_spawn)
    try:
        with pytest.raises(SandboxTimeoutError, match='was killed'):
            await sandbox.run(_background_sleep_command(pid_file), shell=True, timeout=0.01)
    finally:
        release.set()

    await _assert_process_gone(int(pid_file.read_text()))


async def test_cancellation_during_failing_spawn_is_tolerated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A spawn that fails after its run was cancelled has nobody left to receive the error;
    the abandoned-spawn cleanup must consume it instead of leaving it unretrieved."""
    sandbox = LocalSandbox(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def failing_spawn(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        started.set()
        await release.wait()
        raise OSError('spawn failed after abandonment')

    monkeypatch.setattr(asyncio, 'create_subprocess_shell', failing_spawn)
    task = asyncio.create_task(sandbox.run('true', shell=True))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    release.set()
    # Let the abandoned spawn finish and its done-callback consume the failure.
    await asyncio.sleep(0.01)


async def test_kill_tolerates_an_already_exited_group():
    """A command can finish in the instant between the deadline firing and the kill; the
    only benign `killpg` failure is "already exited". Unreachable deterministically through
    `run()` (it's a race), so the teardown helper is pinned directly."""
    process = await asyncio.create_subprocess_exec('true', start_new_session=True)
    await process.wait()
    LocalSandbox._kill(process)  # pyright: ignore[reportPrivateUsage]


async def test_abandoned_spawn_kill_falls_back_to_direct_child_on_denied_killpg(
    monkeypatch: pytest.MonkeyPatch,
):
    """Parity with `_kill`'s `PermissionError` fallback, minus the propagation: nobody is
    left to receive the error on the abandoned path, so the direct child still dies."""
    process = await asyncio.create_subprocess_exec('sleep', '30', start_new_session=True)

    async def completed_spawn() -> asyncio.subprocess.Process | Exception:
        return process

    spawn = asyncio.ensure_future(completed_spawn())
    await spawn

    def deny_killpg(pgid: int, sig: int) -> None:
        raise PermissionError('signal denied')

    monkeypatch.setattr(os, 'killpg', deny_killpg)
    LocalSandbox._kill_abandoned_spawn(spawn)  # pyright: ignore[reportPrivateUsage]
    await process.wait()
    assert process.returncode == -signal.SIGKILL


async def test_owned_root_context_manager_reuse_creates_a_fresh_root():
    """Exiting removes an owned root; re-entering must lazily create a fresh one instead of
    resurrecting the deleted path."""
    sandbox = LocalSandbox()
    async with sandbox:
        first = Path(await sandbox.working_dir())
        assert first.exists()
    assert not first.exists()
    async with sandbox:
        second = Path(await sandbox.working_dir())
        assert second.exists()
        assert second != first
    assert not second.exists()


async def test_facade_follows_backend_across_root_recreation():
    """A `Sandbox` facade held across exit and re-entry must follow the backend to its fresh
    root instead of resurrecting the deleted one (which would also leak it on disk)."""
    backend = LocalSandbox()
    facade = Sandbox(backend)
    async with backend:
        first = Path(await facade.working_dir())
    async with backend:
        await facade.write_text('probe.txt', 'hi')
        second = Path(await facade.working_dir())
        assert second != first
        assert not first.exists()
        assert (await facade.run(['cat', 'probe.txt'])).stdout == 'hi'
    assert not second.exists()


async def test_local_environment_contains_only_allowed_variables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    allowed = {
        'PATH': '/bin:/usr/bin',
        'HOME': str(tmp_path / 'home'),
        'LANG': 'C.UTF-8',
        'TMPDIR': str(tmp_path / 'tmp'),
    }
    for key, value in allowed.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv('LOCAL_SANDBOX_HOST_SECRET', 'do-not-pass')
    monkeypatch.setenv('LOCAL_SANDBOX_EXPLICIT', 'host-value')
    sandbox = LocalSandbox(tmp_path)
    result = await sandbox.run(
        ['/usr/bin/env'],
        env={'LOCAL_SANDBOX_EXPLICIT': 'explicit-value'},
    )

    child_environment = dict(line.split('=', 1) for line in result.stdout.splitlines())
    assert child_environment == {**allowed, 'LOCAL_SANDBOX_EXPLICIT': 'explicit-value'}


async def test_cwd_selects_the_working_directory(tmp_path: Path):
    sandbox = LocalSandbox(tmp_path)
    result = await sandbox.run(['pwd'], cwd=str(tmp_path))
    assert result.stdout.rstrip('\n').endswith(tmp_path.name)


async def test_symlinked_root_with_dotdot_keeps_one_environment(tmp_path: Path):
    """A root spelled through `symlink/..` must not split `run()` and `fs` into two directories.

    The kernel resolves the symlink *before* applying `..` (landing in the link target's
    parent), while lexical normalization deletes the `link` segment as text (landing in the
    spelling's parent) — two different directories. Canonicalizing the root at construction is
    what keeps the protocol's one-environment contract: a file written by a command is visible
    to `fs` reads of the same relative path.
    """
    data = tmp_path / 'data'
    data.mkdir()
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'link').symlink_to(data)

    sandbox = Sandbox(LocalSandbox(repo / 'link' / '..'))
    working_dir = await sandbox.working_dir()
    assert working_dir == str(tmp_path)  # where `chdir` actually lands, canonically spelled

    result = await sandbox.run(['sh', '-c', 'echo hello > from_run.txt'])
    assert result.exit_code == 0
    assert await sandbox.read_text('from_run.txt') == 'hello\n'


async def test_timeout_with_denied_group_kill_still_raises_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The timeout contract promises a `SandboxTimeoutError` even when a hardened host denies the
    group kill: the denial rides along as the cause, and the direct child is still killed."""
    sandbox = LocalSandbox(tmp_path)
    pid_file = tmp_path / 'pid'

    def deny_killpg(pgid: int, sig: int) -> None:
        raise PermissionError('signal denied')

    monkeypatch.setattr(os, 'killpg', deny_killpg)
    with pytest.raises(SandboxTimeoutError, match='denied') as exc_info:
        # `exec` makes the shell's own PID the sleeping direct child.
        await sandbox.run(f'echo $$ > {shlex.quote(str(pid_file))}; exec sleep 30', shell=True, timeout=0.1)
    assert isinstance(exc_info.value.__cause__, PermissionError)
    await _assert_process_gone(int(pid_file.read_text()))


async def test_read_file_on_a_directory_raises(tmp_path: Path):
    (tmp_path / 'adir').mkdir()
    sandbox = Sandbox(LocalSandbox(tmp_path))
    with pytest.raises(IsADirectoryError):
        await sandbox.read_file('adir', limit=5)


async def test_default_temp_root_is_reported_canonically():
    """`working_dir()` must be filesystem-canonical even for the lazily created temp root.

    On macOS, `mkdtemp` hands back a path under the symlinked `/var`; reporting that spelling
    makes every string comparison against kernel-resolved paths (e.g. a command's `pwd -P`)
    silently false. Only the backend can canonicalize its own world, so it must do so before
    reporting.
    """
    async with LocalSandbox() as sandbox:
        working_dir = await sandbox.working_dir()
        assert working_dir == os.path.realpath(working_dir)


async def test_filesystem_round_trip_with_parent_creation(tmp_path: Path):
    backend = LocalSandbox(tmp_path)
    sandbox = Sandbox(backend)
    nested = await sandbox.resolve('a/b/notes.txt')
    await sandbox.write_text('a/b/notes.txt', 'hello')  # the write contract creates parents
    assert await sandbox.read_text('a/b/notes.txt') == 'hello'
    entry = await backend.fs.stat(nested)
    assert (entry.name, entry.is_dir, entry.size) == ('notes.txt', False, 5)

    payload = bytes(range(256))
    blob = await sandbox.resolve('blob.bin')
    await backend.fs.write_bytes(blob, payload)
    assert await backend.fs.read_bytes(blob) == payload

    directory = await sandbox.resolve('a')
    assert (await backend.fs.stat(directory)).is_dir
    names = [entry.name for entry in await backend.fs.list_dir(str(tmp_path))]
    assert names == ['a', 'blob.bin']

    made = await sandbox.resolve('made/deep')
    await backend.fs.make_dir(made)
    await backend.fs.make_dir(made)  # mkdir -p semantics
    assert await backend.fs.exists(made)

    await backend.fs.remove(directory)  # removes the tree
    assert not await backend.fs.exists(nested)
    await backend.fs.remove(blob)
    assert not await backend.fs.exists(blob)
    with pytest.raises(FileNotFoundError):
        await backend.fs.read_bytes(blob)


@pytest.mark.parametrize('operation', ['read_bytes', 'stat', 'list_dir', 'remove'])
async def test_filesystem_reports_missing_paths(tmp_path: Path, operation: str):
    fs = LocalSandbox(tmp_path).fs
    with pytest.raises(FileNotFoundError):
        await getattr(fs, operation)(str(tmp_path / 'missing'))


@pytest.mark.parametrize(
    ('content', 'offset', 'limit', 'expected'),
    [
        ('one\ntwo\nthree\nfour\n', 2, 2, (('two', 'three'), True, None)),
        ('one\ntwo\nthree\n', 2, 5, (('two', 'three'), False, 3)),
        ('one\n', 10, 2, ((), False, None)),
        ('one', 1, 2, (('one',), False, 1)),
    ],
    ids=['inside', 'reaches-eof', 'past-eof', 'no-trailing-newline'],
)
async def test_windowed_read_runs_sed_inside_the_sandbox(
    tmp_path: Path, content: str, offset: int, limit: int, expected: tuple[tuple[str, ...], bool, int | None]
):
    """The real `sed` slice: totals are known only when the window provably reached EOF."""
    sandbox = Sandbox(LocalSandbox(tmp_path))
    await sandbox.write_text('notes.txt', content)

    window = await sandbox.read_file('notes.txt', offset=offset, limit=limit)

    assert (window.lines, window.has_more, window.total_lines) == expected
    assert window.start_line == offset


async def test_list_dir_symlink_sizes_match_stat(tmp_path: Path):
    """A symlinked file reports its target's size (as `stat` does); a broken symlink
    doesn't fail the listing, it just has no size."""
    sandbox = LocalSandbox(tmp_path)
    (tmp_path / 'target.txt').write_text('12345')
    (tmp_path / 'link.txt').symlink_to(tmp_path / 'target.txt')
    (tmp_path / 'broken.txt').symlink_to(tmp_path / 'missing.txt')

    entries = {entry.name: entry for entry in await sandbox.fs.list_dir(str(tmp_path))}
    assert entries['link.txt'].size == 5
    assert entries['link.txt'].size == (await sandbox.fs.stat(str(tmp_path / 'link.txt'))).size
    assert entries['broken.txt'].size is None


def fail_mkdtemp(*args: Any, **kwargs: Any) -> str:
    # Trap: tests using this pass exactly when it is never called.
    raise AssertionError('unused default sandbox created a temporary directory')  # pragma: no cover


async def test_unused_default_sandbox_creates_no_directory(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr('pydantic_ai.sandboxes.local.tempfile.mkdtemp', fail_mkdtemp)
    async with LocalSandbox():
        pass  # never used: the lazy default root must never be created


async def test_temp_root_already_deleted_on_exit_does_not_raise():
    async with LocalSandbox() as sandbox:
        root = Path(await sandbox.working_dir())
        await sandbox.fs.remove(str(root))  # a command or tool may delete the root itself
    assert not root.exists()


async def test_caller_supplied_root_is_never_removed(tmp_path: Path):
    async with LocalSandbox(tmp_path) as sandbox:
        await Sandbox(sandbox).write_text('keep.txt', 'kept')
    assert (tmp_path / 'keep.txt').read_text() == 'kept'


async def test_agent_run_end_to_end(tmp_path: Path):
    def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('execute', {'command': 'echo $((6*7))'})])
        return ModelResponse(parts=[TextPart('done')])

    agent: Agent = Agent(FunctionModel(model_func))
    outputs: list[str] = []

    @agent.tool
    async def execute(ctx: RunContext[Any], command: str) -> str:
        result = await ctx.sandbox.run(command, shell=True, timeout=30)
        outputs.append(result.stdout)
        return result.stdout

    async with LocalSandbox(tmp_path) as sandbox:
        result = await agent.run('compute 6*7 in the sandbox', sandbox=sandbox)

    assert result.output == 'done'
    assert outputs == ['42\n']
