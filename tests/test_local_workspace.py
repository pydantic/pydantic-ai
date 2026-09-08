"""Tests for the shipped minimal `LocalWorkspace` implementation of the workspace protocol."""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import signal
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import anyio
import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai._utils import abandon_threads_on_cancel
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.workspaces import (
    LocalWorkspace,
    SupportsFilesystem,
    Workspace,
    WorkspaceBackend,
    WorkspaceError,
    WorkspaceTimeoutError,
)

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(os.name != 'posix', reason='LocalWorkspace tests drive POSIX shell commands'),
]


async def test_local_workspace_concurrent_first_use_creates_one_root(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[str] = []
    real_mkdtemp: Callable[..., str] = tempfile.mkdtemp

    def counted_mkdtemp(*args: Any, **kwargs: Any) -> str:
        root = cast(str, real_mkdtemp(*args, **kwargs))
        created.append(root)
        return root

    monkeypatch.setattr('pydantic_ai.workspaces.local.tempfile.mkdtemp', counted_mkdtemp)
    workspace = LocalWorkspace()
    assert workspace.ref is None
    async with workspace:
        paths = await asyncio.gather(workspace.working_dir(), workspace.working_dir())
        assert paths[0] == paths[1]
        assert len(created) == 1
        root = Path(paths[0])
        assert root.exists()
    assert not root.exists()


async def test_cancelled_context_exit_removes_owned_root() -> None:
    workspace = LocalWorkspace()
    root: Path | None = None
    try:
        async with anyio.create_task_group() as tg:
            async with workspace:
                root = Path(await workspace.root)
                assert root.exists()
                tg.cancel_scope.cancel()
            assert root is not None
            assert not root.exists()
    finally:
        if root is not None:
            shutil.rmtree(root, ignore_errors=True)


async def test_cancelled_root_acquisition_keeps_ownership_with_abandoned_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_mkdtemp: Callable[..., str] = tempfile.mkdtemp
    started = threading.Event()
    release = threading.Event()
    created: list[Path] = []

    def held_mkdtemp(*args: Any, **kwargs: Any) -> str:
        root = Path(cast(str, real_mkdtemp(*args, **kwargs)))
        created.append(root)
        started.set()
        release.wait()
        return str(root)

    monkeypatch.setattr('pydantic_ai.workspaces.local.tempfile.mkdtemp', held_mkdtemp)
    workspace = LocalWorkspace()
    acquisition_scope: anyio.CancelScope | None = None
    acquisition_finished = anyio.Event()

    async def acquire_root() -> None:
        nonlocal acquisition_scope
        with anyio.CancelScope() as scope:
            acquisition_scope = scope
            try:
                with abandon_threads_on_cancel():
                    await workspace.root
            finally:
                acquisition_finished.set()

    roots: set[Path] = set()
    try:
        async with anyio.create_task_group() as tg:
            try:
                tg.start_soon(acquire_root)
                while not started.is_set():
                    await anyio.sleep(0)
                assert acquisition_scope is not None
                acquisition_scope.cancel()
                await anyio.sleep(0)
                release.set()
                await acquisition_finished.wait()
                root = Path(await workspace.root)
                roots.update(created)
                roots.add(root)

                assert len(created) == 1
                assert created[0].resolve() == root
                assert root.exists()
                async with workspace:
                    assert Path(await workspace.root) == root
                assert not root.exists()
            finally:
                release.set()
    finally:
        release.set()
        for root in roots | set(created):
            shutil.rmtree(root, ignore_errors=True)


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
    # `lax no cover`, not `no cover`: whether the reap lands inside this window is a race, so
    # this is taken on some runs and not others.
    except (FileNotFoundError, ProcessLookupError):  # pragma: lax no cover
        return False
    return state != 'Z'


async def _assert_process_gone(pid: int) -> None:
    for _ in range(200):
        if not _process_running(pid):
            return
        await asyncio.sleep(0.01)
    with suppress(ProcessLookupError):  # pragma: no cover - defensive cleanup before failing
        os.kill(pid, signal.SIGKILL)
    pytest.fail(f'process {pid} survived workspace cleanup')  # pragma: no cover


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
        LocalWorkspace()


async def test_local_workspace_conforms_to_the_protocol(tmp_path: Path):
    workspace = LocalWorkspace(tmp_path)
    assert isinstance(workspace, WorkspaceBackend)
    assert isinstance(workspace, SupportsFilesystem)
    typed: WorkspaceBackend = workspace  # static conformance, checked because tests are type-checked
    assert typed.ref is None
    await typed.working_dir()
    assert typed.ref is None


@pytest.mark.parametrize('operation', ['root', 'cwd', 'fs'])
async def test_relative_paths_are_rejected(tmp_path: Path, operation: str):
    """A relative path would resolve against the host process's working directory, outside the
    workspace root, so every entry point rejects it instead of silently escaping."""
    with pytest.raises(ValueError, match='absolute'):
        if operation == 'root':
            LocalWorkspace('work')
        elif operation == 'cwd':
            await LocalWorkspace(tmp_path).run(['pwd'], cwd='subdir')
        else:
            await LocalWorkspace(tmp_path).write_bytes('outside.txt', b'escape')


async def test_run_argv_and_shell(tmp_path: Path):
    workspace = LocalWorkspace(tmp_path)
    result = await workspace.run(['echo', 'hello'])
    assert (result.exit_code, result.stdout, result.stderr) == (0, 'hello\n', '')
    shell_result = await workspace.run('echo foo | tr a-z A-Z', shell=True)
    assert shell_result.stdout == 'FOO\n'


async def test_shell_discipline(tmp_path: Path):
    workspace = LocalWorkspace(tmp_path)
    with pytest.raises(TypeError, match='requires shell=True'):
        await workspace.run('echo hello')
    with pytest.raises(TypeError, match='single command string'):
        await workspace.run(['echo', 'hello'], shell=True)


async def test_missing_binary_raises(tmp_path: Path):
    """A spawn failure propagates as-is: the argv path execs directly, without a shell."""
    workspace = LocalWorkspace(tmp_path)
    with pytest.raises(FileNotFoundError):
        await workspace.run([str(tmp_path / 'missing-binary')])


async def test_nonzero_exit_is_a_result(tmp_path: Path):
    workspace = LocalWorkspace(tmp_path)
    result = await workspace.run('echo oops >&2; exit 3', shell=True)
    assert result.exit_code == 3
    assert result.stderr == 'oops\n'


async def test_timeout_kills_the_whole_process_group_and_raises(tmp_path: Path):
    workspace = LocalWorkspace(tmp_path)
    pid_file = tmp_path / 'pid'
    timeout = 0.2
    with pytest.raises(WorkspaceTimeoutError, match='was killed') as exc_info:
        # `exec` makes the shell's own PID the sleeping direct child, so the timeout applies to
        # a command that has not completed rather than to a descendant holding a pipe open.
        await workspace.run(f'echo $$ > {shlex.quote(str(pid_file))}; exec sleep 30', shell=True, timeout=timeout)

    error = exc_info.value
    assert isinstance(error, TimeoutError)
    assert error.timeout == timeout

    await _assert_process_gone(int(pid_file.read_text()))


async def test_output_over_safety_cap_kills_the_process_group(tmp_path: Path):
    workspace = LocalWorkspace(tmp_path)
    pid_file = tmp_path / 'pid'
    with pytest.raises(WorkspaceError, match=r'10 MiB.*redirect.*file.*read_file'):
        await workspace.run(
            f"echo $$ > {shlex.quote(str(pid_file))}; exec sh -c 'yes x & yes y >&2 & wait'",
            shell=True,
        )

    await _assert_process_gone(int(pid_file.read_text()))


async def test_background_child_holding_a_pipe_returns_after_the_drain_grace(tmp_path: Path):
    workspace = LocalWorkspace(tmp_path)
    pid_file = tmp_path / 'pid'
    child_pid_file = tmp_path / 'child-pid'
    command = (
        f'echo $$ > {shlex.quote(str(pid_file))}; sleep 30 & echo $! > {shlex.quote(str(child_pid_file))}; echo started'
    )
    started = time.monotonic()
    result = await workspace.run(command, shell=True, timeout=10)

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
    workspace = LocalWorkspace(tmp_path)
    with pytest.raises(WorkspaceTimeoutError) as exc_info:
        await workspace.run('echo stdout; echo stderr >&2; sleep 30', shell=True, timeout=0.2)

    error = exc_info.value
    assert error.stdout == 'stdout\n'
    assert error.stderr == 'stderr\n'


async def test_stdin_is_devnull(tmp_path: Path):
    workspace = LocalWorkspace(tmp_path)
    result = await workspace.run(
        [sys.executable, '-c', 'import sys; print("eof" if sys.stdin.read() == "" else "data")']
    )

    assert result.stdout == 'eof\n'


async def test_cancellation_kills_the_whole_process_group(tmp_path: Path):
    """The kill guarantee is not timeout-only: cancelling the awaiting task (an outer
    `asyncio.wait_for`, a durable runner aborting, a user breaking out of `iter()`) must
    also tear down the process group instead of leaking it."""
    workspace = LocalWorkspace(tmp_path)
    pid_file = tmp_path / 'pid file'
    task = asyncio.create_task(workspace.run(_background_sleep_command(pid_file), shell=True))
    await _wait_for_pid_file(pid_file)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await _assert_process_gone(int(pid_file.read_text()))


async def test_cancellation_during_spawn_still_kills_the_process_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The child is forked before the spawn coroutine finishes, so a cancellation delivered
    mid-spawn must still tear down the group — asyncio's own transport cleanup kills only the
    direct child, and the shell here has already exited."""
    workspace = LocalWorkspace(tmp_path)
    pid_file = tmp_path / 'pid'
    release = asyncio.Event()
    real_create_subprocess_shell = asyncio.create_subprocess_shell

    async def held_spawn(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        process = await real_create_subprocess_shell(*args, **kwargs)
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, 'create_subprocess_shell', held_spawn)
    task = asyncio.create_task(workspace.run(_background_sleep_command(pid_file), shell=True))
    await _wait_for_pid_file(pid_file)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    release.set()
    await _assert_process_gone(int(pid_file.read_text()))


async def test_timeout_during_spawn_still_kills_the_process_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workspace = LocalWorkspace(tmp_path)
    pid_file = tmp_path / 'pid'
    release = asyncio.Event()
    real_create_subprocess_shell = asyncio.create_subprocess_shell

    async def held_spawn(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        process = await real_create_subprocess_shell(*args, **kwargs)
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, 'create_subprocess_shell', held_spawn)
    try:
        with pytest.raises(WorkspaceTimeoutError, match='was killed'):
            # Long enough that the spawn always begins: `held_spawn` then holds it open past the
            # deadline, so the timeout always lands mid-spawn without racing the interpreter's
            # first subprocess start.
            await workspace.run(_background_sleep_command(pid_file), shell=True, timeout=0.5)
    finally:
        release.set()

    await _assert_process_gone(int(pid_file.read_text()))


async def test_cancellation_during_failing_spawn_is_tolerated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A spawn that fails after its run was cancelled has nobody left to receive the error;
    the abandoned-spawn cleanup must consume it instead of leaving it unretrieved."""
    workspace = LocalWorkspace(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def failing_spawn(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        started.set()
        await release.wait()
        raise OSError('spawn failed after abandonment')

    monkeypatch.setattr(asyncio, 'create_subprocess_shell', failing_spawn)
    task = asyncio.create_task(workspace.run('true', shell=True))
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
    LocalWorkspace._kill(process)  # pyright: ignore[reportPrivateUsage]


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
    LocalWorkspace._kill_abandoned_spawn(spawn)  # pyright: ignore[reportPrivateUsage]
    await process.wait()
    assert process.returncode == -signal.SIGKILL


async def test_owned_root_context_manager_reuse_creates_a_fresh_root():
    """Exiting removes an owned root; re-entering must lazily create a fresh one instead of
    resurrecting the deleted path."""
    workspace = LocalWorkspace()
    async with workspace:
        first = Path(await workspace.working_dir())
        assert first.exists()
    assert not first.exists()
    assert workspace.ref is None
    async with workspace:
        second = Path(await workspace.working_dir())
        assert second.exists()
        assert second != first
        assert workspace.ref is None
    assert not second.exists()


async def test_workspace_follows_backend_across_root_recreation():
    """A `Workspace` wrapper held across exit and re-entry must follow the backend to its fresh
    root instead of resurrecting the deleted one (which would also leak it on disk)."""
    backend = LocalWorkspace()
    workspace = Workspace(backend)
    async with backend:
        first = Path(await workspace.working_dir())
    async with backend:
        await workspace.write_text('probe.txt', 'hi')
        second = Path(await workspace.working_dir())
        assert second != first
        assert not first.exists()
        assert (await workspace.run(['cat', 'probe.txt'])).stdout == 'hi'
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
    monkeypatch.setenv('LOCAL_WORKSPACE_HOST_SECRET', 'do-not-pass')
    monkeypatch.setenv('LOCAL_WORKSPACE_EXPLICIT', 'host-value')
    workspace = LocalWorkspace(tmp_path)
    result = await workspace.run(
        ['/usr/bin/env'],
        env={'LOCAL_WORKSPACE_EXPLICIT': 'explicit-value'},
    )

    child_environment = dict(line.split('=', 1) for line in result.stdout.splitlines())
    assert child_environment == {**allowed, 'LOCAL_WORKSPACE_EXPLICIT': 'explicit-value'}


async def test_cwd_selects_the_working_directory(tmp_path: Path):
    workspace = LocalWorkspace(tmp_path)
    result = await workspace.run(['pwd'], cwd=str(tmp_path))
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

    workspace = Workspace(LocalWorkspace(repo / 'link' / '..'))
    working_dir = await workspace.working_dir()
    assert working_dir == str(tmp_path)  # where `chdir` actually lands, canonically spelled

    result = await workspace.run(['sh', '-c', 'echo hello > from_run.txt'])
    assert result.exit_code == 0
    assert await workspace.read_text('from_run.txt') == 'hello\n'


async def test_timeout_with_denied_group_kill_still_raises_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The timeout contract promises a `WorkspaceTimeoutError` even when a hardened host denies the
    group kill: the denial rides along as the cause, and the direct child is still killed."""
    workspace = LocalWorkspace(tmp_path)
    pid_file = tmp_path / 'pid'

    def deny_killpg(pgid: int, sig: int) -> None:
        raise PermissionError('signal denied')

    monkeypatch.setattr(os, 'killpg', deny_killpg)
    with pytest.raises(WorkspaceTimeoutError, match='denied') as exc_info:
        # `exec` makes the shell's own PID the sleeping direct child.
        await workspace.run(f'echo $$ > {shlex.quote(str(pid_file))}; exec sleep 30', shell=True, timeout=0.1)
    assert isinstance(exc_info.value.__cause__, PermissionError)
    await _assert_process_gone(int(pid_file.read_text()))


async def test_read_file_on_a_directory_raises(tmp_path: Path):
    (tmp_path / 'adir').mkdir()
    workspace = Workspace(LocalWorkspace(tmp_path))
    with pytest.raises(IsADirectoryError):
        await workspace.read_file('adir', limit=5)


async def test_default_temp_root_is_reported_canonically():
    """`working_dir()` must be filesystem-canonical even for the lazily created temp root.

    On macOS, `mkdtemp` hands back a path under the symlinked `/var`; reporting that spelling
    makes every string comparison against kernel-resolved paths (e.g. a command's `pwd -P`)
    silently false. Only the backend can canonicalize its own world, so it must do so before
    reporting.
    """
    async with LocalWorkspace() as workspace:
        working_dir = await workspace.working_dir()
        assert working_dir == os.path.realpath(working_dir)


async def test_filesystem_round_trip_with_parent_creation(tmp_path: Path):
    backend = LocalWorkspace(tmp_path)
    workspace = Workspace(backend)
    nested = await workspace.resolve('a/b/notes.txt')
    await workspace.write_text('a/b/notes.txt', 'hello')  # the write contract creates parents
    assert await workspace.read_text('a/b/notes.txt') == 'hello'
    entry = await backend.stat(nested)
    assert (entry.name, entry.is_dir, entry.size) == ('notes.txt', False, 5)

    payload = bytes(range(256))
    blob = await workspace.resolve('blob.bin')
    await backend.write_bytes(blob, payload)
    assert await backend.read_bytes(blob) == payload

    directory = await workspace.resolve('a')
    assert (await backend.stat(directory)).is_dir
    names = [entry.name for entry in await backend.list_dir(str(tmp_path))]
    assert names == ['a', 'blob.bin']

    made = await workspace.resolve('made/deep')
    await backend.make_dir(made)
    await backend.make_dir(made)  # mkdir -p semantics
    assert await backend.exists(made)

    await backend.remove(directory)  # removes the tree
    assert not await backend.exists(nested)
    await backend.remove(blob)
    assert not await backend.exists(blob)
    with pytest.raises(FileNotFoundError):
        await backend.read_bytes(blob)


@pytest.mark.parametrize('operation', ['read_bytes', 'stat', 'list_dir', 'remove'])
async def test_filesystem_reports_missing_paths(tmp_path: Path, operation: str):
    fs = LocalWorkspace(tmp_path)
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
async def test_windowed_read_runs_sed_inside_the_workspace(
    tmp_path: Path, content: str, offset: int, limit: int, expected: tuple[tuple[str, ...], bool, int | None]
):
    """The real `sed` slice: totals are known only when the window provably reached EOF."""
    workspace = Workspace(LocalWorkspace(tmp_path))
    await workspace.write_text('notes.txt', content)

    window = await workspace.read_file('notes.txt', offset=offset, limit=limit)

    assert (window.lines, window.has_more, window.total_lines) == expected
    assert window.start_line == offset


async def test_list_dir_symlink_sizes_match_stat(tmp_path: Path):
    """A symlinked file reports its target's size (as `stat` does); a broken symlink
    doesn't fail the listing, it just has no size."""
    workspace = LocalWorkspace(tmp_path)
    (tmp_path / 'target.txt').write_text('12345')
    (tmp_path / 'link.txt').symlink_to(tmp_path / 'target.txt')
    (tmp_path / 'broken.txt').symlink_to(tmp_path / 'missing.txt')

    entries = {entry.name: entry for entry in await workspace.list_dir(str(tmp_path))}
    assert entries['link.txt'].size == 5
    assert entries['link.txt'].size == (await workspace.stat(str(tmp_path / 'link.txt'))).size
    assert entries['broken.txt'].size is None


def fail_mkdtemp(*args: Any, **kwargs: Any) -> str:
    # Trap: tests using this pass exactly when it is never called.
    raise AssertionError('unused default workspace created a temporary directory')  # pragma: no cover


async def test_unused_default_workspace_creates_no_directory(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr('pydantic_ai.workspaces.local.tempfile.mkdtemp', fail_mkdtemp)
    async with LocalWorkspace():
        pass  # never used: the lazy default root must never be created


async def test_temp_root_already_deleted_on_exit_does_not_raise():
    async with LocalWorkspace() as workspace:
        root = Path(await workspace.working_dir())
        await workspace.remove(str(root))  # a command or tool may delete the root itself
    assert not root.exists()


async def test_failed_owned_root_cleanup_retains_root_for_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = LocalWorkspace()
    real_rmtree = shutil.rmtree
    calls = 0
    roots: set[Path] = set()

    def fail_once(path: str | Path) -> None:
        nonlocal calls
        root = Path(path)
        roots.add(root)
        calls += 1
        if calls == 1:
            raise PermissionError('cleanup denied')
        real_rmtree(root)

    monkeypatch.setattr('pydantic_ai.workspaces.local.shutil.rmtree', fail_once)
    root: Path | None = None
    try:
        with pytest.raises(PermissionError, match='cleanup denied'):
            async with workspace:
                root = Path(await workspace.root)
                roots.add(root)

        assert root is not None
        assert root.exists()
        assert Path(await workspace.root) == root

        async with workspace:
            assert Path(await workspace.root) == root

        assert not root.exists()
    finally:
        for path in roots:
            real_rmtree(path, ignore_errors=True)


async def test_caller_supplied_root_is_never_removed(tmp_path: Path):
    async with LocalWorkspace(tmp_path) as workspace:
        await Workspace(workspace).write_text('keep.txt', 'kept')
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
        result = await ctx.workspace.run(command, shell=True, timeout=30)
        outputs.append(result.stdout)
        return result.stdout

    async with LocalWorkspace(tmp_path) as workspace:
        result = await agent.run('compute 6*7 in the workspace', workspace=workspace)

    assert result.output == 'done'
    assert outputs == ['42\n']
