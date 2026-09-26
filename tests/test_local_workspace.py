"""Tests for the shipped minimal `LocalWorkspaceBackend` implementation of the workspace protocol."""

from __future__ import annotations

import asyncio
import math
import os
import shlex
import signal
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

import anyio
import anyio.abc
import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.workspaces import (
    CommandResult,
    LocalWorkspaceBackend,
    Workspace,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
    local as local_module,
)

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(os.name != 'posix', reason='LocalWorkspaceBackend tests drive POSIX shell commands'),
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


def test_non_posix_platforms_are_rejected_at_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(os, 'name', 'nt')
    with pytest.raises(NotImplementedError, match='only supports POSIX'):
        LocalWorkspaceBackend(tmp_path)


def test_ref_normalizes_dot_segments_without_resolving_symlinks(tmp_path: Path) -> None:
    (tmp_path / 'actual').mkdir()
    (tmp_path / 'link').symlink_to(tmp_path / 'actual')
    spelled = tmp_path / 'link' / '.' / 'folder' / '..'
    ref = LocalWorkspaceBackend(spelled).ref
    assert ref == WorkspaceRef(provider='local', id=str(tmp_path / 'link'))
    assert ref != LocalWorkspaceBackend(tmp_path / 'actual').ref


async def test_ref_names_the_configured_working_dir_without_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The ref is the `~`-expanded spelling the backend was given, available before any operation.

    Symlinks are not resolved for it, unlike `working_dir()`, because reading a ref does no I/O.
    """
    monkeypatch.setenv('HOME', str(tmp_path))
    (tmp_path / 'target').mkdir()
    (tmp_path / 'link').symlink_to(tmp_path / 'target')

    workspace = LocalWorkspaceBackend('~/link/')

    assert workspace.ref == WorkspaceRef(provider='local', id=f'{tmp_path}/link')
    assert await workspace.working_dir() == str((tmp_path / 'target').resolve())
    assert workspace.ref == WorkspaceRef(provider='local', id=f'{tmp_path}/link')


async def test_missing_working_dir_is_unavailable_until_it_exists(tmp_path: Path):
    """The directory is the environment: the ref names it up front, and the first operation checks it is there."""
    workspace = LocalWorkspaceBackend(tmp_path / 'missing')

    assert workspace.ref == WorkspaceRef(provider='local', id=str(tmp_path / 'missing'))
    with pytest.raises(WorkspaceUnavailableError, match='does not exist'):
        await workspace.working_dir()
    with pytest.raises(WorkspaceUnavailableError, match='does not exist'):
        await workspace.run(['pwd'])
    (tmp_path / 'missing').write_text('a file, not a directory')
    with pytest.raises(WorkspaceUnavailableError, match='does not exist'):
        await workspace.working_dir()

    # Nothing was cached by the failed checks, so a directory created afterwards is picked up.
    (tmp_path / 'missing').unlink()
    (tmp_path / 'missing').mkdir()
    assert await workspace.working_dir() == str((tmp_path / 'missing').resolve())


@pytest.mark.parametrize('operation', ['cwd', 'filesystem_path'])
async def test_relative_cwd_and_filesystem_paths_are_rejected(tmp_path: Path, operation: str):
    """A relative path would resolve against the host process's working directory rather than the
    workspace's, so the backend rejects it instead of silently depending on ambient state."""
    with pytest.raises(ValueError, match='absolute'):
        if operation == 'cwd':
            await LocalWorkspaceBackend(tmp_path).run(['pwd'], cwd='subdir')
        else:
            await LocalWorkspaceBackend(tmp_path).write_bytes('relative.txt', b'data')


async def test_relative_working_dir_resolves_against_the_directory_at_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    workspace = LocalWorkspaceBackend('.')
    monkeypatch.chdir('/')

    assert workspace.ref == WorkspaceRef(provider='local', id=str(tmp_path))


@pytest.mark.parametrize(
    ('working_dir', 'expected'), [('~', ''), ('~/project', 'project'), (Path('~/project'), 'project')]
)
async def test_working_dir_expands_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, working_dir: str | Path, expected: str
):
    """A leading `~` names the user's home directory, which is absolute once expanded."""
    monkeypatch.setenv('HOME', str(tmp_path))
    (tmp_path / 'project').mkdir()

    workspace = LocalWorkspaceBackend(working_dir)

    assert await workspace.working_dir() == str((tmp_path / expected).resolve())
    result = await workspace.run(['pwd'])
    assert result.stdout.rstrip('\n') == await workspace.working_dir()


async def test_a_program_that_cannot_run_is_a_result_like_in_sh(tmp_path: Path):
    workspace = LocalWorkspaceBackend(tmp_path)
    missing = str(tmp_path / 'missing-binary')
    (tmp_path / 'script.sh').write_text('#!/bin/sh\n')
    not_executable = str(tmp_path / 'script.sh')

    assert await workspace.run([missing]) == CommandResult(
        exit_code=127, stdout='', stderr=f'{missing}: command not found\n'
    )
    assert await workspace.run([not_executable]) == CommandResult(
        exit_code=126, stdout='', stderr=f'{not_executable}: Permission denied\n'
    )
    # A missing `cwd` is not the program failing to run, so it still raises.
    with pytest.raises(FileNotFoundError):
        await workspace.run(['true'], cwd=str(tmp_path / 'missing-dir'))


async def test_timeout_kills_the_whole_process_group_and_raises(tmp_path: Path):
    workspace = LocalWorkspaceBackend(tmp_path)
    pid_file = tmp_path / 'pid'
    with pytest.raises(WorkspaceTimeoutError, match='was killed') as exc_info:
        # `exec` makes the shell's own PID the sleeping direct child, so the timeout applies to
        # a command that has not completed rather than to a descendant holding a pipe open.
        await workspace.run(f'echo $$ > {shlex.quote(str(pid_file))}; exec sleep 30', shell=True, timeout=2)

    assert isinstance(exc_info.value, TimeoutError)

    await _assert_process_gone(int(pid_file.read_text()))


async def test_output_limit_preserves_the_start_of_both_streams(tmp_path: Path):
    workspace = LocalWorkspaceBackend(tmp_path)
    with pytest.raises(WorkspaceOutputLimitError, match='10 MiB') as exc_info:
        await workspace.run("printf 'out-first\\n'; printf 'err-first\\n' >&2; yes x", shell=True)
    assert exc_info.value.limit == 10 * 1024 * 1024
    assert exc_info.value.stdout.startswith('out-first\n')
    assert exc_info.value.stderr.startswith('err-first\n')


async def test_output_over_safety_cap_kills_the_process_group(tmp_path: Path):
    workspace = LocalWorkspaceBackend(tmp_path)
    pid_file = tmp_path / 'pid'
    with pytest.raises(WorkspaceError, match=r'10 MiB.*redirect.*file'):
        await workspace.run(
            f"echo $$ > {shlex.quote(str(pid_file))}; exec sh -c 'yes x & yes y >&2 & wait'",
            shell=True,
        )

    await _assert_process_gone(int(pid_file.read_text()))


@pytest.mark.parametrize('force_pipe_bound_wait', [False, True], ids=['installed_anyio', 'anyio_before_4_15'])
async def test_background_child_holding_a_pipe_returns_after_the_drain_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force_pipe_bound_wait: bool
):
    """`run()` returns when the command exits even though a background child still holds stdout.

    Before anyio 4.15.0, `Process.wait()` on asyncio also waited for the pipes to close (anyio#1174),
    so `LocalWorkspaceBackend` has a fallback for those versions. Forcing its version gate on runs
    that fallback on whatever anyio is installed; the other case is the installed version's own path.
    """
    if force_pipe_bound_wait:
        monkeypatch.setattr(local_module, '_ANYIO_WAITS_FOR_PIPES', True)
    workspace = LocalWorkspaceBackend(tmp_path)
    pid_file = tmp_path / 'pid'
    child_pid_file = tmp_path / 'child-pid'
    command = (
        f'echo $$ > {shlex.quote(str(pid_file))}; sleep 30 & echo $! > {shlex.quote(str(child_pid_file))}; echo started'
    )
    result = await workspace.run(command, shell=True, timeout=10)

    assert (result.exit_code, result.stdout) == (0, 'started\n')
    await _assert_process_gone(int(pid_file.read_text()))
    child_pid = int(child_pid_file.read_text())
    try:
        assert _process_running(child_pid)
    finally:
        with suppress(ProcessLookupError):
            os.kill(child_pid, signal.SIGKILL)
    await _assert_process_gone(child_pid)


async def test_anyio_4_15_wait_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """With the version gate off, `run()` uses anyio's own `wait()` and `aclose()`, whatever anyio is installed.

    No background child here: on anyio before 4.15 that path waits for the pipes to close.
    """
    monkeypatch.setattr(local_module, '_ANYIO_WAITS_FOR_PIPES', False)
    result = await LocalWorkspaceBackend(tmp_path).run(['echo', 'done'], timeout=10)
    assert (result.exit_code, result.stdout) == (0, 'done\n')


async def test_timeout_keeps_output_printed_before_the_deadline(tmp_path: Path):
    workspace = LocalWorkspaceBackend(tmp_path)
    with pytest.raises(WorkspaceTimeoutError) as exc_info:
        await workspace.run('echo stdout; echo stderr >&2; sleep 30', shell=True, timeout=5)

    error = exc_info.value
    assert error.stdout == 'stdout\n'
    assert error.stderr == 'stderr\n'


async def test_stdin_is_devnull(tmp_path: Path):
    workspace = LocalWorkspaceBackend(tmp_path)
    result = await workspace.run(
        [
            sys.executable,
            '-c',
            'import sys; print("eof" if sys.stdin.read() == "" else "data")',
        ]
    )

    assert result.stdout == 'eof\n'


async def test_cancellation_kills_the_whole_process_group(tmp_path: Path):
    """The kill guarantee is not timeout-only: cancelling the awaiting task (an outer
    `asyncio.wait_for`, a durable runner aborting, a user breaking out of `iter()`) must
    also tear down the process group instead of leaking it."""
    workspace = LocalWorkspaceBackend(tmp_path)
    pid_file = tmp_path / 'pid file'
    task = asyncio.create_task(workspace.run(_background_sleep_command(pid_file), shell=True))
    await _wait_for_pid_file(pid_file)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await _assert_process_gone(int(pid_file.read_text()))


async def test_cancellation_during_spawn_still_kills_the_process_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workspace = LocalWorkspaceBackend(tmp_path)
    pid_file = tmp_path / 'pid'
    release = asyncio.Event()
    real_open_process = anyio.open_process

    async def held_spawn(*args: Any, **kwargs: Any) -> anyio.abc.Process:
        process = await real_open_process(*args, **kwargs)
        await release.wait()
        return process

    monkeypatch.setattr(anyio, 'open_process', held_spawn)
    task = asyncio.create_task(workspace.run(_background_sleep_command(pid_file), shell=True))
    await _wait_for_pid_file(pid_file)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    await _assert_process_gone(int(pid_file.read_text()))


async def test_timeout_during_spawn_still_kills_the_process_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workspace = LocalWorkspaceBackend(tmp_path)
    pid_file = tmp_path / 'pid'
    release = asyncio.Event()
    real_open_process = anyio.open_process

    async def held_spawn(*args: Any, **kwargs: Any) -> anyio.abc.Process:
        process = await real_open_process(*args, **kwargs)
        await release.wait()
        return process

    monkeypatch.setattr(anyio, 'open_process', held_spawn)
    timeout = 0.05
    task = asyncio.create_task(workspace.run(_background_sleep_command(pid_file), shell=True, timeout=timeout))
    try:
        await _wait_for_pid_file(pid_file)
        await asyncio.sleep(timeout * 2)
        assert not task.done()
        release.set()
        with pytest.raises(WorkspaceTimeoutError, match='was killed'):
            await task
    finally:
        release.set()
        # Cancelling a finished task is a no-op, and `asyncio.wait` never re-raises its outcome.
        task.cancel()
        await asyncio.wait([task])

    await _assert_process_gone(int(pid_file.read_text()))


async def test_failing_spawn_after_cancellation_raises_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workspace = LocalWorkspaceBackend(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def failing_spawn(*args: Any, **kwargs: Any) -> anyio.abc.Process:
        started.set()
        await release.wait()
        raise OSError('spawn failed')

    monkeypatch.setattr(anyio, 'open_process', failing_spawn)
    task = asyncio.create_task(workspace.run('true', shell=True))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(OSError, match='spawn failed'):
        await task


async def test_kill_tolerates_an_already_exited_group():
    """A command can finish in the instant between the deadline firing and the kill; the
    only benign `killpg` failure is "already exited". Unreachable deterministically through
    `run()` (it's a race), so the teardown helper is pinned directly."""
    async with await anyio.open_process(['true'], start_new_session=True) as process:
        await process.wait()
        LocalWorkspaceBackend._kill(process)  # pyright: ignore[reportPrivateUsage]


async def test_commands_inherit_only_path_home_and_locale_from_the_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv('LOCAL_WORKSPACE_HOST_SECRET', 'do-not-pass')
    monkeypatch.setenv('PATH', '/host/bin:/usr/bin')
    monkeypatch.setenv('HOME', '/host/home')
    for name in ('LANG', 'LC_ALL', 'LC_CTYPE'):
        monkeypatch.setenv(name, 'C.UTF-8')
    workspace = LocalWorkspaceBackend(tmp_path, env={'SHARED': 'backend', 'HOME': 'backend'})

    result = await workspace.run(['/usr/bin/env'], env={'SHARED': 'call'})

    child_environment = dict(line.split('=', 1) for line in result.stdout.splitlines())
    assert child_environment == {
        'PATH': '/host/bin:/usr/bin',
        'HOME': 'backend',
        'SHARED': 'call',
        'LANG': 'C.UTF-8',
        'LC_ALL': 'C.UTF-8',
        'LC_CTYPE': 'C.UTF-8',
    }


async def test_local_workspace_inherits_utf8_locale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ('LANG', 'LC_ALL', 'LC_CTYPE'):
        monkeypatch.setenv(name, 'en_US.UTF-8')
    workspace = LocalWorkspaceBackend(tmp_path)
    result = await workspace.run(['sh', '-c', 'printf "%s:%s:%s" "$LANG" "$LC_ALL" "$LC_CTYPE"'])
    assert result.stdout == 'en_US.UTF-8:en_US.UTF-8:en_US.UTF-8'


async def test_symlinked_working_dir_with_dotdot_keeps_one_environment(tmp_path: Path):
    """A working directory spelled through `symlink/..` must not split `run()` and `fs` into two directories.

    The kernel resolves the symlink *before* applying `..` (landing in the link target's
    parent), while lexical normalization deletes the `link` segment as text (landing in the
    spelling's parent) — two different directories. Canonicalizing the working directory is
    what keeps the protocol's one-environment contract: a file written by a command is visible
    to `fs` reads of the same relative path.
    """
    data = tmp_path / 'data'
    data.mkdir()
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'link').symlink_to(data)

    workspace = Workspace(LocalWorkspaceBackend(repo / 'link' / '..'))
    working_dir = await workspace.working_dir()
    assert working_dir == str(tmp_path)  # where `chdir` actually lands, canonically spelled

    result = await workspace.run(['sh', '-c', 'echo hello > from_run.txt'])
    assert result.exit_code == 0
    assert await workspace.read_text('from_run.txt') == 'hello\n'


async def test_timeout_with_denied_group_kill_still_raises_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The timeout contract promises a `WorkspaceTimeoutError` even when a hardened host denies the
    group kill: the denial rides along as the cause, and the direct child is still killed."""
    workspace = LocalWorkspaceBackend(tmp_path)
    pid_file = tmp_path / 'pid'

    def deny_killpg(pgid: int, sig: int) -> None:
        raise PermissionError('signal denied')

    monkeypatch.setattr(os, 'killpg', deny_killpg)
    with pytest.raises(WorkspaceTimeoutError, match='denied') as exc_info:
        # `exec` makes the shell's own PID the sleeping direct child.
        await workspace.run(f'echo $$ > {shlex.quote(str(pid_file))}; exec sleep 30', shell=True, timeout=5)
    assert isinstance(exc_info.value.__cause__, PermissionError)
    await _assert_process_gone(int(pid_file.read_text()))


@pytest.mark.parametrize('timeout', [-1, 0, math.nan, math.inf, '5'])
async def test_local_rejects_invalid_command_timeout(tmp_path: Path, timeout: Any):
    with pytest.raises(ValueError, match='timeout must be a positive finite number or None'):
        await LocalWorkspaceBackend(tmp_path).run(['true'], timeout=timeout)


async def test_signal_killed_command_reports_shell_exit_code(tmp_path: Path):
    result = await LocalWorkspaceBackend(tmp_path).run('kill -9 $$', shell=True)
    assert result.exit_code == 137


async def test_removed_workspace_cannot_be_recreated_or_removed(tmp_path: Path):
    root = tmp_path / 'workspace'
    root.mkdir()
    workspace = Workspace(LocalWorkspaceBackend(root))
    await workspace.working_dir()
    root.rmdir()
    with pytest.raises(WorkspaceUnavailableError):
        await workspace.run(['pwd'])
    with pytest.raises(WorkspaceUnavailableError):
        await workspace.write_bytes('sub/file', b'x')
    with pytest.raises(WorkspaceUnavailableError):
        await workspace.make_dir('sub')
    assert not root.exists()

    root.mkdir()
    (root / 'file').write_bytes(b'safe')
    with pytest.raises(ValueError, match='workspace root'):
        await workspace.remove('.')
    with pytest.raises(ValueError, match='workspace root'):
        await workspace.remove(str(tmp_path))
    assert (root / 'file').read_bytes() == b'safe'


async def test_reading_fifo_fails_without_waiting_for_writer(tmp_path: Path):
    fifo = tmp_path / 'fifo'
    os.mkfifo(fifo)
    workspace = Workspace(LocalWorkspaceBackend(tmp_path))
    with anyio.fail_after(2):
        with pytest.raises(OSError, match='not a regular file'):
            await workspace.read_bytes('fifo')


async def test_list_dir_keeps_self_loop_symlink(tmp_path: Path):
    (tmp_path / 'loop').symlink_to('loop')
    entries = await LocalWorkspaceBackend(tmp_path).list_dir(str(tmp_path))
    assert [(entry.name, entry.is_dir, entry.size) for entry in entries] == [('loop', False, None)]


async def test_list_dir_symlink_sizes_match_stat(tmp_path: Path):
    """A symlinked file reports its target's size (as `stat` does); a broken symlink
    doesn't fail the listing, it just has no size."""
    workspace = LocalWorkspaceBackend(tmp_path)
    (tmp_path / 'target.txt').write_text('12345')
    (tmp_path / 'link.txt').symlink_to(tmp_path / 'target.txt')
    (tmp_path / 'broken.txt').symlink_to(tmp_path / 'missing.txt')

    entries = {entry.name: entry for entry in await workspace.list_dir(str(tmp_path))}
    assert entries['link.txt'].size == 5
    assert entries['link.txt'].size == (await workspace.stat(str(tmp_path / 'link.txt'))).size
    assert entries['broken.txt'].size is None


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

    workspace = LocalWorkspaceBackend(tmp_path)
    result = await agent.run('compute 6*7 in the workspace', workspace=workspace)

    assert result.output == 'done'
    assert outputs == ['42\n']
