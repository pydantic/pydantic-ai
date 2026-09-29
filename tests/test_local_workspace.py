"""Tests for the shipped minimal `LocalWorkspaceBackend` implementation of the workspace protocol."""

from __future__ import annotations

import asyncio
import math
import os
import shlex
import signal
import threading
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
    pytest.mark.skipif(os.name != 'posix', reason='LocalWorkspaceBackend tests drive POSIX shell commands'),
]

READINESS_WAIT_TIMEOUT = 10


async def _process_running(pid: int) -> bool:
    result = await anyio.run_process(['ps', '-o', 'stat=', '-p', str(pid)], check=False)
    state = result.stdout.strip()
    return bool(state) and not state.startswith(b'Z')


async def _assert_process_gone(pid: int) -> None:
    with anyio.move_on_after(30):
        while await _process_running(pid):
            await anyio.sleep(0.01)  # pragma: lax no cover - depends on how fast the process dies
        return
    with suppress(ProcessLookupError):  # pragma: no cover - defensive cleanup before failing
        os.kill(pid, signal.SIGKILL)
    pytest.fail(f'process {pid} survived workspace cleanup')  # pragma: no cover


def _background_sleep_command(pid_file: Path) -> str:
    return f'sleep 3600 & echo $! > {shlex.quote(str(pid_file))}'


async def _wait_for_pid_file(pid_file: Path) -> None:
    with anyio.fail_after(30):
        while not (pid_file.exists() and pid_file.read_text(encoding='ascii').strip()):
            await anyio.sleep(0.01)


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


async def test_relative_filesystem_paths_are_rejected(tmp_path: Path):
    """A relative path would resolve against the host process's working directory rather than the
    workspace's, so the backend rejects it instead of silently depending on ambient state."""
    with pytest.raises(ValueError, match='absolute'):
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


async def test_reading_a_directory_closes_its_descriptor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    opened: list[int] = []
    real_open = os.open

    def recording_open(*args: Any, **kwargs: Any) -> int:
        opened.append(real_open(*args, **kwargs))
        return opened[-1]

    monkeypatch.setattr(os, 'open', recording_open)
    with pytest.raises(IsADirectoryError):
        await LocalWorkspaceBackend(tmp_path).read_bytes(str(tmp_path))
    [fd] = opened
    with pytest.raises(OSError):
        os.fstat(fd)


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
    # An unenterable working directory is not the program failing to run, so it still raises.
    locked = tmp_path / 'locked'
    locked.mkdir(mode=0o600)
    try:
        with pytest.raises(PermissionError):
            await LocalWorkspaceBackend(locked).run(['true'])
    finally:
        locked.chmod(0o700)


async def test_timeout_kills_the_whole_process_group_and_raises(tmp_path: Path):
    workspace = LocalWorkspaceBackend(tmp_path)
    pid_file = tmp_path / 'pid'
    with pytest.raises(WorkspaceTimeoutError, match='after 2s and was killed') as exc_info:
        # `exec` makes the shell's own PID the sleeping direct child, so the timeout applies to
        # a command that has not completed rather than to a descendant holding a pipe open.
        await workspace.run(f'echo $$ > {shlex.quote(str(pid_file))}; exec sleep 3600', shell=True, timeout=2)

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
    command = f'echo $$ > {shlex.quote(str(pid_file))}; sleep 3600 & echo $! > {shlex.quote(str(child_pid_file))}; echo started'
    result = await workspace.run(command, shell=True, timeout=10)

    assert (result.exit_code, result.stdout) == (0, 'started\n')
    await _assert_process_gone(int(pid_file.read_text()))
    child_pid = int(child_pid_file.read_text())
    try:
        assert await _process_running(child_pid)
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
        # Simulate a spawn that acknowledges a created process after its deadline.
        with anyio.CancelScope(shield=True):
            await release.wait()
        return process

    monkeypatch.setattr(anyio, 'open_process', held_spawn)
    timeout = 0.05
    task = asyncio.create_task(
        workspace.run(f'echo $$ > {shlex.quote(str(pid_file))}; exec sleep 3600', shell=True, timeout=timeout)
    )
    try:
        await _wait_for_pid_file(pid_file)
        await anyio.sleep(timeout)
        release.set()
        with pytest.raises(WorkspaceTimeoutError, match='during startup'):
            await task
    finally:
        release.set()
        # Cancelling a finished task is a no-op, and `asyncio.wait` never re-raises its outcome.
        task.cancel()
        await asyncio.wait([task])

    await _assert_process_gone(int(pid_file.read_text()))


@pytest.mark.parametrize('mode', ['deadline', 'cancel'])
async def test_stalled_spawn_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str):
    entered = anyio.Event()
    cancelled = anyio.Event()

    async def stalled_spawn(*args: Any, **kwargs: Any) -> anyio.abc.Process:
        entered.set()
        try:
            await anyio.sleep_forever()
        finally:
            cancelled.set()
        raise AssertionError('unreachable')  # pragma: no cover

    monkeypatch.setattr(anyio, 'open_process', stalled_spawn)
    workspace = LocalWorkspaceBackend(tmp_path)
    async with anyio.create_task_group() as tg:

        async def run() -> None:
            if mode == 'deadline':
                with pytest.raises(WorkspaceTimeoutError):
                    await workspace.run(['true'], timeout=0.05)
            else:
                with pytest.raises(anyio.get_cancelled_exc_class()):
                    await workspace.run(['true'])

        tg.start_soon(run)
        await entered.wait()
        if mode == 'cancel':
            tg.cancel_scope.cancel()
        with anyio.fail_after(30, shield=True):
            await cancelled.wait()


async def test_stalled_reap_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workspace = LocalWorkspaceBackend(tmp_path)
    real_close = workspace._close  # pyright: ignore[reportPrivateUsage]
    entered = anyio.Event()

    async def timed_out_after_spawn(
        _process: anyio.abc.Process,
        _stdout_buffer: bytearray,
        _stderr_buffer: bytearray,
        _absolute_deadline: float | None,
    ) -> int:
        raise TimeoutError

    async def stalled_close(process: anyio.abc.Process) -> None:
        entered.set()
        try:
            await anyio.sleep_forever()
        finally:
            # The fake stalls only the first close; release real OS resources after cancellation.
            with anyio.move_on_after(30, shield=True):
                await real_close(process)

    monkeypatch.setattr(workspace, '_close', stalled_close)
    monkeypatch.setattr(workspace, '_wait_and_collect_output', timed_out_after_spawn)
    with anyio.fail_after(30):
        with pytest.raises(WorkspaceTimeoutError):
            await workspace.run(['true'], timeout=30)
    assert entered.is_set()


async def test_stalled_reap_without_a_timeout_is_not_a_command_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workspace = LocalWorkspaceBackend(tmp_path)
    real_close = workspace._close  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(local_module, '_REAP_GRACE', 0.05)
    calls = 0

    async def stalled_close(process: anyio.abc.Process) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:  # stall only the first close; the cleanup close reaps for real
            await anyio.sleep_forever()
        await real_close(process)

    monkeypatch.setattr(workspace, '_close', stalled_close)
    with anyio.fail_after(30):
        with pytest.raises(TimeoutError, match='grace period') as exc_info:
            await workspace.run(['true'])
    assert not isinstance(exc_info.value, WorkspaceTimeoutError)


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


async def test_kill_tolerates_an_already_exited_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_killpg = os.killpg

    def already_exited(pgid: int, sig: int) -> None:
        real_killpg(pgid, sig)
        raise ProcessLookupError

    monkeypatch.setattr(os, 'killpg', already_exited)
    with pytest.raises(WorkspaceTimeoutError):
        await LocalWorkspaceBackend(tmp_path).run(['sh', '-c', 'sleep 3600'], timeout=0.05)


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
        await workspace.run(f'echo $$ > {shlex.quote(str(pid_file))}; exec sleep 3600', shell=True, timeout=5)
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


async def test_deleted_local_environment_rejects_every_operation(tmp_path: Path) -> None:
    root = tmp_path / 'root'
    root.mkdir()
    workspace = Workspace(LocalWorkspaceBackend(root))
    await workspace.working_dir()
    root.rmdir()
    for operation in (
        lambda: workspace.working_dir(),
        lambda: workspace.read_bytes('file'),
        lambda: workspace.write_bytes('file', b'x'),
        lambda: workspace.stat('file'),
        lambda: workspace.list_dir('.'),
        lambda: workspace.make_dir('sub'),
        lambda: workspace.remove('file'),
        lambda: workspace.exists('file'),
        lambda: workspace.realpath('file'),
        lambda: workspace.run(['pwd']),
    ):
        with pytest.raises(WorkspaceUnavailableError):
            await operation()
    assert not root.exists()


async def test_first_local_remove_refuses_canonical_root(tmp_path: Path) -> None:
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'safe').write_bytes(b'safe')
    link = tmp_path / 'alias'
    link.symlink_to(root, target_is_directory=True)
    workspace = Workspace(LocalWorkspaceBackend(link))
    with pytest.raises(ValueError, match='workspace root'):
        await workspace.remove(str(root))
    assert (root / 'safe').read_bytes() == b'safe'


async def test_missing_alias_can_attach_after_directory_is_created(tmp_path: Path) -> None:
    alias = tmp_path / 'alias'
    backend = LocalWorkspaceBackend(alias)
    with pytest.raises(WorkspaceUnavailableError):
        await backend.working_dir()
    target = tmp_path / 'target'
    target.mkdir()
    alias.symlink_to(target, target_is_directory=True)
    assert await backend.working_dir() == str(target.resolve())


@pytest.mark.parametrize('second', ['write', 'read'])
async def test_parallel_access_to_one_file_waits_for_an_in_progress_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, second: str
):
    """Parallel tool calls on one file must not tear it: a write truncates in place from a worker thread."""
    (tmp_path / 'real.txt').write_bytes(b'old')
    (tmp_path / 'alias.txt').symlink_to(tmp_path / 'real.txt')
    backend = LocalWorkspaceBackend(tmp_path)
    entered: list[bytes] = []
    first_entered = threading.Event()
    release = threading.Event()
    real_write_bytes = Path.write_bytes

    def blocking_write_bytes(self: Path, data: bytes) -> int:
        entered.append(data)
        if len(entered) == 1:
            first_entered.set()
            release.wait(READINESS_WAIT_TIMEOUT)
        return real_write_bytes(self, data)

    monkeypatch.setattr(Path, 'write_bytes', blocking_write_bytes)
    read: list[bytes] = []

    async def second_access() -> None:
        # Through a symlink: the lock is keyed by the real path, not the spelling.
        if second == 'write':
            await backend.write_bytes(str(tmp_path / 'alias.txt'), b'second')
        else:
            read.append(await backend.read_bytes(str(tmp_path / 'alias.txt')))

    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        async with anyio.create_task_group() as tg:
            tg.start_soon(backend.write_bytes, str(tmp_path / 'real.txt'), b'first')
            await anyio.to_thread.run_sync(first_entered.wait)
            tg.start_soon(second_access)
            [lock] = backend._file_locks.values()  # pyright: ignore[reportPrivateUsage]
            while lock.users < 2:
                await anyio.sleep(0.001)
            # The second call is queued on the lock and has not touched the file.
            assert entered == [b'first']
            assert read == []
            release.set()

    if second == 'write':
        assert entered == [b'first', b'second']
        assert (tmp_path / 'real.txt').read_bytes() == b'second'
    else:
        assert read == [b'first']
    assert backend._file_locks == {}  # pyright: ignore[reportPrivateUsage]
