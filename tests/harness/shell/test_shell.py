"""Tests for the Shell capability and ShellToolset."""

from __future__ import annotations

import errno
import json
import os
import posixpath
import shlex
import shutil
import signal
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any, NoReturn
from unittest.mock import patch

import anyio
import anyio.to_thread
import pytest
import sniffio

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace
from pydantic_ai.exceptions import ModelRetry, ToolFailed, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import (
    CommandResult,
    FileEntry,
    LocalWorkspaceBackend,
    ReadOnlyWorkspace,
    Workspace,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness._workspace import READ_ONLY_FAILURE
from pydantic_ai_harness.code_mode import CodeMode
from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS, Shell
from pydantic_ai_harness.shell._jobs import Job
from pydantic_ai_harness.shell._policy import is_interactive_command
from pydantic_ai_harness.shell._toolset import ShellToolset

from .._tool_calls import call_tool


def _env_toolset(
    shell_dir: Path,
    *,
    env: Mapping[str, str] | None = None,
    denied_env_patterns: Sequence[str] = (),
) -> ShellToolset[None]:
    """Build a ShellToolset wired for env-control tests, with safe defaults."""
    return ShellToolset(
        allowed_commands=[],
        denied_commands=[],
        denied_operators=[],
        default_timeout=10.0,
        max_output_chars=50_000,
        persist_cwd=False,
        allow_interactive=False,
        env=env,
        denied_env_patterns=denied_env_patterns,
    )


def _shell_toolset(
    shell_dir: Path,
    *,
    max_output_chars: int = 50_000,
    default_timeout: float = 10.0,
) -> ShellToolset[None]:
    return ShellToolset(
        allowed_commands=[],
        denied_commands=[],
        denied_operators=[],
        default_timeout=default_timeout,
        max_output_chars=max_output_chars,
        persist_cwd=False,
        allow_interactive=False,
    )


async def test_command_tools_declare_temporal_budget(tmp_path: Path) -> None:
    toolset = _shell_toolset(tmp_path)
    persistent = ShellToolset(
        allowed_commands=[],
        denied_commands=[],
        denied_operators=[],
        default_timeout=10,
        max_output_chars=50_000,
        persist_cwd=False,
        allow_interactive=False,
        tools=['shell'],
    )
    for name, owner in (('run_command', toolset), ('shell', persistent)):
        tools = await owner.get_tools(_ctx(tmp_path))
        metadata = tools[name].tool_def.metadata
        assert metadata is not None
        assert metadata['temporal'] == {'start_to_close_timeout': timedelta(seconds=300)}


def _raise_oserror(code: int, message: str) -> Callable[..., Awaitable[NoReturn]]:
    """Build a stand-in for `anyio.open_process` that fails with a given errno."""

    async def fail(*args: object, **kwargs: object) -> NoReturn:
        raise OSError(code, message)

    return fail


def _read_env_var(name: str) -> str:
    """Shell command that prints an env var's value, or ABSENT if unset."""
    return f'{sys.executable} -c "import os; print(os.environ.get({name!r}, \'ABSENT\'))"'


def _run_context(workspace: Workspace) -> RunContext[None]:
    """Minimal `RunContext` for invoking toolset methods directly in tests."""
    return RunContext[None](
        deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0, workspace=workspace
    )


def _ctx(working_dir: Path) -> RunContext[None]:
    """A run context whose workspace is this machine, with commands starting in `working_dir`."""
    return _run_context(Workspace(LocalWorkspaceBackend(working_dir)))


async def _call_shell_tool(toolset: ShellToolset[None], working_dir: Path, name: str, **tool_args: Any) -> str:
    ctx = _ctx(working_dir)
    tools = await toolset.get_tools(ctx)
    result = await toolset.call_tool(name, tool_args, ctx, tools[name])
    assert isinstance(result, str)
    return result


def _parse_command_id(result: str) -> str:
    assert 'ID: ' in result, f'Expected "ID: " in result: {result!r}'
    return result.split('ID: ')[1].strip()


async def _job(ts: ShellToolset[None], ctx: RunContext[None], command_id: str) -> Job:
    job = await ts._job(ctx, command_id)  # pyright: ignore[reportPrivateUsage]
    assert job is not None
    return job


class TestIsInteractiveCommand:
    def test_vi(self) -> None:
        assert is_interactive_command('vi file.txt') is True

    def test_vim(self) -> None:
        assert is_interactive_command('vim file.txt') is True

    def test_nano(self) -> None:
        assert is_interactive_command('nano file.txt') is True

    def test_less(self) -> None:
        assert is_interactive_command('less file.txt') is True

    def test_top(self) -> None:
        assert is_interactive_command('top') is True

    def test_sudo(self) -> None:
        assert is_interactive_command('sudo rm -rf /') is True

    def test_ssh(self) -> None:
        assert is_interactive_command('ssh host') is True

    def test_regular_command(self) -> None:
        assert is_interactive_command('ls -la') is False

    def test_echo(self) -> None:
        assert is_interactive_command('echo hello') is False

    def test_grep(self) -> None:
        assert is_interactive_command('grep pattern file') is False

    def test_emacs(self) -> None:
        assert is_interactive_command('emacs file.txt') is True

    def test_man(self) -> None:
        assert is_interactive_command('man ls') is True

    def test_htop(self) -> None:
        assert is_interactive_command('htop') is True

    def test_telnet(self) -> None:
        assert is_interactive_command('telnet localhost 80') is True

    def test_ftp(self) -> None:
        assert is_interactive_command('ftp host') is True

    def test_passwd(self) -> None:
        assert is_interactive_command('passwd') is True

    def test_more(self) -> None:
        assert is_interactive_command('more file.txt') is True

    def test_not_prefix_match(self) -> None:
        assert is_interactive_command('view file.txt') is False
        assert is_interactive_command('vishnu') is False

    def test_leading_spaces(self) -> None:
        assert is_interactive_command('  vi file.txt') is True
        assert is_interactive_command('  sudo rm') is True


@pytest.fixture
def shell_dir(tmp_path: Path) -> Path:
    (tmp_path / 'test.txt').write_text('hello\n')
    (tmp_path / 'subdir').mkdir()
    (tmp_path / 'subdir' / 'nested.txt').write_text('nested\n')
    return tmp_path


@pytest.fixture
def toolset(shell_dir: Path) -> ShellToolset[None]:
    return ShellToolset(
        allowed_commands=[],
        denied_commands=['rm', 'rmdir'],
        denied_operators=[],
        default_timeout=10.0,
        max_output_chars=50_000,
        persist_cwd=False,
        allow_interactive=False,
    )


@pytest.fixture
def persist_toolset(shell_dir: Path) -> ShellToolset[None]:
    return ShellToolset(
        allowed_commands=[],
        denied_commands=[],
        denied_operators=[],
        default_timeout=10.0,
        max_output_chars=50_000,
        persist_cwd=True,
        allow_interactive=False,
    )


class TestCommandValidation:
    async def test_denied_command_blocked(self, toolset: ShellToolset[None]) -> None:
        with pytest.raises(PermissionError, match="'rm' is denied"):
            toolset._check_command('rm -rf /')  # pyright: ignore[reportPrivateUsage]

    async def test_allowed_command_permitted(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=['echo', 'cat'],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        ts._check_command('echo hello')  # pyright: ignore[reportPrivateUsage]
        ts._check_command('cat file.txt')  # pyright: ignore[reportPrivateUsage]

    async def test_allowed_blocks_non_matching(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=['echo'],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        with pytest.raises(PermissionError, match='not in the allowed list'):
            ts._check_command('cat file.txt')  # pyright: ignore[reportPrivateUsage]

    async def test_both_allow_and_deny_raises(self, shell_dir: Path) -> None:
        with pytest.raises(ValueError, match='Specify allowed_commands or denied_commands'):
            ShellToolset(
                allowed_commands=['echo'],
                denied_commands=['rm'],
                denied_operators=[],
                default_timeout=10.0,
                max_output_chars=50_000,
                persist_cwd=False,
                allow_interactive=False,
            )

    async def test_interactive_blocked_by_default(self, toolset: ShellToolset[None]) -> None:
        with pytest.raises(PermissionError, match='Interactive commands'):
            toolset._check_command('vim file.txt')  # pyright: ignore[reportPrivateUsage]

    async def test_interactive_allowed_when_enabled(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=True,
        )
        ts._check_command('vim file.txt')  # pyright: ignore[reportPrivateUsage]

    async def test_denied_operator_blocked(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=['>', '>>'],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        with pytest.raises(PermissionError, match="'>' is not allowed"):
            ts._check_command('echo hello > file.txt')  # pyright: ignore[reportPrivateUsage]

    async def test_denied_operator_passes_when_not_present(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=['>', '>>'],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        ts._check_command('echo hello')  # pyright: ignore[reportPrivateUsage]

    async def test_unparseable_command_allowed(self, toolset: ShellToolset[None]) -> None:
        toolset._check_command("echo 'unterminated")  # pyright: ignore[reportPrivateUsage]

    async def test_empty_command_allowed(self, toolset: ShellToolset[None]) -> None:
        toolset._check_command('')  # pyright: ignore[reportPrivateUsage]

    async def test_denied_operator_substring_match(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=['>>'],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        with pytest.raises(PermissionError, match="'>>' is not allowed"):
            ts._check_command('echo hello >> file.txt')  # pyright: ignore[reportPrivateUsage]

    async def test_shlex_error_returns_early(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=['rm'],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        ts._check_command("echo 'unterminated")  # pyright: ignore[reportPrivateUsage]

    async def test_empty_tokens(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=['echo'],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        ts._check_command('')  # pyright: ignore[reportPrivateUsage]

    def test_first_denied_operator_match(self, toolset: ShellToolset[None]) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=['|', '>'],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        assert ts._first_denied_operator('echo hi | cat') == '|'  # pyright: ignore[reportPrivateUsage]

    def test_first_denied_operator_no_match(self, toolset: ShellToolset[None]) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=['|', '>'],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        assert ts._first_denied_operator('echo hello') is None  # pyright: ignore[reportPrivateUsage]

    def test_first_denied_operator_empty_list(self, toolset: ShellToolset[None]) -> None:
        assert toolset._first_denied_operator('echo hi | cat') is None  # pyright: ignore[reportPrivateUsage]


class TestCwdCapture:
    """The persistent-cwd mechanism records `pwd` out-of-band to a private file inside the
    workspace, so command output can never spoof the tracked directory."""

    async def test_capture_disabled_returns_command_unchanged(
        self, toolset: ShellToolset[None], tmp_path: Path
    ) -> None:
        wrapped, cwd_file = await toolset._build_cwd_capture(_ctx(tmp_path), 'echo hi')  # pyright: ignore[reportPrivateUsage]
        assert wrapped == 'echo hi'
        assert cwd_file is None

    async def test_capture_records_pwd_out_of_band(self, persist_toolset: ShellToolset[None], tmp_path: Path) -> None:
        wrapped, cwd_file = await persist_toolset._build_cwd_capture(_ctx(tmp_path), 'echo hi')  # pyright: ignore[reportPrivateUsage]
        assert cwd_file is not None
        # pwd is redirected to the private capture file, never echoed to stdout
        assert f'pwd > {shlex.quote(cwd_file)}' in wrapped
        assert wrapped.startswith('echo hi')

    async def test_apply_valid_dir_updates_cwd(
        self, persist_toolset: ShellToolset[None], shell_dir: Path, tmp_path: Path
    ) -> None:
        capture = tmp_path / 'cwd'
        capture.write_text(f'{shell_dir / "subdir"}\n')
        await persist_toolset._apply_captured_cwd(_ctx(shell_dir), str(capture))  # pyright: ignore[reportPrivateUsage]
        assert persist_toolset._cwd == str(shell_dir / 'subdir')  # pyright: ignore[reportPrivateUsage]

    async def test_apply_empty_file_keeps_cwd(self, persist_toolset: ShellToolset[None], tmp_path: Path) -> None:
        capture = tmp_path / 'cwd'
        capture.write_text('')
        await persist_toolset._apply_captured_cwd(_ctx(tmp_path), str(capture))  # pyright: ignore[reportPrivateUsage]
        assert persist_toolset._cwd is None  # pyright: ignore[reportPrivateUsage]

    async def test_apply_non_dir_keeps_cwd(self, persist_toolset: ShellToolset[None], tmp_path: Path) -> None:
        capture = tmp_path / 'cwd'
        capture.write_text(str(tmp_path / 'does_not_exist'))
        await persist_toolset._apply_captured_cwd(_ctx(tmp_path), str(capture))  # pyright: ignore[reportPrivateUsage]
        assert persist_toolset._cwd is None  # pyright: ignore[reportPrivateUsage]

    async def test_apply_file_keeps_cwd(
        self, persist_toolset: ShellToolset[None], shell_dir: Path, tmp_path: Path
    ) -> None:
        capture = tmp_path / 'cwd'
        capture.write_text(str(shell_dir / 'test.txt'))
        await persist_toolset._apply_captured_cwd(_ctx(shell_dir), str(capture))  # pyright: ignore[reportPrivateUsage]
        assert persist_toolset._cwd is None  # pyright: ignore[reportPrivateUsage]

    async def test_capture_not_utf8_keeps_cwd(self, persist_toolset: ShellToolset[None], shell_dir: Path) -> None:
        # The wrapper runs `pwd` in the same shell as the model's command, so a
        # shell function named `pwd` decides the bytes written to the capture
        # file. Decoding them raises `UnicodeDecodeError`, a `ValueError` and
        # not an `OSError`, so the guard has to cover both.
        result = await persist_toolset.run_command(_ctx(shell_dir), r"""pwd() { printf '\377\376'; }""")
        assert '[exit code' not in result
        assert persist_toolset._cwd is None  # pyright: ignore[reportPrivateUsage]

    async def test_capture_path_too_long_keeps_cwd(
        self, persist_toolset: ShellToolset[None], shell_dir: Path, tmp_path: Path
    ) -> None:
        # The recorded path is junk the workspace refuses to stat (ENAMETOOLONG); the
        # tracked cwd must survive.
        capture = tmp_path / 'cwd'
        capture.write_text(f'/{"x" * 300}')
        await persist_toolset._apply_captured_cwd(_ctx(shell_dir), str(capture))  # pyright: ignore[reportPrivateUsage]
        assert persist_toolset._cwd is None  # pyright: ignore[reportPrivateUsage]

    async def test_relative_capture_keeps_cwd(self, persist_toolset: ShellToolset[None], tmp_path: Path) -> None:
        # A `pwd` function can print anything; a relative path names no workspace directory.
        result = await persist_toolset.run_command(_ctx(tmp_path), 'pwd() { echo subdir; }')
        assert '[exit code' not in result
        assert persist_toolset._cwd is None  # pyright: ignore[reportPrivateUsage]

    async def test_command_that_exits_early_writes_no_capture(
        self, persist_toolset: ShellToolset[None], tmp_path: Path
    ) -> None:
        # `exit` skips the capture line; the tracked cwd and the cleanup both tolerate its absence.
        result = await persist_toolset.run_command(_ctx(tmp_path), 'cd subdir && exit 0')
        assert result == '(no output)'
        assert persist_toolset._cwd is None  # pyright: ignore[reportPrivateUsage]

    async def test_capture_file_is_removed(self, persist_toolset: ShellToolset[None], tmp_path: Path) -> None:
        # The capture lives in the workspace's job directory and is removed after each command,
        # including one that deletes it first.
        await persist_toolset.run_command(_ctx(tmp_path), 'true')
        base = Path(await persist_toolset._jobs_base(_ctx(tmp_path)))  # pyright: ignore[reportPrivateUsage]
        assert not list(base.glob('cwd-*'))
        await persist_toolset.run_command(_ctx(tmp_path), f'rm -f {shlex.quote(str(base))}/cwd-*')
        assert not list(base.glob('cwd-*'))


class TestForRunIsolation:
    """B3: `get_toolset` builds one shared instance at agent construction, so
    `for_run` must hand each run a fresh copy -- otherwise concurrent runs share
    `_cwd` and corrupt each other."""

    async def test_for_run_returns_fresh_instance(self, persist_toolset: ShellToolset[None], tmp_path: Path) -> None:
        run1 = await persist_toolset.for_run(_ctx(tmp_path))
        run2 = await persist_toolset.for_run(_ctx(tmp_path))
        assert run1 is not persist_toolset
        assert run2 is not run1

    async def test_without_persist_cwd_every_run_shares_the_toolset(self, shell_dir: Path) -> None:
        # No per-run state, so durable execution keeps the toolset it registered.
        shell = Shell[None](id='build_shell')
        toolset = shell.get_toolset()
        assert toolset is shell.get_toolset()
        assert toolset.id == 'build_shell'
        assert await toolset.for_run(_ctx(shell_dir)) is toolset

    async def test_removed_tracked_cwd_resets_before_next_command(
        self, persist_toolset: ShellToolset[None], shell_dir: Path
    ) -> None:
        await persist_toolset.run_command(_ctx(shell_dir), 'cd subdir')
        (shell_dir / 'subdir').rename(shell_dir / 'moved-subdir')
        with pytest.raises(ModelRetry, match='previous directory was removed'):
            await persist_toolset.run_command(_ctx(shell_dir), 'pwd')
        result = await persist_toolset.run_command(_ctx(shell_dir), 'pwd')
        assert str(shell_dir) in result

    async def test_persist_cwd_isolated_across_runs(self, persist_toolset: ShellToolset[None], shell_dir: Path) -> None:
        run1 = await persist_toolset.for_run(_ctx(shell_dir))
        assert isinstance(run1, ShellToolset)
        await run1.run_command(_ctx(shell_dir), 'cd subdir')
        assert run1._cwd == str(shell_dir / 'subdir')  # pyright: ignore[reportPrivateUsage]
        # A second run must start back at the configured root, not inherit run1's cd.
        run2 = await persist_toolset.for_run(_ctx(shell_dir))
        assert isinstance(run2, ShellToolset)
        assert run2._cwd is None  # pyright: ignore[reportPrivateUsage]


class TestDurableJob:
    async def test_retry_of_same_tool_call_reuses_background_job(self, shell_dir: Path) -> None:
        toolset = _shell_toolset(shell_dir)
        ctx = _ctx(shell_dir)
        ctx.run_id = 'durable-run'
        ctx.tool_call_id = 'launch-1'
        command = 'echo one >> side-effects.txt'
        first = await toolset.start_command(ctx, command)
        second = await toolset.start_command(ctx, command)
        assert first == second
        command_id = first.split('ID: ')[-1]
        with anyio.fail_after(5):
            while 'finished' not in await toolset.check_command(ctx, command_id):
                await anyio.sleep(0.05)  # pragma: lax no cover
        assert (shell_dir / 'side-effects.txt').read_text().splitlines() == ['one']


class TestDurableCwd:
    async def test_run_cwd_rehydrates_from_workspace_without_leaking(
        self, persist_toolset: ShellToolset[None], shell_dir: Path
    ) -> None:
        first = _ctx(shell_dir)
        first.run_id = 'first'
        await persist_toolset.run_command(first, 'cd subdir')

        # A different toolset instance stands in for an activity on a new worker.
        other = await persist_toolset.for_run(_ctx(shell_dir))
        assert isinstance(other, ShellToolset)
        assert str(shell_dir / 'subdir') in await other.run_command(first, 'pwd')
        second = _ctx(shell_dir)
        second.run_id = 'second'
        assert str(shell_dir / 'subdir') not in await other.run_command(second, 'pwd')

    async def test_missing_state_after_cd_reports_reset(
        self, persist_toolset: ShellToolset[None], shell_dir: Path
    ) -> None:
        ctx = _ctx(shell_dir)
        ctx.run_id = 'lost-state-run'
        await persist_toolset.run_command(ctx, 'cd subdir')
        state_dir = shell_dir / '.pydantic-ai-harness/shell/run-state'
        for state_file in state_dir.iterdir():
            state_file.unlink()
        with pytest.raises(ModelRetry, match='saved working directory was lost'):
            await persist_toolset.run_command(ctx, 'pwd')

    async def test_removed_run_cwd_retries_and_clears_workspace_state(
        self, persist_toolset: ShellToolset[None], shell_dir: Path
    ) -> None:
        ctx = _ctx(shell_dir)
        ctx.run_id = 'removed-directory-run'
        await persist_toolset.run_command(ctx, 'cd subdir')
        (shell_dir / 'subdir').rename(shell_dir / 'moved-subdir')

        # A new worker reads the saved cwd rather than the original toolset's memory.
        fresh = await persist_toolset.for_run(ctx)
        assert isinstance(fresh, ShellToolset)
        with pytest.raises(ModelRetry, match=f'The previous directory was removed; now in {shell_dir}'):
            await fresh.run_command(ctx, 'pwd')
        assert str(shell_dir) in await fresh.run_command(ctx, 'pwd')


class TestCancelledRunCwd:
    async def test_worker_interruption_retains_cwd_for_recovery(self, shell_dir: Path) -> None:
        shell = Shell(persist_cwd=True)
        ctx = _ctx(shell_dir)
        ctx.run_id = 'interrupted-run'
        toolset = shell.get_toolset()
        await toolset.run_command(ctx, 'cd subdir')

        async def interrupted() -> NoReturn:
            await anyio.sleep_forever()
            raise AssertionError('unreachable')  # pragma: no cover

        async def run() -> None:
            await shell.wrap_run(ctx, handler=interrupted)

        async with anyio.create_task_group() as tg:
            tg.start_soon(run)
            await anyio.sleep(0)
            tg.cancel_scope.cancel()
        fresh = await toolset.for_run(ctx)
        assert isinstance(fresh, ShellToolset)
        assert str(shell_dir / 'subdir') in await fresh.run_command(ctx, 'pwd')


class TestSameRunConcurrentCwd:
    async def test_last_completed_command_wins(self, persist_toolset: ShellToolset[None], shell_dir: Path) -> None:
        (shell_dir / 'other').mkdir()
        ctx = _ctx(shell_dir)
        ctx.run_id = 'parallel-run'
        # Both commands start at the root; the slower completion publishes its cwd last.
        results: list[str] = []
        b_finished = anyio.Event()

        async def run_a() -> None:
            results.append(
                await persist_toolset.run_command(
                    ctx, 'touch started-a; while [ ! -f release-a ]; do sleep 0.01; done; cd subdir'
                )
            )

        async def run_b() -> None:
            results.append(
                await persist_toolset.run_command(
                    ctx, 'touch started-b; while [ ! -f started-a ]; do sleep 0.01; done; cd other'
                )
            )
            b_finished.set()

        async with anyio.create_task_group() as tg:
            tg.start_soon(run_a)
            tg.start_soon(run_b)
            with anyio.fail_after(60):
                await b_finished.wait()
            (shell_dir / 'release-a').touch()
        assert len(results) == 2 and all('[exit code:' not in result for result in results)
        assert str(shell_dir / 'subdir') in await persist_toolset.run_command(ctx, 'pwd')


class TestPersistCwdHardening:
    """B4: regression tests for the old stdout-sentinel footguns -- a command's
    output spoofing the cwd, and `;` silently disabling tracking."""

    async def test_cd_persists_even_with_semicolon(self, persist_toolset: ShellToolset[None], tmp_path: Path) -> None:
        # The old mechanism skipped tracking whenever ';' appeared, silently
        # dropping a real `cd`. The out-of-band capture records it regardless.
        await persist_toolset.run_command(_ctx(tmp_path), 'cd subdir ; true')
        result = await persist_toolset.run_command(_ctx(tmp_path), 'pwd')
        assert 'subdir' in result

    async def test_output_cannot_spoof_cwd(self, persist_toolset: ShellToolset[None], shell_dir: Path) -> None:
        # The old mechanism parsed cwd from stdout, so a command printing the
        # sentinel string could redirect the tracked cwd with no real cd.
        spoof = f'true ; echo __HARNESS_PWD__{shell_dir / "subdir"}'
        await persist_toolset.run_command(_ctx(shell_dir), spoof)
        assert persist_toolset._cwd == str(shell_dir)  # pyright: ignore[reportPrivateUsage]


class TestSpawnFailures:
    """Failures raised by the spawn itself, which reached past `_recoverable`
    when it only caught `PermissionError` and aborted the whole run."""

    async def _toolset_in(self, shell_dir: Path) -> ShellToolset[None]:
        """A toolset whose tracked directory is `shell_dir / 'subdir'`, after the model's own `cd`."""
        toolset = ShellToolset[None](
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=True,
            allow_interactive=False,
        )
        await toolset.run_command(_ctx(shell_dir), 'cd subdir')
        return toolset

    async def test_cwd_deleted(self, shell_dir: Path) -> None:
        # The model's own earlier command can do this: `mv "$PWD" "$PWD-old"`
        # passes the denylist, which only inspects the first token.
        target = shell_dir / 'subdir'
        ts = await self._toolset_in(shell_dir)
        shutil.rmtree(target)
        with pytest.raises(ModelRetry, match='previous directory was removed'):
            await ts.run_command(_ctx(shell_dir), 'echo hello')

    async def test_cwd_replaced_by_file(self, shell_dir: Path) -> None:
        target = shell_dir / 'subdir'
        ts = await self._toolset_in(shell_dir)
        shutil.rmtree(target)
        target.write_text('not a directory\n')
        with pytest.raises(ModelRetry, match='previous directory was removed'):
            await ts.run_command(_ctx(shell_dir), 'echo hello')

    async def test_cwd_deleted_start_command(self, shell_dir: Path) -> None:
        target = shell_dir / 'subdir'
        ts = await self._toolset_in(shell_dir)
        shutil.rmtree(target)
        with pytest.raises(ModelRetry, match='previous directory was removed'):
            await ts.start_command(_ctx(shell_dir), 'sleep 30')

    async def test_message_omits_host_path(self, shell_dir: Path) -> None:
        target = shell_dir / 'subdir'
        ts = await self._toolset_in(shell_dir)
        shutil.rmtree(target)
        with pytest.raises(ModelRetry) as exc_info:
            await ts.run_command(_ctx(shell_dir), 'echo hello')
        assert str(target) not in str(exc_info.value)

    @pytest.mark.parametrize(
        ('command', 'expected'),
        [('echo hi\x00there', 'NUL byte'), ('echo \ud800', 'cannot be encoded for the operating system')],
    )
    async def test_unspawnable_command_string(
        self, toolset: ShellToolset[None], command: str, expected: str, tmp_path: Path
    ) -> None:
        with pytest.raises(ModelRetry, match=expected):
            await toolset.run_command(_ctx(tmp_path), command)

    async def test_unspawnable_command_string_start_command(self, toolset: ShellToolset[None], tmp_path: Path) -> None:
        with pytest.raises(ModelRetry, match='NUL byte'):
            await toolset.start_command(_ctx(tmp_path), 'echo \x00')

    @pytest.mark.parametrize('escaped', ['\udc80', '\udcff'])
    async def test_surrogateescape_command_still_runs(
        self, toolset: ShellToolset[None], escaped: str, tmp_path: Path
    ) -> None:
        # The spawn encodes with `surrogateescape`, which round-trips this range
        # back to the raw byte it came from. Screening the command as plain
        # UTF-8 would reject a command the OS runs.
        result = await toolset.run_command(_ctx(tmp_path), f'echo {escaped}')
        assert '[exit code' not in result

    @pytest.mark.parametrize('env', [{'FOO': 'bar\x00baz'}, {'FO\x00O': 'bar'}, {'FOO': 'bar\ud800'}])
    async def test_unspawnable_env_aborts(self, shell_dir: Path, env: dict[str, str]) -> None:
        # The spawn reports a NUL or an unencodable character as the same
        # `ValueError` wherever it came from. This one came from the
        # application's `env`, so the model cannot fix it and must not be asked
        # to retry.
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
            env=env,
        )
        with pytest.raises(ValueError) as exc_info:
            await ts.run_command(_ctx(shell_dir), 'echo hello')
        assert not isinstance(exc_info.value, ModelRetry)

    async def test_argument_or_environment_too_long_propagates(
        self, toolset: ShellToolset[None], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # E2BIG does not identify whether the model's command or the
        # application's environment crossed the combined platform limit. An
        # application configuration error must not become an unwinnable retry.
        monkeypatch.setattr(anyio, 'open_process', _raise_oserror(errno.E2BIG, 'Argument list too long'))
        with pytest.raises(OSError, match='Argument list too long'):
            await toolset.run_command(_ctx(tmp_path), 'echo hello')

    async def test_non_recoverable_errno_propagates(
        self, toolset: ShellToolset[None], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # A host that can't fork is not something the model can retry its way
        # out of, so it must keep aborting the run.
        monkeypatch.setattr(anyio, 'open_process', _raise_oserror(errno.ENOMEM, 'Cannot allocate memory'))
        with pytest.raises(OSError, match='Cannot allocate memory'):
            await toolset.run_command(_ctx(tmp_path), 'echo hello')


class TestRunCommand:
    async def test_basic_echo(self, toolset: ShellToolset[None], tmp_path: Path) -> None:
        result = await toolset.run_command(_ctx(tmp_path), 'echo hello')
        assert '[stdout]' in result
        assert 'hello' in result

    async def test_stderr_output(self, toolset: ShellToolset[None], tmp_path: Path) -> None:
        result = await toolset.run_command(_ctx(tmp_path), 'echo error >&2')
        assert '[stderr]' in result
        assert 'error' in result

    async def test_mixed_output(self, toolset: ShellToolset[None], tmp_path: Path) -> None:
        result = await toolset.run_command(_ctx(tmp_path), 'echo out && echo err >&2')
        assert '[stdout]' in result
        assert '[stderr]' in result

    async def test_exit_code_reported(self, toolset: ShellToolset[None], tmp_path: Path) -> None:
        result = await toolset.run_command(_ctx(tmp_path), 'exit 42')
        assert '[exit code: 42]' in result

    async def test_exit_code_zero_not_shown(self, toolset: ShellToolset[None], tmp_path: Path) -> None:
        result = await toolset.run_command(_ctx(tmp_path), 'echo ok')
        assert 'exit code' not in result

    async def test_no_output(self, toolset: ShellToolset[None], tmp_path: Path) -> None:
        result = await toolset.run_command(_ctx(tmp_path), 'true')
        assert result == '(no output)'

    async def test_output_truncation(self, shell_dir: Path) -> None:
        ts = _shell_toolset(shell_dir, max_output_chars=50)
        result = await _call_shell_tool(
            ts, shell_dir, 'run_command', command=f'{sys.executable} -c "print(\'x\' * 200)"'
        )
        assert len(result) == 50
        assert 'truncated, showing last 5 chars' in result

    async def test_output_truncation_caps_complete_failure_response(self, shell_dir: Path) -> None:
        ts = _shell_toolset(shell_dir, max_output_chars=200)
        command = f'{sys.executable} -c "import sys; sys.stdout.write(\'x\' * 400); sys.exit(7)"'
        result = await _call_shell_tool(ts, shell_dir, 'run_command', command=command)
        assert len(result) == 200
        assert result.startswith('[... output truncated, showing last 153 chars]\n')
        assert result.endswith('[exit code: 7]')

    async def test_persist_cwd(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=True,
            allow_interactive=False,
        )
        await ts.run_command(_ctx(shell_dir), 'cd subdir')
        result = await ts.run_command(_ctx(shell_dir), 'pwd')
        assert 'subdir' in result

    async def test_persist_cwd_keeps_directory_trailing_space(
        self, persist_toolset: ShellToolset[None], shell_dir: Path
    ) -> None:
        (shell_dir / 'trailing ').mkdir()
        ctx = _ctx(shell_dir)
        await persist_toolset.run_command(ctx, command="cd 'trailing '")
        result = await persist_toolset.run_command(ctx, command='pwd')
        assert 'trailing ' in result
        assert persist_toolset._cwd == str(shell_dir / 'trailing ')  # pyright: ignore[reportPrivateUsage]

    async def test_persist_cwd_only_on_success(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=True,
            allow_interactive=False,
        )
        original = ts._cwd  # pyright: ignore[reportPrivateUsage]
        await ts.run_command(_ctx(shell_dir), 'cd nonexistent_dir_xyz && false')
        assert ts._cwd == original  # pyright: ignore[reportPrivateUsage]

    async def test_denied_command_in_run(self, toolset: ShellToolset[None], tmp_path: Path) -> None:
        # B2: a denied command is model-correctable, so it surfaces as ModelRetry
        # (which pyai feeds back to the model) rather than aborting the run.
        with pytest.raises(ModelRetry, match="'rm' is denied"):
            await toolset.run_command(_ctx(tmp_path), 'rm -rf /')

    async def test_cwd_used(self, toolset: ShellToolset[None], shell_dir: Path) -> None:
        result = await toolset.run_command(_ctx(shell_dir), 'cat test.txt')
        assert 'hello' in result

    async def test_multiline_output(self, toolset: ShellToolset[None], tmp_path: Path) -> None:
        result = await toolset.run_command(_ctx(tmp_path), f'{sys.executable} -c "print(\'a\\nb\\nc\\n\')"')
        assert '[stdout]' in result

    async def test_timeout_reports_value(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=0.5,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.run_command(_ctx(shell_dir), 'sleep 10')
        assert 'timed out after 0.5s' in result

    async def test_custom_timeout_overrides_default(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=30.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.run_command(_ctx(shell_dir), 'sleep 10', timeout_seconds=0.5)
        assert 'timed out after 0.5s' in result

    async def test_persist_cwd_disabled_no_update(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        original = ts._cwd  # pyright: ignore[reportPrivateUsage]
        await ts.run_command(_ctx(shell_dir), 'cd subdir')
        assert ts._cwd == original  # pyright: ignore[reportPrivateUsage]

    async def test_nonzero_exit_shows_code(self, toolset: ShellToolset[None], tmp_path: Path) -> None:
        result = await toolset.run_command(_ctx(tmp_path), 'exit 1')
        assert '[exit code: 1]' in result

    async def test_stdout_stderr_separated_by_newline(self, toolset: ShellToolset[None], tmp_path: Path) -> None:
        result = await toolset.run_command(_ctx(tmp_path), 'echo out && echo err >&2')
        assert '[stdout]\nout\n\n[stderr]\nerr' in result

    async def test_non_ascii_stdout(self, toolset: ShellToolset[None], tmp_path: Path) -> None:
        result = await toolset.run_command(
            _ctx(tmp_path), f'{sys.executable} -c "import sys; sys.stdout.buffer.write(b\'hello \\xff\\xfe world\\n\')"'
        )
        assert 'hello' in result

    async def test_non_ascii_stderr(self, toolset: ShellToolset[None], tmp_path: Path) -> None:
        result = await toolset.run_command(
            _ctx(tmp_path), f'{sys.executable} -c "import sys; sys.stderr.buffer.write(b\'err \\xff\\xfe msg\\n\')"'
        )
        assert 'err' in result

    async def test_stdout_chunk_join(self, toolset: ShellToolset[None], tmp_path: Path) -> None:
        result = await toolset.run_command(_ctx(tmp_path), f"{sys.executable} -c \"print('A' * 100 + 'B' * 100)\"")
        assert 'A' * 100 + 'B' * 100 in result

    async def test_exit_code_fallback_to_zero(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=True,
            allow_interactive=False,
        )
        result = await ts.run_command(_ctx(shell_dir), 'echo ok')
        assert 'exit code' not in result

    async def test_error_message_content(self, shell_dir: Path) -> None:
        with pytest.raises(ValueError, match=r'^Specify allowed_commands or denied_commands, not both\.$'):
            ShellToolset(
                allowed_commands=['echo'],
                denied_commands=['rm'],
                denied_operators=[],
                default_timeout=10.0,
                max_output_chars=50_000,
                persist_cwd=False,
                allow_interactive=False,
            )

    def test_non_positive_max_output_chars_rejected(self, shell_dir: Path) -> None:
        # Matches LocalStackToolset: a cap of 0 would blank every response,
        # including start_command's ID line, leaving its process unstoppable.
        with pytest.raises(ValueError, match=r'max_output_chars must be a positive integer.'):
            _shell_toolset(shell_dir, max_output_chars=0)

    async def test_stdout_chunks_joined_cleanly(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=30.0,
            max_output_chars=500_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.run_command(_ctx(shell_dir), "printf '%05000d\\n' $(seq 1 100)")
        assert 'XXXX' not in result

    async def test_stderr_chunks_joined_cleanly(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=30.0,
            max_output_chars=500_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.run_command(_ctx(shell_dir), "printf '%0500d\\n' $(seq 1 100) >&2")
        assert 'XXXX' not in result

    async def test_persist_cwd_updates_after_cd(self, shell_dir: Path) -> None:
        """CWD should update to the actual directory after a successful cd."""
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=True,
            allow_interactive=False,
        )
        await ts.run_command(_ctx(shell_dir), 'cd subdir')
        assert ts._cwd == str(shell_dir / 'subdir')  # pyright: ignore[reportPrivateUsage]

    async def test_persist_cwd_not_updated_on_failure(self, shell_dir: Path) -> None:
        """CWD should not update if command fails (exit code non-zero)."""
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=True,
            allow_interactive=False,
        )
        original = ts._cwd  # pyright: ignore[reportPrivateUsage]
        await ts.run_command(_ctx(shell_dir), 'false')
        assert ts._cwd == original  # pyright: ignore[reportPrivateUsage]


class TestProcessGroupKill:
    async def test_timeout_kills_subprocess_tree(self, shell_dir: Path) -> None:
        """On timeout, the entire process group should be killed."""
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=0.5,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.run_command(_ctx(shell_dir), 'bash -c "sleep 100 & sleep 100"')
        assert 'timed out' in result

    async def test_timeout_with_output_before_timeout(self, shell_dir: Path) -> None:
        """Output produced before timeout should still result in timeout message."""
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=0.5,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.run_command(_ctx(shell_dir), 'echo before_timeout && sleep 100')
        assert 'timed out' in result

    async def test_start_new_session_used(self, shell_dir: Path) -> None:
        """Verify the child is in a different process group from the parent."""
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        parent_pgrp = os.getpgrp()
        result = await ts.run_command(
            _ctx(shell_dir), f'{sys.executable} -c "import os; print(os.getpgrp() != {parent_pgrp})"'
        )
        assert 'True' in result


class TestBackgroundCommands:
    async def test_start_command_returns_id(self, shell_dir: Path) -> None:
        ts = _shell_toolset(shell_dir)
        result = await _call_shell_tool(ts, shell_dir, 'start_command', command='sleep 100')
        assert 'ID:' in result
        assert 'Started background command' in result
        command_id = _parse_command_id(result)
        await ts.stop_command(_ctx(shell_dir), command_id)

    async def test_start_command_long_echo_is_capped_keeping_id(self, shell_dir: Path) -> None:
        # The command echo is subject to the cap like any other output; the ID
        # line is the tail, so truncation keeps it usable for check/stop calls.
        ts = _shell_toolset(shell_dir, max_output_chars=100)
        result = await _call_shell_tool(ts, shell_dir, 'start_command', command='true ' + 'x' * 200)
        assert len(result) == 100
        assert 'output truncated' in result
        command_id = _parse_command_id(result)
        assert len(command_id) == 32
        await ts.stop_command(_ctx(shell_dir), command_id)

    async def test_check_unknown_id(self, toolset: ShellToolset[None], tmp_path: Path) -> None:
        result = await toolset.check_command(_ctx(tmp_path), 'nonexistent_id')
        assert 'unknown command ID' in result

    async def test_stop_unknown_id(self, toolset: ShellToolset[None], tmp_path: Path) -> None:
        result = await toolset.stop_command(_ctx(tmp_path), 'nonexistent_id')
        assert 'unknown command ID' in result

    async def test_start_and_stop(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command(_ctx(shell_dir), 'echo hello_bg')
        command_id = _parse_command_id(start_result)

        await anyio.sleep(0.5)

        stop_result = await ts.stop_command(_ctx(shell_dir), command_id)
        assert 'stopped' in stop_result
        assert 'hello_bg' in stop_result
        assert stop_result.splitlines()[-2:] == ['[stopped]', '[exit code: 0]']

    async def test_start_and_check_running(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command(_ctx(shell_dir), 'sleep 100')
        command_id = _parse_command_id(start_result)

        check_result = await ts.check_command(_ctx(shell_dir), command_id)
        assert 'running' in check_result
        assert check_result.endswith('[status: running]')

        await ts.stop_command(_ctx(shell_dir), command_id)

    async def test_check_and_stop_respect_output_cap(self, shell_dir: Path) -> None:
        ts = _shell_toolset(shell_dir, max_output_chars=200)
        start_result = await ts.start_command(_ctx(shell_dir), "printf '%0400d' 0; sleep 30")
        command_id = _parse_command_id(start_result)
        await anyio.sleep(0.5)

        try:
            check_result = await _call_shell_tool(ts, shell_dir, 'check_command', command_id=command_id)
            assert len(check_result) == 200
            assert 'output truncated' in check_result
            assert check_result.endswith('[status: running]')
        finally:
            stop_result = await _call_shell_tool(ts, shell_dir, 'stop_command', command_id=command_id)
        assert len(stop_result) == 200
        assert 'output truncated' in stop_result
        stop_lines = stop_result.splitlines()
        assert stop_lines[-2] == '[stopped]'
        assert stop_lines[-1].startswith('[exit code:')

    async def test_new_string_tool_is_capped_at_dispatch(self, shell_dir: Path) -> None:
        ts = _shell_toolset(shell_dir, max_output_chars=1)

        def text() -> str:
            return 'xx'

        ts.add_function(text)
        assert await _call_shell_tool(ts, shell_dir, 'text') == 'x'

    async def test_non_string_tool_result_is_unchanged(self, shell_dir: Path) -> None:
        ts = _shell_toolset(shell_dir, max_output_chars=1)

        def number() -> int:
            return 42

        ts.add_function(number)
        ctx = _ctx(shell_dir)
        tools = await ts.get_tools(ctx)
        assert await ts.call_tool('number', {}, ctx, tools['number']) == 42

    async def test_start_and_check_finished(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command(_ctx(shell_dir), 'echo done_quick')
        command_id = _parse_command_id(start_result)

        await anyio.sleep(0.5)

        check_result = await ts.check_command(_ctx(shell_dir), command_id)
        assert 'finished' in check_result
        assert 'done_quick' in check_result
        assert check_result.splitlines()[-2:] == ['[status: finished]', '[exit code: 0]']

        await ts.stop_command(_ctx(shell_dir), command_id)

    async def test_start_denied_command_raises(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=['rm'],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        with pytest.raises(ModelRetry, match="'rm' is denied"):
            await ts.start_command(_ctx(shell_dir), 'rm -rf /')

    async def test_stop_captures_stderr(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command(_ctx(shell_dir), 'echo err_bg >&2')
        command_id = _parse_command_id(start_result)

        await anyio.sleep(0.5)

        stop_result = await ts.stop_command(_ctx(shell_dir), command_id)
        assert 'err_bg' in stop_result

    async def test_stop_no_output(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command(_ctx(shell_dir), 'true')
        command_id = _parse_command_id(start_result)

        await anyio.sleep(0.5)

        stop_result = await ts.stop_command(_ctx(shell_dir), command_id)
        assert '(no output)' in stop_result

    async def test_check_no_output_yet(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command(_ctx(shell_dir), 'sleep 100')
        command_id = _parse_command_id(start_result)

        check_result = await ts.check_command(_ctx(shell_dir), command_id)
        assert 'no output yet' in check_result

        await ts.stop_command(_ctx(shell_dir), command_id)

    async def test_check_command_captures_stderr(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command(_ctx(shell_dir), 'echo err_check >&2')
        command_id = _parse_command_id(start_result)

        await anyio.sleep(0.5)

        check_result = await ts.check_command(_ctx(shell_dir), command_id)
        assert '[stderr]' in check_result
        assert 'err_check' in check_result

        await ts.stop_command(_ctx(shell_dir), command_id)

    async def test_start_command_uses_cwd(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command(_ctx(shell_dir), 'pwd')
        command_id = _parse_command_id(start_result)

        await anyio.sleep(0.5)

        stop_result = await ts.stop_command(_ctx(shell_dir), command_id)
        assert str(shell_dir) in stop_result

    async def test_stop_removes_from_registry(self, shell_dir: Path) -> None:
        """After stop, the command_id should no longer be known."""
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command(_ctx(shell_dir), 'true')
        command_id = _parse_command_id(start_result)

        await anyio.sleep(0.5)

        await ts.stop_command(_ctx(shell_dir), command_id)

        # Should now be unknown
        check_result = await ts.check_command(_ctx(shell_dir), command_id)
        assert 'unknown command ID' in check_result

    async def test_start_command_cleans_temp_files_on_failure(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        with patch('anyio.open_process', side_effect=OSError('spawn failed')):
            with pytest.raises(OSError, match='spawn failed'):
                await ts.start_command(_ctx(shell_dir), 'echo hi')

    async def test_background_command_outlives_the_run(self, shell_dir: Path) -> None:
        # A conversation is several runs: a later run's toolset checks and stops a job an earlier one started.
        shared = _shell_toolset(shell_dir)
        first = await shared.for_run(_ctx(shell_dir))
        assert isinstance(first, ShellToolset)
        pid_pipe = shell_dir / 'pid'
        os.mkfifo(pid_pipe)
        async with first:
            command_id = _parse_command_id(
                await first.start_command(_ctx(shell_dir), f'echo $$ > {pid_pipe}; exec sleep 300')
            )
            # Reading the FIFO blocks until the command has written its PID: no polling, so no timing.
            with anyio.fail_after(10):
                pid = int(await anyio.to_thread.run_sync(pid_pipe.read_text))

        second = await shared.for_run(_ctx(shell_dir))
        assert isinstance(second, ShellToolset)
        assert (await second.check_command(_ctx(shell_dir), command_id)).endswith('[status: running]')
        stopped = await second.stop_command(_ctx(shell_dir), command_id)
        assert stopped.splitlines()[-2:] == ['[stopped]', '[exit code: 143]']
        await _wait_for_exit(pid)

    @pytest.mark.parametrize('command_id', ['../../etc', 'f' * 32])
    async def test_id_naming_no_job_is_unknown(self, shell_dir: Path, command_id: str) -> None:
        ts = _shell_toolset(shell_dir)
        assert 'unknown command ID' in await ts.check_command(_ctx(shell_dir), command_id)

    async def test_handle_that_is_a_directory_is_unknown(self, shell_dir: Path) -> None:
        ts = _shell_toolset(shell_dir)
        command_id = 'd' * 32
        (Path(await ts._jobs_base(_ctx(shell_dir))) / command_id / 'handle').mkdir(parents=True)  # pyright: ignore[reportPrivateUsage]
        assert 'unknown command ID' in await ts.check_command(_ctx(shell_dir), command_id)

    async def test_unreadable_handle_is_unknown(self, shell_dir: Path) -> None:
        ts = _shell_toolset(shell_dir)
        command_id = 'e' * 32
        job_dir = Path(await ts._jobs_base(_ctx(shell_dir))) / command_id  # pyright: ignore[reportPrivateUsage]
        job_dir.mkdir()
        (job_dir / 'handle').write_text('garbage\n')
        assert 'unknown command ID' in await ts.stop_command(_ctx(shell_dir), command_id)
        # A job whose group was not its own to signal records `-`; checking it signals nothing.
        (job_dir / 'handle').write_text(f'{2**22 + 12345} -\n')
        job = await _job(ts, _ctx(shell_dir), command_id)
        assert job.pgid is None
        assert await ts.check_command(_ctx(shell_dir), command_id) == '(no output yet)\n[status: running]'


class TestEdgeCases:
    async def test_toolset_tool_names(self, toolset: ShellToolset[None]) -> None:
        tool_names = list(toolset.tools.keys())
        assert 'run_command' in tool_names
        assert 'start_command' in tool_names
        assert 'check_command' in tool_names
        assert 'stop_command' in tool_names

    async def test_run_command_uses_actual_cwd(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.run_command(_ctx(shell_dir), 'pwd')
        assert str(shell_dir) in result

    async def test_persist_cwd_requires_all_three_conditions(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=True,
            allow_interactive=False,
        )
        # Successful echo -- sentinel shows same dir, cwd should remain valid
        await ts.run_command(_ctx(shell_dir), 'echo hi')
        assert ts._cwd == str(shell_dir)  # pyright: ignore[reportPrivateUsage]


class TestShellCapability:
    def test_default_construction(self) -> None:
        shell = Shell()
        assert shell.cwd is None
        assert shell.default_timeout == 30.0
        assert 'rm' in shell.denied_commands

    def test_custom_construction(self) -> None:
        shell = Shell(
            allowed_commands=['echo', 'cat'],
            denied_commands=[],
            default_timeout=60.0,
        )
        assert shell.default_timeout == 60.0
        shell.get_toolset()

    async def test_empty_allowlist_keeps_default_denylist(self, tmp_path: Path) -> None:
        toolset = Shell(allowed_commands=[]).get_toolset()

        with pytest.raises(ModelRetry, match="'rm' is denied"):
            await toolset.run_command(_ctx(tmp_path), 'rm --version')
        assert 'hello' in await toolset.run_command(_ctx(tmp_path), 'echo hello')

    def test_explicit_default_denylist_conflicts_with_allowlist(self) -> None:
        denied_commands = Shell().denied_commands
        shell = Shell(allowed_commands=['rm'], denied_commands=denied_commands)

        with pytest.raises(ValueError, match='Specify allowed_commands or denied_commands'):
            shell.get_toolset()

    def test_agent_accepts_allowlist_without_explicit_denylist(self, tmp_path: Path) -> None:
        Agent(TestModel(), capabilities=[Shell(allowed_commands=['ls', 'cat', 'rg'])])

    def test_get_toolset_returns_toolset(self, tmp_path: Path) -> None:
        shell = Shell()
        toolset = shell.get_toolset()
        assert isinstance(toolset, ShellToolset)

    def test_default_denied_commands(self) -> None:
        shell = Shell()
        assert 'rm' in shell.denied_commands
        assert 'dd' in shell.denied_commands
        assert 'shutdown' in shell.denied_commands

    async def test_agent_integration(self, tmp_path: Path) -> None:
        model = TestModel(custom_output_text='done', call_tools=[])
        agent: Agent[None, str] = Agent(model, capabilities=[Shell()])
        result = await agent.run('run echo hello', workspace=LocalWorkspaceBackend(tmp_path))
        assert result.output == 'done'

    async def test_no_workspace_fails_the_run(self) -> None:
        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        agent: Agent[None, str] = Agent(TestModel(call_tools=[]), capabilities=[Shell()])
        with pytest.raises(UserError, match='`Shell` needs a workspace'):
            await agent.run('run echo hello')

    async def test_hand_built_toolset_without_workspace_fails_the_run(self, tmp_path: Path) -> None:
        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        agent: Agent[None, str] = Agent(
            TestModel(call_tools=[]), deps_type=type(None), toolsets=[_shell_toolset(tmp_path)]
        )
        with pytest.raises(UserError, match='`ShellToolset` needs a workspace'):
            await agent.run('run echo hello')


async def _tools_offered_to_model(cwd: Path, *, shell_first: bool) -> dict[str, str | None]:
    """Run an agent with Shell and CodeMode and return the tools the model was offered."""
    offered: dict[str, str | None] = {}

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        offered.update({tool.name: tool.description for tool in info.function_tools})
        return ModelResponse(parts=[TextPart('done')])

    shell = Shell[object]()
    code_mode = CodeMode[object]()
    capabilities: list[AbstractCapability[object]] = [shell, code_mode] if shell_first else [code_mode, shell]
    agent: Agent[None, str] = Agent(FunctionModel(capture), capabilities=capabilities)
    await agent.run('go', workspace=LocalWorkspaceBackend(cwd))
    return offered


class TestCodeModeInterop:
    """`run_command` and `start_command` take a command line, so CodeMode leaves them native.

    Folding them into `run_code` would make the model write a Monty script whose argument is a
    shell script quoted as a Python string. The command-id tools carry no command line, so they
    stay sandboxed like any other tool.
    """

    @pytest.mark.parametrize('shell_first', [True, False], ids=['shell-first', 'code-mode-first'])
    async def test_command_tools_stay_native(self, tmp_path: Path, shell_first: bool) -> None:

        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        tools = await _tools_offered_to_model(tmp_path, shell_first=shell_first)

        assert 'run_command' in tools
        assert 'start_command' in tools
        run_code_description = tools['run_code']
        assert run_code_description is not None
        assert 'async def run_command' not in run_code_description
        assert 'async def start_command' not in run_code_description

    @pytest.mark.parametrize('shell_first', [True, False], ids=['shell-first', 'code-mode-first'])
    async def test_command_id_tools_are_still_sandboxed(self, tmp_path: Path, shell_first: bool) -> None:

        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        tools = await _tools_offered_to_model(tmp_path, shell_first=shell_first)

        assert 'check_command' not in tools
        assert 'stop_command' not in tools
        run_code_description = tools['run_code']
        assert run_code_description is not None
        assert 'async def check_command' in run_code_description
        assert 'async def stop_command' in run_code_description


class TestStopEscalation:
    async def test_finished_job_status_is_read_once(self, shell_dir: Path) -> None:
        ts = _shell_toolset(shell_dir)
        command_id = _parse_command_id(await ts.start_command(_ctx(shell_dir), 'true'))
        job = await _job(ts, _ctx(shell_dir), command_id)
        with anyio.fail_after(10):
            while (await job.status())[0]:
                await anyio.sleep(0.01)  # pragma: lax no cover
        await job.cleanup()
        assert (await job.status())[0] is False

    async def test_stop_signals_group_after_wrapper_exits(self, shell_dir: Path) -> None:
        ts = _shell_toolset(shell_dir)
        command_id = _parse_command_id(await ts.start_command(_ctx(shell_dir), 'exec sleep 30'))
        job = await _job(ts, _ctx(shell_dir), command_id)
        # A published finished status may precede the exit of another group member.
        with patch.object(Job, 'status', return_value=(False, 0)):
            assert '[stopped]' in await ts.stop_command(_ctx(shell_dir), command_id)
        try:
            await _wait_for_exit(job.pid)
        finally:
            try:
                os.kill(job.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    async def test_stop_escalates_to_sigkill(self, shell_dir: Path) -> None:
        """A group that ignores SIGTERM is killed after the grace period, with no exit code to report."""
        ts = _shell_toolset(shell_dir)
        ready = shell_dir / 'ready'
        start = await ts.start_command(_ctx(shell_dir), f"trap '' TERM; echo $$ > {ready}; while :; do sleep 1; done")
        command_id = _parse_command_id(start)
        with anyio.fail_after(10):
            while not ready.exists() or not ready.read_text().strip():
                await anyio.sleep(0.01)  # pragma: lax no cover
        pid = int(ready.read_text())
        with patch('pydantic_ai_harness.shell._jobs._KILL_GRACE_PERIOD', 0.2):
            result = await ts.stop_command(_ctx(shell_dir), command_id)
        assert result.endswith('[stopped]')
        await _wait_for_exit(pid)

    async def test_stop_after_process_already_exited(self, shell_dir: Path) -> None:
        """A job that exited between the status read and the signal is not an error."""
        ts = _shell_toolset(shell_dir)
        command_id = _parse_command_id(await ts.start_command(_ctx(shell_dir), 'true'))
        job = await _job(ts, _ctx(shell_dir), command_id)
        with anyio.fail_after(10):
            while (await job.status())[0]:
                await anyio.sleep(0.01)  # pragma: lax no cover
        job.pgid = None
        job.pid = 2**22 + 12345  # beyond any live PID, so `kill` finds no process
        await job.kill()


_KILL_SCRIPT = 'true 2> /dev/null > "$3"; kill -s "$1" -- "$2"'


class _RecordingKill(LocalWorkspaceBackend):
    """A local backend that records every argv command and can answer the signal command itself."""

    def __init__(self, working_dir: str | Path, *, kill_result: CommandResult | None = None) -> None:
        super().__init__(working_dir)
        self.argv: list[list[str]] = []
        self.kill_result = kill_result

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        if not isinstance(command, str):
            self.argv.append(list(command))
            if self.kill_result is not None and command[:3] == ['sh', '-c', _KILL_SCRIPT]:
                return self.kill_result
        return await super().run(command, shell=shell, env=env, timeout=timeout)


class _NeverReady(LocalWorkspaceBackend):
    """A local backend whose launcher never tells the wrapper to start the command.

    With `launcher_alive`, the wrapper is handed a PID that outlives the launcher (this process),
    so it keeps waiting; otherwise the launcher exits like one cancelled before `launch.ready`.
    """

    def __init__(self, working_dir: str | Path, *, launcher_alive: bool = True) -> None:
        super().__init__(working_dir)
        self.launcher_alive = launcher_alive

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        if isinstance(command, str):
            command = command.replace(': > "$dir/launch.ready"\n', '')
            if self.launcher_alive:
                command = command.replace('"$limit" $$ <', f'"$limit" {os.getpid()} <')
        return await super().run(command, shell=shell, env=env, timeout=timeout)


class TestSignalling:
    async def test_signals_go_through_the_shell_builtin(self, shell_dir: Path) -> None:
        # Slim images ship no `kill` executable, so no argv may start with a bare `kill`.
        backend = _RecordingKill(shell_dir)
        ts = _shell_toolset(shell_dir)
        ctx = _run_context(Workspace(backend))
        command_id = _parse_command_id(await ts.start_command(ctx, 'exec sleep 300'))
        job = await _job(ts, ctx, command_id)
        stopped = await ts.stop_command(ctx, command_id)
        assert stopped.splitlines()[-2:] == ['[stopped]', '[exit code: 143]']
        signals = [argv for argv in backend.argv if argv[:3] == ['sh', '-c', _KILL_SCRIPT]]
        target = f'-{job.pgid}' if job.pgid is not None else str(job.pid)
        stop_file = posixpath.join(job.directory, 'stop')
        assert signals[0] == ['sh', '-c', _KILL_SCRIPT, 'kill', 'TERM', target, stop_file]
        assert all(argv[-3:] == ['0', target, stop_file] for argv in signals[1:])
        assert all(argv[0] != 'kill' for argv in backend.argv)
        await _wait_for_exit(job.pid)

    async def test_stop_before_the_command_starts_never_starts_it(self, shell_dir: Path) -> None:
        # Without `launch.ready` the wrapper waits forever, so the stop always lands before the command.
        backend = _NeverReady(shell_dir)
        ts = _shell_toolset(shell_dir)
        ctx = _run_context(Workspace(backend))
        command_id = _parse_command_id(await ts.start_command(ctx, 'echo started; exec sleep 300'))
        job = await _job(ts, ctx, command_id)
        stopped = await ts.stop_command(ctx, command_id)
        assert stopped.splitlines() == ['(no output)', '[stopped]', '[exit code: 143]']
        await _wait_for_exit(job.pid)

    async def test_a_stop_file_keeps_the_command_from_starting(self, shell_dir: Path) -> None:
        # A SIGTERM lost while the command's subshell forks leaves only the stop file `Job.kill` creates first.
        ts = _shell_toolset(shell_dir)
        ctx = _run_context(Workspace(_NeverReady(shell_dir)))
        command_id = _parse_command_id(await ts.start_command(ctx, 'echo started'))
        job = await _job(ts, ctx, command_id)
        (Path(job.directory) / 'stop').touch()
        (Path(job.directory) / 'launch.ready').touch()
        with anyio.fail_after(30):
            while (status := await job.status())[0]:
                await anyio.sleep(0.05)  # pragma: lax no cover
        assert status == (False, 143)
        assert Path(job.output_path).read_text(encoding='utf-8') == ''
        await job.cleanup()

    async def test_a_launcher_gone_before_start_never_starts_the_command(self, shell_dir: Path) -> None:
        # A launch cancelled before `launch.ready` must not leave its detached wrapper waiting forever.
        ts = _shell_toolset(shell_dir)
        ctx = _run_context(Workspace(_NeverReady(shell_dir, launcher_alive=False)))
        command_id = _parse_command_id(await ts.start_command(ctx, 'echo started'))
        job = await _job(ts, ctx, command_id)
        with anyio.fail_after(30):  # hang guard only
            while (status := await job.status())[0]:
                await anyio.sleep(0.05)  # pragma: lax no cover
        assert status == (False, 143)
        assert Path(job.output_path).read_text(encoding='utf-8') == ''
        await job.cleanup()

    async def test_failed_signal_is_not_reported_as_stopped(self, shell_dir: Path) -> None:
        backend = _RecordingKill(
            shell_dir, kill_result=CommandResult(exit_code=1, stdout='', stderr='kill: Operation not permitted')
        )
        ts = _shell_toolset(shell_dir)
        ctx = _run_context(Workspace(backend))
        command_id = _parse_command_id(await ts.start_command(ctx, 'exec sleep 300'))
        job = await _job(ts, ctx, command_id)
        try:
            tools = await ts.get_tools(ctx)
            with pytest.raises(ToolFailed, match='Operation not permitted'):
                await ts.call_tool('stop_command', {'command_id': command_id}, ctx, tools['stop_command'])
            assert (await job.status())[0]
        finally:
            backend.kill_result = None
            await job.kill()
            await job.cleanup()

    async def test_failed_signal_without_stderr_names_the_signal(self, shell_dir: Path) -> None:
        backend = _RecordingKill(shell_dir, kill_result=CommandResult(exit_code=2, stdout='', stderr=''))
        ts = _shell_toolset(shell_dir)
        ctx = _run_context(Workspace(backend))
        command_id = _parse_command_id(await ts.start_command(ctx, 'exec sleep 300'))
        job = await _job(ts, ctx, command_id)
        try:
            with pytest.raises(WorkspaceError, match=f'Unable to send SIGTERM to job {job.pid}'):
                await job.kill()
        finally:
            backend.kill_result = None
            await job.kill()
            await job.cleanup()


async def _wait_for_exit(pid: int) -> None:
    """Wait for `pid` to be reaped; a process init reaps may linger as a zombie for a moment."""
    with anyio.fail_after(10):
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            await anyio.sleep(0.01)  # pragma: lax no cover


class _KillGroupOnExit(LocalWorkspaceBackend):
    """A local backend that kills each command's process group the moment the command exits, as Sprites does."""

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        assert isinstance(command, str) and shell
        async with await anyio.open_process(
            ['/bin/sh', '-c', command],
            cwd=await self.working_dir(),
            env={**self._env, **(env or {})},
            start_new_session=True,
        ) as process:
            assert process.stdout is not None
            stdout = b''.join([chunk async for chunk in process.stdout])
            exit_code = await process.wait()
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        return CommandResult(exit_code=exit_code, stdout=stdout.decode(), stderr='')


class TestLaunch:
    async def test_the_job_survives_its_launcher_process_group_being_killed(self, tmp_path: Path) -> None:
        # A `setsid` that detaches only after a delay: the launcher must wait for it before exiting.
        bin_dir = tmp_path / 'bin'
        bin_dir.mkdir()
        gate = tmp_path / 'gate'
        os.mkfifo(gate)
        slow_setsid = bin_dir / 'setsid'
        slow_setsid.write_text(
            f'#!{sys.executable}\n'
            'import os, sys\n'
            f'open({str(tmp_path / "ready")!r}, "w").close()\n'
            f'with open({str(gate)!r}, "rb") as fifo: fifo.read(1)\n'
            'os.setsid()\n'
            'os.execvp(sys.argv[1], sys.argv[1:])\n'
        )
        slow_setsid.chmod(0o755)
        backend = _KillGroupOnExit(tmp_path, env={'PATH': f'{bin_dir}:{os.environ["PATH"]}'})
        ts = _shell_toolset(tmp_path)
        ctx = _run_context(Workspace(backend))
        launched = anyio.Event()
        ids: list[str] = []

        async def launch() -> None:
            ids.append(_parse_command_id(await ts.start_command(ctx, 'echo finished')))
            launched.set()

        async with anyio.create_task_group() as tg:
            tg.start_soon(launch)
            with anyio.fail_after(60):
                while not (tmp_path / 'ready').exists():
                    await anyio.sleep(0.01)
                await anyio.to_thread.run_sync(gate.write_bytes, b'go')
                await launched.wait()
        job = await _job(ts, ctx, ids[0])
        with anyio.fail_after(60):
            while (status := await job.status())[0]:
                await anyio.sleep(0.05)  # pragma: lax no cover
        assert status == (False, 0)
        assert Path(job.output_path).read_text(encoding='utf-8') == 'finished\n'

    @pytest.mark.parametrize(('lacking', 'named'), [(('mv',), '`mv`'), (('mv', 'base64'), '`mv` and `base64`')])
    async def test_missing_job_tools_fail_the_launch_by_name(
        self, tmp_path: Path, lacking: tuple[str, ...], named: str
    ) -> None:
        bin_dir = tmp_path / 'bin'
        bin_dir.mkdir()
        for source in (Path('/usr/bin'), Path('/bin')):
            for tool in source.iterdir():
                if tool.name not in lacking and not os.path.lexists(bin_dir / tool.name):
                    (bin_dir / tool.name).symlink_to(tool)
        backend = LocalWorkspaceBackend(tmp_path, env={'PATH': str(bin_dir)})
        with pytest.raises(ToolFailed) as failed:
            await _shell_toolset(tmp_path).start_command(_run_context(Workspace(backend)), 'echo hello')
        assert failed.value.message == (
            f'Shell needs `mv` and `base64` on PATH in the workspace; this image lacks {named}.'
        )


class TestReadBgOutputEdgeCases:
    async def test_missing_logs_read_as_empty(self, shell_dir: Path) -> None:
        """A log removed from the workspace reads as empty rather than failing the check."""
        ts = _shell_toolset(shell_dir)
        command_id = _parse_command_id(await ts.start_command(_ctx(shell_dir), 'exec sleep 300'))
        job = await _job(ts, _ctx(shell_dir), command_id)
        (Path(job.directory) / 'stdout.log').unlink()
        (Path(job.directory) / 'stderr.log').unlink()
        try:
            result = await ts.check_command(_ctx(shell_dir), command_id)
            assert result == '(no output yet)\n[status: running]'
        finally:
            await ts.stop_command(_ctx(shell_dir), command_id)

    async def test_unreadable_log_is_a_failed_call(self, shell_dir: Path) -> None:
        """A log the workspace cannot read is reported to the model as a failed tool call."""
        ts = _shell_toolset(shell_dir)
        command_id = _parse_command_id(await ts.start_command(_ctx(shell_dir), 'exec sleep 300'))
        job = await _job(ts, _ctx(shell_dir), command_id)
        stdout_log = Path(job.directory) / 'stdout.log'
        stdout_log.write_text('secret')
        stdout_log.chmod(0)
        try:
            if os.access(stdout_log, os.R_OK):  # pragma: no cover - root reads regardless of mode bits
                pytest.skip('mode bits do not bind this user')
            with pytest.raises(ToolFailed) as exc_info:
                await ts.check_command(_ctx(shell_dir), command_id)
            assert exc_info.value.message == 'Unable to read job log ' + repr(str(stdout_log)) + '.'
        finally:
            stdout_log.chmod(0o600)
            await ts.stop_command(_ctx(shell_dir), command_id)


class _NoStatSizes(LocalWorkspaceBackend):
    """A backend whose `stat` reports no file sizes, as the workspace protocol allows."""

    async def stat(self, path: str) -> FileEntry:
        return replace(await super().stat(path), size=None)


class TestJobLogSizes:
    async def test_log_is_counted_in_the_workspace_when_stat_has_no_size(self, shell_dir: Path) -> None:
        ts = _shell_toolset(shell_dir)
        ctx = _run_context(Workspace(_NoStatSizes(shell_dir)))
        job = await _job(ts, ctx, _parse_command_id(await ts.start_command(ctx, 'echo finished')))
        with anyio.fail_after(5):
            while (await job.status())[0]:
                await anyio.sleep(0.05)  # pragma: lax no cover
        assert await job.size(job.output_path) == len(b'finished\n')
        assert await job.tail(job.output_path, 100) == b'finished\n'
        with pytest.raises(WorkspaceError, match='Is a directory'):
            await job.size(job.directory)


class TestJobStatusEdgeCases:
    async def test_unparseable_status_counts_as_running(self, shell_dir: Path) -> None:
        ts = _shell_toolset(shell_dir)
        command_id = _parse_command_id(await ts.start_command(_ctx(shell_dir), 'exec sleep 300'))
        job = await _job(ts, _ctx(shell_dir), command_id)
        try:
            (Path(job.directory) / 'status.json').write_text('not json')
            assert (await ts.check_command(_ctx(shell_dir), command_id)).endswith('[status: running]')
            (Path(job.directory) / 'status.json').unlink()
            assert (await ts.check_command(_ctx(shell_dir), command_id)).endswith('[status: running]')
        finally:
            await ts.stop_command(_ctx(shell_dir), command_id)


class TestCleanupBgFilesEdgeCases:
    async def test_cleanup_of_removed_directory(self, shell_dir: Path) -> None:
        """A job directory already removed from the workspace is not an error."""
        ts = _shell_toolset(shell_dir)
        command_id = _parse_command_id(await ts.start_command(_ctx(shell_dir), 'true'))
        job = await _job(ts, _ctx(shell_dir), command_id)
        with anyio.fail_after(10):
            # Removing the directory while the wrapper still publishes its status races the wrapper.
            while (await job.status())[0]:
                await anyio.sleep(0.01)  # pragma: lax no cover
        await job.cleanup()
        await job.cleanup()
        assert not Path(job.directory).exists()


class TestStopCommandAlreadyFinished:
    async def test_stop_already_finished_process(self, shell_dir: Path) -> None:
        """stop_command on an already-finished process skips the kill and reports its exit code."""
        ts = _shell_toolset(shell_dir)
        command_id = _parse_command_id(await ts.start_command(_ctx(shell_dir), 'exit 3'))
        job = await _job(ts, _ctx(shell_dir), command_id)
        with anyio.fail_after(10):
            while (await job.status())[0]:
                await anyio.sleep(0.01)  # pragma: lax no cover
        result = await ts.stop_command(_ctx(shell_dir), command_id)
        assert result.splitlines()[-2:] == ['[stopped]', '[exit code: 3]']


class TestResolveEnv:
    """Unit coverage for the env-resolution branches."""

    def test_adds_nothing_when_unconfigured(self, shell_dir: Path) -> None:
        # No env -> None: commands get the workspace's own environment unchanged.
        assert _env_toolset(shell_dir)._resolve_env() is None  # pyright: ignore[reportPrivateUsage]

    def test_patterns_alone_add_nothing(self, shell_dir: Path) -> None:
        # Patterns filter only an explicit `env`; the workspace environment is its provider's.
        assert _env_toolset(shell_dir, denied_env_patterns=['OPENAI_*'])._resolve_env() is None  # pyright: ignore[reportPrivateUsage]

    def test_explicit_env_is_added(self, shell_dir: Path) -> None:
        resolved = _env_toolset(shell_dir, env={'FOO': 'bar'})._resolve_env()  # pyright: ignore[reportPrivateUsage]
        assert resolved == {'FOO': 'bar'}

    def test_explicit_empty_env(self, shell_dir: Path) -> None:
        assert _env_toolset(shell_dir, env={})._resolve_env() == {}  # pyright: ignore[reportPrivateUsage]

    def test_patterns_strip_from_explicit_env(self, shell_dir: Path) -> None:
        resolved = _env_toolset(
            shell_dir,
            env={'OPENAI_API_KEY': 'secret', 'PATH': '/usr/bin'},
            denied_env_patterns=['OPENAI_*'],
        )._resolve_env()  # pyright: ignore[reportPrivateUsage]
        assert resolved == {'PATH': '/usr/bin'}

    def test_patterns_no_match_keeps_base(self, shell_dir: Path) -> None:
        resolved = _env_toolset(
            shell_dir,
            env={'FOO': 'bar'},
            denied_env_patterns=['OPENAI_*'],
        )._resolve_env()  # pyright: ignore[reportPrivateUsage]
        assert resolved == {'FOO': 'bar'}

    def test_pattern_match_is_case_sensitive(self, shell_dir: Path) -> None:
        # Env var names are case-sensitive on POSIX; lowercase must not match.
        resolved = _env_toolset(
            shell_dir,
            env={'openai_api_key': 'secret'},
            denied_env_patterns=['OPENAI_*'],
        )._resolve_env()  # pyright: ignore[reportPrivateUsage]
        assert resolved == {'openai_api_key': 'secret'}


class TestEnvControlExecution:
    """End-to-end: the resolved env actually reaches spawned subprocesses."""

    async def test_explicit_env_seen_by_command(self, shell_dir: Path) -> None:
        ts = _env_toolset(shell_dir, env={'MY_TOKEN': 'present'})
        result = await ts.run_command(_ctx(shell_dir), _read_env_var('MY_TOKEN'))
        assert 'present' in result

    async def test_host_environment_does_not_reach_commands(
        self, shell_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A local workspace passes on only the host's `PATH`, `HOME`, `LANG`, `LC_ALL` and `LC_CTYPE`.
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'leak-me')
        monkeypatch.setenv('HARNESS_INHERITED', 'yes')
        ts = _env_toolset(shell_dir)
        result = await ts.run_command(
            _ctx(shell_dir), _read_env_var('ANTHROPIC_API_KEY') + '; ' + _read_env_var('HARNESS_INHERITED')
        )
        assert 'leak-me' not in result and 'yes' not in result

    async def test_workspace_environment_passes_through(self, shell_dir: Path) -> None:
        ts = _env_toolset(shell_dir, denied_env_patterns=['ANTHROPIC_*'])
        result = await ts.run_command(_ctx(shell_dir), 'printf "$PATH"')
        assert os.environ['PATH'] in result

    async def test_env_and_patterns_compose_at_spawn(self, shell_dir: Path) -> None:
        # Both set: a pattern strips a key from the explicit env, the rest survives.
        ts = _env_toolset(
            shell_dir,
            env={'SECRET_KEY': 'leak-me', 'KEEP_VAR': 'kept'},
            denied_env_patterns=['SECRET_*'],
        )
        stripped = await ts.run_command(_ctx(shell_dir), _read_env_var('SECRET_KEY'))
        assert 'ABSENT' in stripped
        assert 'leak-me' not in stripped
        survived = await ts.run_command(_ctx(shell_dir), _read_env_var('KEEP_VAR'))
        assert 'kept' in survived

    async def test_background_command_honors_env(self, shell_dir: Path) -> None:
        ts = _env_toolset(shell_dir, env={'BG_TOKEN': 'bg-present'})
        start_result = await ts.start_command(_ctx(shell_dir), _read_env_var('BG_TOKEN'))
        command_id = _parse_command_id(start_result)
        await anyio.sleep(0.5)
        stop_result = await ts.stop_command(_ctx(shell_dir), command_id)
        assert 'bg-present' in stop_result


class TestEnvControlPropagation:
    """The capability and `for_run` carry env control through unchanged."""

    async def test_for_run_propagates_env(self, shell_dir: Path) -> None:
        ts = _env_toolset(shell_dir, env={'FOO': 'bar'}, denied_env_patterns=['OPENAI_*'])
        run_ts = await ts.for_run(_ctx(shell_dir))
        assert isinstance(run_ts, ShellToolset)
        assert run_ts._resolve_env() == {'FOO': 'bar'}  # pyright: ignore[reportPrivateUsage]

    def test_capability_defaults_add_nothing(self) -> None:
        shell = Shell()
        assert shell.env is None
        assert list(shell.denied_env_patterns) == []

    def test_capability_passes_env_to_toolset(self, tmp_path: Path) -> None:
        shell = Shell(
            env={'FOO': 'bar'},
            denied_env_patterns=['OPENAI_*'],
        )
        toolset = shell.get_toolset()
        assert isinstance(toolset, ShellToolset)
        assert toolset._resolve_env() == {'FOO': 'bar'}  # pyright: ignore[reportPrivateUsage]

    def test_llm_pattern_constant_strips_provider_keys(self, tmp_path: Path) -> None:
        env = {name: 'secret' for name in ('ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'OPENROUTER_API_KEY')}
        env |= {'GEMINI_API_KEY': 'secret', 'GOOGLE_API_KEY': 'secret', 'GATEWAY_KEY': 'secret'}
        env |= {'PYDANTIC_AI_GATEWAY_API_KEY': 'secret', 'PATH': '/usr/bin'}
        shell = Shell(env=env, denied_env_patterns=list(LLM_API_KEY_ENV_PATTERNS))
        toolset = shell.get_toolset()
        assert isinstance(toolset, ShellToolset)
        # None of the provider-credential prefixes survive.
        assert toolset._resolve_env() == {'PATH': '/usr/bin'}  # pyright: ignore[reportPrivateUsage]


class TestReadOnlyWorkspace:
    """A read-only workspace refuses `run`, so the shell offers no tools and reports a refusal as a failure."""

    async def test_agent_is_offered_no_shell_tools(self, tmp_path: Path) -> None:
        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        model = TestModel(call_tools=[])
        capabilities: list[AbstractCapability[None]] = [
            Shell[None](tools=['run_command', 'shell']),
            LocalWorkspace[None](tmp_path, read_only=True),
        ]
        await Agent(model, deps_type=type(None), capabilities=capabilities).run('Inspect tools')
        assert model.last_model_request_parameters is not None
        assert model.last_model_request_parameters.function_tools == []

    async def test_get_tools_is_empty(self, tmp_path: Path) -> None:
        ts = _shell_toolset(tmp_path)
        read_only = _run_context(ReadOnlyWorkspace(Workspace(LocalWorkspaceBackend(tmp_path))))
        assert await ts.get_tools(read_only) == {}
        assert set(await ts.get_tools(_ctx(tmp_path))) == {
            'run_command',
            'start_command',
            'check_command',
            'stop_command',
        }

    async def test_refusal_is_a_failed_tool_call(self, tmp_path: Path) -> None:
        ts = _shell_toolset(tmp_path)
        writable = _ctx(tmp_path)
        tools = await ts.get_tools(writable)
        read_only = _run_context(ReadOnlyWorkspace(Workspace(LocalWorkspaceBackend(tmp_path))))
        with pytest.raises(ToolFailed) as exc_info:
            await ts.call_tool('run_command', {'command': 'echo hi'}, read_only, tools['run_command'])
        assert exc_info.value.message == READ_ONLY_FAILURE


class TestDetachedJobRoundTrip:
    async def test_start_check_stop(self, tmp_path: Path) -> None:
        ts = _shell_toolset(tmp_path)
        command_id = _parse_command_id(
            await ts.start_command(_ctx(tmp_path), 'echo started; echo warn >&2; echo made > made.txt; exec sleep 300')
        )
        job_dir = Path((await _job(ts, _ctx(tmp_path), command_id)).directory)
        with anyio.fail_after(10):
            while 'started' not in (checked := await ts.check_command(_ctx(tmp_path), command_id)):
                await anyio.sleep(0.05)  # pragma: lax no cover
        assert checked.endswith('[status: running]')
        assert '[stderr]\nwarn' in checked
        assert (tmp_path / 'made.txt').read_text() == 'made\n'
        assert json.loads((job_dir / 'status.json').read_text())['exit_code'] is None

        stopped = await ts.stop_command(_ctx(tmp_path), command_id)
        assert stopped.splitlines()[-2:] == ['[stopped]', '[exit code: 143]']
        assert 'started' in stopped
        assert not job_dir.exists()
        assert 'unknown command ID' in await ts.check_command(_ctx(tmp_path), command_id)

    async def test_persistent_background_job(self, tmp_path: Path) -> None:
        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        shell = Shell[None](tools=['shell'])
        output = await call_tool(
            [shell],
            'shell',
            {'command': 'exec sleep 300', 'mode': 'background'},
            workspace=LocalWorkspaceBackend(tmp_path),
        )
        pid = int(output.split('PID: ')[1].split()[0])
        stop = output.split('use `')[1].split('`')[0]
        status = Path(output.split('Status: ')[1].splitlines()[0])
        try:
            assert json.loads(status.read_text(encoding='utf-8')) == {'pid': pid, 'exit_code': None}
            assert '"exit_code": null' in output
            # The wait for the published exit code runs in the workspace shell, with the model's own tool.
            wait = f'{stop} && while grep -q \'"exit_code": null\' {shlex.quote(str(status))}; do sleep 0.05; done'
            result = await call_tool(
                [Shell[None](tools=['run_command'], denied_commands=[])],
                'run_command',
                {'command': wait, 'timeout_seconds': 10},
                workspace=LocalWorkspaceBackend(tmp_path),
            )
            assert 'exit code' not in result and 'timed out' not in result
            assert json.loads(status.read_text(encoding='utf-8')) == {'pid': pid, 'exit_code': 143}
        finally:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
            shutil.rmtree(status.parent, ignore_errors=True)


class _RaisingWorkspace(LocalWorkspaceBackend):
    """A local backend whose every command raises the configured workspace error."""

    def __init__(self, working_dir: str | Path, error: WorkspaceError) -> None:
        super().__init__(working_dir)
        self.error = error

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        raise self.error


class _SlowReads(LocalWorkspaceBackend):
    async def read_bytes(self, path: str) -> bytes:
        raise WorkspaceTimeoutError('read timed out after 5 seconds')


class TestWorkspaceFailures:
    """Deliberate workspace failures reach the model as failed calls; a vanished workspace ends the run."""

    @pytest.mark.parametrize(
        ('error', 'message'),
        [
            (WorkspaceTimeoutError('command timed out after 30 seconds'), 'command timed out after 30 seconds'),
            (WorkspaceTimeoutError(''), 'The workspace operation timed out.'),
            (WorkspaceError('backend refused'), 'backend refused'),
            (WorkspaceError(), 'The workspace operation failed (WorkspaceError).'),
        ],
    )
    async def test_failed_call(self, shell_dir: Path, error: WorkspaceError, message: str) -> None:
        ts = _shell_toolset(shell_dir)
        ctx = _run_context(Workspace(_RaisingWorkspace(shell_dir, error)))
        tools = await ts.get_tools(ctx)
        with pytest.raises(ToolFailed) as failed:
            await ts.call_tool('start_command', {'command': 'true'}, ctx, tools['start_command'])
        assert failed.value.message == message

    async def test_timed_out_handle_read_is_not_an_unknown_id(self, shell_dir: Path) -> None:
        ts = _shell_toolset(shell_dir)
        with pytest.raises(ToolFailed, match='timed out'):
            await ts.check_command(_run_context(Workspace(_SlowReads(shell_dir))), 'c' * 32)

    async def test_unavailable_workspace_ends_the_run(self, shell_dir: Path) -> None:
        ts = _shell_toolset(shell_dir)
        ctx = _run_context(Workspace(_RaisingWorkspace(shell_dir, WorkspaceUnavailableError('sandbox expired'))))
        tools = await ts.get_tools(ctx)
        with pytest.raises(WorkspaceUnavailableError, match='sandbox expired'):
            await ts.call_tool('run_command', {'command': 'true'}, ctx, tools['run_command'])
