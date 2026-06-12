"""Tests for the Shell capability and ShellToolset."""

from __future__ import annotations

import os
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from unittest.mock import MagicMock, patch

import anyio
import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS, Shell
from pydantic_ai_harness.shell._toolset import (
    ShellToolset,
    _is_interactive_command,
)


def _env_toolset(
    shell_dir: Path,
    *,
    env: Mapping[str, str] | None = None,
    denied_env_patterns: Sequence[str] = (),
) -> ShellToolset[None]:
    """Build a ShellToolset wired for env-control tests, with safe defaults."""
    return ShellToolset(
        cwd=shell_dir,
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


def _read_env_var(name: str) -> str:
    """Shell command that prints an env var's value, or ABSENT if unset."""
    return f'{sys.executable} -c "import os; print(os.environ.get({name!r}, \'ABSENT\'))"'


def _run_context() -> RunContext[None]:
    """Minimal `RunContext` for invoking `for_run` directly in tests."""
    return RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )


def _parse_command_id(result: str) -> str:
    assert 'ID: ' in result, f'Expected "ID: " in result: {result!r}'
    return result.split('ID: ')[1].strip()


class TestIsInteractiveCommand:
    def test_vi(self) -> None:
        assert _is_interactive_command('vi file.txt') is True

    def test_vim(self) -> None:
        assert _is_interactive_command('vim file.txt') is True

    def test_nano(self) -> None:
        assert _is_interactive_command('nano file.txt') is True

    def test_less(self) -> None:
        assert _is_interactive_command('less file.txt') is True

    def test_top(self) -> None:
        assert _is_interactive_command('top') is True

    def test_sudo(self) -> None:
        assert _is_interactive_command('sudo rm -rf /') is True

    def test_ssh(self) -> None:
        assert _is_interactive_command('ssh host') is True

    def test_regular_command(self) -> None:
        assert _is_interactive_command('ls -la') is False

    def test_echo(self) -> None:
        assert _is_interactive_command('echo hello') is False

    def test_grep(self) -> None:
        assert _is_interactive_command('grep pattern file') is False

    def test_emacs(self) -> None:
        assert _is_interactive_command('emacs file.txt') is True

    def test_man(self) -> None:
        assert _is_interactive_command('man ls') is True

    def test_htop(self) -> None:
        assert _is_interactive_command('htop') is True

    def test_telnet(self) -> None:
        assert _is_interactive_command('telnet localhost 80') is True

    def test_ftp(self) -> None:
        assert _is_interactive_command('ftp host') is True

    def test_passwd(self) -> None:
        assert _is_interactive_command('passwd') is True

    def test_more(self) -> None:
        assert _is_interactive_command('more file.txt') is True

    def test_not_prefix_match(self) -> None:
        assert _is_interactive_command('view file.txt') is False
        assert _is_interactive_command('vishnu') is False

    def test_leading_spaces(self) -> None:
        assert _is_interactive_command('  vi file.txt') is True
        assert _is_interactive_command('  sudo rm') is True


@pytest.fixture
def shell_dir(tmp_path: Path) -> Path:
    (tmp_path / 'test.txt').write_text('hello\n')
    (tmp_path / 'subdir').mkdir()
    (tmp_path / 'subdir' / 'nested.txt').write_text('nested\n')
    return tmp_path


@pytest.fixture
def toolset(shell_dir: Path) -> ShellToolset[None]:
    return ShellToolset(
        cwd=shell_dir,
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
        cwd=shell_dir,
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
            toolset._check_command('rm -rf /')

    async def test_allowed_command_permitted(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=['echo', 'cat'],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        ts._check_command('echo hello')
        ts._check_command('cat file.txt')

    async def test_allowed_blocks_non_matching(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=['echo'],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        with pytest.raises(PermissionError, match='not in the allowed list'):
            ts._check_command('cat file.txt')

    async def test_both_allow_and_deny_raises(self, shell_dir: Path) -> None:
        with pytest.raises(ValueError, match='Specify allowed_commands or denied_commands'):
            ShellToolset(
                cwd=shell_dir,
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
            toolset._check_command('vim file.txt')

    async def test_interactive_allowed_when_enabled(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=True,
        )
        ts._check_command('vim file.txt')

    async def test_denied_operator_blocked(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=['>', '>>'],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        with pytest.raises(PermissionError, match="'>' is not allowed"):
            ts._check_command('echo hello > file.txt')

    async def test_denied_operator_passes_when_not_present(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=['>', '>>'],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        ts._check_command('echo hello')

    async def test_unparseable_command_allowed(self, toolset: ShellToolset[None]) -> None:
        toolset._check_command("echo 'unterminated")

    async def test_empty_command_allowed(self, toolset: ShellToolset[None]) -> None:
        toolset._check_command('')

    async def test_denied_operator_substring_match(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=['>>'],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        with pytest.raises(PermissionError, match="'>>' is not allowed"):
            ts._check_command('echo hello >> file.txt')

    async def test_shlex_error_returns_early(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=['rm'],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        ts._check_command("echo 'unterminated")

    async def test_empty_tokens(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=['echo'],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        ts._check_command('')

    def test_first_denied_operator_match(self, toolset: ShellToolset[None]) -> None:
        ts = ShellToolset(
            cwd=Path('/tmp'),
            allowed_commands=[],
            denied_commands=[],
            denied_operators=['|', '>'],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        assert ts._first_denied_operator('echo hi | cat') == '|'

    def test_first_denied_operator_no_match(self, toolset: ShellToolset[None]) -> None:
        ts = ShellToolset(
            cwd=Path('/tmp'),
            allowed_commands=[],
            denied_commands=[],
            denied_operators=['|', '>'],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        assert ts._first_denied_operator('echo hello') is None

    def test_first_denied_operator_empty_list(self, toolset: ShellToolset[None]) -> None:
        assert toolset._first_denied_operator('echo hi | cat') is None


class TestTruncation:
    def test_within_limit(self, toolset: ShellToolset[None]) -> None:
        assert toolset._truncate('short') == 'short'

    def test_at_limit(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=10,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = ts._truncate('x' * 10)
        assert result == 'x' * 10

    def test_over_limit(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=10,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = ts._truncate('x' * 20)
        assert result.endswith('x' * 10)
        assert 'truncated, showing last 10 chars' in result

    def test_exactly_at_limit_not_truncated(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=10,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = ts._truncate('x' * 10)
        assert result == 'x' * 10
        assert 'truncated' not in result

    def test_one_over_limit_truncated(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=10,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = ts._truncate('x' * 11)
        assert result.endswith('x' * 10)
        assert 'truncated, showing last 10 chars' in result

    def test_keeps_tail_not_head(self, shell_dir: Path) -> None:
        """The tail (where errors and the [stderr] section land) is preserved."""
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=20,
            persist_cwd=False,
            allow_interactive=False,
        )
        text = 'HEAD' + 'x' * 100 + 'TAIL_ERROR'
        result = ts._truncate(text)
        assert result.endswith('TAIL_ERROR')
        assert 'HEAD' not in result
        assert 'truncated' in result

    def test_truncation_marker_wording(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=10,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = ts._truncate('x' * 20)
        assert 'output truncated, showing last 10 chars' in result


class TestCwdCapture:
    """The persistent-cwd mechanism records `pwd` out-of-band via a private temp
    file, so command output can never spoof the tracked directory."""

    def test_capture_disabled_returns_command_unchanged(self, toolset: ShellToolset[None]) -> None:
        wrapped, cwd_file = toolset._build_cwd_capture('echo hi')
        assert wrapped == 'echo hi'
        assert cwd_file is None

    def test_capture_records_pwd_out_of_band(self, persist_toolset: ShellToolset[None]) -> None:
        wrapped, cwd_file = persist_toolset._build_cwd_capture('echo hi')
        assert cwd_file is not None
        try:
            # pwd is redirected to the private temp file, never echoed to stdout
            assert f'pwd > {shlex.quote(str(cwd_file))}' in wrapped
            assert wrapped.startswith('echo hi')
        finally:
            cwd_file.unlink(missing_ok=True)

    def test_apply_valid_dir_updates_cwd(
        self, persist_toolset: ShellToolset[None], shell_dir: Path, tmp_path: Path
    ) -> None:
        capture = tmp_path / 'cwd'
        capture.write_text(f'{shell_dir / "subdir"}\n')
        persist_toolset._apply_captured_cwd(capture)
        assert persist_toolset._cwd == shell_dir / 'subdir'

    def test_apply_empty_file_keeps_cwd(self, persist_toolset: ShellToolset[None], tmp_path: Path) -> None:
        original = persist_toolset._cwd
        capture = tmp_path / 'cwd'
        capture.write_text('')
        persist_toolset._apply_captured_cwd(capture)
        assert persist_toolset._cwd == original

    def test_apply_non_dir_keeps_cwd(self, persist_toolset: ShellToolset[None], tmp_path: Path) -> None:
        original = persist_toolset._cwd
        capture = tmp_path / 'cwd'
        capture.write_text(str(tmp_path / 'does_not_exist'))
        persist_toolset._apply_captured_cwd(capture)
        assert persist_toolset._cwd == original


class TestForRunIsolation:
    """B3: `get_toolset` builds one shared instance at agent construction, so
    `for_run` must hand each run a fresh copy -- otherwise concurrent runs share
    `_cwd`/`_background` and corrupt each other."""

    async def test_for_run_returns_fresh_instance(self, persist_toolset: ShellToolset[None]) -> None:
        run1 = await persist_toolset.for_run(_run_context())
        run2 = await persist_toolset.for_run(_run_context())
        assert run1 is not persist_toolset
        assert run2 is not run1

    async def test_persist_cwd_isolated_across_runs(self, persist_toolset: ShellToolset[None], shell_dir: Path) -> None:
        run1 = await persist_toolset.for_run(_run_context())
        assert isinstance(run1, ShellToolset)
        await run1.run_command('cd subdir')
        assert run1._cwd == shell_dir / 'subdir'
        # A second run must start back at the configured root, not inherit run1's cd.
        run2 = await persist_toolset.for_run(_run_context())
        assert isinstance(run2, ShellToolset)
        assert run2._cwd == shell_dir


class TestPersistCwdHardening:
    """B4: regression tests for the old stdout-sentinel footguns -- a command's
    output spoofing the cwd, and `;` silently disabling tracking."""

    async def test_cd_persists_even_with_semicolon(self, persist_toolset: ShellToolset[None]) -> None:
        # The old mechanism skipped tracking whenever ';' appeared, silently
        # dropping a real `cd`. The out-of-band capture records it regardless.
        await persist_toolset.run_command('cd subdir ; true')
        result = await persist_toolset.run_command('pwd')
        assert 'subdir' in result

    async def test_output_cannot_spoof_cwd(self, persist_toolset: ShellToolset[None], shell_dir: Path) -> None:
        # The old mechanism parsed cwd from stdout, so a command printing the
        # sentinel string could redirect the tracked cwd with no real cd.
        spoof = f'true ; echo __HARNESS_PWD__{shell_dir / "subdir"}'
        await persist_toolset.run_command(spoof)
        assert persist_toolset._cwd == shell_dir


class TestRunCommand:
    async def test_basic_echo(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.run_command('echo hello')
        assert '[stdout]' in result
        assert 'hello' in result

    async def test_stderr_output(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.run_command('echo error >&2')
        assert '[stderr]' in result
        assert 'error' in result

    async def test_mixed_output(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.run_command('echo out && echo err >&2')
        assert '[stdout]' in result
        assert '[stderr]' in result

    async def test_exit_code_reported(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.run_command('exit 42')
        assert '[exit code: 42]' in result

    async def test_exit_code_zero_not_shown(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.run_command('echo ok')
        assert 'exit code' not in result

    async def test_no_output(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.run_command('true')
        assert result == '(no output)'

    async def test_output_truncation(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.run_command(f'{sys.executable} -c "print(\'x\' * 200)"')
        assert 'truncated, showing last 50 chars' in result

    async def test_persist_cwd(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=True,
            allow_interactive=False,
        )
        await ts.run_command('cd subdir')
        result = await ts.run_command('pwd')
        assert 'subdir' in result

    async def test_persist_cwd_only_on_success(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=True,
            allow_interactive=False,
        )
        original = ts._cwd
        await ts.run_command('cd nonexistent_dir_xyz && false')
        assert ts._cwd == original

    async def test_denied_command_in_run(self, toolset: ShellToolset[None]) -> None:
        # B2: a denied command is model-correctable, so it surfaces as ModelRetry
        # (which pyai feeds back to the model) rather than aborting the run.
        with pytest.raises(ModelRetry, match="'rm' is denied"):
            await toolset.run_command('rm -rf /')

    async def test_cwd_used(self, toolset: ShellToolset[None], shell_dir: Path) -> None:
        result = await toolset.run_command('cat test.txt')
        assert 'hello' in result

    async def test_multiline_output(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.run_command(f'{sys.executable} -c "print(\'a\\nb\\nc\\n\')"')
        assert '[stdout]' in result

    async def test_timeout_reports_value(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=0.5,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.run_command('sleep 10')
        assert 'timed out after 0.5s' in result

    async def test_custom_timeout_overrides_default(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=30.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.run_command('sleep 10', timeout_seconds=0.5)
        assert 'timed out after 0.5s' in result

    async def test_persist_cwd_disabled_no_update(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        original = ts._cwd
        await ts.run_command('cd subdir')
        assert ts._cwd == original

    async def test_nonzero_exit_shows_code(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.run_command('exit 1')
        assert '[exit code: 1]' in result

    async def test_stdout_stderr_separated_by_newline(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.run_command('echo out && echo err >&2')
        assert '[stdout]\nout\n\n[stderr]\nerr' in result

    async def test_non_ascii_stdout(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.run_command(
            f'{sys.executable} -c "import sys; sys.stdout.buffer.write(b\'hello \\xff\\xfe world\\n\')"'
        )
        assert 'hello' in result

    async def test_non_ascii_stderr(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.run_command(
            f'{sys.executable} -c "import sys; sys.stderr.buffer.write(b\'err \\xff\\xfe msg\\n\')"'
        )
        assert 'err' in result

    async def test_stdout_chunk_join(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.run_command(f"{sys.executable} -c \"print('A' * 100 + 'B' * 100)\"")
        assert 'A' * 100 + 'B' * 100 in result

    async def test_exit_code_fallback_to_zero(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=True,
            allow_interactive=False,
        )
        result = await ts.run_command('echo ok')
        assert 'exit code' not in result

    async def test_error_message_content(self, shell_dir: Path) -> None:
        with pytest.raises(ValueError, match='^Specify allowed_commands or denied_commands, not both\\.$'):
            ShellToolset(
                cwd=shell_dir,
                allowed_commands=['echo'],
                denied_commands=['rm'],
                denied_operators=[],
                default_timeout=10.0,
                max_output_chars=50_000,
                persist_cwd=False,
                allow_interactive=False,
            )

    async def test_stdout_chunks_joined_cleanly(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=30.0,
            max_output_chars=500_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.run_command("printf '%05000d\\n' $(seq 1 100)")
        assert 'XXXX' not in result

    async def test_stderr_chunks_joined_cleanly(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=30.0,
            max_output_chars=500_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.run_command("printf '%0500d\\n' $(seq 1 100) >&2")
        assert 'XXXX' not in result

    async def test_persist_cwd_updates_after_cd(self, shell_dir: Path) -> None:
        """CWD should update to the actual directory after a successful cd."""
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=True,
            allow_interactive=False,
        )
        await ts.run_command('cd subdir')
        assert ts._cwd == (shell_dir / 'subdir')

    async def test_persist_cwd_not_updated_on_failure(self, shell_dir: Path) -> None:
        """CWD should not update if command fails (exit code non-zero)."""
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=True,
            allow_interactive=False,
        )
        original = ts._cwd
        await ts.run_command('false')
        assert ts._cwd == original


class TestProcessGroupKill:
    async def test_timeout_kills_subprocess_tree(self, shell_dir: Path) -> None:
        """On timeout, the entire process group should be killed."""
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=0.5,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.run_command('bash -c "sleep 100 & sleep 100"')
        assert 'timed out' in result

    async def test_timeout_with_output_before_timeout(self, shell_dir: Path) -> None:
        """Output produced before timeout should still result in timeout message."""
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=0.5,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.run_command('echo before_timeout && sleep 100')
        assert 'timed out' in result

    async def test_start_new_session_used(self, shell_dir: Path) -> None:
        """Verify the child is in a different process group from the parent."""
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        parent_pgrp = os.getpgrp()
        result = await ts.run_command(f'{sys.executable} -c "import os; print(os.getpgrp() != {parent_pgrp})"')
        assert 'True' in result


class TestBackgroundCommands:
    async def test_start_command_returns_id(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.start_command('sleep 100')
        assert 'ID:' in result
        assert 'Started background command' in result
        command_id = _parse_command_id(result)
        await ts.stop_command(command_id)

    async def test_check_unknown_id(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.check_command('nonexistent_id')
        assert 'unknown command ID' in result

    async def test_stop_unknown_id(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.stop_command('nonexistent_id')
        assert 'unknown command ID' in result

    async def test_start_and_stop(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command('echo hello_bg')
        command_id = _parse_command_id(start_result)

        await anyio.sleep(0.5)

        stop_result = await ts.stop_command(command_id)
        assert 'stopped' in stop_result
        assert 'hello_bg' in stop_result

    async def test_start_and_check_running(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command('sleep 100')
        command_id = _parse_command_id(start_result)

        check_result = await ts.check_command(command_id)
        assert 'running' in check_result

        await ts.stop_command(command_id)

    async def test_start_and_check_finished(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command('echo done_quick')
        command_id = _parse_command_id(start_result)

        await anyio.sleep(0.5)

        check_result = await ts.check_command(command_id)
        assert 'finished' in check_result
        assert 'done_quick' in check_result

        await ts.stop_command(command_id)

    async def test_start_denied_command_raises(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=['rm'],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        with pytest.raises(ModelRetry, match="'rm' is denied"):
            await ts.start_command('rm -rf /')

    async def test_stop_captures_stderr(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command('echo err_bg >&2')
        command_id = _parse_command_id(start_result)

        await anyio.sleep(0.5)

        stop_result = await ts.stop_command(command_id)
        assert 'err_bg' in stop_result

    async def test_stop_no_output(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command('true')
        command_id = _parse_command_id(start_result)

        await anyio.sleep(0.5)

        stop_result = await ts.stop_command(command_id)
        assert '(no output)' in stop_result

    async def test_check_no_output_yet(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command('sleep 100')
        command_id = _parse_command_id(start_result)

        check_result = await ts.check_command(command_id)
        assert 'no output yet' in check_result

        await ts.stop_command(command_id)

    async def test_check_command_captures_stderr(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command('echo err_check >&2')
        command_id = _parse_command_id(start_result)

        await anyio.sleep(0.5)

        check_result = await ts.check_command(command_id)
        assert '[stderr]' in check_result
        assert 'err_check' in check_result

        await ts.stop_command(command_id)

    async def test_start_command_uses_cwd(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command('pwd')
        command_id = _parse_command_id(start_result)

        await anyio.sleep(0.5)

        stop_result = await ts.stop_command(command_id)
        assert str(shell_dir) in stop_result

    async def test_stop_removes_from_registry(self, shell_dir: Path) -> None:
        """After stop, the command_id should no longer be known."""
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command('true')
        command_id = _parse_command_id(start_result)

        await anyio.sleep(0.5)

        await ts.stop_command(command_id)

        # Should now be unknown
        check_result = await ts.check_command(command_id)
        assert 'unknown command ID' in check_result

    async def test_start_command_cleans_temp_files_on_failure(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
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
                await ts.start_command('echo hi')
        assert not ts._background

    async def test_aexit_terminates_background_processes(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.start_command('sleep 300')
        command_id = _parse_command_id(result)
        bg = ts._background[command_id]
        stdout_path = Path(bg.stdout_path)
        stderr_path = Path(bg.stderr_path)
        assert stdout_path.exists()
        assert stderr_path.exists()

        await ts.__aexit__(None, None, None)

        assert not ts._background
        assert not stdout_path.exists()
        assert not stderr_path.exists()

    async def test_aexit_noop_when_no_background(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        await ts.__aexit__(None, None, None)
        assert not ts._background

    async def test_aexit_cleans_already_finished_process(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.start_command('echo done')
        command_id = _parse_command_id(result)
        await anyio.sleep(0.5)
        # Mark as finished via check_command
        await ts.check_command(command_id)
        bg = ts._background[command_id]
        assert bg.finished

        await ts.__aexit__(None, None, None)
        assert not ts._background


class TestEdgeCases:
    async def test_toolset_tool_names(self, toolset: ShellToolset[None]) -> None:
        tool_names = list(toolset.tools.keys())
        assert 'run_command' in tool_names
        assert 'start_command' in tool_names
        assert 'check_command' in tool_names
        assert 'stop_command' in tool_names

    async def test_run_command_uses_actual_cwd(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.run_command('pwd')
        assert str(shell_dir) in result

    async def test_persist_cwd_requires_all_three_conditions(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=True,
            allow_interactive=False,
        )
        # Successful echo -- sentinel shows same dir, cwd should remain valid
        await ts.run_command('echo hi')
        assert ts._cwd.is_dir()


class TestShellCapability:
    def test_default_construction(self) -> None:
        shell = Shell()
        assert shell.cwd == '.'
        assert shell.default_timeout == 30.0
        assert 'rm' in shell.denied_commands

    def test_custom_construction(self) -> None:
        shell = Shell(
            cwd='/tmp',
            allowed_commands=['echo', 'cat'],
            denied_commands=[],
            default_timeout=60.0,
        )
        assert shell.default_timeout == 60.0

    def test_get_toolset_returns_toolset(self, tmp_path: Path) -> None:
        shell = Shell(cwd=tmp_path)
        toolset = shell.get_toolset()
        assert isinstance(toolset, ShellToolset)

    def test_default_denied_commands(self) -> None:
        shell = Shell()
        assert 'rm' in shell.denied_commands
        assert 'dd' in shell.denied_commands
        assert 'shutdown' in shell.denied_commands

    @pytest.mark.anyio(backends=['asyncio'])
    async def test_agent_integration(self, tmp_path: Path) -> None:
        import sniffio

        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        model = TestModel(custom_output_text='done', call_tools=[])
        agent: Agent[None, str] = Agent(model, capabilities=[Shell(cwd=tmp_path)])
        result = await agent.run('run echo hello')
        assert result.output == 'done'


class TestKillProcessGroupEdgeCases:
    async def test_sigterm_raises_process_lookup_error(self, tmp_path: Path) -> None:
        """When SIGTERM raises ProcessLookupError, method returns without SIGKILL."""
        ts = ShellToolset(
            cwd=tmp_path,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=5.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        proc = MagicMock()
        proc.pid = 99999
        with patch('os.killpg', side_effect=ProcessLookupError):
            await ts._kill_process_group(proc)
        # No exception raised, method returned early

    async def test_sigkill_escalation(self, tmp_path: Path) -> None:
        """When process doesn't exit within grace period, SIGKILL is sent."""
        ts = ShellToolset(
            cwd=tmp_path,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=5.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        proc = MagicMock()
        proc.pid = 99999

        # Make proc.wait() never complete (simulates process ignoring SIGTERM)
        async def never_return() -> None:
            await anyio.sleep(999)

        proc.wait = never_return

        import signal

        kill_calls: list[tuple[int, int]] = []

        def fake_killpg(pgid: int, sig: int) -> None:
            kill_calls.append((pgid, sig))

        with (
            patch('os.killpg', side_effect=fake_killpg),
            patch('os.getpgid', return_value=12345),
            patch('pydantic_ai_harness.shell._toolset._KILL_GRACE_PERIOD', 0.01),
        ):
            await ts._kill_process_group(proc)

        assert len(kill_calls) == 2
        assert kill_calls[0][1] == signal.SIGTERM
        assert kill_calls[1][1] == signal.SIGKILL

    async def test_sigkill_raises_process_lookup_error(self, tmp_path: Path) -> None:
        """When SIGKILL raises ProcessLookupError (process exited between SIGTERM and SIGKILL)."""
        ts = ShellToolset(
            cwd=tmp_path,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=5.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        proc = MagicMock()
        proc.pid = 99999

        async def never_return() -> None:
            await anyio.sleep(999)

        proc.wait = never_return

        import signal

        call_count = 0

        def fake_killpg(pgid: int, sig: int) -> None:
            nonlocal call_count
            call_count += 1
            if sig == signal.SIGKILL:
                raise ProcessLookupError

        with (
            patch('os.killpg', side_effect=fake_killpg),
            patch('os.getpgid', return_value=12345),
            patch('pydantic_ai_harness.shell._toolset._KILL_GRACE_PERIOD', 0.01),
        ):
            await ts._kill_process_group(proc)

        assert call_count == 2


class TestDrainWithTimeoutEdgeCases:
    async def test_stdout_closed_resource_error(self, tmp_path: Path) -> None:
        """ClosedResourceError on stdout is caught silently after yielding data."""
        ts = ShellToolset(
            cwd=tmp_path,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=5.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        proc = MagicMock()

        # Yield one chunk then raise ClosedResourceError
        class FailingStream:
            def __init__(self) -> None:
                self._yielded = False

            def __aiter__(self) -> FailingStream:
                return self

            async def __anext__(self) -> bytes:
                if not self._yielded:
                    self._yielded = True
                    return b'partial'
                raise anyio.ClosedResourceError

        proc.stdout = FailingStream()
        proc.stderr = None

        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        await ts._drain_with_timeout(stdout_chunks, stderr_chunks, proc)
        assert stdout_chunks == [b'partial']

    async def test_stderr_broken_resource_error(self, tmp_path: Path) -> None:
        """BrokenResourceError on stderr is caught silently after yielding data."""
        ts = ShellToolset(
            cwd=tmp_path,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=5.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        proc = MagicMock()
        proc.stdout = None

        class FailingStream:
            def __init__(self) -> None:
                self._yielded = False

            def __aiter__(self) -> FailingStream:
                return self

            async def __anext__(self) -> bytes:
                if not self._yielded:
                    self._yielded = True
                    return b'partial'
                raise anyio.BrokenResourceError

        proc.stderr = FailingStream()

        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        await ts._drain_with_timeout(stdout_chunks, stderr_chunks, proc)
        assert stderr_chunks == [b'partial']


class TestReadBgOutputEdgeCases:
    def test_stdout_oserror(self, tmp_path: Path) -> None:
        """OSError reading stdout file returns empty string."""
        ts = ShellToolset(
            cwd=tmp_path,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=5.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        bg = MagicMock()
        bg.stdout_path = '/nonexistent/path/stdout'
        bg.stderr_path = '/nonexistent/path/stderr'

        stdout, stderr = ts._read_bg_output(bg)
        assert stdout == ''
        assert stderr == ''

    def test_stderr_oserror_only(self, tmp_path: Path) -> None:
        """OSError reading stderr file only, stdout succeeds."""
        ts = ShellToolset(
            cwd=tmp_path,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=5.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        # Create a valid stdout file but invalid stderr path
        stdout_file = tmp_path / 'stdout.txt'
        stdout_file.write_text('hello')

        bg = MagicMock()
        bg.stdout_path = str(stdout_file)
        bg.stderr_path = '/nonexistent/path/stderr'

        stdout, stderr = ts._read_bg_output(bg)
        assert stdout == 'hello'
        assert stderr == ''


class TestCleanupBgFilesEdgeCases:
    def test_unlink_oserror(self, tmp_path: Path) -> None:
        """OSError on unlink is caught silently."""
        ts = ShellToolset(
            cwd=tmp_path,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=5.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        bg = MagicMock()
        bg.stdout_path = '/nonexistent/path/stdout'
        bg.stderr_path = '/nonexistent/path/stderr'

        # Should not raise
        ts._cleanup_bg_files(bg)


class TestStopCommandAlreadyFinished:
    async def test_stop_already_finished_process(self, shell_dir: Path) -> None:
        """stop_command on an already-finished process skips kill."""
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        # Start a command that finishes immediately
        start_result = await ts.start_command('echo done')
        command_id = _parse_command_id(start_result)

        # Wait for the process to finish
        await anyio.sleep(0.5)

        # Manually mark as finished with exit_code = None (simulates edge case
        # where finished is True but exit_code was never captured)
        bg = ts._background[command_id]
        bg.finished = True
        bg.exit_code = None

        # stop_command should skip the kill branch and handle None exit_code
        result = await ts.stop_command(command_id)
        assert '[stopped:' in result
        assert '[exit code:' not in result


class TestResolveEnv:
    """Unit coverage for the env-resolution branches."""

    def test_inherits_when_unconfigured(self, shell_dir: Path) -> None:
        # Neither env nor patterns set -> None, so the subprocess inherits.
        assert _env_toolset(shell_dir)._resolve_env() is None

    def test_explicit_env_replaces(self, shell_dir: Path) -> None:
        resolved = _env_toolset(shell_dir, env={'FOO': 'bar'})._resolve_env()
        assert resolved == {'FOO': 'bar'}

    def test_explicit_empty_env_is_not_inheritance(self, shell_dir: Path) -> None:
        # {} is a hard boundary (no vars), distinct from None (inherit all).
        assert _env_toolset(shell_dir, env={})._resolve_env() == {}

    def test_patterns_strip_from_inherited(self, shell_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('OPENAI_API_KEY', 'secret')
        monkeypatch.setenv('SAFE_VAR', 'keep')
        resolved = _env_toolset(shell_dir, denied_env_patterns=['OPENAI_*'])._resolve_env()
        assert resolved is not None
        assert 'OPENAI_API_KEY' not in resolved
        assert resolved.get('SAFE_VAR') == 'keep'

    def test_patterns_strip_from_explicit_env(self, shell_dir: Path) -> None:
        resolved = _env_toolset(
            shell_dir,
            env={'OPENAI_API_KEY': 'secret', 'PATH': '/usr/bin'},
            denied_env_patterns=['OPENAI_*'],
        )._resolve_env()
        assert resolved == {'PATH': '/usr/bin'}

    def test_patterns_no_match_keeps_base(self, shell_dir: Path) -> None:
        resolved = _env_toolset(
            shell_dir,
            env={'FOO': 'bar'},
            denied_env_patterns=['OPENAI_*'],
        )._resolve_env()
        assert resolved == {'FOO': 'bar'}

    def test_pattern_match_is_case_sensitive(self, shell_dir: Path) -> None:
        # Env var names are case-sensitive on POSIX; lowercase must not match.
        resolved = _env_toolset(
            shell_dir,
            env={'openai_api_key': 'secret'},
            denied_env_patterns=['OPENAI_*'],
        )._resolve_env()
        assert resolved == {'openai_api_key': 'secret'}


class TestEnvControlExecution:
    """End-to-end: the resolved env actually reaches spawned subprocesses."""

    async def test_explicit_env_seen_by_command(self, shell_dir: Path) -> None:
        ts = _env_toolset(shell_dir, env={'MY_TOKEN': 'present', 'PATH': os.environ['PATH']})
        result = await ts.run_command(_read_env_var('MY_TOKEN'))
        assert 'present' in result

    async def test_explicit_env_hides_inherited_secret(self, shell_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('OPENROUTER_API_KEY', 'leak-me')
        ts = _env_toolset(shell_dir, env={'PATH': os.environ['PATH']})
        result = await ts.run_command(_read_env_var('OPENROUTER_API_KEY'))
        assert 'ABSENT' in result
        assert 'leak-me' not in result

    async def test_denied_pattern_strips_inherited_secret(
        self, shell_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'leak-me')
        ts = _env_toolset(shell_dir, denied_env_patterns=['ANTHROPIC_*'])
        result = await ts.run_command(_read_env_var('ANTHROPIC_API_KEY'))
        assert 'ABSENT' in result
        assert 'leak-me' not in result

    async def test_unstripped_inherited_var_still_visible(
        self, shell_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A var not matched by any pattern is still inherited as before.
        monkeypatch.setenv('HARNESS_KEEP', 'visible')
        ts = _env_toolset(shell_dir, denied_env_patterns=['ANTHROPIC_*'])
        result = await ts.run_command(_read_env_var('HARNESS_KEEP'))
        assert 'visible' in result

    async def test_default_inherits_parent_env(self, shell_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Backward compatible: with no env control, inherited vars pass through.
        monkeypatch.setenv('HARNESS_INHERITED', 'yes')
        ts = _env_toolset(shell_dir)
        result = await ts.run_command(_read_env_var('HARNESS_INHERITED'))
        assert 'yes' in result

    async def test_env_and_patterns_compose_at_spawn(self, shell_dir: Path) -> None:
        # Both set: a pattern strips a key from the explicit env, the rest survives.
        ts = _env_toolset(
            shell_dir,
            env={'SECRET_KEY': 'leak-me', 'KEEP_VAR': 'kept', 'PATH': os.environ['PATH']},
            denied_env_patterns=['SECRET_*'],
        )
        stripped = await ts.run_command(_read_env_var('SECRET_KEY'))
        assert 'ABSENT' in stripped
        assert 'leak-me' not in stripped
        survived = await ts.run_command(_read_env_var('KEEP_VAR'))
        assert 'kept' in survived

    async def test_background_command_honors_env(self, shell_dir: Path) -> None:
        ts = _env_toolset(shell_dir, env={'BG_TOKEN': 'bg-present', 'PATH': os.environ['PATH']})
        start_result = await ts.start_command(_read_env_var('BG_TOKEN'))
        command_id = _parse_command_id(start_result)
        await anyio.sleep(0.5)
        stop_result = await ts.stop_command(command_id)
        assert 'bg-present' in stop_result


class TestEnvControlPropagation:
    """The capability and `for_run` carry env control through unchanged."""

    async def test_for_run_propagates_env(self, shell_dir: Path) -> None:
        ts = _env_toolset(shell_dir, env={'FOO': 'bar'}, denied_env_patterns=['OPENAI_*'])
        run_ts = await ts.for_run(_run_context())
        assert isinstance(run_ts, ShellToolset)
        assert run_ts._resolve_env() == {'FOO': 'bar'}

    def test_capability_defaults_inherit(self) -> None:
        shell = Shell()
        assert shell.env is None
        assert list(shell.denied_env_patterns) == []

    def test_capability_passes_env_to_toolset(self, tmp_path: Path) -> None:
        shell = Shell(
            cwd=tmp_path,
            env={'FOO': 'bar'},
            denied_env_patterns=['OPENAI_*'],
        )
        toolset = shell.get_toolset()
        assert isinstance(toolset, ShellToolset)
        assert toolset._resolve_env() == {'FOO': 'bar'}

    def test_llm_pattern_constant_strips_provider_keys(self, tmp_path: Path) -> None:
        shell = Shell(cwd=tmp_path, denied_env_patterns=list(LLM_API_KEY_ENV_PATTERNS))
        toolset = shell.get_toolset()
        assert isinstance(toolset, ShellToolset)
        resolved = toolset._resolve_env()
        assert resolved is not None
        # None of the provider-credential prefixes survive.
        leaked = {
            name
            for name in resolved
            if name.startswith(('ANTHROPIC_', 'OPENAI_', 'OPENROUTER_', 'GEMINI_', 'GOOGLE_', 'GATEWAY_'))
            or name == 'PYDANTIC_AI_GATEWAY_API_KEY'
        }
        assert leaked == set()
