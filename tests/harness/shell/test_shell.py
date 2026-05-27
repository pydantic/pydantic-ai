"""Tests for the Shell capability and ShellToolset."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.shell import Shell
from pydantic_ai_harness.shell._toolset import (
    _PWD_SENTINEL,
    ShellToolset,
    _is_interactive_command,
)

# ============================================================================
# ============================================================================


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


# ============================================================================
# ============================================================================


@pytest.fixture
def shell_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for shell tests."""
    (tmp_path / 'test.txt').write_text('hello\n')
    (tmp_path / 'subdir').mkdir()
    (tmp_path / 'subdir' / 'nested.txt').write_text('nested\n')
    return tmp_path


@pytest.fixture
def toolset(shell_dir: Path) -> ShellToolset:
    """Create a basic ShellToolset."""
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


class TestCommandValidation:
    async def test_denied_command_blocked(self, toolset: ShellToolset) -> None:
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

    async def test_interactive_blocked_by_default(self, toolset: ShellToolset) -> None:
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

    async def test_unparseable_command_allowed(self, toolset: ShellToolset) -> None:
        toolset._check_command("echo 'unterminated")

    async def test_empty_command_allowed(self, toolset: ShellToolset) -> None:
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


class TestTruncation:
    def test_within_limit(self, toolset: ShellToolset) -> None:
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
        assert result.startswith('x' * 10)
        assert 'truncated at 10 chars' in result

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
        assert result.startswith('x' * 10)
        assert 'truncated at 10 chars' in result

    def test_smart_truncation_with_stderr(self, shell_dir: Path) -> None:
        """When stderr_text is provided and output is over limit, use smart truncation."""
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=100,
            persist_cwd=False,
            allow_interactive=False,
        )
        long_text = 'x' * 200
        result = ts._truncate(long_text, stderr_text='error msg')
        assert 'stdout truncated' in result
        assert len(result) < 200

    def test_smart_truncation_not_triggered_under_limit(self, shell_dir: Path) -> None:
        """When under limit, stderr_text parameter is irrelevant."""
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=100,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = ts._truncate('short', stderr_text='error')
        assert result == 'short'

    def test_truncation_without_stderr_uses_basic(self, shell_dir: Path) -> None:
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
        assert 'output truncated at 10 chars' in result
        assert 'stdout truncated' not in result

    def test_truncation_with_stderr_uses_smart(self, shell_dir: Path) -> None:
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
        result = ts._truncate('x' * 20, stderr_text='err')
        assert 'stdout truncated' in result
        assert 'output truncated' not in result


class TestCwdSentinel:
    def test_wrap_command_appends_sentinel(self, toolset: ShellToolset) -> None:
        result = toolset._wrap_command_for_cwd('echo hello')
        assert _PWD_SENTINEL in result
        assert result == f'echo hello && echo {_PWD_SENTINEL}$(pwd)'

    def test_extract_cwd_no_sentinel(self, toolset: ShellToolset) -> None:
        cleaned, cwd = toolset._extract_cwd_from_output('just some output')
        assert cleaned == 'just some output'
        assert cwd is None

    def test_extract_cwd_with_valid_path(self, toolset: ShellToolset, shell_dir: Path) -> None:
        stdout = f'some output\n{_PWD_SENTINEL}{shell_dir}\n'
        cleaned, cwd = toolset._extract_cwd_from_output(stdout)
        assert 'some output' in cleaned
        assert _PWD_SENTINEL not in cleaned
        assert cwd == shell_dir

    def test_extract_cwd_invalid_path(self, toolset: ShellToolset) -> None:
        stdout = f'output\n{_PWD_SENTINEL}/nonexistent_dir_xyz_999\n'
        cleaned, cwd = toolset._extract_cwd_from_output(stdout)
        assert _PWD_SENTINEL not in cleaned
        assert cwd is None

    def test_extract_cwd_empty_path(self, toolset: ShellToolset) -> None:
        stdout = f'output\n{_PWD_SENTINEL}\n'
        _, cwd = toolset._extract_cwd_from_output(stdout)
        assert cwd is None

    def test_extract_cwd_strips_sentinel_from_output(self, toolset: ShellToolset, shell_dir: Path) -> None:
        """Sentinel line should never appear in output shown to model."""
        stdout = f'line1\nline2\n{_PWD_SENTINEL}{shell_dir}\n'
        cleaned, _ = toolset._extract_cwd_from_output(stdout)
        assert _PWD_SENTINEL not in cleaned
        assert 'line1' in cleaned
        assert 'line2' in cleaned

    def test_extract_cwd_uses_rfind(self, toolset: ShellToolset, shell_dir: Path) -> None:
        """If sentinel appears multiple times, use the LAST one (rfind)."""
        stdout = f'{_PWD_SENTINEL}/fake\nmore output\n{_PWD_SENTINEL}{shell_dir}\n'
        _, cwd = toolset._extract_cwd_from_output(stdout)
        assert cwd == shell_dir

    def test_extract_cwd_cleaned_rstrip(self, toolset: ShellToolset, shell_dir: Path) -> None:
        stdout = f'content\n\n{_PWD_SENTINEL}{shell_dir}\n'
        cleaned, _ = toolset._extract_cwd_from_output(stdout)
        assert not cleaned.endswith('\n')
        assert 'content' in cleaned

    def test_extract_cwd_split_maxsplit(self, toolset: ShellToolset, shell_dir: Path) -> None:
        stdout = f'{_PWD_SENTINEL}{shell_dir}\nextra_line\n'
        _, cwd = toolset._extract_cwd_from_output(stdout)
        assert cwd == shell_dir


class TestRunCommand:
    async def test_basic_echo(self, toolset: ShellToolset) -> None:
        result = await toolset.run_command('echo hello')
        assert '[stdout]' in result
        assert 'hello' in result

    async def test_stderr_output(self, toolset: ShellToolset) -> None:
        result = await toolset.run_command('echo error >&2')
        assert '[stderr]' in result
        assert 'error' in result

    async def test_mixed_output(self, toolset: ShellToolset) -> None:
        result = await toolset.run_command('echo out && echo err >&2')
        assert '[stdout]' in result
        assert '[stderr]' in result

    async def test_exit_code_reported(self, toolset: ShellToolset) -> None:
        result = await toolset.run_command('exit 42')
        assert '[exit code: 42]' in result

    async def test_exit_code_zero_not_shown(self, toolset: ShellToolset) -> None:
        result = await toolset.run_command('echo ok')
        assert 'exit code' not in result

    async def test_timeout(self, shell_dir: Path) -> None:
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
        assert 'timed out' in result

    async def test_custom_timeout(self, toolset: ShellToolset) -> None:
        result = await toolset.run_command('sleep 10', timeout_seconds=0.5)
        assert 'timed out' in result

    async def test_no_output(self, toolset: ShellToolset) -> None:
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
        assert 'truncated at 50 chars' in result

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

    async def test_denied_command_in_run(self, toolset: ShellToolset) -> None:
        with pytest.raises(PermissionError, match="'rm' is denied"):
            await toolset.run_command('rm -rf /')

    async def test_cwd_used(self, toolset: ShellToolset, shell_dir: Path) -> None:
        result = await toolset.run_command('cat test.txt')
        assert 'hello' in result

    async def test_multiline_output(self, toolset: ShellToolset) -> None:
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

    async def test_nonzero_exit_shows_code(self, toolset: ShellToolset) -> None:
        result = await toolset.run_command('exit 1')
        assert '[exit code: 1]' in result

    async def test_zero_exit_no_code(self, toolset: ShellToolset) -> None:
        result = await toolset.run_command('echo success')
        assert 'exit code' not in result

    async def test_stdout_stderr_separated_by_newline(self, toolset: ShellToolset) -> None:
        result = await toolset.run_command('echo out && echo err >&2')
        assert '[stdout]\nout\n\n[stderr]\nerr' in result

    async def test_non_ascii_stdout(self, toolset: ShellToolset) -> None:
        result = await toolset.run_command(
            f'{sys.executable} -c "import sys; sys.stdout.buffer.write(b\'hello \\xff\\xfe world\\n\')"'
        )
        assert 'hello' in result

    async def test_non_ascii_stderr(self, toolset: ShellToolset) -> None:
        result = await toolset.run_command(
            f'{sys.executable} -c "import sys; sys.stderr.buffer.write(b\'err \\xff\\xfe msg\\n\')"'
        )
        assert 'err' in result

    async def test_stdout_chunk_join(self, toolset: ShellToolset) -> None:
        result = await toolset.run_command(f"{sys.executable} -c \"print('A' * 100 + 'B' * 100)\"")
        assert 'A' * 100 + 'B' * 100 in result

    async def test_exact_no_output_message(self, toolset: ShellToolset) -> None:
        result = await toolset.run_command('true')
        assert result == '(no output)'

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

    async def test_persist_cwd_sentinel_stripped_from_output(self, shell_dir: Path) -> None:
        """The pwd sentinel should never appear in output shown to user."""
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
        result = await ts.run_command('echo visible')
        assert _PWD_SENTINEL not in result
        assert 'visible' in result

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
        """Verify the process gets its own session (child is process group leader)."""
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
        result = await ts.run_command(f'{sys.executable} -c "import os; print(os.getpgid(0) == os.getpid())"')
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
        command_id = result.split('ID: ')[1].strip()
        await ts.stop_command(command_id)

    async def test_check_unknown_id(self, toolset: ShellToolset) -> None:
        result = await toolset.check_command('nonexistent_id')
        assert 'unknown command ID' in result

    async def test_stop_unknown_id(self, toolset: ShellToolset) -> None:
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
        command_id = start_result.split('ID: ')[1].strip()

        import anyio

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
        command_id = start_result.split('ID: ')[1].strip()

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
        command_id = start_result.split('ID: ')[1].strip()

        import anyio

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
        with pytest.raises(PermissionError, match="'rm' is denied"):
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
        command_id = start_result.split('ID: ')[1].strip()

        import anyio

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
        command_id = start_result.split('ID: ')[1].strip()

        import anyio

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
        command_id = start_result.split('ID: ')[1].strip()

        check_result = await ts.check_command(command_id)
        assert 'no output yet' in check_result

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
        command_id = start_result.split('ID: ')[1].strip()

        import anyio

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
        command_id = start_result.split('ID: ')[1].strip()

        import anyio

        await anyio.sleep(0.5)

        await ts.stop_command(command_id)

        # Should now be unknown
        check_result = await ts.check_command(command_id)
        assert 'unknown command ID' in check_result


# ============================================================================
# ============================================================================


class TestEdgeCases:
    async def test_toolset_tool_names(self, toolset: ShellToolset) -> None:
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

    def test_wrap_command_uses_correct_sentinel(self, toolset: ShellToolset) -> None:
        result = toolset._wrap_command_for_cwd('ls')
        assert '__HARNESS_PWD__' in result
        assert '$(pwd)' in result

    def test_extract_cwd_rfind_not_find(self, toolset: ShellToolset, shell_dir: Path) -> None:
        stdout = f'{_PWD_SENTINEL}/fake\nstuff\n{_PWD_SENTINEL}{shell_dir}\n'
        _, cwd = toolset._extract_cwd_from_output(stdout)
        assert cwd == shell_dir

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
        # Successful echo — sentinel shows same dir, cwd should remain valid
        await ts.run_command('echo hi')
        assert ts._cwd.is_dir()

    async def test_persist_cwd_false_skips_sentinel(self, shell_dir: Path) -> None:
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
        result = await ts.run_command('echo test')
        assert _PWD_SENTINEL not in result

    async def test_start_new_session_true(self, shell_dir: Path) -> None:
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
        result = await ts.run_command(f'{sys.executable} -c "import os; print(os.getpgid(0) == os.getpid())"')
        assert 'True' in result


# ============================================================================
# ============================================================================


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
        model = TestModel(custom_output_text='done', call_tools=[])
        agent: Agent[None, str] = Agent(model, capabilities=[Shell(cwd=tmp_path)])
        result = await agent.run('run echo hello')
        assert result.output == 'done'
