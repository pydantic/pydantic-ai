"""Shell toolset — gives agents the ability to run commands."""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import tempfile
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import anyio
import anyio.abc
from pydantic_ai.toolsets import FunctionToolset

_PWD_SENTINEL = '__HARNESS_PWD__'
_IO_DRAIN_TIMEOUT: float = 2.0
_KILL_GRACE_PERIOD: float = 2.0


def _is_interactive_command(command: str) -> bool:
    """Detect commands that typically require interactive input."""
    interactive_patterns = [
        r'^(vi|vim|nano|emacs|less|more|top|htop|man)\b',
        r'^sudo\s',
        r'^passwd\b',
        r'^ssh\b',
        r'^telnet\b',
        r'^ftp\b',
    ]
    return any(re.match(p, command.strip()) for p in interactive_patterns)


class _BackgroundProcess:
    """State for a background command using temp files for output."""

    __slots__ = ('proc', 'command', 'stdout_path', 'stderr_path', 'finished', 'exit_code')

    def __init__(
        self,
        proc: anyio.abc.Process,
        command: str,
        stdout_path: str,
        stderr_path: str,
    ) -> None:
        self.proc = proc
        self.command = command
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.finished = False
        self.exit_code: int | None = None


class ShellToolset(FunctionToolset[Any]):
    """Gives an agent the ability to execute shell commands.

    Supports synchronous execution (run_command) and background processes
    (start_command / check_command / stop_command). Output is streamed,
    truncated to fit model context, and labelled with stdout/stderr/exit code.

    Optionally tracks the working directory across calls so ``cd`` persists.
    """

    def __init__(
        self,
        *,
        cwd: Path,
        allowed_commands: Sequence[str],
        denied_commands: Sequence[str],
        denied_operators: Sequence[str],
        default_timeout: float,
        max_output_chars: int,
        persist_cwd: bool,
        allow_interactive: bool,
    ) -> None:
        super().__init__()
        self._cwd = cwd.resolve()
        self._allowed_commands = list(allowed_commands)
        self._denied_commands = list(denied_commands)
        self._denied_operators = list(denied_operators)
        self._default_timeout = default_timeout
        self._max_output_chars = max_output_chars
        self._persist_cwd = persist_cwd
        self._allow_interactive = allow_interactive
        self._background: dict[str, _BackgroundProcess] = {}

        if self._allowed_commands and self._denied_commands:
            raise ValueError('Specify allowed_commands or denied_commands, not both.')

        self.add_function(self.run_command, name='run_command')
        self.add_function(self.start_command, name='start_command')
        self.add_function(self.check_command, name='check_command')
        self.add_function(self.stop_command, name='stop_command')

    def _check_command(self, command: str) -> None:
        """Validate command against allow/deny lists."""
        if not self._allow_interactive and _is_interactive_command(command):
            raise PermissionError(f'Interactive commands are not allowed. Command: {command!r}')

        matched_op = next((op for op in self._denied_operators if op in command), None)
        if matched_op:
            raise PermissionError(f'Shell operator {matched_op!r} is not allowed.')

        try:
            tokens = shlex.split(command)
        except ValueError:
            return
        if not tokens:
            return
        executable = tokens[0]

        if self._denied_commands and executable in self._denied_commands:
            raise PermissionError(f'Command {executable!r} is denied.')
        if self._allowed_commands and executable not in self._allowed_commands:
            raise PermissionError(f'Command {executable!r} is not in the allowed list.')

    def _truncate(self, text: str, *, stderr_text: str = '') -> str:
        """Truncate output, reserving space for stderr when both streams are present."""
        if len(text) <= self._max_output_chars:
            return text
        if not stderr_text:
            return text[: self._max_output_chars] + f'\n[... output truncated at {self._max_output_chars} chars]'

        stderr_budget = min(len(stderr_text) + len('[stderr]\n'), self._max_output_chars // 3)
        stdout_budget = self._max_output_chars - stderr_budget
        truncated = text[:stdout_budget] + f'\n[... stdout truncated at {stdout_budget} chars]'
        return truncated

    def _wrap_command_for_cwd(self, command: str) -> str:
        """Append pwd sentinel to command for cwd tracking."""
        return f'{command} && echo {_PWD_SENTINEL}$(pwd)'

    def _extract_cwd_from_output(self, stdout: str) -> tuple[str, Path | None]:
        """Extract and strip pwd sentinel from stdout.

        Returns (cleaned_stdout, new_cwd_or_none).
        """
        sentinel_idx = stdout.rfind(_PWD_SENTINEL)
        if sentinel_idx == -1:
            return stdout, None
        after_sentinel = stdout[sentinel_idx + len(_PWD_SENTINEL) :]
        path_str = after_sentinel.strip().split('\n', maxsplit=1)[0].strip()
        cleaned = stdout[:sentinel_idx].rstrip('\n')
        if not path_str:
            return cleaned, None
        new_cwd = Path(path_str)
        if new_cwd.is_dir():
            return cleaned, new_cwd
        return cleaned, None

    async def _kill_process_group(self, proc: anyio.abc.Process) -> None:
        """SIGTERM the process group, escalating to SIGKILL after the grace period."""
        pid = proc.pid
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            return

        with anyio.move_on_after(_KILL_GRACE_PERIOD):
            await proc.wait()
            return

        # Still alive after grace period — hard kill
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    async def _drain_with_timeout(
        self,
        stdout_chunks: list[bytes],
        stderr_chunks: list[bytes],
        proc: anyio.abc.Process,
    ) -> None:
        """Drain remaining pipe data after kill (grandchildren may still hold the pipe)."""

        async def _drain_stdout() -> None:
            if proc.stdout is None:
                return
            try:
                async for chunk in proc.stdout:
                    stdout_chunks.append(chunk)
            except (anyio.ClosedResourceError, anyio.BrokenResourceError):
                pass

        async def _drain_stderr() -> None:
            if proc.stderr is None:
                return
            try:
                async for chunk in proc.stderr:
                    stderr_chunks.append(chunk)
            except (anyio.ClosedResourceError, anyio.BrokenResourceError):
                pass

        with anyio.move_on_after(_IO_DRAIN_TIMEOUT):
            async with anyio.create_task_group() as tg:
                tg.start_soon(_drain_stdout)
                tg.start_soon(_drain_stderr)

    async def run_command(self, command: str, *, timeout_seconds: float | None = None) -> str:
        """Execute a shell command and return its output.

        Args:
            command: The shell command to run.
            timeout_seconds: Maximum seconds to wait (default: default_timeout).

        Returns:
            Labeled stdout/stderr output with exit code on non-zero exit.
        """
        self._check_command(command)
        timeout = timeout_seconds if timeout_seconds is not None else self._default_timeout

        actual_command = self._wrap_command_for_cwd(command) if self._persist_cwd else command

        proc = await anyio.open_process(
            actual_command,
            cwd=self._cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        try:
            assert proc.stdout is not None
            assert proc.stderr is not None

            async def _read_stdout() -> None:
                assert proc.stdout is not None
                async for chunk in proc.stdout:
                    stdout_chunks.append(chunk)

            async def _read_stderr() -> None:
                assert proc.stderr is not None
                async for chunk in proc.stderr:
                    stderr_chunks.append(chunk)

            with anyio.fail_after(timeout):
                async with anyio.create_task_group() as tg:
                    tg.start_soon(_read_stdout)
                    tg.start_soon(_read_stderr)
                await proc.wait()
        except TimeoutError:
            await self._kill_process_group(proc)
            with anyio.CancelScope(shield=True):
                await proc.wait()
                await self._drain_with_timeout(stdout_chunks, stderr_chunks, proc)
            return f'[Command timed out after {timeout}s]'
        finally:
            await proc.aclose()

        stdout = b''.join(stdout_chunks).decode('utf-8', errors='replace')
        stderr = b''.join(stderr_chunks).decode('utf-8', errors='replace')

        new_cwd: Path | None = None
        if self._persist_cwd:
            stdout, new_cwd = self._extract_cwd_from_output(stdout)

        parts: list[str] = []
        if stdout:
            parts.append(f'[stdout]\n{stdout}')
        if stderr:
            parts.append(f'[stderr]\n{stderr}')
        output = '\n'.join(parts) if parts else '(no output)'

        output = self._truncate(output, stderr_text=stderr)
        exit_code = proc.returncode if proc.returncode is not None else 0

        if self._persist_cwd and exit_code == 0 and new_cwd is not None:
            self._cwd = new_cwd

        if exit_code != 0:
            return f'{output}\n[exit code: {exit_code}]'
        return output

    async def start_command(self, command: str) -> str:
        """Start a long-running command in the background (e.g. a server or watcher).

        Args:
            command: The shell command to run in the background.

        Returns:
            A message containing the unique command ID for later check/stop calls.
        """
        self._check_command(command)
        command_id = uuid.uuid4().hex[:12]

        stdout_file = tempfile.NamedTemporaryFile(mode='w+b', prefix=f'harness_{command_id}_out_', delete=False)
        stderr_file = tempfile.NamedTemporaryFile(mode='w+b', prefix=f'harness_{command_id}_err_', delete=False)

        proc = await anyio.open_process(
            command,
            cwd=self._cwd,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
        )

        stdout_file.close()
        stderr_file.close()

        bg = _BackgroundProcess(
            proc=proc,
            command=command,
            stdout_path=stdout_file.name,
            stderr_path=stderr_file.name,
        )
        self._background[command_id] = bg

        return f'Started background command: {command!r}\nID: {command_id}'

    def _read_bg_output(self, bg: _BackgroundProcess) -> tuple[str, str]:
        """Read current output from background process temp files."""
        try:
            stdout = Path(bg.stdout_path).read_text(encoding='utf-8', errors='replace')
        except OSError:
            stdout = ''
        try:
            stderr = Path(bg.stderr_path).read_text(encoding='utf-8', errors='replace')
        except OSError:
            stderr = ''
        return stdout, stderr

    def _cleanup_bg_files(self, bg: _BackgroundProcess) -> None:
        """Remove temp files for a background process."""
        try:
            os.unlink(bg.stdout_path)
        except OSError:
            pass
        try:
            os.unlink(bg.stderr_path)
        except OSError:
            pass

    async def check_command(self, command_id: str) -> str:
        """Check the status and recent output of a background command.

        Args:
            command_id: The ID returned by start_command.

        Returns:
            Status and recent output of the background command.
        """
        bg = self._background.get(command_id)
        if bg is None:
            return f'[Error: unknown command ID {command_id!r}]'

        if not bg.finished and bg.proc.returncode is not None:
            bg.exit_code = bg.proc.returncode
            bg.finished = True

        stdout, stderr = self._read_bg_output(bg)

        status = 'finished' if bg.finished else 'running'
        parts = [f'[status: {status}]']
        if bg.finished and bg.exit_code is not None:
            parts.append(f'[exit code: {bg.exit_code}]')
        if stdout:
            parts.append(f'[stdout]\n{self._truncate(stdout)}')
        if stderr:
            parts.append(f'[stderr]\n{self._truncate(stderr)}')
        if not stdout and not stderr:
            parts.append('(no output yet)')

        return '\n'.join(parts)

    async def stop_command(self, command_id: str) -> str:
        """Stop a background command and return its final output.

        Args:
            command_id: The ID returned by start_command.

        Returns:
            Final output and exit status of the stopped command.
        """
        bg = self._background.get(command_id)
        if bg is None:
            return f'[Error: unknown command ID {command_id!r}]'

        if not bg.finished:
            await self._kill_process_group(bg.proc)
            with anyio.CancelScope(shield=True):
                await bg.proc.wait()
            bg.exit_code = bg.proc.returncode
            bg.finished = True

        stdout, stderr = self._read_bg_output(bg)

        self._cleanup_bg_files(bg)
        del self._background[command_id]
        await bg.proc.aclose()

        parts = [f'[stopped: {bg.command!r}]']
        if bg.exit_code is not None:
            parts.append(f'[exit code: {bg.exit_code}]')
        if stdout:
            parts.append(f'[stdout]\n{self._truncate(stdout)}')
        if stderr:
            parts.append(f'[stderr]\n{self._truncate(stderr)}')
        if not stdout and not stderr:
            parts.append('(no output)')

        return '\n'.join(parts)
