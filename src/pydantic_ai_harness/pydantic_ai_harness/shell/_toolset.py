"""Shell toolset — gives agents the ability to run commands."""

from __future__ import annotations

import functools
import os
import re
import shlex
import signal
import subprocess
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Concatenate, ParamSpec

import anyio
import anyio.abc
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

_IO_DRAIN_TIMEOUT: float = 2.0
_KILL_GRACE_PERIOD: float = 2.0

_P = ParamSpec('_P')


def _recoverable(
    fn: Callable[Concatenate[ShellToolset, _P], Awaitable[str]],
) -> Callable[Concatenate[ShellToolset, _P], Awaitable[str]]:
    """Convert model-correctable errors into `ModelRetry`.

    pyai only feeds `ModelRetry` back to the model as a retry prompt; any other
    exception propagates and aborts the whole run. A denied command is something
    the model can recover from (pick an allowed one), so surface it as a retry
    instead of crashing the agent.
    """

    @functools.wraps(fn)
    async def wrapper(self: ShellToolset, *args: _P.args, **kwargs: _P.kwargs) -> str:
        try:
            return await fn(self, *args, **kwargs)
        except PermissionError as e:
            raise ModelRetry(str(e)) from e

    return wrapper


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


class ShellToolset(FunctionToolset[AgentDepsT]):
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
        # The configured starting directory, never mutated by persist_cwd, so
        # `for_run` can hand each run a fresh instance rooted back here.
        self._initial_cwd = self._cwd
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

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Return a fresh instance per run so cwd and background processes are isolated.

        `get_toolset` builds one shared instance at agent construction (see
        `AbstractToolset.for_run`, which defaults to returning `self`). This
        toolset holds mutable per-run state (`_cwd`, `_background`), so without
        an override two concurrent runs would corrupt each other's cwd and kill
        each other's background processes.
        """
        return ShellToolset(
            cwd=self._initial_cwd,
            allowed_commands=self._allowed_commands,
            denied_commands=self._denied_commands,
            denied_operators=self._denied_operators,
            default_timeout=self._default_timeout,
            max_output_chars=self._max_output_chars,
            persist_cwd=self._persist_cwd,
            allow_interactive=self._allow_interactive,
        )

    async def __aexit__(self, *args: Any) -> None:
        """Terminate all remaining background processes and clean up temp files."""
        for bg in self._background.values():
            if not bg.finished:
                await self._kill_process_group(bg.proc)
                with anyio.CancelScope(shield=True):
                    await bg.proc.wait()
                await bg.proc.aclose()
            self._cleanup_bg_files(bg)
        self._background.clear()

    def _first_denied_operator(self, command: str) -> str | None:
        """Return the first denied operator found in command, or None."""
        return next((op for op in self._denied_operators if op in command), None)

    def _check_command(self, command: str) -> None:
        """Validate command against allow/deny lists.

        These checks are best-effort and are not a security boundary — a
        sufficiently motivated agent can bypass them. Use OS-level isolation
        (containers, sandboxes) for hard enforcement.
        """
        if not self._allow_interactive and _is_interactive_command(command):
            raise PermissionError(f'Interactive commands are not allowed. Command: {command!r}')

        matched_op = self._first_denied_operator(command)
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

    def _truncate(self, text: str) -> str:
        """Truncate output to the configured cap, keeping the tail.

        The most useful output — errors, stack traces, exit info, and the
        `[stderr]` section (which callers append last) — lands at the end, so
        the head is dropped and the final `max_output_chars` are kept.
        """
        if len(text) <= self._max_output_chars:
            return text
        marker = f'[... output truncated, showing last {self._max_output_chars} chars]\n'
        return marker + text[-self._max_output_chars :]

    def _build_cwd_capture(self, command: str) -> tuple[str, Path | None]:
        """Wrap a command to record its final working directory out-of-band.

        `pwd` is written to a private temp file whose random path the agent's
        command can't address, so command output can never spoof the tracked
        cwd — unlike parsing a sentinel out of stdout, where any command that
        prints the sentinel string (or one using `;` to skip success-gating)
        could redirect the cwd. Returns the wrapped command plus the temp-file
        path, or the command unchanged and `None` when cwd tracking is off.
        """
        if not self._persist_cwd:
            return command, None
        fd, name = tempfile.mkstemp(prefix='harness_cwd_')
        os.close(fd)
        wrapped = f'{command}\n__harness_ec=$?\npwd > {shlex.quote(name)}\nexit $__harness_ec'
        return wrapped, Path(name)

    def _apply_captured_cwd(self, cwd_file: Path) -> None:
        """Update the persistent cwd from the capture file, ignoring junk."""
        try:
            recorded = cwd_file.read_text(encoding='utf-8').strip()
        except OSError:  # pragma: no cover
            return
        if not recorded:
            return
        candidate = Path(recorded)
        if candidate.is_dir():
            self._cwd = candidate

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

    @_recoverable
    async def run_command(self, command: str, *, timeout_seconds: float | None = None) -> str:
        """Execute a shell command and return its output.

        Args:
            command: The shell command to run.
            timeout_seconds: Maximum seconds to wait (default: 30).

        Returns:
            Labeled stdout/stderr output with exit code on non-zero exit.
        """
        self._check_command(command)
        timeout = timeout_seconds if timeout_seconds is not None else self._default_timeout

        actual_command, cwd_file = self._build_cwd_capture(command)
        try:
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

            parts: list[str] = []
            if stdout:
                parts.append(f'[stdout]\n{stdout}')
            if stderr:
                parts.append(f'[stderr]\n{stderr}')
            output = '\n'.join(parts) if parts else '(no output)'

            output = self._truncate(output)
            exit_code = proc.returncode if proc.returncode is not None else 0

            if cwd_file is not None and exit_code == 0:
                self._apply_captured_cwd(cwd_file)

            if exit_code != 0:
                return f'{output}\n[exit code: {exit_code}]'
            return output
        finally:
            if cwd_file is not None:
                cwd_file.unlink(missing_ok=True)

    @_recoverable
    async def start_command(self, command: str) -> str:
        """Start a long-running command in the background (e.g. a server or watcher).

        Callers MUST call `stop_command(command_id)` when done to terminate the
        process and clean up temporary output files.

        Args:
            command: The shell command to run in the background.

        Returns:
            A message containing the unique command ID for later check/stop calls.
        """
        self._check_command(command)
        command_id = uuid.uuid4().hex[:12]

        stdout_file = tempfile.NamedTemporaryFile(mode='w+b', prefix=f'harness_{command_id}_out_', delete=False)
        stderr_file = tempfile.NamedTemporaryFile(mode='w+b', prefix=f'harness_{command_id}_err_', delete=False)

        try:
            proc = await anyio.open_process(
                command,
                cwd=self._cwd,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
        except BaseException:
            stdout_file.close()
            stderr_file.close()
            os.unlink(stdout_file.name)
            os.unlink(stderr_file.name)
            raise

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
        output_sections: list[str] = []
        if stdout:
            output_sections.append(f'[stdout]\n{stdout}')
        if stderr:
            output_sections.append(f'[stderr]\n{stderr}')
        if output_sections:
            parts.append(self._truncate('\n'.join(output_sections)))
        else:
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
        output_sections: list[str] = []
        if stdout:
            output_sections.append(f'[stdout]\n{stdout}')
        if stderr:
            output_sections.append(f'[stderr]\n{stderr}')
        if output_sections:
            parts.append(self._truncate('\n'.join(output_sections)))
        else:
            parts.append('(no output)')

        return '\n'.join(parts)
