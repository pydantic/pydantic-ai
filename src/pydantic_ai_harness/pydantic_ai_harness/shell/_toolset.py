"""Shell toolset -- gives agents the ability to run commands inside the run's workspace."""

from __future__ import annotations

import fnmatch
import hashlib
import os
import posixpath
import re
import shlex
import uuid
import weakref
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import anyio

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, ToolsetTool
from pydantic_ai.workspaces import Workspace, WorkspaceTimeoutError
from pydantic_ai_harness._output import truncate_tail
from pydantic_ai_harness._warn import SET_WORKING_DIR_ON_THE_WORKSPACE, warn_argument_ignored
from pydantic_ai_harness._workspace import metadata_dir, require_workspace, supports_commands
from pydantic_ai_harness.shell._jobs import CONTROL_TIMEOUT, Job
from pydantic_ai_harness.shell._limits import file_limit_status, limited_script, validate_file_limit
from pydantic_ai_harness.shell._persistent import MAX_FOREGROUND_WAIT, CommandMode, run_persistent_command
from pydantic_ai_harness.shell._policy import is_interactive_command, recoverable

RUN_SCOPED_TOOL_NAMES: tuple[str, ...] = ('run_command', 'start_command', 'check_command', 'stop_command')
"""The default tools. Background commands keep running until they exit, `stop_command` stops them, or the workspace ends."""

PERSISTENT_TOOL_NAME = 'shell'
"""The opt-in tool whose commands return handles to their log and status files."""

SHELL_TOOL_NAMES: tuple[str, ...] = (*RUN_SCOPED_TOOL_NAMES, PERSISTENT_TOOL_NAME)
"""Every tool `Shell` can register, in registration order."""

_OUTPUT_BYTES_PER_CHAR = 4
"""UTF-8 bytes per character at most: reading `4 * max_output_chars` bytes of a log keeps every character the cap keeps."""


_COMMAND_ID = re.compile(r'[0-9a-f]{32}')
"""A background command's ID: the name of its job directory."""


class ShellToolset(FunctionToolset[AgentDepsT]):
    """Gives an agent the ability to execute shell commands in the run's workspace.

    Supports synchronous execution (run_command) and background processes
    (start_command / check_command / stop_command). Output is truncated to fit
    model context and labelled with stdout/stderr/exit code. The opt-in `shell`
    tool instead returns handles to its commands' output and exit status. Every
    command, file, and signal goes through `ctx.workspace`.

    Background commands are not tied to the run: a later run on the same
    workspace can check or stop them by ID.

    Optionally tracks the working directory across calls so `cd` persists.
    """

    def __init__(
        self,
        *,
        allowed_commands: Sequence[str],
        denied_commands: Sequence[str],
        denied_operators: Sequence[str],
        default_timeout: float,
        max_output_chars: int,
        persist_cwd: bool,
        allow_interactive: bool,
        max_file_bytes: int | None = None,
        env: Mapping[str, str] | None = None,
        denied_env_patterns: Sequence[str] = (),
        tools: Sequence[str] = RUN_SCOPED_TOOL_NAMES,
        cwd: Path | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        if cwd is not None:
            warn_argument_ignored('ShellToolset', 'cwd', SET_WORKING_DIR_ON_THE_WORKSPACE, stacklevel=3)
        # The absolute workspace path `persist_cwd` last recorded; `None` means the working directory.
        self._cwd: str | None = None
        self._cwd_run_id: str | None = None
        self._allowed_commands = list(allowed_commands)
        self._denied_commands = list(denied_commands)
        self._denied_operators = list(denied_operators)
        self._default_timeout = default_timeout
        self._max_output_chars = max_output_chars
        validate_file_limit(max_file_bytes, persistent=PERSISTENT_TOOL_NAME in tools)
        if max_file_bytes is not None and persist_cwd:
            raise ValueError(
                'max_file_bytes is not supported with persist_cwd; cwd capture writes a file in the child.'
            )
        self._max_file_bytes = max_file_bytes
        self._persist_cwd = persist_cwd
        self._allow_interactive = allow_interactive
        self._env = dict(env) if env is not None else None
        self._denied_env_patterns = list(denied_env_patterns)
        self._tools = tuple(tools)
        # Each workspace's job directory, looked up on its first command; weak, so no finished run's workspace is kept.
        self._jobs_dirs: weakref.WeakKeyDictionary[Workspace, str] = weakref.WeakKeyDictionary()

        if self._allowed_commands and self._denied_commands:
            raise ValueError('Specify allowed_commands or denied_commands, not both.')
        if max_output_chars <= 0:
            raise ValueError('max_output_chars must be a positive integer.')
        if unknown := sorted(set(self._tools) - set(SHELL_TOOL_NAMES)):
            raise ValueError(f'Unknown shell tools: {", ".join(unknown)}. Available: {", ".join(SHELL_TOOL_NAMES)}.')
        if PERSISTENT_TOOL_NAME in self._tools and not 0 < default_timeout <= MAX_FOREGROUND_WAIT:
            raise ValueError(
                f'default_timeout must be greater than zero and at most {MAX_FOREGROUND_WAIT:g} seconds '
                'for the shell tool.'
            )

        command_metadata = {'code_arg_name': 'command', 'code_arg_language': 'shell'}
        registrations: dict[str, Callable[..., Awaitable[str]]] = {
            'run_command': self.run_command,
            'start_command': self.start_command,
            'check_command': self.check_command,
            'stop_command': self.stop_command,
            PERSISTENT_TOOL_NAME: self.shell,
        }
        for name in SHELL_TOOL_NAMES:
            if name in self._tools:
                metadata: dict[str, Any] = (
                    command_metadata.copy() if name in ('run_command', 'start_command', PERSISTENT_TOOL_NAME) else {}
                )
                if name in ('run_command', PERSISTENT_TOOL_NAME):
                    metadata['temporal'] = {'start_to_close_timeout': timedelta(seconds=MAX_FOREGROUND_WAIT + 30)}
                self.add_function(registrations[name], name=name, metadata=metadata or None)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Return a fresh instance per run when `persist_cwd` tracks a cwd, so each run has its own.

        `get_toolset` builds one shared instance at agent construction (see
        `AbstractToolset.for_run`, which defaults to returning `self`). The tracked
        `_cwd` is per-run state, so without a copy two concurrent runs would corrupt
        each other's cwd. Without `persist_cwd` there is no per-run state and every run
        shares this instance, which durable execution requires of the toolset it registered.
        """
        if not self._persist_cwd:
            return self
        return ShellToolset(
            allowed_commands=self._allowed_commands,
            denied_commands=self._denied_commands,
            denied_operators=self._denied_operators,
            default_timeout=self._default_timeout,
            max_output_chars=self._max_output_chars,
            max_file_bytes=self._max_file_bytes,
            persist_cwd=self._persist_cwd,
            allow_interactive=self._allow_interactive,
            env=self._env,
            denied_env_patterns=self._denied_env_patterns,
            tools=self._tools,
            id=self.id,
        )

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Offer no tools when the workspace cannot execute commands; fail a run with no workspace."""
        if not ctx.workspace.attached:
            require_workspace(ctx.workspace, 'ShellToolset', ctx.messages)
        if not supports_commands(ctx.workspace):
            return {}
        return await super().get_tools(ctx)

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        """Enforce the model-visible output cap at the tool dispatch seam.

        Tools place control metadata (status, exit code, `start_command`'s ID
        line) at the end of their responses, so keeping the tail preserves it
        without any per-tool cases here. Only `str` results are capped; a
        future tool returning rich content (e.g. `ToolReturn`) needs this seam
        extended.
        """
        result = await super().call_tool(name, tool_args, ctx, tool)
        if not isinstance(result, str):
            return result
        return truncate_tail(result, self._max_output_chars)

    def _resolve_env(self) -> dict[str, str] | None:
        """The variables handed to the workspace for each command, on top of its own environment.

        The workspace decides the base environment.
        `None` adds nothing. An explicit `env` is added, minus names that match
        `denied_env_patterns` (glob, via `fnmatch`).
        """
        if self._env is None:
            return None
        if not self._denied_env_patterns:
            return dict(self._env)
        return {
            name: value
            for name, value in self._env.items()
            if not any(fnmatch.fnmatchcase(name, pattern) for pattern in self._denied_env_patterns)
        }

    async def _cwd_for(self, ctx: RunContext[AgentDepsT]) -> str:
        """The absolute workspace directory the next `run_command` or `start_command` command starts in."""
        state: str | None = None
        recorded: str | None = None
        if self._persist_cwd and ctx.run_id is not None:
            state = await self._cwd_state_path(ctx)
            try:
                recorded = (await ctx.workspace.read_bytes(state)).decode('utf-8')
            except FileNotFoundError:
                if self._cwd is not None and self._cwd_run_id == ctx.run_id:
                    # This worker saw a saved cwd for this run; a missing file now is lost state.
                    self._cwd = None
                    raise ModelRetry('The saved working directory was lost; now in the workspace working directory.')
            except UnicodeDecodeError:
                pass
        else:
            recorded = self._cwd
        if recorded is not None:
            try:
                entry = await ctx.workspace.stat(recorded) if posixpath.isabs(recorded) else None
            except (FileNotFoundError, NotADirectoryError):
                entry = None
            if entry is None or not entry.is_dir:
                # Remote shells may report a missing cwd as an exit code; tell the model to retry elsewhere.
                if state is not None:
                    await ctx.workspace.remove(state)
                self._cwd = None
                raise ModelRetry(f'The previous directory was removed; now in {await ctx.workspace.working_dir()}.')
            return recorded
        return await ctx.workspace.working_dir()

    async def _cwd_state_path(self, ctx: RunContext[AgentDepsT]) -> str:
        # A run ID can contain path separators; hash it rather than letting it choose a filename.
        assert ctx.run_id is not None
        run_key = hashlib.sha256(ctx.run_id.encode('utf-8')).hexdigest()
        return posixpath.join(await self._jobs_base(ctx), 'run-state', run_key)

    async def clear_run_cwd(self, ctx: RunContext[AgentDepsT]) -> None:
        """Remove the run's saved directory once its agent run ends."""
        if ctx.run_id is not None:
            try:
                await ctx.workspace.remove(await self._cwd_state_path(ctx))
            except FileNotFoundError:
                pass

    async def _jobs_base(self, ctx: RunContext[AgentDepsT]) -> str:
        """The workspace directory holding job and capture files, looked up once per workspace."""
        if (jobs_dir := self._jobs_dirs.get(ctx.workspace)) is None:
            jobs_dir = self._jobs_dirs[ctx.workspace] = await metadata_dir(ctx.workspace, 'shell')
        return jobs_dir

    async def _job(self, ctx: RunContext[AgentDepsT], command_id: str) -> Job | None:
        """The background command `command_id` names in the workspace, started by this run or an earlier one."""
        if not _COMMAND_ID.fullmatch(command_id):
            return None
        return await Job.attach(ctx.workspace, posixpath.join(await self._jobs_base(ctx), command_id), combined=False)

    def _first_denied_operator(self, command: str) -> str | None:
        """Return the first denied operator found in command, or None."""
        return next((op for op in self._denied_operators if op in command), None)

    def _check_command(self, command: str) -> None:
        """Validate command against allow/deny lists.

        These checks are best-effort and are not a security boundary -- a
        sufficiently motivated agent can bypass them. Use OS-level isolation
        (containers, sandboxes) for hard enforcement.

        Rejecting a command the OS could not accept belongs here rather than in
        `recoverable`: a spawn reports a NUL byte or an unencodable character as
        the same `ValueError` whether it came from `command`, the working
        directory, or a configured `env`, and only the first of those is the
        model's to fix.
        """
        if '\x00' in command:
            raise ModelRetry('The command contains a NUL byte, which cannot be passed to a process.')
        try:
            # `os.fsencode`, not `str.encode`: the spawn encodes with the
            # filesystem encoding and `surrogateescape`, which accepts the
            # \udc80-\udcff range as the raw bytes it round-trips from. Encoding
            # as plain UTF-8 here would reject commands the OS runs happily.
            os.fsencode(command)
        except UnicodeEncodeError as e:
            raise ModelRetry('The command contains characters that cannot be encoded for the operating system.') from e

        if not self._allow_interactive and is_interactive_command(command):
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

    async def _build_cwd_capture(self, ctx: RunContext[AgentDepsT], command: str) -> tuple[str, str | None]:
        """Wrap a command to record its final working directory out-of-band.

        `pwd` is written to a file inside the workspace, so command output can
        never spoof the tracked cwd -- unlike parsing a sentinel out of stdout,
        where any command that prints the sentinel string (or one using `;` to
        skip success-gating) could redirect the cwd. The random file name keeps
        concurrent commands from colliding. Returns the wrapped command plus the
        capture path, or the command unchanged and `None` when cwd tracking is off.
        """
        if not self._persist_cwd:
            return command, None
        name = posixpath.join(await self._jobs_base(ctx), f'cwd-{uuid.uuid4().hex}')
        wrapped = f'{command}\n__harness_ec=$?\npwd > {shlex.quote(name)}\nexit $__harness_ec'
        return wrapped, name

    async def _apply_captured_cwd(self, ctx: RunContext[AgentDepsT], cwd_file: str) -> None:
        """Update the persistent cwd from the capture file, ignoring junk.

        The whole read-and-check is guarded, not just the read: the command it
        belongs to already succeeded, so a capture that isn't UTF-8 (a
        `UnicodeDecodeError`, which is a `ValueError` rather than an `OSError`)
        or a recorded path the workspace refuses to stat (`ENAMETOOLONG`) is
        bookkeeping the toolset can drop, not a tool failure to report.
        """
        try:
            recorded = (await ctx.workspace.read_bytes(cwd_file)).decode('utf-8').removesuffix('\n')
            if not posixpath.isabs(recorded):
                return
            if (await ctx.workspace.stat(recorded)).is_dir:
                if ctx.run_id is None:
                    self._cwd = posixpath.normpath(recorded)
                else:
                    state = await self._cwd_state_path(ctx)
                    await ctx.workspace.make_dir(posixpath.dirname(state))
                    # Parallel calls in one run each use their starting cwd; last completion wins.
                    # Workspace backends do not provide a cross-worker compare-and-swap for this state.
                    await ctx.workspace.write_bytes(state, posixpath.normpath(recorded).encode('utf-8'))
                    self._cwd = posixpath.normpath(recorded)
                    self._cwd_run_id = ctx.run_id
        except (OSError, ValueError):
            return

    async def _remove_capture(self, ctx: RunContext[AgentDepsT], cwd_file: str | None) -> None:
        if cwd_file is None:
            return
        with anyio.move_on_after(CONTROL_TIMEOUT, shield=True):
            try:
                await ctx.workspace.remove(cwd_file)
            except FileNotFoundError:
                pass

    @recoverable
    async def run_command(
        self, ctx: RunContext[AgentDepsT], command: str, *, timeout_seconds: float | None = None
    ) -> str:
        """Execute a shell command and return its output.

        Args:
            ctx: The current agent run context.
            command: The shell command to run.
            timeout_seconds: Maximum seconds to wait (default: 30).

        Returns:
            Labeled stdout/stderr output with exit code on non-zero exit.
        """
        self._check_command(command)
        timeout = timeout_seconds if timeout_seconds is not None else self._default_timeout

        actual_command, cwd_file = await self._build_cwd_capture(ctx, command)
        try:
            try:
                cwd = shlex.quote(await self._cwd_for(ctx))
                result = await ctx.workspace.run(
                    f'cd {cwd} || exit\n{limited_script(actual_command, self._max_file_bytes)}',
                    shell=True,
                    env=self._resolve_env(),
                    timeout=timeout,
                )
            except WorkspaceTimeoutError as e:
                return _labelled(e.stdout, e.stderr, empty=None, trailer=f'[Command timed out after {timeout}s]')

            output = _labelled(result.stdout, result.stderr, empty='(no output)')
            exit_code = result.exit_code

            if cwd_file is not None and exit_code == 0:
                await self._apply_captured_cwd(ctx, cwd_file)

            if exit_code != 0:
                output = f'{output}\n[exit code: {exit_code}]'
                output += file_limit_status(exit_code, self._max_file_bytes)
            return output
        finally:
            await self._remove_capture(ctx, cwd_file)

    @recoverable
    async def shell(
        self,
        ctx: RunContext[AgentDepsT],
        command: str,
        *,
        mode: CommandMode = 'foreground',
        timeout: float | None = None,
    ) -> str:
        """Run a command that keeps running after this call and after the agent run.

        Foreground waits up to `timeout` seconds (at most 270) for the command to
        exit, then returns handles to the same still-running process. Background
        returns the handles immediately. Both return the PID, the path of the
        combined stdout/stderr log, and the path of a JSON status file whose
        `exit_code` is null until the command exits. Read those files with your
        other tools and stop the process with `kill` and the returned PID; no
        notification arrives when it finishes.

        Args:
            ctx: The current agent run context.
            command: The shell command to run.
            mode: `foreground` to wait, `background` to return at once.
            timeout: Seconds to wait in foreground mode (default: the configured timeout).
        """
        self._check_command(command)
        return await run_persistent_command(
            ctx,
            command,
            base=await self._jobs_base(ctx),
            cwd=await ctx.workspace.working_dir(),
            env=self._resolve_env(),
            mode=mode,
            timeout=self._default_timeout if timeout is None else timeout,
        )

    @recoverable
    async def start_command(self, ctx: RunContext[AgentDepsT], command: str) -> str:
        """Start a long-running command in the background (e.g. a server or watcher).

        The command keeps running, after this run too, until it exits or
        `stop_command(command_id)` stops it; call `stop_command` when done to
        also remove its output files.

        Args:
            ctx: The current agent run context.
            command: The shell command to run in the background.

        Returns:
            A message containing the unique command ID for later check/stop calls.
        """
        self._check_command(command)
        # Activity retries share the same tool-call identity, not just the run:
        # separate starts in one run still need distinct job directories.
        job_id = None
        if ctx.run_id is not None and ctx.tool_call_id is not None:
            job_id = hashlib.sha256(f'{ctx.run_id}:{ctx.tool_call_id}'.encode()).hexdigest()[:32]
        job = await Job.launch(
            ctx.workspace,
            command,
            base=await self._jobs_base(ctx),
            cwd=await self._cwd_for(ctx),
            env=self._resolve_env(),
            combined=False,
            file_limit=self._max_file_bytes,
            job_id=job_id,
        )
        return f'Started background command: {command!r}\nID: {posixpath.basename(job.directory)}'

    async def _read_bg_output(self, job: Job) -> tuple[str, str]:
        """The retained tail of a background command's stdout and stderr logs."""
        limit = self._max_output_chars * _OUTPUT_BYTES_PER_CHAR
        stdout = await job.tail(job.output_path, limit)
        stderr = await job.tail(job.stderr_path, limit)
        return stdout.decode('utf-8', errors='replace'), stderr.decode('utf-8', errors='replace')

    @recoverable
    async def check_command(self, ctx: RunContext[AgentDepsT], command_id: str) -> str:
        """Check the status and recent output of a background command.

        Args:
            ctx: The current agent run context.
            command_id: The ID returned by start_command.

        Returns:
            Status and recent output of the background command.
        """
        job = await self._job(ctx, command_id)
        if job is None:
            return f'[Error: unknown command ID {command_id!r}]'

        running, exit_code = await job.status()
        stdout, stderr = await self._read_bg_output(job)

        parts = [
            _labelled(stdout, stderr, empty='(no output yet)'),
            f'[status: {"running" if running else "finished"}]',
        ]
        if exit_code is not None:
            parts.append(f'[exit code: {exit_code}]' + file_limit_status(exit_code, self._max_file_bytes))
        return '\n'.join(parts)

    @recoverable
    async def stop_command(self, ctx: RunContext[AgentDepsT], command_id: str) -> str:
        """Stop a background command and return its final output.

        Args:
            ctx: The current agent run context.
            command_id: The ID returned by start_command.

        Returns:
            Final output and exit status of the stopped command.
        """
        job = await self._job(ctx, command_id)
        if job is None:
            return f'[Error: unknown command ID {command_id!r}]'

        running, exit_code = await job.status()
        if running or job.pgid is not None:
            # The wrapper can publish a finished status while children still occupy its group.
            with anyio.CancelScope(shield=True):
                await job.kill()
                # A group that ignored SIGTERM is killed with SIGKILL, which its wrapper cannot
                # outlive to publish a status: the command is stopped, with no exit code to report.
                _, exit_code = await job.status()

        stdout, stderr = await self._read_bg_output(job)
        await job.cleanup()

        parts = [_labelled(stdout, stderr, empty='(no output)'), '[stopped]']
        if exit_code is not None:
            parts.append(f'[exit code: {exit_code}]' + file_limit_status(exit_code, self._max_file_bytes))
        return '\n'.join(parts)


def _labelled(stdout: str, stderr: str, *, empty: str | None, trailer: str | None = None) -> str:
    """`[stdout]`/`[stderr]` sections, `empty` when both are blank, and an optional last line."""
    sections: list[str] = []
    if stdout:
        sections.append(f'[stdout]\n{stdout}')
    if stderr:
        sections.append(f'[stderr]\n{stderr}')
    if not sections and empty is not None:
        sections.append(empty)
    if trailer is not None:
        sections.append(trailer)
    return '\n'.join(sections)
