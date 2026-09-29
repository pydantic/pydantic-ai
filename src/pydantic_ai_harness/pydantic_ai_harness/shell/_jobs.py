"""Detached jobs: commands started inside the run's workspace that outlive the call that started them.

`Job.launch` runs a short POSIX `sh` launcher through `workspace.run`. The launcher starts a
wrapper shell in its own session (`setsid` when the workspace has it, else `nohup` in the
launcher's process group) and returns once the wrapper is running. The wrapper publishes `status.json` --
`{"pid": <wrapper pid>, "exit_code": null}` -- before running the command, appends the
command's output to log files next to it, and publishes the exit code when the command ends.
Each job's files live in one owner-only directory below `.pydantic-ai-harness/shell` in the
workspace's working directory, never on the host unless the workspace is the host. Status and output are read, and the process group
is signalled, through the same workspace.
"""

from __future__ import annotations

import base64
import json
import posixpath
import shlex
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field

import anyio

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.workspaces import Workspace, WorkspaceError
from pydantic_ai_harness.shell._limits import LIMIT_FUNCTION

CONTROL_TIMEOUT = 30.0
"""Deadline in seconds for one control command (launch, read, signal).

These return at once in a healthy workspace; the deadline keeps a wedged one from hanging the tool call.
"""

_KILL_GRACE_PERIOD = 2.0
"""Seconds a job gets to exit after `SIGTERM` before its process group is sent `SIGKILL`."""

POLL_MIN = 0.05
POLL_MAX = 1.0
"""Bounds of the backoff between polls of a job's status, so a remote workspace is not asked every 50 ms."""

_WRAPPER = f"""{LIMIT_FUNCTION}
dir=$1
publish() {{
  printf '{{"pid": %s, "exit_code": %s}}' "$$" "$1" > "$dir/status.tmp" && mv -f "$dir/status.tmp" "$dir/status.json"
}}
unset __harness_stopped
trap __harness_stopped=1 TERM
publish null
while [ -z "$__harness_stopped" ] && [ ! -e "$dir/launch.ready" ]; do
  kill -0 "$5" 2> /dev/null || [ -e "$dir/launch.ready" ] || __harness_stopped=1
  sleep 0.1 2> /dev/null || [ -n "$__harness_stopped" ] || sleep 1
done
if [ "$2" = combined ]; then out="$dir/output.log"; err="$dir/output.log"; else out="$dir/stdout.log"; err="$dir/stderr.log"; fi
if [ -n "$4" ]; then
  __harness_limit_files "$4" || {{ echo 'Unable to apply max_file_bytes.' >> "$err"; publish 1; exit 1; }}
fi
(
  [ -z "$__harness_stopped" ] && [ ! -e "$dir/stop" ] || exit 143
  exec sh -c "$3"
) < /dev/null >> "$out" 2>> "$err"
publish $?
"""
"""The job's supervisor: arguments are the job directory, the log mode, the command, the file limit, and the launcher's PID.

The `TERM` trap lets the wrapper outlive a `SIGTERM` sent to the whole group long enough to
publish the command's exit status; the command itself runs with default signal handling, since
a trap with an action is reset in a child. Stopping the group therefore still records how the
command ended (`143` for `SIGTERM`); only a `SIGKILL` escalation leaves `exit_code` null.
The trap also records the signal, so a job stopped before its command starts (the wrapper waits
for `launch.ready` first) never starts it and publishes `143` instead. A launcher that dies
before `launch.ready` (a cancelled or timed-out launch) counts as a stop too, so the wrapper
never waits for it forever.

A `SIGTERM` that lands while the command's subshell is being forked, before its default signal
handling is in place, is lost. `Job.kill` therefore creates the job's `stop` file before
signalling, and the subshell checks for it once a signal would end it.
"""

_SIGNAL_SCRIPT = 'true 2> /dev/null > "$3"; kill -s "$1" -- "$2"'
"""Send signal `$1` to `$2`, first creating the stop file `$3` the wrapper checks before starting the command.

`true` rather than `:`, as a failed redirection on a special builtin exits the shell (the job
directory may be gone).
"""

_REQUIRED_TOOLS = ('mv', 'base64')
"""Executables the job files need: the wrapper publishes its status with `mv`, and logs are read back through `base64`."""

_MISSING_TOOLS = 122
"""The launcher's exit status when a tool in `_REQUIRED_TOOLS` is not on the workspace's `PATH`."""

_LAUNCHER = f"""missing=
for tool in {' '.join(_REQUIRED_TOOLS)}; do command -v "$tool" > /dev/null 2>&1 || missing="$missing $tool"; done
if [ -n "$missing" ]; then echo $missing; exit {_MISSING_TOOLS}; fi
if [ "$exclusive" = 1 ]; then
  (umask 077 && mkdir "$dir") || exit 126
else
  (umask 077 && mkdir -p "$dir") || exit 125
fi
if [ "$mode" = combined ]; then : > "$dir/output.log"; else : > "$dir/stdout.log"; : > "$dir/stderr.log"; fi
if command -v setsid > /dev/null 2>&1; then
  setsid sh -c "$wrapper" sh "$dir" "$mode" "$cmd" "$limit" $$ < /dev/null > /dev/null 2>&1 &
  pid=$!; group=$pid
else
  nohup sh -c "$wrapper" sh "$dir" "$mode" "$cmd" "$limit" $$ < /dev/null > /dev/null 2>&1 &
  pid=$!; group=-
  if [ "$(ps -o pgid= -p $$ 2> /dev/null | tr -d ' ')" = "$$" ]; then group=$$; fi
fi
while [ ! -e "$dir/status.json" ]; do sleep 0.1 2> /dev/null || sleep 1; done
: > "$dir/launch.ready"
echo "$pid $group" > "$dir/handle"
echo "$pid $group"
"""
"""Start the wrapper detached and print `<pid> <process group or ->`, also kept in the job's `handle` file.

It first checks for `_REQUIRED_TOOLS` with the shell's `command -v` builtin, which costs no extra
process, and prints the missing ones and exits `_MISSING_TOOLS` instead of launching.

Before returning, the launcher waits for the wrapper to publish its status after `setsid`
has detached it: some workspaces kill the launching process group as soon as it exits.
The workspace run is bounded by `CONTROL_TIMEOUT`. `sleep 1` stands in where `sleep`
takes only whole seconds.

Without `setsid`, the job stays in the launcher's process group, which is only the job's to
signal when the workspace started the launcher as a group leader (the local workspace starts
every command in a new session). Otherwise it is reported as `-` and only the wrapper's PID is
signalled, which does not stop the command.
"""


@dataclass(kw_only=True)
class Job:
    """One detached command and the files that describe it inside the workspace."""

    workspace: Workspace
    directory: str
    pid: int
    """The wrapper's PID, which is also the PID in `status.json`."""
    pgid: int | None
    """The process group to signal, or `None` when only the wrapper's PID is safe to signal."""
    combined: bool
    """Whether stdout and stderr share `output.log`, rather than `stdout.log` and `stderr.log`."""
    _final_status: str | None = field(default=None, init=False, repr=False)

    @classmethod
    async def launch(
        cls,
        workspace: Workspace,
        command: str,
        *,
        base: str,
        cwd: str,
        env: Mapping[str, str] | None,
        combined: bool,
        file_limit: int | None = None,
        job_id: str | None = None,
    ) -> Job:
        directory = posixpath.join(base, job_id or uuid.uuid4().hex)
        mode = 'combined' if combined else 'separate'
        assignments = ' '.join(
            f'{name}={shlex.quote(value)}'
            for name, value in (
                ('dir', directory),
                ('exclusive', '1' if job_id is not None else '0'),
                ('mode', mode),
                ('cmd', command),
                ('limit', '' if file_limit is None else str(file_limit)),
                ('wrapper', _WRAPPER),
            )
        )
        result = await workspace.run(
            f'cd {shlex.quote(cwd)} || exit\n{assignments}\n{_LAUNCHER}', shell=True, env=env, timeout=CONTROL_TIMEOUT
        )
        if result.exit_code == _MISSING_TOOLS:
            # Without these the launcher would wait out `CONTROL_TIMEOUT` for a status that is never
            # published, or a log read would fail looking like the command's own output.
            needed = ' and '.join(f'`{tool}`' for tool in _REQUIRED_TOOLS)
            lacking = ' and '.join(f'`{tool}`' for tool in result.stdout.split())
            raise WorkspaceError(f'Shell needs {needed} on PATH in the workspace; this image lacks {lacking}.')
        if job_id is not None and result.exit_code == 126:
            # An existing claim can be an in-flight launch or a lost reply. Never
            # spawn a second process in either case; a later retry can attach.
            if existing := await cls.attach(workspace, directory, combined=combined):
                return existing
            raise ModelRetry('Background command launch is pending; retry with the same tool call ID.')
        fields = result.stdout.split()
        if result.exit_code != 0 or len(fields) != 2 or not fields[0].isdigit():
            detail = result.stderr.strip() or f'launcher output {result.stdout.strip()!r}'
            raise ModelRetry(f'Shell supervisor exited with {result.exit_code}: {detail}')
        pgid = int(fields[1]) if fields[1].isdigit() else None
        return cls(workspace=workspace, directory=directory, pid=int(fields[0]), pgid=pgid, combined=combined)

    @classmethod
    async def attach(cls, workspace: Workspace, directory: str, *, combined: bool) -> Job | None:
        """The job launched into `directory`, from its `handle` file, or `None` when there is none."""
        try:
            fields = (await workspace.read_bytes(posixpath.join(directory, 'handle'))).decode(errors='replace').split()
        except WorkspaceError:
            raise
        except OSError:  # missing, a directory, or unreadable: no job to attach to
            return None
        if len(fields) != 2 or not fields[0].isdigit():
            return None
        pgid = int(fields[1]) if fields[1].isdigit() else None
        return cls(workspace=workspace, directory=directory, pid=int(fields[0]), pgid=pgid, combined=combined)

    @property
    def output_path(self) -> str:
        """The combined log, or stdout's log when the streams are kept apart."""
        return posixpath.join(self.directory, 'output.log' if self.combined else 'stdout.log')

    @property
    def stderr_path(self) -> str:
        return posixpath.join(self.directory, 'output.log' if self.combined else 'stderr.log')

    @property
    def status_path(self) -> str:
        return posixpath.join(self.directory, 'status.json')

    @property
    def stop_command(self) -> str:
        """The command a model runs to stop the whole job."""
        return f'kill -- -{self.pgid}' if self.pgid is not None else f'kill {self.pid}'

    async def status_text(self) -> str | None:
        """`status.json` as the wrapper published it, or `None` before the first publication."""
        if self._final_status is not None:
            return self._final_status
        try:
            text = (await self.workspace.read_bytes(self.status_path)).decode('utf-8', errors='replace')
        except FileNotFoundError:
            return None
        try:
            exit_code = json.loads(text)['exit_code']
        except (ValueError, KeyError, TypeError):
            exit_code = None
        # A published exit code never changes; reuse it during drain and final rendering.
        if isinstance(exit_code, int) and not isinstance(exit_code, bool):
            self._final_status = text
        return text

    async def status(self) -> tuple[bool, int | None]:
        """`(running, exit_code)`; a job whose status is not yet published counts as running."""
        text = await self.status_text()
        try:
            exit_code = json.loads(text)['exit_code'] if text is not None else None
        except (ValueError, KeyError, TypeError):
            exit_code = None
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            return True, None
        return False, exit_code

    async def size(self, path: str) -> int:
        try:
            size = (await self.workspace.stat(path)).size
        except FileNotFoundError:
            return 0
        if size is not None:
            return size
        # The backend's `stat` reports no size; count the bytes in the workspace instead.
        result = await self.workspace.run(f'wc -c < {shlex.quote(path)}', shell=True, timeout=CONTROL_TIMEOUT)
        if result.exit_code != 0 or not result.stdout.strip().isdigit():
            raise WorkspaceError(result.stderr.strip() or f'Unable to size job log {path!r}.')
        return int(result.stdout)

    async def read(self, path: str, offset: int, length: int) -> bytes:
        """Up to `length` bytes of `path` from `offset`, read inside the workspace so only they cross the wire."""
        if length <= 0:
            return b''
        quoted = shlex.quote(path)
        result = await self.workspace.run(
            # A pipeline reports only `base64`'s status, so check the log is readable before it.
            f'test -f {quoted} || exit 66; test -r {quoted} || exit 67; '
            f'tail -c +{offset + 1} {quoted} | head -c {length} | base64',
            shell=True,
            timeout=CONTROL_TIMEOUT,
        )
        if result.exit_code == 66:
            return b''
        if result.exit_code != 0:
            raise WorkspaceError(result.stderr.strip() or f'Unable to read job log {path!r}.')
        return base64.b64decode(result.stdout)

    async def tail(self, path: str, max_bytes: int) -> bytes:
        """The last `max_bytes` bytes of `path`."""
        size = await self.size(path)
        start = max(0, size - max_bytes)
        return await self.read(path, start, size - start)

    async def kill(self) -> None:
        """Stop the job's process group: `SIGTERM`, then `SIGKILL` if it is still running after the grace period."""
        if not await self._signal('TERM'):
            return
        if self.pgid is None:
            return
        # The wrapper's status says nothing about children left in its process group.
        # When it is already finished, avoid waiting a grace period for zombie members.
        if not (await self.status())[0]:
            if await self._signal('0'):
                await self._signal('KILL')
            return
        interval = POLL_MIN
        with anyio.move_on_after(_KILL_GRACE_PERIOD):
            while await self._signal('0'):
                await anyio.sleep(interval)
                interval = min(interval * 2, POLL_MAX)
            return
        await self._signal('KILL')

    async def _signal(self, name: str) -> bool:
        """Whether the signal reached a process; a job that already exited is not an error.

        The shell's `kill` builtin sends it, so no `kill` executable is needed: slim images
        such as Debian's ship none. A failure other than "no such process" raises, rather than
        reporting a job stopped that may still be running.
        """
        target = f'-{self.pgid}' if self.pgid is not None else str(self.pid)
        result = await self.workspace.run(
            ['sh', '-c', _SIGNAL_SCRIPT, 'kill', name, target, posixpath.join(self.directory, 'stop')],
            timeout=CONTROL_TIMEOUT,
        )
        if result.exit_code == 0:
            return True
        if 'no such process' in result.stderr.lower():
            return False
        raise WorkspaceError(result.stderr.strip() or f'Unable to send SIG{name} to job {self.pid}.')

    async def cleanup(self) -> None:
        """Remove the job's directory from the workspace."""
        try:
            await self.workspace.remove(self.directory)
        except FileNotFoundError:
            pass
