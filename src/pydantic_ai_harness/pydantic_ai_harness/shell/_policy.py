"""Command policy shared by the shell tools: retryable failures and interactive-command detection."""

from __future__ import annotations

import errno
import functools
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Concatenate, ParamSpec, TypeVar

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.workspaces import CommandResult, Workspace, WorkspaceCommand, WorkspaceError
from pydantic_ai_harness._workspace import raise_tool_failure

_P = ParamSpec('_P')
_ToolsetT = TypeVar('_ToolsetT')

# Spawning a command fails with a bare `OSError` for causes that have no
# dedicated subclass, and with `FileNotFoundError`/`NotADirectoryError` for
# causes that do. The errno says whose fault it is: these are the model's, and
# it can act on them. Every other errno (EMFILE, ENOMEM) is the host's, and must
# keep aborting the run rather than sending the model into a retry loop it
# can't win. E2BIG can be either's, so `run_model_command` tells them apart.
#
# ENOENT and ENOTDIR reach here only from the working directory, since the
# command string is handed to a shell that always exists -- a command whose own
# executable is missing is reported by that shell on stderr, not by the spawn.
#
# Keyed by `OSError.errno`, which the stdlib types as `int | None`.
_RECOVERABLE_ERRNOS: dict[int | None, str] = {
    errno.ENOENT: 'The working directory no longer exists.',
    errno.ENOTDIR: 'The working directory is no longer a directory.',
}

_TOO_LONG = (
    'The command is too long to start; write the script to a file, with a file tool or in several shorter '
    'commands, then run the file.'
)


async def run_model_command(
    workspace: Workspace,
    build: Callable[[str], WorkspaceCommand],
    command: str,
    *,
    env: Mapping[str, str] | None,
    timeout: float | None,
) -> CommandResult:
    """Run the model's `command` as `build` wraps it, sending an `E2BIG` back to the model only if it's to blame.

    `build` returns the toolset's own invocation around the command: a string for the shell, or an argv.

    The OS refuses to start a command whose arguments and environment together are too large (`E2BIG`),
    so the errno alone does not say whether the model's command or the application's environment is to
    blame. The same invocation and environment without the model's command does, given to `true`, which
    ignores its arguments: when that starts, the model can fix the failure by sending less, so it is asked
    to. When it fails as well, the original error propagates, as an unwinnable retry would only postpone
    the end of the run.
    """
    invocation = build(command)
    try:
        return await workspace.run(invocation, shell=isinstance(invocation, str), env=env, timeout=timeout)
    except OSError as error:
        if error.errno != errno.E2BIG:
            raise
        without = build('')
        try:
            await workspace.run(
                ['true', *([without] if isinstance(without, str) else without)], env=env, timeout=timeout
            )
        except OSError:
            raise error from None
        raise ModelRetry(_TOO_LONG) from error


def recoverable(
    fn: Callable[Concatenate[_ToolsetT, _P], Awaitable[str]],
) -> Callable[Concatenate[_ToolsetT, _P], Awaitable[str]]:
    """Convert model-correctable errors into `ModelRetry`, and workspace refusals into `ToolFailed`.

    pyai only feeds `ModelRetry` back to the model as a retry prompt; any other
    exception propagates and aborts the whole run. A denied command, a command
    the OS refuses to spawn, and a working directory the model's own earlier
    command destroyed are all things the model can recover from, so surface them
    as a retry instead of crashing the agent. A read-only workspace, an expired
    control deadline, or another deliberate workspace failure is reported as a
    failed call instead (see `raise_tool_failure`); a workspace that is gone
    still ends the run.
    """

    @functools.wraps(fn)
    async def wrapper(self: _ToolsetT, *args: _P.args, **kwargs: _P.kwargs) -> str:
        try:
            return await fn(self, *args, **kwargs)
        # Before `PermissionError` and `OSError`: `WorkspaceReadOnlyError` is a `PermissionError`
        # and `WorkspaceTimeoutError` a `TimeoutError`, and neither is a policy denial.
        except WorkspaceError as e:
            raise_tool_failure(e)
        except PermissionError as e:
            raise ModelRetry(str(e)) from e
        except OSError as e:
            reason = _RECOVERABLE_ERRNOS.get(e.errno)
            if reason is None:
                raise
            # `str(e)` embeds the absolute host path; the reason alone doesn't.
            raise ModelRetry(reason) from e

    return wrapper


def is_interactive_command(command: str) -> bool:
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
