"""Command policy shared by the shell tools: retryable failures and interactive-command detection."""

from __future__ import annotations

import errno
import functools
import re
from collections.abc import Awaitable, Callable
from typing import Concatenate, ParamSpec, TypeVar

from pydantic_ai.exceptions import ModelRetry

_P = ParamSpec('_P')
_ToolsetT = TypeVar('_ToolsetT')

# Spawning a command fails with a bare `OSError` for causes that have no
# dedicated subclass, and with `FileNotFoundError`/`NotADirectoryError` for
# causes that do. The errno says whose fault it is: these are the model's, and
# it can act on them. Every other errno (EMFILE, ENOMEM) is the host's, and must
# keep aborting the run rather than sending the model into a retry loop it
# can't win.
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


def recoverable(
    fn: Callable[Concatenate[_ToolsetT, _P], Awaitable[str]],
) -> Callable[Concatenate[_ToolsetT, _P], Awaitable[str]]:
    """Convert model-correctable errors into `ModelRetry`.

    pyai only feeds `ModelRetry` back to the model as a retry prompt; any other
    exception propagates and aborts the whole run. A denied command, a command
    the OS refuses to spawn, and a working directory the model's own earlier
    command destroyed are all things the model can recover from, so surface them
    as a retry instead of crashing the agent.
    """

    @functools.wraps(fn)
    async def wrapper(self: _ToolsetT, *args: _P.args, **kwargs: _P.kwargs) -> str:
        try:
            return await fn(self, *args, **kwargs)
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
