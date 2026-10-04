"""Private helpers shared by workspace provider backends."""

from __future__ import annotations

import asyncio
import math
import posixpath
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

import anyio

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import WorkspaceCommand, WorkspaceTimeoutError

# Optional, not a dependency: it used to arrive only transitively, and AnyIO dropped it in 4.12.
try:
    import sniffio as _sniffio
except ModuleNotFoundError:  # pragma: no cover - exercised by the clean-import test in a subprocess
    _sniffio = None


def safe_credential_reason(error: Exception) -> str:
    """Classify a provider credential rejection without copying its possibly secret-bearing text."""
    message = str(error).lower()
    # Provider errors can embed the rejected token; emit only fixed labels.
    if 'malformed' in message or 'invalid format' in message:
        return 'API key is malformed'
    if 'expired' in message:
        return 'Credential expired'
    if 'missing' in message or 'not configured' in message:
        return 'Credential missing'
    return 'Credentials rejected'


def running_on_asyncio() -> bool:
    """Whether the caller runs on asyncio rather than Trio.

    Inspired by AnyIO's private `current_async_library`. With `sniffio` installed, ask it: Trio records itself
    there, so the answer holds even for Trio guest mode on an asyncio loop. Without it, Trio cannot be running,
    because Trio depends on `sniffio`, so a running asyncio loop means asyncio. If Trio ever drops `sniffio`,
    only guest mode would be misread, as AnyIO would misread it too.
    """
    if _sniffio is None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        return True
    try:
        return _sniffio.current_async_library() == 'asyncio'
    except _sniffio.AsyncLibraryNotFoundError:
        return False


# asyncio holds only weak references to tasks, so a detached stop needs a strong one until it ends.
_pending_stops: set[asyncio.Task[None]] = set()
# Extra wait past the stop grace so a stop cut off at the deadline can finish its cleanup.
_STOP_SETTLE = 0.5


async def stop_shielded(stop: Callable[[], Awaitable[object]], *, grace: float = 2.0) -> None:
    """Attempt to stop a command under cancellation without cancelling its shared sandbox."""

    async def best_effort_stop() -> None:
        try:
            await stop()
        except Exception:
            # Preserve the command timeout/cancellation if a provider's stop request fails.
            pass

    if running_on_asyncio():

        async def bounded_stop() -> None:
            with anyio.move_on_after(grace, shield=True):
                await best_effort_stop()

        # asyncio-only on purpose: a native task.cancel() bypasses AnyIO shields and would cancel an
        # AnyIO task-group child with its caller, and AnyIO has no detached task. This child owns
        # its own grace even if a second cancel interrupts the caller waiting for it.
        child = asyncio.create_task(bounded_stop())
        _pending_stops.add(child)
        child.add_done_callback(_pending_stops.discard)
        # The child already bounds itself by `grace`; the margin lets its cleanup (e.g. a
        # provider's "may still be running" log) finish before we return, instead of racing it.
        try:
            with anyio.move_on_after(grace + _STOP_SETTLE, shield=True):
                await asyncio.shield(child)
        except asyncio.CancelledError:
            # A native cancel thrown into this wait takes as `__context__` the exception each frame the
            # throw resumes is handling, here the cancellation that started this stop. AnyIO cancel scopes
            # follow `__context__` and would claim it as their own. Re-raised from a step that resumed
            # normally, it keeps no such context. The shield stops the caller's cancelled scope from
            # cancelling every checkpoint; a repeated native cancel folds into this one.
            with anyio.CancelScope(shield=True):
                while True:
                    try:
                        await asyncio.sleep(0)
                    except asyncio.CancelledError:
                        continue
                    break
            raise
    else:
        with anyio.move_on_after(grace, shield=True):
            async with anyio.create_task_group() as group:
                group.start_soon(best_effort_stop)


@asynccontextmanager
async def command_deadline(
    timeout: float | None,
    *,
    stop: Callable[[], Awaitable[object]],
    output: Callable[[], tuple[str, str]] = lambda: ('', ''),
) -> AsyncGenerator[None, None]:
    """Bound only the command phase, after sandbox acquisition; stop on timeout or cancellation."""
    with anyio.move_on_after(timeout) as scope:
        try:
            yield
        except BaseException:
            await stop_shielded(stop)
            raise
    if scope.cancelled_caught:
        stdout, stderr = output()
        raise WorkspaceTimeoutError(f'Command timed out after {timeout:g} seconds', stdout=stdout, stderr=stderr)


def absolute_path(name: str, value: str | None) -> str | None:
    """Validate an absolute POSIX path without changing symlink traversal."""
    if value is None:
        return None
    if not posixpath.isabs(value):
        raise ValueError(f'{name} must be an absolute workspace path or None, got {value!r}.')
    return value


def command_argv(command: WorkspaceCommand, shell: bool) -> list[str]:
    """The argv that runs `command`, with the same `shell` rules as core's local backend.

    A shell string runs under `/bin/sh -c`; an argv sequence runs as given. An empty argv is
    refused: a provider that quotes it with `shlex.join` would get `''`, which a shell runs as a
    successful no-op.
    """
    if isinstance(command, str):
        if not shell:
            raise TypeError('a string command requires shell=True; pass an argv sequence otherwise')
        return ['/bin/sh', '-c', command]
    if isinstance(command, bytes):
        raise TypeError('a bytes command is not supported; pass a string or argv sequence')
    if shell:
        raise TypeError('an argv sequence cannot be combined with shell=True; pass a single command string')
    if not command:
        raise TypeError('an argv sequence needs at least the program to run')
    for argument in command:
        if type(argument) is not str:
            raise TypeError('argv elements must be strings')
        if '\x00' in argument:
            raise ValueError('argv elements must not contain NUL bytes')
    return list(command)


def check_working_dir(value: str | None) -> None:
    """Raise `UserError` unless a provider's `working_dir` is an absolute POSIX path or `None`."""
    if value is not None and not posixpath.isabs(value):
        raise UserError(f'working_dir must be an absolute POSIX path or None, got {value!r}.')


def check_integer(name: str, value: int | None, *, minimum: int = 1, optional: bool = False) -> None:
    """Raise `UserError` unless `value` is an integer of at least `minimum`, or `None` when `optional`.

    `bool` is rejected: it is an `int` subclass, but `True` is never a meant count.
    """
    if (type(value) is int and value >= minimum) or (value is None and optional):
        return
    raise UserError(f'{name} must be an integer of at least {minimum}{" or None" if optional else ""}, got {value!r}.')


def check_timeout(timeout: float | None) -> None:
    """Raise `ValueError` unless a command `timeout` is a positive finite number or `None`, as core's backends do."""
    if timeout is not None and (not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError(f'timeout must be a positive finite number or None, got {timeout!r}.')
