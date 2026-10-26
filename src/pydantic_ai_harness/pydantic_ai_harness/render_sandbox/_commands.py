"""Command execution for the public Pydantic workspace protocol.

Only Render's public asynchronous exec API is used. A small remote supervisor
supplies process-group identity, byte-preserving output and direct-child exit
semantics that are not exposed by Render's streaming endpoint itself.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import uuid
from collections.abc import AsyncGenerator, Mapping
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any, cast

import anyio
from render import RenderAsync
from render.client.errors import ClientError
from render.experimental.sandbox import SandboxExecEvent, SandboxExecExit

from pydantic_ai.workspaces import (
    CommandResult,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness._workspace_provider import stop_shielded

_CLAIM_ROOT = '/tmp'
# Linux limits each exec argument, and Render passes the whole launch as one `bash -c` argument.
_MAX_LAUNCH_BYTES = 126_000
# After the direct child exits its bytes are already in the pipes; a background child still writing to
# an inherited pipe gets this long before the result returns, as core's local backend allows.
_DRAIN_GRACE = 2.0
# Like core's local backend: past 10 MiB the command is stopped, and the error carries a 64 KiB preview.
_MAX_OUTPUT_BYTES = 10 * 1024 * 1024
_OUTPUT_PREVIEW_BYTES = 64 * 1024
_STOP_TIMEOUT = 12.0


def _launch(source: str, operation: str, config: Mapping[str, object]) -> str:
    payload = json.dumps(dict(config), separators=(',', ':'), ensure_ascii=True)
    # -I keeps a module in the working directory, such as a project's json.py, from shadowing the stdlib.
    command = shlex.join(['python3', '-I', '-c', source, operation, payload])
    if len(command.encode()) > _MAX_LAUNCH_BYTES:
        raise ValueError('Command and environment exceed the remote exec argument limit')
    return command


@dataclass
class _Capture:
    """Parses the supervisor's JSON-line frames; only the supervisor writes this stream."""

    stdout: bytearray = field(default_factory=bytearray)
    stderr: bytearray = field(default_factory=bytearray)
    buffer: str = ''
    terminal: str | None = None
    exit_code: int | None = None

    def text(self, limit: int | None = None) -> tuple[str, str]:
        return (
            self.stdout[:limit].decode('utf-8', errors='replace'),
            self.stderr[:limit].decode('utf-8', errors='replace'),
        )

    def overflow(self) -> WorkspaceOutputLimitError:
        stdout, stderr = self.text(_OUTPUT_PREVIEW_BYTES)
        return WorkspaceOutputLimitError(
            'Command output exceeded 10 MiB; redirect output to a file',
            limit=_MAX_OUTPUT_BYTES,
            stdout=stdout,
            stderr=stderr,
        )

    def feed(self, data: str) -> None:
        *lines, self.buffer = (self.buffer + data).split('\n')
        for line in lines:
            try:
                self.frame(json.loads(line))
            except (ValueError, TypeError, KeyError) as error:
                raise WorkspaceError('Invalid command supervisor frame') from error

    def frame(self, frame: dict[str, Any]) -> None:
        kind = frame['kind']
        if kind == 'output':
            (self.stdout if frame['stream'] == 'stdout' else self.stderr).extend(base64.b64decode(frame['data']))
        elif kind == 'exit':
            self.terminal, self.exit_code = kind, int(frame['code'])
        elif kind in ('limit', 'timeout', 'stopped', 'fenced', 'cwd'):
            self.terminal = kind
        else:
            raise WorkspaceError(f'Command supervisor reported {kind!r}')


async def _consume(client: RenderAsync, sandbox_id: str, launch: str, capture: _Capture) -> None:
    # The SDK annotates an iterator but returns an async generator, closed here if the caller stops early.
    stream = cast(AsyncGenerator[SandboxExecEvent], client.experimental.sandboxes.exec(sandbox_id, launch))
    sdk_exit: int | None = None
    async with aclosing(stream):
        async for event in stream:
            # Only a crashed supervisor writes to stderr, and that is not a valid frame either.
            if isinstance(event, SandboxExecExit):
                sdk_exit = event.exit_code
            else:
                capture.feed(event.data)
    if sdk_exit != 0 or capture.buffer or capture.terminal is None:
        raise WorkspaceError('Command supervisor stream ended without a complete result')


async def _stop(client: RenderAsync | None, sandbox_id: str, command: PreparedCommand) -> None:
    # Its own client when the caller supplied none: the stop can outlive the operation that opened the run's client.
    async with client_scope(client) as scope:
        capture = _Capture()
        await _consume(scope, sandbox_id, command.stop, capture)
        if capture.terminal != 'stopped':
            raise WorkspaceError('Command stop did not acknowledge its fence')


_MISSING_CREDENTIALS = 'Set RENDER_API_KEY and RENDER_WORKSPACE_ID in the environment, or pass a configured `client=`.'


def open_client() -> RenderAsync:
    if not os.environ.get('RENDER_WORKSPACE_ID'):
        raise WorkspaceUnavailableError(_MISSING_CREDENTIALS)
    try:
        return RenderAsync()
    except ValueError as error:
        raise WorkspaceUnavailableError(_MISSING_CREDENTIALS) from error


async def close_client(client: RenderAsync) -> None:
    # Render 1.2 exposes no close on `RenderAsync`; its pool lives on the sandbox API client.
    await client.experimental.sandboxes.client.get_async_httpx_client().aclose()


@asynccontextmanager
async def client_scope(client: RenderAsync | None) -> AsyncGenerator[RenderAsync]:
    """Yield `client`, or one opened from the environment and closed afterwards."""
    if client is not None:
        yield client
        return
    opened = open_client()
    try:
        yield opened
    finally:
        with anyio.CancelScope(shield=True):
            await close_client(opened)


_SOURCE = files('pydantic_ai_harness.render_sandbox').joinpath('_command_runner.py').read_text()


@dataclass(frozen=True)
class PreparedCommand:
    """A command's supervisor launch and stop requests, built before any sandbox exists."""

    launch: str
    stop: str


def prepare_command(
    argv: list[str],
    *,
    working_dir: str,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> PreparedCommand:
    """Build the supervisor requests, raising `ValueError` if they exceed the exec argument limit.

    The backend calls this before acquiring a sandbox, so a command too large to send creates nothing.
    """
    claim = f'{_CLAIM_ROOT}/pydantic-render-command-{uuid.uuid4().hex}'
    config: dict[str, object] = {
        'claim': claim,
        'argv': argv,
        'cwd': working_dir,
        'env': dict(env or {}),
        'timeout': timeout,
        'limit': _MAX_OUTPUT_BYTES,
        'drain': _DRAIN_GRACE,
    }
    return PreparedCommand(
        launch=_launch(_SOURCE, 'run', config),
        stop=_launch(_SOURCE, 'stop', {'claim': claim}),
    )


async def run_command(
    client: RenderAsync,
    sandbox_id: str,
    command: PreparedCommand,
    *,
    timeout: float | None,
    stop_client: RenderAsync | None,
) -> CommandResult:
    """Run once, stopping the command if it may still be running after a timeout, cancellation or stream failure.

    Exceptions from Render propagate unchanged, and no command is replayed.
    """
    capture = _Capture()
    deadline = asyncio.timeout(timeout)
    try:
        async with deadline:
            await _consume(client, sandbox_id, command.launch, capture)
    except BaseException as error:
        # No stop once the supervisor reported the end, or when Render refused the exec outright.
        refused = isinstance(error, ClientError) and error.__cause__ is None
        if capture.terminal is None and not refused:
            await stop_shielded(lambda: _stop(stop_client, sandbox_id, command), grace=_STOP_TIMEOUT)
        if isinstance(error, TimeoutError) and deadline.expired():
            stdout, stderr = capture.text()
            raise WorkspaceTimeoutError('Command deadline exceeded', stdout=stdout, stderr=stderr) from error
        raise
    stdout, stderr = capture.text()
    if capture.terminal == 'limit':
        raise capture.overflow()
    if capture.terminal == 'cwd':
        raise WorkspaceUnavailableError('The working directory no longer exists or is not usable in the sandbox')
    if capture.terminal == 'timeout':
        raise WorkspaceTimeoutError('Command deadline exceeded', stdout=stdout, stderr=stderr)
    if capture.terminal != 'exit' or capture.exit_code is None:
        raise WorkspaceError('Command was fenced before it could start')
    return CommandResult(exit_code=capture.exit_code, stdout=stdout, stderr=stderr)
