"""Command execution for the public Pydantic workspace protocol.

Only Render's public asynchronous exec API is used. A small remote supervisor
supplies process-group identity, byte-preserving output and direct-child exit
semantics that are not exposed by Render's streaming endpoint itself.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import shlex
import uuid
from collections.abc import AsyncGenerator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from importlib.resources import files
from typing import TypeGuard

import anyio
from render import RenderAsync
from render.client.errors import ClientError
from render.experimental.sandbox import SandboxExecOutput

from pydantic_ai.workspaces import (
    CommandResult,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness._workspace_provider import stop_shielded

_CLAIM_ROOT = '/tmp'
_MAX_FRAME_CHARS = 131_072
# Linux limits each exec argument, and Render passes the whole launch as one `bash -c` argument.
_MAX_LAUNCH_BYTES = 126_000
# After the direct child exits its bytes are already in the pipes; a background child still writing to
# an inherited pipe gets this long before the result returns, as core's local backend allows.
_DRAIN_GRACE = 2.0
_MAX_OUTPUT_BYTES = 1_048_576
_STOP_TIMEOUT = 12.0


def _launch(source: str, operation: str, config: Mapping[str, object]) -> str:
    payload = json.dumps(dict(config), separators=(',', ':'), ensure_ascii=True)
    # -I keeps a module in the working directory, such as a project's json.py, from shadowing the stdlib.
    command = shlex.join(['python3', '-I', '-c', source, operation, payload])
    if len(command.encode()) > _MAX_LAUNCH_BYTES:
        raise ValueError('Command and environment exceed the remote exec argument limit')
    return command


def _json_object(value: object) -> TypeGuard[dict[str, object]]:
    # Callers obtain objects from json.loads, which produces string dictionary keys.
    return isinstance(value, dict)


def _object(value: object) -> dict[str, object]:
    if not _json_object(value):
        raise WorkspaceError('Invalid command supervisor frame')
    # These objects come from JSON decoding, which already guarantees text keys.
    return value


@dataclass
class _Capture:
    nonce: str
    limit: int
    stdout: bytearray = field(default_factory=bytearray)
    stderr: bytearray = field(default_factory=bytearray)
    buffer: str = ''
    terminal: str | None = None
    exit_code: int | None = None

    def text(self) -> tuple[str, str]:
        return self.stdout.decode('utf-8', errors='replace'), self.stderr.decode('utf-8', errors='replace')

    def overflow(self) -> WorkspaceOutputLimitError:
        stdout, stderr = self.text()
        return WorkspaceOutputLimitError(
            'Command output exceeded 1 MiB; redirect output to a file',
            limit=self.limit,
            stdout=stdout,
            stderr=stderr,
        )

    def feed(self, data: str) -> None:
        pieces = data.split('\n')
        for index, piece in enumerate(pieces):
            self.buffer += piece
            if len(self.buffer) > _MAX_FRAME_CHARS:
                raise WorkspaceError('Command supervisor frame exceeded its bound')
            if index < len(pieces) - 1:
                line, self.buffer = self.buffer, ''
                if line:
                    try:
                        parsed: object = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise WorkspaceError('Invalid command supervisor JSON') from exc
                    self.frame(_object(parsed))

    def frame(self, frame: dict[str, object]) -> None:
        if frame.get('nonce') != self.nonce:
            raise WorkspaceError('Command supervisor identity does not match')
        kind = frame.get('kind')
        if self.terminal is not None:
            raise WorkspaceError('Command supervisor sent data after its terminal frame')
        if kind == 'started':
            return
        if kind == 'output':
            stream, encoded = frame.get('stream'), frame.get('data')
            if stream not in ('stdout', 'stderr') or not isinstance(encoded, str):
                raise WorkspaceError('Invalid command output frame')
            try:
                data = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise WorkspaceError('Invalid base64 command output') from exc
            remaining = max(0, self.limit - len(self.stdout) - len(self.stderr))
            (self.stdout if stream == 'stdout' else self.stderr).extend(data[:remaining])
            if len(data) > remaining:
                raise self.overflow()
            return
        if kind in ('exit', 'limit', 'timeout', 'stopped', 'fenced', 'cwd'):
            assert isinstance(kind, str)
            self.terminal = kind
            if kind == 'exit':
                code = frame.get('code')
                if type(code) is not int:
                    raise WorkspaceError('Missing direct command exit code')
                self.exit_code = code
            return
        if kind == 'error':
            raise WorkspaceError('Remote command supervisor failed')
        raise WorkspaceError('Unknown command supervisor frame')


async def _consume(client: RenderAsync, sandbox_id: str, launch: str, capture: _Capture) -> None:
    stream = client.experimental.sandboxes.exec(sandbox_id, launch)
    sdk_exit: int | None = None
    async with AsyncExitStack() as stack:
        if isinstance(stream, AsyncGenerator):
            stack.push_async_callback(stream.aclose)
        async for event in stream:
            if isinstance(event, SandboxExecOutput):
                if event.stream != 'stdout':
                    # The supervisor reserves stderr for interpreter-level failures.
                    raise WorkspaceError('Command supervisor emitted unexpected stderr')
                capture.feed(event.data)
            else:
                sdk_exit = event.exit_code
    if sdk_exit != 0 or capture.buffer or capture.terminal is None:
        raise WorkspaceError('Command supervisor stream ended without a complete result')


async def _stop(client: RenderAsync | None, sandbox_id: str, command: PreparedCommand) -> None:
    # Its own client when the caller supplied none: the stop can outlive the operation that opened the run's client.
    async with client_scope(client) as scope:
        capture = _Capture(command.nonce, 4096)
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

    nonce: str
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
    nonce = uuid.uuid4().hex
    claim = f'{_CLAIM_ROOT}/pydantic-render-command-{nonce}'
    config: dict[str, object] = {
        'claim': claim,
        'nonce': nonce,
        'argv': argv,
        'cwd': working_dir,
        'env': dict(env or {}),
        'timeout': timeout,
        'limit': _MAX_OUTPUT_BYTES,
        'drain': _DRAIN_GRACE,
    }
    return PreparedCommand(
        nonce=nonce,
        launch=_launch(_SOURCE, 'run', config),
        stop=_launch(_SOURCE, 'stop', {'claim': claim, 'nonce': nonce}),
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
    capture = _Capture(command.nonce, _MAX_OUTPUT_BYTES)
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
