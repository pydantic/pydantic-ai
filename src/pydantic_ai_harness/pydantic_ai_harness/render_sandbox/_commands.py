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
import shlex
import uuid
from collections.abc import AsyncGenerator, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from importlib.resources import files
from typing import TypeGuard

import anyio
from render import RenderAsync
from render.experimental.sandbox import SandboxExecExit, SandboxExecOutput

from pydantic_ai.workspaces import (
    CommandResult,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceTimeoutError,
)

_CLAIM_ROOT = '/tmp'
_MAX_FRAME_CHARS = 131_072
# Linux limits each exec argument, and Render passes the whole launch as one `bash -c` argument.
_MAX_LAUNCH_BYTES = 126_000
# After the direct child exits its bytes are already in the pipes; a background child still writing to
# an inherited pipe gets this long before the result returns, as core's local backend allows.
_DRAIN_GRACE = 2.0


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
            'Command output exceeded max_output_bytes; redirect output to a file',
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
        if kind in ('exit', 'limit', 'timeout', 'stopped', 'fenced'):
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
                if sdk_exit is not None:
                    raise WorkspaceError('Render output followed its terminal exit')
                if event.stream != 'stdout':
                    # The supervisor reserves stderr for interpreter-level failures.
                    raise WorkspaceError('Command supervisor emitted unexpected stderr')
                capture.feed(event.data)
            elif type(event) is SandboxExecExit:
                if sdk_exit is not None or type(event.exit_code) is not int:
                    raise WorkspaceError('Invalid Render exec terminal event')
                sdk_exit = event.exit_code
            else:
                raise WorkspaceError('Invalid Render exec event')
    if sdk_exit != 0 or capture.buffer or capture.terminal is None:
        raise WorkspaceError('Command supervisor stream ended without a complete result')


async def _stop_shielded(
    client: RenderAsync, sandbox_id: str, launch: str, nonce: str, stop_timeout: float
) -> asyncio.CancelledError | None:
    async def stop() -> None:
        async with asyncio.timeout(stop_timeout):
            capture = _Capture(nonce, 4096)
            await _consume(client, sandbox_id, launch, capture)
            if capture.terminal != 'stopped':
                raise WorkspaceError('Command stop did not acknowledge its fence')

    # The child task owns the timeout. Neither AnyIO level cancellation nor
    # repeated native Task.cancel calls may abandon an already-dispatched stop.
    task = asyncio.create_task(stop())
    interrupted: asyncio.CancelledError | None = None
    with anyio.CancelScope(shield=True):
        while not task.done():
            try:
                # Wait for completion without forwarding cancellation or installing
                # shield's Python 3.14+ detached-error logger. Read the result below.
                await asyncio.wait((task,))
            except asyncio.CancelledError as exc:
                interrupted = exc
    try:
        task.result()
    except BaseException as error:
        if interrupted is not None:
            interrupted.add_note('Command stop also failed; inspect the retained sandbox reference')
            raise interrupted from error
        raise
    return interrupted


_SOURCE = files('pydantic_ai_harness.render_sandbox').joinpath('_command_runner.py').read_text()


@dataclass(frozen=True)
class PreparedCommand:
    """A command's supervisor launch and stop requests, built before any sandbox exists."""

    nonce: str
    launch: str
    stop: str
    max_output_bytes: int


def prepare_command(
    argv: list[str],
    *,
    working_dir: str,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    max_output_bytes: int = 1_048_576,
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
        'limit': max_output_bytes,
        'drain': _DRAIN_GRACE,
    }
    return PreparedCommand(
        nonce=nonce,
        launch=_launch(_SOURCE, 'run', config),
        stop=_launch(_SOURCE, 'stop', {'claim': claim, 'nonce': nonce}),
        max_output_bytes=max_output_bytes,
    )


async def run_command(
    client: RenderAsync,
    sandbox_id: str,
    command: PreparedCommand,
    *,
    timeout: float | None = None,
    stop_timeout: float = 12.0,
) -> CommandResult:
    """Run once, preserving the sandbox on timeout, cancellation or stream failure.

    The remote supervisor independently checks its received payload. Exceptions from
    Render propagate unchanged, including after an uncertain start. No command
    is replayed. Stop failures remain explicit in exception notes and causes.
    """
    nonce, launch, stop, max_output_bytes = command.nonce, command.launch, command.stop, command.max_output_bytes
    capture = _Capture(nonce, max_output_bytes)
    try:
        deadline = asyncio.timeout(timeout)
        try:
            async with deadline:
                await _consume(client, sandbox_id, launch, capture)
        except TimeoutError as exc:
            if not deadline.expired():
                raise  # A provider TimeoutError retains its original identity.
            stdout, stderr = capture.text()
            raise WorkspaceTimeoutError('Command deadline exceeded', stdout=stdout, stderr=stderr) from exc
        stdout, stderr = capture.text()
        if capture.terminal == 'limit':
            raise capture.overflow()
        if capture.terminal == 'timeout':
            raise WorkspaceTimeoutError('Command deadline exceeded', stdout=stdout, stderr=stderr)
        if capture.terminal != 'exit' or capture.exit_code is None:
            raise WorkspaceError('Command was fenced before it could start')
        return CommandResult(exit_code=capture.exit_code, stdout=stdout, stderr=stderr)
    except BaseException as original:
        try:
            interrupted = await _stop_shielded(client, sandbox_id, stop, nonce, stop_timeout)
        except BaseException as stop_error:
            if isinstance(stop_error, asyncio.CancelledError) and not isinstance(original, asyncio.CancelledError):
                # A new cancellation during failure cleanup still takes priority.
                raise stop_error from original
            original.add_note(
                f'Command stop was not confirmed in sandbox {sandbox_id}; '
                'retain its reference for inspection or explicit disposal'
            )
            # Preserve cancellation and provider exception identity. Frameworks
            # may overwrite implicit __context__, so make cleanup evidence explicit.
            raise original from stop_error
        if interrupted is not None and not isinstance(original, asyncio.CancelledError):
            raise interrupted from original
        raise
