"""Run one command through sandboxd's streaming `ProcessService.Start`.

Workaround: the SDK's `sandbox.commands.run()` wraps the unary `Execute`, which buffers all output
in one message (over gRPC's 4 MiB receive limit it fails with `RESOURCE_EXHAUSTED`), returns
nothing when its deadline passes, and fails with `INTERNAL` when a background child keeps stdout
open. `Start` streams 4 KiB chunks, so this keeps whatever arrived before a timeout. Replace with the SDK's own `Start` wrapper once it exists
(https://github.com/kubernetes-sigs/agent-sandbox/issues/1603; bounded output in
https://github.com/kubernetes-sigs/agent-sandbox/issues/1918).

sandboxd behaviour relied on, verified 2026-10-19 against sandboxd from agent-sandbox v1.0.7 and main:

* cancelling the call SIGKILLs the command's process group, so timeouts and cancelled runs stop
  the foreground work; a `setsid` child may survive, which the workspace protocol allows;
* a command killed by signal n exits with 128+n (sandboxd v1.0.6 reported -1);
* when a background child keeps stdout open, the stream ends about 5 seconds after the command exits.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import anyio

from pydantic_ai.workspaces import CommandResult, WorkspaceOutputLimitError, WorkspaceTimeoutError
from pydantic_ai_harness.kubernetes_sandbox._wire import (
    decode_start_response,
    encode_start_request,
)

if TYPE_CHECKING:
    import grpc.aio

_SERVICE = '/process.v1.ProcessService'

# Output is collected in this process, so `yes` or a large `cat` stops at the cap instead of growing it.
_MAX_OUTPUT_BYTES = 10 * 1024 * 1024
_OUTPUT_PREVIEW_BYTES = 64 * 1024


def _identity(data: bytes) -> bytes:
    return data


async def start_command(
    channel: grpc.aio.Channel,
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    timeout: float | None,
) -> CommandResult:
    """Run `argv` in sandboxd's root directory, returning its complete output.

    Raises `WorkspaceTimeoutError` with the output that arrived when `timeout` passes, and
    `WorkspaceOutputLimitError` when the output passes 10 MiB; either way the command is killed.
    """
    start = channel.unary_stream(
        f'{_SERVICE}/Start', request_serializer=_identity, response_deserializer=decode_start_response
    )
    call = start(encode_start_request(argv, env))
    stdout = bytearray()
    stderr = bytearray()
    exit_code: int | None = None
    with anyio.move_on_after(timeout) as scope:
        try:
            async for event in call:
                if event.kind == 'exit':
                    exit_code = event.number
                elif event.kind != 'unknown':
                    (stdout if event.kind == 'stdout' else stderr).extend(event.data)
                    if len(stdout) + len(stderr) > _MAX_OUTPUT_BYTES:
                        raise WorkspaceOutputLimitError(
                            'Command output exceeded the 10 MiB limit; redirect it to a file and read part of it instead',
                            limit=_MAX_OUTPUT_BYTES,
                            stdout=stdout[:_OUTPUT_PREVIEW_BYTES].decode(errors='replace'),
                            stderr=stderr[:_OUTPUT_PREVIEW_BYTES].decode(errors='replace'),
                        )
        except asyncio.CancelledError:
            # Closing a channel cancels its calls, so another command's lost connection can end this
            # one with a cancellation no one asked for. Raised as the connection error it is.
            task = asyncio.current_task()
            if task is not None and not task.cancelling():
                raise ConnectionError('The connection to sandboxd closed while the command was running') from None
            raise
        finally:
            # On timeout or cancellation this makes sandboxd kill the process group; after the exit
            # event it is a no-op.
            call.cancel()
    if scope.cancelled_caught:
        raise WorkspaceTimeoutError(
            f'Command timed out after {timeout:g} seconds',
            stdout=stdout.decode(errors='replace'),
            stderr=stderr.decode(errors='replace'),
        )
    if exit_code is None:
        raise RuntimeError('sandboxd ended the command stream without an exit event')
    return CommandResult(
        exit_code=exit_code, stdout=stdout.decode(errors='replace'), stderr=stderr.decode(errors='replace')
    )
