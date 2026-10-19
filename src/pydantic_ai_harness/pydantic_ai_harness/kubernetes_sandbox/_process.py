"""Run one command through sandboxd's streaming `ProcessService.Start`.

Workaround: the SDK's `sandbox.commands.run()` wraps the unary `Execute`, which buffers all output
in one message (over gRPC's 4 MiB receive limit it fails with `RESOURCE_EXHAUSTED`) and returns
nothing when its deadline passes. `Start` streams 4 KiB chunks, so this keeps whatever arrived
before a timeout. Replace with the SDK's own `Start` wrapper once it exists
(https://github.com/kubernetes-sigs/agent-sandbox/issues/1603; bounded output in
https://github.com/kubernetes-sigs/agent-sandbox/issues/1918).

sandboxd behaviour relied on, verified 2026-10-19 against `sandboxd:latest-main` (agent-sandbox v1.0.7):

* cancelling the call SIGKILLs the command's process group, so timeouts and cancelled runs stop
  the foreground work; a `setsid` child survives, which the workspace protocol allows;
* a command killed by signal n exits with 128+n;
* when a background child keeps stdout open, the stream ends about 5 seconds after the command exits.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import anyio

from pydantic_ai.workspaces import CommandResult, WorkspaceTimeoutError
from pydantic_ai_harness.kubernetes_sandbox._wire import (
    decode_start_response,
    encode_close_stdin,
    encode_start_request,
)

if TYPE_CHECKING:
    import grpc.aio

_SERVICE = '/process.v1.ProcessService'


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

    Raises `WorkspaceTimeoutError` with the output that arrived when `timeout` passes.
    """
    import grpc

    start = channel.unary_stream(
        f'{_SERVICE}/Start', request_serializer=_identity, response_deserializer=decode_start_response
    )
    write_stdin = channel.unary_unary(
        f'{_SERVICE}/WriteStdin', request_serializer=_identity, response_deserializer=_identity
    )
    call = start(encode_start_request(argv, env))
    stdout = bytearray()
    stderr = bytearray()
    exit_code: int | None = None
    with anyio.move_on_after(timeout) as scope:
        try:
            async for event in call:
                if event.kind == 'init':
                    await _close_stdin(write_stdin, event.number)
                elif event.kind == 'stdout':
                    stdout += event.data
                elif event.kind == 'stderr':
                    stderr += event.data
                elif event.kind == 'exit':
                    exit_code = event.number
        except grpc.aio.AioRpcError as error:
            # Workaround: sandboxd reports a program it cannot start as a `NOT_FOUND` error before
            # any event, where the workspace protocol (like a shell) wants exit code 127.
            # Not tracked upstream yet.
            if error.code() == grpc.StatusCode.NOT_FOUND and exit_code is None and not stdout and not stderr:
                return CommandResult(exit_code=127, stdout='', stderr=f'{error.details()}\n')
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


async def _close_stdin(write_stdin: grpc.aio.UnaryUnaryMultiCallable[bytes, bytes], process_id: int) -> None:
    import grpc

    # Workaround: `Start` gives the command an open stdin pipe, so a command that reads stdin would
    # wait forever. The protocol wants stdin at EOF, so close it as soon as the process id arrives.
    # Not tracked upstream yet (a `Start` option for a closed stdin would remove this call).
    try:
        await write_stdin(encode_close_stdin(process_id))
    except grpc.aio.AioRpcError as error:
        # A command that already exited races this call: sandboxd answers `NOT_FOUND` once the
        # process is unregistered, or `INTERNAL: ... file already closed` just before. Its stdin is
        # closed either way.
        already_closed = error.code() == grpc.StatusCode.INTERNAL and 'already closed' in (error.details() or '')
        if error.code() != grpc.StatusCode.NOT_FOUND and not already_closed:
            raise
