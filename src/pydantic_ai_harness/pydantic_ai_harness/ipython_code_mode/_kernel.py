"""A persistent Jupyter kernel that `IPythonCodeMode` runs model-written code in."""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Protocol, Self

import anyio

from pydantic_ai.messages import BinaryImage

try:
    import zmq
    import zmq.asyncio
    from jupyter_client.asynchronous.client import AsyncKernelClient
    from jupyter_client.manager import AsyncKernelManager
# Covered only in CI lanes that install the harness without the `ipython` extra.
except ImportError as _import_error:  # pragma: lax no cover
    raise ImportError(
        '`jupyter-client` and `ipykernel` are required for IPythonCodeMode. '
        'Install them with: pip install "pydantic-ai-harness[ipython]"'
    ) from _import_error


class _KernelManager(Protocol):
    """The part of `AsyncKernelManager` used here, typed: jupyter-client leaves its async methods' results unknown."""

    def start_kernel(self, *, cwd: str | None) -> Awaitable[None]: ...
    def shutdown_kernel(self, now: bool = False) -> Awaitable[None]: ...
    def is_alive(self) -> Awaitable[bool]: ...
    def interrupt_kernel(self) -> Awaitable[None]: ...
    def client(self, *, context: zmq.asyncio.Context) -> AsyncKernelClient: ...


_RESULT_MIME = 'application/vnd.pydantic-ai.result+json'
"""Mimetype of the JSON form of a cell's last expression, added by a display formatter the bootstrap registers."""

_IMAGE_MIMES = ('image/png', 'image/jpeg', 'image/gif', 'image/webp')

MAX_OUTPUT_CHARS = 10 * 1024 * 1024
"""Printed and displayed text kept per cell; the rest is dropped and counted."""

_LIVENESS_POLL_SECS = 1.0
"""How often a running cell checks that the kernel process is still alive."""

_INTERRUPT_GRACE_SECS = 5.0
"""How long an interrupted cell gets to stop before the kernel is shut down."""

_SHUTDOWN_WAIT_SECS = 5.0
"""How long `shutdown()` waits for a running cell, before interrupting it and again before killing its kernel."""

_ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')

# Runs once in each new kernel. Defines how injected functions reach the host: a call sends an
# `input_request` over the Jupyter stdin channel with a per-kernel token, and the host answers it
# with the tool's JSON result. The stdin channel is the one channel a kernel can use while a cell
# runs, so no extra socket or port is opened. It also registers a display formatter that renders a
# cell's last expression as JSON when it can be, so results reach the model as data, not a repr.
_BOOTSTRAP = """\
def _pydantic_ai_bootstrap(token, mime):
    import json
    from IPython import get_ipython
    from IPython.core.formatters import BaseFormatter

    shell = get_ipython()
    kernel = shell.kernel

    class ToolCallError(Exception):
        '''A function backed by an agent tool failed; the message carries the tool's error.'''

    def call(name, kwargs):
        reply = json.loads(kernel.raw_input(token + json.dumps({{'name': name, 'args': kwargs}}, default=repr)))
        if 'error' in reply:
            raise ToolCallError(reply['error'])
        return reply['value']

    def define(names):
        def make(name):
            def function(**kwargs):
                __tracebackhide__ = True
                return call(name, kwargs)
            function.__name__ = function.__qualname__ = name
            return function
        shell.user_ns.update({{name: make(name) for name in names}}, ToolCallError=ToolCallError)

    class ResultFormatter(BaseFormatter):
        format_type = mime

        def __call__(self, obj):
            try:
                return json.dumps(obj, allow_nan=False)
            except Exception:
                return None

    shell.display_formatter.formatters[mime] = ResultFormatter(parent=shell.display_formatter)
    shell.pydantic_ai_define_functions = define

_pydantic_ai_bootstrap({token!r}, {mime!r})
del _pydantic_ai_bootstrap
"""

CellOutcome = Literal['ok', 'error', 'interrupted', 'restarted', 'died']
"""How a cell ended.

- `'ok'`: it ran to completion.
- `'error'`: it raised; `CellResult.error` has the traceback.
- `'interrupted'`: it ran past the timeout and stopped on an interrupt. The kernel keeps its state.
- `'restarted'`: it ran past the timeout and ignored the interrupt, so the kernel was shut down.
- `'died'`: the kernel process exited, before or while the cell ran.
"""


@dataclass
class CellResult:
    """What one cell produced."""

    outcome: CellOutcome = 'ok'
    output: str = ''
    """Printed text and the text of non-image displays, in order, capped at `MAX_OUTPUT_CHARS`."""
    result: Any = None
    """The last expression: its JSON value, an image, or its `repr` when it has no JSON form."""
    images: list[BinaryImage] = field(default_factory=list[BinaryImage])
    """Images the cell displayed, such as `matplotlib` figures, in order."""
    error: str | None = None
    """The traceback, without terminal colors, when the cell raised."""


@dataclass
class _Collector:
    """Accumulates the iopub messages of one cell into a `CellResult`."""

    cell: CellResult = field(default_factory=CellResult)
    chunks: list[str] = field(default_factory=list[str])
    size: int = 0
    dropped: int = 0
    idle: bool = False

    def add(self, msg_type: str, content: Mapping[str, Any]) -> None:
        if msg_type == 'stream':
            self._write(content['text'])
        elif msg_type == 'execute_result':
            self.cell.result = _decode_result(content['data'])
        elif msg_type == 'display_data':
            data: Mapping[str, Any] = content['data']
            image = _decode_image(data)
            if image is not None:
                self.cell.images.append(image)
            elif 'text/plain' in data:
                self._write(f'{data["text/plain"]}\n')
        elif msg_type == 'error':
            traceback: Sequence[str] = content['traceback']
            self.cell.error = _ANSI_ESCAPE.sub('', '\n'.join(traceback))
            self.cell.outcome = 'error'
        elif msg_type == 'status' and content['execution_state'] == 'idle':
            self.idle = True

    def _write(self, text: str) -> None:
        kept = text[: max(MAX_OUTPUT_CHARS - self.size, 0)]
        if kept:
            self.chunks.append(kept)
            self.size += len(kept)
        self.dropped += len(text) - len(kept)

    def finish(self) -> CellResult:
        output = ''.join(self.chunks)
        if self.dropped:
            output += f'\n[output truncated: {self.dropped} more characters were dropped]'
        self.cell.output = output
        return self.cell


def _decode_image(data: Mapping[str, Any]) -> BinaryImage | None:
    for mime in _IMAGE_MIMES:
        encoded = data.get(mime)
        if isinstance(encoded, str):
            try:
                return BinaryImage(data=base64.b64decode(encoded), media_type=mime)
            except ValueError:
                # Code can display any string under an image mimetype; the text form is used instead.
                return None
    return None


def _decode_result(data: Mapping[str, Any]) -> Any:
    image = _decode_image(data)
    if image is not None:
        return image
    encoded = data.get(_RESULT_MIME)
    if isinstance(encoded, str):
        return json.loads(encoded)
    return data.get('text/plain')


def _kernel_manager(kernel_name: str, context: zmq.asyncio.Context) -> _KernelManager:
    """A manager for a new kernel, talking to it over Unix sockets in the temp directory where the platform has them.

    A TCP kernel listens on local ports that other users of the machine can reach, and ipykernel
    warns about it on every start. Windows has no IPC transport, so it keeps Jupyter's TCP default.
    The caller owns `context`: not every supported jupyter-client closes one it creates itself.
    """
    if sys.platform == 'win32':  # pragma: no cover
        return AsyncKernelManager(kernel_name=kernel_name, context=context)
    ip = os.path.join(tempfile.gettempdir(), f'pydantic-ai-kernel-{secrets.token_hex(6)}')
    return AsyncKernelManager(kernel_name=kernel_name, transport='ipc', ip=ip, context=context)


@dataclass
class IPythonKernel:
    """A persistent Jupyter kernel that `IPythonCodeMode` runs code in.

    Without one, `IPythonCodeMode` starts a kernel when the model first runs code and shuts it down
    when the agent run ends. Pass the same instance to `IPythonCodeMode(kernel=...)` to keep the
    kernel, with its variables, imports, and loaded data, across runs: a conversation of several
    runs then shares one namespace. You own its lifecycle, so shut it down with `async with` or
    `shutdown()`:

    ```python {test="skip"}
    from pydantic_ai import Agent
    from pydantic_ai_harness import IPythonCodeMode
    from pydantic_ai_harness.ipython_code_mode import IPythonKernel


    async def main():
        async with IPythonKernel() as kernel:
            agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[IPythonCodeMode(kernel=kernel)])
            result = await agent.run('Load data.csv into a DataFrame named `df`.')
            result = await agent.run('Plot `df`.', message_history=result.all_messages())
    ```

    The kernel runs one cell at a time, so runs that share it wait for each other. It is bound to the
    event loop it was started on.
    """

    kernel_name: str = 'python3'
    """The Jupyter kernelspec to launch.

    The default `'python3'` runs ipykernel with the agent's own Python interpreter, so code can import
    whatever the agent's environment has installed. Name another installed kernelspec to run code in a
    different environment; it must be an IPython kernel.
    """

    _: KW_ONLY

    cwd: str | Path | None = None
    """Working directory of the kernel process; unset, it inherits the agent process's."""

    startup_timeout: float = 60.0
    """Seconds to wait for a new kernel to answer before giving up."""

    _manager: _KernelManager | None = field(default=None, init=False, repr=False)
    _client: AsyncKernelClient | None = field(default=None, init=False, repr=False)
    _context: zmq.asyncio.Context | None = field(default=None, init=False, repr=False)
    _token: str = field(default='', init=False, repr=False)

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.shutdown()

    async def start(self) -> None:
        """Start the kernel process, unless it is already running."""
        async with self._lock:
            await self._ensure_started()

    async def restart(self) -> None:
        """Replace the kernel process with a fresh one, discarding its state."""
        async with self._lock:
            await self._stop()
            await self._ensure_started()

    async def shutdown(self) -> None:
        """Stop the kernel process. A later `start()` or cell starts a fresh one.

        Shielded, so a run ending in cancellation still stops the process. A cell another run is still
        running gets 5 seconds to finish, is then interrupted, and 5 seconds later its kernel is killed,
        so this returns within about 10 seconds.
        """
        with anyio.CancelScope(shield=True):
            with anyio.move_on_after(_SHUTDOWN_WAIT_SECS):
                async with self._lock:
                    await self._stop()
                    return
            await self._interrupt()
            with anyio.move_on_after(_SHUTDOWN_WAIT_SECS):
                async with self._lock:
                    await self._stop()
                    return
            # The run still holding the lock, in a cell or a tool call it made, finds the process gone the
            # next time it polls the kernel, and finishes stopping it.
            await self._kill()

    async def _interrupt(self) -> None:
        # Unset only while the run holding the lock is still starting the kernel.
        if self._manager is not None:  # pragma: no branch
            await self._manager.interrupt_kernel()

    async def _kill(self) -> None:
        # Unset only while the run holding the lock is still starting the kernel.
        if self._manager is not None:  # pragma: no branch
            await self._manager.shutdown_kernel(now=True)

    @cached_property
    def _lock(self) -> anyio.Lock:
        """Serializes cells, created on first use so it binds to the event loop that uses the kernel."""
        return anyio.Lock()

    async def _ensure_started(self) -> None:
        if self._manager is not None:
            return
        context = zmq.asyncio.Context()
        manager = _kernel_manager(self.kernel_name, context)
        try:
            await manager.start_kernel(cwd=None if self.cwd is None else str(self.cwd))
        except BaseException:
            context.destroy(linger=0)
            raise
        # Given the same context, as a client left to create its own keeps it open after `stop_channels()`
        # on jupyter-client before 8.7, until the garbage collector finds it.
        client = manager.client(context=context)
        client.start_channels()
        self._manager, self._client, self._context = manager, client, context
        self._token = f'\x00pydantic-ai-tool-call:{secrets.token_hex(16)}\x00'
        try:
            await client.wait_for_ready(timeout=self.startup_timeout)
            bootstrap = await self._run_cell(_BOOTSTRAP.format(token=self._token, mime=_RESULT_MIME), silent=True)
            if bootstrap.outcome != 'ok':
                raise RuntimeError(
                    f'Kernel {self.kernel_name!r} could not be set up for code mode; it must be an IPython kernel.'
                    f'\n{bootstrap.error or ""}'
                )
        except BaseException:
            with anyio.CancelScope(shield=True):
                await self._stop()
            raise

    async def _stop(self) -> None:
        manager, client, context = self._manager, self._client, self._context
        self._manager = self._client = self._context = None
        if client is not None:
            client.stop_channels()
        if manager is not None:
            await manager.shutdown_kernel(now=True)
        if context is not None:
            context.destroy(linger=0)

    async def _execute(
        self,
        code: str,
        *,
        functions: Sequence[str],
        call_function: Callable[[str], Awaitable[str]],
        timeout: float | None,
    ) -> CellResult:
        """Run one cell, answering the calls its code makes to `functions` with `call_function`.

        `call_function` takes the JSON request a function sent and returns the JSON reply, a
        `value` or an `error`. Time spent in it doesn't count toward `timeout`.
        """
        async with self._lock:
            await self._ensure_started()
            if functions:
                # Every cell, so a function the code overwrote or `%reset` removed is back, and one the catalog
                # dropped answers with an error. A kernel that died since the last cell is found here.
                defined = await self._run_cell(
                    f'get_ipython().pydantic_ai_define_functions({list(functions)!r})', silent=True
                )
                if defined.outcome != 'ok':
                    return defined
            return await self._run_cell(code, call_function=call_function, timeout=timeout)

    async def _run_cell(  # noqa: C901
        self,
        code: str,
        *,
        silent: bool = False,
        call_function: Callable[[str], Awaitable[str]] | None = None,
        timeout: float | None = None,
    ) -> CellResult:
        manager, client = self._manager, self._client
        assert manager is not None and client is not None, 'the kernel is started before a cell runs'
        # `stop_on_error=False`: by default ipykernel aborts the requests queued behind a cell that fails,
        # which would drop the next call's cell after a timeout or a cancelled run interrupted this one.
        msg_id = client.execute(code, silent=silent, store_history=not silent, allow_stdin=True, stop_on_error=False)
        iopub, stdin = client.iopub_channel.socket, client.stdin_channel.socket
        poller = zmq.asyncio.Poller()
        poller.register(iopub, zmq.POLLIN)
        poller.register(stdin, zmq.POLLIN)
        collector = _Collector()
        deadline = None if timeout is None else time.monotonic() + timeout
        interrupted = False
        try:
            while not collector.idle:
                # Checked every iteration, not only when the kernel is quiet: a cell that keeps printing
                # never leaves the poll empty.
                if deadline is not None and time.monotonic() >= deadline:
                    if interrupted:
                        await self._stop()
                        return CellResult(outcome='restarted')
                    await manager.interrupt_kernel()
                    interrupted = True
                    deadline = time.monotonic() + _INTERRUPT_GRACE_SECS
                wait = _LIVENESS_POLL_SECS
                if deadline is not None:
                    wait = max(min(wait, deadline - time.monotonic()), 0)
                events = dict(await poller.poll(int(wait * 1000)))
                if not events:
                    if not await manager.is_alive():
                        await self._stop()
                        return CellResult(outcome='died')
                    continue

                if stdin in events:
                    request = await client.get_stdin_msg(timeout=0)
                    # A request from an earlier, interrupted cell can't arrive: interrupting it ends its wait.
                    if request['parent_header'].get('msg_id') == msg_id:  # pragma: no branch
                        prompt: str = request['content']['prompt']
                        reply = ''
                        if call_function is not None and prompt.startswith(self._token):
                            started = time.monotonic()
                            reply = await call_function(prompt[len(self._token) :])
                            if deadline is not None and not interrupted:
                                deadline += time.monotonic() - started
                        client.input(reply)
                if iopub in events:
                    message = await client.get_iopub_msg(timeout=0)
                    # Skips other requests' messages, such as startup `kernel_info` replies; whether any arrive
                    # during a given cell depends on timing.
                    if message['parent_header'].get('msg_id') == msg_id:  # pragma: no branch
                        collector.add(message['msg_type'], message['content'])
        except BaseException:
            # The run was cancelled mid-cell: stop the cell so a kernel that outlives the run is free.
            with anyio.CancelScope(shield=True):
                # Unset only when stopping the kernel itself failed.
                if self._manager is not None:  # pragma: no branch
                    if await manager.is_alive():
                        await self._interrupt_abandoned_cell(msg_id)
                    else:
                        # `shutdown()` killed it while this cell waited on a tool call; close the channels.
                        await self._stop()
            raise
        cell = collector.finish()
        if interrupted:
            # Also when the code caught the `KeyboardInterrupt` and finished: it ran past the timeout.
            cell.outcome = 'interrupted'
        return cell

    async def _interrupt_abandoned_cell(self, msg_id: str) -> None:
        """Interrupt a cell whose run was cancelled, and stop the kernel if the cell keeps running.

        The lock is released right after, so a cell still running would leave the next one queued
        behind it with nothing to stop it.
        """
        manager, client = self._manager, self._client
        assert manager is not None and client is not None, 'only called while the kernel is running'
        await manager.interrupt_kernel()
        with anyio.move_on_after(_INTERRUPT_GRACE_SECS):
            while True:
                message = await client.get_iopub_msg()
                content = message['content']
                if (
                    message['parent_header'].get('msg_id') == msg_id
                    and message['msg_type'] == 'status'
                    and content['execution_state'] == 'idle'
                ):
                    return
        await self._stop()
