"""A minimal Chrome DevTools Protocol client, and a launcher for a throwaway Chrome, used by `CDPBrowser`.

External-service assumptions (Chrome or Chromium, and the DevTools Protocol). The test suite drives an in-process
fake of the protocol, so these are what it cannot show. Re-check them against a real browser with
`scripts/cdp_browser_smoke.py` before changing launch, interception, dialog or teardown handling.

- `--remote-debugging-port=0` makes Chrome pick a free port and write it, with the browser target's path, as the
  first two lines of `DevToolsActivePort` in the profile directory.
- `Target.setAutoAttach` with `flatten: true` on the browser connection attaches to every top-level target (tabs,
  popups, service and shared workers), and on a page or worker session to its children (out-of-process iframes,
  dedicated workers). `waitForDebuggerOnStart` holds each new target until `Runtime.runIfWaitingForDebugger`.
  Source: <https://chromedevtools.github.io/devtools-protocol/tot/Target/#method-setAutoAttach>.
- `Fetch.enable` on a target pauses each request the target makes in a `Fetch.requestPaused` event until
  `Fetch.continueRequest` or `Fetch.failRequest` answers it. Source:
  <https://chromedevtools.github.io/devtools-protocol/tot/Fetch/>.
- A shared worker target has a `Fetch` domain, and its requests are paused on its own session.
- A dedicated worker target has no `Fetch` domain (`Fetch.enable` answers "'Fetch.enable' wasn't found"); its
  requests are paused on the session of the page that started it.
- A page target's main frame id is its target id.
- Chrome's child processes (renderer, GPU, utility) join its process group, and can still be writing to the profile
  after the browser process exits.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import anyio
from anyio.abc import Process

from pydantic_ai_harness._typeguards import is_str_keyed_dict

try:
    from websockets.asyncio.client import ClientConnection, connect
    from websockets.exceptions import ConnectionClosed
except ImportError as _import_error:
    raise ImportError(
        'websockets is required for CDPBrowser. Install it with: pip install "pydantic-ai-harness[cdp-browser]"'
    ) from _import_error

MAX_MESSAGE_SIZE = 64 * 1024 * 1024
"""Largest CDP message accepted, in bytes; a full-page screenshot arrives base64-encoded in one message."""

STARTUP_TIMEOUT_SECONDS = 15.0
"""How long Chrome gets to open its DevTools port."""

SHUTDOWN_GRACE_SECONDS = 5.0
"""How long Chrome gets to exit after `Browser.close`, before its process group is killed."""

PROFILE_REMOVAL_ATTEMPTS = 20
"""How many times the profile directory removal is tried while Chrome's last child processes exit."""

CLOSED_ERRORS = (ConnectionError, ConnectionClosed)
"""What a command raises when the DevTools connection is gone."""

EventHandler = Callable[[str | None, dict[str, object]], Awaitable[None]]
"""Called with an event's session id (`None` for the browser connection) and its params."""


class CDPError(RuntimeError):
    """Chrome answered a command with a protocol error, such as an unknown method or invalid params."""


@asynccontextmanager
async def launch_chrome(
    *, extra_args: Sequence[str] = (), headless: bool = True, chrome_path: str | None = None, sandbox: bool = True
) -> AsyncGenerator[CDPClient]:
    """Start Chrome with a throwaway profile and yield a `CDPClient` connected to it.

    Downloads are denied, since they would write to the host filesystem. On exit Chrome is asked to close, then its
    whole process group is killed, and only then is the profile removed: removing it while a child process is still
    writing fails with "Directory not empty".
    """
    # The search stats candidate paths, so it runs off the event loop.
    executable = chrome_path or await anyio.to_thread.run_sync(find_chrome)
    profile = await anyio.to_thread.run_sync(partial(tempfile.mkdtemp, prefix='pydantic-ai-chrome-'))
    try:
        args = [
            executable,
            f'--user-data-dir={profile}',
            '--remote-debugging-port=0',
            '--no-first-run',
            '--no-default-browser-check',
            *extra_args,
            'about:blank',
        ]
        if headless:
            args.insert(1, '--headless=new')
        if not sandbox:
            args.insert(1, '--no-sandbox')
        process = await start_process(args)
        try:
            ws_url = await read_devtools_url(Path(profile), process)
            async with connect(ws_url, max_size=MAX_MESSAGE_SIZE) as ws:
                client = CDPClient(ws)
                async with client.run():
                    try:
                        await client.send('Browser.setDownloadBehavior', {'behavior': 'deny'})
                        yield client
                    finally:
                        await close_browser(client, process)
        finally:
            await kill_process_group(process)
    finally:
        await remove_profile(Path(profile))


def find_chrome() -> str:
    """Locate a Chrome or Chromium binary: `CHROME_PATH` if set, otherwise the usual install locations."""
    if path := os.environ.get('CHROME_PATH'):
        return path
    if sys.platform == 'darwin':
        candidates = ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome']
    elif sys.platform == 'win32':
        roots = [os.environ.get(var) for var in ('PROGRAMFILES', 'PROGRAMFILES(X86)', 'LOCALAPPDATA')]
        candidates = [rf'{root}\Google\Chrome\Application\chrome.exe' for root in roots if root]
    else:
        candidates = ['google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser']
    for candidate in candidates:
        # A bare name is only looked up on `PATH`, never in the working directory.
        if found := shutil.which(candidate) or (
            Path(candidate).is_absolute() and Path(candidate).is_file() and candidate
        ):
            return found
    raise FileNotFoundError(
        'Chrome not found, set `CHROME_PATH` or pass `chrome_path` to the Chrome or Chromium binary'
    )


# The suite replaces this with a fake process; `scripts/cdp_browser_smoke.py` runs it against a real Chrome.
async def start_process(args: list[str]) -> Process:  # pragma: no cover
    """Start Chrome as the leader of a new process group, so teardown can reach every process it starts."""
    return await anyio.open_process(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


async def read_devtools_url(profile: Path, process: Process) -> str:
    """Wait for Chrome to write `DevToolsActivePort`, and return its browser websocket URL."""
    port_file = anyio.Path(profile / 'DevToolsActivePort')
    try:
        with anyio.fail_after(STARTUP_TIMEOUT_SECONDS):
            while True:
                if await port_file.exists() and len(lines := (await port_file.read_text()).splitlines()) >= 2:
                    return f'ws://127.0.0.1:{lines[0]}{lines[1]}'
                if process.returncode is not None:
                    raise RuntimeError(f'Chrome exited with code {process.returncode} before opening its DevTools port')
                await anyio.sleep(0.05)
    except TimeoutError:
        raise RuntimeError(f'Chrome did not open its DevTools port within {STARTUP_TIMEOUT_SECONDS:g}s') from None


async def close_browser(client: CDPClient, process: Process) -> None:
    """Ask Chrome to close, and give it `SHUTDOWN_GRACE_SECONDS` to exit, whatever cancellation is under way."""
    with anyio.CancelScope(shield=True), anyio.move_on_after(SHUTDOWN_GRACE_SECONDS):
        # The connection closes as Chrome exits, so the command's answer may never arrive.
        with suppress(CDPError, *CLOSED_ERRORS):
            await client.send('Browser.close')
        await process.wait()


async def kill_process_group(process: Process) -> None:
    """Kill Chrome and every process it started, and reap Chrome itself.

    The group is killed even after Chrome exited on its own, since its children can outlive it. Windows has no
    process groups, so there Chrome alone is killed.
    """
    with anyio.CancelScope(shield=True):
        # An exited Chrome, or an empty group, has nothing left to signal.
        with suppress(ProcessLookupError):
            if sys.platform == 'win32':
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        await process.wait()


async def remove_profile(profile: Path) -> None:
    """Remove the profile directory off the event loop, retrying while Chrome's last child processes exit."""
    with anyio.CancelScope(shield=True):
        for _ in range(PROFILE_REMOVAL_ATTEMPTS):
            await anyio.to_thread.run_sync(partial(shutil.rmtree, profile, ignore_errors=True))
            if not await anyio.Path(profile).exists():
                return
            await anyio.sleep(0.1)


class CDPClient:
    """A CDP connection: numbered requests matched to responses, events fanned out to waiters and handlers.

    One websocket carries the browser target and every attached target, distinguished by `sessionId`.
    """

    def __init__(self, ws: ClientConnection) -> None:
        self._ws = ws
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future[dict[str, object]]] = {}
        self._event_waiters: list[tuple[str | None, str, asyncio.Future[dict[str, object]]]] = []
        self._handlers: dict[str, EventHandler] = {}
        self._handler_tasks: set[asyncio.Task[None]] = set()
        self._session_futures: list[asyncio.Future[CDPSession]] = []

    @asynccontextmanager
    async def run(self) -> AsyncGenerator[None]:
        """Read messages in the background for the lifetime of the block, and stop every handler at its end."""
        reader = asyncio.create_task(self._read_loop(), name='cdp-read-loop')
        try:
            yield
        finally:
            reader.cancel()
            for task in self._handler_tasks:
                task.cancel()
            with anyio.CancelScope(shield=True):
                await asyncio.gather(reader, *self._handler_tasks, return_exceptions=True)

    def session_future(self) -> asyncio.Future[CDPSession]:
        """Return a future for a session the caller resolves, failed like every waiter if the connection closes first."""
        future: asyncio.Future[CDPSession] = asyncio.get_running_loop().create_future()
        self._session_futures.append(future)
        return future

    def on(self, method: str, handler: EventHandler) -> None:
        """Call `handler` for every `method` event, each in its own task so it can send commands of its own."""
        self._handlers[method] = handler

    async def send(
        self, method: str, params: dict[str, object] | None = None, session_id: str | None = None
    ) -> dict[str, object]:
        """Send one CDP command and return its `result`, raising `CDPError` on a protocol error.

        Cancelling the call, for example by a deadline around it, stops the wait and forgets the command: a late
        answer is dropped.
        """
        message_id = next(self._ids)
        message: dict[str, object] = {'id': message_id, 'method': method, 'params': params or {}}
        if session_id is not None:
            message['sessionId'] = session_id
        future = self._pending[message_id] = asyncio.get_running_loop().create_future()
        try:
            await self._ws.send(json.dumps(message))
            response = await future
        finally:
            self._pending.pop(message_id, None)
        if is_str_keyed_dict(error := response.get('error')):
            raise CDPError(f'{method} failed: {error.get("message")}')
        result = response.get('result')
        return result if is_str_keyed_dict(result) else {}

    def wait_for_event(self, method: str, session_id: str | None = None) -> asyncio.Future[dict[str, object]]:
        """Return a future for the next `method` event; create it *before* the command that triggers it.

        Cancel the future to stop waiting.
        """
        future: asyncio.Future[dict[str, object]] = asyncio.get_running_loop().create_future()
        self._event_waiters.append((session_id, method, future))
        return future

    async def _read_loop(self) -> None:
        try:
            async for raw in self._ws:
                message: object = json.loads(raw)
                if not is_str_keyed_dict(message):
                    continue
                message_id = message.get('id')
                if isinstance(message_id, int):
                    if (future := self._pending.get(message_id)) is not None and not future.done():
                        future.set_result(message)
                elif isinstance(method := message.get('method'), str):
                    self._dispatch_event(method, message)
        finally:
            self._fail_waiters()

    def _fail_waiters(self) -> None:
        """Fail every outstanding request and event waiter once the connection is gone, so none hangs."""
        error = ConnectionError('Chrome DevTools connection closed')
        waiters: list[asyncio.Future[dict[str, object]] | asyncio.Future[CDPSession]] = [
            *self._pending.values(),
            *(future for _, _, future in self._event_waiters),
            *self._session_futures,
        ]
        self._event_waiters.clear()
        for future in waiters:
            if not future.done():
                future.set_exception(error)

    def _dispatch_event(self, method: str, message: dict[str, object]) -> None:
        session_id = message.get('sessionId')
        session_id = session_id if isinstance(session_id, str) else None
        params = message.get('params')
        params = params if is_str_keyed_dict(params) else {}
        remaining: list[tuple[str | None, str, asyncio.Future[dict[str, object]]]] = []
        for waiter in self._event_waiters:
            waiter_session, waiter_method, future = waiter
            if future.done():
                continue
            if waiter_method == method and waiter_session == session_id:
                future.set_result(params)
            else:
                remaining.append(waiter)
        self._event_waiters = remaining
        if (handler := self._handlers.get(method)) is not None:
            task = asyncio.create_task(self._run_handler(handler, session_id, params), name=f'cdp-handler-{method}')
            self._handler_tasks.add(task)
            task.add_done_callback(self._handler_tasks.discard)

    async def _run_handler(self, handler: EventHandler, session_id: str | None, params: dict[str, object]) -> None:
        # A target can go away while its handler is still answering for it; that is not a failure of the run.
        with suppress(CDPError, *CLOSED_ERRORS):
            await handler(session_id, params)


@dataclass
class CDPSession:
    """A `CDPClient` bound to one attached target."""

    client: CDPClient
    session_id: str

    async def send(self, method: str, params: dict[str, object] | None = None) -> dict[str, object]:
        return await self.client.send(method, params, self.session_id)

    def wait_for_event(self, method: str) -> asyncio.Future[dict[str, object]]:
        return self.client.wait_for_event(method, self.session_id)
