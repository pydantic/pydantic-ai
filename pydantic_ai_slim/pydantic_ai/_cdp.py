"""A minimal Chrome DevTools Protocol client, and a launcher for a throwaway Chrome, used by the `Browser` capability."""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

# Largest CDP message accepted, in bytes; full-page screenshots arrive base64-encoded in one message.
MAX_MESSAGE_SIZE = 64 * 1024 * 1024


@asynccontextmanager
async def launch_chrome(*, headless: bool = True, chrome_path: str | None = None) -> AsyncGenerator[CDPSession]:
    """Start Chrome with a throwaway profile and yield a `CDPSession` for its one open tab."""
    try:
        from websockets.asyncio.client import connect
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            'Please install `websockets` to use the `Browser` capability, '
            'you can use the `browser` optional group — `pip install "pydantic-ai-slim[browser]"`'
        ) from e

    with tempfile.TemporaryDirectory(prefix='pydantic-ai-chrome-') as profile:
        args = [
            chrome_path or find_chrome(),
            f'--user-data-dir={profile}',
            '--remote-debugging-port=0',
            '--no-first-run',
            '--no-default-browser-check',
            'about:blank',
        ]
        if headless:
            args.insert(1, '--headless=new')
        process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            ws_url = await _read_devtools_url(Path(profile))
            async with connect(ws_url, max_size=MAX_MESSAGE_SIZE) as ws:
                client = CDPClient(ws)
                async with client.run():
                    # downloads would write to the host filesystem
                    await client.send('Browser.setDownloadBehavior', {'behavior': 'deny'})
                    yield await client.attach_to_first_page()
        finally:
            process.terminate()
            process.wait()


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
        if found := shutil.which(candidate) or (Path(candidate).is_file() and candidate):
            return found
    raise FileNotFoundError(
        'Chrome not found, set `CHROME_PATH` or pass `chrome_path` to the Chrome or Chromium binary'
    )


async def _read_devtools_url(profile: Path) -> str:
    """Wait for Chrome to write `DevToolsActivePort`, which holds its debugging port and browser path."""
    port_file = profile / 'DevToolsActivePort'
    with anyio.fail_after(15):
        while not port_file.exists() or len(lines := port_file.read_text().splitlines()) < 2:
            await anyio.sleep(0.05)
    return f'ws://127.0.0.1:{lines[0]}{lines[1]}'


class CDPClient:
    """A CDP connection: numbered requests matched to responses, events fanned out to waiters.

    One websocket carries the browser target and every attached page, distinguished by `sessionId`.
    """

    def __init__(self, ws: ClientConnection):
        self._ws = ws
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._event_waiters: list[tuple[str | None, str, asyncio.Future[dict[str, Any]]]] = []

    @asynccontextmanager
    async def run(self) -> AsyncGenerator[None]:
        """Read messages in the background for the lifetime of the block."""
        reader = asyncio.create_task(self._read_loop())
        try:
            yield
        finally:
            reader.cancel()
            with suppress(asyncio.CancelledError):
                await reader

    async def send(self, method: str, params: dict[str, Any] | None = None, session_id: str | None = None) -> Any:
        """Send one CDP command and return its `result`, raising `RuntimeError` on a protocol error."""
        message_id = next(self._ids)
        message: dict[str, Any] = {'id': message_id, 'method': method, 'params': params or {}}
        if session_id:
            message['sessionId'] = session_id
        future = self._pending[message_id] = asyncio.get_running_loop().create_future()
        await self._ws.send(json.dumps(message))
        response = await future
        if error := response.get('error'):
            raise RuntimeError(f'{method} failed: {error["message"]}')
        return response['result']

    def wait_for_event(self, method: str, session_id: str | None = None) -> asyncio.Future[dict[str, Any]]:
        """Return a future for the next `method` event; create it *before* the command that triggers it."""
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._event_waiters.append((session_id, method, future))
        return future

    async def attach_to_first_page(self) -> CDPSession:
        """Attach to the tab Chrome opened at startup and enable the `Page` domain for its events."""
        targets = await self.send('Target.getTargets')
        target_id = next(t['targetId'] for t in targets['targetInfos'] if t['type'] == 'page')
        attached = await self.send('Target.attachToTarget', {'targetId': target_id, 'flatten': True})
        session = CDPSession(self, attached['sessionId'])
        await session.send('Page.enable')
        return session

    async def _read_loop(self) -> None:
        try:
            async for raw in self._ws:
                message = json.loads(raw)
                if 'id' in message:
                    # a cancelled `send()` leaves its future here, already done
                    if (future := self._pending.pop(message['id'], None)) and not future.done():
                        future.set_result(message)
                else:
                    self._dispatch_event(message)
        finally:
            self._fail_waiters()

    def _fail_waiters(self) -> None:
        """Fail every outstanding request and event waiter once the connection is gone, so none hangs."""
        error = ConnectionError('Chrome DevTools connection closed')
        waiters = [*self._pending.values(), *(future for _, _, future in self._event_waiters)]
        self._pending.clear()
        self._event_waiters.clear()
        for future in waiters:
            if not future.done():
                future.set_exception(error)

    def _dispatch_event(self, message: dict[str, Any]) -> None:
        remaining: list[tuple[str | None, str, asyncio.Future[dict[str, Any]]]] = []
        for session_id, method, future in self._event_waiters:
            if method == message['method'] and session_id == message.get('sessionId') and not future.done():
                future.set_result(message.get('params', {}))
            elif not future.done():
                remaining.append((session_id, method, future))
        self._event_waiters = remaining


@dataclass
class CDPSession:
    """A `CDPClient` bound to one attached page target."""

    client: CDPClient
    session_id: str

    async def send(self, method: str, params: dict[str, Any] | None = None) -> Any:
        return await self.client.send(method, params, self.session_id)

    def wait_for_event(self, method: str) -> asyncio.Future[dict[str, Any]]:
        return self.client.wait_for_event(method, self.session_id)
