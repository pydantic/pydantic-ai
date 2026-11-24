"""An in-process stand-in for Chrome: a DevTools websocket endpoint, and the process `launch_chrome` starts.

`FakeChrome` answers the commands `CDPBrowser` sends the way Chrome does, over a real websocket, so the client's
framing, response matching, event fan-out and teardown all run for real. Only the browser is faked: what Chrome
itself does with these commands is checked against a real one by `scripts/cdp_browser_smoke.py`.

A `Runtime.evaluate` expression that is a JSON object scripts the page: `{"pause": {...}}` makes it issue a request
that the egress guard must answer, and `{"dialog": "<type>"}` opens a dialog that blocks the command until it is
answered. The expression `hang` never answers, and `disconnect` closes the connection.
"""

from __future__ import annotations

import base64
import itertools
import json
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeGuard

import anyio
from websockets.asyncio.server import ServerConnection

PAGE_TARGET_ID = 'PAGE-TARGET'
PAGE_SESSION = 'page-session'


@dataclass
class FakeProcess:
    """Stands in for Chrome's `anyio.abc.Process`: exits when Chrome is closed or its process group is killed."""

    pid: int
    returncode: int | None = None
    exited: anyio.Event = field(default_factory=anyio.Event)

    async def wait(self) -> int:
        await self.exited.wait()
        assert self.returncode is not None
        return self.returncode

    def exit(self, code: int) -> None:
        self.returncode = code
        self.exited.set()

    def kill(self) -> None:
        """What Windows teardown calls; like asyncio's, it raises for a process that already exited."""
        if self.returncode is not None:
            raise ProcessLookupError(self.pid)
        self.exit(-9)


def is_object_dict(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict)


def pause_expression(
    url: str, *, resource_type: str = 'Document', frame_id: str = PAGE_TARGET_ID, session_id: str = PAGE_SESSION
) -> str:
    """A `Runtime.evaluate` expression that makes the fake page issue a request, and returns the guard's verdict."""
    return json.dumps(
        {'pause': {'url': url, 'resourceType': resource_type, 'frameId': frame_id, 'sessionId': session_id}}
    )


@dataclass
class FakeChrome:
    """The browser end of the DevTools connection, with knobs for the cases a real Chrome produces rarely."""

    port: int = 0
    received: list[tuple[str | None, str, dict[str, object]]] = field(
        default_factory=list[tuple[str | None, str, dict[str, object]]]
    )
    """Every command received, as `(session id, method, params)`."""
    verdicts: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    """How the guard answered each paused request, as `(url, 'continue' | 'fail')`."""
    errors: dict[str, str] = field(default_factory=dict[str, str])
    """Methods to answer with a protocol error, and its message."""
    error_targets: set[str] = field(default_factory=set[str])
    """Target types whose `Fetch.enable` is answered with an error."""
    unanswered_targets: set[str] = field(default_factory=set[str])
    """Target types whose `Fetch.enable` is never answered."""
    redirects: dict[str, str] = field(default_factory=dict[str, str])
    """Navigation URLs that redirect, and where to: the redirect is paused as a request of its own, as Chrome does."""
    close_after_auto_attach: bool = False
    """Close the connection once the browser connection turns on auto-attach, before any tab is attached."""
    hanging: anyio.Event = field(default_factory=anyio.Event)
    """Set when the `hang` expression arrives."""
    bare_replies: set[str] = field(default_factory=set[str])
    """Methods to answer with no `result` at all."""
    child_targets: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    """Targets attached when the page session turns on auto-attach, as `targetInfo` dicts with a `sessionId`."""
    browser_events: list[object] = field(default_factory=list[object])
    """Raw messages sent when the browser connection turns on auto-attach, before the page is attached."""
    attach_page: bool = True
    write_port: bool = True
    exit_on_start: bool = False
    ignore_close: bool = False
    screenshot: bytes = b'\x89PNG fake image'
    launches: list[list[str]] = field(default_factory=list[list[str]])
    processes: list[FakeProcess] = field(default_factory=list['FakeProcess'])
    killed_groups: list[int] = field(default_factory=list[int])
    _ws: ServerConnection | None = None
    _ids: itertools.count[int] = field(default_factory=lambda: itertools.count(1))
    _verdict_events: dict[str, tuple[anyio.Event, list[str]]] = field(
        default_factory=dict[str, tuple[anyio.Event, list[str]]]
    )
    _dialog_answer: tuple[anyio.Event, list[bool]] | None = None

    @property
    def process(self) -> FakeProcess:
        return self.processes[-1]

    async def start(self, args: list[str]) -> FakeProcess:
        """Replaces `start_process`: write `DevToolsActivePort` to the profile, as Chrome does once it listens."""
        self.launches.append(args)
        process = FakeProcess(pid=4_000_000 + len(self.processes))
        self.processes.append(process)
        profile = Path(next(arg for arg in args if arg.startswith('--user-data-dir=')).split('=', 1)[1])
        if self.write_port:
            await anyio.Path(profile / 'DevToolsActivePort').write_text(f'{self.port}\n/devtools/browser/fake\n')
        if self.exit_on_start:
            process.exit(21)
        return process

    def killpg(self, pgid: int, sig: int) -> None:
        """Replaces `os.killpg`: an exited Chrome has no group left to signal, as on Linux."""
        process = next(p for p in self.processes if p.pid == pgid)
        assert sig == signal.SIGKILL
        if process.returncode is not None:
            raise ProcessLookupError(pgid)
        self.killed_groups.append(pgid)
        process.exit(-signal.SIGKILL)

    async def handle(self, ws: ServerConnection) -> None:
        self._ws = ws
        async with anyio.create_task_group() as tg:
            async for raw in ws:
                message: dict[str, object] = json.loads(raw)
                tg.start_soon(self._answer, message)

    async def send_raw(self, message: object) -> None:
        assert self._ws is not None
        await self._ws.send(json.dumps(message))

    async def _event(self, method: str, params: dict[str, object], session_id: str | None) -> None:
        message: dict[str, object] = {'method': method, 'params': params}
        if session_id is not None:
            message['sessionId'] = session_id
        await self.send_raw(message)

    async def _reply(self, message: dict[str, object], result: dict[str, object]) -> None:
        reply: dict[str, object] = {'id': message['id'], 'result': result}
        if 'sessionId' in message:
            reply['sessionId'] = message['sessionId']
        await self.send_raw(reply)

    async def _attach(self, info: dict[str, object], parent: str | None) -> None:
        session_id = info['sessionId']
        target_info = {k: v for k, v in info.items() if k != 'sessionId'}
        await self._event(
            'Target.attachedToTarget',
            {'sessionId': session_id, 'targetInfo': target_info, 'waitingForDebugger': True},
            parent,
        )

    async def _pause(self, request: dict[str, object]) -> str:
        request_id = f'request-{next(self._ids)}'
        decided = anyio.Event()
        verdict: list[str] = []
        self._verdict_events[request_id] = (decided, verdict)
        await self._event(
            'Fetch.requestPaused',
            {
                'requestId': request_id,
                'request': {'url': request['url'], 'method': 'GET'},
                'frameId': request['frameId'],
                'resourceType': request['resourceType'],
            },
            str(request['sessionId']),
        )
        await decided.wait()
        self.verdicts.append((str(request['url']), verdict[0]))
        return verdict[0]

    async def _answer(self, message: dict[str, object]) -> None:  # noqa: C901
        method = str(message['method'])
        raw_params = message.get('params')
        params = raw_params if is_object_dict(raw_params) else {}
        session_id = message.get('sessionId')
        session_id = session_id if isinstance(session_id, str) else None
        self.received.append((session_id, method, params))
        if method in self.errors:
            await self.send_raw({'id': message['id'], 'error': {'code': -32000, 'message': self.errors[method]}})
            return
        if method in self.bare_replies:
            await self.send_raw({'id': message['id']})
            return
        if method == 'Target.setAutoAttach':
            await self._reply(message, {})
            if session_id is None and self.close_after_auto_attach:
                assert self._ws is not None
                await self._ws.close()
            elif session_id is None:
                for event in self.browser_events:
                    await self.send_raw(event)
                if self.attach_page:
                    await self._attach({'sessionId': PAGE_SESSION, 'targetId': PAGE_TARGET_ID, 'type': 'page'}, None)
            elif session_id == PAGE_SESSION:
                for child in self.child_targets:
                    await self._attach(child, PAGE_SESSION)
        elif method == 'Fetch.enable' and self._target_type(session_id) in self.unanswered_targets:
            return
        elif method == 'Fetch.enable' and self._target_type(session_id) in self.error_targets:
            await self.send_raw(
                {'id': message['id'], 'error': {'code': -32601, 'message': "'Fetch.enable' wasn't found"}}
            )
        elif method in ('Fetch.continueRequest', 'Fetch.failRequest'):
            decided, verdict = self._verdict_events.pop(str(params['requestId']))
            verdict.append('continue' if method == 'Fetch.continueRequest' else 'fail')
            decided.set()
            await self._reply(message, {})
        elif method == 'Page.navigate':
            url = str(params['url'])
            outcome = await self._pause(
                {'url': url, 'resourceType': 'Document', 'frameId': PAGE_TARGET_ID, 'sessionId': PAGE_SESSION}
            )
            if outcome == 'continue' and url in self.redirects:
                outcome = await self._pause(
                    {
                        'url': self.redirects[url],
                        'resourceType': 'Document',
                        'frameId': PAGE_TARGET_ID,
                        'sessionId': PAGE_SESSION,
                    }
                )
            if outcome == 'fail':
                await self._reply(message, {'frameId': PAGE_TARGET_ID, 'errorText': 'net::ERR_BLOCKED_BY_CLIENT'})
            else:
                await self._reply(message, {'frameId': PAGE_TARGET_ID})
                await self._event('Page.loadEventFired', {'timestamp': 1.0}, session_id)
        elif method == 'Runtime.evaluate':
            await self._evaluate(message, str(params['expression']), session_id)
        elif method == 'Page.handleJavaScriptDialog':
            assert self._dialog_answer is not None
            answered, accepted = self._dialog_answer
            accepted.append(bool(params['accept']))
            answered.set()
            await self._reply(message, {})
        elif method == 'Page.captureScreenshot':
            await self._reply(message, {'data': base64.b64encode(self.screenshot).decode()})
        elif method == 'Browser.close':
            if not self.ignore_close:
                self.process.exit(0)
                assert self._ws is not None
                await self._ws.close()
        else:
            await self._reply(message, {})

    async def _evaluate(self, message: dict[str, object], expression: str, session_id: str | None) -> None:
        if expression == 'hang':
            self.hanging.set()
            return
        if expression == 'disconnect':
            assert self._ws is not None
            await self._ws.close()
            return
        if expression.startswith('{'):
            script: dict[str, object] = json.loads(expression)
            if is_object_dict(pause := script.get('pause')):
                outcome = await self._pause(pause)
                await self._reply(message, {'result': {'type': 'string', 'value': outcome}})
                return
            answered, accepted = self._dialog_answer = (anyio.Event(), [])
            await self._event('Page.javascriptDialogOpening', {'type': script['dialog']}, session_id)
            await answered.wait()
            await self._reply(message, {'result': {'type': 'boolean', 'value': accepted[0]}})
            return
        await self._reply(message, {'result': {'type': 'string', 'value': 'Example Domain'}})

    def _target_type(self, session_id: str | None) -> str:
        if session_id == PAGE_SESSION:
            return 'page'
        return next(str(child['type']) for child in self.child_targets if child['sessionId'] == session_id)
