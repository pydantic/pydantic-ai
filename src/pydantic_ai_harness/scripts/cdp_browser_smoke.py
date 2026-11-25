"""Hand-run smoke test for the CDP Browser capability.

Not collected by pytest and not wired into CI: it launches a real Chrome, which the in-process suite deliberately
avoids. Run it where Chrome or Chromium is installed, pointing `CHROME_PATH` at the binary when it is not on `PATH`:

    CHROME_PATH=/path/to/chrome uv run python src/pydantic_ai_harness/scripts/cdp_browser_smoke.py

It drives the public `CDPBrowser` surface through scripted agent runs against local servers, and checks:

- a navigation and a `Runtime.evaluate` on a real page,
- teardown: runs under `blockbuster`, so a blocking call on the event loop fails the check, and after each of
  several runs its profile directory is gone and, where `/proc` lists processes, no Chrome process started for
  that run is left,
- the egress policy against a real local server: the private-address block on the `navigate` pre-check and its
  decimal-IP spelling, and `blocked_domains` on a navigation a page script starts and an in-page `fetch`, with a
  `block_private_addresses=False` control that reaches the server for the plain-address pre-check and the `fetch`,
- the private-name block: a hostname resolving to loopback through a public wildcard DNS service (skipped when
  that name does not resolve),
- the connection proxy: with the private-address block on, `http://example.com/` and `https://example.com/` load
  through it (skipped without internet access), and a name whose lookup answers a public address for the request
  guard and loopback when Chrome connects, the window a rebinding record exploits, never reaches the local server,
  with a `block_private_addresses=False` control that reaches it,
- the request guard in targets of their own: a cross-site iframe, a dedicated worker, a shared worker and a popup
  each fail to reach a host the policy refuses, and the iframe and workers reach it when nothing refuses it,
- service worker registration is blocked,
- dialogs are dismissed, so `alert` and `confirm` do not block the page,
- a screenshot comes back as image content,
- a command that never answers times out, and the tab answers the next one,
- a WebSocket to a host the policy refuses never reaches it, and one to an allowed host connects and echoes.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from blockbuster import blockbuster_ctx
from websockets.asyncio.server import ServerConnection, serve

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import _browser_egress as egress_module, _browser_proxy as proxy_module
from pydantic_ai_harness.cdp_browser import CDPBrowser, EgressPolicy

_SECRET = 'cdp-browser-smoke-secret'
_PROFILE_PREFIX = 'pydantic-ai-chrome-'
# Set CDP_BROWSER_SMOKE_NO_SANDBOX=1 where Chrome's sandbox cannot start.
_SANDBOX = not os.environ.get('CDP_BROWSER_SMOKE_NO_SANDBOX')


def _evaluate(expression: str, timeout: float = 30) -> tuple[str, dict[str, object]]:
    return (
        'browser_cdp',
        {
            'method': 'Runtime.evaluate',
            'params': {'expression': expression, 'returnByValue': True, 'awaitPromise': True},
            'timeout': timeout,
        },
    )


def _navigate(url: str) -> tuple[str, dict[str, object]]:
    return (
        'browser_cdp',
        {'method': 'Page.navigate', 'params': {'url': url}, 'wait_for_event': 'Page.loadEventFired'},
    )


async def _run_tools(browser: CDPBrowser[object], calls: list[tuple[str, dict[str, object]]]) -> list[str]:
    """Drive a fixed sequence of tool calls through one agent run, and return each result or retry prompt as text."""
    results: list[str] = []
    index = 0

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal index
        for part in messages[-1].parts:
            if isinstance(part, ToolReturnPart):
                results.append(f'RETRY: {part.content}' if part.outcome == 'retried' else str(part.content))
        if index < len(calls):
            name, args = calls[index]
            index += 1
            return ModelResponse(parts=[ToolCallPart(tool_name=name, args=args)])
        return ModelResponse(parts=[TextPart('done')])

    await Agent(FunctionModel(model), capabilities=[browser]).run('smoke')
    return results


Route = tuple[str, bytes]


class _Server:
    """A loopback HTTP server with fixed responses, which counts the requests it answers."""

    def __init__(self, server: asyncio.Server, hits: list[str]) -> None:
        self.server = server
        self.port: int = server.sockets[0].getsockname()[1]
        self.hits = hits

    async def __aenter__(self) -> _Server:
        return self

    async def __aexit__(self, *args: object) -> None:
        self.server.close()
        await self.server.wait_closed()


async def _serve(host: str, routes: dict[str, Route]) -> _Server:
    """Serve `routes` over HTTP on a loopback address."""
    hits: list[str] = []

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        request_line = (await reader.readline()).decode()
        while (await reader.readline()) not in (b'\r\n', b''):
            pass
        if ' ' not in request_line:
            # A connection Chrome opened speculatively and closed without a request: nothing was asked or served.
            writer.close()
            return
        path = request_line.split(' ')[1]
        hits.append(path)
        content_type, body = routes.get(path, ('text/plain', b'not found'))
        writer.write(
            f'HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nAccess-Control-Allow-Origin: *\r\n'
            f'Content-Length: {len(body)}\r\nConnection: close\r\n\r\n'.encode()
            + body
        )
        await writer.drain()
        writer.close()

    return _Server(await asyncio.start_server(handle, host, 0), hits)


_SECRET_ROUTES: dict[str, Route] = {'/': ('text/html', f'<title>secret</title><p>{_SECRET}</p>'.encode())}


def _chrome_processes(profile_prefix: str) -> set[int]:
    """Processes whose command line names a profile directory this capability created."""
    pids: set[int] = set()
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                cmdline = (entry / 'cmdline').read_bytes()
            except OSError:
                continue
            if profile_prefix.encode() in cmdline:
                pids.add(int(entry.name))
    return pids


def _profiles() -> set[str]:
    return {name for name in os.listdir(tempfile.gettempdir()) if name.startswith(_PROFILE_PREFIX)}


async def _check_navigate_and_teardown() -> None:
    """Several runs under blockbuster, each leaving no profile directory and, with `/proc`, no Chrome process behind."""
    # macOS and Windows have no `/proc` to list processes from.
    procfs = Path('/proc').is_dir()
    async with await _serve('127.0.0.1', {'/': ('text/html', b'<title>smoke page</title>')}) as server:
        before_profiles = _profiles()
        before_processes = _chrome_processes(_PROFILE_PREFIX) if procfs else set[int]()
        for _ in range(3):
            with blockbuster_ctx():
                results = await _run_tools(
                    CDPBrowser[object](chrome_sandbox=_SANDBOX, block_private_addresses=False),
                    [_navigate(f'http://127.0.0.1:{server.port}/'), _evaluate('document.title')],
                )
            assert "'value': 'smoke page'" in results[-1], results
            assert _profiles() == before_profiles, f'profile left behind: {_profiles() - before_profiles}'
            if procfs:
                leftover = _chrome_processes(_PROFILE_PREFIX) - before_processes
                assert not leftover, f'Chrome processes left behind: {sorted(leftover)}'
    if procfs:
        print('navigate + evaluate ok; 3 runs under blockbuster left no Chrome process and no profile')
    else:
        print('navigate + evaluate ok; 3 runs under blockbuster left no profile; process check SKIPPED (no /proc)')


async def _check_private_address_block() -> None:
    """The pre-check, a decimal IP, a script navigation and a fetch; only the pre-check and fetch have controls."""
    async with (
        await _serve('127.0.0.1', _SECRET_ROUTES) as secret,
        await _serve('127.0.0.2', {'/': ('text/html', b'<title>start</title>')}) as start,
    ):
        url = f'http://127.0.0.1:{secret.port}/'
        start_url = f'http://127.0.0.2:{start.port}/'

        blocked = await _run_tools(
            CDPBrowser[object](chrome_sandbox=_SANDBOX),
            [_navigate(url), _navigate(f'http://2130706433:{secret.port}/')],
        )
        assert all(r.startswith('RETRY:') and 'private or link-local' in r for r in blocked), blocked
        assert not secret.hits, secret.hits
        await _run_tools(CDPBrowser[object](chrome_sandbox=_SANDBOX, block_private_addresses=False), [_navigate(url)])
        assert secret.hits, 'control failed, server unreachable'
        print('navigate refuses a private address and its decimal spelling ok (plain-address control reached it)')

        # The start page is private too, so the policy refuses the secret's host by name instead.
        secret.hits.clear()
        policy = EgressPolicy(blocked_domains=['127.0.0.1'], block_private_addresses=False)
        fetch = f"fetch('{url}').then(r => r.text(), e => 'refused: ' + e.message)"
        results = await _run_tools(
            CDPBrowser[object](chrome_sandbox=_SANDBOX, policy=policy),
            [
                _navigate(start_url),
                _evaluate(f"location.href = '{url}'; 'started'"),
                _evaluate('new Promise(r => setTimeout(r, 1500)).then(() => document.title)'),
                _navigate(start_url),
                _evaluate(fetch),
            ],
        )
        assert not secret.hits, (secret.hits, results)
        assert 'refused: Failed to fetch' in results[-1], results
        open_results = await _run_tools(
            CDPBrowser[object](chrome_sandbox=_SANDBOX, block_private_addresses=False),
            [_navigate(start_url), _evaluate(fetch)],
        )
        assert _SECRET in open_results[-1], f'control failed, fetch never reached the server: {open_results}'
        print('script navigation and in-page fetch to a refused host blocked ok (fetch control reached the server)')


async def _check_private_name_block() -> None:
    """A hostname pointing at loopback must be refused, which needs real DNS."""
    try:
        await asyncio.get_running_loop().getaddrinfo('127.0.0.1.nip.io', None)
    except OSError:
        print('private-name block SKIPPED (127.0.0.1.nip.io did not resolve)')
        return
    async with await _serve('127.0.0.1', _SECRET_ROUTES) as secret:
        name_url = f'http://127.0.0.1.nip.io:{secret.port}/'
        (blocked,) = await _run_tools(CDPBrowser[object](chrome_sandbox=_SANDBOX), [_navigate(name_url)])
        assert blocked.startswith('RETRY:') and 'private or link-local' in blocked, blocked
        assert not secret.hits, secret.hits
        await _run_tools(
            CDPBrowser[object](chrome_sandbox=_SANDBOX, block_private_addresses=False), [_navigate(name_url)]
        )
        assert secret.hits, 'control failed, server unreachable'
    print('private-name navigation blocked ok (control reached the server)')


async def _check_public_pages_through_the_proxy() -> None:
    """With the address block on, Chrome's `http` and `https` connections go through the proxy and still load."""
    try:
        await asyncio.get_running_loop().getaddrinfo('example.com', None)
    except OSError:
        print('public pages through the proxy SKIPPED (example.com did not resolve)')
        return
    for url in ('http://example.com/', 'https://example.com/'):
        results = await _run_tools(
            CDPBrowser[object](chrome_sandbox=_SANDBOX), [_navigate(url), _evaluate('document.title')]
        )
        assert "'value': 'Example Domain'" in results[-1], (url, results)
    print('http and https pages load through the connection proxy ok')


async def _check_rebinding_name() -> None:
    """A name that answers a public address for the guard and loopback when Chrome connects reaches nothing.

    The guard's lookups are answered with a public address, and the proxy's lookup goes to the real resolver, which
    answers `127.0.0.1.nip.io` with loopback. Chrome itself would get loopback too, so a direct connection would
    reach the server.
    """
    try:
        await asyncio.get_running_loop().getaddrinfo('127.0.0.1.nip.io', None)
    except OSError:
        print('rebinding name SKIPPED (127.0.0.1.nip.io did not resolve)')
        return
    real_getaddrinfo = egress_module.getaddrinfo

    async def public_for_the_guard(host: str) -> tuple[str, ...]:
        return ('93.184.216.34',)

    async def real_answer_at_connect(host: str) -> tuple[str, ...] | None:
        try:
            return await real_getaddrinfo(host)
        except OSError:
            return None

    async with await _serve('127.0.0.1', _SECRET_ROUTES) as secret:
        url = f'http://127.0.0.1.nip.io:{secret.port}/'
        # An earlier check's real answer would otherwise reach the guard from the cache.
        egress_module.resolution_cache.clear()
        egress_module.getaddrinfo = public_for_the_guard
        proxy_module.resolve_host = real_answer_at_connect  # pyright: ignore[reportPrivateImportUsage]
        try:
            (navigated,) = await _run_tools(CDPBrowser[object](chrome_sandbox=_SANDBOX), [_navigate(url)])
        finally:
            egress_module.getaddrinfo = real_getaddrinfo
            proxy_module.resolve_host = egress_module.resolve_host  # pyright: ignore[reportPrivateImportUsage]
            egress_module.resolution_cache.clear()
        assert 'errorText' in navigated, navigated
        assert not secret.hits, (secret.hits, navigated)
        await _run_tools(CDPBrowser[object](chrome_sandbox=_SANDBOX, block_private_addresses=False), [_navigate(url)])
        assert secret.hits, 'control failed, server unreachable'
    print(f'rebinding name never reached the server ok ({navigated}; control reached it)')


def _targets_page(secret_url: str, child_url: str) -> bytes:
    worker = f"fetch('{secret_url}').then(r => r.text(), e => 'refused: ' + e.message).then(t => postMessage(t));"
    shared = (
        f"onconnect = c => fetch('{secret_url}').then(r => r.text(), e => 'refused: ' + e.message)"
        '.then(t => c.ports[0].postMessage(t));'
    )
    return f"""<title>targets</title>
<script>
window.results = {{}};
addEventListener('message', e => {{ if (e.data && e.data.from) window.results[e.data.from] = e.data.text; }});
const w = new Worker(URL.createObjectURL(new Blob([{worker!r}], {{type: 'text/javascript'}})));
w.onmessage = e => {{ window.results.worker = e.data; }};
const s = new SharedWorker(URL.createObjectURL(new Blob([{shared!r}], {{type: 'text/javascript'}})));
s.port.onmessage = e => {{ window.results.shared = e.data; }};
</script>
<iframe src="{child_url}"></iframe>""".encode()


def _child_page(secret_url: str) -> bytes:
    return f"""<script>
fetch('{secret_url}').then(r => r.text(), e => 'refused: ' + e.message)
  .then(t => parent.postMessage({{from: 'iframe', text: t}}, '*'));
</script>""".encode()


async def _check_other_targets() -> None:
    """A cross-site iframe, a dedicated worker and a popup are guarded like the page itself."""
    async with await _serve('127.0.0.1', _SECRET_ROUTES) as secret:
        secret_url = f'http://localhost:{secret.port}/'
        async with await _serve('127.0.0.3', {'/child': ('text/html', _child_page(secret_url))}) as child:
            page_html = _targets_page(secret_url, f'http://127.0.0.3:{child.port}/child')
            async with await _serve('127.0.0.2', {'/': ('text/html', page_html)}) as page:
                page_url = f'http://127.0.0.2:{page.port}/'
                wait = 'new Promise(r => setTimeout(r, 2000)).then(() => JSON.stringify(window.results))'
                popup = f"window.open('{secret_url}'); new Promise(r => setTimeout(r, 1500)).then(() => 'opened')"
                policy = EgressPolicy(blocked_domains=['localhost'], block_private_addresses=False)
                results = await _run_tools(
                    CDPBrowser[object](chrome_sandbox=_SANDBOX, policy=policy),
                    [_navigate(page_url), _evaluate(wait), _evaluate(popup)],
                )
                assert not secret.hits, (secret.hits, results)
                observed = results[1]
                assert observed.count('refused: Failed to fetch') == 3, observed
                assert child.hits, 'the iframe never loaded, so it proves nothing'
                print('cross-site iframe, dedicated worker, shared worker and popup to a refused host blocked ok')

                open_results = await _run_tools(
                    CDPBrowser[object](chrome_sandbox=_SANDBOX, block_private_addresses=False),
                    [_navigate(page_url), _evaluate(wait)],
                )
                assert open_results[-1].count(_SECRET) == 3, f'control failed: {open_results}'
                print('control: the same iframe and workers reach the server when nothing refuses it')


async def _check_page_behavior() -> None:
    """Service worker registration, dialogs, screenshots, and a command that never answers."""
    async with await _serve('127.0.0.1', {'/': ('text/html', b'<title>behavior</title><p>hi</p>')}) as server:
        results = await _run_tools(
            CDPBrowser[object](chrome_sandbox=_SANDBOX, block_private_addresses=False),
            [
                _navigate(f'http://127.0.0.1:{server.port}/'),
                _evaluate("navigator.serviceWorker.register('/sw.js').then(r => String(r))"),
                _evaluate("alert('hello'); confirm('sure?')"),
                ('browser_cdp', {'method': 'Page.captureScreenshot'}),
                _evaluate('while (true) {}', timeout=2),
                _evaluate('1 + 1'),
            ],
        )
        _, service_worker, dialogs, screenshot, hang, after = results
        assert "'value': 'undefined'" in service_worker, service_worker
        assert "'value': False" in dialogs, dialogs
        assert screenshot.startswith('Screenshot attached (image/png'), screenshot
        assert hang.startswith('RETRY:') and 'did not answer within 2s' in hang, hang
        assert "'value': 2" in after, after
    print('service worker blocked, dialogs dismissed, screenshot attached, hung command recovered ok')


async def _check_websocket_guard() -> None:
    """A socket to a refused host never reaches it, and one to an allowed host connects and carries data."""
    handshakes: list[str] = []

    async def echo(connection: ServerConnection) -> None:
        handshakes.append(connection.request.path if connection.request else '')
        async for message in connection:
            await connection.send(message)

    async with serve(echo, '127.0.0.1', 0) as server:
        port = next(iter(server.sockets)).getsockname()[1]
        socket = (
            f"new Promise(r => {{ const s = new WebSocket('ws://127.0.0.1:{port}/socket'); "
            "s.onerror = () => r('refused'); s.onopen = () => s.send('ping'); s.onmessage = e => r(e.data); })"
        )
        by_name = EgressPolicy(blocked_domains=['127.0.0.1'], block_private_addresses=False)
        for browser in (
            CDPBrowser[object](chrome_sandbox=_SANDBOX),
            CDPBrowser[object](chrome_sandbox=_SANDBOX, policy=by_name),
        ):
            results = await _run_tools(browser, [_evaluate(socket)])
            assert 'refused' in results[0], results
        assert not handshakes, handshakes
        results = await _run_tools(
            CDPBrowser[object](chrome_sandbox=_SANDBOX, block_private_addresses=False), [_evaluate(socket)]
        )
        assert 'ping' in results[0], f'control failed, the socket never echoed: {results}'
        assert handshakes == ['/socket'], handshakes
        print('WebSocket to a refused host blocked ok (control connected and echoed)')


async def _main() -> None:
    await _check_navigate_and_teardown()
    await _check_private_address_block()
    await _check_private_name_block()
    await _check_public_pages_through_the_proxy()
    await _check_rebinding_name()
    await _check_other_targets()
    await _check_page_behavior()
    await _check_websocket_guard()
    print('all checks passed')


if __name__ == '__main__':
    asyncio.run(_main())
