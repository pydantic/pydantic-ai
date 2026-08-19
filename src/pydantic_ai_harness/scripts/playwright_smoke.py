"""Hand-run smoke test for the Playwright browser capability.

Not collected by pytest and not wired into CI: it launches a real Chromium and
reaches the public internet, which the mocked unit suite deliberately avoids. Run
it after installing the browser binary to verify the real integration end to end:

    playwright install chromium
    uv run python scripts/playwright_smoke.py

It exercises the public `PlaywrightBrowser` surface and checks twelve things:

- lazy launch plus a real navigation to https://example.com (prints the title),
- the allowlist bounce (a disallowed host returns an error result, not content),
- the private-address block against a real local server, including the decimal
  spelling of an IP (which only Chromium's canonicalization resolves) and an
  in-page `fetch`, each with an opt-out control proving the server is reachable,
- the private-name block: a hostname resolving to loopback through a public
  wildcard DNS service is refused, which needs real DNS and so cannot be checked
  by the mocked suite,
- a `storage_state` round-trip: a cookie captured from a real context is visible
  to the agent after relaunching the capability with that state,
- embedded content: text inside a real cross-origin iframe reaches the model, a
  `snapshot` ref from that frame clicks inside it, and `wait_for` matches content
  that appears there,
- the browser event log: console output and a failed request land in
  `console_messages` / `network_requests` (a refused request is checked in the
  private-address scenario, where the guard is what refuses it),
- the WebSocket guard: a socket to a loopback port is refused and recorded, while
  a public socket still connects, sends and receives through it,
- tabs: a `target="_blank"` link opens a second tab that stays open, `tabs` lists
  it, selects it, reads it, and closes it,
- dialogs: a real `confirm` is dismissed by default and accepted after
  `handle_next_dialog`, and a `prompt` is answered with the given text,
- keystroke-level typing and a disappearance wait: a type-ahead handler that
  `fill` never triggers fires under `sequential=True`, and `wait_for(gone=True)`
  returns once a spinner is hidden,
- attaching over `cdp_url` to a browser that already holds a session: the run gets
  its own context (the existing cookie is not visible), the allowlist still bounces,
  and the host browser's own page survives teardown,
- clean teardown: each scenario runs its own capability, whose `wrap_run` closes
  the browser when the run ends. After it prints `all checks passed`, confirm no
  Chromium lingered, e.g. `pgrep -fl chromium` shows nothing this script started.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import socket
from tempfile import TemporaryDirectory

from playwright.async_api import StorageState, async_playwright
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness.playwright import EgressPolicy, PlaywrightBrowser

_COOKIE = 'smoke_session=abc123'
_SECRET = 'private-address-smoke-secret'


async def _run_tools(
    browser: PlaywrightBrowser[object],
    calls: list[tuple[str, dict[str, object]]],
    resolve_ref: tuple[str, int] | None = None,
    pause: float = 0.0,
) -> list[str]:
    """Drive a fixed sequence of tool calls through one agent run, in order.

    Emitting one scripted `ToolCallPart` per model turn keeps every call in the
    same run, so the launched browser (and its `storage_state` session) persists
    across the sequence. Returns each tool's stringified result.
    """
    results: list[str] = []
    index = 0

    def _with_ref(args: dict[str, object]) -> dict[str, object]:
        """Substitute `aria-ref=REF` with the ref of the named node in the last snapshot.

        A ref is only knowable at run time, so the scripted call carries a
        placeholder and the ref is read out of the snapshot that preceded it.
        """
        if resolve_ref is None or args.get('selector') != 'aria-ref=REF':
            return args
        label, snapshot_index = resolve_ref
        line = next(line for line in results[snapshot_index].splitlines() if label in line and '[ref=' in line)
        return {**args, 'selector': f'aria-ref={line.split("[ref=")[1].split("]")[0]}'}

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal index
        # A real run spends a model round trip between tool calls; this scripted one
        # spends none, which is short enough that a browser event triggered by the
        # previous call (a tab opening) may not have been delivered yet.
        await asyncio.sleep(pause)
        for part in messages[-1].parts:
            if isinstance(part, ToolReturnPart):
                results.append(str(part.content))
        if index < len(calls):
            name, args = calls[index]
            index += 1
            return ModelResponse(parts=[ToolCallPart(tool_name=name, args=_with_ref(args))])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model), capabilities=[browser])
    await agent.run('smoke')
    return results


async def _check_navigate() -> None:
    """Lazy launch, then navigate to a real page and report its title."""
    (result,) = await _run_tools(PlaywrightBrowser[object](), [('navigate', {'url': 'https://example.com'})])
    assert 'Title:' in result, result
    title_line = next(line for line in result.splitlines() if line.startswith('Title:'))
    print(f'navigate ok -- {title_line}')


async def _check_allowlist_bounce() -> None:
    """A navigation to a host outside the allowlist returns an error result."""
    browser = PlaywrightBrowser[object](allowed_domains=['example.com'])
    (result,) = await _run_tools(browser, [('navigate', {'url': 'https://www.iana.org/'})])
    assert 'not in allowed_domains' in result, result
    print('allowlist bounce ok')


async def _serve_secret() -> tuple[asyncio.Server, int]:
    """Serve a fixed body over HTTP on a loopback port, so a leak is observable."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readline()
        body = _SECRET.encode()
        writer.write(
            b'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n'
            b'Access-Control-Allow-Origin: *\r\n'
            b'Content-Length: ' + str(len(body)).encode() + b'\r\n\r\n' + body
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, '127.0.0.1', 0)
    return server, server.sockets[0].getsockname()[1]


async def _check_private_address_block() -> None:
    """The block must survive both an exotic IP spelling and an in-page fetch.

    `2130706433` is not parseable as an address by `ipaddress`, so it reaches the
    block only because something canonicalizes it: the system resolver, which
    answers `127.0.0.1` and so decides it at the pre-check, or failing that
    Chromium, which is why the route guard sees a canonical URL. The `fetch` never
    goes through `navigate` at all, so it rests on the route guard alone, and the
    mocked suite feeds that guard already-canonical URLs. Each assertion is paired
    with a `block_private_addresses=False` control, so a server that simply failed
    to start could not pass the check by accident.
    """
    server, port = await _serve_secret()
    async with server:
        decimal_url = f'http://2130706433:{port}/'
        fetch = f"fetch('http://127.0.0.1:{port}/').then(r => r.text())"

        (blocked_nav,) = await _run_tools(PlaywrightBrowser[object](), [('navigate', {'url': decimal_url})])
        assert _SECRET not in blocked_nav, blocked_nav
        # Refused with the reason rather than as a bare network failure, whichever
        # layer canonicalized the spelling first.
        assert 'private or link-local' in blocked_nav, blocked_nav
        (open_nav,) = await _run_tools(
            PlaywrightBrowser[object](block_private_addresses=False), [('navigate', {'url': decimal_url})]
        )
        assert _SECRET in open_nav, f'control failed, server unreachable: {open_nav}'
        print('decimal-IP navigation blocked ok (control reached the server)')

        blocked_fetch, refusals = await _run_tools(
            PlaywrightBrowser[object](), [('execute_js', {'script': fetch}), ('network_requests', {})]
        )
        assert _SECRET not in blocked_fetch, blocked_fetch
        # This one has no pre-check to reach it, so the route guard is what refuses
        # it, and the refusal is recorded with its reason.
        assert 'request_blocked' in refusals and 'private or link-local' in refusals, refusals
        open_fetch = await _run_tools(
            PlaywrightBrowser[object](block_private_addresses=False),
            [('navigate', {'url': f'http://127.0.0.1:{port}/'}), ('execute_js', {'script': fetch})],
        )
        assert _SECRET in open_fetch[-1], f'control failed, fetch never reached the server: {open_fetch}'
        print('in-page fetch to a private address blocked ok (control reached the server)')


async def _check_private_name_block() -> None:
    """A hostname pointing at a private address must be refused, not followed.

    The host is an ordinary name, so nothing in the URL says where it goes; only
    resolving it does. `nip.io` is a public wildcard resolver that answers with the
    address embedded in the name, which is the same shape as an attacker pointing a
    record they control at the metadata endpoint. This check therefore needs real
    DNS, which is exactly why the mocked suite cannot make it.
    """
    server, port = await _serve_secret()
    async with server:
        name_url = f'http://127.0.0.1.nip.io:{port}/'
        try:
            await asyncio.get_running_loop().getaddrinfo('127.0.0.1.nip.io', None)
        except OSError:  # pragma: no cover - network-dependent
            print('private-name block SKIPPED (127.0.0.1.nip.io did not resolve)')
            return

        (blocked,) = await _run_tools(PlaywrightBrowser[object](), [('navigate', {'url': name_url})])
        assert _SECRET not in blocked, blocked
        (open_nav,) = await _run_tools(
            PlaywrightBrowser[object](block_private_addresses=False), [('navigate', {'url': name_url})]
        )
        assert _SECRET in open_nav, f'control failed, server unreachable: {open_nav}'
        print('private-name navigation blocked ok (control reached the server)')


async def _serve_page(html: str) -> tuple[asyncio.Server, int]:
    """Serve one HTML document over HTTP on a loopback port."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readline()
        body = html.encode()
        writer.write(
            b'HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n'
            b'Content-Length: ' + str(len(body)).encode() + b'\r\n\r\n' + body
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, '127.0.0.1', 0)
    return server, server.sockets[0].getsockname()[1]


async def _check_blocked_redirect_message() -> None:
    """A click the guard refuses names the destination, not Chromium's error page.

    Only a real browser produces the state this is about: the guard aborts the
    request mid-navigation and Chromium is left on `chrome-error://chromewebdata/`,
    a URL the test doubles never generate on their own.
    """
    server, port = await _serve_page('<a href="https://example.org/invite">join</a>')
    async with server:
        browser = PlaywrightBrowser[object](allowed_domains=['127.0.0.1'], block_private_addresses=False)
        _, clicked = await _run_tools(
            browser,
            [('navigate', {'url': f'http://127.0.0.1:{port}/'}), ('click', {'selector': 'a'})],
        )
        assert 'example.org/invite' in clicked, clicked
        assert 'chrome-error' not in clicked, clicked
        print('blocked-redirect message names the refused URL ok')


_NO_CORS_FETCH = "fetch('https://example.com/', {mode: 'no-cors'}).then(() => 'sent').catch(e => 'failed: ' + e)"


async def _check_page_request_egress() -> None:
    """A page's own fetch answers to the allowlist; its images do not.

    The mocked suite drives the guard with synthetic requests, so only a real
    Chromium proves the resource types it reports line up with the policy's kinds
    -- and that a page whose assets come from elsewhere still renders.
    """
    page = b'<html><body><h1>Local page</h1><img src="https://example.com/favicon.ico"></body></html>'
    page_server, page_port = await _serve_html(page)
    async with page_server:
        browser = PlaywrightBrowser[object](
            policy=EgressPolicy(allowed_domains=['127.0.0.1'], block_private_addresses=False)
        )
        _, fetched, refusals = await _run_tools(
            browser,
            [
                ('navigate', {'url': f'http://127.0.0.1:{page_port}/'}),
                ('execute_js', {'script': _NO_CORS_FETCH}),
                ('network_requests', {'errors_only': True}),
            ],
        )
        # A cross-origin fetch is opaque either way, so the verdict is whether it left at all.
        assert 'sent' not in fetched, fetched
        assert 'request_blocked' in refusals, refusals
        print('page fetch to a non-allowlisted host refused ok')

        # The control: the same host is reachable once the allowlist names it.
        allowed = PlaywrightBrowser[object](
            policy=EgressPolicy(allowed_domains=['127.0.0.1', 'example.com'], block_private_addresses=False)
        )
        _, reachable = await _run_tools(
            allowed,
            [
                ('navigate', {'url': f'http://127.0.0.1:{page_port}/'}),
                ('execute_js', {'script': _NO_CORS_FETCH}),
            ],
        )
        assert 'sent' in reachable, f'control failed, the allowlisted fetch did not complete: {reachable}'
        print('page fetch to an allowlisted host still completes ok')


async def _capture_storage_state() -> StorageState:
    """Log a cookie into a real context and hand back its storage state."""
    async with async_playwright() as pw:
        chromium = await pw.chromium.launch(headless=True)
        try:
            context = await chromium.new_context()
            page = await context.new_page()
            await page.goto('https://example.com')
            await page.evaluate(f"document.cookie = '{_COOKIE}; path=/'")
            return await context.storage_state()
        finally:
            await chromium.close()


async def _check_storage_state_round_trip() -> None:
    """A cookie captured from a real context is visible to the agent after relaunch."""
    browser = PlaywrightBrowser[object](storage_state=await _capture_storage_state())
    _, cookies = await _run_tools(
        browser,
        [('navigate', {'url': 'https://example.com'}), ('execute_js', {'script': 'document.cookie'})],
    )
    assert _COOKIE in cookies, cookies
    print('storage_state round-trip ok')


async def _check_cdp_attach() -> None:
    """Attaching over CDP gets a fresh context, not the sessions already open there.

    The mocked suite cannot answer this: it turns on whether a real Chrome honours
    `Target.createBrowserContext` for a browser Playwright did not launch. A cookie
    is logged into the attached browser's own default context, then the capability
    attaches and reads `document.cookie` on the same origin. The allowlist is
    checked in the same run, since the route guard is installed on the context the
    capability creates rather than on the browser it connected to.
    """
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    with TemporaryDirectory() as user_data_dir:
        async with async_playwright() as pw:
            default_context = await pw.chromium.launch_persistent_context(
                user_data_dir, headless=True, args=[f'--remote-debugging-port={port}']
            )
            try:
                page = await default_context.new_page()
                await page.goto('https://example.com')
                await page.evaluate(f"document.cookie = '{_COOKIE}; path=/'")
                assert _COOKIE in await page.evaluate('document.cookie')

                browser = PlaywrightBrowser[object](cdp_url=f'http://127.0.0.1:{port}', allowed_domains=['example.com'])
                _, cookies, bounced = await _run_tools(
                    browser,
                    [
                        ('navigate', {'url': 'https://example.com'}),
                        ('execute_js', {'script': 'document.cookie'}),
                        ('navigate', {'url': 'https://www.iana.org/'}),
                    ],
                )
                assert _COOKIE not in cookies, f'attached run inherited the default context session: {cookies}'
                assert 'not in allowed_domains' in bounced, bounced
                assert not default_context.pages[0].is_closed(), 'teardown closed a page it did not create'
            finally:
                await default_context.close()
    print('cdp attach ok -- isolated context, allowlist enforced, host browser left open')


async def _serve_html(body: bytes) -> tuple[asyncio.Server, int]:
    """Serve a fixed HTML body over HTTP on a loopback port."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readline()
        writer.write(
            b'HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n'
            b'Content-Length: ' + str(len(body)).encode() + b'\r\n\r\n' + body
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, '127.0.0.1', 0)
    return server, server.sockets[0].getsockname()[1]


_INNER_FRAME = (
    b'<html><body><h1>Inner schedule</h1>'
    b'<button id="more">Show more</button>'
    b'<div id="late" style="display:none">LATE-CONTENT</div>'
    b"<script>document.getElementById('more').onclick = () => "
    b"{ document.getElementById('late').style.display = 'block' }</script>"
    b'</body></html>'
)


async def _check_embedded_frame() -> None:
    """Content inside a real iframe: readable, clickable by ref, waitable."""
    inner, inner_port = await _serve_html(_INNER_FRAME)
    outer, outer_port = await _serve_html(
        f'<html><body><h1>Conference</h1><iframe src="http://127.0.0.1:{inner_port}/" '
        f'width="600" height="400"></iframe></body></html>'.encode()
    )
    async with inner, outer:
        browser = PlaywrightBrowser[object](block_private_addresses=False)
        page_text, snapshot, _, waited = await _run_tools(
            browser,
            [
                ('navigate', {'url': f'http://127.0.0.1:{outer_port}/'}),
                ('snapshot', {}),
                ('click', {'selector': 'aria-ref=REF'}),
                ('wait_for', {'text': 'LATE-CONTENT', 'timeout_ms': 3000}),
            ],
            resolve_ref=('Show more', 1),
        )
        assert 'Inner schedule' in page_text, f'iframe text missing from the page read: {page_text}'
        assert 'f1e' in snapshot, f'snapshot carries no frame-scoped refs: {snapshot}'
        assert 'timed out' not in waited, waited
    print('embedded frame ok -- read, clicked by ref, and waited inside the iframe')


async def _check_browser_event_log() -> None:
    """Console output, a failed request, and a refused request all reach the tools."""
    server, port = await _serve_html(
        b'<html><body><script>console.error("page boom");'
        b'fetch("http://127.0.0.1:1/never").catch(() => {});'
        b'</script></body></html>'
    )
    async with server:
        browser = PlaywrightBrowser[object](block_private_addresses=False, allowed_domains=['127.0.0.1'])
        _, console, network = await _run_tools(
            browser,
            [
                ('navigate', {'url': f'http://127.0.0.1:{port}/'}),
                ('console_messages', {'errors_only': True}),
                ('network_requests', {}),
            ],
        )
        assert 'page boom' in console, console
        assert '127.0.0.1:1/never' in network, network
    print('browser event log ok -- console error and failed request recorded')


_WS_GUID = b'258EAFA5-E914-47DA-95CA-C5AB0DC85B11'


async def _check_tabs() -> None:
    """A `target="_blank"` link opens a tab that stays open, and `tabs` drives it."""
    second, second_port = await _serve_html(b'<html><body><h1>SECOND-TAB</h1></body></html>')
    first, first_port = await _serve_html(
        f'<html><body><h1>First</h1>'
        f'<a id="open" href="http://127.0.0.1:{second_port}/" target="_blank">Open</a>'
        f'</body></html>'.encode()
    )
    async with first, second:
        browser = PlaywrightBrowser[object](block_private_addresses=False)
        _, _, listed, selected, closed = await _run_tools(
            browser,
            [
                ('navigate', {'url': f'http://127.0.0.1:{first_port}/'}),
                ('click', {'selector': '#open'}),
                ('tabs', {'action': 'list'}),
                ('tabs', {'action': 'select', 'index': 1}),
                ('tabs', {'action': 'close', 'index': 1}),
            ],
            pause=0.3,
        )
        assert listed.count('\n') == 1, f'the opened tab is missing from the list: {listed}'
        assert '(active)' in listed.splitlines()[0], f'the popup took over as active tab: {listed}'
        assert 'SECOND-TAB' in selected, f'selecting the tab did not read it: {selected}'
        assert closed.startswith('Closed tab 1.'), closed
    print('tabs ok -- popup kept, listed, selected, read and closed')


_DIALOG_PAGE = (
    b'<html><body><h1>Dialogs</h1><div id="out">none</div>'
    b'<button id="ask">Ask</button><button id="name">Name</button>'
    b"<script>document.getElementById('ask').onclick = () => "
    b"{ document.getElementById('out').textContent = confirm('Sure?') ? 'CONFIRMED' : 'CANCELLED' };"
    b"document.getElementById('name').onclick = () => "
    b"{ document.getElementById('out').textContent = 'NAME:' + prompt('Who?') };"
    b'</script></body></html>'
)


async def _check_dialogs() -> None:
    """A real `confirm` takes the cancelling branch by default and the accepting one when armed."""
    server, port = await _serve_html(_DIALOG_PAGE)
    async with server:
        browser = PlaywrightBrowser[object](block_private_addresses=False)
        _, dismissed, _, accepted, _, answered = await _run_tools(
            browser,
            [
                ('navigate', {'url': f'http://127.0.0.1:{port}/'}),
                ('click', {'selector': '#ask'}),
                ('handle_next_dialog', {'accept': True}),
                ('click', {'selector': '#ask'}),
                ('handle_next_dialog', {'accept': True, 'prompt_text': 'Ada'}),
                ('click', {'selector': '#name'}),
            ],
        )
        assert 'CANCELLED' in dismissed, f'an unarmed confirm was not dismissed: {dismissed}'
        assert 'CONFIRMED' in accepted, f'an armed confirm was not accepted: {accepted}'
        assert 'NAME:Ada' in answered, f'the prompt was not answered with the given text: {answered}'
    print('dialogs ok -- confirm dismissed, confirm accepted, prompt answered')


_TYPEAHEAD_PAGE = (
    b'<html><body><input id="q"><div id="hint"></div><div id="spinner">Loading</div>'
    b"<script>document.getElementById('q').addEventListener('keydown', () => "
    b"{ document.getElementById('hint').textContent = 'TYPEAHEAD-FIRED' });"
    b"setTimeout(() => { document.getElementById('spinner').style.display = 'none' }, 800);"
    b'</script></body></html>'
)


async def _check_typing_and_waiting() -> None:
    """`sequential=True` dispatches real key events, and `gone=True` waits for a spinner to go."""
    server, port = await _serve_html(_TYPEAHEAD_PAGE)
    async with server:
        browser = PlaywrightBrowser[object](block_private_addresses=False)
        _, waited, _, after_fill, _, after_keys = await _run_tools(
            browser,
            [
                ('navigate', {'url': f'http://127.0.0.1:{port}/'}),
                ('wait_for', {'text': 'Loading', 'gone': True, 'timeout_ms': 5000}),
                ('type_text', {'selector': '#q', 'text': 'ada'}),
                ('get_text', {'selector': '#hint'}),
                ('type_text', {'selector': '#q', 'text': 'ada', 'sequential': True}),
                ('get_text', {'selector': '#hint'}),
            ],
        )
        assert 'timed out' not in waited, waited
        assert 'TYPEAHEAD-FIRED' not in after_fill, f'fill dispatched key events after all: {after_fill}'
        assert 'TYPEAHEAD-FIRED' in after_keys, f'sequential typing dispatched no key events: {after_keys}'
    print('typing and waiting ok -- key events dispatched, spinner waited out')


async def _serve_websocket_secret() -> tuple[asyncio.Server, int]:
    """Serve `_SECRET` over a real WebSocket on a loopback port.

    Hand-rolled rather than pulled from a library: the handshake is a hash and one
    unmasked text frame, and the script stays dependency-free like the HTTP
    servers above.
    """

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        key = b''
        while True:
            line = await reader.readline()
            if line in (b'\r\n', b''):
                break
            if line.lower().startswith(b'sec-websocket-key:'):
                key = line.split(b':', 1)[1].strip()
        accept = base64.b64encode(hashlib.sha1(key + _WS_GUID).digest())
        writer.write(
            b'HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n'
            b'Connection: Upgrade\r\nSec-WebSocket-Accept: ' + accept + b'\r\n\r\n'
        )
        payload = _SECRET.encode()
        writer.write(bytes([0x81, len(payload)]) + payload)
        await writer.drain()
        await asyncio.sleep(5)

    server = await asyncio.start_server(handle, '127.0.0.1', 0)
    return server, server.sockets[0].getsockname()[1]


async def _check_websocket_block() -> None:
    """A WebSocket to a private address is refused; a public one still works.

    `context.route` never sees a WebSocket, so this exercises the separate socket
    guard. The proof that the guard refused it is the recorded entry, not the
    failure itself: Chromium's own private-network protection also stops a public
    page from reaching loopback, so a socket that merely fails proves nothing.
    """
    server, port = await _serve_websocket_secret()
    async with server:
        script = (
            'new Promise(resolve => {'
            f"const s = new WebSocket('ws://127.0.0.1:{port}/');"
            "s.onmessage = e => resolve('received:' + e.data);"
            "s.onerror = () => resolve('error');"
            "s.onclose = () => resolve('closed');"
            "setTimeout(() => resolve('timeout'), 3000)})"
        )
        _, result, network = await _run_tools(
            PlaywrightBrowser[object](),
            [
                ('navigate', {'url': 'http://example.com/'}),
                ('execute_js', {'script': script, 'timeout_ms': 8000}),
                ('network_requests', {}),
            ],
        )
        assert _SECRET not in result, result
        assert f'request_blocked ws://127.0.0.1:{port}' in network, f'the socket guard did not refuse it: {network}'
    print('websocket to a private address refused ok (recorded by the guard)')

    echo = (
        'new Promise(resolve => {'
        "const s = new WebSocket('wss://echo.websocket.org/');"
        "s.onopen = () => s.send('ping');"
        "s.onmessage = e => resolve('received');"
        "s.onerror = () => resolve('error');"
        "setTimeout(() => resolve('timeout'), 8000)})"
    )
    _, echoed = await _run_tools(
        PlaywrightBrowser[object](),
        [('navigate', {'url': 'https://example.com/'}), ('execute_js', {'script': echo, 'timeout_ms': 15000})],
    )
    assert echoed == 'received', f'a permitted socket did not survive the guard: {echoed}'
    print('public websocket still connects through the guard ok')


async def _main() -> None:
    """Run every smoke scenario in sequence."""
    await _check_navigate()
    await _check_allowlist_bounce()
    await _check_private_address_block()
    await _check_private_name_block()
    await _check_blocked_redirect_message()
    await _check_page_request_egress()
    await _check_storage_state_round_trip()
    await _check_embedded_frame()
    await _check_browser_event_log()
    await _check_websocket_block()
    await _check_tabs()
    await _check_dialogs()
    await _check_typing_and_waiting()
    await _check_cdp_attach()
    print('all checks passed')


if __name__ == '__main__':
    asyncio.run(_main())
