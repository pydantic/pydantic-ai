"""Exercise the shipped UI and its real HTTP/WebSocket boundary without provider requests."""

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

import pytest
from anyio.from_thread import start_blocking_portal
from playwright.async_api import Browser, Dialog, Page, Route, async_playwright, expect

from pydantic_ai import Agent, models
from pydantic_ai.models.test import TestModel
from pydantic_clai2.builtin_plugins.web.app import create_app
from pydantic_clai2.builtin_plugins.web.server import WebServer
from tests.clai2.test_web import TOKEN
from tests.clai2.test_web_sessions import Fleet

HOSTILE = """Markdown survived

<script>window.alert('script')</script>
<img src="/missing" onerror="window.alert('image')">
<svg onload="window.alert('svg')"><a href="javascript:alert('svg-link')">SVG link</a></svg>
<iframe srcdoc="<script>alert('frame')</script>"></iframe>
<style>body {display:none}</style><form><input autofocus onfocus="alert('form')"></form>

[JavaScript link](javascript:alert%281%29)
[Data link](data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==)
<a href="javascript:alert(1)" onclick="alert('click')">Raw link</a>
[Safe external](https://example.invalid/guide)
"""


@dataclass
class WebUI:
    url: str
    alpha: str
    beta: str


@pytest.fixture(scope='module')
def web_ui(tmp_path_factory: pytest.TempPathFactory) -> Iterator[WebUI]:
    """Share one process, server, and two live sessions across browser cases."""
    root = tmp_path_factory.mktemp('browser-ui')
    with pytest.MonkeyPatch.context() as patch, start_blocking_portal() as portal:
        patch.setenv('HOME', str(root / 'home'))
        patch.setenv('XDG_CONFIG_HOME', str(root / 'config'))
        patch.setenv('XDG_STATE_HOME', str(root / 'state'))
        patch.setattr(models, 'ALLOW_MODEL_REQUESTS', False)
        fleet = Fleet(
            root=root,
            portal=portal,
            agents={
                'alpha': Agent(TestModel(call_tools=[], custom_output_text=HOSTILE)),
                'beta': Agent(TestModel(call_tools=[], custom_output_text='Beta answer')),
            },
        )
        try:
            alpha = fleet.new('alpha')
            beta = fleet.new('beta')
            for name, conversation, title in (
                ('alpha', alpha, 'Browser Alpha'),
                ('beta', beta, 'Browser Beta'),
            ):
                portal.call(fleet.runtimes[name].session.commit_messages, ())
                response = fleet.post(f'/api/sessions/{conversation}/rename', {'title': title})
                assert response.status_code == 204, response.text
            server = portal.call(
                lambda: WebServer.serve(
                    lambda port: create_app(service=fleet.service, token=TOKEN, hosts=frozenset({f'127.0.0.1:{port}'})),
                    host='127.0.0.1',
                    port=0,
                )
            )
            try:
                yield WebUI(url=f'http://127.0.0.1:{server.port}', alpha=alpha, beta=beta)
            finally:
                portal.call(server.stop)
        finally:
            fleet.exit()


@pytest.fixture(scope='module')
async def chromium() -> AsyncIterator[Browser]:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        try:
            yield browser
        finally:
            await browser.close()


@pytest.fixture
async def page(chromium: Browser) -> AsyncIterator[Page]:
    async with await chromium.new_context() as context:
        page = await context.new_page()
        page.set_default_timeout(10_000)
        yield page


async def test_hostile_markdown_and_external_links(page: Page, web_ui: WebUI) -> None:
    dialogs: list[str] = []

    async def dismiss_dialog(dialog: Dialog) -> None:
        dialogs.append(dialog.message)
        await dialog.dismiss()

    page.on('dialog', dismiss_dialog)
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    await page.locator('.sidebar').get_by_text('Browser Alpha', exact=True).click()
    await expect(page.locator('.session.current .session-title')).to_have_text('Browser Alpha')
    await expect(page.locator('.session.opening')).to_have_count(0)
    composer = page.get_by_role('combobox', name='Message CLAI')
    await expect(composer).to_be_enabled()
    await composer.fill('Render the hostile example')
    async with page.expect_response(f'**/api/sessions/{web_ui.alpha}/input') as sent:
        await composer.press('Enter')
    assert (await sent.value).ok
    markdown = page.locator('.markdown').filter(has_text='Markdown survived').last
    await expect(markdown).to_be_visible()
    assert (
        await markdown.locator(
            'script, iframe, style, form, input, [onclick], [onerror], [onload], [onfocus], '
            'a[href^="javascript:"], a[href^="data:"]'
        ).count()
        == 0
    )
    assert dialogs == []

    link = markdown.get_by_role('link', name='Safe external', exact=True)
    await expect(link).to_have_attribute('target', '_blank')
    await expect(link).to_have_attribute('rel', 'noreferrer noopener')
    requests: list[dict[str, str]] = []

    async def external(route: Route) -> None:
        requests.append(route.request.headers)
        assert TOKEN not in route.request.url
        await route.fulfill(status=200, content_type='text/html', body='<h1>External page</h1>')

    await page.context.route('https://example.invalid/**', external)
    async with page.expect_popup() as popup:
        await link.click()
    opened = await popup.value
    await opened.wait_for_load_state()
    assert await opened.evaluate('window.opener === null')
    assert len(requests) == 1
    assert 'referer' not in requests[0]
    assert 'x-clai-token' not in requests[0]


async def test_browser_token_host_and_websocket_origin(page: Page, web_ui: WebUI) -> None:
    for path in ('/', '/?token=wrong', '/assets/app.js', '/api/state?token=' + TOKEN):
        response = await page.goto(web_ui.url + path)
        assert response is not None and response.status == 403

    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    await expect(page.get_by_title('Connected', exact=True)).to_be_visible()
    assert await page.evaluate("fetch('/api/state').then(response => response.status)") == 403
    assert (
        await page.evaluate(
            "fetch('/api/state', {headers: {'x-clai-token': 'wrong'}}).then(response => response.status)"
        )
        == 403
    )
    for suffix in ('', '?token=wrong'):
        assert (
            await page.evaluate(
                """suffix => new Promise(resolve => {
              const socket = new WebSocket(`ws://${location.host}/ws${suffix}`);
              socket.onopen = () => { socket.close(); resolve('opened'); };
              socket.onclose = () => resolve('refused');
            })""",
                suffix,
            )
            == 'refused'
        )

    # The same loopback server reached by another hostname supplies a real foreign origin.
    response = await page.goto(web_ui.url.replace('127.0.0.1', 'localhost') + f'/?token={TOKEN}')
    assert response is not None and response.status == 403
    assert (
        await page.evaluate(
            """url => new Promise(resolve => {
          const socket = new WebSocket(url);
          socket.onopen = () => { socket.close(); resolve('opened'); };
          socket.onclose = () => resolve('refused');
        })""",
            web_ui.url.replace('http:', 'ws:') + '/ws?token=' + TOKEN,
        )
        == 'refused'
    )


async def test_session_switch_and_reconnect(page: Page, web_ui: WebUI) -> None:
    # Capture real browser sockets while leaving their transport and handlers intact.
    await page.add_init_script("""
      window.claiTestSockets = [];
      const OriginalWebSocket = window.WebSocket;
      window.WebSocket = class extends OriginalWebSocket {
        constructor(...args) { super(...args); window.claiTestSockets.push(this); }
      };
    """)
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    await page.locator('.sidebar').get_by_text('Browser Beta', exact=True).click()
    await expect(page.locator('.session.current .session-title')).to_have_text('Browser Beta')
    await expect(page.locator('.session.opening')).to_have_count(0)
    composer = page.get_by_role('combobox', name='Message CLAI')
    await expect(composer).to_be_enabled()
    await composer.fill('Only Beta should receive this')
    async with page.expect_response(f'**/api/sessions/{web_ui.beta}/input') as sent:
        await composer.press('Enter')
    assert (await sent.value).ok
    await expect(page.locator('.markdown').filter(has_text='Beta answer')).to_be_visible()
    await page.locator('.sidebar').get_by_text('Browser Alpha', exact=True).click()
    await expect(page.locator('.session.current .session-title')).to_have_text('Browser Alpha')
    await expect(page.locator('.session.opening')).to_have_count(0)
    await expect(composer).to_be_enabled()
    await expect(page.locator('.markdown').filter(has_text='Beta answer')).to_have_count(0)
    await page.clock.install()
    await page.evaluate('window.claiTestSockets[0].close()')
    await expect(page.get_by_title('Reconnecting…', exact=True)).to_be_visible()
    await page.clock.run_for(500)
    await page.clock.resume()
    await expect(page.get_by_title('Connected', exact=True)).to_be_visible()
    assert await page.evaluate('window.claiTestSockets.length') == 2
    await composer.fill('Alpha after reconnect')
    async with page.expect_request(f'**/api/sessions/{web_ui.alpha}/input') as request:
        await composer.press('Enter')
    assert (await request.value).post_data_json == {'text': 'Alpha after reconnect'}
    await expect(page.locator('.markdown').filter(has_text='Markdown survived').last).to_be_visible()
    await expect(page.locator('.markdown').filter(has_text='Beta answer')).to_have_count(0)
