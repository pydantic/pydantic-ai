"""Plugins' side panels in the web UI, and whether each page can be framed there."""

from pathlib import Path

import anyio
import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_clai2.builtin_plugins.web import framing, service as service_module
from pydantic_clai2.builtin_plugins.web.framing import FrameCheck, fetch_headers, frameable
from pydantic_clai2.builtin_plugins.web.protocol import PROTOCOL, TurnEnded
from pydantic_clai2.config import PluginSettings
from tests.clai2.test_web import TOKEN, MakeWeb, Web, make_web as make_web, receive_until

UI = 'http://127.0.0.1:8123'
LOGFIRE_CSP = (
    "default-src 'self'; frame-ancestors 'self' https://pydantic.dev https://ai.pydantic.dev https://docs.pydantic.dev "
    'https://logfire.pydantic.dev https://pydantic-dev-site.pages.dev https://*.pydantic-dev-site.pages.dev '
    'http://localhost;'
)
"""What Logfire sends today: `http://localhost` without a port means port 80 alone."""


@pytest.mark.parametrize(
    'headers, page, expected',
    [
        pytest.param({}, 'https://example.com/', True, id='no-headers'),
        pytest.param({'x-frame-options': 'DENY'}, 'https://example.com/', False, id='xfo-deny'),
        pytest.param({'x-frame-options': 'sameorigin'}, 'https://example.com/', False, id='xfo-sameorigin'),
        pytest.param({'x-frame-options': 'SAMEORIGIN'}, f'{UI}/page', True, id='xfo-sameorigin-same'),
        pytest.param({'x-frame-options': 'ALLOW-FROM http://127.0.0.1:8123'}, 'https://a.example/', True, id='xfo-old'),
        pytest.param({'x-frame-options': 'ALLOWALL'}, 'https://example.com/', True, id='xfo-allowall'),
        pytest.param({'x-frame-options': 'deny, sameorigin'}, f'{UI}/', False, id='xfo-conflicting'),
        pytest.param({'content-security-policy': LOGFIRE_CSP}, 'https://logfire-us.pydantic.dev/', False, id='logfire'),
        pytest.param(
            {'content-security-policy': 'frame-ancestors http://127.0.0.1:*', 'x-frame-options': 'DENY'},
            'https://logfire-us.pydantic.dev/',
            True,
            id='loopback-any-port-beats-xfo',
        ),
        pytest.param({'content-security-policy': "frame-ancestors 'none'"}, f'{UI}/', False, id='none'),
        pytest.param({'content-security-policy': "frame-ancestors 'self'"}, f'{UI}/x', True, id='self-same'),
        pytest.param({'content-security-policy': "frame-ancestors 'self'"}, 'https://a.example/', False, id='self'),
        pytest.param({'content-security-policy': 'frame-ancestors *'}, 'https://a.example/', True, id='star'),
        pytest.param(
            {'content-security-policy': 'frame-ancestors http:'}, 'https://a.example/', True, id='http-scheme'
        ),
        pytest.param({'content-security-policy': 'frame-ancestors https:'}, 'https://a.example/', False, id='https'),
        pytest.param(
            {'content-security-policy': 'frame-ancestors http://127.0.0.1:8123'}, 'https://a.example/', True, id='port'
        ),
        pytest.param(
            {'content-security-policy': 'frame-ancestors http://127.0.0.1:9000'},
            'https://a.example/',
            False,
            id='other',
        ),
        pytest.param(
            {'content-security-policy': 'frame-ancestors http://127.0.0.1'}, 'https://a.example/', False, id='port-80'
        ),
        pytest.param(
            {'content-security-policy': 'frame-ancestors HTTP://127.0.0.1:*/ui/'}, 'https://a.example/', True, id='path'
        ),
        pytest.param(
            {'content-security-policy': 'frame-ancestors http://*.0.0.1:*'}, 'https://a.example/', True, id='wildcard'
        ),
        pytest.param(
            {'content-security-policy': 'frame-ancestors http://*.example.com:*'}, 'https://a.example/', False, id='sub'
        ),
        pytest.param(
            {'content-security-policy': 'frame-ancestors 127.0.0.1:*'}, 'https://a.example/', False, id='bare-on-https'
        ),
        pytest.param(
            {'content-security-policy': 'frame-ancestors 127.0.0.1:*'}, 'http://a.example/', True, id='bare-on-http'
        ),
        pytest.param({'content-security-policy': 'frame-ancestors [::1]'}, 'http://a.example/', False, id='unparsed'),
        pytest.param(
            {'content-security-policy': ';; frame-ancestors *, frame-ancestors https://a.example'},
            'https://a.example/',
            False,
            id='every-policy-must-allow',
        ),
        pytest.param({'content-security-policy': "default-src 'self'"}, 'https://a.example/', True, id='no-directive'),
    ],
)
def test_frameable(headers: dict[str, str], page: str, expected: bool) -> None:
    assert frameable(httpx.Headers(headers), page=page, ancestor=UI) is expected


def test_an_ancestor_without_a_host_matches_no_host_source() -> None:
    headers = httpx.Headers({'content-security-policy': 'frame-ancestors http://127.0.0.1:*'})
    assert not frameable(headers, page='https://a.example/', ancestor='http:')


async def test_headers_are_fetched_once_per_page_that_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(framing, 'FETCH_TIMEOUT', 0.05)
    fetched: list[str] = []

    async def fetch(url: str) -> httpx.Headers:
        fetched.append(url)
        if 'down' in url:
            raise httpx.ConnectError('unreachable')
        if 'odd' in url:
            raise httpx.InvalidURL('no such address')
        if 'slow' in url:
            await anyio.sleep(10)
        return httpx.Headers({'content-security-policy': 'frame-ancestors http://127.0.0.1:*'})

    check = FrameCheck(fetch)
    assert await check('https://up.example/live?turn=1', ancestor=UI)
    assert await check('https://up.example/live?turn=2#top', ancestor=UI)
    assert not await check('https://up.example/live', ancestor='http://localhost:8123')
    assert not await check('https://down.example/', ancestor=UI)
    assert not await check('https://down.example/?again', ancestor=UI)
    assert not await check('https://odd.example/', ancestor=UI)
    assert not await check('https://slow.example/', ancestor=UI)
    # The web UI's own policy frames `http` pages only from this machine: those are not even read.
    assert not await check('http://intranet.example/', ancestor=UI)
    # A page that could not be read is not framed, and is read again next time.
    assert fetched == [
        'https://up.example/live?turn=1',
        'https://down.example/',
        'https://down.example/?again',
        'https://odd.example/',
        'https://slow.example/',
    ]


async def test_fetch_headers_reads_a_page_that_answers_itself() -> None:
    requested: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        if request.url.path == '/old':
            return httpx.Response(302, headers={'location': 'http://127.0.0.1:8080/action'})
        return httpx.Response(200, headers={'x-frame-options': 'DENY'}, text='<html>')

    transport = httpx.MockTransport(answer)
    headers = await fetch_headers('https://a.example/new', transport=transport)
    assert headers['x-frame-options'] == 'DENY'
    # A redirect is not followed, so a page cannot send the server's request on to this machine.
    with pytest.raises(httpx.HTTPStatusError, match='answered 302'):
        await fetch_headers('https://a.example/old', transport=transport)
    assert requested == ['/new', '/old']


PANELS = '''
import asyncio

from pydantic_clai2.plugins import PanelView, Plugin, SidePanel

NOT_PAGES = {
    'local': 'javascript:alert(1)',
    'unclosed': 'http://[::1/',
    'hostless': 'https://',
    'spaced': 'http://exa mple.com/',
}


class Traces(Plugin):
    """Side panels that follow the turns."""

    def __init__(self, host, settings):
        super().__init__(host, settings)
        self.turns = 0

    def get_side_panels(self):
        return (
            SidePanel(id='live', title='Live', icon='traces', view=self.live),
            SidePanel(id='setup', title='Setup', view=self.setup),
            SidePanel(id='broken', title='Broken', view=self.broken),
            SidePanel(id='silent', title='Silent', view=self.silent),
            SidePanel(id='slow', title='Slow', view=self.slow),
            *(SidePanel(id=name, title=name.title(), view=self.offer(url)) for name, url in NOT_PAGES.items()),
        )

    async def on_turn_start(self, event):
        self.turns += 1

    async def live(self):
        return PanelView(url=f'https://traces.example/live?turn={self.turns}')

    async def setup(self):
        return PanelView(note='Set it up first.')

    async def broken(self):
        raise RuntimeError('no traces')

    async def silent(self):
        raise LookupError()

    async def slow(self):
        await asyncio.sleep(60)

    def offer(self, url):
        async def view():
            return PanelView(url=url)

        return view


class Framed(Plugin):
    """A panel whose page allows this UI to frame it."""

    def get_side_panels(self):
        return (
            SidePanel(id='page', title='Page', view=self.page),
            SidePanel(id='intranet', title='Intranet', view=self.intranet),
        )

    async def intranet(self):
        return PanelView(url='http://intranet.example/page')

    async def page(self):
        return PanelView(url='http://127.0.0.1:9999/page')
'''


def panel_plugins(tmp_path: Path) -> list[PluginSettings]:
    path = tmp_path / 'traces.py'
    path.write_text(PANELS)
    return [PluginSettings(id=name.lower(), factory=f'traces:{name}', path=str(path)) for name in ('Traces', 'Framed')]


def panels(web_: Web) -> list[dict[str, JsonValue]]:
    response = web_.get('/api/panels')
    assert response.status_code == 200
    return TypeAdapter(list[dict[str, JsonValue]]).validate_python(response.json())


def test_side_panels_follow_the_turns(make_web: MakeWeb, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each loaded plugin's panels, as their views say now, framed only where the page allows it."""
    monkeypatch.setattr(service_module, 'VIEW_TIMEOUT', 0.05)
    web_ = make_web(Agent(TestModel(call_tools=[], custom_output_text='ok')), extra=panel_plugins(tmp_path))
    fetched: list[str] = []

    async def fetch(url: str) -> httpx.Headers:
        fetched.append(url)
        # The page's origin is the address it was opened at: the test client's.
        allowed = 'https://pydantic.dev' if url.startswith('https://traces') else 'http://testserver'
        return httpx.Headers({'content-security-policy': f'frame-ancestors {allowed}'})

    web_.plugin.service.frames = FrameCheck(fetch)
    traces = {
        'plugin': 'traces',
        'id': 'live',
        'title': 'Live',
        'icon': 'traces',
        'url': 'https://traces.example/live?turn=0',
        'note': '',
        'embed': False,
    }
    rest = [
        {**traces, 'id': 'setup', 'title': 'Setup', 'icon': 'panel', 'url': None, 'note': 'Set it up first.'},
        {**traces, 'id': 'broken', 'title': 'Broken', 'icon': 'panel', 'url': None, 'note': 'Broken failed: no traces'},
        {
            **traces,
            'id': 'silent',
            'title': 'Silent',
            'icon': 'panel',
            'url': None,
            'note': 'Silent failed: LookupError',
        },
        {**traces, 'id': 'slow', 'title': 'Slow', 'icon': 'panel', 'url': None, 'note': 'Slow did not answer in time.'},
        *(
            {
                **traces,
                'id': name,
                'title': name.title(),
                'icon': 'panel',
                'url': None,
                'note': f'{name.title()} offered a page that is not on the web; only http and https open here.',
            }
            for name in ('local', 'unclosed', 'hostless', 'spaced')
        ),
        {**traces, 'plugin': 'framed', 'id': 'page', 'title': 'Page', 'icon': 'panel'}
        | {'url': 'http://127.0.0.1:9999/page', 'embed': True},
        # The page's own policy frames `http` only from this machine, whatever this page allows.
        {**traces, 'plugin': 'framed', 'id': 'intranet', 'title': 'Intranet', 'icon': 'panel'}
        | {'url': 'http://intranet.example/page'},
    ]
    assert panels(web_) == [traces, *rest]
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        socket.receive_text()
        running = web_.run()
        web_.post('/api/input', {'text': 'go'})
        receive_until(socket, TurnEnded)
        assert panels(web_) == [{**traces, 'url': 'https://traces.example/live?turn=1'}, *rest]
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)
    # Each page's headers were read once, however often its query changed.
    assert fetched == ['https://traces.example/live?turn=0', 'http://127.0.0.1:9999/page']


def test_the_page_may_frame_side_panels(make_web: MakeWeb) -> None:
    web_ = make_web(Agent(TestModel()))
    policy = web_.client.get(f'/?token={TOKEN}').headers['content-security-policy']
    assert "frame-src 'self' https: http://127.0.0.1:* http://localhost:*;" in policy
    assert panels(web_) == []
    web_.plugin.host.plugins = None
    assert panels(web_) == []
