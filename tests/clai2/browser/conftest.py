"""Real-browser regressions run once in the dedicated CLAI 2 web CI job."""

from collections.abc import AsyncIterator, Callable, Generator
from contextlib import AbstractContextManager, contextmanager

import pytest
from playwright.async_api import Browser, Page, async_playwright

from pydantic_clai2.builtin_plugins.web.app import create_app
from pydantic_clai2.builtin_plugins.web.server import WebServer
from tests.clai2.test_web import TOKEN
from tests.clai2.test_web_sessions import Fleet


@pytest.fixture(scope='module')
def anyio_backend() -> str:
    """Keep the browser's shared asynchronous fixtures on one asyncio runner."""
    return 'asyncio'


@contextmanager
def served(fleet: Fleet) -> Generator[str]:
    """Serve `fleet`'s web UI on a free loopback port, at the address this yields."""
    server = fleet.portal.call(
        lambda: WebServer.serve(
            lambda port: create_app(service=fleet.service, token=TOKEN, hosts=frozenset({f'127.0.0.1:{port}'})),
            host='127.0.0.1',
            port=0,
        )
    )
    try:
        yield f'http://127.0.0.1:{server.port}'
    finally:
        fleet.portal.call(server.stop)


Serve = Callable[[Fleet], AbstractContextManager[str]]


@pytest.fixture(scope='session')
def serve() -> Serve:
    """`served`, for a module's fixture that starts its own fleet."""
    return served


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
