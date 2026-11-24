"""Shared fixtures for the CDPBrowser tests: a fake Chrome, and no live DNS."""

from __future__ import annotations

import importlib.util

# `websockets` comes with the `cdp-browser` extra, so an install without it can't import these modules. Ignore them
# at collection. A conditional expression rather than an `if` statement: branch coverage traces statement arcs, and
# no single environment can take both arms of an install-dependent branch.
collect_ignore = ['test_cdp_browser.py', '_fake_chrome.py'] if importlib.util.find_spec('websockets') is None else []

if not collect_ignore:  # pragma: no branch
    import os
    from collections.abc import AsyncGenerator

    import pytest
    from websockets.asyncio.server import serve

    from pydantic_ai_harness import _browser_egress as egress_module
    from pydantic_ai_harness.cdp_browser import _client as client_module

    from ._fake_chrome import FakeChrome

    @pytest.fixture(autouse=True)
    def no_real_dns(monkeypatch: pytest.MonkeyPatch) -> None:
        """Answer every lookup the private-address block makes with a public address, so no test issues live DNS.

        A lookup that does not answer is a refusal, so returning nothing here would refuse every request. The
        cache is module-level, so it is emptied between tests.
        """
        egress_module.resolution_cache.clear()

        async def public_address(host: str) -> tuple[str, ...]:
            return ('93.184.216.34',)

        monkeypatch.setattr(egress_module, 'getaddrinfo', public_address)

    @pytest.fixture
    async def chrome(monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[FakeChrome]:
        """A fake Chrome behind a real websocket, which every launch in the test connects to."""
        fake = FakeChrome()
        async with serve(fake.handle, '127.0.0.1', 0) as server:
            address: tuple[str, int] = next(iter(server.sockets)).getsockname()
            fake.port = address[1]
            # No test depends on a Chrome being installed; the ones about finding it unset this.
            monkeypatch.setenv('CHROME_PATH', '/fake/chrome')
            monkeypatch.setattr(client_module, 'start_process', fake.start)
            monkeypatch.setattr(os, 'killpg', fake.killpg)
            yield fake
