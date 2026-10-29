"""Shared fixtures for OpenSandbox tests."""

from __future__ import annotations

import importlib.util
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

_HAS_OPENSANDBOX = importlib.util.find_spec('opensandbox') is not None
collect_ignore = (
    []
    if _HAS_OPENSANDBOX
    else ['test_backend.py', 'test_conformance.py', 'test_opensandbox_live.py', 'test_opensandbox_sandbox.py']
)

if TYPE_CHECKING or _HAS_OPENSANDBOX:  # pragma: no branch - the SDK-installed and slim jobs take opposite branches
    import httpx
    from opensandbox.config import ConnectionConfig

    from .fake_opensandbox import API_KEY, DOMAIN, FakeOpenSandbox, OwnedTransport, reap


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Gate the live tier, before any live fixture can create a sandbox.

    It runs only with `PYDANTIC_AI_HARNESS_OPENSANDBOX_LIVE=1` and `OPEN_SANDBOX_DOMAIN` naming a server.
    Otherwise it skips, unless `OPENSANDBOX_REQUIRE_LIVE=1` (CI's live job sets it), where it fails instead:
    a live job that tested nothing must not pass as a silent skip.
    """
    if item.get_closest_marker('opensandbox_live') is None:
        return
    opted_in = os.getenv('PYDANTIC_AI_HARNESS_OPENSANDBOX_LIVE') == '1'
    if opted_in and os.getenv('OPEN_SANDBOX_DOMAIN'):  # pragma: no cover - live job
        return
    reason = 'requires PYDANTIC_AI_HARNESS_OPENSANDBOX_LIVE=1 and OPEN_SANDBOX_DOMAIN naming a server'
    if os.getenv('OPENSANDBOX_REQUIRE_LIVE') == '1':  # pragma: no cover - live job
        pytest.fail(f'OPENSANDBOX_REQUIRE_LIVE=1 but the live tier cannot run: it {reason}.')
    pytest.skip(reason)


@pytest.fixture(autouse=True)
def _no_real_server(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the SDK's default server at a name that cannot resolve, for every test but the live tier.

    The SDK falls back to `localhost:8080`, where a developer may run a real server: a unit test that forgets
    the fake's `connection` must fail instead of creating real sandboxes there.
    """
    if 'opensandbox_live' in request.keywords:  # pragma: no cover - live tier runs without coverage
        return
    monkeypatch.setenv('OPEN_SANDBOX_DOMAIN', 'unit-tests-use-the-fake-server.invalid')
    monkeypatch.delenv('OPEN_SANDBOX_API_KEY', raising=False)


if _HAS_OPENSANDBOX:  # pragma: no branch - the fixture cannot be defined without the SDK the fake runs

    @pytest.fixture
    async def server(tmp_path_factory: pytest.TempPathFactory) -> AsyncIterator[FakeOpenSandbox]:
        """The fake server; pass its `connection` to the backend or capability."""
        fake = FakeOpenSandbox(pid_dir=str(tmp_path_factory.mktemp('pids')))
        yield fake
        await reap(fake)

    @pytest.fixture
    def owned_connection(server: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch) -> ConnectionConfig:
        """A connection without a transport of its own, as an application passes one: the SDK opens and closes
        its own transports, which reach the fake server and count on it."""

        def transport(**_: object) -> OwnedTransport:
            return OwnedTransport(server)

        monkeypatch.setattr(httpx, 'AsyncHTTPTransport', transport)
        return ConnectionConfig(domain=DOMAIN, api_key=API_KEY)

    @pytest.fixture
    def working_dir(tmp_path: Path) -> str:
        """A host directory for the fake's commands to run in; resolved, as `pwd -P` reports it."""
        return str(tmp_path.resolve())
