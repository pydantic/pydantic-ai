"""The `updates` plugin shows a newer final release in the status row, and never blocks or fails startup."""

import io
from importlib import metadata
from pathlib import Path

import anyio
import httpx2
import pytest
from cassetter import use_cassette
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_clai2._app import DEFAULT_PLUGINS, create_shell
from pydantic_clai2.builtin_plugins import updates
from pydantic_clai2.config import Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import PluginHost, SessionEnd, SessionStart


def host(*, terminal: bool = True) -> PluginHost[None]:
    return PluginHost(name='updates', console=Console(file=io.StringIO(), force_terminal=terminal), settings={})


def installed(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    def fake_version(name: str) -> str:
        assert name == 'pydantic-clai2'
        return version

    monkeypatch.setattr(updates.metadata, 'version', fake_version)


def published(monkeypatch: pytest.MonkeyPatch, version: str | None) -> list[str]:
    checks: list[str] = []

    async def fake_latest() -> str | None:
        checks.append('checked')
        return version

    monkeypatch.setattr(updates, 'latest_version', fake_latest)
    return checks


async def lifecycle(plugin: PluginHost[None]) -> str:
    """Load the plugin, let the background check finish, read the status row, then unload."""
    session = SessionStart(agent=Agent(TestModel()), settings=Settings())
    for handler in plugin.handlers:
        await handler(session)
    await anyio.wait_all_tasks_blocked()
    text = ''.join(segment() for segment in plugin.status_segments)
    for handler in plugin.handlers:
        await handler(SessionEnd(reason='exit'))
    return text


@pytest.mark.parametrize(
    ('version', 'expected'),
    [
        ('0.51.0', (0, 51)),
        ('0.51', (0, 51)),
        ('1.0.0', (1,)),
        ('0.0', (0,)),
        ('0.52.10', (0, 52, 10)),
        ('0.52.0rc1', None),
        ('0.52.0.post1', None),
        ('0.51.1.dev487+a8ded2e50', None),
        ('0.51.0+local', None),
        ('v0.51.0', None),
        ('', None),
    ],
)
def test_final_release(version: str, expected: tuple[int, ...] | None) -> None:
    assert updates.final_release(version) == expected


@pytest.mark.parametrize(
    ('current', 'latest', 'expected'),
    [
        ('0.51.0', '0.52.0', 'clai2 0.52.0 available'),
        ('0.51.0', '0.51.10', 'clai2 0.51.10 available'),
        ('0.51.0', '0.51', ''),
        ('0.52.0', '0.51.9', ''),
        ('0.51.0', '0.52.0rc1', ''),
        ('0.51.0', None, ''),
    ],
)
async def test_status_row(current: str, latest: str | None, expected: str, monkeypatch: pytest.MonkeyPatch) -> None:
    installed(monkeypatch, current)
    checks = published(monkeypatch, latest)
    plugin = host()
    updates.activate(plugin)
    assert ''.join(segment() for segment in plugin.status_segments) == ''
    assert await lifecycle(plugin) == expected
    assert checks == ['checked']


@pytest.mark.parametrize('version', ['0.51.1.dev487+a8ded2e50', None])
async def test_source_checkouts_and_missing_metadata_never_check(
    version: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if version is None:

        def missing(name: str) -> str:
            raise metadata.PackageNotFoundError(name)

        monkeypatch.setattr(updates.metadata, 'version', missing)
    else:
        installed(monkeypatch, version)
    checks = published(monkeypatch, '99.0.0')
    plugin = host()
    updates.activate(plugin)
    assert plugin.handlers == []
    assert plugin.status_segments == []
    assert checks == []


def test_redirected_output_never_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    installed(monkeypatch, '0.51.0')
    checks = published(monkeypatch, '99.0.0')
    plugin = host(terminal=False)
    updates.activate(plugin)
    assert plugin.handlers == []
    assert checks == []


async def test_unload_cancels_a_pending_check(monkeypatch: pytest.MonkeyPatch) -> None:
    installed(monkeypatch, '0.51.0')
    started = anyio.Event()
    cleaned = anyio.Event()

    async def hang() -> str | None:
        started.set()
        try:
            await anyio.sleep_forever()
        finally:
            cleaned.set()
        raise AssertionError('unreachable')  # pragma: no cover

    monkeypatch.setattr(updates, 'latest_version', hang)
    plugin = host()
    updates.activate(plugin)
    session = SessionStart(agent=Agent(TestModel()), settings=Settings())
    for handler in plugin.handlers:
        await handler(session)
    await started.wait()
    for handler in plugin.handlers:
        await handler(SessionEnd(reason='exit'))
    assert cleaned.is_set()


async def test_unload_before_start_is_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    installed(monkeypatch, '0.51.0')
    plugin = host()
    updates.activate(plugin)
    for handler in plugin.handlers:
        await handler(SessionEnd(reason='error'))


async def test_startup_does_not_wait_for_pypi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    installed(monkeypatch, '0.51.0')
    release = anyio.Event()

    async def slow() -> str | None:
        await release.wait()
        return '0.52.0'

    monkeypatch.setattr(updates, 'latest_version', slow)
    shell = create_shell(
        Agent(TestModel()),
        deps=None,
        plugins=(),
        usage_limits=None,
        settings=None,
        project=ProjectSettings(),
        console=Console(file=io.StringIO(), force_terminal=True),
        store=SettingsStore(tmp_path / 'settings.db'),
        builtin_plugins=[plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'updates'],
    )
    await shell.loader.load_all()
    assert [segment() for segment in shell.loader.status_segments()] == ['']
    release.set()
    await anyio.wait_all_tasks_blocked()
    assert [segment() for segment in shell.loader.status_segments()] == ['clai2 0.52.0 available']
    await shell.loader.command(['disable', 'updates'])
    assert shell.loader.status_segments() == []
    await shell.loader.close('exit')


@pytest.mark.vcr
async def test_recorded_pypi_lookup() -> None:
    """Recorded against the real PyPI JSON API; `pydantic-clai2` had no release on PyPI yet."""
    with use_cassette(Path(__file__).parent / 'cassettes/updates_pypi.yaml', record_mode='none'):
        assert await updates.latest_version() is None


@pytest.mark.parametrize(
    ('response', 'expected'),
    [
        (httpx2.Response(200, json={'info': {'version': '0.52.0', 'yanked': False}}), '0.52.0'),
        (httpx2.Response(200, json={'info': {'version': '0.52.0'}}), '0.52.0'),
        (httpx2.Response(200, json={'info': {'version': '0.52.0', 'yanked': True}}), None),
        (httpx2.Response(200, json={'releases': {}}), None),
        (httpx2.Response(200, text='<html>maintenance</html>'), None),
        (httpx2.Response(404, json={'message': 'Not Found'}), None),
        (httpx2.Response(301, headers={'Location': 'https://untrusted.example/'}), None),
    ],
)
async def test_pypi_responses(response: httpx2.Response, expected: str | None) -> None:
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return response

    assert await updates.latest_version(transport=httpx2.MockTransport(respond)) == expected
    assert [str(request.url) for request in requests] == [updates.PYPI_URL]


async def test_network_failure_is_quiet() -> None:
    def fail(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError('offline', request=request)

    assert await updates.latest_version(transport=httpx2.MockTransport(fail)) is None
