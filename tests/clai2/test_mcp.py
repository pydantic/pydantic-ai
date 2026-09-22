"""MCP plugin configuration, discovery, and core-managed tool execution."""

import io
import os
import sys
from pathlib import Path

import httpx
import pytest
from pydantic import JsonValue, ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from rich.console import Console

from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.mcp import activate, http_client
from pydantic_clai2.plugin_loader import PluginLoader
from pydantic_clai2.plugins import PluginHost, SessionStart
from pydantic_clai2.settings_store import SettingsStore


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def make_host(settings: dict[str, JsonValue]) -> PluginHost[None]:
    host: PluginHost[None] = PluginHost(name='mcp', console=Console(file=io.StringIO()), settings=settings)
    activate(host)
    return host


async def test_http_client_rejects_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[httpx.Request] = []

    async def respond(transport: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(307, headers={'location': 'http://127.0.0.1/private'}, request=request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, 'handle_async_request', respond)
    async with http_client(headers={'X-Test': 'present'}) as client:
        response = await client.post('https://example.com/mcp')
        assert response.status_code == 307
        assert len(requests) == 1
        assert requests[0].headers['X-Test'] == 'present'
        assert client.timeout.read == 300
    async with http_client(timeout=httpx.Timeout(5), auth=httpx.BasicAuth('user', 'password')) as client:
        assert client.timeout.read == 5
        assert isinstance(client.auth, httpx.BasicAuth)


async def test_empty_builtin() -> None:
    declaration = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'mcp')
    assert declaration.enabled
    assert declaration.factory == 'pydantic_clai2.mcp'
    host = make_host({})
    assert not host.capabilities
    assert 'No MCP servers configured' in await host.commands.execute_async('/mcp')
    assert 'Usage:' in await host.commands.execute_async('/mcp nope')
    assert 'Unknown or disabled' in await host.commands.execute_async('/mcp tools missing')
    command = next(iter(host.commands))
    assert tuple(command.complete([])) == ('list', 'tools')
    assert tuple(command.complete(['tools', ''])) == ()


async def test_configuration_is_lazy_and_listing_redacts_secrets() -> None:
    host = make_host(
        {
            'servers': {
                'remote': {'transport': 'http', 'url': 'https://example.com/mcp?secret=value'},
                'local': {'transport': 'stdio', 'command': 'not-a-real-program', 'env': {'TOKEN': 'secret'}},
                'off': {'transport': 'stdio', 'command': 'not-a-real-program', 'enabled': False},
            }
        }
    )
    assert len(host.capabilities) == 2
    assert await host.commands.execute_async('/mcp list') == (
        'remote: http, enabled\nlocal: stdio, enabled\noff: stdio, disabled'
    )
    assert 'Unknown or disabled' in await host.commands.execute_async('/mcp tools off')
    assert tuple(next(iter(host.commands)).complete(['tools', ''])) == ('remote', 'local')


@pytest.mark.parametrize(
    'server',
    [
        {'transport': 'stdio', 'command': ''},
        {'transport': 'stdio', 'command': 'python', 'typo': True},
        {'transport': 'http', 'url': 'file:///tmp/server'},
        {'transport': 'http'},
        {'transport': 'sse', 'url': 'https://example.com'},
    ],
)
def test_invalid_configuration(server: JsonValue) -> None:
    with pytest.raises(ValidationError):
        make_host({'servers': {'test': server}})


@pytest.mark.parametrize('name', ['bad-name', 'a_b', '0server'])
def test_invalid_name(name: str) -> None:
    with pytest.raises(ValidationError):
        make_host({'servers': {name: {'transport': 'stdio', 'command': 'python'}}})


async def test_loader_persistence_disable_and_project_trust(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    builtin = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'mcp')
    project = PluginSettings(
        id='mcp',
        factory='pydantic_clai2.mcp',
        enabled=False,
        settings={'servers': {'local': {'transport': 'stdio', 'command': 'not-a-program'}}},
    )
    commands = Commands()
    agent = Agent(TestModel(), deps_type=type(None))
    loader: PluginLoader[None] = PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=commands,
        session_start=lambda: SessionStart(agent=agent, settings=store.load()),
        builtin=(builtin,),
        project=(project,),
    )
    try:
        await loader.load_all()
        assert not loader.capabilities()
        assert not list(commands)
        await loader.enable('mcp')
        assert len(loader.capabilities()) == 1
        assert await commands.execute_async('/mcp') == 'local: stdio, enabled'
        assert store.plugins()[0].settings == project.settings
        await loader.disable('mcp')
        assert not loader.capabilities()
        assert not list(commands)
        await loader.enable('mcp')
        await loader.reload('mcp')
        assert len(loader.capabilities()) == 1
        assert len(list(commands)) == 1
    finally:
        await loader.close('exit')
    assert not list(commands)
    assert SettingsStore(store.path).plugins()[0].settings == project.settings


@pytest.mark.parametrize('with_tool', [True, False])
async def test_real_stdio_discovery_and_agent_run(tmp_path: Path, with_tool: bool) -> None:
    script = tmp_path / 'server.py'
    pid_file = tmp_path / 'server.pid'
    script.write_text(
        'import os\nfrom pathlib import Path\n'
        f'Path({str(pid_file)!r}).write_text(str(os.getpid()))\n'
        'from mcp.server.fastmcp import FastMCP\n'
        'server = FastMCP("test")\n'
        + ('@server.tool()\ndef ping() -> str:\n    return "pong"\n' if with_tool else '')
        + 'server.run()\n'
    )
    host = make_host({'servers': {'local': {'transport': 'stdio', 'command': sys.executable, 'args': [str(script)]}}})
    expected = 'local_ping' if with_tool else 'No tools provided by local.'
    assert await host.commands.execute_async('/mcp tools local') == expected
    if sys.platform != 'win32':
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)
    # A second connection exercises cleanup after the discovery context closes.
    result = await Agent(TestModel(), deps_type=type(None), capabilities=host.capabilities).run(
        'Use the available tools.'
    )
    if with_tool:
        assert 'pong' in result.output
    else:
        assert result.output == 'success (no tool calls)'
    if sys.platform != 'win32':
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)
