"""MCP configuration, storage, lifecycle, and core-managed tool execution."""

import io
import logging
import os
import socket
import stat
import subprocess
import sys
from pathlib import Path
from typing import Literal, cast

import anyio
import httpx
import httpx2
import pytest
from pydantic import HttpUrl, JsonValue, ValidationError
from rich.console import Console

import pydantic_clai2.mcp._settings as mcp_settings
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.mcp import (
    HTTPServer,
    MCPPlugin,
    MCPServers,
    MCPSettings,
    MCPStore,
    SSEServer,
    StdioServer,
    http_client,
)
from pydantic_clai2.plugins import LoadedPlugin, PluginHost, SessionEnd, SessionStart, collect
from pydantic_clai2.plugins.loader import PluginLoader


def make_plugin(settings: dict[str, JsonValue], store: MCPStore | None = None) -> LoadedPlugin[None]:
    host: PluginHost[None] = PluginHost(name='mcp', console=Console(file=io.StringIO()), settings=settings)
    return collect(MCPPlugin(host, host.settings(MCPSettings), store=store))


def write_server(tmp_path: Path, *, with_tool: bool = True) -> tuple[Path, Path]:
    script = tmp_path / 'server.py'
    pid_file = tmp_path / 'server.pid'
    script.write_text(
        'import os, sys\nfrom pathlib import Path\n'
        f'Path({str(pid_file)!r}).write_text(str(os.getpid()))\n'
        'print("server booted", file=sys.stderr, flush=True)\n'
        'from mcp.server.fastmcp import FastMCP\n'
        'server = FastMCP("test")\n'
        + ('@server.tool()\ndef ping() -> str:\n    return "pong"\n' if with_tool else '')
        + 'server.run()\n'
    )
    return script, pid_file


def assert_exited(pid_file: Path) -> None:
    if sys.platform != 'win32':  # pragma: no branch
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text(encoding='utf-8')), 0)


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
    async with http_client(follow_redirects=True) as client:
        assert not client.follow_redirects, 'FastMCP asks for redirects; the endpoint stays fixed'


async def test_http_client_is_the_httpx_fastmcp_drives(monkeypatch: pytest.MonkeyPatch) -> None:
    """FastMCP 4's sign-in is an `httpx2.Auth`, which a legacy `httpx` client rejects as an invalid `auth`."""

    def fastmcp_4_client(
        headers: dict[str, str] | None = None, timeout: httpx2.Timeout | None = None, auth: httpx2.Auth | None = None
    ) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(headers=headers, timeout=timeout, auth=auth, follow_redirects=True)

    monkeypatch.setattr(mcp_settings, 'create_mcp_http_client', fastmcp_4_client)
    sign_in = httpx2.BasicAuth('user', 'password')
    async with http_client(auth=cast(httpx.Auth, sign_in), follow_redirects=True) as client:
        assert isinstance(client, httpx2.AsyncClient)
        assert client.auth is cast(httpx.Auth, sign_in)
        assert not client.follow_redirects


async def test_builtin_is_enabled_and_the_dashboard_is_the_front_door() -> None:
    declaration = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'mcp')
    assert declaration.enabled
    assert declaration.factory == 'pydantic_clai2.mcp'
    plugin = make_plugin({})
    assert len(plugin.capabilities) == 1
    dashboard = await plugin.commands.execute_async('/mcp')
    assert 'No MCP servers yet' in dashboard
    assert '/mcp install' in dashboard
    assert '/plugins' not in dashboard
    await plugin.dispatch(SessionEnd(reason='exit'))


def test_store_round_trip_is_private_and_fails_loudly(tmp_path: Path) -> None:
    store = MCPStore(tmp_path / 'config')
    assert store.load().servers == {}
    assert not store.delete('ghost')
    store.put('local', StdioServer(type='stdio', command='python'))
    assert list(MCPStore(tmp_path / 'config').load().servers) == ['local']
    if sys.platform != 'win32':  # pragma: no branch
        assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert '"enabled"' not in store.path.read_text(), 'defaults are not written'
    assert store.delete('local')
    store.path.write_text('{"servers": {"bad_name": {"type": "stdio", "command": "x"}}}')
    with pytest.raises(ValueError, match=r'mcp.json'):
        store.load()


def test_default_store_uses_the_clai_config_folder(tmp_path: Path) -> None:
    assert MCPStore().path == tmp_path / 'config' / 'pydantic-clai2' / 'mcp.json'


@pytest.mark.parametrize(
    'server',
    [
        {'type': 'stdio', 'command': ''},
        {'type': 'stdio', 'command': 'python', 'typo': True},
        {'type': 'http', 'url': 'file:///tmp/server'},
        {'type': 'http'},
        {'type': 'http', 'url': 'https://example.com', 'auth': 'basic'},
        {'type': 'http', 'url': 'http://example.com/mcp', 'auth': 'oauth'},
        {'type': 'sse', 'url': 'https://example.com', 'auth': 'oauth', 'headers': {'authorization': 'x'}},
        {'type': 'http', 'url': 'https://u:p@example.com', 'auth': 'oauth'},
        {'type': 'stdio', 'command': 'x', 'timeout': 0},
        {'type': 'websocket', 'url': 'https://example.com'},
    ],
)
def test_invalid_configuration(server: JsonValue) -> None:
    with pytest.raises(ValidationError):
        make_plugin({'servers': {'test': server}})


@pytest.mark.parametrize('name', ['bad_name', 'a b', '0server', 'x' * 65])
def test_invalid_name(name: str) -> None:
    with pytest.raises(ValidationError):
        make_plugin({'servers': {name: {'type': 'stdio', 'command': 'python'}}})


async def test_plugin_settings_servers_still_load_read_only(tmp_path: Path) -> None:
    plugin = make_plugin(
        {
            'servers': {
                'remote': {'transport': 'http', 'url': 'https://user:pw@example.com:8443/mcp?secret=value'},
                'off': {'transport': 'stdio', 'command': 'not-a-real-program', 'enabled': False},
            }
        },
        MCPStore(tmp_path / 'config', workspace=tmp_path),
    )
    dashboard = await plugin.commands.execute_async('/mcp list')
    assert 'remote' in dashboard and 'plugin' in dashboard
    assert 'secret' not in dashboard
    status = await plugin.commands.execute_async('/mcp status remote')
    assert 'https://example.com:8443/mcp' in status
    assert 'pw' not in status and 'secret' not in status
    with pytest.raises(ValueError, match='/plugins'):
        await plugin.commands.execute_async('/mcp remove remote')
    assert await plugin.commands.execute_async('/mcp logs remote') == 'No log entries for remote yet.'
    assert 'Stopped off' in await plugin.commands.execute_async('/mcp stop off')
    assert '- off' in await plugin.commands.execute_async('/mcp')


async def test_loader_persistence_disable_and_project_plugin_trust(tmp_path: Path) -> None:
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
        assert 'local' in await commands.execute_async('/mcp')
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


async def test_start_stop_restart_logs_and_agent_use(tmp_path: Path) -> None:
    script, pid_file = write_server(tmp_path)
    store = MCPStore(tmp_path / 'config', workspace=tmp_path)
    plugin = make_plugin({}, store)
    run = plugin.commands.execute_async
    store.put('local', StdioServer(type='stdio', command=sys.executable, args=[str(script)]))
    assert 'o local' in await run('/mcp'), 'installed servers are ready without a start'

    result = await Agent(TestModel(), deps_type=type(None), capabilities=plugin.capabilities).run('Use tools.')
    assert 'pong' in result.output
    assert '+ local' in await run('/mcp'), 'the first prompt connects a ready server and keeps it'
    held = pid_file.read_text()
    assert 'already running' in await run('/mcp start local')
    dashboard = await run('/mcp')
    assert '+ local' in dashboard and '1 tools' in dashboard and '1/1 running' in dashboard
    assert 'local_ping' in await run('/mcp status local')
    result = await Agent(TestModel(), deps_type=type(None), capabilities=plugin.capabilities).run('Use tools.')
    assert 'pong' in result.output
    assert pid_file.read_text() == held, 'a connected server keeps one process across runs'

    assert 'Started local' in await run('/mcp restart local')
    assert pid_file.read_text() != held
    assert 'Stopped local' in await run('/mcp stop local')
    assert_exited(pid_file)
    assert store.load().servers['local'].enabled is False, 'stop persists for user servers'
    result = await Agent(TestModel(), deps_type=type(None), capabilities=plugin.capabilities).run('Use tools.')
    assert result.output == 'success (no tool calls)'

    logs = await run('/mcp logs local 50')
    assert 'server booted' in logs and '[clai] started with 1 tools' in logs and '[clai] stopped' in logs
    assert logs.count('\n') <= 51
    assert len((await run('/mcp logs local 1')).splitlines()) == 2
    assert (
        (await run('/mcp logs local 1'))
        .splitlines()[0]
        .endswith(f'(last 1 of {logs.splitlines()[0].split()[-2]} lines)')
    )
    with pytest.raises(ValueError, match='Usage'):
        await run('/mcp logs local lots')

    assert 'Started local' in await run('/mcp start-all')
    assert 'local_ping' in await run('/mcp tools local')
    await plugin.dispatch(SessionEnd(reason='exit'))
    assert_exited(pid_file)
    assert 'Stopped local' in await run('/mcp stop-all')


async def test_tools_without_start_and_empty_server(tmp_path: Path) -> None:
    script, pid_file = write_server(tmp_path, with_tool=False)
    store = MCPStore(tmp_path / 'config', workspace=tmp_path)
    plugin = make_plugin({}, store)
    store.put('local', StdioServer(type='stdio', command=sys.executable, args=[str(script)]))
    assert await plugin.commands.execute_async('/mcp tools local') == 'No tools provided by local.'
    assert_exited(pid_file)


async def test_failed_start_is_reported_and_logged(tmp_path: Path) -> None:
    store = MCPStore(tmp_path / 'config', workspace=tmp_path)
    plugin = make_plugin({}, store)
    run = plugin.commands.execute_async
    store.put('broken', StdioServer(type='stdio', command=sys.executable, args=['-c', 'raise SystemExit(3)']))
    message = await run('/mcp start broken')
    assert message.startswith('Could not start broken') and '/mcp logs broken' in message
    assert '! broken' in await run('/mcp')
    assert 'start failed' in await run('/mcp logs broken')
    result = await Agent(TestModel(), deps_type=type(None), capabilities=plugin.capabilities).run('Use tools.')
    assert result.output == 'success (no tool calls)', 'a failed server is left out of runs'


async def test_env_references_resolve_at_connect_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('GITHUB_TOKEN', raising=False)
    store = MCPStore(tmp_path / 'config', workspace=tmp_path)
    plugin = make_plugin({}, store)
    store.put(
        'github',
        HTTPServer(
            type='http', url=HttpUrl('https://api.example.com/mcp'), headers={'Authorization': 'Bearer $GITHUB_TOKEN'}
        ),
    )
    assert '$GITHUB_TOKEN' in store.path.read_text(), 'the saved file holds the reference, not a value'
    assert 'set GITHUB_TOKEN in your environment' in await plugin.commands.execute_async('/mcp')
    assert 'cannot connect' in await plugin.commands.execute_async('/mcp start github')
    monkeypatch.setenv('GITHUB_TOKEN', 'token-value')
    assert 'o github' in await plugin.commands.execute_async('/mcp')
    servers = MCPServers(store)
    assert servers.state(servers.get('github')) == 'ready'


@pytest.mark.parametrize('kind', ['stdio', 'http'])
def test_env_reference_order_and_template_escapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: Literal['stdio', 'http']
) -> None:
    monkeypatch.delenv('FIRST_TOKEN', raising=False)
    monkeypatch.delenv('SECOND_TOKEN', raising=False)
    values = {'first': '${FIRST_TOKEN}:$SECOND_TOKEN:$FIRST_TOKEN', 'second': '$$LITERAL:$9:${'}
    server = (
        StdioServer(type='stdio', command='python', env=values)
        if kind == 'stdio'
        else HTTPServer(type='http', url=HttpUrl('https://api.example.com/mcp'), headers=values)
    )
    store = MCPStore(tmp_path / 'config', workspace=tmp_path)
    store.put('configured', server)
    servers = MCPServers(store)
    entry = servers.get('configured')
    assert servers.state(entry) == 'error'
    assert servers.problem(entry) == 'set FIRST_TOKEN, SECOND_TOKEN in your environment'
    monkeypatch.setenv('FIRST_TOKEN', 'first-value')
    assert servers.problem(entry) == 'set SECOND_TOKEN in your environment'
    monkeypatch.setenv('SECOND_TOKEN', 'second-value')
    assert servers.state(entry) == 'ready'


async def test_reconfiguring_or_removing_releases_the_connection(tmp_path: Path) -> None:
    script, pid_file = write_server(tmp_path)
    store = MCPStore(tmp_path / 'config', workspace=tmp_path)
    servers = MCPServers(store)
    store.put('local', StdioServer(type='stdio', command=sys.executable, args=[str(script)]))
    assert 'Started' in await servers.start('local')
    first = pid_file.read_text()
    store.put('local', StdioServer(type='stdio', command=sys.executable, args=[str(script), '--x']))
    assert 'Started' in await servers.start('local'), 'a changed server reconnects instead of reusing'
    assert pid_file.read_text() != first
    store.put('local', StdioServer(type='stdio', command=sys.executable, args=[str(script), '--y']))
    await servers.sync()
    assert_exited(pid_file)
    assert servers.state(servers.get('local')) == 'ready'
    assert 'Started' in await servers.start('local')
    await servers.remove('local')
    assert_exited(pid_file)
    with pytest.raises(ValueError, match='Unknown MCP server: local'):
        servers.get('local')


@pytest.mark.parametrize(('kind', 'path'), [('http', 'mcp'), ('sse', 'sse')])
async def test_real_remote_server(tmp_path: Path, kind: str, path: str) -> None:
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    script = tmp_path / 'remote_server.py'
    script.write_text(
        'from mcp.server.fastmcp import FastMCP\n'
        f'server = FastMCP("web", host="127.0.0.1", port={port}, log_level="WARNING")\n'
        '@server.tool()\ndef ping() -> str:\n    return "pong"\n'
        f'server.run(transport={"streamable-http" if kind == "http" else "sse"!r})\n'
    )
    process = subprocess.Popen([sys.executable, str(script)], stderr=subprocess.DEVNULL)
    try:
        store = MCPStore(tmp_path / 'config', workspace=tmp_path)
        plugin = make_plugin({}, store)
        run = plugin.commands.execute_async
        url = HttpUrl(f'http://127.0.0.1:{port}/{path}')
        store.put(
            'web', HTTPServer(type='http', url=url) if kind == 'http' else SSEServer(type='sse', url=url, timeout=10)
        )
        # Wait for the port before the first MCP connection: an SSE client that fails to connect
        # leaves its memory streams unclosed, which `filterwarnings = error` reports.
        with anyio.fail_after(10):
            while True:
                try:
                    await (await anyio.connect_tcp('127.0.0.1', port)).aclose()
                    break
                except OSError:
                    await anyio.sleep(0.05)
        message = ''
        for _ in range(100):  # pragma: no branch
            message = await run('/mcp restart web')
            if message.startswith('Started'):
                break
            await anyio.sleep(0.1)  # pragma: lax no cover
        assert message.startswith('Started web with 1 tools'), message
        result = await Agent(TestModel(), deps_type=type(None), capabilities=plugin.capabilities).run('Use tools.')
        assert 'pong' in result.output
        await plugin.dispatch(SessionEnd(reason='exit'))
    finally:
        process.terminate()
        process.wait()


def test_remote_options_build_the_right_transport(tmp_path: Path) -> None:
    store = MCPStore(tmp_path / 'config', workspace=tmp_path)
    store.put('old', SSEServer(type='sse', url=HttpUrl('https://example.com/sse'), timeout=12))
    store.put('signin', HTTPServer(type='http', url=HttpUrl('https://example.com/mcp'), auth='oauth'))
    store.put('plain', HTTPServer(type='http', url=HttpUrl('http://localhost:9/mcp'), auth='oauth', timeout=5))
    servers = MCPServers(store, {'legacy': StdioServer(type='stdio', command='x', timeout=7)})
    signin = servers.get('signin').server
    assert isinstance(signin, HTTPServer) and signin.init_timeout() == 330
    assert servers.get('plain').server.model_dump()['timeout'] == 5
    assert [entry.server.type for entry in servers.entries()] == ['sse', 'http', 'http', 'stdio']


def test_plugin_settings_accept_the_old_transport_key() -> None:
    assert MCPSettings(servers={'x': StdioServer(type='stdio', command='x')}).servers['x'].type == 'stdio'
    plugin = make_plugin({'servers': {'old': {'transport': 'stdio', 'command': 'x'}}})
    assert len(plugin.capabilities) == 1


def test_fastmcp_info_logging_stays_off_the_prompt() -> None:
    assert logging.getLogger('fastmcp.client.auth.oauth').getEffectiveLevel() >= logging.WARNING


async def test_a_server_that_cannot_connect_does_not_fail_the_prompt(tmp_path: Path) -> None:
    store = MCPStore(tmp_path / 'config', workspace=tmp_path)
    plugin = make_plugin({}, store)
    store.put('broken', StdioServer(type='stdio', command=sys.executable, args=['-c', 'raise SystemExit(3)']))
    result = await Agent(TestModel(), deps_type=type(None), capabilities=plugin.capabilities).run('Use tools.')
    assert result.output == 'success (no tool calls)'
    assert '! broken' in await plugin.commands.execute_async('/mcp')
