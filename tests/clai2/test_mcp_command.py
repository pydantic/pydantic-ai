"""The `/mcp` command surface: the add/edit server form, trust, help, and completion."""

import io
import json
import sys
from pathlib import Path

import pytest
from menu_script import Script, pick, typed
from pydantic import HttpUrl
from rich.console import Console
from termflow.tui.menu import MenuResult  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.textinput import TextInputResult  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2.commands import Commands
from pydantic_clai2.mcp import (
    EXAMPLES,
    HELP,
    PROJECT_MCP_FILE,
    HTTPServer,
    MCPCommand,
    MCPServers,
    MCPStore,
    ServerForm,
    SSEServer,
    StdioServer,
    activate,
    edit_in_editor,
    run_form,
)
from pydantic_clai2.plugins import PluginHost

ESC = MenuResult(cancelled=True)


def make(
    tmp_path: Path, script: Script | None = None, edits: list[str | None] | None = None
) -> tuple[MCPCommand, MCPStore]:
    store = MCPStore(tmp_path / 'config', workspace=tmp_path / 'repo')
    (tmp_path / 'repo' / '.git').mkdir(parents=True, exist_ok=True)
    pending = iter(edits or [])
    command = MCPCommand(servers=MCPServers(store), editor=lambda _: next(pending))
    if script is not None:
        command.runners = script.runners
    return command, store


def stdio(command: str = 'python', *args: str) -> StdioServer:
    return StdioServer(type='stdio', command=command, args=list(args))


async def test_help_errors_and_usage(tmp_path: Path) -> None:
    command, _ = make(tmp_path)
    assert await command(['help']) == HELP
    for sub in ('install', 'start', 'stop', 'restart', 'status', 'logs', 'edit', 'remove', 'trust', 'start-all'):
        assert f'/mcp {sub}' in HELP
    assert 'search' not in HELP and 'catalog' not in HELP
    with pytest.raises(ValueError, match='Unknown MCP subcommand: nope'):
        await command(['nope'])
    with pytest.raises(ValueError, match='Usage: /mcp install'):
        await command(['install', 'github'])
    with pytest.raises(ValueError, match='Usage: /mcp start NAME'):
        await command(['start'])
    with pytest.raises(ValueError, match='Unknown MCP server: ghost. Known: none'):
        await command(['status', 'ghost'])
    assert await command(['status']) == await command([])
    assert await command(['start-all']) == 'No MCP servers to start.'
    assert await command(['stop-all']) == 'No MCP servers to stop.'


async def test_install_stdio_through_the_form(tmp_path: Path) -> None:
    config = json.dumps({'command': 'uvx', 'args': ['my-server'], 'env': {'TOKEN': '$MY_TOKEN'}})
    script = Script(lists=[pick('name'), pick('json'), pick('save')], choices=[], texts=[typed(' files ')])
    command, store = make(tmp_path, script, edits=[config])
    message = await command(['install'])
    assert message.startswith('Added files. The agent can use it on your next prompt')
    assert 'Set MY_TOKEN' in message
    assert store.load().servers['files'] == StdioServer(
        type='stdio', command='uvx', args=['my-server'], env={'TOKEN': '$MY_TOKEN'}
    )
    assert '! files' in await command([]), 'MY_TOKEN is not set'


async def test_install_remote_with_type_swap_and_oauth(tmp_path: Path) -> None:
    script = Script(
        lists=[pick('type'), pick('target'), pick('target'), pick('oauth'), pick('name'), pick('save')],
        choices=[pick('http')],
        texts=[typed('not a url'), typed('https://mcp.example.com/mcp'), typed('docs')],
    )
    command, store = make(tmp_path, script)
    assert (await command(['install'])).startswith('Added docs.')
    saved = store.load().servers['docs']
    assert isinstance(saved, HTTPServer)
    assert str(saved.url) == 'https://mcp.example.com/mcp', 'an invalid URL is ignored, the valid one saved'
    assert saved.auth == 'oauth' and saved.timeout == 330, 'OAuth allows time for the browser sign-in'
    assert saved.headers is None, 'switching on OAuth drops the Authorization header'


async def test_form_saves_only_valid_servers(tmp_path: Path) -> None:
    store = MCPStore(tmp_path / 'config')
    store.put('taken', stdio())
    form = ServerForm(store)
    assert not form.save() and form.status == 'Save failed: Server name is required'
    form.name = 'taken'
    assert not form.save() and 'taken already exists' in (form.status or '')
    form.name = 'bad_name'
    assert form.name_problem(form.name) is not None
    form.name = 'fresh-one'
    form.config = '{not json'
    assert not form.save() and 'Invalid JSON' in (form.status or '')
    assert 'JSON Configuration (INVALID)' in [item.label for item in form.items()]
    form.config = '{}'
    assert form.problem() == 'command: Field required'
    form.config = '{"command": "x", "typo": 1}'
    assert form.problem() == 'typo: Extra inputs are not permitted'
    form.select_type('http')
    form.config = '{"url": "https://example.com/mcp", "auth": "oauth", "headers": {"Authorization": "x"}}'
    assert 'Authorization' in (form.problem() or '')
    form.config = '{"type": "stdio", "url": "https://example.com/mcp"}'
    assert form.problem() is None, 'the Server Type row wins over a stale "type" key'
    assert form.save() and isinstance(store.load().servers['fresh-one'], HTTPServer)


def test_form_rows_preview_and_examples(tmp_path: Path) -> None:
    form = ServerForm(MCPStore(tmp_path / 'config'))
    labels = [item.label for item in form.items()]
    assert labels == [
        'Server Name: (not set)',
        'Server Type: stdio',
        'Command: npx -y @modelcontextprotocol/server-filesystem /path/to/dir',
        'JSON Configuration (valid)',
        'Load example for stdio',
        'Save & Install',
        'Cancel',
    ]
    assert 'Add Custom MCP Server' in form.preview() and 'Configuration is valid' in form.preview()
    assert 'OAuth' not in form.preview()
    form.select_type('sse')
    assert form.config == EXAMPLES['sse'], 'an untouched example follows the type'
    assert 'OAuth sign-in: off' in [item.label for item in form.items()]
    assert 'OAuth signs in' in form.preview()
    form.config = '{"url": "https://example.com/sse"}'
    form.select_type('http')
    assert form.config == '{"url": "https://example.com/sse"}', 'edited configuration is kept'
    form.toggle_oauth()
    assert form.oauth and 'OAuth sign-in: on' in [item.label for item in form.items()]
    form.toggle_oauth()
    assert not form.oauth and json.loads(form.config) == {'url': 'https://example.com/sse'}
    form.config = '{"url": "https://example.com/sse", "auth": "oauth", "timeout": 60}'
    form.toggle_oauth()
    assert json.loads(form.config) == {'url': 'https://example.com/sse', 'timeout': 60}, 'a chosen timeout stays'
    form.config = json.dumps({'url': 'https://example.com/mcp', 'headers': {'authorization': 'x', 'X-Team': 'a'}})
    form.toggle_oauth()
    assert json.loads(form.config)['headers'] == {'X-Team': 'a'} and 'Removed the Authorization' in (form.status or '')
    form.config = 'broken'
    form.toggle_oauth()
    assert form.status == 'Fix the JSON before switching OAuth' and not form.oauth
    assert 'Fix the JSON' in form.preview() and 'Invalid: Invalid JSON' in form.preview()
    form.load_example()
    assert form.config == EXAMPLES['http'] and form.status is None


def test_json_fallback_when_no_editor_runs(tmp_path: Path) -> None:
    store = MCPStore(tmp_path / 'config')
    script = Script(
        lists=[pick('json'), pick('json'), pick('json'), pick('cancel')],
        choices=[],
        texts=[typed('{"command": "uvx"}'), TextInputResult(cancelled=True), typed('{broken')],
    )
    form = ServerForm(store)
    form.config = 'not json yet'
    assert not run_form(form, script.runners, editor=lambda _: None)
    assert json.loads(form.config) == {'command': 'uvx'}
    assert script.opened == ['list', 'text', 'list', 'text', 'list', 'text', 'list']


def test_form_cancel_paths(tmp_path: Path) -> None:
    store = MCPStore(tmp_path / 'config')
    form = ServerForm(store)
    script = Script(
        lists=[pick('name'), pick('type'), pick('example'), pick('save'), pick('stray'), ESC],
        choices=[ESC],
        texts=[TextInputResult(cancelled=True)],
    )
    assert not run_form(form, script.runners, editor=lambda _: None)
    assert form.status == 'Save failed: Server name is required'
    assert form.type == 'stdio' and form.name == ''


async def test_install_cancelled(tmp_path: Path) -> None:
    command, store = make(tmp_path, Script(lists=[pick('cancel')], choices=[], texts=[]))
    assert await command(['install']) == 'Exited custom server form.'
    assert store.load().servers == {}


async def test_missing_program_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PATH', str(tmp_path / 'empty'))
    script = Script(lists=[pick('name'), pick('save')], choices=[], texts=[typed('fs')])
    command, _ = make(tmp_path, script)
    assert 'npx is not on PATH' in await command(['install'])


async def test_edit_prefills_renames_and_cancels(tmp_path: Path) -> None:
    script = Script(
        lists=[pick('name'), pick('json'), pick('save'), ESC],
        choices=[],
        texts=[typed('renamed')],
    )
    command, store = make(tmp_path, script, edits=[json.dumps({'command': 'uvx', 'args': ['new'], 'timeout': 5})])
    store.put('local', StdioServer(type='stdio', command='python', env={'A': 'b'}))
    form = ServerForm(store, name='local', server=store.load().servers['local'])
    assert json.loads(form.config) == {'type': 'stdio', 'command': 'python', 'env': {'A': 'b'}}
    assert 'Edit MCP Server' in form.preview() and 'Save changes' in [item.label for item in form.items()]
    assert form.name_problem('local') is None, 'keeping the name is not a clash'
    assert (await command(['edit', 'local'])).startswith('Updated renamed.')
    assert list(store.load().servers) == ['renamed']
    assert store.load().servers['renamed'] == StdioServer(type='stdio', command='uvx', args=['new'], timeout=5)
    assert await command(['edit', 'renamed']) == 'No changes.'


async def test_edit_restarts_a_running_server(tmp_path: Path) -> None:
    server = tmp_path / 'server.py'
    server.write_text(
        'from mcp.server.fastmcp import FastMCP\nserver = FastMCP("t")\n'
        '@server.tool()\ndef ping() -> str:\n    return "pong"\nserver.run()\n'
    )
    config = json.dumps({'command': sys.executable, 'args': [str(server)], 'cwd': str(tmp_path)})
    command, store = make(tmp_path, Script(lists=[pick('json'), pick('save')], choices=[], texts=[]), edits=[config])
    store.put('local', stdio(sys.executable, str(server)))
    await command(['start', 'local'])
    lines = (await command(['edit', 'local'])).splitlines()
    assert lines[0].startswith('Updated local.')
    assert lines[-1] == 'Started local with 1 tools. The agent can use them on your next prompt.'
    await command.servers.close()


async def test_edit_refuses_servers_mcp_does_not_own(tmp_path: Path) -> None:
    store = MCPStore(tmp_path / 'config', workspace=tmp_path)
    command = MCPCommand(servers=MCPServers(store, {'legacy': stdio()}))
    with pytest.raises(ValueError, match='/plugins'):
        await command(['edit', 'legacy'])


def test_editor_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    edit = tmp_path / 'edit.py'
    edit.write_text('import sys, pathlib\npathlib.Path(sys.argv[1]).write_text("{\\"command\\": \\"edited\\"}")\n')
    monkeypatch.setenv('VISUAL', f'{sys.executable} {edit}')
    assert edit_in_editor('{}') == '{"command": "edited"}'
    monkeypatch.setenv('VISUAL', f'{sys.executable} -c "raise SystemExit(1)"')
    assert edit_in_editor('{}') is None
    monkeypatch.setenv('VISUAL', str(tmp_path / 'no-such-editor'))
    assert edit_in_editor('{}') is None
    monkeypatch.setenv('VISUAL', 'vim "unclosed')
    assert edit_in_editor('{}') is None, 'an unparsable editor falls back to the one-line input'


async def test_project_file_trust(tmp_path: Path) -> None:
    command, store = make(tmp_path)
    assert 'No .clai/mcp_servers.json' in await command(['trust'])
    project = tmp_path / 'repo' / PROJECT_MCP_FILE
    project.parent.mkdir()
    project.write_text('{"servers": {"team": {"type": "stdio", "command": "team-server"}}}')

    dashboard = await command([])
    assert 'not trusted, so its servers are not loaded' in dashboard and 'team' not in dashboard.split('\n\n')[0]
    assert (await command(['trust'])).endswith('untrusted')
    assert 'Servers loaded: team' in await command(['trust', 'accept'])
    assert 'project' in await command([])
    assert 'team' in await command(['status', 'team'])
    with pytest.raises(ValueError, match='change that file instead'):
        await command(['edit', 'team'])
    with pytest.raises(ValueError, match='change that file instead'):
        await command(['remove', 'team'])
    assert 'Stopped team' in await command(['stop', 'team'])
    assert '- team' in await command([]), 'project servers stop for the session only'
    assert 'team-server' in project.read_text()

    project.write_text('{"servers": {"team": {"type": "stdio", "command": "other"}}}')
    assert 'changed since you trusted it' in await command([])
    assert (await command(['trust', 'status'])).endswith('changed')
    assert 'Revoked trust' in await command(['trust', 'revoke'])
    assert 'was not trusted' in await command(['trust', 'revoke'])
    with pytest.raises(ValueError, match='Usage: /mcp trust'):
        await command(['trust', 'maybe'])

    project.write_text('{"servers": {"bad_name": {}}}')
    store.trust(project)
    with pytest.raises(ValueError, match='mcp_servers.json'):
        await command([])
    project.unlink()
    project.mkdir()
    assert store.trust_state(project) == 'changed', 'an unreadable file fails closed'
    assert store.project_servers() == {}


def test_user_servers_shadow_project_servers(tmp_path: Path) -> None:
    command, store = make(tmp_path)
    project = tmp_path / 'repo' / PROJECT_MCP_FILE
    project.parent.mkdir()
    project.write_text('{"servers": {"shared": {"type": "stdio", "command": "project"}}}')
    store.trust(project)
    store.put('shared', stdio('mine'))
    [only] = command.servers.entries()
    assert only.source == 'user' and isinstance(only.server, StdioServer) and only.server.command == 'mine'


async def test_dashboard_lists_every_type(tmp_path: Path) -> None:
    command, store = make(tmp_path)
    store.put('local', stdio())
    store.put('docs', HTTPServer(type='http', url=HttpUrl('https://example.com/mcp')))
    store.put('old', SSEServer(type='sse', url=HttpUrl('https://example.com/sse')))
    dashboard = await command([])
    assert 'stdio' in dashboard and 'http' in dashboard and 'sse' in dashboard
    assert '0/3 running, 3 available to the agent' in dashboard
    assert await command(['remove', 'local']) == 'Removed local.'


async def test_completion(tmp_path: Path) -> None:
    command, store = make(tmp_path)
    store.put('local', stdio())
    assert 'install' in command.complete([]) and 'start-all' in command.complete([''])
    assert 'search' not in command.complete([])
    assert list(command.complete(['start', ''])) == ['local']
    assert tuple(command.complete(['install', ''])) == ()
    assert tuple(command.complete(['trust', ''])) == ('status', 'accept', 'revoke')
    store.path.write_text('not json')
    assert tuple(command.complete(['start', ''])) == ()


async def test_registered_completion_through_the_command_registry(tmp_path: Path) -> None:
    host: PluginHost[None] = PluginHost(name='mcp', console=Console(file=io.StringIO()), settings={})
    store = MCPStore(tmp_path / 'config', workspace=tmp_path)
    activate(host, store=store)
    registry = Commands()
    registry.register_many(host.commands)
    store.put('local', stdio())
    [mcp] = list(registry)
    assert 'local' in mcp.complete(['logs', ''])


def test_url_and_command_rows(tmp_path: Path) -> None:
    form = ServerForm(MCPStore(tmp_path / 'config'))
    assert form.target_label == 'Command'
    assert form.target_problem('uvx "unclosed') == 'No closing quotation'
    assert form.target_problem('  ') == 'Enter the program to run, then its arguments'
    assert form.target_problem('uvx my-server') is None
    form.set_target('uvx my-server --flag "a b"')
    assert json.loads(form.config)['args'] == ['my-server', '--flag', 'a b']
    assert form.target() == "uvx my-server --flag 'a b'"
    form.set_target('my-server')
    assert 'args' not in json.loads(form.config)
    form.config = '{"command": 3, "args": "x"}'
    assert form.target() == ''
    form.config = '{"command": "x", "args": "x"}'
    assert form.target() == 'x'
    form.select_type('sse')
    assert form.target_label == 'URL' and form.target() == ''
    assert form.target_problem('ftp://example.com') == 'Enter an http:// or https:// URL'
    form.set_target(' https://example.com/sse ')
    assert form.target() == 'https://example.com/sse'
    assert 'URL: https://example.com/sse' in [item.label for item in form.items()]
    form.config = 'broken'
    assert form.target() == ''
    form.set_target('https://example.com/sse')
    assert form.status == 'Fix the JSON before editing the URL'


async def test_surplus_arguments_are_rejected(tmp_path: Path) -> None:
    command, store = make(tmp_path)
    store.put('local', stdio())
    for args in (['remove', 'local', 'oops'], ['start', 'local', 'x'], ['start-all', 'typo'], ['stop-all', 'x']):
        with pytest.raises(ValueError, match='unexpected'):
            await command(args)
    with pytest.raises(ValueError, match='Usage: /mcp trust'):
        await command(['trust', 'accept', 'junk'])
    assert 'local' in store.load().servers


async def test_symlinked_project_file_is_never_trusted(tmp_path: Path) -> None:
    command, store = make(tmp_path)
    elsewhere = tmp_path / 'trusted.json'
    elsewhere.write_text('{"servers": {"team": {"type": "stdio", "command": "team-server"}}}')
    project = tmp_path / 'repo' / PROJECT_MCP_FILE
    project.parent.mkdir()
    project.symlink_to(elsewhere)
    with pytest.raises(ValueError, match='symlink'):
        await command(['trust', 'accept'])
    data = store.load()
    store.save(data.model_copy(update={'trusted_projects': {str(project.absolute()): 'x'}}))
    assert store.trust_state(project) == 'untrusted' and store.project_servers() == {}


async def test_revoke_disconnects_project_servers(tmp_path: Path) -> None:
    server = tmp_path / 'server.py'
    server.write_text('from mcp.server.fastmcp import FastMCP\nFastMCP("t").run()\n')
    command, _ = make(tmp_path)
    project = tmp_path / 'repo' / PROJECT_MCP_FILE
    project.parent.mkdir()
    project.write_text(
        json.dumps({'servers': {'team': {'type': 'stdio', 'command': sys.executable, 'args': [str(server)]}}})
    )
    await command(['trust', 'accept'])
    assert 'Started team' in await command(['start', 'team'])
    assert 'Revoked' in await command(['trust', 'revoke'])
    assert command.servers._connections == {}  # pyright: ignore[reportPrivateUsage]


def test_oauth_toggle_restores_a_short_timeout(tmp_path: Path) -> None:
    form = ServerForm(MCPStore(tmp_path / 'config'))
    form.select_type('http')
    form.config = '{"url": "https://example.com/mcp", "timeout": 30}'
    form.toggle_oauth()
    assert json.loads(form.config)['timeout'] == 330
    form.toggle_oauth()
    assert json.loads(form.config) == {'url': 'https://example.com/mcp', 'timeout': 30}
    form.config = '{"url": "https://example.com/mcp", "timeout": 330}'
    form.toggle_oauth()
    form.toggle_oauth()
    assert json.loads(form.config)['timeout'] == 330, 'a timeout the user chose is kept'
