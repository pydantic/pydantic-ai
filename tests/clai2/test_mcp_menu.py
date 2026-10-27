"""`/mcp`'s declared menu: each server with its state, managed through `Dialogs`, as the web UI shows it."""

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from fastmcp.client.auth.oauth import TokenStorageAdapter
from mcp.shared.auth import OAuthToken
from pydantic import HttpUrl

from pydantic_clai2.mcp import (
    PROJECT_MCP_FILE,
    HTTPServer,
    MCPCommand,
    MCPServers,
    MCPStore,
    OAuthSignIn,
    ServersSource,
    StdioServer,
    TokenStore,
    servers_menu,
)
from pydantic_clai2.mcp.runtime import ServerEntry, State
from pydantic_clai2.mcp.settings import Servers
from pydantic_clai2.plugins.menus import FieldRow, Option, SettingsMenu
from tests.clai2.conftest import stored_accounts
from tests.clai2.dialog_script import Answer, Back, Confirm, Finish, Pick, ScriptedDialogs, Type

DEAD = 'http://127.0.0.1:9/mcp'
"""A loopback port nothing listens on: connecting fails at once, without the network."""
Vault = dict[tuple[str, str], str]


def make(tmp_path: Path, plugin_servers: Servers | None = None) -> tuple[MCPCommand, MCPStore, SettingsMenu]:
    store = MCPStore(tmp_path / 'config', workspace=tmp_path / 'repo')
    (tmp_path / 'repo' / '.git').mkdir(parents=True, exist_ok=True)
    command = MCPCommand(servers=MCPServers(store, plugin_servers))
    return command, store, servers_menu(command)


def rows(menu: SettingsMenu) -> dict[str, tuple[FieldRow, str]]:
    return {row.key: (row, menu.source.current(row)) for row in menu.source.rows()}


def stdio(command: str = 'python', **env: str) -> StdioServer:
    return StdioServer(type='stdio', command=command, args=['server.py'], env=env or None)


async def sign_in_to(name: str) -> None:
    token = OAuthToken(access_token='access', token_type='Bearer', refresh_token='refresh', expires_in=3600)
    await TokenStorageAdapter(TokenStore(name), server_url=DEAD).set_tokens(token)


async def test_rows_show_each_server_with_its_state(tmp_path: Path, vault: Vault) -> None:
    _, store, menu = make(tmp_path, {'legacy': stdio()})
    store.put('local', stdio())
    store.put('off', stdio().model_copy(update={'enabled': False}))
    store.put('secret', stdio(TOKEN='$CLAI_TEST_UNSET_TOKEN'))
    store.put('docs', HTTPServer(type='http', url=HttpUrl(DEAD), auth='oauth'))
    store.put('plain', HTTPServer(type='http', url=HttpUrl('https://example.com/mcp')))
    shown = rows(menu)
    assert list(shown) == [
        'server.local',
        'server.off',
        'server.secret',
        'server.docs',
        'server.plain',
        'server.legacy',
        'add',
    ], 'no trust row without project files'
    local, state = shown['server.local']
    assert (local.label, local.description, state) == ('local', 'stdio · python server.py', 'ready')
    assert local.note == 'Connects on your next prompt'
    assert shown['server.off'][1] == 'stopped' and shown['server.off'][0].note == 'Off until you start it'
    assert shown['server.secret'][1] == 'error'
    assert shown['server.secret'][0].note == 'set CLAI_TEST_UNSET_TOKEN in your environment'
    assert shown['server.docs'][1] == 'sign-in' and shown['server.docs'][0].note == 'Sign in to use it'
    assert shown['server.plain'][1] == 'ready', 'no OAuth, nothing to sign in to'
    assert shown['server.legacy'][0].note == 'Connects on your next prompt · From /plugins settings for mcp'
    assert shown['add'][1] == ''
    await sign_in_to('docs')
    assert rows(menu)['server.docs'][1] == 'ready'

    # Every row is an action, found as the rows are now: none has a value to type.
    assert list(menu.actions) == list(shown) and len(menu.actions) == len(shown)
    assert 'server.added-later' in menu.actions and 'nope' not in menu.actions
    assert menu.source.problem(local, 'x') == 'local has no value to set; choose it instead.'
    with pytest.raises(ValueError, match='has no value to set'):
        menu.source.apply(local, 'x')
    with pytest.raises(ValueError, match='has no value to set'):
        menu.source.reset(local)


async def test_running_servers_report_their_tools_and_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A held connection is `running`, which only a real server reaches, so its state is stood in for here."""
    command, store, menu = make(tmp_path)
    store.put('local', stdio())
    states: dict[str, State] = {'local': 'running'}
    servers = command.servers

    def state(entry: ServerEntry) -> State:
        return states[entry.name]

    def tools(entry: ServerEntry) -> tuple[str, ...]:
        return ('local_ping', 'local_pong')

    monkeypatch.setattr(servers, 'state', state)
    monkeypatch.setattr(servers, 'tools', tools)
    assert rows(menu)['server.local'][0].note == '2 tools'
    dialogs = ScriptedDialogs([Back()])
    assert await menu.actions['server.local'](dialogs) == []
    assert [option.value for option in dialogs.asked[0].options] == [
        'restart',
        'stop',
        'tools',
        'logs',
        'edit',
        'remove',
    ]
    states['local'] = 'error'
    assert rows(menu)['server.local'][0].note == 'Could not connect', 'an error without a recorded problem'


async def test_manage_starts_stops_and_shows_a_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    command, store, menu = make(tmp_path)
    store.put('local', stdio(TOKEN='$CLAI_TEST_UNSET_TOKEN'))
    manage = menu.actions['server.local']

    dialogs = ScriptedDialogs([Pick(value='stop')])
    assert await manage(dialogs) == ['Stopped local. Its tools are unavailable until /mcp start local.']
    choose = dialogs.asked[0]
    assert choose.title == 'local' and 'state    error' in choose.description
    assert [option.value for option in choose.options] == ['start', 'stop', 'tools', 'logs', 'edit', 'remove']
    assert store.load().servers['local'].enabled is False, 'saved, as /mcp stop saves it'

    dialogs = ScriptedDialogs([Pick(value='start')])
    assert await manage(dialogs) == [
        'local is enabled but cannot connect: set CLAI_TEST_UNSET_TOKEN in your environment.'
    ]
    assert [option.label for option in dialogs.asked[0].options][:1] == ['Start'], 'a stopped server only starts'

    async def listed(name: str) -> list[str]:
        return [f'{name}_ping']

    command.servers.log('local', 'started with 1 tools')
    monkeypatch.setattr(command.servers, 'list_tools', listed)
    dialogs = ScriptedDialogs([Pick(value='tools'), Pick(value='back'), Pick(value='logs'), Back(), Back()])
    assert await manage(dialogs) == []
    _, tools, _, logs, _ = dialogs.asked
    assert (tools.title, tools.description) == ('local tools', 'local_ping')
    assert logs.title == 'local logs' and logs.description.endswith('[clai] started with 1 tools')

    async def refused(name: str) -> list[str]:
        raise OSError()

    monkeypatch.setattr(command.servers, 'list_tools', refused)
    dialogs = ScriptedDialogs([Pick(value='tools'), Back(), Back()])
    await manage(dialogs)
    assert dialogs.asked[1].description == 'Could not list its tools: OSError'


async def test_manage_edits_and_removes_a_server(tmp_path: Path) -> None:
    _, store, menu = make(tmp_path)
    store.put('local', stdio())
    dialogs = ScriptedDialogs([Pick(value='edit'), Pick(value='name'), Type(text='renamed'), Pick(value='save')])
    [message] = await menu.actions['server.local'](dialogs)
    assert message.startswith('Updated renamed.') and list(store.load().servers) == ['renamed']
    assert dialogs.asked[1].title == 'Edit MCP Server'

    remove = menu.actions['server.renamed']
    dialogs = ScriptedDialogs([Pick(value='remove'), Confirm(yes=False), Back()])
    assert await remove(dialogs) == [] and 'renamed' in store.load().servers, 'keeping it changes nothing'
    assert dialogs.asked[1].title == 'Remove renamed?'
    assert await remove(ScriptedDialogs([Pick(value='remove'), Confirm(yes=True)])) == ['Removed renamed.']
    assert store.load().servers == {}
    with pytest.raises(ValueError, match='Unknown MCP server: renamed'):
        await remove(ScriptedDialogs([]))


async def test_manage_signs_in_and_out(tmp_path: Path, vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    _, store, menu = make(tmp_path)
    store.put('docs', HTTPServer(type='http', url=HttpUrl(DEAD), auth='oauth', timeout=5))

    async def sign_in(method: OAuthSignIn, *, show: Callable[[str], object]) -> None:
        show('Open https://auth.example.com/authorize')
        await sign_in_to('docs')

    monkeypatch.setattr(OAuthSignIn, 'sign_in', sign_in)
    manage = menu.actions['server.docs']
    dialogs = ScriptedDialogs([Pick(value='sign-in'), Finish()])
    [message] = await manage(dialogs)
    signed_in, started = message.splitlines()
    assert signed_in == 'Signed in to docs.' and started.startswith('Could not start docs:')
    assert 'sign-out' not in [option.value for option in dialogs.asked[0].options]
    assert dialogs.asked[1].kind == 'wait' and dialogs.asked[1].shown == ['Open https://auth.example.com/authorize']

    dialogs = ScriptedDialogs([Pick(value='sign-out')])
    assert (await manage(dialogs))[0].startswith('Signed out of docs.')
    assert 'sign-out' in [option.value for option in dialogs.asked[0].options]
    assert stored_accounts() == set()


async def test_project_servers_are_trusted_from_the_menu(tmp_path: Path) -> None:
    _, _, menu = make(tmp_path)
    project = tmp_path / 'repo' / PROJECT_MCP_FILE
    project.parent.mkdir()
    project.write_text(json.dumps({'servers': {'team': {'type': 'stdio', 'command': 'team-server'}}}))
    trust_row, state = rows(menu)['trust']
    assert state == 'untrusted' and trust_row.note == str(project.resolve())
    assert list(rows(menu)) == ['trust', 'add'], 'an untrusted file loads no servers'

    assert await menu.actions['trust'](ScriptedDialogs([Back()])) == []
    dialogs = ScriptedDialogs([Pick(value='accept')])
    [message] = await menu.actions['trust'](dialogs)
    assert message.startswith('Trusted') and 'Servers loaded: team' in message
    description = dialogs.asked[0].description
    assert f'{project.resolve()} (untrusted)\n' in description and '"command": "team-server"' in description
    shown = rows(menu)
    assert shown['trust'][1] == 'trusted' and list(shown) == ['server.team', 'trust', 'add']
    assert shown['server.team'][0].note == f'Connects on your next prompt · From {project.resolve()}'

    dialogs = ScriptedDialogs([Back()])
    await menu.actions['server.team'](dialogs)
    assert 'edit' not in [option.value for option in dialogs.asked[0].options]
    assert dialogs.asked[0].description.endswith('change that file instead.')

    project.write_text(json.dumps({'servers': {}}))
    assert rows(menu)['trust'][1] == 'changed'
    assert await menu.actions['trust'](ScriptedDialogs([Pick(value='revoke')])) == [
        f'Revoked trust in {project.resolve()}.'
    ]


class EditedWhileOpen(ScriptedDialogs):
    """Rewrites `path` while the first list is open, as an agent's turn or a `git pull` could."""

    def __init__(self, answers: Sequence[Answer], *, path: Path) -> None:
        super().__init__(answers)
        self._path = path

    async def choose(
        self, *, title: str, options: Sequence[Option], current: str | None = None, description: str = ''
    ) -> str | None:
        self._path.write_text(json.dumps({'servers': {'team': {'type': 'stdio', 'command': 'other'}}}))
        return await super().choose(title=title, options=options, current=current, description=description)


async def test_trust_covers_only_what_the_user_reviewed(tmp_path: Path) -> None:
    _, store, menu = make(tmp_path)
    project = tmp_path / 'repo' / PROJECT_MCP_FILE
    project.parent.mkdir()
    project.write_text(json.dumps({'servers': {'team': {'type': 'stdio', 'command': 'team-server'}}}))
    dialogs = EditedWhileOpen([Pick(value='accept')], path=project)
    assert await menu.actions['trust'](dialogs) == [
        'A project MCP file changed while you reviewed it; review it again.'
    ]
    assert store.trust_state(project.resolve()) == 'untrusted'


def test_review_reads_only_regular_files(tmp_path: Path) -> None:
    store = MCPStore(tmp_path / 'config', workspace=tmp_path)
    elsewhere = tmp_path / 'elsewhere.json'
    elsewhere.write_text('{}')
    link = tmp_path / 'link.json'
    link.symlink_to(elsewhere)
    assert store.review(link) == 'A symlink, which is never trusted.'
    assert store.review(tmp_path).startswith('Cannot read it:'), 'a folder is not a file to read'


async def test_add_opens_the_install_form(tmp_path: Path) -> None:
    _, store, menu = make(tmp_path)
    assert await menu.actions['add'](ScriptedDialogs([Pick(value='cancel')])) == ['Exited custom server form.']
    dialogs = ScriptedDialogs(
        [
            Pick(value='name'),
            Type(text='files'),
            Pick(value='json'),
            Type(text='{"command": "uvx"}'),
            Pick(value='save'),
        ]
    )
    [message] = await menu.actions['add'](dialogs)
    assert message.startswith('Added files.') and store.load().servers['files'] == StdioServer(
        type='stdio', command='uvx'
    )
    assert isinstance(menu.source, ServersSource) and menu.source.title == 'MCP servers'
    assert menu.searchable is False
