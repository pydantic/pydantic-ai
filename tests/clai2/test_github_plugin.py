"""The built-in `github` plugin: a settings menu for `GitHub`'s options; tokens come from `gh` or `/keys`."""

import io
from pathlib import Path

import pytest
from pydantic import JsonValue
from rich.console import Console
from termflow.tui.menu import MenuResult
from termflow.tui.textinput import TextInputResult

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.github import GITHUB_MCP_URL, GitHub
from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.builtin_plugins.github import ENTERPRISE, SETUP, GitHubSource, enterprise_url
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import api_keys
from pydantic_clai2.config.api_keys import SavedKey
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.gh_cli import GhToken
from pydantic_clai2.plugins import PluginHost, SessionStart
from pydantic_clai2.plugins.loader import PluginError, PluginLoader
from pydantic_clai2.plugins.menus import Option, SettingsMenu
from pydantic_clai2.ui.menus import dialogs
from pydantic_clai2.ui.menus.field_menu import CUSTOM, FieldRow
from pydantic_clai2.ui.menus.plugin_menu import PluginMenu, open_plugins_menu
from tests.clai2.dialog_script import Back, Confirm, Pick, ScriptedDialogs, Type
from tests.clai2.menu_script import Script, pick, typed

pytestmark = pytest.mark.subprocess(
    reason='drives the fake `gh` CLI (`tests/clai2/fake_gh_cli.py`) as a real child process'
)

BUILTIN = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'github')
CLOSE = MenuResult(cancelled=True)
TOKEN_LABEL = 'GitHub token (saved in /keys as GITHUB_TOKEN)'


class Shell:
    """A loader plus what a test inspects: printed output and the settings file."""

    def __init__(self, tmp_path: Path, settings: dict[str, JsonValue] | None = None, *, terminal: bool = True) -> None:
        self.path = tmp_path / 'config.db'
        self.store = SettingsStore(self.path)
        self.output = io.StringIO()
        declaration = BUILTIN if settings is None else BUILTIN.model_copy(update={'settings': settings})
        self.loader: PluginLoader[None] = PluginLoader(
            store=self.store,
            console=Console(file=self.output, width=200, force_terminal=terminal),
            commands=Commands(),
            session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=self.store.load()),
            builtin=(declaration,),
        )

    def saved(self) -> dict[str, JsonValue]:
        [declaration] = self.store.plugins()
        assert declaration.enabled
        return declaration.settings

    async def act(self, row: str, answers: ScriptedDialogs) -> list[str]:
        """Run a row's action as the web UI does: through its own dialogs, then reload the plugin."""
        return list(await self.loader.edit_settings('github', lambda menu: menu.actions[row](answers)))


def script(
    monkeypatch: pytest.MonkeyPatch,
    lists: list[MenuResult],
    choices: list[MenuResult] | None = None,
    texts: list[TextInputResult] | None = None,
) -> Script:
    """Drive the terminal menu and its actions' dialogs."""
    scripted = Script(lists=[*lists, CLOSE], choices=choices or [], texts=texts or [])
    monkeypatch.setattr(dialogs, 'RUNNERS', scripted.runners)
    return scripted


GH = GhToken(hostname='github.com', setup=SETUP)


def github(auth: SavedKey | GhToken = GH, *, read_only: bool = True) -> list[GitHub[None]]:
    return [GitHub[None](defer_loading=True, auth=auth, url=GITHUB_MCP_URL, read_only=read_only)]


def test_declared_disabled_with_no_settings() -> None:
    assert BUILTIN.factory == 'pydantic_clai2.builtin_plugins.github'
    assert not BUILTIN.enabled
    assert BUILTIN.settings == {}


def test_saved_key_is_resolved_on_every_run() -> None:
    token = SavedKey(name='GITHUB_TOKEN', setup=SETUP)
    api_keys.save_key(name='GITHUB_TOKEN', value='first')
    assert token(None) == 'first'
    api_keys.save_key(name='GITHUB_TOKEN', value='replaced')
    assert token(None) == 'replaced'
    api_keys.delete_key(name='GITHUB_TOKEN')
    with pytest.raises(UserError, match=r'GITHUB_TOKEN is missing\. Run /plugins configure github'):
        token(None)


async def test_enable_opens_the_menu_and_every_option_saves_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('GITHUB_TOKEN', 'the-environment-is-not-a-source')
    shown = script(
        monkeypatch,
        lists=[pick('login'), pick('url'), pick('read_only'), pick('toolsets'), pick('include_instructions')],
        choices=[pick('key'), pick(ENTERPRISE), pick('false'), pick(CUSTOM), pick('false')],
        texts=[typed(' ghp_new '), typed('octocorp.ghe.com'), typed('repos, issues')],
    )
    shell = Shell(tmp_path)
    assert await shell.loader.command(['enable', 'github']) == '\n'.join(
        [
            'Enabled github.',
            'GitHub uses the saved key GITHUB_TOKEN. Manage it in /keys.',
            'Saved Enterprise URL.',
            'Saved Tools.',
            'Saved Tool groups.',
            'Saved Server instructions.',
        ]
    )
    assert shown.opened.count('list') == 6
    assert api_keys.load_keys()['GITHUB_TOKEN'].get_secret_value() == 'ghp_new'
    assert shell.saved() == {
        'login': 'key',
        'token': {'name': 'GITHUB_TOKEN'},
        'url': 'https://copilot-api.octocorp.ghe.com/mcp',
        'read_only': False,
        'toolsets': ['repos', 'issues'],
        'include_instructions': False,
    }
    assert b'ghp_new' not in shell.path.read_bytes()
    assert shell.loader.capabilities() == [
        GitHub[None](
            defer_loading=True,
            auth=SavedKey(name='GITHUB_TOKEN', setup=SETUP),
            url='https://copilot-api.octocorp.ghe.com/mcp',
            read_only=False,
            toolsets=['repos', 'issues'],
            include_instructions=False,
        )
    ]


async def test_without_a_terminal_enable_warns_and_says_where_to_configure(tmp_path: Path) -> None:
    shell = Shell(tmp_path, terminal=False)
    assert await shell.loader.command(['enable', 'github']) == (
        'Enabled github.\nConfigure GitHub from a terminal: /plugins configure github'
    )
    assert 'GitHub has no token: the GitHub CLI is not signed in to github.com.' in shell.output.getvalue()


async def test_declares_its_rows_and_the_sign_in_and_host_actions(tmp_path: Path) -> None:
    shell = Shell(tmp_path, terminal=False)
    await shell.loader.enable('github')

    async def declared(menu: SettingsMenu) -> list[str]:
        assert menu.source.title == 'GitHub' and menu.searchable
        assert set(menu.actions) == {'login', 'url'}
        return [row.key for row in menu.source.rows()]

    keys = await shell.loader.edit_settings('github', declared)
    assert keys == ['login', 'url', 'read_only', 'toolsets', 'include_instructions']


async def test_reopening_repicks_a_saved_key_and_host_without_reinstalling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_keys.save_key(name='WORK_GITHUB', value='work-secret')
    shell = Shell(tmp_path, {'url': 'https://copilot-api.octocorp.ghe.com/mcp', 'read_only': False})
    script(monkeypatch, lists=[])
    await shell.loader.command(['enable', 'github'])
    script(
        monkeypatch,
        lists=[pick('login'), pick('url')],
        choices=[pick('key'), pick('WORK_GITHUB'), pick(GITHUB_MCP_URL)],
    )
    assert await shell.loader.command(['configure', 'github']) == (
        'GitHub uses the saved key WORK_GITHUB. Manage it in /keys.\nSaved GitHub host.'
    )
    assert shell.saved() == {
        'login': 'key',
        'token': {'name': 'WORK_GITHUB'},
        'url': GITHUB_MCP_URL,
        'read_only': False,
        'toolsets': None,
        'include_instructions': True,
    }
    assert b'work-secret' not in shell.path.read_bytes()
    assert shell.loader.capabilities() == github(SavedKey(name='WORK_GITHUB', setup=SETUP), read_only=False)


async def test_sign_in_row_asks_where_the_token_comes_from_then_for_a_masked_token(tmp_path: Path) -> None:
    shell = Shell(tmp_path, terminal=False)
    await shell.loader.enable('github')
    answers = ScriptedDialogs([Pick(value='key'), Type(text='ghp_typed')])
    assert await shell.act('login', answers) == ['GitHub uses the saved key GITHUB_TOKEN. Manage it in /keys.']
    choose, token = answers.asked
    assert (choose.title, choose.current) == ('Choose Sign-in', 'gh')
    assert choose.options == (
        Option(value='gh', label='GitHub CLI (browser)'),
        Option(value='key', label='Token saved in /keys'),
    )
    assert (token.kind, token.title, token.secret) == ('text', TOKEN_LABEL, True)
    assert api_keys.load_keys()['GITHUB_TOKEN'].get_secret_value() == 'ghp_typed'
    assert shell.loader.capabilities() == github(SavedKey(name='GITHUB_TOKEN', setup=SETUP))


@pytest.mark.parametrize('answers', [[Back()], [Pick(value='key'), Back()], [Pick(value='key'), Type(text='   ')]])
async def test_cancelled_or_blank_token_changes_nothing(tmp_path: Path, answers: list[Back | Pick | Type]) -> None:
    shell = Shell(tmp_path, terminal=False)
    await shell.loader.enable('github')
    assert await shell.act('login', ScriptedDialogs(answers)) == []
    assert api_keys.load_keys() == {}
    assert shell.store.plugins()[0].settings == {}


async def test_replacing_a_shared_key_needs_confirmation(tmp_path: Path) -> None:
    api_keys.save_key(name='GITHUB_TOKEN', value='shared')
    shell = Shell(tmp_path, terminal=False)
    await shell.loader.enable('github')
    kept = ScriptedDialogs([Pick(value='key'), Pick(value='enter'), Type(text='other'), Confirm(yes=False)])
    assert await shell.act('login', kept) == []
    confirm = kept.asked[-1]
    assert (confirm.title, confirm.no) == ('GITHUB_TOKEN is already in /keys', 'Keep the saved token')
    assert api_keys.load_keys()['GITHUB_TOKEN'].get_secret_value() == 'shared'
    replaced = ScriptedDialogs([Pick(value='key'), Pick(value='enter'), Type(text='other'), Confirm(yes=True)])
    assert await shell.act('login', replaced) == ['GitHub uses the saved key GITHUB_TOKEN. Manage it in /keys.']
    assert api_keys.load_keys()['GITHUB_TOKEN'].get_secret_value() == 'other'


def test_save_key_without_replace_refuses_an_existing_name() -> None:
    api_keys.save_key(name='SHARED', value='first', replace=False)
    with pytest.raises(api_keys.KeyExistsError, match='SHARED is already saved'):
        api_keys.save_key(name='SHARED', value='second', replace=False)
    assert api_keys.load_keys()['SHARED'].get_secret_value() == 'first'


async def test_host_row_offers_github_or_an_enterprise_host(tmp_path: Path) -> None:
    shell = Shell(tmp_path, terminal=False)
    await shell.loader.enable('github')
    answers = ScriptedDialogs([Pick(value=ENTERPRISE), Type(text='octocorp.ghe.com')])
    assert await shell.act('url', answers) == ['Saved Enterprise URL.']
    choose, text = answers.asked
    assert (choose.title, choose.current) == ('Choose GitHub host', GITHUB_MCP_URL)
    assert choose.options == (
        Option(value=GITHUB_MCP_URL, label='github.com'),
        Option(value=ENTERPRISE, label='GitHub Enterprise Cloud (ghe.com)'),
    )
    assert (text.title, text.placeholder) == ('New value for Enterprise URL', f'current: {GITHUB_MCP_URL}')
    assert text.description == 'Your ghe.com host, such as octocorp.ghe.com, or the full MCP URL.'
    assert shell.saved()['url'] == 'https://copilot-api.octocorp.ghe.com/mcp'
    back = ScriptedDialogs([Pick(value=GITHUB_MCP_URL)])
    assert await shell.act('url', back) == ['Saved GitHub host.']
    assert back.asked[0].current == ENTERPRISE
    assert shell.saved()['url'] == GITHUB_MCP_URL


async def test_enterprise_url_is_checked_as_it_is_typed(tmp_path: Path) -> None:
    shell = Shell(tmp_path, terminal=False)
    await shell.loader.enable('github')
    with pytest.raises(AssertionError, match=r'would not accept .*Use an https:// URL'):
        await shell.act('url', ScriptedDialogs([Pick(value=ENTERPRISE), Type(text='http://octocorp.example.com')]))
    assert shell.saved() == {}


@pytest.mark.parametrize(
    'answers', [[Back()], [Pick(value=ENTERPRISE), Back()], [Pick(value=ENTERPRISE), Type(text=' ')]]
)
async def test_cancelled_host_choice_changes_nothing(tmp_path: Path, answers: list[Back | Pick | Type]) -> None:
    shell = Shell(tmp_path, terminal=False)
    await shell.loader.enable('github')
    assert await shell.act('url', ScriptedDialogs(answers)) == []
    assert shell.saved() == {}


async def test_closing_the_menu_without_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[])
    await shell.loader.enable('github')
    script(monkeypatch, lists=[pick('url')], choices=[CLOSE])
    assert await shell.loader.configure('github') == 'GitHub: nothing changed.'


@pytest.mark.parametrize(
    ('typed_url', 'url'),
    [
        ('octocorp.ghe.com', 'https://copilot-api.octocorp.ghe.com/mcp'),
        ('https://octocorp.ghe.com/', 'https://copilot-api.octocorp.ghe.com/mcp'),
        ('copilot-api.octocorp.ghe.com', 'https://copilot-api.octocorp.ghe.com/mcp'),
        ('https://copilot-api.octocorp.ghe.com/mcp/readonly', 'https://copilot-api.octocorp.ghe.com/mcp/readonly'),
        ('https://mcp-proxy.example.com/github', 'https://mcp-proxy.example.com/github'),
    ],
)
def test_enterprise_url_accepts_a_ghe_host_or_a_full_url(typed_url: str, url: str) -> None:
    assert enterprise_url(typed_url) == url


def test_menu_validates_and_resets_like_saving_would() -> None:
    host = PluginHost[None](
        name='github', console=Console(file=io.StringIO()), settings={'read_only': False, 'login': 'key'}
    )
    source = GitHubSource(host)
    rows = {row.key: row for row in source.rows()}
    assert rows['login'].note == 'no token'
    assert source.current(rows['login']) == 'GITHUB_TOKEN in /keys'
    enterprise = FieldRow(key='enterprise_url', label='Enterprise URL', description='', default=GITHUB_MCP_URL)
    assert source.problem(enterprise, 'http://octocorp.example.com') == 'Value error, Use an https:// URL.'
    assert source.problem(rows['toolsets'], 'Repos') == (
        'Value error, Tool groups are lowercase names such as repos, separated by commas.'
    )
    assert source.problem(rows['read_only'], 'maybe') == 'Input should be a valid boolean'
    assert source.problem(rows['toolsets'], 'repos,issues') is None
    assert source.current(rows['read_only']) == 'false'
    assert source.reset(rows['read_only']) == 'Reset Tools.'
    assert source.current(rows['read_only']) == 'true'
    assert source.current(rows['toolsets']) == 'default'
    assert source.current(rows['url']) == GITHUB_MCP_URL
    api_keys.save_key(name='GITHUB_TOKEN', value='saved')
    assert {row.key: row for row in source.rows()}['login'].note == ''
    assert source.reset(rows['login']) == 'Reset Sign-in.'
    assert source.current(rows['login']) == 'gh'


@pytest.mark.parametrize(
    'settings',
    [
        {'token': 'ghp_inline_secret'},
        {'token': {'name': 'ghp_inline_secret'}},
        {'token': {'name': 'X', 'value': 'secret'}},
        {'toolsets': []},
        {'url': 'http://api.githubcopilot.com/mcp/'},
        {'auth': 'secret'},
        {'login': 'oauth'},
    ],
)
async def test_settings_cannot_hold_a_secret_or_invalid_options(tmp_path: Path, settings: dict[str, JsonValue]) -> None:
    shell = Shell(tmp_path, settings)
    with pytest.raises(PluginError):
        await shell.loader.enable('github')
    assert shell.loader.capabilities() == []


async def test_add_replacing_the_builtin_opens_the_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[pick('read_only')], choices=[pick('true')])
    assert await shell.loader.command(
        ['add', 'github', 'pydantic_clai2.builtin_plugins.github', '{"read_only": false}']
    ) == ('Replaced built-in github.\nSaved Tools.')
    assert shell.loader.capabilities() == github()


async def test_configure_needs_a_loaded_plugin_with_a_menu(tmp_path: Path) -> None:
    shell = Shell(tmp_path)
    with pytest.raises(ValueError, match='not loaded; enable it before configuring'):
        await shell.loader.command(['configure', 'github'])
    shell.store.save_plugin(
        BUILTIN.model_copy(update={'id': 'plain', 'factory': 'pydantic_clai2.builtin_plugins.repo_context'})
    )
    assert await shell.loader.command(['enable', 'plain']) == 'Enabled plain.'
    with pytest.raises(ValueError, match='no settings menu'):
        await shell.loader.configure('plain')


async def test_enabling_from_plugins_menu_opens_the_settings_menu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[pick('read_only')], choices=[pick('false')])
    notices: list[str | None] = []

    def run(menu: PluginMenu[None]) -> MenuResult:
        notices.append(menu.notice)
        github_row, save_and_close = menu.items()
        if len(notices) > 1:
            return MenuResult(item=save_and_close)
        result = menu.toggle(Redraw(), github_row)
        assert result is not None
        return result

    assert await open_plugins_menu(shell.loader, run=run) == 'Saved Tools.'
    assert notices == [None, 'Saved Tools.']
    assert shell.loader.capabilities() == github(read_only=False)


class Redraw:
    def replace_items(self, items: object) -> None:
        pass
