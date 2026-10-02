"""The built-in `github` plugin: a settings menu for `GitHub`'s options; tokens come from `gh` or `/keys`."""

import io
from pathlib import Path

import anyio
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
from pydantic_clai2.config.api_keys import KeyReference, SavedKey
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.gh_cli import GhToken
from pydantic_clai2.plugins import PluginHost, SessionStart
from pydantic_clai2.plugins.loader import PluginError, PluginLoader
from pydantic_clai2.ui.menus.field_menu import CUSTOM, FieldRow
from pydantic_clai2.ui.menus.plugin_menu import PluginMenu, open_plugins_menu
from tests.clai2.menu_script import Script, pick, typed

BUILTIN = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'github')
CLOSE = MenuResult(cancelled=True)


class Shell:
    """A loader plus what a test inspects: printed output and the settings file."""

    def __init__(self, tmp_path: Path, settings: dict[str, JsonValue] | None = None) -> None:
        self.path = tmp_path / 'config.db'
        self.store = SettingsStore(self.path)
        self.output = io.StringIO()
        declaration = BUILTIN if settings is None else BUILTIN.model_copy(update={'settings': settings})
        self.loader: PluginLoader[None] = PluginLoader(
            store=self.store,
            console=Console(file=self.output, width=200),
            commands=Commands(),
            session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=self.store.load()),
            builtin=(declaration,),
        )

    def saved(self) -> dict[str, JsonValue]:
        [declaration] = self.store.plugins()
        assert declaration.enabled
        return declaration.settings


def script(
    monkeypatch: pytest.MonkeyPatch,
    lists: list[MenuResult],
    choices: list[MenuResult] | None = None,
    texts: list[TextInputResult] | None = None,
) -> Script:
    scripted = Script(lists=[*lists, CLOSE], choices=choices or [], texts=texts or [])
    monkeypatch.setattr('pydantic_clai2.builtin_plugins.github.RUNNERS', scripted.runners)
    return scripted


def key_choice(monkeypatch: pytest.MonkeyPatch, choice: str | KeyReference | None) -> list[str]:
    """Answer `prompt_api_key`, whose saved-key list needs a real terminal, and record its labels."""
    labels: list[str] = []

    async def prompt_api_key(*, prompt: object, label: str, optional: bool = False) -> str | KeyReference | None:
        labels.append(label)
        return choice

    monkeypatch.setattr('pydantic_clai2.builtin_plugins.github.prompt_api_key', prompt_api_key)
    return labels


GH = GhToken(hostname='github.com', setup=SETUP)


def github(auth: SavedKey | GhToken = GH, *, read_only: bool = True) -> list[GitHub[None]]:
    return [GitHub[None](auth=auth, url=GITHUB_MCP_URL, read_only=read_only)]


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
    labels = key_choice(monkeypatch, ' ghp_new ')
    shown = script(
        monkeypatch,
        lists=[pick('login'), pick('url'), pick('read_only'), pick('toolsets'), pick('include_instructions')],
        choices=[pick('key'), pick(ENTERPRISE), pick('false'), pick(CUSTOM), pick('false')],
        texts=[typed('octocorp.ghe.com'), typed('repos, issues')],
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
    assert 'GitHub has no token: the GitHub CLI is not signed in to github.com.' in shell.output.getvalue()
    assert labels == ['GitHub token (saved in /keys as GITHUB_TOKEN)']
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
            auth=SavedKey(name='GITHUB_TOKEN', setup=SETUP),
            url='https://copilot-api.octocorp.ghe.com/mcp',
            read_only=False,
            toolsets=['repos', 'issues'],
            include_instructions=False,
        )
    ]


async def test_reopening_repicks_a_saved_key_and_host_without_reinstalling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_keys.save_key(name='WORK_GITHUB', value='work-secret')
    shell = Shell(tmp_path, {'url': 'https://copilot-api.octocorp.ghe.com/mcp', 'read_only': False})
    script(monkeypatch, lists=[])
    await shell.loader.command(['enable', 'github'])
    key_choice(monkeypatch, KeyReference(name='WORK_GITHUB'))
    script(monkeypatch, lists=[pick('login'), pick('url')], choices=[pick('key'), pick(GITHUB_MCP_URL)])
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


async def test_new_token_is_typed_masked_when_no_keys_are_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[])
    await shell.loader.enable('github')
    script(monkeypatch, lists=[pick('login')], choices=[pick('key')], texts=[typed('ghp_typed')])
    await shell.loader.configure('github')
    assert api_keys.load_keys()['GITHUB_TOKEN'].get_secret_value() == 'ghp_typed'


@pytest.mark.parametrize('answer', [TextInputResult(cancelled=True), typed('   ')])
async def test_cancelled_or_blank_token_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: TextInputResult
) -> None:
    shell = Shell(tmp_path)
    await shell.loader.enable('github')
    script(monkeypatch, lists=[pick('login')], choices=[pick('key')], texts=[answer])
    assert await shell.loader.configure('github') == 'GitHub settings unchanged.'
    assert api_keys.load_keys() == {}
    assert shell.store.plugins()[0].settings == {}


async def test_replacing_a_shared_key_needs_confirmation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api_keys.save_key(name='GITHUB_TOKEN', value='shared')
    shell = Shell(tmp_path)
    await shell.loader.enable('github')
    key_choice(monkeypatch, 'other')
    script(
        monkeypatch,
        lists=[pick('login'), pick('login')],
        choices=[pick('key'), pick(False), pick('key'), pick(True)],
    )
    assert await shell.loader.configure('github') == 'GitHub uses the saved key GITHUB_TOKEN. Manage it in /keys.'
    assert api_keys.load_keys()['GITHUB_TOKEN'].get_secret_value() == 'other'


async def test_a_key_created_while_picking_still_needs_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = Shell(tmp_path)
    await shell.loader.enable('github')

    async def prompt_api_key(*, prompt: object, label: str, optional: bool = False) -> str | KeyReference | None:
        api_keys.save_key(name='GITHUB_TOKEN', value='from-another-process')
        return 'typed-here'

    monkeypatch.setattr('pydantic_clai2.builtin_plugins.github.prompt_api_key', prompt_api_key)
    script(monkeypatch, lists=[pick('login')], choices=[pick('key'), CLOSE])
    assert await shell.loader.configure('github') == 'GitHub settings unchanged.'
    assert api_keys.load_keys()['GITHUB_TOKEN'].get_secret_value() == 'from-another-process'


def test_save_key_without_replace_refuses_an_existing_name() -> None:
    api_keys.save_key(name='SHARED', value='first', replace=False)
    with pytest.raises(api_keys.KeyExistsError, match='SHARED is already saved'):
        api_keys.save_key(name='SHARED', value='second', replace=False)
    assert api_keys.load_keys()['SHARED'].get_secret_value() == 'first'


@pytest.mark.parametrize(
    ('choice', 'text'),
    [(CLOSE, None), (pick(ENTERPRISE), TextInputResult(cancelled=True)), (pick(ENTERPRISE), typed(' '))],
)
async def test_cancelled_host_choice_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, choice: MenuResult, text: TextInputResult | None
) -> None:
    shell = Shell(tmp_path)
    await shell.loader.enable('github')
    script(monkeypatch, lists=[pick('url')], choices=[choice], texts=[text] if text else [])
    assert await shell.loader.configure('github') == 'GitHub settings unchanged.'


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


async def test_cancelling_configure_cancels_an_open_token_picker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = Shell(tmp_path)
    await shell.loader.enable('github')
    opened = anyio.Event()
    finished: list[str] = []

    async def prompt_api_key(*, prompt: object, label: str, optional: bool = False) -> str | KeyReference | None:
        opened.set()
        try:
            await anyio.sleep_forever()
        finally:
            finished.append(label)
        return None  # pragma: no cover -- unreachable; keeps the signature honest

    monkeypatch.setattr('pydantic_clai2.builtin_plugins.github.prompt_api_key', prompt_api_key)
    script(monkeypatch, lists=[pick('login')], choices=[pick('key')])
    with anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(shell.loader.configure, 'github')
            await opened.wait()
            tasks.cancel_scope.cancel()
    assert finished == ['GitHub token (saved in /keys as GITHUB_TOKEN)']


class Redraw:
    def replace_items(self, items: object) -> None:
        pass
