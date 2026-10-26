"""The `github` plugin's GitHub CLI sign-in, driven through a fake `gh` (see `conftest.fake_gh`)."""

import io
import time
import webbrowser
from collections.abc import Callable
from pathlib import Path

import anyio
import pytest
from pydantic import JsonValue
from rich.console import Console
from termflow.tui.menu import MenuResult

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.github import GITHUB_MCP_URL, GitHub
from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.builtin_plugins.github import SETUP
from pydantic_clai2.commands import Commands
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.gh_cli import INSTALL, GhLogin, GhToken, gh_command, gh_host, gh_token, start_login
from pydantic_clai2.plugins import SessionStart
from pydantic_clai2.plugins.loader import PluginLoader
from pydantic_clai2.ui.menus import dialogs
from tests.clai2.conftest import FakeGh
from tests.clai2.dialog_script import Answer, Back, Finish, Pick, ScriptedDialogs
from tests.clai2.menu_script import UNTIL_CLOSED, Script, pick

pytestmark = pytest.mark.subprocess(
    reason='drives the fake `gh` CLI (`tests/clai2/fake_gh_cli.py`) as a real child process'
)

BUILTIN = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'github')
CLOSE = MenuResult(cancelled=True)
DEVICE = 'https://github.com/login/device'
GHE = 'https://copilot-api.octocorp.ghe.com/mcp'
CODE = f'Enter ABCD-1234 at {DEVICE}'


class Shell:
    def __init__(self, tmp_path: Path, settings: dict[str, JsonValue] | None = None, *, terminal: bool = False) -> None:
        self.store = SettingsStore(tmp_path / 'config.db')
        self.output = io.StringIO()
        self.loader: PluginLoader[None] = PluginLoader(
            store=self.store,
            console=Console(file=self.output, width=200, force_terminal=terminal),
            commands=Commands(),
            session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=self.store.load()),
            builtin=(BUILTIN.model_copy(update={'settings': settings or {}}),),
        )

    def login(self) -> JsonValue:
        [declaration] = self.store.plugins()
        return declaration.settings.get('login')

    async def sign_in(self, *answers: Answer) -> tuple[list[str], ScriptedDialogs]:
        """Run the Sign-in row's action as the web UI does, through scripted dialogs."""
        asked = ScriptedDialogs([Pick(value='gh'), *answers])
        return await self.sign_in_through(asked), asked

    async def sign_in_through(self, asked: ScriptedDialogs) -> list[str]:
        return list(await self.loader.edit_settings('github', lambda menu: menu.actions['login'](asked)))


async def test_browser_sign_in_through_gh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh) -> None:
    shell = Shell(tmp_path, {'login': 'key'})
    await shell.loader.enable('github')
    assert 'GITHUB_TOKEN is not in /keys' in shell.output.getvalue()
    messages, asked = await shell.sign_in(Finish())
    assert messages == ['Signed in to GitHub as octocat.', 'GitHub uses your GitHub CLI login for github.com.']
    [choose, wait] = asked.asked
    assert (choose.title, choose.current) == ('Choose Sign-in', 'key')
    assert (wait.kind, wait.title, wait.shown) == ('wait', 'Signing in to GitHub', [CODE])
    assert 'gh copied it to your clipboard' in wait.description
    assert fake_gh.opened == [DEVICE]
    assert 'auth login --hostname github.com --web --clipboard' in fake_gh.calls()
    assert shell.login() == 'gh'
    assert shell.loader.capabilities() == [
        GitHub[None](defer_loading=True, auth=GhToken(hostname='github.com', setup=SETUP), read_only=True)
    ]
    assert GhToken(hostname='github.com', setup=SETUP)(None) == 'gho_browser'


async def test_the_terminal_waiting_screen_closes_once_gh_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh
) -> None:
    shell = Shell(tmp_path, {'login': 'key'}, terminal=True)
    await shell.loader.enable('github')
    shown = Script(lists=[pick('login'), CLOSE], choices=[pick('gh'), UNTIL_CLOSED], texts=[])
    monkeypatch.setattr(dialogs, 'RUNNERS', shown.runners)
    with anyio.fail_after(20):
        assert await shell.loader.configure('github') == (
            'Signed in to GitHub as octocat.\nGitHub uses your GitHub CLI login for github.com.'
        )
    assert shown.opened == ['list', 'choice', 'choice', 'list']
    assert fake_gh.opened == [DEVICE]
    assert shell.login() == 'gh'


async def test_an_existing_gh_login_is_used_without_a_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh
) -> None:
    fake_gh.sign_in('octocorp.ghe.com')
    shell = Shell(tmp_path, {'login': 'key', 'url': GHE})
    await shell.loader.enable('github')
    assert (await shell.sign_in())[0] == ['GitHub uses your GitHub CLI login for octocorp.ghe.com.']
    assert fake_gh.opened == []
    assert shell.loader.capabilities() == [
        GitHub[None](
            defer_loading=True, auth=GhToken(hostname='octocorp.ghe.com', setup=SETUP), url=GHE, read_only=True
        )
    ]


async def test_signed_in_gh_loads_without_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh
) -> None:
    fake_gh.sign_in()
    shell = Shell(tmp_path, terminal=True)
    monkeypatch.setattr(dialogs, 'RUNNERS', Script(lists=[CLOSE], choices=[], texts=[]).runners)
    assert await shell.loader.command(['enable', 'github']) == 'Enabled github.\nGitHub: nothing changed.'
    assert shell.output.getvalue() == ''


@pytest.mark.parametrize(
    ('mode', 'message'),
    [
        ('fail', 'gh auth login failed: error: the code expired'),
        ('early', 'gh auth login failed: error: authentication failed before a code was issued'),
        ('no-token', 'Signed in to GitHub.'),
    ],
)
async def test_a_failed_sign_in_keeps_the_previous_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, mode: str, message: str
) -> None:
    fake_gh.next_login(mode)
    shell = Shell(tmp_path, {'login': 'key'})
    await shell.loader.enable('github')
    assert (await shell.sign_in(Finish()))[0] == [message]
    assert shell.login() == 'key'


@pytest.mark.parametrize('answers', [[Back()], [Pick(value='gh'), Back()]])
async def test_cancelling_sign_in_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, answers: list[Answer]
) -> None:
    fake_gh.next_login('hang')
    shell = Shell(tmp_path, {'login': 'key'})
    await shell.loader.enable('github')
    asked = ScriptedDialogs(answers)
    with anyio.fail_after(20):
        assert await shell.loader.edit_settings('github', lambda menu: menu.actions['login'](asked)) == []
    assert shell.login() == 'key'
    assert not (fake_gh.state / 'github.com').exists()


async def test_sign_in_continues_when_no_browser_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh
) -> None:
    def no_browser(url: str) -> bool:
        raise webbrowser.Error('no browser')

    monkeypatch.setattr('pydantic_clai2.builtin_plugins.github.OPEN_BROWSER', no_browser)
    shell = Shell(tmp_path)
    await shell.loader.enable('github')
    messages, asked = await shell.sign_in(Finish())
    assert messages[0] == 'Signed in to GitHub as octocat.'
    assert asked.asked[-1].shown == [CODE], 'the link is on screen instead'


async def test_without_gh_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('pydantic_clai2.gh_cli.gh_command', lambda: None)
    shell = Shell(tmp_path)
    await shell.loader.enable('github')
    assert f'The GitHub CLI (gh) is not installed. {INSTALL}' in shell.output.getvalue()
    assert (await shell.sign_in())[0] == [f'The GitHub CLI (gh) is not installed. {INSTALL}']
    with pytest.raises(UserError, match='not installed'):
        GhToken(hostname='github.com', setup=SETUP)(None)


def test_gh_token_ignores_token_variables(monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh) -> None:
    monkeypatch.setenv('GH_TOKEN', 'from-the-environment')
    monkeypatch.setenv('GITHUB_TOKEN', 'from-the-environment')
    token = GhToken(hostname='github.com', setup=SETUP)
    with pytest.raises(UserError, match=r'no login for github\.com\. Run /plugins configure github'):
        token(None)
    fake_gh.sign_in(token='gho_first')
    assert token(None) == 'gho_first'
    fake_gh.sign_in(token='gho_switched')
    assert token(None) == 'gho_switched'


@pytest.mark.parametrize(
    ('url', 'host'),
    [(GITHUB_MCP_URL, 'github.com'), (GHE, 'octocorp.ghe.com'), ('https://mcp.example.com/github', 'mcp.example.com')],
)
def test_gh_host(url: str, host: str) -> None:
    assert gh_host(url) == host


def test_gh_command_finds_gh_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PATH', str(tmp_path))
    assert gh_command() is None
    executable = tmp_path / 'gh'
    executable.write_text('#!/bin/sh\n')
    executable.chmod(0o755)
    assert gh_command() == [str(executable)]


def test_a_stalled_gh_auth_token_fails_instead_of_hanging(monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh) -> None:
    monkeypatch.setattr('pydantic_clai2.gh_cli.TOKEN_TIMEOUT', 0.3)
    fake_gh.hang_on_token()
    with pytest.raises(UserError, match=r'gh auth token did not answer within 0\.3 seconds'):
        gh_token('github.com')


async def test_a_login_that_shows_no_code_gives_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh
) -> None:
    monkeypatch.setattr('pydantic_clai2.gh_cli.CODE_TIMEOUT', 0.3)
    fake_gh.next_login('silent')
    shell = Shell(tmp_path, {'login': 'key'})
    await shell.loader.enable('github')
    messages, asked = await shell.sign_in(Finish())
    assert messages == ['gh auth login showed no code within 0.3 seconds.']
    assert asked.asked[-1].shown == []
    assert shell.login() == 'key'


def test_start_login_stops_when_asked(fake_gh: FakeGh) -> None:
    fake_gh.next_login('silent')
    assert start_login('github.com', stopping=lambda: True) is None


async def cancel_once_gh_starts(shell: Shell, fake_gh: FakeGh) -> None:
    """Start the sign-in, and cancel it as closing the web dialog or the terminal menu would."""
    asked = ScriptedDialogs([Pick(value='gh'), Finish()])
    with anyio.fail_after(20):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(shell.sign_in_through, asked)
            while not any(call.startswith('auth login') for call in fake_gh.calls()):
                await anyio.sleep(0.02)
            tasks.cancel_scope.cancel()


async def test_cancelling_stops_a_silent_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh
) -> None:
    fake_gh.next_login('silent')
    shell = Shell(tmp_path, {'login': 'key'})
    await shell.loader.enable('github')
    await cancel_once_gh_starts(shell, fake_gh)
    assert shell.login() == 'key'


async def test_cancelling_as_the_code_arrives_still_stops_gh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh
) -> None:
    fake_gh.next_login('hang')
    started: list[GhLogin] = []

    def slow_start(hostname: str, *, stopping: Callable[[], bool]) -> GhLogin | str | None:
        login = start_login(hostname)
        assert isinstance(login, GhLogin)
        started.append(login)
        while not stopping():  # The code arrived, but the sign-in is cancelled before it is shown.
            time.sleep(0.01)
        return login

    monkeypatch.setattr('pydantic_clai2.builtin_plugins.github.start_login', slow_start)
    shell = Shell(tmp_path, {'login': 'key'})
    await shell.loader.enable('github')
    await cancel_once_gh_starts(shell, fake_gh)
    with anyio.fail_after(10):
        while not started or started[0].process.poll() is None:
            await anyio.sleep(0.02)
    assert shell.login() == 'key'


async def test_a_stalled_check_after_sign_in_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh
) -> None:
    monkeypatch.setattr('pydantic_clai2.gh_cli.TOKEN_TIMEOUT', 0.5)
    fake_gh.next_login('stall-token')
    shell = Shell(tmp_path, {'login': 'key'})
    await shell.loader.enable('github')
    assert (await shell.sign_in(Finish()))[0] == [
        'Signed in to GitHub as octocat.',
        'gh auth token did not answer within 0.5 seconds.',
    ]
    assert shell.login() == 'key'
