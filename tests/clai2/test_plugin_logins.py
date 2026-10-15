"""Plugins add `/login NAME` sign-ins next to CLAI's own `openai-codex` and `github-copilot`."""

import io
from collections.abc import Sequence
from pathlib import Path
from typing import Generic, TypeVar

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_clai2 import chat
from pydantic_clai2.auth import CodexAuth, login_command
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import Settings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.models import login_names
from pydantic_clai2.plugins import Plugin, PluginHost, PluginLogin, SessionStart, load_plugin
from pydantic_clai2.plugins.loader import PluginLoader

PromptT = TypeVar('PromptT')

PLUGIN = """
from collections.abc import Sequence

from pydantic_clai2.plugins import Plugin, PluginLogin


async def sign_in() -> str:
    return '{MESSAGE}'


class ClaudeLogin(Plugin):
    def get_logins(self) -> Sequence[PluginLogin]:
        return (PluginLogin(name='claude', handler=sign_in, models=('claude-test:a', 'claude-test:b')),)
"""


async def signed_in() -> str:
    return 'Signed in.'


class ClaudeLogin(Plugin):
    def get_logins(self) -> Sequence[PluginLogin]:
        return (PluginLogin(name='claude', handler=signed_in),)


async def test_loaded_plugin_keeps_its_logins() -> None:
    loaded = load_plugin(ClaudeLogin, PluginHost[None](name='p', console=Console(file=io.StringIO()), settings={}))
    assert loaded.logins == (PluginLogin(name='claude', handler=signed_in),)
    assert await loaded.logins[0].handler() == 'Signed in.'


async def test_a_plugin_login_saves_its_models_only_once_it_succeeds(tmp_path: Path) -> None:
    async def refused() -> str:
        raise UserError('Sign-in refused.')

    store = SettingsStore(tmp_path / 'config.db')
    plugins = {
        login.name: login
        for login in (
            PluginLogin(name='claude', handler=signed_in, models=('claude-test:a', 'claude-test:b')),
            PluginLogin(name='refused', handler=refused, models=('claude-test:c',)),
        )
    }
    codex = CodexAuth(Console(file=io.StringIO()))
    with pytest.raises(UserError, match='refused'):
        await login_command(['refused'], codex=codex, plugins=plugins, store=store)
    assert store.models() == []
    assert await login_command(['claude'], codex=codex, plugins=plugins, store=store) == 'Signed in.'
    assert store.models() == ['claude-test:a', 'claude-test:b']


@pytest.mark.parametrize('name', ['', 'Claude', '1claude', 'claude code', 'claude:code'])
def test_rejects_a_malformed_login_name(name: str) -> None:
    with pytest.raises(ValueError, match=r'Login name .* must start with a lowercase letter'):
        PluginLogin(name=name, handler=signed_in)


@pytest.mark.parametrize('name', ['codex', 'copilot', 'openai-codex', 'github-copilot'])
def test_rejects_a_login_clai_already_has(name: str) -> None:
    with pytest.raises(ValueError, match='sign-in CLAI already has'):
        PluginLogin(name=name, handler=signed_in)


def test_login_names_list_clai_sign_ins_first_then_plugins_sorted() -> None:
    assert login_names() == ('openai-codex', 'github-copilot')
    assert login_names(['zeta', 'claude', 'claude']) == ('openai-codex', 'github-copilot', 'claude', 'zeta')


async def test_loader_merges_logins_and_the_later_plugin_wins(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.plugins_dir.mkdir(parents=True, exist_ok=True)
    (store.plugins_dir / 'a_first.py').write_text(PLUGIN.replace('{MESSAGE}', 'first'))
    (store.plugins_dir / 'b_second.py').write_text(PLUGIN.replace('{MESSAGE}', 'second'))
    loader: PluginLoader[None] = PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
    )
    await loader.load_all()
    logins = loader.logins()
    assert list(logins) == ['claude']
    assert await logins['claude'].handler() == 'second'
    await loader.disable('b_second')
    await loader.disable('a_first')
    assert loader.logins() == {}


async def test_shell_runs_and_completes_a_plugin_login(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    values = ['/login claude', '/exit']
    completions: list[set[str]] = []

    class Prompt(Generic[PromptT]):
        def __init__(self, **kwargs: object) -> None:
            self.completer = kwargs['completer']

        async def prompt_async(self, label: str, **kwargs: object) -> str:
            document = Document('/login ', 7)
            completions.append({item.text for item in self.completer.get_completions(document, CompleteEvent())})  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType]
            return values.pop(0)

    monkeypatch.setattr('pydantic_clai2._app.PromptSession', Prompt)
    store = SettingsStore(tmp_path / 'config.db')
    store.plugins_dir.mkdir(parents=True, exist_ok=True)
    (store.plugins_dir / 'claude_login.py').write_text(PLUGIN.replace('{MESSAGE}', 'Signed in to Claude Code.'))
    output = io.StringIO()
    await chat(
        Agent('test'), deps=None, settings=Settings(model='test'), console=Console(file=output, width=200), store=store
    )
    assert 'Signed in to Claude Code.' in output.getvalue()
    assert completions[0] == {'openai-codex', 'github-copilot', 'claude'}
    assert {'claude-test:a', 'claude-test:b'} <= set(store.models())
