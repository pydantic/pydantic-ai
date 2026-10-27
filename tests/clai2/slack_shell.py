"""A CLAI shell with only the `slack` plugin loaded, scripted terminal widgets, and scripted dialogs."""

import io
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import TypeGuard

import pytest
from pydantic import JsonValue
from rich.console import Console
from termflow.tui.menu import MenuResult
from termflow.tui.textinput import TextInputResult

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.slack import Slack
from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.builtin_plugins import slack as slack_plugin
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import PluginHost, SessionStart, TurnStart
from pydantic_clai2.plugins.loader import PluginLoader
from pydantic_clai2.plugins.menus import SettingsMenu
from pydantic_clai2.ui.menus import dialogs
from tests.clai2.dialog_script import Answer, ScriptedDialogs
from tests.clai2.menu_script import Script

BUILTIN = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'slack')
ENABLED = BUILTIN.model_copy(update={'enabled': True})
CONTEXT = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
CLOSE = MenuResult(cancelled=True)
ESC = TextInputResult(cancelled=True)


def is_slack(capability: object) -> TypeGuard[Slack[None]]:
    return isinstance(capability, Slack)


@dataclass
class Shell:
    plugins: PluginLoader[None]
    store: SettingsStore
    output: io.StringIO

    def slack(self) -> Slack[None]:
        [capability] = self.plugins.capabilities()
        assert is_slack(capability)
        assert capability.defer_loading
        return capability

    async def turn_token(self) -> str | None:
        """The token the Slack connection would use for a turn started now."""
        await self.plugins.fire(TurnStart(text='hi'))
        auth = self.slack().auth
        assert callable(auth)
        token = auth(CONTEXT)
        assert not isinstance(token, Awaitable)
        return token

    def saved(self) -> dict[str, JsonValue]:
        [declaration] = self.store.plugins()
        return declaration.settings

    def host(self) -> PluginHost[None]:
        [entry] = self.plugins.entries()
        assert entry.loaded is not None
        return entry.loaded.host

    def source(self) -> 'slack_plugin.SlackSource[None]':
        return slack_plugin.SlackSource(self.host())

    async def act(self, key: str, answers: Sequence[Answer]) -> tuple[list[str], ScriptedDialogs]:
        """Run a row's action as the web UI does, answering its dialogs in order; its messages and the dialogs."""
        scripted = ScriptedDialogs(answers)

        async def edit(menu: SettingsMenu) -> list[str]:
            return [*await menu.actions[key](scripted)]

        messages = await self.plugins.edit_settings('slack', edit)
        assert scripted.unused == []
        return messages, scripted


async def shell(declaration: PluginSettings = ENABLED, *, terminal: bool = True) -> Shell:
    store = SettingsStore()
    output = io.StringIO()
    plugins: PluginLoader[None] = PluginLoader(
        store=store,
        # A terminal opens the settings menu; no colours keep the output plain to assert on.
        console=Console(file=output, width=200, force_terminal=terminal, color_system=None),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=(declaration,),
    )
    await plugins.load_all()
    return Shell(plugins, store, output)


def script(
    monkeypatch: pytest.MonkeyPatch,
    lists: list[MenuResult],
    choices: list[MenuResult] | None = None,
    texts: list[TextInputResult] | None = None,
) -> Script:
    scripted = Script(lists=[*lists, CLOSE], choices=choices or [], texts=texts or [])
    monkeypatch.setattr(dialogs, 'RUNNERS', scripted.runners)
    return scripted
