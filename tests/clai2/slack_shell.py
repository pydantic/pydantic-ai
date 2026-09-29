"""A CLAI shell with only the `slack` plugin loaded, and scripted settings-menu widgets."""

import io
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
from pydantic_clai2 import DEFAULT_PLUGINS, slack as slack_plugin
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.plugin_loader import PluginLoader
from pydantic_clai2.plugins import PluginHost, SessionStart, TurnStart
from pydantic_clai2.settings_store import SettingsStore
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
        return capability

    async def turn_token(self) -> str | None:
        """The token the Slack connection would use for a turn started now."""
        await self.plugins.fire(TurnStart(text='hi'))
        auth = self.slack().auth
        assert callable(auth)
        return auth(CONTEXT)

    def saved(self) -> dict[str, JsonValue]:
        [declaration] = self.store.plugins()
        return declaration.settings

    def host(self) -> PluginHost[None]:
        [entry] = self.plugins.entries()
        assert entry.host is not None
        return entry.host

    def source(self) -> 'slack_plugin.SlackSource[None]':
        return slack_plugin.SlackSource(self.host())


async def shell(declaration: PluginSettings = ENABLED) -> Shell:
    store = SettingsStore()
    output = io.StringIO()
    plugins: PluginLoader[None] = PluginLoader(
        store=store,
        console=Console(file=output, width=200),
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
    monkeypatch.setattr(slack_plugin, 'RUNNERS', scripted.runners)
    return scripted
