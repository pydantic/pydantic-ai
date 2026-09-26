"""`/plugins` offers only the curated built-ins; other capabilities are added on purpose."""

import asyncio
import io
from collections.abc import Coroutine, Sequence
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from rich.console import Console
from termflow.tui import MenuItem  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.plugin_loader import PluginLoader
from pydantic_clai2.plugin_menu import PluginMenu
from pydantic_clai2.plugins import SessionStart
from pydantic_clai2.settings_store import SettingsStore

CURATED = {'coder', 'ask_user', 'repo_context', 'compaction', 'persistence', 'logfire', 'notifications', 'mcp'}


class Menu:
    def replace_items(self, items: Sequence[MenuItem]) -> None:
        self.items = items


def _loader(store: SettingsStore, builtin: Sequence[PluginSettings]) -> PluginLoader[None]:
    return PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=builtin,
    )


def _apply(action: Coroutine[object, object, object]) -> None:
    asyncio.run(action)


def test_builtins_are_the_curated_enabled_set() -> None:
    assert sorted(plugin.id for plugin in DEFAULT_PLUGINS) == sorted(CURATED)
    assert all(plugin.enabled for plugin in DEFAULT_PLUGINS)


def test_menu_offers_no_uncurated_harness_capabilities(tmp_path: Path) -> None:
    menu = PluginMenu(_loader(SettingsStore(tmp_path / 'settings.db'), DEFAULT_PLUGINS), apply=_apply)
    assert {item.value for item in menu.items()} == CURATED


def test_capability_saved_from_the_old_catalog_still_loads(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    store.save_plugin(
        PluginSettings(
            id='tool_output_limits',
            factory='pydantic_ai_harness.tool_output_limits:ToolOutputLimits',
            enabled=True,
        )
    )
    plugins = _loader(store, ())
    asyncio.run(plugins.load_all())
    assert len(plugins.capabilities()) == 1
    menu = PluginMenu(plugins, apply=_apply)
    [item] = menu.items()
    assert '(built-in)' not in menu.details(item)
    assert 'enabled, loaded' in menu.details(item)
    menu.remove(Menu(), item)
    assert store.plugins() == []
    assert plugins.entries() == []
    assert plugins.capabilities() == []
