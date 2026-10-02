"""`/plugins` offers only the curated built-ins; other capabilities are added on purpose."""

import asyncio
import io
from collections.abc import Coroutine, Sequence
from pathlib import Path

import pytest
from rich.console import Console
from termflow.tui import MenuItem

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import SessionStart
from pydantic_clai2.plugins.loader import PluginLoader
from pydantic_clai2.ui.menus.plugin_menu import Configure, PluginMenu

CURATED = {
    'coder',
    'ask_user',
    'repo_context',
    'compaction',
    'persistence',
    'observability',
    'notifications',
    'herdr',
    'mcp',
    'day_ai',
    'ordinal',
    'github',
    'google_workspace',
    'pylon',
    'notion',
    'slack',
    'logfire_mcp',
    'posthog',
    'grain',
    'linear',
}
OPT_IN = {
    'herdr',
    'day_ai',
    'github',
    'google_workspace',
    'grain',
    'linear',
    'logfire_mcp',
    'notion',
    'ordinal',
    'posthog',
    'pylon',
    'slack',
}


class Menu:
    def replace_items(self, items: Sequence[MenuItem]) -> None:
        self.items = items


def _loader(
    store: SettingsStore, builtin: Sequence[PluginSettings], *, project: Sequence[PluginSettings] = ()
) -> PluginLoader[None]:
    return PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=builtin,
        project=project,
    )


def _apply(action: Coroutine[object, object, object]) -> None:
    asyncio.run(action)


def test_builtins_are_the_curated_set_with_opt_in_integrations_off() -> None:
    assert sorted(plugin.id for plugin in DEFAULT_PLUGINS) == sorted(CURATED)
    assert {plugin.id for plugin in DEFAULT_PLUGINS if not plugin.enabled} == OPT_IN


def test_menu_offers_no_uncurated_harness_capabilities(tmp_path: Path) -> None:
    menu = PluginMenu(_loader(SettingsStore(tmp_path / 'settings.db'), DEFAULT_PLUGINS), apply=_apply)
    *rows, _save_and_close = menu.items()
    assert {item.value for item in rows} == CURATED | OPT_IN


def test_saved_logfire_opens_observability_setup_when_enabled(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    store.save_plugin(
        PluginSettings(
            id='logfire',
            factory='pydantic_clai2.builtin_plugins.logfire',
            enabled=False,
            settings={'send_to_logfire': False, 'include_content': False},
        )
    )
    builtin = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'observability')
    plugins = _loader(store, (builtin,))
    menu = PluginMenu(plugins, apply=_apply)
    item, _save_and_close = menu.items()
    assert item.value == 'observability' and '○ observability' in item.label
    assert plugins.entries()[0].state == 'disabled'
    try:
        result = menu.toggle(Menu(), item)
        assert result is not None and result.item is not None
        assert result.item.value == Configure('observability')
        assert len(plugins.entries()) == 1 and len(plugins.capabilities()) == 1
        (saved,) = store.plugins()
        assert saved.id == 'observability' and saved.enabled
        assert saved.settings == {'send_to_logfire': False, 'include_content': False}
        assert menu.toggle(Menu(), item) is None
        assert not store.plugins()[0].enabled
    finally:
        asyncio.run(plugins.close('exit'))


@pytest.mark.parametrize('source', ['project', 'folder', 'builtin'])
async def test_legacy_logfire_sources_share_one_runtime_identity(tmp_path: Path, source: str) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    legacy = PluginSettings(
        id='logfire',
        factory='pydantic_clai2.builtin_plugins.logfire',
        enabled=False,
        settings={'send_to_logfire': False, 'include_content': False},
    )
    project: tuple[PluginSettings, ...] = (legacy,)
    if source == 'folder':
        store.plugins_dir.mkdir()
        path = store.plugins_dir / 'logfire.py'
        path.write_text(
            'from pydantic_clai2.builtin_plugins.logfire import LogfirePlugin\n\n\n'
            'class Observability(LogfirePlugin):\n'
            '    pass\n'
        )
        store.save_plugin(legacy.model_copy(update={'path': str(path)}))
        project = ()
    builtin = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'observability')
    if source == 'builtin':
        builtin = legacy.model_copy(update={'enabled': True})
        store.save_plugin(legacy)
        project = ()
    plugins = _loader(store, (builtin,), project=project)
    try:
        await plugins.load_all()
        assert [entry.name for entry in plugins.entries()] == ['observability']
        assert not plugins.capabilities()
        await plugins.enable('observability')
        assert len(plugins.capabilities()) == 1
        await plugins.disable('observability')
        assert not plugins.capabilities()
        assert not store.plugins()[0].enabled
    finally:
        await plugins.close('exit')


async def test_legacy_logfire_add_replaces_observability_without_duplicate_hosts(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    builtin = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'observability')
    plugins = _loader(store, (builtin.model_copy(update={'settings': {'send_to_logfire': False}}),))
    try:
        await plugins.load_all()
        message = await plugins.command(['add', 'logfire', 'pydantic_ai_harness.tool_output_limits:ToolOutputLimits'])
        assert message == 'Replaced built-in observability.'
        await plugins.load_all()
        assert [entry.name for entry in plugins.entries()] == ['observability']
        assert len(plugins.capabilities()) == 1
        assert await plugins.command(['disable', 'logfire']) == 'Disabled observability.'
        assert not plugins.capabilities()
    finally:
        await plugins.close('exit')


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
    item, _save_and_close = menu.items()
    assert 'built-in' not in menu.details(item)
    assert 'installed' in item.description and 'on' in item.description
    menu.remove(Menu(), item)
    assert store.plugins() == []
    assert plugins.entries() == []
    assert plugins.capabilities() == []
