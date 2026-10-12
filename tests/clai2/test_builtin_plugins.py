"""`/plugins` offers only the curated built-ins; other capabilities are added on purpose."""

import asyncio
import io
from collections.abc import Coroutine, Sequence
from pathlib import Path

import pytest
from rich.console import Console
from termflow.tui import MenuItem

from pydantic_ai import Agent
from pydantic_ai.capabilities import AgentCapability, Instrumentation
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.compaction import FallbackCompaction, SlidingWindowCompaction
from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.builtin_plugins.coder import CoderPlugin, CoderSettings
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import SessionStart
from pydantic_clai2.plugins.loader import PluginError, PluginLoader
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


async def test_coder_keeps_the_plugins_it_includes_off(tmp_path: Path) -> None:
    """`coder` includes delegation, so `subagents` greys out; `compaction` runs alongside it."""
    store = SettingsStore(tmp_path / 'settings.db')
    # A row saved from the former harness catalog, under the id it offered.
    store.save_plugin(PluginSettings(id='subagents', factory='pydantic_ai_harness.subagents:SubAgents', enabled=False))
    coder, compaction = (
        next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == name) for name in ('coder', 'compaction')
    )
    delegating = coder.model_copy(update={'settings': {**coder.settings, 'sub_agents': True}})
    plugins = _loader(store, (delegating, compaction))
    menu = PluginMenu(plugins, apply=_apply)
    try:
        await plugins.load_all()
        assert {entry.name: entry.state for entry in plugins.entries()} == {
            'coder': 'enabled, loaded',
            'compaction': 'enabled, loaded',
            'subagents': 'included in coder',
        }
        _coder, compaction_row, greyed_subagents, _save_and_close = menu.items()
        assert not compaction_row.disabled and greyed_subagents.disabled
        assert greyed_subagents.label.split() == ['○', 'subagents', 'off', 'in', 'coder']
        with pytest.raises(ValueError, match='subagents is included in coder; disable coder to use it'):
            await plugins.enable('subagents')
        assert not store.plugins()[0].enabled, 'a refused enable saves nothing'
        with pytest.raises(ValueError, match='subagents is included in coder; disable coder to use it'):
            await plugins.configure('subagents')

        await plugins.disable('coder')
        states = {entry.name: entry.state for entry in plugins.entries()}
        assert states == {'coder': 'disabled', 'compaction': 'enabled, loaded', 'subagents': 'disabled'}
        await plugins.enable('coder')
        assert plugins.entries()[2].state == 'included in coder'
        assert (await plugins.remove('coder')).startswith('coder is built in')
        assert plugins.entries()[2].state == 'included in coder'
        assert await plugins.remove('subagents') == 'Removed subagents.'
    finally:
        await plugins.close('exit')


async def test_compaction_is_on_beside_coder_with_one_chain(tmp_path: Path) -> None:
    """`coder` binds no history compaction, so the built-in chain and `/compact` are on, once."""
    store = SettingsStore(tmp_path / 'settings.db')
    coder, compaction = (
        next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == name) for name in ('coder', 'compaction')
    )
    assert compaction.enabled
    commands = Commands()
    plugins = PluginLoader[None](
        store=store,
        console=Console(file=io.StringIO()),
        commands=commands,
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=(coder, compaction),
    )

    def chains() -> int:
        return sum(isinstance(capability, FallbackCompaction) for capability in plugins.capabilities())

    try:
        await plugins.load_all()
        assert [entry.state for entry in plugins.entries()] == ['enabled, loaded', 'enabled, loaded']
        assert chains() == 1 and 'compact' in commands
        # Enabling or loading it again is a no-op: no second chain, no duplicate `/compact`.
        await plugins.enable('compaction')
        await plugins.load_all()
        assert chains() == 1 and 'compact' in commands
        assert store.plugins()[0].enabled
    finally:
        await plugins.close('exit')


async def test_coder_binding_history_compaction_keeps_compaction_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `coder` that compacts history itself includes `compaction`, so two chains never run."""
    store = SettingsStore(tmp_path / 'settings.db')
    coder, compaction = (
        next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == name) for name in ('coder', 'compaction')
    )
    get_capabilities = CoderPlugin.get_capabilities

    def compacting(self: CoderPlugin[None]) -> Sequence[AgentCapability[None]]:
        return (*get_capabilities(self), SlidingWindowCompaction[None](max_messages=40))

    monkeypatch.setattr(CoderPlugin, 'get_capabilities', compacting)
    commands = Commands()
    plugins = PluginLoader[None](
        store=store,
        console=Console(file=io.StringIO()),
        commands=commands,
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=(coder, compaction),
    )
    try:
        await plugins.load_all()
        assert plugins.entries()[1].state == 'included in coder' and 'compact' not in commands
        with pytest.raises(ValueError, match='compaction is included in coder; disable coder to use it'):
            await plugins.enable('compaction')
        with pytest.raises(ValueError, match='compaction is included in coder'):
            await plugins.command(['add', 'compaction', 'pydantic_clai2.builtin_plugins.compaction'])
        with pytest.raises(ValueError, match='compaction is included in coder; disable coder to use it'):
            await plugins.configure('compaction')
        assert store.plugins() == [], 'a refused enable or add saves nothing'
        await plugins.disable('coder')
        assert plugins.entries()[1].state == 'enabled, loaded' and 'compact' in commands
        await plugins.enable('coder')
        assert plugins.entries()[1].state == 'included in coder' and 'compact' not in commands
    finally:
        await plugins.close('exit')


async def test_coder_without_sub_agents_leaves_subagents_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`coder` binds no `SubAgents` with `sub_agents` off, so a separate one still delegates."""
    store = SettingsStore(tmp_path / 'settings.db')
    store.save_plugin(PluginSettings(id='subagents', factory='pydantic_ai_harness.subagents:SubAgents'))
    coder = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'coder')
    plugins = _loader(store, (coder.model_copy(update={'settings': {**coder.settings, 'sub_agents': False}}),))

    def configure_sub_agents(enabled: bool) -> None:
        loaded = plugins.entries()[0].loaded
        assert loaded is not None

        async def save() -> str:
            loaded.host.save_settings(CoderSettings(sub_agents=enabled))
            return 'saved'

        monkeypatch.setattr(loaded.plugin, 'configure', save)

    try:
        await plugins.load_all()
        assert [entry.state for entry in plugins.entries()] == ['enabled, loaded', 'enabled, loaded']

        configure_sub_agents(True)
        await plugins.configure('coder')
        assert plugins.entries()[1].state == 'included in coder'

        configure_sub_agents(False)
        await plugins.configure('coder')
        assert plugins.entries()[1].state == 'enabled, loaded'
    finally:
        await plugins.close('exit')


async def test_failed_coder_reload_brings_back_what_it_included(tmp_path: Path) -> None:
    """A `coder` that no longer loads includes nothing, so a separate `SubAgents` runs again."""
    store = SettingsStore(tmp_path / 'settings.db')
    store.save_plugin(PluginSettings(id='subagents', factory='pydantic_ai_harness.subagents:SubAgents'))
    coder = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'coder')
    delegating = coder.model_copy(update={'settings': {**coder.settings, 'sub_agents': True}})
    plugins = _loader(store, (delegating,))
    try:
        await plugins.load_all()
        assert plugins.entries()[1].state == 'included in coder'
        store.save_plugin(coder.model_copy(update={'settings': {'sub_agents': 'yes'}}))
        with pytest.raises(PluginError):
            await plugins.reload('coder')
        assert [entry.loaded is not None for entry in plugins.entries()] == [False, True]
    finally:
        await plugins.close('exit')


async def test_disabling_coder_loads_the_rest_when_one_it_included_fails(tmp_path: Path) -> None:
    """A released plugin that fails to load is reported without stopping the others."""
    store = SettingsStore(tmp_path / 'settings.db')
    factory = 'pydantic_ai_harness.subagents:SubAgents'
    store.save_plugin(PluginSettings(id='broken_subagents', factory=factory, settings={'nonsense': True}))
    store.save_plugin(PluginSettings(id='subagents', factory=factory))
    coder = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'coder')
    delegating = coder.model_copy(update={'settings': {**coder.settings, 'sub_agents': True}})
    plugins = _loader(store, (delegating,))
    try:
        await plugins.load_all()
        await plugins.disable('coder')
        states = {entry.name: entry.state for entry in plugins.entries()}
        assert states['broken_subagents'].startswith('enabled, failed:')
        assert (states['coder'], states['subagents']) == ('disabled', 'enabled, loaded')
    finally:
        await plugins.close('exit')


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
        assert len(plugins.entries()) == 1
        assert sum(isinstance(capability, Instrumentation) for capability in plugins.capabilities()) == 1
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
        assert sum(isinstance(capability, Instrumentation) for capability in plugins.capabilities()) == 1
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
