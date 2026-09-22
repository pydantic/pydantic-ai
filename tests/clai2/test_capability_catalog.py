"""Harness catalog completeness and opt-in behavior through the real plugin menu."""

import ast
import asyncio
import io
from collections.abc import Coroutine, Sequence
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from rich.console import Console
from termflow.tui import MenuItem  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.capability_catalog import HARNESS_PLUGINS
from pydantic_clai2.commands import Commands
from pydantic_clai2.plugin_loader import PluginLoader
from pydantic_clai2.plugin_menu import PluginMenu
from pydantic_clai2.plugins import SessionStart
from pydantic_clai2.settings_store import SettingsStore


def test_catalog_covers_public_harness_capabilities() -> None:
    root = Path(__file__).parents[2] / 'pydantic_ai_harness'
    bases = {'AbstractCapability', 'CombinedCapability', 'NativeOrLocalTool', 'BaseDurabilityCapability'}
    expected: set[str] = set()
    for path in root.rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef) and any(ast.unparse(base).split('[')[0] in bases for base in node.bases):
                module = path.parent.relative_to(root).as_posix().replace('/', '.')
                exports = ast.parse((path.parent / '__init__.py').read_text())
                assert any(
                    isinstance(statement, ast.ImportFrom) and any(alias.name == node.name for alias in statement.names)
                    for statement in exports.body
                )
                expected.add(f'pydantic_ai_harness.{module}:{node.name}')
    integrated = {
        'pydantic_ai_harness.coder:Coder',
        'pydantic_ai_harness.ask_user:AskUser',
        'pydantic_ai_harness.repo_context:RepoContext',
    }
    assert {plugin.factory for plugin in HARNESS_PLUGINS} | integrated == expected
    assert all(not plugin.enabled for plugin in HARNESS_PLUGINS)
    assert len({plugin.id for plugin in DEFAULT_PLUGINS}) == len(DEFAULT_PLUGINS)
    assert {plugin.id for plugin in DEFAULT_PLUGINS if plugin.enabled} == {
        'coder',
        'ask_user',
        'repo_context',
        'compaction',
        'persistence',
        'logfire',
        'notifications',
        'mcp',
    }


class Menu:
    def replace_items(self, items: Sequence[MenuItem]) -> None:
        self.items = items


def test_catalog_menu_toggle_persistence_reset_and_errors(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'settings.db')

    def loader() -> PluginLoader[None]:
        return PluginLoader(
            store=store,
            console=Console(file=io.StringIO()),
            commands=Commands(),
            session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
            builtin=HARNESS_PLUGINS,
        )

    def apply(action: Coroutine[object, object, object]) -> None:
        asyncio.run(action)

    plugins = loader()
    asyncio.run(plugins.load_all())
    assert plugins.capabilities() == []
    assert store.plugins() == []
    menu = PluginMenu(plugins, apply=apply)
    rows = {item.value: item for item in menu.items()}
    assert len(rows) == len(HARNESS_PLUGINS)
    assert all(item.label.startswith('[ ]') for item in rows.values())
    redraw = Menu()
    item = rows['tool_output_limits']
    menu.toggle(redraw, item)
    assert menu.notice is None
    assert len(plugins.capabilities()) == 1
    assert 'enabled, loaded' in menu.details(item)
    fresh = loader()
    asyncio.run(fresh.load_all())
    assert len(fresh.capabilities()) == 1
    menu.toggle(redraw, item)
    assert 'state   disabled' in menu.details(item)
    menu.remove(redraw, item)
    assert store.plugins() == []
    assert 'state   disabled' in menu.details(item)
    assert len(menu.items()) == len(HARNESS_PLUGINS)
    menu.toggle(redraw, rows['input_guardrail'])
    assert menu.notice is not None
    assert 'error   none' not in menu.details(rows['input_guardrail'])
    assert plugins.capabilities() == []
    asyncio.run(fresh.close('exit'))
