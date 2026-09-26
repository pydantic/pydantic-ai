"""The `/plugins` menu, driven headless."""

import asyncio
import io
from collections.abc import Coroutine, Sequence
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from rich.console import Console
from termflow.tui import MenuItem  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2.commands import Commands
from pydantic_clai2.plugin_loader import PluginLoader
from pydantic_clai2.plugin_menu import PluginMenu, open_plugins_menu
from pydantic_clai2.plugins import SessionStart
from pydantic_clai2.settings_store import SettingsStore

PLUGIN = 'from pydantic_clai2.plugins import PluginHost\ndef activate(host: PluginHost) -> None:\n    pass\n'


class FakeMenu:
    def __init__(self) -> None:
        self.redraws: list[Sequence[MenuItem]] = []

    def replace_items(self, items: Sequence[MenuItem]) -> None:
        self.redraws.append(items)


def make_loader(tmp_path: Path, *names: str) -> PluginLoader[None]:
    store = SettingsStore(tmp_path / 'config.db')
    store.plugins_dir.mkdir()
    for name in names:
        (store.plugins_dir / f'{name}.py').write_text(PLUGIN)
    return PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
    )


def run_now(action: Coroutine[object, object, object]) -> None:
    asyncio.run(action)


def test_rows_details_and_keys(tmp_path: Path) -> None:
    loader = make_loader(tmp_path, 'alpha', 'beta')
    menu = PluginMenu(loader, apply=run_now)
    labels = [item.label for item in menu.items()]
    assert labels[0].startswith('[ ] alpha') and labels[1].startswith('[ ] beta')
    fake = FakeMenu()
    alpha = menu.items()[0]
    menu.toggle(fake, alpha)
    assert fake.redraws[-1][0].label.startswith('[x] alpha')
    assert 'state   enabled, loaded' in menu.details(alpha)
    assert 'adds    0 commands' in menu.details(alpha)
    menu.reload(fake, alpha)
    assert fake.redraws[-1][0].label.startswith('[x] alpha')
    menu.toggle(fake, alpha)
    assert fake.redraws[-1][0].label.startswith('[ ] alpha')
    assert 'state   disabled' in menu.details(alpha)
    menu.remove(fake, alpha)
    assert 'state   disabled' in menu.details(alpha)
    assert menu.details(MenuItem('stray', value=None)) == ''
    assert menu.details(MenuItem('typed', value=0)) == ''
    assert len(fake.redraws) == 4
    assert menu.close(fake, alpha).item is alpha
    assert menu.build() is not None


def test_errors_become_a_notice(tmp_path: Path) -> None:
    loader = make_loader(tmp_path, 'broken', 'fine')
    (loader.plugins_dir / 'broken.py').write_text('raise RuntimeError("nope")')
    menu = PluginMenu(loader, apply=run_now)
    fake = FakeMenu()
    broken, fine = menu.items()
    menu.toggle(fake, broken)
    assert menu.notice is not None and 'RuntimeError: nope' in menu.notice
    assert 'notice  ' in menu.details(broken) and 'error   RuntimeError: nope' in menu.details(broken)
    assert menu.details(MenuItem('stray', value=None)) == menu.notice
    menu.toggle(fake, fine)
    assert menu.notice is None
    for handler in (menu.toggle, menu.reload, menu.remove):
        handler(fake, MenuItem('stray', value=None))
    assert len(fake.redraws) == 5


def test_empty_state(tmp_path: Path) -> None:
    loader = make_loader(tmp_path)
    items = PluginMenu(loader, apply=run_now).items()
    assert len(items) == 1 and items[0].disabled and str(loader.plugins_dir) in items[0].label


async def test_open_menu_applies_actions_from_the_menu_thread(tmp_path: Path) -> None:
    loader = make_loader(tmp_path, 'gamma')

    def run(menu: PluginMenu[None]) -> None:
        menu.toggle(FakeMenu(), menu.items()[0])

    assert await open_plugins_menu(loader, run=run) == ''
    assert loader.entries()[0].host is not None
    listing = await loader.command(['list'])
    assert 'gamma:' in listing and '(enabled, loaded)' in listing


@pytest.mark.parametrize('names', [(), ('gamma',)])
async def test_open_menu_closes_quietly_without_changes(tmp_path: Path, names: tuple[str, ...]) -> None:
    loader = make_loader(tmp_path, *names)

    def run(menu: PluginMenu[None]) -> None:
        assert menu.items()

    assert await open_plugins_menu(loader, run=run) == ''
    assert all(entry.host is None for entry in loader.entries())


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'
