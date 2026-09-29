"""The `/plugins` menu, driven headless."""

import asyncio
import io
import sys
from collections.abc import Coroutine, Sequence
from pathlib import Path

import pytest
from rich.console import Console
from termflow.tui import MenuItem
from termflow.tui.menu import MenuResult

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_clai2.commands import Commands
from pydantic_clai2.field_menu import is_save_and_close
from pydantic_clai2.plugin_loader import PluginLoader
from pydantic_clai2.plugin_menu import Configure, PluginMenu, open_plugins_menu
from pydantic_clai2.plugins import SessionStart
from pydantic_clai2.settings_store import SettingsStore

PLUGIN = 'from pydantic_clai2.plugins import PluginHost\ndef activate(host: PluginHost) -> None:\n    pass\n'
TUNED = """
from pydantic import BaseModel
from pydantic_clai2.plugins import PluginHost


class Settings(BaseModel):
    greeting: str = 'hi'


def activate(host: PluginHost) -> None:
    settings = host.settings(Settings)

    @host.configure
    async def configure() -> str:
        if host.name == 'grumpy':
            host.save_settings(Settings(greeting='grr'))
            raise ValueError('grumpy refuses to be configured')
        if settings.greeting != 'hi':
            return f'{host.name} already says {settings.greeting}.'
        host.save_settings(Settings(greeting='hello'))
        return f'Configured {host.name}.'
"""


class FakeMenu:
    def __init__(self) -> None:
        self.redraws: list[Sequence[MenuItem]] = []

    def replace_items(self, items: Sequence[MenuItem]) -> None:
        self.redraws.append(items)


def make_loader(tmp_path: Path, *names: str, tuned: tuple[str, ...] = ()) -> PluginLoader[None]:
    store = SettingsStore(tmp_path / 'config.db')
    store.plugins_dir.mkdir()
    for name in names:
        (store.plugins_dir / f'{name}.py').write_text(PLUGIN)
    for name in tuned:
        (store.plugins_dir / f'{name}.py').write_text(TUNED)
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
    broken, fine, _ = menu.items()
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
    assert len(items) == 2 and items[0].disabled and str(loader.plugins_dir) in items[0].label
    assert is_save_and_close(items[1])


async def test_open_menu_applies_actions_from_the_menu_thread(tmp_path: Path) -> None:
    loader = make_loader(tmp_path, 'gamma')

    def run(menu: PluginMenu[None]) -> MenuResult:
        assert menu.toggle(FakeMenu(), menu.items()[0]) is None
        return MenuResult(item=menu.items()[-1])

    assert await open_plugins_menu(loader, run=run) == ''
    assert loader.entries()[0].host is not None
    listing = await loader.command(['list'])
    assert 'gamma:' in listing and '(enabled, loaded)' in listing


@pytest.mark.parametrize('names', [(), ('gamma',)])
async def test_open_menu_closes_quietly_without_changes(tmp_path: Path, names: tuple[str, ...]) -> None:
    loader = make_loader(tmp_path, *names)

    def run(menu: PluginMenu[None]) -> MenuResult:
        assert menu.items()
        return MenuResult(cancelled=True)

    assert await open_plugins_menu(loader, run=run) == ''
    assert all(entry.host is None for entry in loader.entries())


def test_save_and_close_is_the_last_row(tmp_path: Path) -> None:
    loader = make_loader(tmp_path, 'alpha')
    menu = PluginMenu(loader, apply=run_now)
    *_, last = menu.items()
    assert is_save_and_close(last)
    assert menu.details(last) == 'Leave this menu. Each change was saved as you made it.'
    menu.notice = 'Configured alpha.'
    assert menu.details(last) == 'Configured alpha.\nLeave this menu. Each change was saved as you made it.'
    assert menu.toggle(FakeMenu(), last) is None and menu.configure(FakeMenu(), last) is None


def test_enabling_hands_back_the_settings_menu_only_when_there_is_one(tmp_path: Path) -> None:
    loader = make_loader(tmp_path, 'plain', tuned=('tuned',))
    menu = PluginMenu(loader, apply=run_now)
    fake = FakeMenu()
    plain, tuned, _ = menu.items()
    assert menu.toggle(fake, plain) is None
    assert fake.redraws[-1][0].label.startswith('[x] plain')
    result = menu.toggle(fake, tuned)
    assert result is not None and result.item is not None and result.item.value == Configure('tuned')
    assert menu.toggle(fake, tuned) is None, 'disabling never opens a settings menu'
    assert fake.redraws[-1][1].label.startswith('[ ] tuned')


def test_configure_key(tmp_path: Path) -> None:
    loader = make_loader(tmp_path, 'plain', tuned=('tuned',))
    menu = PluginMenu(loader, apply=run_now)
    fake = FakeMenu()
    plain, tuned, _ = menu.items()
    assert menu.configure(fake, tuned) is None
    assert menu.notice == 'Enable tuned to configure it.'
    menu.toggle(fake, plain)
    assert menu.configure(fake, plain) is None
    assert menu.notice == 'plain has no settings menu.'
    menu.toggle(fake, tuned)
    result = menu.configure(fake, tuned)
    assert result is not None and result.item is not None and result.item.value == Configure('tuned')


async def test_enabling_from_the_menu_opens_the_settings_menu_then_returns(tmp_path: Path) -> None:
    loader = make_loader(tmp_path, tuned=('grumpy', 'tuned'))
    notices: list[str | None] = []

    def run(menu: PluginMenu[None]) -> MenuResult:
        notices.append(menu.notice)
        grumpy, tuned, save_and_close = menu.items()
        steps = [
            lambda: menu.toggle(FakeMenu(), tuned),
            lambda: menu.configure(FakeMenu(), tuned),
            lambda: menu.toggle(FakeMenu(), grumpy),
            lambda: MenuResult(item=save_and_close),
        ]
        result = steps[len(notices) - 1]()
        assert result is not None
        return result

    expected = ['Configured tuned.', 'tuned already says hello.', 'grumpy refuses to be configured']
    assert await open_plugins_menu(loader, run=run) == '\n'.join(expected)
    assert notices == [None, *expected]
    grumpy, tuned = loader.entries()
    assert grumpy.host is not None, 'a failing settings menu leaves the plugin on'
    assert grumpy.declaration.settings == {'greeting': 'grr'}, 'and loaded with what it saved before failing'
    assert tuned.host is not None and tuned.declaration.settings == {'greeting': 'hello'}


async def test_enable_and_add_commands_open_the_settings_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loader = make_loader(tmp_path, 'plain', tuned=('tuned',))
    assert await loader.command(['enable', 'plain']) == 'Enabled plain.'
    assert await loader.command(['enable', 'tuned']) == 'Enabled tuned.\nConfigured tuned.'
    assert await loader.command(['enable', 'tuned']) == 'Enabled tuned.', 'already on: no menu'
    assert await loader.command(['disable', 'tuned']) == 'Disabled tuned.'
    with pytest.raises(ValueError, match='tuned is not loaded; enable it before configuring'):
        await loader.command(['configure', 'tuned'])
    with pytest.raises(ValueError, match='plain has no settings menu'):
        await loader.command(['configure', 'plain'])
    module = tmp_path / 'modules'
    module.mkdir()
    (module / 'tuned_module.py').write_text(TUNED)
    monkeypatch.setattr(sys, 'path', [str(module), *sys.path])
    assert await loader.command(['add', 'added', 'tuned_module']) == 'Added and loaded added.\nConfigured added.'
