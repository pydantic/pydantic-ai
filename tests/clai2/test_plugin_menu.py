"""The `/plugins` menu, driven headless."""

import asyncio
import io
import re
import sys
from collections.abc import Coroutine, Sequence
from pathlib import Path

import pytest
from rich.console import Console
from termflow.tui import MenuItem
from termflow.tui.keys import Key
from termflow.tui.menu import MenuResult

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import SessionStart
from pydantic_clai2.plugins.describe import describe
from pydantic_clai2.plugins.loader import TURN_NOTICE, PluginEntry, PluginLoader
from pydantic_clai2.ui.menus.field_menu import is_save_and_close
from pydantic_clai2.ui.menus.plugin_menu import Configure, PluginMenu, open_plugins_menu


def unstyled(text: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


PLUGIN = 'from pydantic_clai2.plugins import Plugin\nclass Quiet(Plugin):\n    pass\n'
TUNED = """
from pydantic import BaseModel
from pydantic_clai2.plugins import Plugin


class Settings(BaseModel):
    greeting: str = 'hi'


class Tuned(Plugin[Settings]):
    async def configure(self) -> str:
        host = self.host
        if host.name == 'grumpy':
            host.save_settings(Settings(greeting='grr'))
            raise ValueError('grumpy refuses to be configured')
        if self.settings.greeting != 'hi':
            return f'{host.name} already says {self.settings.greeting}.'
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


def test_details_show_the_docstring_summary(tmp_path: Path) -> None:
    loader = make_loader(tmp_path)
    (loader.plugins_dir / 'documented.py').write_text(f'"""Say `hello`\n  on start.\n\nInternals."""\n{PLUGIN}')
    menu = PluginMenu(loader, apply=run_now)
    item = menu.items()[0]
    assert unstyled(menu.details(item)).splitlines()[3:6] == [
        'Say hello on start.',
        '',
        f'source   {loader.plugins_dir / "documented.py"}',
    ]
    (loader.plugins_dir / 'documented.py').write_text(f'"""Say goodbye."""\n{PLUGIN}')
    assert 'Say hello on start.' in unstyled(menu.details(item)), 'read once, not on every repaint'
    menu.reload(FakeMenu(), item)
    assert 'Say goodbye.' in unstyled(menu.details(item))


def test_describe_reads_source_without_importing_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'clai_described.py').write_text(
        '"""The module."""\nraise RuntimeError("imported")\nclass Plain:\n    pass\nclass Tool:\n    """The tool."""\n'
    )
    (tmp_path / 'clai_broken_source.py').write_text('def (:\n')
    (tmp_path / 'clai_namespace').mkdir()
    monkeypatch.setattr(sys, 'path', [str(tmp_path), *sys.path])

    def text(factory: str, *, project: bool = False, path: Path | None = None) -> str:
        declaration = PluginSettings(id='x', factory=factory)
        return describe(PluginEntry[None](declaration=declaration, path=path, project=project))

    assert text('clai_described:Tool') == 'The tool.'
    assert text('clai_described:Plain') == 'The module.'
    assert text('clai_described:missing') == 'The module.'
    assert text('clai_described', project=True) == 'The module.'
    assert text('x', project=True, path=tmp_path / 'clai_described.py') == 'The module.'
    assert text('clai_broken_source') == ''
    assert text('clai_missing_package.plugin') == ''
    assert text('pydantic_clai2.no_such_module') == ''
    assert text('clai_namespace') == ''
    assert text('sys') == ''
    assert 'clai_described' not in sys.modules
    package = tmp_path / 'clai_parent_must_not_run'
    package.mkdir()
    (package / '__init__.py').write_text('raise RuntimeError("parent imported")')
    (package / 'tools.py').write_text('class Tool:\n    """A safe description."""\n')
    assert text('clai_parent_must_not_run.tools:Tool') == 'A safe description.'
    assert text('clai_parent_must_not_run.tools:Tool', project=True) == 'A safe description.'
    assert 'clai_parent_must_not_run' not in sys.modules
    injected = tmp_path / 'injected.py'
    injected.write_text('"""Before\\x1b]52;c;payload\\x07 after."""')
    assert text('x', path=injected) == 'Before]52;c;payload after.'


def test_remove_refreshes_the_restored_description(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / 'clai_description_cache'
    package.mkdir()
    (package / '__init__.py').write_text('')
    (package / 'custom.py').write_text('"""Custom tools."""')
    (package / 'stock.py').write_text('"""Stock tools."""')
    monkeypatch.setattr(sys, 'path', [str(tmp_path), *sys.path])
    store = SettingsStore(tmp_path / 'cache.db')
    store.save_plugin(PluginSettings(id='tools', factory='clai_description_cache.custom', enabled=False))
    loader: PluginLoader[None] = PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=(PluginSettings(id='tools', factory='clai_description_cache.stock', enabled=False),),
    )
    menu = PluginMenu(loader, apply=run_now)
    item = menu.items()[0]
    assert 'Custom tools.' in unstyled(menu.details(item))
    menu.remove(FakeMenu(), item)
    assert 'Stock tools.' in unstyled(menu.details(item))
    assert 'Custom tools.' not in unstyled(menu.details(item))


def test_rows_details_and_keys(tmp_path: Path) -> None:
    loader = make_loader(tmp_path, 'alpha', 'beta')
    menu = PluginMenu(loader, apply=run_now)
    labels = [item.label for item in menu.items()]
    assert labels[0].startswith('○ alpha') and labels[1].startswith('○ beta')
    fake = FakeMenu()
    alpha = menu.items()[0]
    menu.toggle(fake, alpha)
    assert fake.redraws[-1][0].label.startswith('● alpha')
    assert unstyled(fake.redraws[-1][0].description) == 'on     drop-in'
    assert unstyled(menu.details(alpha)).splitlines() == [
        'alpha',
        'on · drop-in',
        '',
        f'source   {loader.plugins_dir / "alpha.py"}',
        'provides nothing yet',
        'settings none',
    ]
    menu.reload(fake, alpha)
    assert fake.redraws[-1][0].label.startswith('● alpha')
    menu.toggle(fake, alpha)
    assert fake.redraws[-1][0].label.startswith('○ alpha')
    assert unstyled(menu.details(alpha)).splitlines()[1] == 'off · drop-in'
    assert 'settings turn on to see' in unstyled(menu.details(alpha))
    menu.remove(fake, alpha)
    assert unstyled(menu.details(alpha)).splitlines()[1] == 'off · drop-in'
    assert menu.details(MenuItem('stray', value=None)) == ''
    assert menu.details(MenuItem('typed', value=0)) == ''
    assert len(fake.redraws) == 4
    assert menu.close(fake, alpha).item is alpha
    assert menu.build() is not None


def run_keys(
    menu: PluginMenu[None], keys: Sequence[str], *, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> tuple[MenuResult, list[str]]:
    """Press `keys`; `frames[i]` is the screen shown when key `i` was read, kept when a key repaints nothing."""
    frames: list[str] = []
    inputs = iter(keys)

    def read_key() -> str:
        frames.append(unstyled(capsys.readouterr().out) or (frames[-1] if frames else ''))
        return next(inputs)

    monkeypatch.setattr('pydantic_clai2.ui.menus.plugin_menu.menu_key', read_key)
    return menu.build().run(), frames


FOOTER = '/ search · space on/off · c configure · r reload · d remove · esc close'
EMPTY = '(type to filter)'


@pytest.mark.parametrize(
    ('keys', 'query', 'name'),
    [
        pytest.param(['/', *'coder', Key.ENTER], 'coder', 'coDer', id='letters-filter-ignoring-case'),
        pytest.param(['/', *'quiet', Key.ENTER], 'quiet', 'quiet', id='q-filters-while-searching'),
        pytest.param(['/', *'codex', Key.BACKSPACE, 'r', Key.ENTER], 'coder', 'coDer', id='backspace-edits'),
        pytest.param(['/', 'e', Key.DOWN, Key.ENTER], 'e', 'coDer', id='arrows-move-between-matches'),
        pytest.param(
            ['/', *'zz', Key.ENTER, Key.BACKSPACE, Key.BACKSPACE, *'beta', Key.ENTER],
            'beta',
            'beta',
            id='enter-with-no-matches-keeps-searching',
        ),
        pytest.param(['/', *'bet', Key.ENTER, '/', 'a', Key.ENTER], 'beta', 'beta', id='slash-resumes-the-query'),
        pytest.param(['/', *'beta', Key.ENTER, Key.ESCAPE], EMPTY, 'alpha', id='esc-clears-the-kept-filter'),
        pytest.param(['/', *'zz', Key.ESCAPE], EMPTY, 'alpha', id='esc-leaves-search-with-no-matches'),
        pytest.param([Key.DOWN, '/', Key.ESCAPE], EMPTY, 'beta', id='esc-from-an-empty-search-keeps-the-cursor'),
        pytest.param(
            [Key.DOWN, '/', Key.BACKSPACE, Key.ESCAPE],
            EMPTY,
            'beta',
            id='backspace-in-an-empty-search-keeps-the-cursor',
        ),
        pytest.param(
            ['/', 'x', Key.BACKSPACE, Key.ENTER, Key.DOWN, '/', Key.ESCAPE],
            EMPTY,
            'beta',
            id='esc-after-an-emptied-search-keeps-the-cursor',
        ),
        pytest.param(['x', 'b', Key.BACKSPACE, Key.END], EMPTY, None, id='typing-outside-search-does-nothing'),
    ],
)
def test_search_and_navigation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    keys: list[str],
    query: str,
    name: str | None,
) -> None:
    """Enter ends a search keeping the matches; a second Enter picks the highlighted row."""
    loader = make_loader(tmp_path, 'alpha', 'beta', 'coDer', 'quiet')
    result, frames = run_keys(
        PluginMenu(loader, apply=run_now), [*keys, Key.ENTER], monkeypatch=monkeypatch, capsys=capsys
    )
    assert result.item is not None
    if name is None:
        assert is_save_and_close(result.item)
    else:
        assert result.item.value == name
    assert f'search: {query}' in frames[-1]
    assert FOOTER in frames[-1]
    if query != EMPTY:
        assert '○ alpha' not in frames[-1]
    assert all(entry.loaded is None for entry in loader.entries()), 'typing must never run a plugin action'


def test_hotkeys_act_on_the_kept_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    loader = make_loader(tmp_path, 'alpha', 'directory')
    result, frames = run_keys(
        PluginMenu(loader, apply=run_now),
        ['/', *'directory', Key.ENTER, ' ', 'r', ' ', 'd', 'q'],
        monkeypatch=monkeypatch,
        capsys=capsys,
    )
    assert result.item is not None and result.item.value == 'directory'
    for frame in frames[-5:]:
        assert 'search: directory' in frame and '○ alpha' not in frame
    assert '● directory' in frames[-4] and '● directory' in frames[-3]
    assert '○ directory' in frames[-2] and '○ directory' in frames[-1]
    assert all(entry.loaded is None for entry in loader.entries())


@pytest.mark.parametrize('keys', [['/', Key.ENTER], ['/', 'x', Key.BACKSPACE, Key.ENTER]])
def test_enter_on_an_empty_search_lets_one_esc_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], keys: list[str]
) -> None:
    """With nothing to keep, Enter returns to plain hotkeys, so a single Esc closes."""
    loader = make_loader(tmp_path, 'alpha')
    result, frames = run_keys(
        PluginMenu(loader, apply=run_now), [*keys, Key.ESCAPE], monkeypatch=monkeypatch, capsys=capsys
    )
    assert result.cancelled
    assert len(frames) == len(keys) + 1


@pytest.mark.parametrize('key', [' ', 'c'])
def test_configure_from_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], key: str
) -> None:
    loader = make_loader(tmp_path, 'alpha', tuned=('tuned',))
    if key == 'c':
        run_now(loader.enable('tuned'))
    result, _ = run_keys(
        PluginMenu(loader, apply=run_now), ['/', *'tuned', Key.ENTER, key], monkeypatch=monkeypatch, capsys=capsys
    )
    assert result.item is not None and result.item.value == Configure('tuned')


@pytest.mark.parametrize('keys', [['ctrl-c'], [Key.ESCAPE, Key.ESCAPE]])
@pytest.mark.parametrize('names', [(), ('alpha',)])
def test_no_matches_and_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    keys: list[str],
    names: tuple[str, ...],
) -> None:
    loader = make_loader(tmp_path, *names)
    result, frames = run_keys(
        PluginMenu(loader, apply=run_now), ['/', *'missing', Key.ENTER, *keys], monkeypatch=monkeypatch, capsys=capsys
    )
    assert result.cancelled and result.item is None
    assert '(no matches)' in frames[len(frames) - len(keys)]


@pytest.mark.parametrize('close', [[Key.ESCAPE, Key.ESCAPE], [Key.ESCAPE, 'q']])
def test_removing_the_last_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], close: list[str]
) -> None:
    loader = make_loader(tmp_path, 'alpha')
    store = SettingsStore(tmp_path / 'config.db')
    store.save_plugin(PluginSettings(id='installed', factory='clai_missing.plugin', enabled=False))
    result, frames = run_keys(
        PluginMenu(loader, apply=run_now),
        ['/', *'installed', Key.ENTER, 'd', *close],
        monkeypatch=monkeypatch,
        capsys=capsys,
    )
    if close[-1] == 'q':
        assert result.item is not None and result.item.value == 'alpha', 'Esc cleared the filter, q closes on a row'
    else:
        assert result.cancelled
    assert 'search: installed' in frames[-len(close)] and '(no matches)' in frames[-len(close)]
    assert [entry.name for entry in loader.entries()] == ['alpha']


async def test_details_say_that_a_running_turn_keeps_its_plugins(tmp_path: Path) -> None:
    loader = make_loader(tmp_path, 'alpha')
    menu = PluginMenu(loader, apply=run_now)
    alpha = menu.items()[0]
    assert TURN_NOTICE not in ' '.join(unstyled(menu.details(alpha)).split())
    async with loader.turn():
        assert ' '.join(unstyled(menu.details(alpha)).split()).endswith(TURN_NOTICE)


def test_errors_become_a_notice(tmp_path: Path) -> None:
    loader = make_loader(tmp_path, 'broken', 'fine')
    (loader.plugins_dir / 'broken.py').write_text('raise RuntimeError("nope")')
    menu = PluginMenu(loader, apply=run_now)
    fake = FakeMenu()
    broken, fine, _ = menu.items()
    menu.toggle(fake, broken)
    assert menu.notice is not None and 'RuntimeError: nope' in menu.notice
    details = unstyled(menu.details(broken))
    assert details.splitlines()[1] == 'failed · drop-in'
    assert '\nerror\n' in details and 'RuntimeError: nope' in details
    assert unstyled(menu.details(MenuItem('stray', value=None))) == menu.notice
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
    assert loader.entries()[0].loaded is not None
    listing = await loader.command(['list'])
    assert 'gamma:' in listing and '(enabled, loaded)' in listing


@pytest.mark.parametrize('names', [(), ('gamma',)])
async def test_open_menu_closes_quietly_without_changes(tmp_path: Path, names: tuple[str, ...]) -> None:
    loader = make_loader(tmp_path, *names)

    def run(menu: PluginMenu[None]) -> MenuResult:
        assert menu.items()
        return MenuResult(cancelled=True)

    assert await open_plugins_menu(loader, run=run) == ''
    assert all(entry.loaded is None for entry in loader.entries())


def test_save_and_close_is_the_last_row(tmp_path: Path) -> None:
    loader = make_loader(tmp_path, 'alpha')
    menu = PluginMenu(loader, apply=run_now)
    *_, last = menu.items()
    assert is_save_and_close(last)
    assert menu.details(last) == 'Leave this menu. Each change was saved as you made it.'
    menu.notice = 'Configured alpha.'
    assert unstyled(menu.details(last)) == 'Configured alpha.\nLeave this menu. Each change was saved as you made it.'
    assert menu.toggle(FakeMenu(), last) is None and menu.configure(FakeMenu(), last) is None


def test_enabling_hands_back_the_settings_menu_only_when_there_is_one(tmp_path: Path) -> None:
    loader = make_loader(tmp_path, 'plain', tuned=('tuned',))
    menu = PluginMenu(loader, apply=run_now)
    fake = FakeMenu()
    plain, tuned, _ = menu.items()
    assert menu.toggle(fake, plain) is None
    assert fake.redraws[-1][0].label.startswith('● plain')
    result = menu.toggle(fake, tuned)
    assert result is not None and result.item is not None and result.item.value == Configure('tuned')
    assert menu.toggle(fake, tuned) is None, 'disabling never opens a settings menu'
    assert fake.redraws[-1][1].label.startswith('○ tuned')


def test_rows_name_the_origin_and_status(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.save_plugin(PluginSettings(id='installed', factory='clai_missing.plugin', enabled=False))
    loader = PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=(PluginSettings(id='shipped', factory='clai_missing.shipped'),),
        project=(PluginSettings(id='filed', factory='clai_missing.filed', enabled=False),),
    )
    menu = PluginMenu(loader, apply=run_now)
    rows = {item.value: unstyled(item.description) for item in menu.items()[:-1]}
    assert rows == {'filed': 'off    project', 'installed': 'off    installed', 'shipped': 'idle   built-in'}
    assert menu.build() is not None


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
    assert 'settings press c to configure' in unstyled(menu.details(tuned))
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
    assert grumpy.loaded is not None, 'a failing settings menu leaves the plugin on'
    assert grumpy.declaration.settings == {'greeting': 'grr'}, 'and loaded with what it saved before failing'
    assert tuned.loaded is not None and tuned.declaration.settings == {'greeting': 'hello'}


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
