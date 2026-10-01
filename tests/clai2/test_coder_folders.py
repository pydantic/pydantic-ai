"""Drive Coder's folder settings through real widgets and temporary directories."""

import io
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import JsonValue
from rich.console import Console
from rich.text import Text
from termflow.tui import MenuItem
from termflow.tui.menu import Menu, MenuResult
from termflow.tui.textinput import TextInput, TextInputResult

from pydantic_clai2.builtin_plugins.coder import CoderSettings, CoderSource
from pydantic_clai2.builtin_plugins.coder_folders import DirectoryPicker, FolderAction, FolderMenu, run_coder_flow
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import PluginHost
from pydantic_clai2.ui.menus.field_menu import Runners, save_and_close_item
from tests.clai2.menu_script import Script, pick, typed


@pytest.fixture
def folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FolderMenu[object]:
    monkeypatch.chdir(tmp_path)
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    host = PluginHost[object](
        name='coder', console=Console(file=io.StringIO()), settings={'instructions': 'Keep this.', 'sub_agents': True}
    )
    return FolderMenu(CoderSource(host), project=tmp_path)


class Keyboard:
    """Exercise widget validation, key bindings and rendering without a terminal owner."""

    def __init__(self, keys: list[str], *, width: int = 80) -> None:
        self.keys = iter(keys)
        self.width = width
        self.output = io.StringIO()

    def prepare(self, widget: Menu | TextInput) -> None:
        widget._use_alt_screen = False  # pyright: ignore[reportPrivateUsage]
        widget._read_key = lambda: next(self.keys)  # pyright: ignore[reportPrivateUsage]
        widget._output = self.output  # pyright: ignore[reportPrivateUsage]
        widget._size = lambda: (self.width, 30)  # pyright: ignore[reportPrivateUsage]

    def menu(self, menu: Menu) -> MenuResult:
        self.prepare(menu)
        return menu.run()

    def text(self, widget: TextInput) -> TextInputResult:
        self.prepare(widget)
        return widget.run()

    @property
    def runners(self) -> Runners:
        return Runners(run_list=self.menu, run_choice=self.menu, run_text=self.text)


def render_frame(menu: Menu) -> list[str]:
    output = io.StringIO()
    menu._use_alt_screen = False  # pyright: ignore[reportPrivateUsage]
    menu._read_key = lambda: 'escape'  # pyright: ignore[reportPrivateUsage]
    menu._output = output  # pyright: ignore[reportPrivateUsage]
    assert menu.run().cancelled
    return Text.from_ansi(output.getvalue()).plain.splitlines()


def test_configure_flow_preserves_order_preferences_and_updates_summary(folders: FolderMenu[object]) -> None:
    for name in ('first', 'replacement'):
        (folders.project / name).mkdir()
    script = Script(
        lists=[
            pick('agent_folders'),
            pick(FolderAction(kind='name')),
            pick(FolderAction(kind='path')),
            pick(1),
            MenuResult(item=save_and_close_item()),
            MenuResult(item=save_and_close_item()),
        ],
        choices=[pick(FolderAction(kind='edit', index=1))],
        texts=[typed('agents'), typed('first'), typed('./replacement')],
    )
    assert run_coder_flow(folders.source, runners=script.runners) == ['Agent folder saved.'] * 3
    assert folders.source.host.settings(CoderSettings).model_dump(mode='json') == {
        'instructions': 'Keep this.',
        'sub_agents': True,
        'agent_folders': ['agents', './replacement'],
    }
    row = folders.source.rows()[1]
    assert row.display(folders.source.current(row)) == '2 selected'


@pytest.mark.parametrize('width', [50, 80, 140])
def test_keyboard_add_edit_remove_and_empty_state(folders: FolderMenu[object], width: int) -> None:
    (folders.project / 'agents').mkdir()
    (folders.project / 'other agents').mkdir()
    keyboard = Keyboard(
        [
            'a',
            *'agents',
            'enter',
            'e',
            'ctrl-u',
            *'other agents',
            'enter',
            'enter',
            'down',
            'down',
            'enter',
            'escape',
        ],
        width=width,
    )
    assert folders.run(runners=keyboard.runners) == [
        'Agent folder saved.',
        'Agent folder saved.',
        'Folder removed. Files were not changed.',
    ]
    assert folders.folders() == []
    assert (folders.project / 'agents').is_dir()
    assert (folders.project / 'other agents').is_dir()
    rendered = Text.from_ansi(keyboard.output.getvalue()).plain
    assert 'No folders selected.' in rendered
    assert 'Remove from search (keep files)' in rendered


@pytest.mark.parametrize(
    ('text', 'named', 'error'),
    [
        ('', False, 'Enter a directory path.'),
        (' ', True, 'Enter a folder name.'),
        ('no/such/name', True, 'Use letters'),
        ('bad\x00path', False, 'control characters'),
        ('bad\x1bpath', False, 'control characters'),
        ('./missing', False, 'does not exist'),
        ('./file.txt', False, 'not a directory'),
        ('agents', True, 'already in the list'),
    ],
)
def test_rejected_input_does_not_mutate_settings(
    folders: FolderMenu[object], text: str, named: bool, error: str
) -> None:
    folders.source.apply(folders.source.rows()[1], '["agents"]')
    (folders.project / 'file.txt').write_text('not a directory')
    assert error in (folders.problem(text, named=named, index=None) or '')
    keyboard = Keyboard(['enter', 'escape'])
    editor = folders.editor(named=named, index=None)
    editor.set_text(text)
    assert keyboard.text(editor).cancelled
    assert error in Text.from_ansi(keyboard.output.getvalue()).plain
    assert folders.folders() == ['agents']


def test_duplicate_aliases_and_editing_current_entry(folders: FolderMenu[object]) -> None:
    directory = folders.project / 'shared'
    directory.mkdir()
    (folders.project / 'alias').symlink_to(directory, target_is_directory=True)
    folders.source.apply(folders.source.rows()[1], '["./shared"]')
    for spelling in ('shared', './shared', str(directory), './alias', './shared/../shared'):
        assert folders.problem(spelling, named=False, index=None) == 'This folder is already in the list.'
        assert folders.problem(spelling, named=False, index=0) is None
    uppercase = directory.with_name('SHARED')
    if uppercase.exists():
        assert directory.samefile(uppercase)
        assert folders.problem(str(uppercase), named=False, index=None) == 'This folder is already in the list.'
    assert folders.problem('agents', named=True, index=None) is None
    assert folders.value(' shared ', named=False) == './shared'
    assert folders.value('~', named=False) == str(Path.home())
    assert folders.value('~/shared', named=False) == '~/shared'


def test_cancellation_at_each_level_keeps_saved_values(folders: FolderMenu[object]) -> None:
    folders.source.apply(folders.source.rows()[1], '["agents"]')
    script = Script(
        lists=[
            pick(0),
            pick(0),
            pick(FolderAction(kind='edit', index=0)),
            pick(FolderAction(kind='browse')),
            MenuResult(cancelled=True),
        ],
        choices=[MenuResult(cancelled=True), pick(None), MenuResult(cancelled=True)],
        texts=[TextInputResult(cancelled=True)],
    )
    assert folders.run(runners=script.runners) == []
    assert folders.folders() == ['agents']
    assert folders.action(save_and_close_item(), kind='remove') is None
    keyboard = Keyboard(['d', 'escape'])
    assert keyboard.menu(folders.build(initial=4)).cancelled


def test_browser_navigation_hidden_folders_and_empty_selection(folders: FolderMenu[object]) -> None:
    hidden = folders.project / '.agents'
    hidden.mkdir()
    (hidden / 'ignored.txt').write_text('not a folder')
    picker = DirectoryPicker(start=folders.project, project=folders.project)
    script = Script(lists=[], choices=[pick(hidden), pick(hidden.parent), pick(hidden), pick(True)], texts=[])
    assert picker.run(runners=script.runners) == hidden
    keyboard = Keyboard(['escape'])
    keyboard.menu(picker.build())
    rendered = Text.from_ansi(keyboard.output.getvalue()).plain
    assert 'No subdirectories.' in rendered
    assert 'ignored.txt' not in rendered
    keyboard = Keyboard([*'.agents', 'enter', 'enter'])
    assert DirectoryPicker(start=folders.project, project=folders.project).run(runners=keyboard.runners) == hidden


def test_browser_goto_home_relative_and_cancel(folders: FolderMenu[object]) -> None:
    picker = DirectoryPicker(start=folders.project, project=folders.project)
    assert picker.path('~') == Path.home()
    assert picker.path('home') == Path.home()
    assert picker.path_problem(' ') == 'Enter a directory path.'
    assert 'does not exist' in (picker.path_problem('missing') or '')
    assert 'Invalid directory' in (picker.path_problem('\x00') or '')
    script = Script(
        lists=[],
        choices=[pick('path'), pick('path'), pick(True)],
        texts=[TextInputResult(cancelled=True), typed('~/')],
    )
    assert picker.run(runners=script.runners) == Path.home()


def test_browse_add_replace_and_duplicate(folders: FolderMenu[object]) -> None:
    directory = folders.project / 'one'
    directory.mkdir()
    script = Script(
        lists=[
            pick(FolderAction(kind='browse')),
            pick(FolderAction(kind='browse')),
            pick(FolderAction(kind='browse', index=0)),
            MenuResult(cancelled=True),
        ],
        choices=[pick(directory), pick(True), pick(directory), pick(True), pick(folders.project), pick(True)],
        texts=[],
    )
    assert folders.run(runners=script.runners) == ['Agent folder saved.'] * 2
    assert folders.folders() == [str(folders.project)]


def test_missing_saved_directory_can_be_browsed_away_or_removed(folders: FolderMenu[object]) -> None:
    folders.source.apply(folders.source.rows()[1], '["./missing"]')
    assert 'does not exist' in folders.details(MenuItem('', value=0))
    keyboard = Keyboard(['enter', 'down', 'enter', 'enter', 'enter', 'd', 'escape'])
    assert folders.run(runners=keyboard.runners) == ['Agent folder saved.', 'Folder removed. Files were not changed.']
    assert folders.folders() == []


def test_browser_unreadable_directory_is_recoverable(
    folders: FolderMenu[object], monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(self: Path) -> object:
        raise PermissionError(13, 'Permission denied', str(self))

    monkeypatch.setattr(Path, 'iterdir', denied)
    picker = DirectoryPicker(start=folders.project, project=folders.project)
    keyboard = Keyboard(['escape'])
    assert picker.run(runners=keyboard.runners) is None
    rendered = Text.from_ansi(keyboard.output.getvalue()).plain
    assert 'Cannot read directory:' in rendered
    assert 'denied' in rendered
    assert 'Cannot read directory' in (folders.problem('./', named=False, index=None) or '')


def test_previews_wrap_long_paths_and_escape_controls(
    folders: FolderMenu[object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr('pydantic_clai2.builtin_plugins.coder_folders.terminal_size', lambda: (80, 40))
    long_path = './' + '/'.join(['long-directory-name'] * 6)
    folders.source.host.save_settings(CoderSettings(agent_folders=[long_path, 'agents', 'bad\x1b[31m']))
    details = folders.details(MenuItem('', value=0))
    assert long_path in details.replace('\n', '')
    assert all(len(line) <= 37 for line in details.splitlines())
    named = folders.details(MenuItem('', value=1)).replace('\n', '')
    assert '.claude/agents' in named
    assert 'Missing named locations are skipped.' in named
    assert '\x1b' not in folders.details(MenuItem('', value=2))
    assert 'Changes are already saved' in folders.details(save_and_close_item())


def test_save_failure_stays_in_menu_and_preserves_settings(folders: FolderMenu[object]) -> None:
    def fail_save(settings: dict[str, JsonValue]) -> None:
        raise OSError('Disk is full')

    folders.source = CoderSource(
        PluginHost[object](name='coder', console=Console(file=io.StringIO()), settings={}, save_settings=fail_save)
    )
    script = Script(
        lists=[pick(FolderAction(kind='name')), MenuResult(cancelled=True)], choices=[], texts=[typed('agents')]
    )
    assert folders.run(runners=script.runners) == []
    assert folders.notice == 'Could not save: Disk is full'
    assert folders.folders() == []


def test_symlink_loop_does_not_block_other_edits(folders: FolderMenu[object]) -> None:
    loop = folders.project / 'loop'
    loop.symlink_to(loop, target_is_directory=True)
    folders.source.host.save_settings(CoderSettings(agent_folders=['./loop'], sub_agents=False))
    assert folders.problem('./loop', named=False, index=None) is not None
    details = ' '.join(folders.details(MenuItem('', value=0)).split())
    assert 'Cannot open directory' in details or 'not a directory' in details
    assert folders.problem('./home', named=False, index=None) is None
    script = Script(
        lists=[pick(FolderAction(kind='browse', index=0)), MenuResult(cancelled=True)],
        choices=[pick(folders.project), pick(True)],
        texts=[],
    )
    assert folders.run(runners=script.runners) == ['Agent folder saved.']
    assert folders.folders() == [str(folders.project)]
    keyboard = Keyboard(['escape'])
    keyboard.menu(folders.build())
    assert 'Sub-agents are disabled' in Text.from_ansi(keyboard.output.getvalue()).plain


def test_directory_disappears_between_validation_and_listing(
    folders: FolderMenu[object], monkeypatch: pytest.MonkeyPatch
) -> None:
    real_iterdir = Path.iterdir
    calls = 0

    def vanished(path: Path) -> Iterator[Path]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FileNotFoundError(2, 'Directory disappeared')
        return real_iterdir(path)

    monkeypatch.setattr(Path, 'iterdir', vanished)
    picker = DirectoryPicker(start=folders.project, project=folders.project)
    keyboard = Keyboard(['escape'])
    keyboard.menu(picker.build())
    assert 'Cannot list directory' in Text.from_ansi(keyboard.output.getvalue()).plain


def test_browser_rechecks_selection_and_shortcuts(folders: FolderMenu[object]) -> None:
    picker = DirectoryPicker(start=folders.project / 'missing', project=folders.project)
    script = Script(lists=[], choices=[pick(True), pick(None), pick(folders.project), pick(True)], texts=[])
    assert picker.run(runners=script.runners) == folders.project
    keyboard = Keyboard(['ctrl-l', 'ctrl-u', *'home', 'enter', 'left', 'enter'])
    assert picker.run(runners=keyboard.runners) == folders.project


def test_full_browser_page_keeps_title_and_footer_on_screen(
    folders: FolderMenu[object], monkeypatch: pytest.MonkeyPatch
) -> None:
    for index in range(30):
        (folders.project / f'directory-{index}').mkdir()
    monkeypatch.setattr('pydantic_clai2.builtin_plugins.coder_folders.terminal_size', lambda: (50, 24))
    frame = render_frame(DirectoryPicker(start=folders.project, project=folders.project).build())
    assert frame[0] == 'Browse local directories'
    assert 'Esc back' in frame[-1]
    assert len(frame) < 24
    assert all(len(line) < 50 for line in frame)


@pytest.mark.parametrize('navigation', ['ctrl-l', 'left'])
def test_browser_shortcuts_recover_when_search_has_no_matches(folders: FolderMenu[object], navigation: str) -> None:
    picker = DirectoryPicker(start=Path.home(), project=folders.project)
    keys = [*'no-matching-directory', navigation]
    if navigation == 'ctrl-l':
        keys += ['ctrl-u', *str(folders.project), 'enter']
    keys += ['enter']
    keyboard = Keyboard(keys)
    assert picker.run(runners=keyboard.runners) == folders.project


def test_existing_control_characters_never_reach_text_input(folders: FolderMenu[object]) -> None:
    unsafe = './agent\x1b[2Jfolder'
    directory = folders.project / unsafe
    directory.mkdir()
    folders.source.host.save_settings(CoderSettings(agent_folders=[unsafe]))
    editor = folders.editor(named=False, index=0)
    assert editor.text == ''
    keyboard = Keyboard(['escape'])
    assert keyboard.text(editor).cancelled
    assert '\x1b[2J' not in keyboard.output.getvalue()
    picker = DirectoryPicker(start=directory, project=folders.project)
    keyboard = Keyboard(['ctrl-l', 'escape', 'escape'])
    assert picker.run(runners=keyboard.runners) is None
    assert '\x1b[2J' not in keyboard.output.getvalue()
    assert folders.folders() == [unsafe]


def test_sqlite_save_failure_keeps_menu_and_previous_settings(
    folders: FolderMenu[object], monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SettingsStore(folders.project / 'settings.db')
    saved_folders = [f'group{index}' for index in range(20)]
    declaration = PluginSettings(
        id='coder',
        factory='pydantic_clai2.builtin_plugins.coder',
        settings=CoderSettings(agent_folders=saved_folders).model_dump(mode='json'),
    )
    store.save_plugin(declaration)

    def persist(settings: dict[str, JsonValue]) -> None:
        store.save_plugin(declaration.model_copy(update={'settings': settings}))

    folders.source = CoderSource(
        PluginHost[object](
            name='coder', console=Console(file=io.StringIO()), settings=declaration.settings, save_settings=persist
        )
    )
    connect = sqlite3.connect

    def readonly(path: Path, *, timeout: float) -> sqlite3.Connection:
        connection = connect(path, timeout=timeout)
        connection.execute('PRAGMA query_only = ON')
        return connection

    monkeypatch.setattr(sqlite3, 'connect', readonly)
    script = Script(
        lists=[pick(FolderAction(kind='name')), MenuResult(cancelled=True)], choices=[], texts=[typed('agents')]
    )
    assert folders.run(runners=script.runners) == []
    assert 'readonly' in folders.notice
    assert folders.folders() == saved_folders
    assert store.plugins() == [declaration]
    monkeypatch.setattr('pydantic_clai2.builtin_plugins.coder_folders.terminal_size', lambda: (50, 24))
    frame = render_frame(folders.build())
    assert frame[0] == 'Agent folders'
    assert 'Esc' in frame[-1]
    assert folders.notice in ' '.join(line.strip() for line in frame)
    assert len(frame) < 24
    assert all(len(line) < 50 for line in frame)
