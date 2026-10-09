"""Drive Coder's folder settings through real widgets and temporary directories."""

import io
import sqlite3
from collections import deque
from collections.abc import Iterator, Sequence
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
from pydantic_clai2.ui.menus.field_menu import FieldRow, Runners, save_and_close_item
from tests.clai2.menu_script import Script, pick, typed

# One fixed terminal for every test, independent of the host's (CI sets a wide `COLUMNS`).
TERMINAL = (80, 30)


class Keyboard:
    """Drive real widgets through the public key, terminal-size and stdout seams."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        self.monkeypatch = monkeypatch
        self.capsys = capsys
        # A `(columns, rows)` entry resizes the terminal, then reports a key-less poll so the widget repaints.
        self.keys: deque[str | tuple[int, int]] = deque()
        monkeypatch.setattr('pydantic_clai2.builtin_plugins.coder_folders.menu_key', self.read)
        self.resize(*TERMINAL)

    def read(self) -> str:
        key = self.keys.popleft()
        if isinstance(key, str):
            return key
        self.resize(*key)
        return ''

    def resize(self, columns: int, rows: int = TERMINAL[1]) -> None:
        self.monkeypatch.setenv('COLUMNS', str(columns))
        self.monkeypatch.setenv('LINES', str(rows))

    def press(self, keys: Sequence[str | tuple[int, int]]) -> None:
        """Queue keys for the next widgets and start a fresh transcript."""
        self.keys.extend(keys)
        self.capsys.readouterr()

    def menu(self, menu: Menu) -> MenuResult:
        return menu.run()

    def text(self, widget: TextInput) -> TextInputResult:
        return widget.run()

    @property
    def runners(self) -> Runners:
        return Runners(run_list=self.menu, run_choice=self.menu, run_text=self.text)

    @property
    def transcript(self) -> str:
        """Raw terminal output since the last `press`."""
        return self.capsys.readouterr().out

    def frame(self, menu: Menu) -> list[str]:
        """The physical rows of the menu's only frame."""
        self.press(['escape'])
        assert menu.run().cancelled
        return last_frame(self.transcript)


def last_frame(transcript: str) -> list[str]:
    """The physical rows of the final repaint; each full repaint starts by homing the cursor."""
    return Text.from_ansi(transcript.split('\x1b[H')[-1]).plain.rstrip('\n').splitlines()


@pytest.fixture
def keyboard(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> Keyboard:
    return Keyboard(monkeypatch, capsys)


@pytest.fixture
def folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keyboard: Keyboard) -> FolderMenu[object]:
    monkeypatch.chdir(tmp_path)
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    host = PluginHost[object](
        name='coder', console=Console(file=io.StringIO()), settings={'instructions': 'Keep this.', 'sub_agents': True}
    )
    return FolderMenu(CoderSource(host), project=tmp_path)


def folders_row(folders: FolderMenu[object]) -> FieldRow:
    return next(row for row in folders.source.rows() if row.key == 'agent_folders')


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
    row = folders_row(folders)
    assert row.display(folders.source.current(row)) == '2 selected'


@pytest.mark.parametrize('width', [50, 80, 140])
def test_keyboard_add_edit_remove_and_empty_state(
    folders: FolderMenu[object], width: int, monkeypatch: pytest.MonkeyPatch, keyboard: Keyboard
) -> None:
    keyboard.resize(width)
    (folders.project / 'agents').mkdir()
    (folders.project / 'other agents').mkdir()
    keyboard.press(
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
        ]
    )
    assert folders.run(runners=keyboard.runners) == [
        'Agent folder saved.',
        'Agent folder saved.',
        'Folder removed. Files were not changed.',
    ]
    assert folders.folders() == []
    assert (folders.project / 'agents').is_dir()
    assert (folders.project / 'other agents').is_dir()
    rendered = Text.from_ansi(keyboard.transcript).plain
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
    folders: FolderMenu[object], text: str, named: bool, error: str, keyboard: Keyboard
) -> None:
    folders.source.apply(folders_row(folders), '["agents"]')
    (folders.project / 'file.txt').write_text('not a directory')
    assert error in (folders.problem(text, named=named, index=None) or '')
    keyboard.press(['enter', 'escape'])
    editor = folders.editor(named=named, index=None)
    editor.set_text(text)
    assert keyboard.text(editor).cancelled
    assert error in Text.from_ansi(keyboard.transcript).plain
    assert folders.folders() == ['agents']


def test_duplicate_aliases_and_editing_current_entry(folders: FolderMenu[object]) -> None:
    directory = folders.project / 'shared'
    directory.mkdir()
    (folders.project / 'alias').symlink_to(directory, target_is_directory=True)
    folders.source.apply(folders_row(folders), '["./shared"]')
    for spelling in ('shared', './shared', str(directory), './alias', './shared/../shared'):
        assert folders.problem(spelling, named=False, index=None) == 'This folder is already in the list.'
        assert folders.problem(spelling, named=False, index=0) is None
    uppercase = directory.with_name('SHARED')
    if uppercase.exists():  # pragma: lax no cover - only case-insensitive filesystems (macOS, Windows)
        assert directory.samefile(uppercase)
        assert folders.problem(str(uppercase), named=False, index=None) == 'This folder is already in the list.'
    assert folders.problem('agents', named=True, index=None) is None
    assert folders.value(' shared ', named=False) == './shared'
    assert folders.value('~', named=False) == str(Path.home())
    assert folders.value('~/shared', named=False) == '~/shared'


def test_cancellation_at_each_level_keeps_saved_values(folders: FolderMenu[object], keyboard: Keyboard) -> None:
    folders.source.apply(folders_row(folders), '["agents"]')
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
    keyboard.press(['d', 'escape'])
    assert keyboard.menu(folders.build(initial=4)).cancelled


def test_browser_navigation_hidden_folders_and_empty_selection(folders: FolderMenu[object], keyboard: Keyboard) -> None:
    hidden = folders.project / '.agents'
    hidden.mkdir()
    (hidden / 'ignored.txt').write_text('not a folder')
    picker = DirectoryPicker(start=folders.project, project=folders.project)
    script = Script(lists=[], choices=[pick(hidden), pick(hidden.parent), pick(hidden), pick(True)], texts=[])
    assert picker.run(runners=script.runners) == hidden
    keyboard.press(['escape'])
    keyboard.menu(picker.build())
    rendered = Text.from_ansi(keyboard.transcript).plain
    assert 'No subdirectories.' in rendered
    assert 'ignored.txt' not in rendered
    keyboard.press([*'.agents', 'enter', 'enter'])
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


@pytest.mark.parametrize(
    ('saved', 'problem', 'browse'),
    [
        # The browser opens in the missing directory, so step up to its parent first.
        ('./missing', 'does not exist', ['enter', 'enter']),
        # An unknown user has no directory to open, so the browser starts in the project.
        ('~nosuchuser-coder/agents', 'Cannot open directory', ['enter']),
    ],
)
def test_broken_saved_directory_can_be_browsed_away_or_removed(
    folders: FolderMenu[object], keyboard: Keyboard, saved: str, problem: str, browse: list[str]
) -> None:
    folders.source.host.save_settings(CoderSettings(agent_folders=[saved]))
    assert problem in ' '.join(folders.details(MenuItem('', value=0)).split())
    keyboard.press(['enter', 'down', 'enter', *browse, 'd', 'escape'])
    assert folders.run(runners=keyboard.runners) == ['Agent folder saved.', 'Folder removed. Files were not changed.']
    assert folders.folders() == []


def test_browser_unreadable_directory_is_recoverable(
    folders: FolderMenu[object], monkeypatch: pytest.MonkeyPatch, keyboard: Keyboard
) -> None:
    def denied(self: Path) -> object:
        raise PermissionError(13, 'Permission denied', str(self))

    monkeypatch.setattr(Path, 'iterdir', denied)
    picker = DirectoryPicker(start=folders.project, project=folders.project)
    keyboard.press(['escape'])
    assert picker.run(runners=keyboard.runners) is None
    rendered = Text.from_ansi(keyboard.transcript).plain
    assert 'Cannot read directory:' in rendered
    assert 'denied' in rendered
    assert 'Cannot read directory' in (folders.problem('./', named=False, index=None) or '')


def test_previews_wrap_long_paths_and_escape_controls(
    folders: FolderMenu[object], monkeypatch: pytest.MonkeyPatch, keyboard: Keyboard
) -> None:
    keyboard.resize(80, 40)
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


def test_symlink_loop_does_not_block_other_edits(folders: FolderMenu[object], keyboard: Keyboard) -> None:
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
    keyboard.press(['escape'])
    keyboard.menu(folders.build())
    assert 'Sub-agents are disabled' in Text.from_ansi(keyboard.transcript).plain


def test_directory_disappears_between_validation_and_listing(
    folders: FolderMenu[object], monkeypatch: pytest.MonkeyPatch, keyboard: Keyboard
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
    keyboard.press(['escape'])
    keyboard.menu(picker.build())
    assert 'Cannot list directory' in Text.from_ansi(keyboard.transcript).plain


def test_browser_rechecks_selection_and_shortcuts(folders: FolderMenu[object], keyboard: Keyboard) -> None:
    picker = DirectoryPicker(start=folders.project / 'missing', project=folders.project)
    script = Script(lists=[], choices=[pick(True), pick(None), pick(folders.project), pick(True)], texts=[])
    assert picker.run(runners=script.runners) == folders.project
    keyboard.press(['ctrl-l', 'ctrl-u', *'home', 'enter', 'left', 'enter'])
    assert picker.run(runners=keyboard.runners) == folders.project


def test_full_browser_page_keeps_title_and_footer_on_screen(
    folders: FolderMenu[object], monkeypatch: pytest.MonkeyPatch, keyboard: Keyboard
) -> None:
    for index in range(30):
        (folders.project / f'directory-{index}').mkdir()
    keyboard.resize(50, 24)
    frame = keyboard.frame(DirectoryPicker(start=folders.project, project=folders.project).build())
    assert frame[0] == 'Browse local directories'
    assert 'Esc back' in frame[-1]
    assert len(frame) < 24
    assert all(len(line) < 50 for line in frame)


def test_entry_actions_keep_title_on_screen_with_long_path(
    folders: FolderMenu[object], monkeypatch: pytest.MonkeyPatch, keyboard: Keyboard
) -> None:
    long_path = folders.project.joinpath(*[f'segment-{index:02d}' for index in range(40)])
    folders.source.host.save_settings(CoderSettings(agent_folders=[str(long_path)]))
    keyboard.resize(80, 24)
    frame = keyboard.frame(folders.actions(0))
    assert frame[0] == 'Manage agent folder'
    assert 'Esc back' in frame[-1]
    assert len(frame) < 24


def test_notices_fit_after_terminal_shrinks(folders: FolderMenu[object], keyboard: Keyboard) -> None:
    keyboard.resize(140, 24)
    folders.source.host.save_settings(CoderSettings(agent_folders=[f'group{index}' for index in range(20)]))
    folders.notice = 'Could not save: attempt to write a readonly database'
    menu = folders.build(initial=1)
    keyboard.press([(50, 24), 'escape'])
    assert menu.run().cancelled
    last = last_frame(keyboard.transcript)
    assert last[0] == 'Agent folders'
    assert len(last) < 24
    assert all(len(line) < 50 for line in last)
    assert folders.notice in ' '.join(line.strip() for line in last)
    assert menu.highlighted is not None and menu.highlighted.value == 1


@pytest.mark.parametrize('navigation', ['ctrl-l', 'left'])
def test_browser_shortcuts_recover_when_search_has_no_matches(
    folders: FolderMenu[object], navigation: str, keyboard: Keyboard
) -> None:
    picker = DirectoryPicker(start=Path.home(), project=folders.project)
    keys = [*'no-matching-directory', navigation]
    if navigation == 'ctrl-l':
        keys += ['ctrl-u', *str(folders.project), 'enter']
    keys += ['enter']
    keyboard.press(keys)
    assert picker.run(runners=keyboard.runners) == folders.project


def test_existing_control_characters_never_reach_text_input(folders: FolderMenu[object], keyboard: Keyboard) -> None:
    # Windows forbids control characters in file names, so the saved entry deliberately does not exist.
    unsafe = './agent\x1b[2Jfolder'
    directory = folders.project / unsafe
    folders.source.host.save_settings(CoderSettings(agent_folders=[unsafe]))
    editor = folders.editor(named=False, index=0)
    assert editor.text == ''
    keyboard.press(['escape'])
    assert keyboard.text(editor).cancelled
    assert '\x1b[2Jfolder' not in keyboard.transcript
    picker = DirectoryPicker(start=directory, project=folders.project)
    keyboard.press(['ctrl-l', 'escape', 'escape'])
    assert picker.run(runners=keyboard.runners) is None
    assert '\x1b[2Jfolder' not in keyboard.transcript
    assert folders.folders() == [unsafe]


def test_sqlite_save_failure_keeps_menu_and_previous_settings(
    folders: FolderMenu[object], monkeypatch: pytest.MonkeyPatch, keyboard: Keyboard
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
    keyboard.resize(50, 24)
    script = Script(
        lists=[pick(FolderAction(kind='name')), MenuResult(cancelled=True)], choices=[], texts=[typed('agents')]
    )
    frames: list[list[str]] = []

    def run_list(menu: Menu) -> MenuResult:
        frames.append(keyboard.frame(menu))
        return script.run_list(menu)

    assert folders.run(runners=Runners(run_list=run_list, run_text=script.run_text)) == []
    assert 'readonly' in folders.notice
    assert folders.folders() == saved_folders
    assert store.plugins() == [declaration]
    frame = frames[-1]
    assert frame[0] == 'Agent folders'
    assert 'Esc' in frame[-1]
    assert folders.notice in ' '.join(line.strip() for line in frame)
    assert len(frame) < 24
    assert all(len(line) < 50 for line in frame)
