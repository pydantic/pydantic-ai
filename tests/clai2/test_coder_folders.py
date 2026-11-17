"""Drive Coder's folder settings through scripted dialogs and temporary directories."""

import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import JsonValue
from rich.console import Console

from pydantic_clai2.builtin_plugins.coder import CoderPlugin, CoderSettings, CoderSource
from pydantic_clai2.builtin_plugins.coder_folders import DirectoryPicker, FolderMenu, edit_agent_folders
from pydantic_clai2.plugins import PluginHost
from pydantic_clai2.plugins.menus import FieldRow
from tests.clai2.dialog_script import Back, Pick, ScriptedDialogs, Type


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


def folders_row(folders: FolderMenu[object]) -> FieldRow:
    return next(row for row in folders.source.rows() if row.key == 'agent_folders')


def flat(text: str) -> str:
    """Text as the web UI shows it: descriptions break sentences into short lines for the terminal."""
    return ' '.join(text.split())


def labels(dialogs: ScriptedDialogs, index: int) -> list[str]:
    return [option.label for option in dialogs.asked[index].options]


async def test_add_edit_and_remove_preserve_order_and_preferences(folders: FolderMenu[object]) -> None:
    for name in ('first', 'replacement'):
        (folders.project / name).mkdir()
    script = ScriptedDialogs(
        [
            Pick(value='name'),
            Type(text='agents'),
            Pick(value='path'),
            Type(text='first'),
            Pick(value='1'),
            Pick(value='edit'),
            Type(text='./replacement'),
            Pick(value='0'),
            Pick(value='remove'),
            Pick(value='close'),
        ]
    )
    plugin = CoderPlugin[object].from_host(folders.source.host)
    action = plugin.settings_menu().actions['agent_folders']
    assert await action(script) == ['Agent folder saved.'] * 3 + ['Folder removed. Files were not changed.']
    assert folders.source.host.settings(CoderSettings).model_dump(mode='json') == {
        'instructions': 'Keep this.',
        'sub_agents': True,
        'agent_folders': ['./replacement'],
    }
    assert (folders.project / 'first').is_dir()
    first = script.asked[0]
    assert (first.kind, first.title, flat(first.description)) == (
        'choose',
        'Agent folders',
        'No folders selected. Disk agents are off.',
    )
    assert labels(script, 0) == [
        '+ Add directory path...',
        '+ Browse local directories...',
        '+ Add folder name...',
        'Save & close',
    ]
    assert 'Names search .agents' in flat(first.options[0].description)
    assert flat(first.options[-1].description) == 'Changes are already saved. Return to Coder settings.'
    named = script.asked[1]
    assert (named.title, named.label, named.placeholder, named.initial) == (
        'Add agent folder',
        'Name',
        'agents or global',
        '',
    )
    assert script.asked[2].description == 'Agent folder saved.'
    assert script.asked[3].label == 'Path'
    assert labels(script, 4)[:2] == ['1. agents', '2. first/']
    assert 'Missing named locations are skipped.' in script.asked[4].options[0].description
    manage = script.asked[5]
    assert manage.title == 'Manage agent folder'
    assert labels(script, 5) == [
        'Edit name or path...',
        'Replace by browsing...',
        'Remove from search (keep files)',
        'Back',
    ]
    assert 'Relative path\n./first' in manage.description
    assert (script.asked[6].title, script.asked[6].initial) == ('Edit agent folder', './first')
    row = folders_row(folders)
    assert row.display(folders.source.current(row)) == '1 selected'


async def test_backing_out_at_each_level_keeps_saved_values(folders: FolderMenu[object]) -> None:
    folders.source.apply(folders_row(folders), '["agents"]')
    script = ScriptedDialogs(
        [
            Pick(value='0'),
            Back(),
            Pick(value='0'),
            Pick(value='back'),
            Pick(value='0'),
            Pick(value='edit'),
            Back(),
            Pick(value='path'),
            Type(text=''),
            Pick(value='browse'),
            Back(),
            Back(),
        ]
    )
    assert await folders.run(script) == []
    assert folders.folders() == ['agents']
    assert script.asked[6].label == 'Name'
    assert script.unused == []


@pytest.mark.parametrize(
    ('text', 'named', 'error'),
    [
        ('', False, 'Enter a directory path.'),
        (' ', True, 'Enter a folder name.'),
        ('no/such/name', True, 'Use letters'),
        ('bad\x00path', False, 'control characters'),
        ('bad\x1bpath', False, 'control characters'),
        ('~/missing', False, 'does not exist'),
        ('~/file.txt', False, 'not a directory'),
        ('agents', True, 'already in the list'),
    ],
)
def test_rejected_input(folders: FolderMenu[object], text: str, named: bool, error: str) -> None:
    folders.source.apply(folders_row(folders), '["agents"]')
    (Path.home() / 'file.txt').write_text('not a directory')
    assert error in (folders.problem(text, named=named, index=None) or '')


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
    # An entry whose location cannot be resolved never blocks others, and an unresolvable candidate is reported.
    folders.source.host.save_settings(CoderSettings(agent_folders=['~nosuchuser-coder/agents']))
    assert folders.problem('./shared', named=False, index=None) is None
    assert 'Invalid directory' in (folders.problem('~nosuchuser-coder/other', named=False, index=None) or '')


async def test_relative_paths_are_saved_for_projects_that_have_them(folders: FolderMenu[object]) -> None:
    script = ScriptedDialogs([Pick(value='path'), Type(text='team-agents'), Back()])
    assert await folders.run(script) == ['Agent folder saved.']
    assert folders.folders() == ['./team-agents']
    assert folders.problem('./team-agents/', named=False, index=None) == 'This folder is already in the list.'
    details = ' '.join(folders.details(0).split())
    assert 'does not exist or is not a directory. Skipped in this project.' in details
    assert 'Resolved against each project CLAI runs in.' in details
    (folders.project / 'team-agents').mkdir()
    assert 'Directory is readable.' in ' '.join(folders.details(0).split())
    folders.source.host.save_settings(CoderSettings(agent_folders=[str(folders.project)]))
    assert folders.details(0).startswith(f'Directory path\n{folders.project}')


async def test_browse_add_replace_and_duplicate(folders: FolderMenu[object]) -> None:
    directory = folders.project / 'one'
    directory.mkdir()
    script = ScriptedDialogs(
        [
            Pick(value='browse'),
            Pick(value='open:one'),
            Pick(value='use'),
            Pick(value='browse'),
            Pick(value='open:one'),
            Pick(value='use'),
            Pick(value='0'),
            Pick(value='browse'),
            Pick(value='project'),
            Pick(value='use'),
            Pick(value='close'),
        ]
    )
    assert await folders.run(script) == ['Agent folder saved.'] * 2
    assert folders.folders() == [str(folders.project)]
    assert script.asked[6].description == 'This folder is already in the list.'
    # Replacing browses from the entry's own directory.
    assert f'Current directory\n{directory}' in script.asked[8].description


async def test_browser_navigation_hidden_folders_and_empty_directory(folders: FolderMenu[object]) -> None:
    hidden = folders.project / '.agents'
    hidden.mkdir()
    (hidden / 'ignored.txt').write_text('not a folder')
    picker = DirectoryPicker(start=folders.project, project=folders.project)
    script = ScriptedDialogs(
        [Pick(value='open:.agents'), Pick(value='parent'), Pick(value='open:.agents'), Pick(value='use')]
    )
    assert await picker.run(script) == hidden
    first = script.asked[0]
    assert first.title == 'Browse local directories'
    assert [option.label for option in first.options] == [
        'Use this directory',
        '.. (parent directory)',
        'Go to path / view current...',
        'Project directory',
        'Home directory',
        '.agents/',
        'home/',
    ]
    assert first.options[5].description == str(hidden)
    assert 'absolute path' in flat(first.description)
    inside = script.asked[1]
    assert flat(inside.description).startswith('No subdirectories. This directory can be used.')
    assert 'ignored.txt' not in str(inside.options)


async def test_browser_goto_home_root_and_backing_out(folders: FolderMenu[object]) -> None:
    picker = DirectoryPicker(start=folders.project, project=folders.project)
    assert picker.path('~') == Path.home()
    assert picker.path('home') == Path.home()
    assert picker.path_problem(' ') == 'Enter a directory path.'
    assert 'does not exist' in (picker.path_problem('missing') or '')
    assert 'Invalid directory' in (picker.path_problem('\x00') or '')
    script = ScriptedDialogs(
        [Pick(value='path'), Back(), Pick(value='path'), Type(text=''), Pick(value='path'), Type(text='~/'), Back()]
    )
    assert await picker.run(script) is None
    assert picker.current == Path.home()
    go_to = script.asked[1]
    assert (go_to.title, go_to.label, go_to.initial) == ('Go to directory', 'Path', str(folders.project))
    root = DirectoryPicker(start=Path(folders.project.anchor), project=folders.project)
    script = ScriptedDialogs([Pick(value='home'), Pick(value='use')])
    assert await root.run(script) == Path.home()
    assert '.. (parent directory)' not in labels(script, 0)


async def test_browser_rechecks_selection(folders: FolderMenu[object]) -> None:
    picker = DirectoryPicker(start=folders.project / 'missing', project=folders.project)
    script = ScriptedDialogs([Pick(value='use'), Pick(value='use'), Pick(value='project'), Pick(value='use')])
    assert await picker.run(script) == folders.project
    assert flat(script.asked[0].options[0].description) == 'Directory does not exist or is not a directory.'
    assert flat(script.asked[1].description).startswith('Directory does not exist')


@pytest.mark.parametrize(
    ('saved', 'problem'),
    [
        # The browser opens in the missing directory, so step up to its parent first.
        ('./missing', 'does not exist'),
        # An unknown user has no directory to open, so the browser starts in the project.
        ('~nosuchuser-coder/agents', 'Cannot open directory'),
    ],
)
async def test_broken_saved_directory_can_be_browsed_away_or_removed(
    folders: FolderMenu[object], saved: str, problem: str
) -> None:
    folders.source.host.save_settings(CoderSettings(agent_folders=[saved]))
    assert problem in ' '.join(folders.details(0).split())
    script = ScriptedDialogs([Pick(value='0'), Pick(value='browse'), Pick(value='project'), Pick(value='use'), Back()])
    assert await folders.run(script) == ['Agent folder saved.']
    assert folders.folders() == [str(folders.project)]


async def test_browser_unreadable_directory_is_recoverable(
    folders: FolderMenu[object], monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(self: Path) -> object:
        raise PermissionError(13, 'Permission denied', str(self))

    monkeypatch.setattr(Path, 'iterdir', denied)
    picker = DirectoryPicker(start=folders.project, project=folders.project)
    script = ScriptedDialogs([Back()])
    assert await picker.run(script) is None
    assert flat(script.asked[0].description).startswith('Cannot read directory: Permission denied')
    assert 'Cannot read directory' in (folders.problem(str(folders.project), named=False, index=None) or '')


async def test_directory_disappears_between_validation_and_listing(
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
    _options, description = DirectoryPicker(start=folders.project, project=folders.project).options()
    assert flat(description).startswith('Cannot list directory: Directory disappeared')


async def test_save_failure_stays_in_menu_and_preserves_settings(folders: FolderMenu[object]) -> None:
    def fail_save(settings: dict[str, JsonValue]) -> None:
        raise OSError('Disk is full')

    folders.source = CoderSource(
        PluginHost[object](name='coder', console=Console(file=io.StringIO()), settings={}, save_settings=fail_save)
    )
    script = ScriptedDialogs([Pick(value='name'), Type(text='agents'), Back()])
    assert await folders.run(script) == []
    assert flat(script.asked[2].description) == 'Could not save: Disk is full No folders selected. Disk agents are off.'
    assert folders.folders() == []


async def test_symlink_loop_does_not_block_other_edits(folders: FolderMenu[object]) -> None:
    loop = folders.project / 'loop'
    loop.symlink_to(loop, target_is_directory=True)
    folders.source.host.save_settings(CoderSettings(agent_folders=['./loop'], sub_agents=False))
    assert folders.problem(str(loop), named=False, index=None) is not None
    details = ' '.join(folders.details(0).split())
    assert 'Cannot open directory' in details or 'not a directory' in details
    assert folders.problem('./home', named=False, index=None) is None
    script = ScriptedDialogs([Pick(value='0'), Pick(value='browse'), Pick(value='project'), Pick(value='use'), Back()])
    assert await folders.run(script) == ['Agent folder saved.']
    assert folders.folders() == [str(folders.project)]
    assert flat(script.asked[0].description) == 'Sub-agents are disabled in Coder settings.'


async def test_existing_control_characters_are_escaped_and_never_offered_as_text(folders: FolderMenu[object]) -> None:
    # Windows forbids control characters in file names, so the saved entry deliberately does not exist.
    unsafe = './agent\x1b[2Jfolder'
    folders.source.host.save_settings(CoderSettings(agent_folders=[unsafe]))
    script = ScriptedDialogs(
        [Pick(value='0'), Pick(value='edit'), Back(), Pick(value='0'), Pick(value='browse'), Pick(value='path')]
        + [Back(), Back(), Back()]
    )
    assert await folders.run(script) == []
    shown = [
        text
        for asked in script.asked
        for text in (asked.description, asked.initial, *(option.label + option.description for option in asked.options))
    ]
    assert not any('\x1b' in text for text in shown)
    assert '\\x1b[2Jfolder/' in labels(script, 0)[0]
    assert script.asked[2].initial == ''
    assert script.asked[6].initial == ''
    assert folders.folders() == [unsafe]


async def test_the_action_lists_folders_for_the_working_directory(folders: FolderMenu[object]) -> None:
    script = ScriptedDialogs([Pick(value='browse'), Pick(value='use'), Back()])
    assert await edit_agent_folders(folders.source, script) == ['Agent folder saved.']
    assert folders.folders() == [str(folders.project)]
