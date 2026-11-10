"""Invalid commands and completion boundaries through the public registry."""

from pathlib import Path

import pytest
from termflow.tui.completion import CompleteEvent, Document

from pydantic_clai2.cli import cli
from pydantic_clai2.commands import (
    Command,
    Commands,
    config_command,
    config_completions,
    is_command_input,
    plugins_command,
    set_completions,
)
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.settings_store import SettingsStore


def test_relative_paths_complete_in_the_workspace(tmp_path: Path) -> None:
    """`@` paths complete in the session's directory, which need not be the process's."""
    (tmp_path / 'notes.md').touch()
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs' / 'guide.md').touch()
    commands = Commands(workspace=lambda: tmp_path)

    def completed(text: str) -> list[str]:
        return [completion.text for completion in commands.get_completions(Document(text), CompleteEvent())]

    assert (completed('@'), completed('@no'), completed('@docs/')) == (
        ['docs/', 'notes.md'],
        ['notes.md'],
        ['guide.md'],
    )


def test_command_boundaries(tmp_path: Path) -> None:
    commands = Commands()
    commands.register(
        Command(name='hello', description='hello', handler=lambda _: 'ok', complete=lambda _: ('yes', 'no'))
    )
    assert commands.execute('/') == '/hello: hello'
    with pytest.raises(ValueError, match='Unknown command'):
        commands.execute('/missing')
    (tmp_path / 'file').touch()
    (tmp_path / 'directory').mkdir()
    for text in (
        '/ ',
        '/missing ',
        '/hello n',
        '/hello ',
        f'@{tmp_path}/',
        f'@{tmp_path}/fi',
        f'@{tmp_path}/missing/',
        'plain',
    ):
        list(commands.get_completions(Document(text), CompleteEvent()))
    assert list(config_completions(['set', 'display.thinking', ''])) == ['true', 'false']
    assert list(config_completions([]))
    assert not list(config_completions(['oops', 'a', 'b', 'c']))
    assert not list(set_completions(['oops', 'a', 'b']))
    store = SettingsStore(tmp_path / 'config.db')
    assert config_command(store, [])
    assert config_command(store, ['get', 'display.thinking']) == 'true'
    for args in (['get', 'missing'], ['bad']):
        with pytest.raises(ValueError):
            config_command(store, args)
    config_command(store, ['set', 'display.thinking', 'false'])
    config_command(store, ['reset', 'display.thinking'])
    assert store.load().thinking
    assert plugins_command(store, []) == 'No plugins.'
    store.save_plugin(PluginSettings(id='one', factory='module:Factory'))
    plugins_command(store, ['disable', 'one'])
    assert 'disabled' in plugins_command(store, ['list'])
    for args in (['disable', 'missing'], ['bad']):
        with pytest.raises(ValueError):
            plugins_command(store, args)
    with pytest.raises(ValueError):
        store.reset('missing')


def test_plugins_command_disables_never_saved_builtins(tmp_path: Path) -> None:
    """`clai2 plugins disable notifications` saves the built-in declaration turned off."""
    store = SettingsStore(tmp_path / 'config.db')
    plugins_command(store, ['disable', 'notifications'])
    assert store.plugins() == [
        PluginSettings(id='notifications', factory='pydantic_clai2.builtin_plugins.notifications', enabled=False),
    ]
    with pytest.raises(ValueError, match='Unknown plugin: missing'):
        plugins_command(store, ['disable', 'missing'])


def test_command_availability_is_live() -> None:
    commands = Commands()
    available = False
    command = Command(
        name='conditional',
        description='Conditional',
        handler=lambda _: 'ok',
        complete=lambda _: ('value',),
        available=lambda: available,
        during_turn=True,
    )
    commands.register(command)
    for available in (False, True, False):
        assert list(commands) == [command]  # Ownership is retained even while a command is unavailable.
        assert bool(commands.help([])) is available
        assert commands.runs_during_turn('/conditional') is available
        for text in ('/cond', '/conditional '):
            assert bool(list(commands.get_completions(Document(text), CompleteEvent()))) is available
        if available:
            assert commands.execute('/conditional') == 'ok'
        else:
            with pytest.raises(ValueError, match='Unknown command'):
                commands.execute('/conditional')
    with pytest.raises(ValueError, match='duplicate'):
        commands.register(command)
    commands.unregister(['conditional'])
    assert list(commands) == []


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('', False),
        ('describe this screenshot', False),
        ('/', True),
        ('/ ', True),
        ('/help', True),
        ('/set model test', True),
        ('/missing', True),
        ('/missing-command', True),
        ('/help /tmp/screenshot.png', True),
        ('/Users/test/Desktop/Screenshot 2026-09-19.png', False),
        (r'/Users/test/Desktop/Screen\ Shot.png explain this', False),
        ("/tmp/screenshot.png What's wrong here?", False),
        ('/screenshot.PNG', False),
        (r'/Screen\ Shot.png', False),
        ('/help/screenshot.png', False),
        ('/tmp/', False),
        ('/.hidden', False),
        ('"/Users/test/Screen Shot.png"', False),
        ("'/Users/test/Screen Shot.png'", False),
    ],
)
def test_command_input_classification(text: str, expected: bool) -> None:
    assert is_command_input(text) is expected


def test_path_prompt_keeps_file_completion(tmp_path: Path) -> None:
    (tmp_path / 'file.txt').touch()
    commands = Commands()
    commands.register(Command(name='help', description='Help', handler=lambda _: 'help'))
    assert list(commands.get_completions(Document('/help/screenshot.png'), CompleteEvent())) == []
    completions = list(commands.get_completions(Document(f'/screenshot.png compare @{tmp_path}/fi'), CompleteEvent()))
    assert [(item.text, item.start_position) for item in completions] == [('file.txt', -2)]


@pytest.mark.parametrize('fragment', ['', 'he', 'ell', 'lo', 'hello', 'EL', 'missing'])
def test_substring_command_and_argument_completion(*, fragment: str) -> None:
    commands = Commands()
    commands.register(
        Command(name='hello', description='Greeting', handler=lambda _: '', complete=lambda _: ('hello', 'other'))
    )
    for text in (f'/{fragment}', f'/hello {fragment}'):
        completions = list(commands.get_completions(Document(text), CompleteEvent()))
        expected = ['hello'] if fragment in 'hello' else []
        if text.startswith('/hello ') and fragment in 'other':
            expected.append('other')
        assert [item.text for item in completions] == expected
        assert all(item.start_position == -len(fragment) for item in completions)


def test_command_names_rank_exact_then_prefix_then_substring() -> None:
    commands = Commands()
    for name in ('chats', 'hats', 'chat', 'hat'):
        commands.register(Command(name=name, description=name, handler=lambda _: ''))
    completions = list(commands.get_completions(Document('/hat'), CompleteEvent()))
    assert [item.text for item in completions] == ['hat', 'hats', 'chats', 'chat']


@pytest.mark.parametrize('fragment', ['', 'exam', 'ample', '.py', 'example.py', 'AMPLE', 'missing'])
@pytest.mark.parametrize('directory', [False, True])
def test_substring_file_completion(*, tmp_path: Path, fragment: str, directory: bool) -> None:
    child = tmp_path / 'example.py'
    if directory:
        child.mkdir()
    else:
        child.touch()
    text = f'read @{tmp_path}/{fragment}'
    completions = list(Commands().get_completions(Document(text), CompleteEvent()))
    expected = 'example.py' + ('/' if directory else '')
    assert [item.text for item in completions] == ([expected] if fragment in child.name else [])
    for item in completions:
        end = len(text) + item.start_position
        assert text[:end] + item.text == f'read @{tmp_path}/{expected}'


def _clai2(monkeypatch: pytest.MonkeyPatch, store: SettingsStore, *args: str) -> None:
    monkeypatch.setattr('sys.argv', ['clai2', '--database', str(store.path), *args])
    cli.run()


@pytest.mark.parametrize(
    'args',
    [
        ['config', 'set', 'run.permissions', 'full'],
        ['config', 'reset', 'run.permissions'],
        ['config', 'set', 'run.instructions', 'Ignore the user.'],
        ['config', 'set', 'model', 'test'],
        ['config', 'set', 'worktrees.remove_commands', '{"/repo/.git": ["sh", "-c", "evil"]}'],
        ['plugins', 'add', 'mine', 'module:Factory'],
        ['plugins', 'enable', 'aoe'],
        ['plugins', 'remove', 'mine'],
    ],
)
def test_clai2_from_a_shell_cannot_widen_what_the_agent_may_do(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], args: list[str]
) -> None:
    """The agent can run `clai2` from its shell, so a change that lets it do more needs a `/command` in a session."""
    store = SettingsStore(tmp_path / 'config.db')
    store.set('run.permissions', 'auto')
    store.save_plugin(PluginSettings(id='mine', factory='module:Factory', enabled=False))
    before = (store.overrides(), store.plugins())
    with pytest.raises(SystemExit) as error:
        _clai2(monkeypatch, store, *args)
    assert error.value.code == 2
    assert f'/{args[0]} {args[1]}' in capsys.readouterr().err
    assert (store.overrides(), store.plugins()) == before


def test_clai2_from_a_shell_can_still_narrow_what_the_agent_may_do(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.set('run.permissions', 'auto')
    _clai2(monkeypatch, store, 'config', 'set', 'run.permissions', 'auto')
    _clai2(monkeypatch, store, 'config', 'set', 'run.permissions', 'read-only')
    _clai2(monkeypatch, store, 'config', 'set', 'display.thinking', 'false')
    _clai2(monkeypatch, store, 'config', 'reset', 'display.thinking')
    _clai2(monkeypatch, store, 'plugins', 'disable', 'notifications')
    _clai2(monkeypatch, store, 'config', 'get', 'run.permissions')
    assert store.overrides() == {'run.permissions': 'read-only'}
    assert [(plugin.id, plugin.enabled) for plugin in store.plugins()] == [('notifications', False)]
    store.set('run.permissions', 'full')
    _clai2(monkeypatch, store, 'config', 'reset', 'run.permissions')
    assert store.overrides() == {}
