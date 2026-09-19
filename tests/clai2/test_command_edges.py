"""Invalid commands and completion boundaries through the public registry."""

from pathlib import Path

import pytest
from termflow.tui.completion import CompleteEvent, Document  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2.commands import (
    Command,
    Commands,
    config_command,
    config_completions,
    is_command_input,
    plugins_command,
    set_completions,
)
from pydantic_clai2.settings_store import SettingsStore


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
    plugins_command(store, ['add', 'one', 'module:Factory'])
    plugins_command(store, ['add', 'two', 'module:Factory', '{}'])
    plugins_command(store, ['disable', 'one'])
    assert 'disabled' in plugins_command(store, ['list'])
    plugins_command(store, ['enable', 'one'])
    for args in (['enable', 'missing'], ['bad']):
        with pytest.raises(ValueError):
            plugins_command(store, args)
    with pytest.raises(ValueError):
        store.reset('missing')


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
    assert [item.text for item in completions] == ['le.txt']
