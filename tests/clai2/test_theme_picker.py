"""Termflow palette selection through the shell, picker, and settings."""

import io
from pathlib import Path

import anyio
import pytest
from menu_script import Script, make_context, pick
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from rich.console import Console
from termflow.themes import PALETTES  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui import MenuItem  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import MenuResult  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2 import chat, theme
from pydantic_clai2.commands import config_command, config_completions, set_completions
from pydantic_clai2.config import Settings
from pydantic_clai2.field_menu import FieldMenu
from pydantic_clai2.set_menu import open_settings_menu
from pydantic_clai2.settings_store import SettingsStore
from pydantic_clai2.theme_picker import build_theme_picker, theme_command


async def test_picker_and_settings_share_registry_and_persistence(tmp_path: Path) -> None:
    context, applied = make_context(tmp_path)
    assert theme.names() == ('default', *PALETTES)
    assert build_theme_picker(context).highlighted == MenuItem('default (current)', value='default')
    script = Script(lists=[pick('github_light')], choices=[], texts=[])
    with theme.use(lambda: context.settings.theme):
        assert await theme_command(context, [], runners=script.runners) == 'Saved display.theme. Applied.'
        assert theme.current() is PALETTES['github_light']
        assert SettingsStore(context.store.path).load().theme == 'github_light'
        assert build_theme_picker(context).highlighted == MenuItem('github_light (current)', value='github_light')
        for name in theme.names():
            await theme_command(context, [name])
            assert theme.current() is PALETTES.get(name)
        for name in ('unknown', 'pydantic', 'light', 'system'):
            with pytest.raises(ValidationError, match='Unknown theme'):
                await theme_command(context, [name])
        with pytest.raises(ValueError, match='Usage: /theme'):
            await theme_command(context, ['github_light', 'extra'])
        assert context.settings.theme == context.store.load().theme == list(PALETTES)[-1]

        def edit(menu: FieldMenu) -> list[str]:
            row = menu.row_for('display.theme')
            assert row is not None and row.choices == theme.names()
            result = menu.apply(row, 'tokyo_night')
            assert theme.current() is PALETTES['tokyo_night']
            return [result]

        assert await open_settings_menu(context, run=edit) == 'Saved display.theme. Applied.'
        assert context.reset_setting('display.theme').startswith('Reset')
        assert theme.current() is None
    assert applied == ['display.theme'] * (len(theme.names()) + 3)
    assert context.store.overrides() == {}
    config_command(context.store, ['set', 'display.theme', 'github_light'])
    assert config_command(context.store, ['get', 'display.theme']) == '"github_light"'
    config_command(context.store, ['reset', 'display.theme'])
    assert context.store.load().theme == 'default'
    assert tuple(set_completions(['display.theme', ''])) == theme.names()
    assert tuple(config_completions(['set', 'display.theme', ''])) == theme.names()


@pytest.mark.parametrize('result', [MenuResult(cancelled=True), MenuResult(), pick(0)])
async def test_cancel_keeps_preference(tmp_path: Path, result: MenuResult) -> None:
    context, applied = make_context(tmp_path)
    script = Script(lists=[result], choices=[], texts=[])
    assert await theme_command(context, [], runners=script.runners) == 'No changes.'
    assert applied == [] and context.store.overrides() == {}


@pytest.mark.parametrize('key', ['escape', 'ctrl-c', 'enter'])
@pytest.mark.parametrize('width', [80, 120])
def test_picker_keyboard_and_preview(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, width: int) -> None:
    context, _ = make_context(tmp_path)
    output = io.StringIO()
    monkeypatch.setattr('sys.stdout', output)
    monkeypatch.setenv('COLUMNS', str(width))
    monkeypatch.setenv('LINES', '30')
    keys = iter([*'github', key])
    monkeypatch.setattr('pydantic_clai2.theme_picker.menu_key', lambda: next(keys))
    result = build_theme_picker(context).run()
    if key == 'enter':
        assert result.item == MenuItem('github_light', value='github_light')
    else:
        assert result.cancelled
    assert 'Select theme' in output.getvalue() and 'Summarize this change' in output.getvalue()
    assert 'Ask a follow-up' in output.getvalue()
    assert '\x1b]' not in output.getvalue()
    assert context.store.overrides() == {} and theme.current() is None


@pytest.mark.parametrize('terminal', [False, True])
@pytest.mark.parametrize(('initial', 'selected'), [('default', 'github_light'), ('tokyo_night', 'default')])
async def test_shell_applies_saved_theme_and_resets_on_exit(
    tmp_path: Path, terminal: bool, initial: str, selected: str
) -> None:
    changed, done = anyio.Event(), anyio.Event()

    class Output(io.StringIO):
        def write(self, text: str) -> int:
            if 'Saved display.theme.' in text:
                changed.set()
            return super().write(text)

    output = Output()
    store = SettingsStore(tmp_path / 'config.db')
    store.set('display.theme', initial)

    async def run() -> None:
        await chat(
            Agent(TestModel()),
            deps=None,
            settings=Settings(model=None, theme=store.load().theme),
            store=store,
            console=Console(file=output, force_terminal=terminal, width=100, height=24),
        )
        done.set()

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(run)
            pipe.send_text(f'/theme {selected}\n')
            await changed.wait()
            pipe.send_text('/exit\n')
            await done.wait()
    assert store.load().theme == selected
    text = output.getvalue()
    if terminal:
        name = selected if initial == 'default' else initial
        assert f'\x1b]11;{PALETTES[name].bg}\x07' in text
        assert text.count('\x1b]104\x07\x1b]111\x07\x1b]110\x07') == 1
    else:
        assert '\x1b]' not in text
    assert theme.current() is None


async def test_default_shell_leaves_terminal_palette_untouched(tmp_path: Path) -> None:
    output = io.StringIO()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(10):
        pipe.send_text('/theme default\n/exit\n')
        await chat(
            Agent(TestModel()),
            deps=None,
            console=Console(file=output, force_terminal=True, color_system='truecolor', width=100, height=24),
            store=SettingsStore(tmp_path / 'config.db'),
        )
    text = output.getvalue()
    assert '\x1b]' not in text
    assert '\x1b[38;2;229;32;233m' in text or '\x1b[95m' in text
    assert theme.current() is None


async def test_palette_scope_restores_after_outer_cancellation() -> None:
    output = io.StringIO()
    with anyio.CancelScope() as scope:
        with theme.use(lambda: 'github_light', output=output):
            assert theme.current() is PALETTES['github_light']
            scope.cancel()
            await anyio.sleep_forever()
    assert theme.current() is None
    assert output.getvalue().endswith('\x1b]104\x07\x1b]111\x07\x1b]110\x07')
