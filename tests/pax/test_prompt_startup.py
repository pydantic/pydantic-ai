"""The transcript covers startup and plugin lifecycle output, and scrollback gets each line once."""

import io
from pathlib import Path

import anyio
import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from rich.text import Text

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_pax import chat
from pydantic_ai_pax.cli.self_update import installed
from pydantic_ai_pax.config import Settings
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.ui.prompt.prompt_surface import ENTER, LEAVE, PromptSurface
from pydantic_ai_pax.ui.prompt.prompt_transcript import TranscriptBuffer
from tests._timeouts import READINESS_WAIT_TIMEOUT
from tests.pax.surface_terminal import SurfaceTerminal


async def test_startup_and_plugin_messages_are_captured_once_before_editor_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[TranscriptBuffer] = []

    class Surface(PromptSurface):
        def paint(self, rows: tuple[str, ...], *, select_pinned: bool = False) -> None:
            if not captured:
                captured.append(self.transcript)
                text = '\n'.join(Text.from_ansi(row).plain for row in self.transcript.frame(width=200, height=200).rows)
                assert '/new starts a session' in text
                assert f'pydantic-ai-pax {installed().label}' in text
                assert text.count('PLUGIN_LOAD_NOTICE') == 1
            super().paint(rows, select_pinned=select_pinned)

    monkeypatch.setattr('pydantic_ai_pax.ui.prompt.live_prompt.PromptSurface', Surface)
    store = SettingsStore(tmp_path / 'config.db')
    store.plugins_dir.mkdir()
    (store.plugins_dir / 'notice.py').write_text(
        'from pydantic_ai_pax.plugins import Plugin\n'
        'class Notice(Plugin):\n'
        '    async def on_session_start(self, event):\n'
        "        self.host.console.print('PLUGIN_LOAD_NOTICE')\n"
        '    async def on_session_end(self, event):\n'
        "        self.host.console.print('PLUGIN_END_NOTICE')\n"
    )
    terminal = SurfaceTerminal(width=80, height=24)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(10):
        pipe.send_text('/exit\r')
        await chat(
            Agent(TestModel()),
            deps=None,
            console=Console(file=terminal, force_terminal=True, width=80, height=24),
            store=store,
        )
    text = '\n'.join(Text.from_ansi(row).plain for row in captured[0].frame(width=200, height=200).rows)
    assert text.count('PLUGIN_LOAD_NOTICE') == text.count('PLUGIN_END_NOTICE') == 1
    assert not terminal.alternate
    scrollback = '\n'.join([*terminal.history, *terminal.lines()])
    assert scrollback.count('PLUGIN_LOAD_NOTICE') == scrollback.count('PLUGIN_END_NOTICE') == 1
    assert scrollback.count('/new starts a session') == 1
    assert scrollback.index('PLUGIN_LOAD_NOTICE') < scrollback.index('Goodbye.') < scrollback.index('PLUGIN_END_NOTICE')


@pytest.mark.parametrize('command', ['/clear', 'clear', '/new'])
async def test_clear_returns_to_the_start_screen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str) -> None:
    surfaces: list[PromptSurface] = []

    class Surface(PromptSurface):
        def paint(self, rows: tuple[str, ...], *, select_pinned: bool = False) -> None:
            surfaces.append(self)
            super().paint(rows, select_pinned=select_pinned)

    monkeypatch.setattr('pydantic_ai_pax.ui.prompt.live_prompt.PromptSurface', Surface)
    output = io.StringIO()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(10):
        pipe.send_text(f'hello\r{command}\r/exit\r')
        await chat(
            Agent(TestModel()),
            deps=None,
            console=Console(file=output, force_terminal=True, width=80, height=24),
            store=SettingsStore(tmp_path / 'config.db'),
            # A compact panel prints the conversation into scrollback when a command takes over.
            settings=Settings(full_screen=True),
        )
    text = '\n'.join(Text.from_ansi(row).plain for row in surfaces[0].transcript.frame(width=200, height=200).rows)
    assert text.count('/new starts a session') == 1
    assert '> hello' not in text
    # The exit printout forgets the cleared conversation too, so it never reappears on screen.
    printed = Text.from_ansi(output.getvalue().rsplit(LEAVE, 1)[1]).plain
    assert '> hello' not in printed
    assert printed.index('/new starts a session') < printed.index('Goodbye.')


@pytest.mark.parametrize('full_screen', [False, True])
async def test_the_editor_starts_compact_unless_the_setting_asks_for_full_screen(
    tmp_path: Path, full_screen: bool
) -> None:
    output = io.StringIO()
    with (
        create_pipe_input() as pipe,
        create_app_session(input=pipe, output=DummyOutput()),
        anyio.fail_after(READINESS_WAIT_TIMEOUT),
    ):
        # A command between turns keeps a compact panel compact.
        pipe.send_text('hello\r/set display.thinking false\r/exit\r')
        await chat(
            Agent(TestModel()),
            deps=None,
            console=Console(file=output, force_terminal=True, width=80, height=24),
            settings=Settings(full_screen=full_screen),
            store=SettingsStore(tmp_path / 'config.db'),
        )
    assert (ENTER in output.getvalue()) is full_screen
    assert 'Saved display.thinking.' in output.getvalue()
