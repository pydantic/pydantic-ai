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
from pydantic_clai2 import chat
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.ui.prompt.prompt_surface import LEAVE, PromptSurface
from pydantic_clai2.ui.prompt.prompt_transcript import TranscriptBuffer
from tests.clai2.surface_terminal import SurfaceTerminal


async def test_startup_and_plugin_messages_are_captured_once_before_editor_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[TranscriptBuffer] = []

    class Surface(PromptSurface):
        def paint(self, rows: tuple[str, ...]) -> None:
            if not captured:
                captured.append(self.transcript)
                text = '\n'.join(Text.from_ansi(row).plain for row in self.transcript.frame(width=200, height=200).rows)
                assert '/new starts a session' in text
                assert text.count('PLUGIN_LOAD_NOTICE') == 1
            super().paint(rows)

    monkeypatch.setattr('pydantic_clai2.ui.prompt.live_prompt.PromptSurface', Surface)
    store = SettingsStore(tmp_path / 'config.db')
    store.plugins_dir.mkdir()
    (store.plugins_dir / 'notice.py').write_text(
        'from pydantic_clai2.plugins import Plugin\n'
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
        def paint(self, rows: tuple[str, ...]) -> None:
            surfaces.append(self)
            super().paint(rows)

    monkeypatch.setattr('pydantic_clai2.ui.prompt.live_prompt.PromptSurface', Surface)
    output = io.StringIO()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(10):
        pipe.send_text(f'hello\r{command}\r/exit\r')
        await chat(
            Agent(TestModel()),
            deps=None,
            console=Console(file=output, force_terminal=True, width=80, height=24),
            store=SettingsStore(tmp_path / 'config.db'),
        )
    text = '\n'.join(Text.from_ansi(row).plain for row in surfaces[0].transcript.frame(width=200, height=200).rows)
    assert text.count('/new starts a session') == 1
    assert '> hello' not in text
    # The exit printout forgets the cleared conversation too, so it never reappears on screen.
    printed = Text.from_ansi(output.getvalue().rsplit(LEAVE, 1)[1]).plain
    assert '> hello' not in printed
    assert printed.index('/new starts a session') < printed.index('Goodbye.')
