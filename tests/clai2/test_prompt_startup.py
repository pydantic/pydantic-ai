"""The replay cache covers startup and plugin lifecycle output, not just turns."""

import io
from pathlib import Path

import anyio
import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from rich.console import Console
from rich.text import Text

from pydantic_clai2 import chat
from pydantic_clai2.prompt_surface import PromptSurface
from pydantic_clai2.prompt_transcript import TranscriptBuffer
from pydantic_clai2.settings_store import SettingsStore


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


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

    monkeypatch.setattr('pydantic_clai2.live_prompt.PromptSurface', Surface)
    store = SettingsStore(tmp_path / 'config.db')
    store.plugins_dir.mkdir()
    (store.plugins_dir / 'notice.py').write_text(
        'def activate(host):\n'
        "    host.console.print('PLUGIN_LOAD_NOTICE')\n"
        "    @host.on('session_end')\n"
        '    async def end(event):\n'
        "        host.console.print('PLUGIN_END_NOTICE')\n"
    )
    output = io.StringIO()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(10):
        pipe.send_text('/exit\n')
        await chat(
            Agent(TestModel()),
            deps=None,
            console=Console(file=output, force_terminal=True, width=80, height=24),
            store=store,
        )
    text = '\n'.join(Text.from_ansi(row).plain for row in captured[0].frame(width=200, height=200).rows)
    assert text.count('PLUGIN_LOAD_NOTICE') == text.count('PLUGIN_END_NOTICE') == 1
    assert output.getvalue().count('PLUGIN_LOAD_NOTICE') == output.getvalue().count('PLUGIN_END_NOTICE') == 1
