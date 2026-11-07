"""`read_line`: one line of input pinned under the transcript, driven headless."""

import asyncio
import io
from collections.abc import Generator

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from rich.text import Text

from pydantic_clai2.ui.menus.menu_worker import holding_output
from pydantic_clai2.ui.prompt.line_input import LineInput, read_line
from pydantic_clai2.ui.prompt.prompt_surface import LEAVE, PromptSurface
from pydantic_clai2.ui.prompt.question_input import Paste, TranscriptKey
from tests.clai2.surface_terminal import SurfaceTerminal

URL = 'https://example.com/oauth/authorize?state=abc'
PROMPT = 'Or paste the address your browser ended on (Esc cancels): '


@pytest.fixture
def key_pipe(monkeypatch: pytest.MonkeyPatch) -> Generator[PipeInput]:
    with create_pipe_input() as pipe:
        monkeypatch.setattr('pydantic_clai2.ui.prompt.question_input.create_input', lambda: pipe)
        yield pipe


def live_console(terminal: SurfaceTerminal) -> tuple[PromptSurface, Console]:
    """The live editor's surface as a slash command finds it: painted once, then released."""
    surface = PromptSurface(output=terminal, size=lambda: (terminal.width, terminal.height))
    surface.paint(('> /login',))
    surface.release()
    return surface, Console(file=surface, width=terminal.width, height=terminal.height, color_system=None)


async def test_output_printed_before_the_prompt_stays_on_screen(key_pipe: PipeInput) -> None:
    """A plugin's sign-in URL was invisible: its prompt drew on the main screen, away from the transcript."""
    terminal = SurfaceTerminal(width=80, height=24)
    surface, console = live_console(terminal)
    key_pipe.send_text('the-code\r')
    with holding_output(surface.held):
        console.print(URL)
        assert await read_line(PROMPT, console=console) == 'the-code'
    assert terminal.alternate
    assert LEAVE not in terminal.getvalue(), 'the panel stays on screen'
    screen = [line for line in terminal.lines() if line]
    assert screen[-3:] == [URL, PROMPT.rstrip(), 'the-code']


async def test_hidden_answer_is_counted_and_pastes_join_into_one_line(key_pipe: PipeInput) -> None:
    terminal = SurfaceTerminal(width=80, height=24)
    surface, console = live_console(terminal)
    key_pipe.send_text('\x1b[200~https://example.com/\r\ncallback\x1b[201~x\x7f\r')
    with holding_output(surface.held):
        assert await read_line(PROMPT, console=console, hidden=True) == 'https://example.com/callback'
    assert 'callback' not in terminal.getvalue()
    assert [line for line in terminal.lines() if line][-1] == '[28 characters]'


@pytest.mark.parametrize('key', ['\x1b', '\x03'])
async def test_escape_and_ctrl_c_cancel(key_pipe: PipeInput, key: str) -> None:
    _, console = live_console(SurfaceTerminal(width=80, height=24))
    key_pipe.send_text(f'typed{key}')
    assert await read_line(PROMPT, console=console) is None


async def test_cancelling_the_reader_closes_the_prompt(key_pipe: PipeInput, monkeypatch: pytest.MonkeyPatch) -> None:
    """A browser callback that wins the race cancels the paste prompt."""
    terminal = SurfaceTerminal(width=80, height=24)
    _, console = live_console(terminal)
    shown = asyncio.Event()
    loop = asyncio.get_running_loop()
    frame = LineInput.frame

    def showing(line: LineInput, *, width: int) -> tuple[str, ...]:
        loop.call_soon_threadsafe(shown.set)
        return frame(line, width=width)

    monkeypatch.setattr(LineInput, 'frame', showing)
    reading = asyncio.create_task(read_line(PROMPT, console=console))
    await shown.wait()
    reading.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reading
    assert PROMPT.rstrip() not in terminal.lines(), 'the prompt rows are released'


def test_keys_edit_the_answer_and_transcript_keys_scroll() -> None:
    terminal = SurfaceTerminal(width=40, height=12)
    surface = PromptSurface(output=terminal, size=lambda: (40, 12))
    for index in range(30):
        surface.write(f'line {index}\n')

    def keys() -> Generator[str | Paste | TranscriptKey]:
        yield TranscriptKey(key='pageup')
        yield from 'ab'
        yield 'ctrl-r'  # no history to search: ignored
        yield 'left'
        yield Paste(text='X')
        yield ''
        yield 'enter'

    assert LineInput(prompt='Code: ').run(surface=surface, keys=keys().__next__) == 'aXb'
    assert surface.view.anchor is not None, 'Page Up scrolled the transcript, not the answer'


def test_long_answer_wraps_and_keeps_the_cursor_in_view() -> None:
    line = LineInput(prompt='Paste: ')
    line.feed(Paste(text='x' * 200))
    rows = [Text.from_ansi(row).plain for row in line.frame(width=40)]
    assert rows[0] == 'Paste: '
    assert len(rows) == 4
    assert rows[-1].endswith(' ')
    assert LineInput(prompt='Paste: ', hidden=True).frame(width=40)[1] == '\x1b[7m \x1b[27m'


def test_hidden_answers_are_edited_only_at_the_end() -> None:
    """The count cannot show where a moved cursor is, so moving it would edit text the user cannot see."""
    line = LineInput(prompt='Code: ', hidden=True)
    for key in ['a', 'left', 'b', 'home', 'c', 'left', 'backspace']:
        line.feed(key)
    assert line.answer.text == 'ab'
    assert Text.from_ansi(line.frame(width=40)[1]).plain == '[2 characters] '
    line.feed('backspace')
    assert Text.from_ansi(line.frame(width=40)[1]).plain == '[1 character] '


async def test_without_the_live_editor_it_is_a_plain_prompt() -> None:
    console = Console(file=io.StringIO())
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text('pasted value\r')
        assert await read_line('> ', console=console) == 'pasted value'
        pipe.send_text('\x03')
        assert await read_line('> ', console=console) is None
