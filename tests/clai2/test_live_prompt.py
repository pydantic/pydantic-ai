"""Exercise native editor state and ownership without a prompt-toolkit renderer."""

import asyncio
import io
import signal
import sys
from collections.abc import AsyncGenerator, Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Event as ThreadEvent

import anyio
import pytest
from anyio.to_thread import run_sync as in_worker
from PIL import Image
from prompt_toolkit.application import create_app_session
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai import PartStartEvent, TextPart, ThinkingPart
from pydantic_ai.messages import BinaryContent
from rich.console import Console
from rich.text import Text
from surface_terminal import SurfaceTerminal
from termflow.tui.completion import Completion  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2 import StreamRenderer, theme
from pydantic_clai2.commands import Command, Commands
from pydantic_clai2.image_input import ImageInput
from pydantic_clai2.interrupts import Interrupts
from pydantic_clai2.live_prompt import LivePrompt
from pydantic_clai2.prompt_completion import CompletionWorker


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@asynccontextmanager
async def editor(*, output: io.StringIO | None = None) -> AsyncGenerator[tuple[LivePrompt, PipeInput, io.StringIO]]:
    output = output if output is not None else io.StringIO()
    console = Console(file=output, force_terminal=True, width=80, height=24)
    commands = Commands()
    commands.register(Command(name='help', description='Help', handler=lambda args: 'help'))
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(5):
        live = LivePrompt(
            console=console,
            commands=commands,
            history=InMemoryHistory(),
            images=ImageInput(),
            interrupts=Interrupts(),
            toolbar=lambda: [('', 'ready')],
            clock=lambda: 0,
        )
        async with live.opened():
            yield live, pipe, output
        assert console.file is output
        assert output.getvalue().endswith('\x1b[?25h\x1b[?2026l')


async def test_input_queue_and_controls() -> None:
    async with editor() as (live, pipe, _):
        pipe.send_text('  \nfirst\nsecond\n')
        assert await live.read() == 'first'
        assert await live.read() == 'second'
        pipe.send_text('discard\x03')
        with pytest.raises(KeyboardInterrupt):
            await live.read()
        assert live.buffer.text == ''
        pipe.send_text('keep\x01\x04\n')
        assert await live.read() == 'eep'
        pipe.send_text('\x04')
        with pytest.raises(EOFError):
            await live.read()


@pytest.mark.parametrize(
    'sequence',
    ['\x1b\x7f', '\x1b\x08', '\x1b[27;3;127~', '\x1b[27;3;8~', '\x1b[127;3u', '\x1b[8;3u'],
)
async def test_option_backspace_deletes_word_before_cursor(sequence: str) -> None:
    async with editor() as (live, pipe, _):
        pipe.send_text('one two three' + '\x1b[D' * 6 + sequence + '\n')
        assert await live.read() == 'one  three'
        pipe.send_text('one\x1b[13;2utwo' + sequence + 'three\n')
        assert await live.read() == 'one\nthree'


async def test_completed_and_partial_output_never_repaint_editor() -> None:
    async with editor() as (live, _, output):
        live.buffer.replace('retained draft')
        live.paint()
        start = len(output.getvalue())
        for chunk in ('streaming ', 'partial', '\n', 'next line\n'):
            live.console.file.write(chunk)
            live.console.file.flush()
        assert output.getvalue()[start:] == 'streaming partial\nnext line\n'
        assert live.buffer.text == 'retained draft'
        live.paint()
        assert output.getvalue()[start:] == 'streaming partial\nnext line\n'


@pytest.mark.parametrize('thinking', [False, True])
async def test_real_termflow_writes_do_not_clear_input(thinking: bool) -> None:
    async with editor() as (live, _, output):
        live.buffer.replace('retained draft')
        live.paint()
        start = len(output.getvalue())
        renderer = StreamRenderer(live.console, stop_loading=lambda: None)
        part = ThinkingPart(content='A thinking burst') if thinking else TextPart(content='A response burst\n')
        await renderer.on_stream_event(PartStartEvent(index=0, part=part))
        await renderer.finish()
        assert live.buffer.text == 'retained draft'
        text = output.getvalue()[start:]
        assert 'A thinking burst' in Text.from_ansi(text).plain if thinking else 'A response burst' in text
        for forbidden in ('\x1b[J', '\x1b[2K', '\x1b[?25h', '┌', '└'):
            assert forbidden not in text


async def test_menu_suspension_preserves_draft_and_output() -> None:
    async with editor() as (live, _, output):
        live.buffer.replace('draft')
        live.paint()
        live.output.write('partial')
        async with live.suspended():
            start = len(output.getvalue())
            live.paint()
            assert output.getvalue()[start:] == ''
            async with live.suspended():
                live.console.print('menu output')
                assert 'menu output' in output.getvalue()
            assert live.buffer.text == 'draft'
        assert 'draft' in Text.from_ansi(output.getvalue()).plain


@pytest.mark.parametrize('menu', [False, True])
async def test_outer_cancellation_releases_terminal_and_tasks(menu: bool) -> None:
    before = asyncio.all_tasks()
    with anyio.CancelScope() as scope:
        async with editor() as (live, _, _):
            if menu:
                async with live.suspended():
                    scope.cancel()
                    await anyio.sleep_forever()
            else:
                scope.cancel()
                await anyio.sleep_forever()
    assert scope.cancelled_caught
    assert asyncio.all_tasks() <= before


@pytest.mark.parametrize('key', ['\x03', '\x1b'])
async def test_interrupt_targets_work_and_preserves_draft(key: str) -> None:
    async with editor() as (live, pipe, _):
        started, done = anyio.Event(), anyio.Event()

        async def operation() -> None:
            started.set()
            await anyio.sleep_forever()

        async def run() -> None:
            assert not await live.interrupts.run(operation())
            done.set()

        live.buffer.replace('retained draft')
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(run)
            await started.wait()
            assert any('Working ⠋' in Text.from_ansi(row).plain for row in live.frame())
            pipe.send_text(key)
            await done.wait()
        assert not any('Working' in row for row in live.frame())
        assert live.buffer.text == 'retained draft'


async def test_closed_input_reports_eof() -> None:
    async with editor() as (live, pipe, _):
        pipe.close()
        with pytest.raises(EOFError):
            await live.read()


async def test_paste_is_atomic_and_alt_word_editing_works() -> None:
    async with editor() as (live, pipe, _):
        pipe.send_text('\x1b[200~one\ntwo\x1b[201~\x1bbX\n')
        assert await live.read() == 'Xone\ntwo'
        assert live.buffer.text == ''


async def test_image_paste_and_failure_notice(monkeypatch: pytest.MonkeyPatch) -> None:
    async with editor() as (live, _, _):
        image = BinaryContent(data=b'png', media_type='image/png')
        monkeypatch.setattr('pydantic_clai2.live_prompt.clipboard_images', lambda: [image])
        live.feed('alt-v')
        assert live.images.resolve(live.buffer.text) == ('', [image])

        def fail() -> list[BinaryContent]:
            raise ValueError('clipboard unavailable')

        monkeypatch.setattr('pydantic_clai2.live_prompt.clipboard_images', fail)
        live.feed('ctrl-v')
        assert 'clipboard unavailable' in live.images.notice
        live.feed('paste', 'plain\r\ntext')
        assert live.buffer.text.endswith('plain\ntext')


async def test_queue_labels_and_small_terminal() -> None:
    async with editor() as (live, _, _):
        for text in ('/tmp/shot.png', '/help'):
            live.buffer.replace(text)
            live.accept()
        frame = '\n'.join(Text.from_ansi(row).plain for row in live.frame())
        assert 'Follow-up: /tmp/shot.png' in frame
        assert 'Command: /help' in frame
        assert await live.read() == '/tmp/shot.png'
        assert await live.read() == '/help'
        live.console.size = (3, 3)
        assert len(live.frame()) == 1


async def test_completion_acceptance_and_cycling() -> None:
    ready = anyio.Event()

    class Output(io.StringIO):
        def write(self, text: str) -> int:
            if 'Hello command' in text:
                ready.set()
            return super().write(text)

    async with editor(output=Output()) as (live, pipe, _):
        live.commands.register(Command(name='hello', description='Hello command', handler=lambda args: 'hi'))
        pipe.send_text('/he')
        await ready.wait()
        pipe.send_text('\t\t\n\n')
        assert await live.read() == '/hello'
        ready = anyio.Event()
        pipe.send_text('/hell')
        await ready.wait()
        pipe.send_text('\t\n')
        assert await live.read() == '/hello'
        live.feed('tab')
        live.feed('backtab')
        live.feed('escape')


async def test_literal_paths_attach_images_and_queue_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / 'shot.png'
    Image.new('RGB', (2, 2)).save(path)
    async with editor() as (live, _, _):
        live.feed('paste', str(path))
        caption, images = live.images.resolve(live.buffer.text)
        assert caption == '' and len(images) == 1
        for i in range(10):
            live.submit(f'message {i}')
        rows = '\n'.join(Text.from_ansi(row).plain for row in live.frame())
        assert '+6 more queued' in rows
        assert 'queued: 10' in rows


async def test_history_search_and_multiline_submission() -> None:
    async with editor() as (live, pipe, _):
        pipe.send_text('history entry\n')
        assert await live.read() == 'history entry'
        pipe.send_text('\x12history\n\n')
        assert await live.read() == 'history entry'
        pipe.send_text('first\x1b\rsecond\n')
        assert await live.read() == 'first\nsecond'


async def test_footer_warning_and_control_bytes_are_safe() -> None:
    async with editor() as (live, _, _):
        live.toolbar = lambda: [(theme.WARNING, 'warning\x1b[2J')]
        footer = live.frame()[-1]
        assert theme.sgr(theme.WARNING) in footer
        assert '\x1b[2J' not in footer
        assert r'\x1b[2J' in footer


@pytest.mark.parametrize('sequence', ['\x1b[13;2u', '\x1b[27;2;13~', '\x1b\r'])
async def test_shift_enter_inserts_newline_and_plain_enter_submits(sequence: str) -> None:
    async with editor() as (live, pipe, _):
        pipe.send_text(f'first{sequence}second\r')
        assert await live.read() == 'first\nsecond'
        assert live.queued_messages == ()
        assert live.buffer.text == ''


@pytest.mark.parametrize('colorterm', ['', 'truecolor'])
@pytest.mark.parametrize('width', [10, 80])
async def test_spinner_uses_tool_accent_without_coloring_border(
    monkeypatch: pytest.MonkeyPatch, colorterm: str, width: int
) -> None:
    monkeypatch.setenv('COLORTERM', colorterm)
    async with editor() as (live, _, _):
        live.console.size = (width, 24)

        async def operation() -> None:
            top = live.frame()[0]
            plain = Text.from_ansi(top).plain
            assert len(plain) == width
            if '⠋' in plain:
                assert f'{theme.sgr(theme.ACCENT)}⠋\x1b[0m{theme.sgr(theme.MUTED)}' in top
            else:
                assert theme.sgr(theme.ACCENT) not in top
            border_style = Text.from_ansi(top).get_style_at_offset(live.console, len(plain) - 1)
            assert not border_style.bold

        assert await live.interrupts.run(operation())
        assert theme.sgr(theme.ACCENT) not in live.frame()[0]


@pytest.mark.parametrize('dismiss', [False, True])
async def test_completion_refresh_keeps_popup_without_selecting_stale_results(
    monkeypatch: pytest.MonkeyPatch, dismiss: bool
) -> None:
    held = False
    started, release, finished = anyio.Event(), anyio.Event(), anyio.Event()

    async def compute(self: CompletionWorker, operation: Callable[[], list[Completion]]) -> list[Completion]:
        if held:
            started.set()
            await release.wait()
        result = operation()
        finished.set()
        return result

    monkeypatch.setattr(CompletionWorker, 'run', compute)
    terminal = SurfaceTerminal(width=80, height=24)
    async with editor(output=terminal) as (live, pipe, _):
        live.commands.register(Command(name='hello', description='Hello', handler=lambda args: 'hello'))
        live.output.write('transcript tail\n')
        pipe.send_text('/')
        await finished.wait()
        assert any('/help' in row for row in live.frame())
        height = len(live.frame())
        history = terminal.history.copy()
        held = True
        finished = anyio.Event()
        pipe.send_text('h')
        await started.wait()
        assert len(live.frame()) == height
        assert any('/help' in row for row in live.frame())
        live.feed('tab')
        live.feed('down')
        assert live.buffer.text == '/h'
        assert all('\x1b[7m' not in row for row in live.frame()[-3:-1])
        if dismiss:
            live.feed('escape')
        release.set()
        await finished.wait()
        assert terminal.history == history
        if dismiss:
            assert not any('/help' in row for row in live.frame())
        else:
            assert len(live.frame()) == height
            live.feed('tab')
            live.feed('enter')
            assert live.buffer.text == '/help'


async def test_suggestions_expand_the_prompt_box() -> None:
    ready = anyio.Event()

    class Output(io.StringIO):
        def write(self, text: str) -> int:
            if 'Hello command' in text:
                ready.set()
            return super().write(text)

    async with editor(output=Output()) as (live, pipe, _):
        live.commands.register(Command(name='hello', description='Hello command', handler=lambda args: 'hi'))
        pipe.send_text('/he')
        await ready.wait()
        plain = [Text.from_ansi(row).plain for row in live.frame()]
        dashes = [index for index, row in enumerate(plain) if row and set(row) == {'─'}]
        top, bottom = dashes[0], dashes[-1]
        inside = plain[top + 1 : bottom]
        assert inside[0].startswith('/he')
        assert inside[1].startswith('/help Help')
        assert inside[2].startswith('/hello Hello command')
        assert all('│' not in row and not row.startswith('>') for row in inside)
        assert all('help' not in row for row in plain[bottom + 1 :])
        live.feed('down')
        frame = live.frame()
        selected = next(row for row in frame if 'help' in row and '\x1b[7m' in row)
        assert selected.startswith('\x1b[7m') and selected.endswith('\x1b[0m')


async def test_completion_iteration_is_bounded_before_materializing() -> None:
    ready = anyio.Event()
    produced: list[int] = []

    class Output(io.StringIO):
        def write(self, text: str) -> int:
            if 'candidate0' in text:
                ready.set()
            return super().write(text)

    def candidates(args: list[str]) -> Iterator[str]:
        for index in range(100):
            produced.append(index)
            yield f'candidate{index}'
        raise AssertionError('completion consumed beyond its bound')

    async with editor(output=Output()) as (live, pipe, _):
        live.commands.register(
            Command(name='many', description='Many candidates', handler=lambda args: '', complete=candidates)
        )
        pipe.send_text('/many ')
        await ready.wait()
        assert produced == list(range(100))
        async with live.suspended():
            assert live.buffer.text == '/many '


async def test_recalled_history_gets_completions_and_completed_draft_survives_navigation() -> None:
    ready = anyio.Event()

    class Output(io.StringIO):
        def write(self, text: str) -> int:
            if '/help' in text:
                ready.set()
            return super().write(text)

    async with editor(output=Output()) as (live, pipe, _):
        live.buffer.history = ['/he']
        live.buffer.replace('original draft')
        pipe.send_text('\x1b[A')
        await ready.wait()
        live.feed('tab')
        assert live.buffer.text == '/help'
        live.feed('escape')
        live.feed('up')
        assert live.buffer.text == '/he'
        live.feed('down')
        assert live.buffer.text == '/help'


@pytest.mark.parametrize('menu', [False, True])
async def test_blocked_completion_does_not_hold_terminal_ownership(menu: bool) -> None:
    started = anyio.Event()
    release, finished = ThreadEvent(), ThreadEvent()
    output = io.StringIO()
    loop = asyncio.get_running_loop()

    def blocked(args: list[str]) -> list[str]:
        loop.call_soon_threadsafe(started.set)
        try:
            assert release.wait(timeout=5)
            return ['late result']
        finally:
            finished.set()

    try:
        async with editor(output=output) as (live, pipe, _):
            live.commands.register(
                Command(name='blocked', description='Blocked', handler=lambda args: '', complete=blocked)
            )
            pipe.send_text('/blocked ')
            await started.wait()
            if menu:
                async with live.suspended():
                    assert not finished.is_set()
                    assert live.buffer.text == '/blocked '
                    live.commands.unregister(['blocked'])
        assert not finished.is_set()
        assert output.getvalue().endswith('\x1b[?25h\x1b[?2026l')
        before = output.getvalue()
    finally:
        release.set()
        assert await in_worker(finished.wait, 5)
    assert output.getvalue() == before


async def test_completion_exception_is_reported_without_ending_editor() -> None:
    failed, recovered = anyio.Event(), anyio.Event()

    class Output(io.StringIO):
        def write(self, text: str) -> int:
            if 'Completion unavailable' in text:
                failed.set()
            if '/help' in text:
                recovered.set()
            return super().write(text)

    def broken(args: list[str]) -> list[str]:
        raise RuntimeError('broken provider')

    async with editor(output=Output()) as (live, pipe, _):
        live.commands.register(Command(name='broken', description='Broken', handler=lambda args: '', complete=broken))
        pipe.send_text('/broken ')
        await failed.wait()
        assert live.buffer.text == '/broken '
        assert any('broken provider' in row for row in live.frame())
        pipe.send_text('\x15/he')
        await recovered.wait()
        live.feed('tab')
        assert live.buffer.text == '/help'
        assert not any('Completion unavailable' in row for row in live.frame())


@pytest.mark.skipif(sys.platform == 'win32', reason='SIGWINCH is POSIX-only')
async def test_live_resize_signal_schedules_viewport_clear_without_losing_draft() -> None:
    cleared = anyio.Event()

    class Output(io.StringIO):
        def write(self, text: str) -> int:
            if '\x1b[2J' in text:
                cleared.set()
            return super().write(text)

    async with editor(output=Output()) as (live, _, _):
        live.buffer.replace('retained during resize signal')
        live.paint()
        signal.raise_signal(signal.SIGWINCH)
        await cleared.wait()
        assert live.buffer.text == 'retained during resize signal'
