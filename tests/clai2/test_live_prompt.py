"""Input remains owned by the editor while turns and terminal widgets run."""

import asyncio
import io
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import anyio
import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application, create_app_session
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from pydantic_clai2.interrupts import Interrupts
from pydantic_clai2.live_prompt import LivePrompt

TIMEOUT = 10


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@asynccontextmanager
async def editor() -> AsyncGenerator[tuple[LivePrompt, PipeInput, io.StringIO]]:
    output = io.StringIO()
    console = Console(file=output, force_terminal=True)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(TIMEOUT):
        prompt = PromptSession[str]()
        live = LivePrompt(prompt, console, prepare=lambda: None, interrupts=Interrupts())
        async with live.opened():
            yield live, pipe, output
        assert console.file is output
        assert not prompt.app.is_running


async def test_submissions_and_input_controls() -> None:
    async with editor() as (live, pipe, _):
        pipe.send_text('  \nfirst\nsecond\n')
        assert await live.read() == 'first'
        assert await live.read() == 'second'
        pipe.send_text('discard\x03')
        with pytest.raises(KeyboardInterrupt):
            await live.read()
        assert live.prompt.default_buffer.text == ''
        # Ctrl-D with text edits the buffer rather than exiting.
        pipe.send_text('keep\x01\x04\n')
        assert await live.read() == 'eep'
        pipe.send_text('\x04')
        with pytest.raises(EOFError):
            await live.read()


async def test_streaming_preview_and_menu_preserve_draft() -> None:
    async with editor() as (live, pipe, output):
        drafted = anyio.Event()

        def changed(buffer: Buffer) -> None:
            drafted.set()

        live.prompt.default_buffer.on_text_changed += changed
        pipe.send_text('unfinished draft')
        await drafted.wait()
        previewed = anyio.Event()

        def rendered(app: Application[str]) -> None:
            screen = app.renderer.last_rendered_screen
            if screen is not None:
                rows = [''.join(cell.char for cell in row.values()) for row in screen.data_buffer.values()]
                if any('streaming partial' in row for row in rows):
                    assert any('> unfinished draft' in row for row in rows)
                    previewed.set()

        live.prompt.app.after_render += rendered
        assert not live.console.file.isatty()
        live.console.file.write('streaming partial')
        live.console.file.flush()
        assert live.output.pending == 'streaming partial'
        assert 'streaming partial' not in output.getvalue()
        await previewed.wait()
        live.console.file.write(' line\nnext partial')
        await live.output.lines.join()
        assert 'streaming partial line\n' in output.getvalue()
        assert live.output.pending == 'next partial'
        async with live.suspended():
            assert 'next partial\n' in output.getvalue()
            async with live.suspended():
                live.console.print('menu output', markup=False)
                assert 'menu output' in output.getvalue()
            assert live.prompt.default_buffer.text == 'unfinished draft'
        pipe.send_text('\n')
        assert await live.read() == 'unfinished draft'
        live.console.file.write('final partial')
    assert output.getvalue().endswith('final partial\n')


async def test_closed_input_reports_eof() -> None:
    async with editor() as (live, pipe, _):
        pipe.close()
        with pytest.raises(EOFError):
            await live.read()


async def test_busy_interrupt_cancels_work_not_editor() -> None:
    async with editor() as (live, pipe, _):
        started = anyio.Event()
        stopped = anyio.Event()
        cleaned = anyio.Event()

        async def operation() -> None:
            try:
                started.set()
                await anyio.sleep_forever()
            finally:
                cleaned.set()

        async def work() -> None:
            assert not await live.interrupts.run(operation())
            stopped.set()

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(work)
            await started.wait()
            pipe.send_text('retained\x03')
            await stopped.wait()
            assert cleaned.is_set()
            assert live.prompt.app.is_running
            assert live.prompt.default_buffer.text == 'retained'
            pipe.send_text('\n')
            assert await live.read() == 'retained'


@pytest.mark.parametrize('menu', [False, True])
async def test_outer_cancellation_restores_console_and_drains_workers(menu: bool) -> None:
    original = io.StringIO()
    console = Console(file=original, force_terminal=True)
    before = asyncio.all_tasks()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(TIMEOUT):
        prompt = PromptSession[str]()
        live = LivePrompt(prompt, console, prepare=lambda: None, interrupts=Interrupts())
        with anyio.CancelScope() as scope:
            async with live.opened():
                console.print('pending output')
                if menu:
                    async with live.suspended():
                        scope.cancel()
                        await anyio.sleep_forever()
                else:
                    scope.cancel()
                    await anyio.sleep_forever()
        assert scope.cancelled_caught
        assert console.file is original
        assert not prompt.app.is_running
    assert asyncio.all_tasks() <= before


async def test_queue_preview_tracks_pending_messages_above_editor() -> None:
    async with editor() as (live, pipe, _):
        frames = [anyio.Event() for _ in range(3)]

        def rendered(app: Application[str]) -> None:
            screen = app.renderer.last_rendered_screen
            if screen is None:
                return
            rows = [''.join(cell.char for cell in row.values()) for row in screen.data_buffer.values()]
            text = '\n'.join(rows)
            if '> draft' not in text:
                return
            if 'Follow-up: first' in text and 'Command: /usage' in text:
                assert text.index('Follow-up: first') < text.index('Command: /usage') < text.index('> draft')
                frames[0].set()
            elif 'Follow-up:' not in text and 'Command: /usage' in text:
                frames[1].set()
            elif 'Follow-up:' not in text and 'Command:' not in text:
                frames[2].set()

        live.prompt.app.after_render += rendered
        pipe.send_text('first\n/usage\ndraft')
        await frames[0].wait()
        assert live.queued_messages == ('first', '/usage')
        assert await live.read() == 'first'
        await frames[1].wait()
        assert await live.read() == '/usage'
        await frames[2].wait()
        assert live.queued_messages == ()
        assert live.prompt.default_buffer.text == 'draft'


async def test_queue_preview_is_bounded_and_does_not_modify_messages() -> None:
    async with editor() as (live, pipe, _):
        live.console.size = (24, 9)
        messages = ['one\ntwo\x1b[31m', '界' * 40, 'third', '/usage']
        for message in messages:
            live.prompt.default_buffer.text = message
            live.prompt.default_buffer.validate_and_handle()
        lines = live.queue_preview()[0][1].splitlines()
        assert lines[0] == 'Follow-up: one two[31m'
        assert lines[1].startswith('Follow-up: 界') and lines[1].endswith('…')
        assert lines[2] == '+2 more queued'
        assert '\x1b' not in '\n'.join(lines)
        for message in messages:
            assert await live.read() == message
        assert live.queued_messages == ()
        drafted = anyio.Event()

        def changed(buffer: Buffer) -> None:
            drafted.set()

        live.prompt.default_buffer.on_text_changed += changed
        pipe.send_text('\x04draft')
        await drafted.wait()
        assert live.queued_messages == ()
        assert live.queue_preview()[0][1] == ''
        with pytest.raises(EOFError):
            await live.read()
