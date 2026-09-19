"""Input remains owned by the editor while turns and terminal widgets run."""

import asyncio
import io
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager

import anyio
import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application, create_app_session
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Always
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai.messages import BinaryContent
from rich.console import Console

from pydantic_clai2.image_input import ImageInput
from pydantic_clai2.interrupts import Interrupts
from pydantic_clai2.live_prompt import LivePrompt

TIMEOUT = 10


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@asynccontextmanager
async def editor(
    *, clock: Callable[[], float] = time.monotonic, images: ImageInput | None = None
) -> AsyncGenerator[tuple[LivePrompt, PipeInput, io.StringIO]]:
    output = io.StringIO()
    console = Console(file=output, force_terminal=True)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(TIMEOUT):
        prompt = PromptSession[str](key_bindings=images.bindings() if images is not None else None)
        live = LivePrompt(prompt, console, prepare=lambda: None, interrupts=Interrupts(), clock=clock)
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


@pytest.mark.parametrize('key', ['\x03', '\x1b'])
async def test_busy_interrupt_cancels_work_not_editor(key: str) -> None:
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
            edited = anyio.Event()

            def changed(buffer: Buffer) -> None:
                if buffer.text == 'Zretained':
                    edited.set()

            live.prompt.default_buffer.on_text_changed += changed
            pipe.send_text('retained\x1b[D\x1bbZ')
            await edited.wait()
            assert not cleaned.is_set()
            pipe.send_text(key)
            await stopped.wait()
            assert cleaned.is_set()
            assert live.prompt.app.is_running
            assert live.prompt.default_buffer.text == 'Zretained'
            pipe.send_text('\n')
            assert await live.read() == 'Zretained'


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


@pytest.mark.parametrize('outcome', ['completed', 'cancelled', 'failed'])
async def test_working_animation_is_on_top_border_without_changing_draft(outcome: str) -> None:
    now = 0.0
    started = anyio.Event()
    finish = anyio.Event()
    finished = anyio.Event()
    frames = [anyio.Event(), anyio.Event()]
    idle = anyio.Event()

    async with editor(clock=lambda: now) as (live, pipe, _):
        live.prompt.layout.current_window.height = 1
        live.prompt.layout.current_window.dont_extend_height = Always()
        painted: list[str] = []

        def rendered(app: Application[str]) -> None:
            screen = app.renderer.last_rendered_screen
            if screen is None:
                return
            rows = [''.join(cell.char for cell in row.values()) for row in screen.data_buffer.values()]
            text = '\n'.join(rows)
            if '> draft' not in text:
                return
            for index, spinner in enumerate(('⠋', '⠙')):
                if f'Working {spinner}' in text:
                    painted[:] = rows
                    frames[index].set()
            if finished.is_set() and 'Working' not in text:
                idle.set()

        live.prompt.app.after_render += rendered
        assert live.working_title() == []

        async def operation() -> None:
            started.set()
            await finish.wait()
            if outcome == 'failed':
                raise ValueError('test failure')

        async def run() -> None:
            if outcome == 'failed':
                with pytest.raises(ValueError, match='test failure'):
                    await live.interrupts.run(operation())
            else:
                assert await live.interrupts.run(operation()) == (outcome == 'completed')
            finished.set()
            live.prompt.app.invalidate()

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(run)
            await started.wait()
            pipe.send_text('draft')
            await frames[0].wait()
            top = next(index for index, row in enumerate(painted) if '┌' in row)
            bottom = next(index for index, row in enumerate(painted) if '└' in row)
            assert '┌─ Working ⠋' in painted[top]
            assert bottom - top == 2
            assert '> draft' in painted[top + 1]
            now = 0.1
            live.prompt.app.invalidate()
            await frames[1].wait()
            assert live.prompt.default_buffer.text == 'draft'
            if outcome == 'cancelled':
                pipe.send_text('\x1b')
            else:
                finish.set()
            await idle.wait()
        assert live.working_title() == []
        assert live.prompt.default_buffer.text == 'draft'


async def test_alt_v_during_work_attaches_without_cancelling(monkeypatch: pytest.MonkeyPatch) -> None:
    images = ImageInput()
    image = BinaryContent(data=b'png', media_type='image/png')
    monkeypatch.setattr('pydantic_clai2.image_input.clipboard_images', lambda: [image])
    async with editor(images=images) as (live, pipe, _):
        started = anyio.Event()
        stopped = anyio.Event()
        attached = anyio.Event()

        async def operation() -> None:
            started.set()
            await anyio.sleep_forever()

        async def work() -> None:
            assert not await live.interrupts.run(operation())
            stopped.set()

        def changed(buffer: Buffer) -> None:
            if '[image:' in buffer.text:
                attached.set()

        live.prompt.default_buffer.on_text_changed += changed
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(work)
            await started.wait()
            pipe.send_text('caption\x1bv')
            await attached.wait()
            assert not stopped.is_set()
            assert live.interrupts.active
            assert images.resolve(live.prompt.default_buffer.text) == ('caption', [image])
            pipe.send_text('\x1b')
            await stopped.wait()
            assert images.resolve(live.prompt.default_buffer.text) == ('caption', [image])


async def test_queue_labels_screenshot_paths_as_follow_ups() -> None:
    async with editor() as (live, _, _):
        for text in ('/tmp/shot.png', '/screenshot.PNG', '/help', '/unknown-command'):
            live.prompt.default_buffer.text = text
            live.prompt.default_buffer.validate_and_handle()
        assert live.queue_preview()[0][1].splitlines() == [
            'Follow-up: /tmp/shot.png',
            'Follow-up: /screenshot.PNG',
            'Command: /help',
            'Command: /unknown-command',
        ]
