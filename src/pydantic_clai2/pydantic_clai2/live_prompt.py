"""An editable prompt with sequential submissions and output above the editor."""

import asyncio
import io
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import IO

import anyio
from prompt_toolkit import PromptSession
from prompt_toolkit.application import in_terminal
from prompt_toolkit.application.current import set_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition, is_done
from prompt_toolkit.formatted_text import ANSI, FormattedText, to_formatted_text
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent, merge_key_bindings
from prompt_toolkit.layout import ConditionalContainer, FormattedTextControl, HSplit, VSplit, Window
from rich.console import Console
from rich.text import Text

from . import theme
from .commands import is_command_input
from .interrupts import Interrupts

_WORKING_FRAMES = ('⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏')


class PromptOutput(io.StringIO):
    """Buffer complete lines so a redraw cannot overwrite a partially streamed line."""

    def __init__(self, original: IO[str]) -> None:
        """Retain the console destination rather than replacing the process streams."""
        super().__init__()
        self.original = original
        self.pending = ''
        self.lines: asyncio.Queue[str] = asyncio.Queue()
        self.direct = False

    def isatty(self) -> bool:
        """Preserve terminal detection for Rich and Termflow."""
        return self.original.isatty()

    def write(self, text: str) -> int:
        """Queue complete lines; menu output bypasses the editor."""
        if self.direct:
            return self.original.write(text)
        self.pending += text
        before, separator, self.pending = self.pending.rpartition('\n')
        if separator:
            self.lines.put_nowait(before + separator)
        return len(text)

    def flush(self) -> None:
        """Leave incomplete streaming lines buffered until a boundary."""
        if self.direct:
            self.original.flush()

    async def drain(self) -> None:
        """Finish an unterminated line at a turn or menu boundary."""
        if self.pending:
            self.lines.put_nowait(self.pending + '\n')
            self.pending = ''
        await self.lines.join()

    async def run(self) -> None:
        """Serialize writes with prompt-toolkit terminal ownership."""
        while True:
            text = await self.lines.get()
            try:
                async with in_terminal():
                    self.original.write(text)
                    self.original.flush()
            finally:
                self.lines.task_done()


class LivePrompt:
    """Own the editor and output worker until the shell exits, including cancellation."""

    def __init__(
        self,
        prompt: PromptSession[str],
        console: Console,
        *,
        prepare: Callable[[], None],
        interrupts: Interrupts,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Keep all input and output resources scoped to one shell."""
        self.prompt = prompt
        self.console = console
        self.prepare = prepare
        self.interrupts = interrupts
        self._clock = clock
        self._submissions: deque[str | KeyboardInterrupt | EOFError] = deque()
        self._submitted = asyncio.Event()
        self.output = PromptOutput(console.file)

    def working_title(self) -> FormattedText:
        """Animate the top border without adding a row to the editable area."""
        if not self.interrupts.active:
            return FormattedText([])
        frame = _WORKING_FRAMES[int(self._clock() * 10) % len(_WORKING_FRAMES)]
        return FormattedText([(theme.MUTED, ' Working '), (theme.ACCENT, frame), ('', ' ')])

    async def read(self) -> str:
        """Consume submissions in order without overlapping agent runs."""
        await self._submitted.wait()
        value = self._submissions.popleft()
        if not self._submissions:
            self._submitted.clear()
        self.prompt.app.invalidate()
        if isinstance(value, BaseException):
            raise value
        return value

    def _submit(self, value: str | KeyboardInterrupt | EOFError) -> None:
        self._submissions.append(value)
        self._submitted.set()
        self.prompt.app.invalidate()

    @property
    def queued_messages(self) -> tuple[str, ...]:
        """Pending text in execution order, excluding input control signals."""
        return tuple(value for value in self._submissions if isinstance(value, str))

    def queue_preview(self) -> FormattedText:
        """Show compact follow-up previews without letting a large queue fill the terminal."""
        messages = self.queued_messages
        limit = max(1, self.console.height // 3 - 1)
        lines: list[str] = []
        for message in messages[:limit]:
            label = 'Command' if is_command_input(message) else 'Follow-up'
            printable = ''.join(char for char in message if char.isprintable() or char.isspace())
            text = Text(f'{label}: {" ".join(printable.split())}')
            text.truncate(max(1, self.console.width), overflow='ellipsis')
            lines.append(text.plain)
        if len(messages) > limit:
            lines.append(f'+{len(messages) - limit} more queued')
        return FormattedText([(theme.MUTED, '\n'.join(lines))])

    def accept(self, buffer: Buffer) -> bool:
        """Submit the current buffer without ending the editor application."""
        text = buffer.text.strip()
        if text:
            self._submit(text)
        return False

    def bindings(self) -> KeyBindings:
        """Keep interrupts aimed at the turn rather than the editor task."""
        keys = KeyBindings()

        @keys.add('c-c')
        def interrupt(event: KeyPressEvent) -> None:
            if not self.interrupts.cancel():
                event.current_buffer.reset()
                self._submit(KeyboardInterrupt())

        @keys.add('escape', filter=Condition(lambda: self.interrupts.active))
        def escape(event: KeyPressEvent) -> None:
            self.interrupts.cancel(exit_on_repeat=False)

        @keys.add('c-d')
        def eof(event: KeyPressEvent) -> None:
            if event.current_buffer.text:
                event.current_buffer.delete()
            else:
                self._submit(EOFError())

        return keys

    @asynccontextmanager
    async def suspended(self) -> AsyncGenerator[None]:
        """Let a command or question menu own input without losing the draft."""
        if self.output.direct:
            yield
            return
        await self.output.drain()
        async with in_terminal():
            self.output.direct = True
            try:
                yield
            finally:
                self.output.direct = False

    @asynccontextmanager
    async def opened(self) -> AsyncGenerator[None]:
        """Scope both workers to the shell and restore the console on every exit."""
        started = anyio.Event()
        container = self.prompt.layout.container
        assert isinstance(container, HSplit)
        editor = container.children[0]
        assert isinstance(editor, ConditionalContainer)
        frame = editor.content
        assert isinstance(frame, HSplit)
        original_border = frame.children[0]
        working_border = VSplit(
            [
                Window(width=1, char='┌'),
                Window(width=1, char='─'),
                Window(FormattedTextControl(self.working_title), dont_extend_width=True),
                Window(char='─'),
                Window(width=1, char='┐'),
            ],
            height=1,
            style='class:frame.border',
        )
        preview = ConditionalContainer(
            Window(FormattedTextControl(lambda: ANSI(self.output.pending)), dont_extend_height=True),
            filter=Condition(lambda: bool(self.output.pending)) & ~is_done,
        )
        queue_preview = ConditionalContainer(
            Window(FormattedTextControl(self.queue_preview), dont_extend_height=True),
            filter=Condition(lambda: bool(self.queued_messages)) & ~is_done,
        )
        toolbar = self.prompt.bottom_toolbar
        accept_handler = self.prompt.default_buffer.accept_handler

        def prepare() -> None:
            self.prepare()
            frame.children[0] = working_border
            container.children[0:0] = [preview, queue_preview]
            self.prompt.default_buffer.accept_handler = self.accept
            started.set()

        async def edit() -> None:
            try:
                await self.prompt.prompt_async(
                    '> ',
                    pre_run=prepare,
                    key_bindings=merge_key_bindings(
                        [self.prompt.key_bindings, self.bindings()] if self.prompt.key_bindings else [self.bindings()]
                    ),
                    handle_sigint=False,
                    show_frame=~is_done & Condition(lambda: self.console.width >= 4 and self.console.height >= 6),
                    refresh_interval=0.1,
                    bottom_toolbar=lambda: [
                        *to_formatted_text(toolbar),
                        ('', f' | queued: {len(self.queued_messages)}') if self.queued_messages else ('', ''),
                    ],
                )
            except EOFError:
                self._submit(EOFError())

        original = self.console.file
        with set_app(self.prompt.app):
            async with anyio.create_task_group() as workers:
                workers.start_soon(edit)
                await started.wait()
                self.console.file = self.output
                workers.start_soon(self.output.run)
                try:
                    yield
                    await self.output.drain()
                finally:
                    self.console.file = original
                    self.prompt.bottom_toolbar = toolbar
                    self.prompt.default_buffer.accept_handler = accept_handler
                    frame.children[0] = original_border
                    container.children.remove(preview)
                    container.children.remove(queue_preview)
                    workers.cancel_scope.cancel()
