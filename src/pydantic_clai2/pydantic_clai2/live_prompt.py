"""Pinned editor and scrollback ownership, without a PromptSession renderer."""

import asyncio
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import asynccontextmanager
from itertools import islice

import anyio
from PIL import Image
from prompt_toolkit.application.current import get_app_session
from prompt_toolkit.history import History
from rich.console import Console
from termflow.ansi.utils import visible_length  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.completion import CompleteEvent, Completion, Document  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.layout import truncate  # pyright: ignore[reportMissingTypeStubs]

from . import theme
from .commands import Commands, is_command_input
from .image_input import ImageInput, clipboard_images, pasted_paths, read_images
from .interrupts import Interrupts
from .prompt_buffer import PromptBuffer
from .prompt_completion import CompletionWorker
from .prompt_keys import PromptKeys
from .prompt_resize import resize_notifications
from .prompt_surface import PromptSurface
from .prompt_transcript import TranscriptBuffer
from .tool_output import terminal_text


class LivePrompt:
    """One terminal surface, one keyboard reader, sequential queued submissions."""

    def __init__(
        self,
        *,
        console: Console,
        commands: Commands,
        history: History,
        images: ImageInput,
        interrupts: Interrupts,
        toolbar: Callable[[], list[tuple[str, str]]],
        steer: Callable[[str], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
        transcript: TranscriptBuffer | None = None,
        chords: Mapping[str, Callable[[], str]] | None = None,
        pinned: Callable[[], str] = lambda: '',
    ) -> None:
        """Bind editing state, terminal ownership and per-session services.

        `chords` maps a two-key sequence such as `'ctrl-x ctrl-s'` to an action returning a
        footer notice. `pinned` returns an optional styled row painted above the footer.
        """
        self.console = console
        self.commands = commands
        self.history = history
        self.images = images
        self.interrupts = interrupts
        self.toolbar = toolbar
        self.steer = steer
        self.clock = clock
        self.chords = dict(chords or {})
        self.pinned = pinned
        self.notice = ''
        self._chord_prefix = ''
        self.buffer = PromptBuffer(history=list(reversed(list(history.load_history_strings()))))
        self.output = PromptSurface(output=console.file, size=lambda: console.size, transcript=transcript)
        self.keys = PromptKeys(
            source=get_app_session().input,
            feed=self.feed,
            eof=lambda: self.submit(EOFError()),
        )
        self._submissions: deque[str | KeyboardInterrupt | EOFError] = deque()
        self._submitted = asyncio.Event()
        self._suspended = False
        self._completions: list[Completion] = []
        self._selection = -1
        self._completion_pending = False
        self._completion_revision = 0
        self._complete = anyio.Event()
        self._completion_owner = anyio.Lock()
        self._completion_scope: anyio.CancelScope | None = None
        self._completion_worker = CompletionWorker()
        self._completion_error = ''
        self._opened = False

    @property
    def queued_messages(self) -> tuple[str, ...]:
        """Pending text, excluding control signals."""
        return tuple(item for item in self._submissions if isinstance(item, str))

    def submit(self, value: str | KeyboardInterrupt | EOFError) -> None:
        """Publish a submission without ending or replacing the editor."""
        self._submissions.append(value)
        self._submitted.set()
        self.paint()

    async def read(self) -> str:
        """Consume queued submissions in order."""
        await self._submitted.wait()
        value = self._submissions.popleft()
        if not self._submissions:
            self._submitted.clear()
        self.paint()
        if isinstance(value, BaseException):
            raise value
        return value

    def paste(self, text: str | None) -> None:
        """Attach clipboard/path images, or insert a literal bracketed paste."""
        self.images.retain([self.buffer.text, *self.queued_messages])
        self.images.notice = ''
        try:
            paths = pasted_paths(text) if text is not None else []
            if text is not None and not paths:
                self.buffer.insert(text, paste=True)
            else:
                self.buffer.insert(self.images.attach(read_images(paths) if text is not None else clipboard_images()))
        except (OSError, ValueError, NotImplementedError, Image.DecompressionBombError) as exc:
            self.images.notice = f'Image paste failed: {exc}. Linux requires wl-paste (Wayland) or xclip (X11).'

    def feed(self, key: str, data: str = '') -> None:
        """Route editing, completion and interrupts without rendering a widget tree."""
        self.notice = ''
        if not self._chord(key):
            self._route(key, data)
        self.paint()

    def _route(self, key: str, data: str) -> None:
        if key == 'ctrl-c':
            if not self.interrupts.cancel():
                self.buffer.replace('')
                self.buffer.search = None
                self.submit(KeyboardInterrupt())
        elif key == 'escape' and self.interrupts.active:
            self.interrupts.cancel(exit_on_repeat=False)
        elif key == 'ctrl-d':
            if self.buffer.text:
                self.buffer.edit('delete')
            else:
                self.submit(EOFError())
        elif key in ('paste', 'ctrl-v', 'alt-v'):
            self.paste(data if key == 'paste' else None)
        elif self.buffer.search is not None:
            self.buffer.search_key(key)
        elif key in ('tab', 'backtab'):
            self.complete(backwards=key == 'backtab')
        elif key == 'enter':
            self.accept()
        elif key == 'alt-enter':
            self.steer_queued()
        elif key in ('shift-enter', 'ctrl-j'):
            self.buffer.insert('\n')
        elif key in ('up', 'down') and self._completions:
            self.complete(backwards=key == 'up', accept_single=False)
        elif key == 'escape':
            self.dismiss_completions()
        else:
            self.buffer.edit(key)
        if key not in ('tab', 'backtab', 'escape') and (key not in ('up', 'down') or not self._completions):
            self.refresh_completions()

    def _chord(self, key: str) -> bool:
        """Consume a chord prefix or its completion; any other second key acts on its own."""
        if self._chord_prefix:
            chord, self._chord_prefix = f'{self._chord_prefix} {key}', ''
            if chord in self.chords:
                self.notice = self.chords[chord]()
                return True
            return False
        if any(chord.startswith(f'{key} ') for chord in self.chords):
            self._chord_prefix = key
            return True
        return False

    def accept(self) -> None:
        """Accept a completion or queue the nonempty draft."""
        if self._selection >= 0:
            self.accept_completion()
            return
        text = self.buffer.text.strip()
        if text:
            self.history.append_string(text)
            self.buffer.history.append(text)
            self.buffer.history_index = None
            self.buffer.replace('')
            self.submit(text)

    def steer_queued(self) -> None:
        """Promote the oldest follow-up without bypassing commands or control signals."""
        if not self._submissions or self.steer is None:
            return
        text = self._submissions[0]
        if not isinstance(text, str) or is_command_input(text) or not self.steer(text):
            return
        self._submissions.popleft()
        if not self._submissions:
            self._submitted.clear()

    def complete(self, *, backwards: bool, accept_single: bool = True) -> None:
        """Cycle suggestions, accepting a sole candidate immediately."""
        if self._completion_pending:
            return
        if len(self._completions) == 1 and accept_single:
            self._selection = 0
            self.accept_completion()
        elif self._completions:
            self._selection = (
                len(self._completions) - 1
                if backwards and self._selection < 0
                else (self._selection + (-1 if backwards else 1)) % len(self._completions)
            )

    def accept_completion(self) -> None:
        """Apply the selected Termflow completion to its original prefix."""
        item = self._completions[self._selection]
        start = max(0, self.buffer.cursor + item.start_position)
        self.buffer.replace_range(start, self.buffer.cursor, item.text)
        self.buffer.history_index = None
        self.dismiss_completions()

    def dismiss_completions(self) -> None:
        """Close the popup and invalidate any in-flight lookup."""
        self._completions = []
        self._selection = -1
        self._completion_pending = False
        self._completion_revision += 1
        self._completion_error = ''

    def refresh_completions(self) -> None:
        """Compute file/command suggestions off the input loop, ignoring stale results."""
        self._selection = -1
        self._completion_revision += 1
        self._completion_error = ''
        self._completion_pending = bool(self.buffer.text) and self.buffer.search is None
        # Retain the displayed rows until their replacements arrive. Removing
        # them here makes the terminal band shrink and grow on every key.
        if not self._completion_pending:
            self._completions = []
        self._complete.set()

    async def completion_loop(self) -> None:
        """Keep optional completion providers from blocking or terminating the shell."""
        while True:
            await self._complete.wait()
            self._complete = anyio.Event()
            text, cursor = self.buffer.text, self.buffer.cursor
            revision = self._completion_revision
            if not self._completion_pending:
                continue
            error = ''
            items: list[Completion] = []
            async with self._completion_owner:
                with anyio.CancelScope() as scope:
                    self._completion_scope = scope
                    try:
                        items = await self._completion_worker.run(
                            lambda: list(
                                islice(
                                    self.commands.get_completions(
                                        Document(text, cursor), CompleteEvent(text_inserted=True)
                                    ),
                                    100,
                                )
                            ),
                        )
                    except Exception as exc:  # noqa: BLE001 -- optional suggestions must not end a session.
                        error = f'Completion unavailable: {exc}'
                    finally:
                        self._completion_scope = None
                if scope.cancel_called:
                    continue
            if revision == self._completion_revision and (text, cursor) == (self.buffer.text, self.buffer.cursor):
                self._completion_pending = False
                self._completion_error = error
                self._completions = items
                self.paint()

    def frame(self) -> tuple[str, ...]:
        """Build the reserved rows; transcript contents are deliberately absent."""
        width, height = self.console.size
        width, height = max(1, width), max(2, height)
        muted, reset = theme.sgr(theme.MUTED), '\x1b[0m'
        if width < 6 or height < 6:
            return tuple(self.buffer.rows(width=width, limit=1))
        rows: list[str] = []
        queue_limit = max(1, height // 6)
        for text in self.queued_messages[:queue_limit]:
            label = 'Command' if is_command_input(text) else 'Follow-up'
            rows.append(muted + truncate(f'{label}: {" ".join(terminal_text(text).split())}', width) + reset)
        if len(self.queued_messages) > queue_limit:
            rows.append(muted + f'+{len(self.queued_messages) - queue_limit} more queued' + reset)
        title = ''
        if self.interrupts.active:
            spinner = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'[int(self.clock() * 10) % 10]
            title = truncate(f' Working {spinner} | Enter: queue | Alt+Enter: steer queued ', width)
            title = title.replace(spinner, f'{theme.sgr(theme.ACCENT)}{spinner}{reset}{muted}')
        rows.append(muted + title + '─' * max(0, width - visible_length(title)) + reset)
        # The box has no side borders and no prompt marker: the draft and the
        # suggestions are plain rows between the top and bottom rules, so no
        # row can drift out of alignment with the corners.
        # The pinned row only takes a spare row: `paint` keeps `height - 2` rows, and the title,
        # one draft row, the rule, and the footer come first.
        pinned = self.pinned() if height - len(rows) - 5 >= 1 else ''
        inner = max(1, height - len(rows) - 4 - bool(pinned))
        popup_want = min(6, len(self._completions))
        draft = self.buffer.rows(width=width, limit=max(1, min(height // 3, inner - popup_want)))
        rows.extend(draft)
        popup_limit = max(0, min(6, len(self._completions), inner - len(draft)))
        start = max(0, self._selection - popup_limit + 1)
        for index, item in enumerate(self._completions[start : start + popup_limit], start=start):
            line = truncate(
                ' '.join(terminal_text(f'{item.display or item.text}  {item.display_meta or ""}').split()), width
            )
            rows.append(('\x1b[7m' if index == self._selection else muted) + line + reset)
        rows.append(muted + '─' * width + reset)
        if pinned:
            rows.append(truncate(pinned, width) + reset)
        if self.buffer.search is not None:
            footer = f'reverse-i-search: {self.buffer.search}'
        else:
            notice = self.notice or self.images.notice or self._completion_error
            footer = (
                ' '.join(terminal_text(notice).split())
                if notice
                else ''.join(
                    (theme.sgr(style) if style else muted) + ' '.join(terminal_text(text).splitlines())
                    for style, text in self.toolbar()
                )
            )
            if self.queued_messages:
                footer += f' | queued: {len(self.queued_messages)}'
        rows.append(muted + truncate(footer, width) + reset)
        return tuple(rows)

    def paint(self) -> None:
        """Draw only when the editor owns the terminal."""
        if self._opened and not self._suspended:
            self.output.paint(self.frame())

    @asynccontextmanager
    async def suspended(self) -> AsyncGenerator[None]:
        """Hand input and terminal margins to a menu without losing the draft."""
        if self._suspended:
            yield
            return
        self._suspended = True
        self.keys.stop()
        self.dismiss_completions()
        if self._completion_scope is not None:
            self._completion_scope.cancel()
        try:
            async with self._completion_owner:
                await self.output.drain()
                self.output.release()
                yield
        finally:
            self._suspended = False
            self.refresh_completions()
            self.paint()
            self.keys.start()

    @asynccontextmanager
    async def opened(self) -> AsyncGenerator[None]:
        """Scope keyboard, resize/status polling and output ownership to the shell."""

        async def refresh() -> None:
            while True:
                self.paint()
                await anyio.sleep(0.1)

        loop = asyncio.get_running_loop()

        def resized() -> None:
            self.output.resize_notice()
            loop.call_soon_threadsafe(self.paint)

        original = self.console.file
        self._opened = True
        self.console.file = self.output
        try:
            with resize_notifications(resized):
                self.paint()
                self.keys.start()
                async with anyio.create_task_group() as tasks:
                    tasks.start_soon(refresh)
                    tasks.start_soon(self.completion_loop)
                    try:
                        yield
                        await self.output.drain()
                    finally:
                        tasks.cancel_scope.cancel()
        finally:
            self._opened = False
            self.keys.stop()
            self._completion_worker.close()
            self.console.file = original
            self.output.release()
