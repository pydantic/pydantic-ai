"""Pinned editor and scrollback ownership, without a PromptSession renderer."""

import asyncio
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from itertools import islice

import anyio
from PIL import Image
from prompt_toolkit.application.current import get_app_session
from prompt_toolkit.history import History
from rich.console import Console
from termflow.ansi.utils import visible_length
from termflow.tui.completion import CompleteEvent, Completion, Document
from termflow.tui.layout import truncate

from pydantic_clai2.cli.shell_passthrough import shell_command
from pydantic_clai2.commands import Commands, expand_bare_command, is_command_input
from pydantic_clai2.ui import telemetry
from pydantic_clai2.ui.menus.menu_worker import holding_output
from pydantic_clai2.ui.prompt.image_input import ImageInput, clipboard_images, pasted_paths, read_images
from pydantic_clai2.ui.prompt.interrupts import Interrupts
from pydantic_clai2.ui.prompt.prompt_buffer import PromptBuffer
from pydantic_clai2.ui.prompt.prompt_completion import CompletionWorker
from pydantic_clai2.ui.prompt.prompt_keys import PromptKeys
from pydantic_clai2.ui.prompt.prompt_resize import resize_notifications
from pydantic_clai2.ui.prompt.prompt_surface import TRANSCRIPT_KEYS, PromptSurface
from pydantic_clai2.ui.prompt.prompt_transcript import TranscriptBuffer
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering.spinners import BUILTIN_SPINNERS, DEFAULT_SPINNER, Spinner
from pydantic_clai2.ui.rendering.tool_output import terminal_text


@dataclass(eq=False)
class _Queued:
    """A queued prompt compared by identity, so an edit finds it even after the queue shifts."""

    text: str
    recorded: str = ''
    """The raw draft `accept` saved to history for this prompt, before command expansion."""


class PromptWakeup(Exception):
    """An automated continuation is ready; the editor draft is not a submission."""


class PromptRewind(Exception):
    """Open the rewind picker without submitting or clearing the draft."""


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
        run_now: Callable[[str], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
        transcript: TranscriptBuffer | None = None,
        chords: Mapping[str, Callable[[], str]] | None = None,
        pinned: Callable[[], str] = lambda: '',
        spinner: Callable[[], Spinner] = lambda: BUILTIN_SPINNERS[DEFAULT_SPINNER],
        panel: Callable[[str], Sequence[str]] = lambda _: (),
    ) -> None:
        """Bind editing state, terminal ownership and per-session services.

        `chords` maps a key or sequence such as `'ctrl-b'` or `'ctrl-x ctrl-s'` to an action returning a
        footer notice. `pinned` returns an optional styled row painted above the footer.
        `run_now` may take an accepted draft instead of queueing it, returning whether it did.
        `spinner` returns the working animation; it is read on every frame, so a new choice shows at once.
        `panel` receives the current spinner frame and returns styled rows painted above the queue,
        such as running forks; it is read on every repaint, including between turns.
        """
        self.console = console
        self.commands = commands
        self.history = history
        self.images = images
        self.interrupts = interrupts
        self.toolbar = toolbar
        self.steer = steer
        self.run_now = run_now
        self.clock = clock
        self.chords = dict(chords or {})
        self.pinned = pinned
        self.spinner = spinner
        self.panel = panel
        self.notice = ''
        self._chord_prefix = ''
        self._last_escape: float | None = None
        self.buffer = PromptBuffer(history=list(reversed(list(history.load_history_strings()))))
        self.output = PromptSurface(output=console.file, size=lambda: console.size, transcript=transcript)
        self.keys = PromptKeys(
            source=get_app_session().input,
            feed=self.feed,
            eof=lambda: self.submit(EOFError()),
        )
        self._submissions: deque[_Queued | KeyboardInterrupt | EOFError | PromptRewind] = deque()
        # The queued prompt the draft would rewrite on Enter, and the queue as a recall walk found it.
        self._editing: _Queued | None = None
        self._recall_queue: list[_Queued] = []
        self._recall_target: _Queued | None = None
        self._search_target: _Queued | None = None
        self._submitted = asyncio.Event()
        self._wake_pending = False
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
        return tuple(item.text for item in self._queued())

    def _queued(self) -> list[_Queued]:
        return [item for item in self._submissions if isinstance(item, _Queued)]

    def submit(self, value: str | KeyboardInterrupt | EOFError | PromptRewind) -> None:
        """Publish a submission without ending or replacing the editor."""
        self._enqueue(_Queued(value) if isinstance(value, str) else value)

    def _enqueue(self, value: _Queued | KeyboardInterrupt | EOFError | PromptRewind) -> None:
        self._submissions.append(value)
        self._submitted.set()
        self.paint()

    def wake(self) -> None:
        """Wake an idle reader once, without changing its draft, history, or queue."""
        self._wake_pending = True
        self._submitted.set()

    def _discard(self, entry: _Queued) -> None:
        self._submissions.remove(entry)
        if not self._submissions and not self._wake_pending:
            self._submitted.clear()

    async def read(self) -> str:
        """Consume queued submissions in order."""
        await self._submitted.wait()
        if not self._submissions:
            self._wake_pending = False
            self._submitted.clear()
            raise PromptWakeup
        value = self._submissions.popleft()
        if not self._submissions and not self._wake_pending:
            self._submitted.clear()
        self.paint()
        if isinstance(value, BaseException):
            raise value
        return value.text

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
        if key in TRANSCRIPT_KEYS:
            # Leave the draft alone, and the notice unless a drag copied text.
            if self.output.transcript_key(key, data):
                self.notice = 'Copied the selection to the clipboard.'
                self.paint()
            return
        self.notice = ''
        if key != 'escape':
            self._last_escape = None
        if not self._chord(key):
            self._route(key, data)
        self.paint()

    def _route(self, key: str, data: str) -> None:
        cycles = key in ('up', 'down') and self._arrows_cycle_completions()
        if key == 'ctrl-c':
            self.interrupt()
        elif key == 'escape':
            self.escape()
        elif key == 'ctrl-d':
            if self.buffer.text:
                self.buffer.edit('delete')
            else:
                self.submit(EOFError())
        elif key == 'ctrl-l':
            # As in Claude Code: a clean screen; the conversation, draft, and queue stay.
            self.output.clear(keep_current=self.interrupts.active)
        elif key in ('paste', 'ctrl-v', 'alt-v'):
            self.paste(data if key == 'paste' else None)
        elif key == 'ctrl-r' or self.buffer.search is not None:
            self.search(key)
        elif key in ('tab', 'backtab'):
            self.complete(backwards=key == 'backtab')
        elif key == 'enter':
            self.accept()
        elif key == 'alt-enter':
            self.steer_draft() if self.buffer.text.strip() else self.steer_queued()
        elif key in ('shift-enter', 'ctrl-j'):
            self.buffer.insert('\n')
        elif cycles:
            self.complete(backwards=key == 'up', accept_single=False)
        elif key in ('up', 'down'):
            self.recall(backwards=key == 'up')
        else:
            self.buffer.edit(key)
        if key not in ('tab', 'backtab', 'escape') and not cycles:
            self.refresh_completions()

    def escape(self) -> None:
        """Cancel or dismiss first; only consecutive idle presses request a rewind."""
        previous, self._last_escape = self._last_escape, None
        if self.interrupts.active:
            cancelled = self.interrupts.cancel(exit_on_repeat=False)
            telemetry.record('prompt interrupt', key='escape', cancelled_turn=cancelled)
        elif self.buffer.search is not None:
            self.search('escape')
        elif self._completions or self._completion_pending:
            self.dismiss_completions()
        elif not self._submissions:
            now = self.clock()
            if previous is not None and now - previous <= 0.5:
                self.submit(PromptRewind())
            else:
                self._last_escape = now
                self.notice = 'Press Esc again to rewind the conversation.'

    def restore_draft(self, text: str) -> None:
        """Replace the draft after a rewind, without editing a queued submission."""
        self._editing = None
        self.buffer.search = None
        self.buffer.history_index = None
        self.buffer.replace(text)
        self.refresh_completions()

    def _arrows_cycle_completions(self) -> bool:
        """Whether Up/Down move through the popup rather than the draft and history.

        A history walk owns the arrows, so recalling a `/command` shows its suggestions
        without trapping the walk on them. Tab picks a suggestion and hands the arrows to
        the popup; editing the recalled text ends the walk, as does typing a fresh draft.
        """
        return bool(self._completions) and (self.buffer.history_index is None or self._selection >= 0)

    def search(self, key: str) -> None:
        """Search history; a picked match is a new prompt, not an edit of a recalled queued one."""
        if self.buffer.search is None:
            self._search_target = self._editing
        self.buffer.edit(key)
        # Cancelling (or finding nothing) leaves the original text, and with it the edit target.
        self._editing = self._search_target if self.buffer.text == self.buffer.search_original else None

    def interrupt(self) -> None:
        """Cancel running work, or else drop the draft and signal the reader."""
        cancelled = self.interrupts.cancel()
        telemetry.record('prompt interrupt', key='ctrl-c', cancelled_turn=cancelled)
        if not cancelled:
            self.buffer.replace('')
            self.buffer.search = None
            self._editing = None
            self.submit(KeyboardInterrupt())

    def _chord(self, key: str) -> bool:
        """Consume a chord prefix or its completion; any other second key acts on its own."""
        if key in self.chords:
            self.notice = self.chords[key]()
            return True
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
        """Accept a completion that changes the draft, or else queue the nonempty draft."""
        if self._selection >= 0:
            item = self._completions[self._selection]
            if self.buffer.text[self._completion_start(item) : self.buffer.cursor] != item.text:
                self.accept_completion()
                return
            # The highlighted command is already typed in full, so Enter runs it.
            self.dismiss_completions()
        text = self.buffer.text.strip()
        target, self._editing = self._editing, None
        if target is not None and target not in self._submissions:
            # The run took the prompt while it was being edited, so the edit becomes a new follow-up.
            target = None
        if not text and target is None:
            return
        recalled = self.buffer.history_index is not None
        self.buffer.history_index = None
        self.buffer.replace('')
        if target is not None and not text:
            self._discard(target)
            telemetry.record('prompt submitted', route='discarded queued')
            return
        command = expand_bare_command(text)
        if target is not None and command == target.text:
            return
        self.output.view.follow()
        self.history.append_string(text)
        self.buffer.history.append(text)
        if self.run_now is not None and self.run_now(command):
            route = 'run now'
            if target is not None:
                self._discard(target)
        elif target is not None:
            route = 'edited queued'
            target.text, target.recorded = command, text
        else:
            route = 'queued' if self.interrupts.active else 'submitted'
            self._enqueue(_Queued(command, recorded=text))
        telemetry.record('prompt submitted', route=route, recalled=recalled, **_submission(command, self.commands))

    def recall(self, *, backwards: bool) -> None:
        """Walk queued prompts, newest first, before command history.

        The queue holds the most recent input, so Up reaches it before older history, as in
        shell history. A recalled queued prompt keeps its place in the queue: Enter rewrites
        it, and clearing the draft before Enter removes it.
        """
        if self.buffer.history_index is None:
            self._recall_queue = self._queued()
            self._recall_target = self._editing
        self.buffer.vertical(
            backwards=backwards,
            queued=tuple(entry.text for entry in self._recall_queue),
            recorded=tuple(entry.recorded for entry in self._recall_queue),
        )
        offset = self.buffer.recall_offset
        if offset == 0:
            self._editing = self._recall_target
        elif offset is not None:
            self._editing = self._recall_queue[offset] if -offset <= len(self._recall_queue) else None

    def _steered(self, text: str) -> bool:
        """Hand prompt text to the active run; commands and shell lines never steer."""
        command = expand_bare_command(text)
        return (
            self.steer is not None
            and not is_command_input(command)
            and shell_command(command) is None
            and self.steer(command)
        )

    def steer_draft(self) -> None:
        """Send the draft to the active run now, or accept it as Enter would when it cannot steer."""
        text = self.buffer.text.strip()
        if not self._steered(text):
            self.accept()
            return
        target, self._editing = self._editing, None
        # Steered text joins the running turn, so show its response, as submitting does.
        self.output.view.follow()
        self.history.append_string(text)
        self.buffer.history.append(text)
        self.buffer.history_index = None
        self.buffer.replace('')
        if target is not None and target in self._submissions:
            # The steered text was an edit of a queued prompt, so it no longer waits for its own turn.
            self._discard(target)
        telemetry.record('prompt steer', steered=True, source='draft')

    def steer_queued(self) -> None:
        """Promote the oldest follow-up without bypassing commands, shell lines, or control signals."""
        if not self._submissions or self.steer is None:
            return
        head = self._submissions[0]
        if not isinstance(head, _Queued) or not self._steered(head.text):
            telemetry.record('prompt steer', steered=False, source='queue')
            return
        self.output.view.follow()
        self._discard(head)
        telemetry.record('prompt steer', steered=True, source='queue')

    def complete(self, *, backwards: bool, accept_single: bool = True) -> None:
        """Cycle suggestions, accepting a sole candidate immediately, or look them up if none are shown."""
        if self._completion_pending:
            return
        if not self._completions:
            # Escape dismissed the popup (or nothing was looked up yet): Tab asks again.
            self.refresh_completions()
        elif len(self._completions) == 1 and accept_single:
            self._selection = 0
            self.accept_completion()
        else:
            self._selection = (
                len(self._completions) - 1
                if backwards and self._selection < 0
                else (self._selection + (-1 if backwards else 1)) % len(self._completions)
            )

    def accept_completion(self) -> None:
        """Apply the selected Termflow completion to its original prefix."""
        item = self._completions[self._selection]
        telemetry.record(
            'completion accepted',
            source='command' if is_command_input(self.buffer.text) else 'path',
            candidates=len(self._completions),
        )
        self.buffer.replace_range(self._completion_start(item), self.buffer.cursor, item.text)
        self.dismiss_completions()

    def _completion_start(self, item: Completion) -> int:
        """Where the fragment `item` replaces begins in the draft."""
        return max(0, self.buffer.cursor + item.start_position)

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
                self._selection = 0 if items and self._highlights_best_match(text[:cursor]) else -1
                self.paint()

    def _highlights_best_match(self, typed: str) -> bool:
        """Whether the first suggestion starts highlighted, as if Tab had picked it.

        Only while a command name is typed, where the registry ranks the best match first.
        Argument candidates keep their provider's order, so Enter still submits a typed
        argument, and a history walk keeps the arrows (see `_arrows_cycle_completions`).
        """
        return (
            self.buffer.history_index is None and is_command_input(typed) and not any(char.isspace() for char in typed)
        )

    def frame(self) -> tuple[str, ...]:
        """Build the reserved rows; transcript contents are deliberately absent."""
        width, height = self.console.size
        width, height = max(1, width), max(2, height)
        muted, reset = theme.sgr(theme.MUTED), '\x1b[0m'
        if width < 6 or height < 6:
            return tuple(self.buffer.rows(width=width, limit=1))
        # `paint` keeps `height - 2` rows; the title, one draft row, the rule, and the footer need four.
        room = height - 6
        limit = max(1, height // 6)
        panel = [truncate(row, width) + reset for row in self.panel(self.spinner().frame(self.clock()))]
        rows = _capped(panel, limit=limit, room=room, more=muted + '+{} more' + reset)
        queued = [
            muted
            + truncate(
                f'{"Command" if is_command_input(entry.text) else "Follow-up"}'
                f'{" (editing)" if entry is self._editing else ""}: {" ".join(terminal_text(entry.text).split())}',
                width,
            )
            + reset
            for entry in self._queued()
        ]
        rows += _capped(queued, limit=limit, room=room - len(rows), more=muted + '+{} more queued' + reset)
        title = ''
        if self.interrupts.active:
            head, glyph = ' Working ', self.spinner().frame(self.clock())
            # Queue and steer hints only matter once there is something to send, matching pi and Claude Code.
            hints = '| Enter: queue | Alt+Enter: steer ' if self.queued_messages or self.buffer.text.strip() else ''
            title = truncate(f'{head}{glyph} {hints}', width)
            if title.startswith(head + glyph):
                title = f'{head}{theme.sgr(theme.ACCENT)}{glyph}{reset}{muted}{title[len(head + glyph) :]}'
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
        self._last_escape = None
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
                if self._suspended:
                    self.output.refresh()
                else:
                    self.paint()
                # A spinner faster than the status poll gets a repaint per frame, but only while it shows.
                await anyio.sleep(min(0.1, self.spinner().interval) if self.interrupts.active else 0.1)

        loop = asyncio.get_running_loop()

        def resized() -> None:
            self.output.resize_notice()
            loop.call_soon_threadsafe(self.paint)

        original = self.console.file
        self._opened = True
        self.console.file = self.output
        try:
            with resize_notifications(resized), holding_output(self.output.held):
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
            self.output.restore()


def _capped(rows: list[str], *, limit: int, room: int, more: str) -> list[str]:
    """Up to `limit` rows plus a `more` count for the rest, never taller than `room`."""
    if len(rows) <= min(limit, room):
        return rows
    if room <= 0:
        return []
    shown = rows[: min(limit, room - 1)]
    return [*shown, more.format(len(rows) - len(shown))]


def _submission(text: str, commands: Commands) -> dict[str, telemetry.Attribute]:
    """What kind of input was submitted, its length, and, for a prompt the sink records content of, its text.

    Only a prompt's text is recorded, as `prompt`, since agent spans already carry it. A `!` line never reaches
    the agent and can hold a secret such as `!export KEY=...`, so it records only its kind and length. A command
    records only its registered name, never its arguments: any plugin can add a command, and arguments such as
    `/plugins add` settings JSON or an MCP server's headers can hold a secret CLAI cannot recognize.
    """
    if is_command_input(text):
        name = text.split(maxsplit=1)[0].removeprefix('/')
        return {'kind': 'command', 'command': name if name in commands else 'unknown', 'chars': len(text)}
    if shell_command(text) is not None:
        return {'kind': 'shell', 'chars': len(text)}
    return {'kind': 'prompt', 'chars': len(text), **telemetry.prompt_text(text)}
