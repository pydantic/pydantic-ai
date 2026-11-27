"""Pinned editor and scrollback ownership, without a PromptSession renderer."""

import asyncio
import math
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from itertools import islice
from typing import Literal

import anyio
from PIL import Image
from prompt_toolkit.application.current import get_app_session
from prompt_toolkit.history import History
from rich.console import Console, ConsoleDimensions
from termflow.ansi.utils import visible_length
from termflow.tui.completion import CompleteEvent, Completion, Document
from termflow.tui.layout import truncate

from pydantic_ai_pax.commands import Commands, expand_bare_command, is_command_input
from pydantic_ai_pax.config.input_bindings import (
    BusyDelivery,
    InputAction,
    InputBindings,
    KeyboardLabels,
    canonical_key,
)
from pydantic_ai_pax.ui import telemetry
from pydantic_ai_pax.ui.menus.menu_worker import holding_output
from pydantic_ai_pax.ui.prompt.image_input import ATTACHED, ImageInput, clipboard_images, pasted_paths, read_images
from pydantic_ai_pax.ui.prompt.input_setup import KeyTest
from pydantic_ai_pax.ui.prompt.interrupts import Interrupts
from pydantic_ai_pax.ui.prompt.prompt_buffer import PromptBuffer
from pydantic_ai_pax.ui.prompt.prompt_completion import CompletionWorker
from pydantic_ai_pax.ui.prompt.prompt_keys import PromptKeys, ctrl_enter_steers
from pydantic_ai_pax.ui.prompt.prompt_questions import PromptQuestions
from pydantic_ai_pax.ui.prompt.prompt_queue import QueuedPrompt, can_steer, input_kind
from pydantic_ai_pax.ui.prompt.prompt_resize import resize_notifications
from pydantic_ai_pax.ui.prompt.prompt_selection import LINK_END, LINK_START, hyperlink
from pydantic_ai_pax.ui.prompt.prompt_surface import TRANSCRIPT_KEYS, PromptSurface
from pydantic_ai_pax.ui.prompt.prompt_transcript import TranscriptBuffer
from pydantic_ai_pax.ui.rendering import theme
from pydantic_ai_pax.ui.rendering.duration import format_duration
from pydantic_ai_pax.ui.rendering.spinners import BUILTIN_SPINNERS, DEFAULT_SPINNER, Spinner
from pydantic_ai_pax.ui.rendering.status import LinkText
from pydantic_ai_pax.ui.rendering.tool_output import terminal_text

IDLE_REPAINT = 1.0
"""Seconds between timed repaints of an idle editor, and how long it keeps repainting fast after animating.

Timed repaints pick up status that no key, output, or resize announces, such as a plugin's status
segment. They run every `ANIMATED_REPAINT` seconds while a turn runs or the panel shows forks or tasks;
an idle editor sleeps between them.
"""

ANIMATED_REPAINT = 0.1
"""Seconds between timed repaints while the editor animates, unless the spinner is faster."""


class PromptWakeup(Exception):
    """An automated continuation is ready; the editor draft is not a submission."""


class PromptRewind(Exception):
    """Open the rewind picker without submitting or clearing the draft."""


_Submission = QueuedPrompt | KeyboardInterrupt | EOFError | PromptRewind

SendRoute = Literal['save', 'choose-save', 'question', 'choose', 'send', 'after-turn', 'next-checkpoint']
"""What Send does with the draft now; each caller words it for its own surface."""


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
        queue_changed: Callable[[], None] = lambda: None,
        bindings: Callable[[], InputBindings] = InputBindings,
        keyboard_labels: Callable[[], KeyboardLabels] = lambda: 'neutral',
        busy_delivery: Callable[[], BusyDelivery] = lambda: 'after-turn',
        remember_busy_delivery: Callable[[BusyDelivery], object] = lambda _: None,
        input_changed: Callable[[], bool] = lambda: False,
        queue_in_toolbar: Callable[[], bool] | None = None,
        full_screen: bool = False,
    ) -> None:
        """Bind editing state, terminal ownership and per-session services.

        `chords` maps a key or sequence such as `'ctrl-b'` or `'ctrl-x ctrl-s'` to an action returning a
        footer notice. `pinned` returns an optional styled row painted above the footer.
        `run_now` may take an accepted draft instead of queueing it, returning whether it did.
        `spinner` returns the working animation; it is read on every frame, so a new choice shows at once.
        `panel` receives the current spinner frame and returns styled rows painted above the queue,
        such as running forks; it is read on every repaint, including between turns.
        `queue_changed` is told whenever a prompt joins, leaves, or is rewritten in the queue.
        `queue_in_toolbar=None` appends the queue count here; a callback leaves it to the toolbar,
        except over a notice when the callback returns `True`.
        `full_screen` takes the whole terminal from the start; otherwise the panel starts compact and grows.
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
        self.chords = {'ctrl-o': self.toggle_details, **(chords or {})}
        self.pinned = pinned
        self.spinner = spinner
        self.panel = panel
        self.queue_changed = queue_changed
        self.bindings = bindings
        self.keyboard_labels = keyboard_labels
        self.busy_delivery = busy_delivery
        self.remember_busy_delivery = remember_busy_delivery
        self.input_changed = input_changed
        self.key_test: KeyTest | None = None
        # A shell's `SessionRuntime.bind` replaces this with the shown session's questions and preference.
        self.questions = PromptQuestions(
            paint=self.paint, preference=lambda: 'ask', remember=lambda _: None, show=self.console.print
        )
        self.queue_in_toolbar = queue_in_toolbar
        self.ctrl_enter_steers = ctrl_enter_steers()
        """Whether Ctrl+Enter (and Ctrl+J, which older terminals send for it) steers, as Alt+Enter does.

        Otherwise Ctrl+Enter is Enter and Ctrl+J inserts a newline.
        """
        self.notice = ''
        self._chord_prefix = ''
        self._last_escape: float | None = None
        self.buffer = PromptBuffer(history=list(reversed(list(history.load_history_strings()))))
        self.output = PromptSurface(output=console.file, size=lambda: console.size, transcript=transcript)
        self.output.full_screen = full_screen
        self.keys = PromptKeys(
            source=get_app_session().input,
            feed=self.feed,
            eof=lambda: self.submit(EOFError()),
            unknown=self.unrecognized_key,
        )
        self._submissions: deque[_Submission] = deque()
        # The queued prompt taken out of the queue while the draft edits it, the submission it returns after
        # (`None` for the front), and the draft from before the recall that took it, which Esc restores.
        self._editing: QueuedPrompt | None = None
        self._editing_after: _Submission | None = None
        self._shelved_draft = ''
        self._queue_open = False
        self._delivery_choice = False
        self._queue_armed = False
        self._queue_selected: QueuedPrompt | None = None
        self._action_draft: PromptBuffer | None = None
        self._recall_queue: list[QueuedPrompt] = []
        self._recall_target: QueuedPrompt | None = None
        self._search_target: QueuedPrompt | None = None
        self._submitted = asyncio.Event()
        self._animate = asyncio.Event()
        """Set to cut short the timed repaint's idle sleep, as work starts."""
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
        self._panel_shown = False
        self._animated_at = -math.inf
        self._painted_at = -math.inf
        self._painted_size: ConsoleDimensions | None = None

    @property
    def queued_messages(self) -> tuple[str, ...]:
        """Pending text, excluding control signals, including the queued prompt being edited."""
        return tuple(item.text for item in self.queued_prompts())

    @property
    def drafts(self) -> tuple[str, ...]:
        """Every text the editor may still send, whose image attachments must stay.

        That is the draft, the one Esc restores while a queued prompt is edited, the one Down restores during a
        recall walk of the draft or of the composer a queue edit set aside, the one a search hides, and the
        queued prompts.
        """
        shelved = (self._shelved_draft,) if self._editing is not None else ()
        composers = (self.buffer,) if self._action_draft is None else (self.buffer, self._action_draft)
        walked = tuple(composer.saved_draft for composer in composers if composer.history_index is not None)
        searched = (self.buffer.search_original,) if self.buffer.search is not None else ()
        action_draft = (self._action_draft.text,) if self._action_draft is not None else ()
        return (self.buffer.text, *shelved, *walked, *searched, *action_draft, *self.queued_messages)

    def queued_prompts(self) -> list[QueuedPrompt]:
        """The queued prompts, oldest first, without control signals.

        The one being edited is out of the queue so it cannot run, but is listed at the place it returns to.
        """
        items = list(self._submissions)
        if self._editing is not None:
            items.insert(self._return_index(), self._editing)
        return [item for item in items if isinstance(item, QueuedPrompt)]

    def runnable_prompts(self) -> list[QueuedPrompt]:
        """The queued prompts that can run, without the one being edited, which the pending count leaves out."""
        return [item for item in self._submissions if isinstance(item, QueuedPrompt)]

    def _return_index(self) -> int:
        after = self._editing_after
        # Once the submission before it has run, so has everything ahead of it: it goes first.
        return self._submissions.index(after) + 1 if after is not None and after in self._submissions else 0

    def _edit(self, entry: QueuedPrompt | None) -> None:
        """Take `entry` out of the queue while the draft edits it, returning the one edited before unchanged.

        A prompt being edited cannot run, so a turn that ends meanwhile starts the next one instead.
        An entry the run took before the walk reached it is only text in the draft, as history is.
        """
        if entry is self._editing:
            return
        if self._editing is not None:
            self._put_back(self._editing)
        if entry is None or entry not in self._submissions:
            return
        index = self._submissions.index(entry)
        self._editing_after = self._submissions[index - 1] if index else None
        self._submissions.remove(entry)
        if not self._submissions and not self._wake_pending:
            self._submitted.clear()
        self._editing = entry

    def _put_back(self, entry: QueuedPrompt) -> None:
        """Return the prompt taken out to edit to its place in the queue."""
        self._submissions.insert(self._return_index(), entry)
        self._editing = None
        self._submitted.set()

    def _drop_editing(self) -> None:
        """Forget the prompt being edited, which leaves the queue for good, telling `queue_changed`."""
        if self._editing is not None:
            self._editing = None
            self.queue_changed()

    def _keep_edit_as_draft(self) -> None:
        """Leave the prompt being edited in the draft only, followed by the draft Up set aside.

        While the recall walk lasts, Down still restores that draft, so it is not added twice. Joined, a
        `/command` or `!` line on either side would turn the other into part of the command, or itself into
        prompt text, so the set-aside draft goes to history instead, where Up finds it. A composer a queue
        action set aside is not joined: sending the draft brings it back whole.
        """
        if self._editing is None:
            return
        if self._action_draft is None and self.buffer.history_index is None and (shelved := self._shelved_draft):
            edit = self.buffer.text.strip()
            joined = '\n'.join(filter(None, [edit, shelved]))
            # A bare `!` edit is a prompt alone, but joined it would send the draft to the shell.
            if joined == shelved or all(
                input_kind(expand_bare_command(text)) == 'prompt' for text in (edit, shelved, joined)
            ):
                self.buffer.replace(joined)
                self.refresh_completions()
            else:
                self.history.append_string(shelved)
                self.buffer.history.append(shelved)
        self._drop_editing()

    def take_queued(self) -> list[QueuedPrompt]:
        """Remove pending entries without changing their IDs, as another session reads this editor next."""
        self._edit(None)
        queued = self.queued_prompts()
        for entry in queued:
            self._discard(entry)
        return queued

    def submit(self, value: str | QueuedPrompt | KeyboardInterrupt | EOFError | PromptRewind) -> None:
        """Publish a submission without ending or replacing the editor."""
        self._enqueue(QueuedPrompt(value) if isinstance(value, str) else value)

    def rewrite(self, entry: QueuedPrompt, text: str) -> None:
        """Replace a queued prompt's text, keeping its place, as Enter on a recalled one does."""
        entry.text = text
        self.queue_changed()
        self.paint()

    def discard(self, entry: QueuedPrompt) -> None:
        """Drop a queued prompt, as clearing a recalled one before Enter does.

        Dropping the one being edited ends the edit: the draft stays, and Enter queues it as a new prompt.
        """
        if entry is self._editing:
            self._delivery_choice = False
            self._keep_edit_as_draft()
        else:
            self._discard(entry)
        self.paint()

    def _enqueue(self, value: _Submission) -> None:
        self._submissions.append(value)
        self._submitted.set()
        if isinstance(value, QueuedPrompt):
            self.queue_changed()
        self.paint()

    def wake(self) -> None:
        """Wake an idle reader once, without changing its draft, history, or queue."""
        self._wake_pending = True
        self._submitted.set()

    def _discard(self, entry: QueuedPrompt) -> None:
        # Removing the anchor does not consume the earlier work ahead of a held edit.
        if entry is self._editing_after:
            index = self._submissions.index(entry)
            self._editing_after = self._submissions[index - 1] if index else None
        self._submissions.remove(entry)
        if not self._submissions and not self._wake_pending:
            self._submitted.clear()
        self.queue_changed()

    async def read(self) -> str:
        """Consume queued submissions in order, first showing the status the shell set up for this read.

        With nothing else queued the shell is idle, so a queued prompt being edited is just the draft now, unless an
        explicit delivery choice still holds it.
        """
        # Keep the reserved edit while a delivery choice spans completion of the surrounding queue.
        if not self._submissions and not self._delivery_choice:
            self._keep_edit_as_draft()
        self.paint()
        await self._submitted.wait()
        # What a read returns starts work, whose spinner animates from its first frame, not the next idle one.
        self._animated_at = self.clock()
        self._animate.set()
        if not self._submissions:
            if not self._delivery_choice:
                self._keep_edit_as_draft()
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
        self.images.retain(self.drafts)
        self.images.notice = ''
        try:
            paths = pasted_paths(text) if text is not None else []
            if text is not None and not paths:
                self.buffer.insert(text, paste=True)
            else:
                self.buffer.insert(self.images.attach(read_images(paths) if text is not None else clipboard_images()))
                self.images.notice = ATTACHED.replace('Enter', self.key_label('send'))
        except (OSError, ValueError, NotImplementedError, Image.DecompressionBombError) as exc:
            self.images.paste_failed(exc, clock=self.clock)

    def feed(self, key: str, data: str = '') -> None:
        """Route editing, completion and interrupts without rendering a widget tree."""
        if self.key_test is not None:
            self.calibrate(key, data)
            self.paint()
            return
        key = self.effective_key(key)
        if key in TRANSCRIPT_KEYS:
            # Leave the draft alone, and the notice unless a drag copied text.
            if self.output.transcript_key(key, data):
                self.notice = 'Copied the selection to the clipboard.'
                self.paint()
            return
        if self.queue_key(key, data):
            self.paint()
            return
        width, height = self.console.size
        if self.questions.menu is not None and (key == 'f2' or self.questions.focused) and min(width, height) < 6:
            # Below six rows a focused question cannot show its choice, so no key may choose one unseen.
            self.questions.focused = False
            self.notice = 'Enlarge the terminal to review the question; nothing was chosen.'
            self.paint()
            return
        if self.questions.feed(
            key, data, draft=self.buffer, bindings=self.bindings(), queue=lambda: self.accept(follow_up=True)
        ):
            self.paint()
            return
        self.notice = ''
        if key != 'escape':
            self._last_escape = None
        if not self._chord(key):
            self._route(key, data)
        self.paint()

    def effective_key(self, key: str) -> str:
        """Fold Ctrl+Enter into Enter, or into the steer binding where Ctrl+Enter steers.

        Windows Terminal sends Ctrl+Enter as Ctrl+J unless it reports modified keys, so Ctrl+J steers there too.
        """
        if self.ctrl_enter_steers and key in ('ctrl-enter', 'ctrl-j'):
            return canonical_key(self.bindings().steer)
        return 'enter' if key == 'ctrl-enter' else key

    def key_label(self, action: InputAction) -> str:
        """Name an action's key as hints show it.

        Newline names Ctrl+J, which every terminal can send, and steer names Ctrl+Enter where Ctrl+Enter steers,
        since Ctrl+J steers there too.
        """
        if action == 'steer' and self.ctrl_enter_steers:
            return 'Ctrl+Enter'
        if action == 'newline' and not self.ctrl_enter_steers:
            return 'Ctrl+J'
        return self.bindings().label(action, keyboard=self.keyboard_labels())

    def queue_key(self, key: str, data: str) -> bool:
        """Visible queue actions keep an entry identity, never a position that can race turn completion."""
        if self._delivery_choice:
            return self.delivery_key(key, data)
        if key == 'f4':
            self._queue_open = not self._queue_open
            self._queue_selected = next(iter(self.runnable_prompts()), None)
            self._queue_armed = False
            # A key that closes the panel reaches the editor, so it must not confirm a question focused before F4.
            self.questions.focused = self.questions.routing = False
            asyncio.get_running_loop().call_soon(self.arm_queue)
            return True
        if not self._queue_open:
            return False
        if not self._queue_armed:
            if key == 'paste':
                self.paste(data)
            elif len(key) == 1 and key.isprintable():
                self.buffer.insert(key)
            return True
        if key == 'm':
            mode = 'next-checkpoint' if self.busy_delivery() == 'after-turn' else 'after-turn'
            self.remember_busy_delivery(mode)
            return True
        if key in ('escape', 'ctrl-c'):
            self._queue_open = False
            return True
        return self.queue_entry_key(key)

    def delivery_key(self, key: str, data: str) -> bool:
        """Require fresh, explicit intent before changing a mixed queue's delivery order."""
        if key == 'paste':
            self.paste(data)
        elif not self._queue_armed:
            if len(key) == 1 and key.isprintable():
                self.buffer.insert(key)
        elif key in ('escape', 'ctrl-c'):
            self._delivery_choice = False
            self.notice = ''
        elif key == 'q':
            self._delivery_choice = False
            self.notice = ''
            self.accept(follow_up=True, keep_position=True)
        elif key == 's':
            if self._steered(self.buffer.text.strip()):
                self.finish_steering(self.buffer.text.strip())
            else:
                self.notice = 'Cannot steer; nothing was sent.'
        return True

    def choose_delivery(self) -> None:
        """Hold the entire draft/edit unchanged while the user chooses actual delivery timing."""
        self._delivery_choice = True
        self._queue_open = False
        self._queue_armed = False
        asyncio.get_running_loop().call_soon(self.arm_queue)

    def queue_entry_key(self, key: str) -> bool:
        """Apply an explicit action only while the selected entry is still runnable; other keys close the panel.

        A key the panel does not handle falls through to the editor, so typing after F4 keeps its letters.
        """
        entries = self.runnable_prompts()
        entry = self._queue_selected
        if entry is None:
            # The panel opened on an empty queue, so it offers only the delivery mode.
            self._queue_open = False
            return False
        if entry not in entries:
            self._queue_open = False
            self.notice = 'That message is no longer queued.'
            return True
        match key:
            case 'up' | 'down':
                self._queue_selected = entries[(entries.index(entry) + (-1 if key == 'up' else 1)) % len(entries)]
            case 'e':
                self.edit_queue_entry(entry)
                self._queue_open = False
            case 'd':
                self.discard(entry)
                self._queue_open = False
            case 's':
                if self._steered(entry.text):
                    self.discard(entry)
                else:
                    self.notice = 'Cannot steer this input now. Commands and shell lines never steer.'
                self._queue_open = False
            case _:
                self._queue_open = False
                return False
        return True

    def arm_queue(self) -> None:
        """Only fresh input after opening the queue can activate its actions."""
        self._queue_armed = True

    def edit_queue_entry(self, entry: QueuedPrompt) -> None:
        """Use the prerequisite's held-entry lifecycle, retaining the entire earlier composer."""
        if self._editing is not None:
            self.stop_editing()
        self.questions.focused = self.questions.routing = False
        # A history search can end an edit and leave the composer set aside; a second edit keeps that composer.
        if self._action_draft is None:
            self._action_draft = self.buffer
        self._shelved_draft = self._action_draft.text
        self._edit(entry)
        self.buffer = PromptBuffer(history=self.buffer.history)
        self.buffer.replace(entry.text)
        self.refresh_completions()

    def restore_action_draft(self) -> None:
        """Finish an explicit queue action without losing the earlier cursor or undo history."""
        if self._action_draft is not None:
            self.buffer, self._action_draft = self._action_draft, None
            self.refresh_completions()

    def unrecognized_key(self, _: str) -> None:
        """Report unsupported controls without echoing bytes that could contain pasted text."""
        if self.key_test is not None:
            self.key_test.report = (
                'Unsupported terminal key report (contents hidden). No action. '
                'Try Ctrl+J for newline or a tested alternative binding. Ctrl+G closes.'
            )
            self.paint()

    def calibrate(self, key: str, data: str) -> None:
        """Observe keys without executing actions; retain text typed during the test."""
        assert self.key_test is not None
        if key == 'ctrl-g':
            # Keep the rest of this decoder batch in calibration, even held Enter.
            asyncio.get_running_loop().call_soon(self.end_calibration)
            return
        self.key_test.observe(
            key,
            keyboard=self.keyboard_labels(),
            action=self.input_action(key),
            steer=self.key_label('steer'),
            newline=self.key_label('newline'),
        )
        if key == 'paste' or (len(key) == 1 and key.isprintable()):
            self.buffer.insert(data if key == 'paste' else key, paste=True)

    def input_action(self, key: str) -> str:
        """Describe the action calibration suppresses in this composer context."""
        action = self.bindings().action(self.effective_key(key))
        if action is not None:
            return self.binding_action(action)
        match key:
            case 'ctrl-c':
                return 'interrupt' if self.interrupts.active else 'clear draft / exit gesture'
            case 'escape':
                return (
                    'stop editing'
                    if self._editing is not None
                    else 'interrupt'
                    if self.interrupts.active
                    else 'dismiss / rewind gesture'
                )
            case 'ctrl-d':
                return 'delete' if self.buffer.text else 'exit'
            case 'ctrl-l':
                return 'clear transcript'
            case 'ctrl-v' | 'alt-v':
                return 'paste image'
            case 'ctrl-r':
                return 'search history'
            case 'tab' | 'backtab':
                return 'complete'
            case 'f2':
                return 'focus question' if self.questions.menu is not None else 'no pending question'
            case 'f4':
                return 'queue actions'
            case _:
                return 'edit'

    def binding_action(self, action: InputAction) -> str:
        """Describe remappable actions using the same current state as their handlers."""
        match action:
            case 'send':
                match self.send_route():
                    case 'save':
                        return 'save queued edit'
                    case 'choose-save':
                        return 'choose save delivery'
                    case 'question':
                        return self.questions.hint(
                            send_label=self.key_label('send'), newline_label=self.key_label('newline')
                        )
                    case 'choose':
                        return 'choose delivery'
                    case 'send':
                        return 'send'
                    case 'after-turn':
                        return 'after turn'
                    case 'next-checkpoint':
                        return 'next checkpoint'
            case 'steer':
                if not self.buffer.text.strip():
                    return (
                        'steer oldest queued'
                        if self.interrupts.active and self._submissions
                        else 'no active steering target'
                    )
                return 'steer' if self.interrupts.active else 'send'
            case 'newline':
                return 'newline'

    def send_route(self) -> SendRoute:
        """Whether Send saves, answers, or delivers this draft, and when."""
        if self._editing is not None:
            return (
                'choose-save'
                if self.busy_delivery() == 'next-checkpoint' and can_steer(expand_bare_command(self.buffer.text))
                else 'save'
            )
        if self.questions.menu is not None:
            return 'question'
        if (
            self._submissions
            and self.busy_delivery() == 'next-checkpoint'
            and can_steer(expand_bare_command(self.buffer.text))
        ):
            return 'choose'
        if not self.interrupts.active:
            return 'send'
        if not can_steer(expand_bare_command(self.buffer.text)):
            return 'after-turn'
        return self.busy_delivery()

    def end_calibration(self) -> None:
        """Return to the unchanged composer once the explicit close key's batch ends."""
        self.key_test = None
        self.paint()

    def _route(self, key: str, data: str) -> None:
        cycles = key in ('up', 'down') and self._arrows_cycle_completions()
        action = self.bindings().action(key)
        match key:
            case 'ctrl-c':
                self.interrupt()
            case 'escape':
                self.escape()
            case 'ctrl-d':
                if self.buffer.text:
                    self.buffer.edit('delete')
                else:
                    self.submit(EOFError())
            case 'ctrl-l':
                # As in Claude Code: a clean screen; the conversation, draft, and queue stay.
                self.output.clear(keep_current=self.interrupts.active)
            case 'paste' | 'ctrl-v' | 'alt-v':
                self.paste(data if key == 'paste' else None)
            case _ if key == 'ctrl-r' or self.buffer.search is not None:
                self.search(key)
            case 'tab' | 'backtab':
                self.complete(backwards=key == 'backtab')
            case _ if action == 'send':
                self.accept()
            case _ if action == 'steer':
                self.steer_draft() if self.buffer.text.strip() else self.steer_queued()
            case _ if action == 'newline':
                self.buffer.insert('\n')
            case 'up' | 'down' if cycles:
                self.complete(backwards=key == 'up', accept_single=False)
            case 'up' | 'down':
                self.recall(backwards=key == 'up')
            case _:
                self.buffer.edit(key)
        if key not in ('tab', 'backtab', 'escape') and not cycles:
            self.refresh_completions()

    def toggle_details(self) -> str:
        """Expand or collapse the output of every finished command in the transcript, for Ctrl+O."""
        return 'Showing command output.' if self.output.toggle_details() else 'Hiding command output.'

    def escape(self) -> None:
        """Stop editing a queued prompt, then cancel or dismiss; only consecutive idle presses request a rewind."""
        previous, self._last_escape = self._last_escape, None
        if self._editing is not None:
            self.stop_editing()
        elif self.interrupts.active:
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

    def stop_editing(self) -> None:
        """Return the queued prompt being edited unchanged, or remove it if its draft was cleared.

        Either way the draft from before the recall comes back, as in Claude Code, where Esc on a
        queued prompt being edited ends the edit rather than the turn.
        """
        assert self._editing is not None
        if self.buffer.text.strip():
            self._put_back(self._editing)
        else:
            self._remove_editing()
        self.buffer.search = None
        self.buffer.history_index = None
        if self._action_draft is None:
            self.buffer.replace(self._shelved_draft)
        else:
            self.restore_action_draft()
        self.refresh_completions()

    def _remove_editing(self) -> None:
        """Remove the queued prompt whose draft was cleared, as the hint `Clear it to remove` says."""
        self._drop_editing()
        self.notice = 'Removed the queued message.'
        telemetry.record('prompt submitted', route='discarded queued')

    def restore_draft(self, text: str) -> None:
        """Replace the draft after a rewind, without editing a queued submission."""
        self._edit(None)
        self.buffer.search = None
        self.buffer.history_index = None
        self.buffer.replace(text)
        self.refresh_completions()

    def _arrows_cycle_completions(self) -> bool:
        """Whether Up/Down move through the popup rather than the draft and history.

        A history walk owns the arrows, so recalling a `/command` shows its suggestions
        without trapping the walk on them. Tab picks a suggestion and hands the arrows to
        the popup. Editing recalled text keeps the walk only when it protects a nonempty
        original draft; editing the restored draft ends the walk.
        """
        return bool(self._completions) and (self.buffer.history_index is None or self._selection >= 0)

    def search(self, key: str) -> None:
        """Search history; a picked match is a new prompt, not an edit of a recalled queued one."""
        if self.buffer.search is None:
            self._search_target = self._editing
        self.buffer.edit(key)
        # Cancelling (or finding nothing) leaves the original text, and with it the edit target.
        self._edit(self._search_target if self.buffer.text == self.buffer.search_original else None)

    def interrupt(self) -> None:
        """Cancel running work, or else drop the draft and signal the reader.

        Cancelling while a queued prompt is edited ends the edit as Esc does, before the queue runs on.
        """
        cancelled = self.interrupts.cancel()
        telemetry.record('prompt interrupt', key='ctrl-c', cancelled_turn=cancelled)
        if not cancelled:
            self.buffer.replace('')
            self.buffer.search = None
            self._edit(None)
            self.submit(KeyboardInterrupt())
        elif (editing := self._editing) is not None:
            # Only Enter saves an edit: the prompt runs unchanged, or is removed if cleared, as Esc leaves it, and Up
            # finds the edit in history.
            if (edit := self.buffer.text.strip()) and expand_bare_command(edit) != editing.text:
                self.history.append_string(edit)
                self.buffer.history.append(edit)
            self.stop_editing()

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

    def accept(self, *, follow_up: bool = False, keep_position: bool = False) -> None:
        """Accept a completion, route a pending question, or choose delivery for the nonempty draft."""
        if self._selection >= 0:
            item = self._completions[self._selection]
            if self.buffer.text[self._completion_start(item) : self.buffer.cursor] != item.text:
                self.accept_completion()
                return
            # The highlighted command is already typed in full, so Enter runs it.
            self.dismiss_completions()
        if (
            not follow_up
            and self._editing is None
            and input_kind(self.buffer.text) == 'prompt'
            and self.questions.route(self.buffer)
        ):
            return
        text = self.buffer.text.strip()
        if not keep_position and text and self.checkpoint_delivery(expand_bare_command(text)):
            return
        # The prompt being edited is out of the queue: each outcome below returns it or leaves it out.
        target, self._editing = self._editing, None
        if not text and target is None:
            return
        recalled = self.buffer.history_index is not None
        self.buffer.history_index = None
        self.buffer.reset()
        if target is not None and self._action_draft is None:
            # Every way out of an edit brings back the draft set aside to make it, as Esc does.
            self.buffer.replace(self._shelved_draft)
            self.refresh_completions()
        if target is not None and not text:
            self.queue_changed()
            telemetry.record('prompt submitted', route='discarded queued')
            self.restore_action_draft()
            return
        command = expand_bare_command(text)
        if target is not None and command == target.text:
            self._put_back(target)
            self.restore_action_draft()
            return
        self.output.view.follow()
        self.history.append_string(text)
        self.buffer.history.append(text)
        route = self._deliver(command, recorded=text, target=target)
        self.restore_action_draft()
        telemetry.record('prompt submitted', route=route, recalled=recalled, **_submission(command, self.commands))

    def checkpoint_delivery(self, command: str) -> bool:
        """Use native delivery only when unambiguous; otherwise keep the draft for an explicit choice."""
        if self.busy_delivery() != 'next-checkpoint' or not can_steer(command):
            return False
        if not self.interrupts.active and self._editing is None and not self._submissions:
            return False
        if self._editing is not None or self._submissions:
            self.choose_delivery()
        elif self._steered(command):
            self.finish_steering(self.buffer.text.strip())
        else:
            self.choose_delivery()
            self.notice = 'Cannot steer; nothing was sent.'
        return True

    def _deliver(self, command: str, *, recorded: str, target: QueuedPrompt | None) -> str:
        """Run, rewrite, or queue an accepted draft, returning the route telemetry records."""
        if self.run_now is not None and self.run_now(command):
            if target is not None:
                self.queue_changed()
            return 'run now'
        if target is not None:
            target.text, target.recorded = command, recorded
            self._put_back(target)
            self.queue_changed()
            return 'edited queued'
        route = 'queued' if self.interrupts.active else 'submitted'
        self._enqueue(QueuedPrompt(command, recorded=recorded))
        return route

    def recall(self, *, backwards: bool) -> None:
        """Walk queued prompts, newest first, before command history.

        The queue holds the most recent input, so Up reaches it before older history, as in
        shell history. A recalled queued prompt leaves the queue while it is edited, and keeps its
        place: Enter puts it back rewritten, Esc unchanged, and clearing the draft removes it.
        """
        if self.buffer.history_index is None:
            self._recall_queue = self.queued_prompts()
            self._recall_target = self._editing
            if self._editing is None:
                self._shelved_draft = self.buffer.text
        self.buffer.vertical(
            backwards=backwards,
            queued=tuple(entry.text for entry in self._recall_queue),
            recorded=tuple(entry.recorded for entry in self._recall_queue),
        )
        offset = self.buffer.recall_offset
        if offset == 0:
            self._edit(self._recall_target)
        elif offset is not None:
            self._edit(self._recall_queue[offset] if -offset <= len(self._recall_queue) else None)

    def _steered(self, text: str) -> bool:
        """Hand prompt text to the active run; commands and shell lines never steer."""
        command = expand_bare_command(text)
        return self.steer is not None and can_steer(command) and self.steer(command)

    def steer_draft(self) -> None:
        """Send the draft to the active run now, or accept it as Enter would when it cannot steer."""
        text = self.buffer.text.strip()
        if not self._steered(text):
            if (
                self.busy_delivery() == 'next-checkpoint'
                and self.interrupts.active
                and can_steer(expand_bare_command(text))
            ):
                self.choose_delivery()
                self.notice = 'Cannot steer; nothing was sent.'
            else:
                self.accept()
            return
        self.finish_steering(text)

    def finish_steering(self, text: str) -> None:
        """Consume a draft only after native steering accepted it, restoring any earlier composer."""
        if self.busy_delivery() == 'next-checkpoint':
            self.notice = f'Next checkpoint (sent to active run): {" ".join(terminal_text(text).split())}'
        self._delivery_choice = False
        # The steered text may be an edit of a queued prompt, which then no longer waits for its own turn.
        edited = self._editing is not None
        self._drop_editing()
        # Steered text joins the running turn, so show its response, as submitting does.
        self.output.view.follow()
        self.history.append_string(text)
        self.buffer.history.append(text)
        self.buffer.history_index = None
        self.buffer.reset()
        if edited and self._action_draft is None:
            # As every other way out of an edit, steering it brings back the draft set aside by Up.
            self.buffer.replace(self._shelved_draft)
            self.refresh_completions()
        self.restore_action_draft()
        telemetry.record('prompt steer', steered=True, source='draft')

    def steer_queued(self) -> None:
        """Promote the oldest follow-up without bypassing commands, shell lines, or control signals."""
        if not self._submissions or self.steer is None:
            return
        head = self._submissions[0]
        if not isinstance(head, QueuedPrompt) or not self._steered(head.text):
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

    def input_control_frame(self, *, width: int, height: int) -> tuple[str, ...] | None:
        """Render explicit key testing or queue actions without a second input owner."""
        if self.key_test is not None:
            return tuple(
                self.key_test.report[start : start + width] for start in range(0, len(self.key_test.report), width)
            )[: max(1, height - 2)]
        if self._delivery_choice:
            return self.delivery_frame(width=width, height=height)
        if self.questions.focused:
            rows = self.questions.rows(
                width=width, height=height, send_label=self.key_label('send'), newline_label=self.key_label('newline')
            )
            # A focused question the composer frame would cut takes the frame, so its chosen row stays visible.
            if len(rows) > height - 6 - (self._editing is not None):
                return rows[: max(1, height - 2)]
        if not self._queue_open:
            return None
        entry = self._queue_selected
        if entry is None:
            return (
                truncate(f'New busy: {self.busy_delivery().replace("-", " ")} - m change - Esc close', width),
                *self.buffer.rows(width=width, limit=max(1, height - 3)),
            )
        preview = ' '.join(terminal_text(entry.text).split())
        return (
            truncate('e Edit - d Remove - s Steer - Esc close', width),
            truncate(f'New busy: {self.busy_delivery().replace("-", " ")} - m change', width),
            truncate('Queue: Up/Down choose (after turn)', width),
            truncate(preview, width),
            *self.buffer.rows(width=width, limit=max(1, height - 6)),
        )

    def delivery_frame(self, *, width: int, height: int) -> tuple[str, ...]:
        """Show actual timing and failed steering alongside explicit delivery choices."""
        timing = 'after turn' if self.interrupts.active else 'separate turn'
        steer = (
            's Steer/promote: next checkpoint'
            if can_steer(expand_bare_command(self.buffer.text))
            else 'Steer unavailable: commands/shell/blank'
        )
        notice = (truncate(' '.join(terminal_text(self.notice).split()), width),) if self.notice else ()
        return (
            truncate('Delivery: next checkpoint requested', width),
            truncate(steer, width),
            truncate(f'q Keep queue position: {timing}', width),
            truncate('Esc cancel: keep draft/edit', width),
            *notice,
            truncate('Earlier queued entries stay after turn', width),
            *self.buffer.rows(width=width, limit=max(1, height - 8 - len(notice))),
        )[: max(1, height - 2)]

    def frame(self) -> tuple[str, ...]:
        """Build the reserved rows; transcript contents are deliberately absent."""
        width, height = self.console.size
        width, height = max(1, width), max(2, height)
        muted, reset = theme.sgr(theme.MUTED), '\x1b[0m'
        controls = self.input_control_frame(width=width, height=height)
        if controls is not None:
            return controls
        if width < 6 or height < 6 + (self._editing is not None):
            self._panel_shown = False
            return tuple(self.buffer.rows(width=width, limit=1))
        # `paint` keeps `height - 2` rows; the title, one draft row, the rule, and the footer need four.
        room = height - 6
        limit = max(1, height // 6)
        question_rows = list(
            self.questions.rows(
                width=width,
                height=height,
                send_label=self.key_label('send'),
                newline_label=self.key_label('newline'),
            )
        )
        panel = [truncate(row, width) + reset for row in self.panel(self.spinner().frame(self.clock()))]
        if question_rows:
            panel = [*question_rows, *panel]
        self._panel_shown = bool(panel)
        rows = _capped(
            panel,
            limit=max(limit, len(question_rows)),
            room=room - (self._editing is not None),
            more=muted + '+{} more' + reset,
        )
        # Only opt-in checkpoint delivery has two timings for a preview to tell apart.
        timing = ' (after turn)' if self.busy_delivery() == 'next-checkpoint' else ''
        queued = [
            muted
            + truncate(
                f'{"Command" if is_command_input(entry.text) else "Follow-up"}{timing}: '
                f'{" ".join(terminal_text(entry.text).split())}',
                width,
            )
            + reset
            for entry in self.runnable_prompts()
        ]
        if self._editing is not None:
            # Keep the non-runnable state visible even when the queue preview is capped.
            rows.insert(0, muted + truncate("Editing - won't send while you edit it", width) + reset)
        rows += _capped(queued, limit=limit, room=room - len(rows), more=muted + '+{} more queued' + reset)
        title = self._working_title(width, muted=muted) if self.interrupts.active else ''
        rows.append(muted + title + '─' * max(0, width - visible_length(title)) + reset)
        # The box has no side borders: the draft and the suggestions are rows between the top and
        # bottom rules, so no row can drift out of alignment with the corners. The draft wraps two
        # cells narrower, after a `❯ ` marker on its first row shown (a tall draft scrolls to the
        # cursor, and the marker stays visible) and an indent on the rest.
        # The pinned row only takes a spare row: `paint` keeps `height - 2` rows, and the title,
        # one draft row, the rule, and the footer come first.
        pinned = self.pinned() if height - len(rows) - 5 >= 1 else ''
        inner = max(1, height - len(rows) - 4 - bool(pinned))
        popup_want = min(6, len(self._completions))
        draft = self.buffer.rows(width=width - 2, limit=max(1, min(height // 3, inner - popup_want)))
        rows.append(f'{theme.sgr(theme.ACCENT)}❯{reset} {draft[0]}')
        rows.extend(map('  {}'.format, draft[1:]))
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
        rows.append(muted + truncate(self._footer(width, muted=muted), width) + reset)
        # Hints use spare space, without displacing the draft, pinned status, or pending work. Until an input
        # setting changes, they show only while a question waits in the composer.
        if len(rows) < height - 2 and (self.input_changed() or self.questions.menu is not None):
            rows.insert(-1 - bool(pinned), muted + truncate(self.input_hint(), width) + reset)
        return tuple(rows)

    def _working_title(self, width: int, *, muted: str) -> str:
        """The rule above the draft while work runs: the animation, its time, and the keys that act on it.

        The first of the candidate rows that fits wins; a draft's own keys outrank the Esc hint and the time.
        """
        head, glyph, reset = '─ Working ', self.spinner().frame(self.clock()), '\x1b[0m'
        elapsed = format_duration(self.interrupts.elapsed)
        if self._editing is not None:
            # Esc stops the edit rather than interrupting, so the edit's keys replace the Esc hint.
            keys = self.working_keys()
            candidates = [[elapsed, *keys], keys, ['Esc to stop editing']]
        else:
            # Esc, unlike Ctrl+C, never counts toward the double press that quits.
            candidates = [[elapsed, 'Esc to interrupt']]
            # Queue and steer hints only matter once there is something to send.
            if keys := self.working_keys():
                candidates = [[elapsed, 'Esc to interrupt', *keys], [elapsed, *keys], keys]
            candidates.append(['Esc to interrupt'])
        title = ''
        for parts in candidates:
            title = f'{head}{glyph} {" · ".join(parts)} '
            if visible_length(title) <= width:
                break
        title = truncate(title, width)
        if title.startswith(head + glyph):
            title = f'{head}{theme.sgr(theme.ACCENT)}{glyph}{reset}{muted}{title[len(head + glyph) :]}'
        return title

    def working_keys(self) -> list[str]:
        """The working title's keys: queue edits and question routing come before generic sending.

        After-turn delivery keeps "Enter to queue"; only opt-in delivery names its timing.
        """
        send = self.key_label('send')
        if self._editing is not None:
            save = 'choose delivery' if self.send_route() == 'choose-save' else 'save'
            return [f'{send} to {save}', 'Esc to stop editing', 'Clear it to remove']
        if self.questions.menu is not None:
            return [self.questions.hint(send_label=send, newline_label=self.key_label('newline'))]
        if not (self.queued_messages or self.buffer.text.strip()):
            return []
        route = self.send_route()
        if self.busy_delivery() == 'after-turn':
            action = 'queue'
        elif route == 'choose':
            action = 'choose delivery'
        else:
            action = 'send at next checkpoint' if route == 'next-checkpoint' else 'queue after turn'
        return [f'{send} to {action}', f'{self.key_label("steer")} to steer']

    def input_hint(self) -> str:
        """Keep effective controls discoverable without truncating notices or model status."""
        send, newline = self.key_label('send'), self.key_label('newline')
        if self._editing is not None:
            steer = self.key_label('steer')
            action = 'Choose delivery' if self.send_route() == 'choose-save' else 'Save/requeue'
            return f'{send}: {action} - Esc: Stop editing - {steer}: Steer/send - Clear to remove'
        if self.questions.menu is not None:
            return self.questions.hint(send_label=send, newline_label=newline)
        if self.queued_messages:
            return f'F4 Queue: Edit / Remove / Steer / Delivery - {newline} newline - /input setup'
        action = self.binding_action('send')
        return f'{send}: {action} - F4 Delivery - {newline} newline - /input'

    def _footer(self, width: int, *, muted: str) -> str:
        """The bottom row: the history search, or else a notice or the toolbar, with the queue's length."""
        if self.buffer.search is not None:
            return f'reverse-i-search: {self.buffer.search}'
        notice = self.notice or self.images.notice or self._completion_error
        queued = ''
        # The count is of what can run, so a prompt being edited is not in it.
        if (runnable := self.runnable_prompts()) and (
            self.queue_in_toolbar is None or (notice and self.queue_in_toolbar())
        ):
            queued = f'{muted} | queued: {len(runnable)}'
        footer = (
            ' '.join(terminal_text(notice).split())
            if notice
            else ''.join((theme.sgr(style) if style else muted) + _toolbar_text(text) for style, text in self.toolbar())
        )
        # Cut the notice or toolbar, not the queue count after it.
        footer = truncate(footer, max(0, width - visible_length(queued)))
        # A cut can drop a link's end, which would link the rest of the row.
        return footer + (LINK_END if LINK_START in footer else '') + queued

    def paint(self) -> None:
        """Draw only when the editor owns the terminal."""
        if self._opened and not self._suspended:
            self.output.paint(self.frame())
            self._painted_at = self.clock()
            self._painted_size = self.console.size

    def tick(self) -> float:
        """Repaint for status changes when due, returning the seconds until the next tick.

        Every `ANIMATED_REPAINT` seconds while a turn runs, the panel shows forks or tasks, or for
        `IDLE_REPAINT` after either; `IDLE_REPAINT` seconds after the last frame otherwise. A spinner
        faster than that gets a repaint per frame, but only while it shows.
        The paste-error deadline also caps the wait; expiry repaints an unsuspended editor through this
        existing loop, without adding periodic polling.
        """
        now = self.clock()
        if self.interrupts.active or self._panel_shown:
            self._animated_at = now
        expired = self.images.expire_notice()
        if not self._suspended and (
            expired
            or now - self._animated_at < IDLE_REPAINT
            or now - self._painted_at >= IDLE_REPAINT
            # Hosts without SIGWINCH find a resize only here.
            or self.console.size != self._painted_size
        ):
            self.paint()
        else:
            # Output written too soon after the last frame waits for this, idle or not.
            self.output.refresh()
        if self.interrupts.active:
            return min(ANIMATED_REPAINT, self.spinner().interval, self.images.notice_remaining())
        # The frame just painted may have found the panel's first fork or task.
        if self._panel_shown or now - self._animated_at < IDLE_REPAINT:
            return min(ANIMATED_REPAINT, self.images.notice_remaining())
        due = self._painted_at + IDLE_REPAINT - now
        # A menu owning the screen leaves the last frame behind; nothing is due until it closes.
        return min(due if due > 0 else IDLE_REPAINT, self.images.notice_remaining())

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
                self.output.release(turn_running=self.interrupts.active)
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
                with anyio.move_on_after(self.tick()):
                    await self._animate.wait()
                self._animate.clear()

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
    `/plugins add` settings JSON or an MCP server's headers can hold a secret Pax cannot recognize.
    """
    kind = input_kind(text)
    if kind == 'command':
        name = text.split(maxsplit=1)[0].removeprefix('/')
        return {'kind': kind, 'command': name if name in commands else 'unknown', 'chars': len(text)}
    if kind == 'shell':
        return {'kind': kind, 'chars': len(text)}
    return {'kind': kind, 'chars': len(text), **telemetry.prompt_text(text)}


def _toolbar_text(text: str) -> str:
    """One toolbar fragment on one row; a link becomes an OSC 8 hyperlink, which a click opens on either screen."""
    shown = ' '.join(terminal_text(text).splitlines())
    return hyperlink(shown, url=text.url) if isinstance(text, LinkText) else shown
