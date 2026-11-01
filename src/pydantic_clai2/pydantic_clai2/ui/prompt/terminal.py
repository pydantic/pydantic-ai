"""The terminal CLAI runs in: its console, editor, and transcript, showing one live session at a time."""

import io
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

from anyio import CancelScope, Event, create_task_group
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import History
from rich.console import Console

from pydantic_clai2.commands import Commands
from pydantic_clai2.plugins import SessionEndReason
from pydantic_clai2.ui.prompt.completion_adapter import COMPLETION_STYLE, PromptCompleter
from pydantic_clai2.ui.prompt.image_input import ImageInput
from pydantic_clai2.ui.prompt.live_prompt import LivePrompt
from pydantic_clai2.ui.prompt.prompt_transcript import TranscriptBuffer
from pydantic_clai2.ui.rendering import theme

if TYPE_CHECKING:
    from pydantic_clai2.cli.self_update import WebHandoff
    from pydantic_clai2.runtime.session_runtime import SessionRuntime

DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT')


class _Nowhere(io.StringIO):
    """Where a session prints while the terminal shows another: it keeps nothing."""

    def write(self, text: str, /) -> int:
        return len(text)


NOWHERE = _Nowhere()


class Terminal(Generic[DepsT, OutputT]):
    """The terminal CLAI runs in, showing one live session of its process at a time.

    The session shown prints to the console, reads the editor (or the line prompt when the console is
    not a terminal), and takes Ctrl-C. Every other one prints nowhere and reads only what `send_input`
    gives it, as one the browser opened does. `/resume` of a session live in another shows that one.
    """

    def __init__(
        self, *, console: Console, transcript: TranscriptBuffer, history: History, shows_others: bool = True
    ) -> None:
        """`transcript` is the output retained for redrawing, kept across `/reload`; `history` the prompts recalled."""
        self.console = console
        self.shows_others = shows_others
        """Whether `/resume` shows another live session here; `clai2 --web` keeps its first one."""
        self.lines = not console.is_terminal
        """Whether it reads lines, decided once: a command that holds the terminal makes the console no terminal."""
        self.transcript = transcript
        self.history = history
        self.images = ImageInput()
        """Images pasted for the next prompt of the session shown."""
        self.editor: LivePrompt | None = None
        """The editor, while `drive` runs and the console is a terminal."""
        self.prompt: PromptSession[str] | None = None
        """The line prompt, when the console is not a terminal."""
        self.shown: SessionRuntime[DepsT, OutputT] | None = None
        self.home: SessionRuntime[DepsT, OutputT] | None = None
        """The session CLAI started with, while `drive` runs it."""
        self.ended_in: SessionRuntime[DepsT, OutputT] | None = None
        """The session whose end ended `drive`: the one shown then."""
        self.home_closed = False
        """Whether `drive` closed its session itself, as it ended while another was shown."""
        self.served: WebHandoff | None = None
        """The web UI for `/update` to serve again, from the session whose end ended `drive`."""
        self._reason: SessionEndReason = 'error'
        self._ended = Event()
        self._completer = PromptCompleter(Commands())
        """Completes the commands of the session shown, in the line prompt."""

    def session_console(self) -> Console:
        """A console for a session this terminal may show later: it prints nowhere until then, in its colours."""
        return Console(
            file=NOWHERE, color_system=_color_system(self.console), legacy_windows=self.console.legacy_windows
        )

    def attach(self, runtime: 'SessionRuntime[DepsT, OutputT]') -> None:
        """Show `runtime` from now on, handing it the console output; the session shown before prints nowhere."""
        previous, self.shown = self.shown, runtime
        if previous is not None and previous is not runtime:
            # Each session has a console of its own: the one shown before keeps printing, to nowhere.
            runtime.console.file, previous.console.file = previous.console.file, NOWHERE
            previous.shown_in(None)
        if self.lines:
            self._completer.commands = runtime.commands
            if self.prompt is None:
                self.prompt = self._line_prompt()
        elif self.editor is not None:
            self.editor.commands, self.editor.interrupts = runtime.commands, runtime.interrupts
        runtime.shown_in(self)

    async def show(self, runtime: 'SessionRuntime[DepsT, OutputT]') -> None:
        """Show `runtime`, live in this process, as `/resume` would: on a clear screen, with its history."""
        self.attach(runtime)
        await runtime.sessions.show_resumed()
        runtime.console.print(
            f'Showing session {runtime.session.conversation_id}, which is live in this CLAI.', markup=False
        )

    def ended(self, runtime: 'SessionRuntime[DepsT, OutputT]', reason: SessionEndReason) -> None:
        """Note that the session shown, `runtime`, ended for `reason`, which ends `drive`.

        Its web UI hand-off is the one `/update` read before uv replaced the installed code.
        """
        self.ended_in, self._reason, self.served = runtime, reason, runtime.handoff()
        self._ended.set()

    async def drive(self, home: 'SessionRuntime[DepsT, OutputT]') -> SessionEndReason:
        """Run `home`, the session CLAI started with, until the session shown ends; returns that one's reason.

        `home` stays live while another session is shown: it ends with its input, or by idle closing.
        The terminal shows it again before `drive` returns.
        """
        self._ended, self.ended_in, self._reason, self.home_closed = Event(), None, 'error', False
        self.home = home
        async with self._opened(home):
            try:
                async with create_task_group() as tasks:
                    tasks.start_soon(self._run_home, home)
                    await self._ended.wait()
                    tasks.cancel_scope.cancel()
            except BaseExceptionGroup as group:
                if len(group.exceptions) == 1:
                    raise group.exceptions[0] from None
                raise  # pragma: no cover -- only one task raises: the other waits.
            finally:
                if self.shown is not home:
                    self.attach(home)
        return self._reason

    async def _run_home(self, home: 'SessionRuntime[DepsT, OutputT]') -> None:
        reason: SessionEndReason = 'exit'  # When idle closing ends it.
        with home.process.live.driving(home):
            reason = await home.loop()
        if self.shown is not home:
            # It ended while another session was shown, which goes on.
            self.home_closed = True
            with CancelScope(shield=True):
                await home.close(reason)

    @asynccontextmanager
    async def _opened(self, home: 'SessionRuntime[DepsT, OutputT]') -> AsyncGenerator[None]:
        """Open the editor, when the console is a terminal, for `home` and the sessions shown after it."""
        if self.lines:
            yield
            return
        self.editor = LivePrompt(
            console=self.console,
            commands=self._shown().commands,
            history=self.history,
            images=self.images,
            interrupts=self._shown().interrupts,
            toolbar=lambda: self._shown().status.toolbar(),
            steer=lambda text: self._shown().steer(text),
            run_now=lambda text: self._shown().run_now(text),
            transcript=self.transcript,
            chords={
                'ctrl-x ctrl-s': lambda: self._shown().speculation.toggle(),
                'ctrl-b': lambda: self._shown().tasks.promote(),
            },
            pinned=lambda: self._shown().speculation.row(),
            spinner=lambda: self._shown().spinners.active(),
            panel=lambda glyph: (*self._shown().tasks.rows(glyph), *self._shown().forks.rows(glyph)),
            queue_changed=lambda: self._shown().live_session.changed(),
        )
        try:
            async with self.editor.opened():
                self._shown().bind(self.editor)
                yield
        finally:
            self._shown().bind(None)
            self.editor = None

    def _shown(self) -> 'SessionRuntime[DepsT, OutputT]':
        assert self.shown is not None, 'a terminal shows a session from its first `attach`'
        return self.shown

    def _line_prompt(self) -> PromptSession[str]:
        prompt = PromptSession[str](
            history=self.history,
            completer=self._completer,
            complete_while_typing=True,
            style=COMPLETION_STYLE,
            reserve_space_for_menu=6,
            bottom_toolbar=lambda: FormattedText(
                [
                    (theme.color(style), text)
                    for style, text in (
                        [(theme.MUTED, self.images.notice)] if self.images.notice else self._shown().status.toolbar()
                    )
                ]
            ),
        )
        prompt.key_bindings = self.images.bindings()
        return prompt


def _color_system(console: Console) -> Literal['standard', '256', 'truecolor', 'windows'] | None:
    match console.color_system:
        case 'standard' | '256' | 'truecolor' | 'windows' as system:
            return system
        case _:
            return None
