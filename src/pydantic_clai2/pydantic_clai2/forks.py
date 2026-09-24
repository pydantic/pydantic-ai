"""`/fork`: run a copy of the conversation in the background while you keep working.

Adapted from Code Puppy's `fork` plugin. A fork is a true fork: the retained
history is copied at the moment `/fork` runs, and the copy seeds a separate
`Session` with its own saved conversation. Later foreground turns never reach
the fork, and the fork never changes the foreground history. When there is no
history yet, or the copy fails, the fork starts with a fresh context.

Commands run between turns, so a `/fork` typed during a turn is queued like any
other command. Completion output waits until no turn or command owns the
terminal, then prints one fork banner, the Markdown response, and the saved
session id that `/resume` continues. Cancelling a turn cancels running forks;
exiting or reloading CLAI cancels them too.
"""

import asyncio
import copy
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar

from anyio import move_on_after
from pydantic_ai import AgentStreamEvent, PartStartEvent, TextPart
from pydantic_ai.messages import ModelMessage
from rich.console import Console
from rich.table import Table
from rich.text import Text

from . import theme
from ._rendering import StreamRenderer
from ._session import Session
from .errors import error_message
from .plugins import HostEvent, TurnEnd, TurnStart
from .status import Status
from .tool_output import terminal_text

DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT')

ForkStatus = Literal['running', 'done', 'failed', 'cancelled']

USAGE = (
    'Usage: /fork [@model] PROMPT   run a copy of this conversation in the background\n'
    '       /fork cancel ID         stop a running fork\n'
    '       /forks                  list forks'
)
_STATUS_STYLES: dict[ForkStatus, str] = {
    'running': theme.WARNING,
    'done': theme.SUCCESS,
    'failed': theme.ERROR,
    'cancelled': theme.MUTED,
}
_PROMPT_PREVIEW = 60


@dataclass(kw_only=True)
class ForkRecord:
    """Bookkeeping for one background run."""

    fork_id: int
    model: str
    prompt: str
    started_at: float
    task: 'asyncio.Task[None]'
    status: ForkStatus = 'running'
    elapsed: float | None = None
    session_id: str | None = field(default=None)
    progress: Status = field(default_factory=lambda: Status(activity='starting'))
    """The child's live activity, fed by its stream events, for the editor's fork rows."""
    announced: bool = False

    @property
    def tag(self) -> str:
        """How messages name this fork."""
        return f'fork #{self.fork_id}'


async def _ignore(event: HostEvent) -> None:
    pass


def _activity_style(activity: str) -> str:
    # Matches Code Puppy's sub-agent rows: thinking, tool calls, and writing each get a colour.
    if activity == 'thinking':
        return theme.sgr(theme.THINKING)
    if activity.startswith(('tool: ', 'running: ')):
        return theme.sgr(theme.WARNING)
    if activity == 'responding':
        return theme.sgr(theme.SUCCESS)
    return theme.sgr(theme.MUTED)


def parse_fork_args(text: str) -> tuple[str | None, str]:
    """Split `[@model] prompt`; no `@model` means the foreground model."""
    if not text.startswith('@'):
        return None, text
    model, *rest = text.split(maxsplit=1)
    return model[1:] or None, ''.join(rest).strip()


class Forks(Generic[DepsT, OutputT]):
    """One shell's forks. `spawn` builds a child session configured like the foreground one."""

    def __init__(
        self,
        *,
        console: Console,
        history: Callable[[], Sequence[ModelMessage]],
        spawn: Callable[[str | None, Sequence[ModelMessage]], Session[DepsT, OutputT]],
        models: Callable[[], Iterable[str]] = lambda: (),
        fire: Callable[[HostEvent], Awaitable[None]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind the foreground history, child-session factory, and the shell's plugin hooks.

        `fire` receives each fork's `TurnStart` and `TurnEnd`, like a foreground turn.
        """
        self.console = console
        self._history = history
        self._spawn = spawn
        self._models = models
        self._fire = fire or _ignore
        self._clock = clock
        self._store_ready = False
        self._records: dict[int, ForkRecord] = {}
        self._busy = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._terminal = asyncio.Lock()
        self._held: list[tuple[str, str]] = []

    @property
    def records(self) -> tuple[ForkRecord, ...]:
        """Every fork started by this shell, oldest first."""
        return tuple(self._records.values())

    @asynccontextmanager
    async def busy(self) -> AsyncGenerator[None]:
        """Hold fork output while a turn or command owns the terminal.

        Entering waits for a fork announcement already rendering, so the two never interleave.
        """
        async with self._terminal:
            self._busy += 1
            self._idle.clear()
        try:
            yield
        finally:
            self._busy -= 1
            if not self._busy:
                self._idle.set()
                held, self._held = self._held, []
                for text, style in held:
                    self._print(text, style)

    def _notify(self, text: str, style: str) -> None:
        # One-line notices print now when idle; otherwise `busy()` prints them on release.
        if self._busy:
            self._held.append((text, style))
        else:
            self._print(text, style)

    def _print(self, text: str, style: str) -> None:
        self.console.print(text, style=theme.color(style), markup=False)
        self.console.print()

    async def fork_command(self, args: list[str]) -> str:
        """`/fork`: receives the unparsed argument text so prompts keep their quotes."""
        text = args[0].strip() if args else ''
        if not text:
            return USAGE
        head, *rest = text.split(maxsplit=1)
        if head == 'cancel':
            return self.cancel(''.join(rest).strip())
        model, prompt = parse_fork_args(text)
        if not prompt:
            raise ValueError('Fork what, exactly? Usage: /fork [@model] PROMPT')
        return await self.start(prompt, model=model)

    def complete(self, args: list[str]) -> Iterable[str]:
        """Suggest `cancel` and saved models as `@model` for the first argument."""
        if len(args) <= 1:
            return ('cancel', *(f'@{name}' for name in self._models()))
        return ()

    async def start(self, prompt: str, *, model: str | None = None) -> str:
        """Apply `turn_start`, copy the history now, then run the child without waiting for it.

        A plugin that cancels or rejects the prompt refuses the fork, as it would refuse a turn.
        """
        begin = TurnStart(text=prompt)
        # Every `turn_start` gets its `turn_end`, as in the foreground, so plugins can close per-turn state.
        try:
            await self._fire(begin)
        except Exception as exc:
            await self._fire(TurnEnd(text=begin.text, outcome='failed', error=exc))
            raise
        if begin.cancelled:
            await self._fire(TurnEnd(text=begin.text, outcome='cancelled'))
            raise ValueError(f'Fork cancelled by a plugin: {begin.cancel_reason or "no reason given"}')
        prompt = begin.text
        try:
            session = await self._prepare(model)
        except Exception as exc:
            await self._fire(TurnEnd(text=prompt, outcome='failed', error=exc))
            raise
        fork_id = len(self._records) + 1
        progress = Status(activity='starting')

        async def observe(event: AgentStreamEvent) -> None:
            progress.observe(event)

        session.on_stream_event = observe
        record = ForkRecord(
            fork_id=fork_id,
            model=session.model or 'agent default',
            prompt=prompt,
            started_at=self._clock(),
            task=asyncio.create_task(self._run(fork_id, session, prompt), name=f'fork-{fork_id}'),
            progress=progress,
        )
        self._records[fork_id] = record
        return (
            f'fork #{fork_id} ({record.model}) started in the background. '
            'Results print when it finishes; /forks shows status.'
        )

    async def _prepare(self, model: str | None) -> Session[DepsT, OutputT]:
        session = self._spawn(model, self._snapshot())
        if session.conversations is not None and not self._store_ready:
            # Concurrent first connections to a brand-new sessions.db can fail on the store's WAL switch.
            # Open it here, while the command owns the shell and before any fork runs.
            await session.conversations.listing(limit=1)
            self._store_ready = True
        return session

    def _snapshot(self) -> list[ModelMessage]:
        # Deep copy, so nothing the fork does to its messages can reach the foreground history.
        try:
            return copy.deepcopy(list(self._history()))
        except Exception:  # noqa: BLE001 -- a failed copy must not block the fork.
            self.console.print(
                "/fork couldn't copy the current conversation. Forking with a fresh context.",
                style=theme.color(theme.WARNING),
            )
            return []

    async def _run(self, fork_id: int, session: Session[DepsT, OutputT], prompt: str) -> None:
        try:
            result = await session.prompt(prompt)
        except asyncio.CancelledError:
            record = self._finish(fork_id, 'cancelled', session)
            self._notify(f'{record.tag} cancelled after {record.elapsed:.1f}s', theme.MUTED)
            with move_on_after(5, shield=True):
                await self._fire(TurnEnd(text=prompt, outcome='cancelled'))
            raise
        except Exception as exc:  # noqa: BLE001 -- a failed fork reports and never reaches the shell.
            record = self._finish(fork_id, 'failed', session)
            first_line = (error_message(exc).strip().splitlines() or [''])[0]
            self._notify(
                f'{record.tag} failed after {record.elapsed:.1f}s: {type(exc).__name__}: {first_line}', theme.ERROR
            )
            await self._fire(TurnEnd(text=prompt, outcome='failed', error=exc))
            return
        record = self._finish(fork_id, 'done', session)
        await self._fire(TurnEnd(text=prompt, outcome='completed', result=result))
        while True:
            await self._idle.wait()
            async with self._terminal:
                # A turn or command may have started between the wake-up and taking the lock.
                if not self._busy:
                    await self._announce(record, result.output)
                    return

    def _finish(self, fork_id: int, status: ForkStatus, session: Session[DepsT, OutputT]) -> ForkRecord:
        record = self._records[fork_id]
        record.status = status
        record.elapsed = self._clock() - record.started_at
        if session.conversations is not None and session.summary.revision:
            record.session_id = session.summary.id
        return record

    async def _announce(self, record: ForkRecord, output: OutputT) -> None:
        header = Text(f' FORK #{record.fork_id} RESPONSE ', style=f'bold white on {theme.color(theme.THINKING)}')
        header.append(' ')
        header.append(record.model, style=f'bold {theme.color(theme.INFO)}')
        record.announced = True
        self.console.print()
        self.console.print(header)
        if isinstance(output, str):
            renderer = StreamRenderer(self.console, stop_loading=lambda: None)
            await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart(content=output)))
            await renderer.finish()
        else:
            # Structured output prints as-is, like a foreground turn, rather than as Markdown.
            self.console.print(str(output), markup=False)
            self.console.print()
        done = f'{record.tag} finished in {record.elapsed:.1f}s.'
        if record.session_id is not None:
            done += f' Continue it with /resume {record.session_id}'
        self._print(done, theme.SUCCESS)

    def rows(self, glyph: str) -> tuple[str, ...]:
        """Live editor rows: one per running fork, and one per finished fork still waiting to print.

        `glyph` is the current frame of the user's `/spinner`, so forks animate with the main turn.
        """
        reset, muted = '\x1b[0m', theme.sgr(theme.MUTED)
        rows: list[str] = []
        for record in self._records.values():
            if record.status == 'running':
                seconds = self._clock() - record.started_at
                mark = theme.sgr(theme.ACCENT) + glyph
                activity = record.progress.activity
                state = _activity_style(activity) + terminal_text(activity)
            elif record.status == 'done' and not record.announced:
                seconds = record.elapsed or 0.0
                mark = theme.sgr(theme.SUCCESS) + '\u2713'
                state = muted + 'done, prints after this turn'
            else:
                continue
            minutes, secs = divmod(int(seconds), 60)
            rows.append(
                f'\x1b[7m{theme.sgr(theme.THINKING, bold=True)} FORK #{record.fork_id} {reset} '
                f'{theme.sgr(theme.INFO)}{terminal_text(record.model)}{reset}  '
                f'{mark}{reset} {muted}{minutes:02d}:{secs:02d}{reset}  {state}{reset}'
            )
        return tuple(rows)

    def cancel(self, raw_id: str) -> str:
        """`/fork cancel ID`."""
        try:
            fork_id = int(raw_id)
        except ValueError:
            raise ValueError(f"Usage: /fork cancel ID. '{raw_id}' is not a fork id.") from None
        record = self._records.get(fork_id)
        if record is None:
            raise ValueError(f'No fork #{fork_id}. Try /forks.')
        if record.status != 'running':
            return f'fork #{fork_id} already {record.status}.'
        record.task.cancel()
        return f'Cancelling {record.tag}...'

    def cancel_running(self) -> int:
        """Cancel every running fork, as a cancelled turn does. Returns how many were running."""
        running = [record for record in self._records.values() if record.status == 'running']
        for record in running:
            record.task.cancel()
        return len(running)

    async def close(self) -> None:
        """Cancel and join every fork task, including ones waiting to print."""
        tasks = [record.task for record in self._records.values() if not record.task.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def status_command(self, args: list[str]) -> str:
        """`/forks`: print the table and return the totals line."""
        if args:
            raise ValueError('Usage: /forks')
        if not self._records:
            return 'No forks yet. Start one with /fork [@model] PROMPT.'
        table = Table(title='Forks', header_style=theme.color(theme.ACCENT), border_style=theme.color(theme.MUTED))
        table.add_column('ID', justify='right')
        table.add_column('Model', style=theme.color(theme.INFO))
        table.add_column('Status')
        table.add_column('Time', justify='right')
        table.add_column('Session')
        table.add_column('Prompt', overflow='ellipsis', no_wrap=True, max_width=_PROMPT_PREVIEW)
        now = self._clock()
        for record in self._records.values():
            elapsed = record.elapsed if record.elapsed is not None else now - record.started_at
            table.add_row(
                str(record.fork_id),
                Text(record.model),
                Text(record.status, style=theme.color(_STATUS_STYLES[record.status])),
                f'{elapsed:.1f}s',
                Text(record.session_id or ''),
                Text(' '.join(record.prompt.split())),
            )
        self.console.print(table)
        counts = {status: 0 for status in _STATUS_STYLES}
        for record in self._records.values():
            counts[record.status] += 1
        return ', '.join(f'{count} {status}' for status, count in counts.items() if count)
