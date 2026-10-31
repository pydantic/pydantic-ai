"""The live sessions of one CLAI process: which session holds which conversation, opening, closing, idle closing.

Every `SessionRuntime` joins its process's `LiveRuntimes` when it is built and leaves it when it closes.
The terminal's session and the session `clai2 --web` starts with run under their own driver. Sessions
opened with `open` or `new` run here, each as a task of `serving`, and close after an idle hour.
"""

import logging
import time
from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from anyio import CancelScope, Event, create_task_group, sleep
from anyio.abc import TaskGroup, TaskStatus

from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.plugins import LiveSession, SessionEndReason
from pydantic_clai2.plugins.process_plugins import JoinedSessions, LiveRuntime, handling
from pydantic_clai2.runtime.live_registry import LiveRegistry
from pydantic_clai2.runtime.session import continuing_fork
from pydantic_clai2.runtime.session_workspace import workspace_directory

IDLE_AFTER = 3600.0
"""Seconds a session opened with `open` or `new` stays live with nothing running, queued, or watching it."""
CHECK_EVERY = 60.0
"""Seconds between looks for idle sessions."""
_LOGGER = logging.getLogger(__name__)


class Served(Protocol):
    """A live session as `LiveRuntimes` runs it; `SessionRuntime` is one."""

    @property
    def live_session(self) -> LiveRuntime:
        """The session as `host.live` lists it."""
        ...

    @property
    def context(self) -> CommandContext:
        """Its settings."""
        ...

    @property
    def idle(self) -> bool:
        """Whether nothing runs or waits in it."""
        ...

    async def serve(self, *, resume: str | None, ready: Callable[[], None]) -> SessionEndReason:
        """Restore `resume`, load plugins, call `ready`, and take input until it ends or is cancelled."""
        ...


Builder = Callable[[Path | None], Served]
"""Builds a session in a directory, by default the one CLAI started in, ready to `serve`."""


@dataclass(kw_only=True)
class _Live:
    runtime: Served
    active: float
    """When it was last seen busy, watched, or changing, by the registry's clock."""
    scope: CancelScope | None = None
    """Cancels a session `serving` runs; `None` for one its own driver runs."""
    ready: bool = True
    """False while `open` or `new` loads it; until then it is not listed and not closed for being idle."""
    closing: bool = False
    """Set once it starts closing, so `open` waits for it to close rather than return it."""
    done: Event = field(default_factory=Event)
    """Set once it has closed."""


class LiveRuntimes(JoinedSessions):
    """The process's live sessions by `LiveSession.id`, as `host.live` reaches them; a conversation is held by one."""

    def __init__(
        self,
        *,
        conversations: SqliteConversationStore,
        clock: Callable[[], float] = time.monotonic,
        idle_after: float = IDLE_AFTER,
        check_every: float = CHECK_EVERY,
    ) -> None:
        """`clock`, `idle_after`, and `check_every` time idle closing."""
        self._live: dict[str, _Live] = {}
        super().__init__(lambda: [live.runtime.live_session for live in self._live.values()])
        self._conversations = conversations
        self.clock = clock
        self.idle_after = idle_after
        self.check_every = check_every
        self.build: Builder | None = None
        """Builds the sessions `open` and `new` start; `None` where CLAI opens no others, as with `-p`."""
        self.focused: str | None = None
        """The session the terminal shows, never closed for being idle; `None` while no terminal shows one."""
        self._group: TaskGroup | None = None
        self._opening: dict[str, Event] = {}
        """Conversations being opened or resumed, set once that is done."""
        self._switching: dict[str, Event] = {}
        """Sessions resuming another conversation, by ID, set once they have."""
        self._building = False
        self.registry = LiveRegistry(self.listing)

    def listing(self) -> list[LiveSession]:
        """Every live session, oldest first, leaving out those `open` and `new` are still loading."""
        return [live.runtime.live_session for live in self._live.values() if live.ready]

    def add(self, runtime: Served) -> None:
        """Count `runtime` among the live sessions, following the settings the others save; once is enough."""
        if runtime.live_session.id in self._live:
            return
        runtime.context.saved = lambda key: self._saved(runtime, key)
        self._live[runtime.live_session.id] = _Live(runtime=runtime, active=self.clock(), ready=not self._building)
        self.changed(runtime.live_session)

    def leaving(self, runtime: Served) -> None:
        """Mark `runtime` closing: it still holds its conversation, but `open` waits for it to close."""
        if (live := self._live.get(runtime.live_session.id)) is not None:
            live.closing = True
            self.changed(runtime.live_session)

    def remove(self, runtime: Served) -> None:
        """Stop counting `runtime`, once it has closed."""
        if (live := self._live.pop(runtime.live_session.id, None)) is not None:
            live.done.set()
            self.changed(runtime.live_session)

    def changed(self, session: LiveRuntime) -> None:
        if (live := self._live.get(session.id)) is not None:
            live.active = self.clock()
        super().changed(session)

    def holding(self, conversation_id: str) -> LiveSession | None:
        """The live session holding `conversation_id`, even one still opening or closing."""
        live = self._holder(conversation_id)
        return live.runtime.live_session if live is not None else None

    def runtime(self, conversation_id: str) -> Served | None:
        """The live session holding `conversation_id`, unless it is still opening or already closing."""
        live = self._holder(conversation_id)
        return live.runtime if live is not None and live.ready and not live.closing else None

    @contextmanager
    def driving(self, runtime: Served) -> Generator[None]:
        """Let `close`, idle closing included, end the loop its own driver runs for `runtime` in the block.

        The terminal runs the session CLAI started with so; it is closed only while the terminal shows another.
        """
        live = self._live[runtime.live_session.id]
        try:
            with CancelScope() as live.scope:
                yield
        finally:
            live.scope = None

    def _holder(self, conversation_id: str) -> _Live | None:
        return next(
            (
                live
                for live in self._live.values()
                if live.runtime.live_session.conversation.conversation_id == conversation_id
            ),
            None,
        )

    @asynccontextmanager
    async def serving(self) -> AsyncGenerator[None]:
        """Run the sessions `open` and `new` start, and close idle ones; leaving closes them all.

        What the block raises propagates as it is, not in an exception group.
        """
        try:
            async with AsyncExitStack() as stack:
                try:
                    await stack.enter_async_context(self.registry.serving(self.subscribe))
                except OSError as exc:
                    _LOGGER.warning('Live session discovery unavailable: %s', exc)
                async with create_task_group() as group:
                    # Set at once: a browser may open a session as soon as the web UI it serves is up.
                    self._group = group
                    group.start_soon(self._close_idle_ones)
                    try:
                        # A new sessions database switches to WAL on first use, which fails at once when two sessions'
                        # first saves race, so make it before the sessions opened from here run.
                        await self._conversations.listing(limit=1)
                        yield
                    finally:
                        self._group = None
                        group.cancel_scope.cancel()
        except BaseExceptionGroup as errors:
            if len(errors.exceptions) == 1:
                raise errors.exceptions[0] from None
            raise  # pragma: no cover -- the block and idle closing never both raise.

    async def open(self, conversation_id: str, /) -> LiveSession:
        """The live session holding `conversation_id`, else a new one resuming it.

        One another process runs continues in the fork of it an earlier open saved, in any CLAI, if there is one.
        """
        while True:
            live = self._holder(conversation_id)
            switching = self._switching.get(live.runtime.live_session.id) if live is not None else None
            if live is not None and live.ready and not live.closing and switching is None:
                return live.runtime.live_session
            waiting = (
                self._opening.get(conversation_id)
                or switching
                or (live.done if live is not None and live.closing else None)
            )
            if waiting is None:
                break
            await waiting.wait()
        opened = self._opening[conversation_id] = Event()
        try:
            saved = await self._conversations.get(conversation_id=conversation_id)
            continuing = await continuing_fork(self._conversations, saved.summary)
            if continuing is None:
                return await self._start(workspace_directory(saved.summary.workspace), resume=conversation_id)
        finally:
            del self._opening[conversation_id]
            opened.set()
        return await self.open(continuing.conversation_id)

    async def new(self, *, workspace: Path | None = None) -> LiveSession:
        return await self._start(None if workspace is None else workspace_directory(str(workspace)), resume=None)

    @contextmanager
    def claim(self, conversation_id: str, *, holder: LiveSession) -> Generator[None]:
        """Reserve `conversation_id` while `holder` resumes it, so `open` waits and no other session resumes it.

        Raises `ValueError` when another live session holds it, or is opening or resuming it, or when
        `holder` is already resuming another. Claiming again what `holder` claimed reserves nothing more.
        """
        switching = self._switching.get(holder.id)
        if switching is not None and switching is self._opening.get(conversation_id):
            yield
            return
        if switching is not None:
            raise ValueError('This session is already switching to another conversation.')
        held = self.holding(conversation_id)
        if (held is not None and held is not holder) or conversation_id in self._opening:
            raise ValueError(f'Session {conversation_id} is open in another live session of this CLAI.')
        claimed = self._opening[conversation_id] = self._switching[holder.id] = Event()
        try:
            yield
        finally:
            del self._opening[conversation_id], self._switching[holder.id]
            claimed.set()

    async def close(self, live_id: str, /) -> None:
        live = self._live.get(live_id)
        if live is None or not live.ready:
            raise LookupError(f'No live session {live_id}')
        if live.scope is None or live_id == self.focused:
            raise ValueError(
                f'Live session {live_id} runs under its own driver, such as the terminal; it closes there.'
            )
        live.closing = True
        live.scope.cancel()
        await live.done.wait()

    async def close_idle(self) -> None:
        """Close each session `serving` runs that has been idle and unwatched for `idle_after` seconds."""
        now = self.clock()
        for live_id, live in list(self._live.items()):
            # A session may have closed by itself while an earlier one closed.
            if self._live.get(live_id) is not live or live.scope is None or live.closing or not live.ready:
                continue
            if live_id == self.focused or live.runtime.live_session.watchers or not live.runtime.idle:
                live.active = now
            elif now - live.active >= self.idle_after:
                await self.close(live_id)

    async def _close_idle_ones(self) -> None:
        while True:
            await sleep(self.check_every)
            await self.close_idle()

    async def _start(self, workspace: Path | None, *, resume: str | None) -> LiveSession:
        if self._group is None or self.build is None:
            return await super().new(workspace=workspace)
        # What `build` adds is not listed until it has loaded.
        self._building = True
        try:
            runtime = self.build(workspace)
        finally:
            self._building = False
        if resume is None:
            return await self._group.start(self._serve, runtime, resume)
        # The conversation `open` reserved is this session's to resume.
        self._switching[runtime.live_session.id] = self._opening[resume]
        try:
            return await self._group.start(self._serve, runtime, resume)
        finally:
            del self._switching[runtime.live_session.id]

    async def _serve(self, runtime: Served, resume: str | None, *, task_status: TaskStatus[LiveSession]) -> None:
        live = self._live[runtime.live_session.id]

        def started() -> None:
            live.ready = True
            task_status.started(runtime.live_session)
            self.changed(runtime.live_session)

        try:
            # Its task starts in the caller's context, which may name another session as the one handled.
            with CancelScope() as live.scope, handling(runtime.live_session):
                await runtime.serve(resume=resume, ready=started)
        except Exception:
            if not live.ready:
                raise
            # One session failing must not end the others.
            _LOGGER.exception('Live session %s failed', runtime.live_session.id)

    def _saved(self, origin: Served, key: str) -> None:
        """Have every other live session follow a setting `origin` saved; `model` and `run.*` stay per session."""
        if key == 'model' or key.startswith('run.'):
            return
        for live in list(self._live.values()):
            if live.runtime is not origin:
                try:
                    live.runtime.context.adopt(key, origin.context.settings)
                except Exception:
                    # Saved and applied where it was set; one session failing to follow must not undo that.
                    _LOGGER.exception('Live session %s did not follow %s', live.runtime.live_session.id, key)
