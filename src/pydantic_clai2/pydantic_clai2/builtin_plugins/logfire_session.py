"""A plugin-owned session root, shared by UI events and agent instrumentation."""

import asyncio
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from uuid import uuid4
from weakref import ref

import logfire
from anyio import move_on_after, run_process
from opentelemetry.context import Context
from opentelemetry.trace import Span, get_current_span

from pydantic_ai import AgentRunResult, RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, Instrumentation, WrapRunHandler
from pydantic_clai2.plugins import SessionEndReason
from pydantic_clai2.ui.telemetry import SCOPE, parent_span

_TRACINGS: list[ref['SessionTracing']] = []
"""Every started session tracing, weakly: each enabled copy of the `observability` plugin has one."""
ROOT_LIFETIME = 12 * 60 * 60
"""Seconds a session root stays open: Logfire rejects a span that started more than a day before it is exported."""


def _live_tracings() -> list['SessionTracing']:
    live = [tracing for weak in _TRACINGS if (tracing := weak()) is not None]
    _TRACINGS[:] = [ref(tracing) for tracing in live]
    return live


@dataclass(kw_only=True)
class SessionTracing(AbstractCapability[None]):
    """Keep agent runs under the session without replacing a nested run's parent."""

    instance: logfire.Logfire
    session_id: Callable[[], str | None]
    id: str | None = 'clai2_session_tracing'
    _email: str | None = field(default=None, init=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False)
    """The loop the session started on, while it is active."""
    _rotation: asyncio.TimerHandle | None = field(default=None, init=False)
    _roots: dict[str, Span] = field(default_factory=dict[str, Span], init=False)
    _fallback_id: str = field(default_factory=lambda: str(uuid4()), init=False)
    # Bounded: a nested run's error that its parent handled never reaches a turn's end to be looked up.
    _run_errors: deque[Exception] = field(default_factory=lambda: deque[Exception](maxlen=16), init=False)

    def start(self, email: str | None) -> None:
        """Open the current conversation's root; `email`, when known, identifies the user on roots only."""
        if self not in _live_tracings():
            _TRACINGS.append(ref(self))
        self._email = email
        self._loop = asyncio.get_running_loop()
        self.root()

    def root(self) -> Span | None:
        """Reuse a saved conversation's root, including when it is resumed later in this shell.

        A root ended by `ROOT_LIFETIME` is still returned inside its own trace, so a run that was under way keeps
        its nesting; anything else gets a new root for the same conversation.
        """
        loop = self._loop
        if loop is None:
            return None
        session_id = self._bind_identity()
        current = self._roots.get(session_id)
        if current is None or (
            not current.is_recording()
            and get_current_span().get_span_context().trace_id != current.get_span_context().trace_id
        ):
            root = (
                self.instance.config.get_tracer_provider()
                .get_tracer(SCOPE)
                .start_span(
                    'CLAI session',
                    context=Context(),
                    attributes={
                        'agent_session_id': session_id,
                        'logfire.msg': 'CLAI session',
                        'logfire.tags': self._tags(),
                        **self._user(),
                    },
                )
            )
            self._roots[session_id] = root
            # Also before `--resume` picks the conversation, or for a host with no saved conversations: the email
            # is known now, and binding the root to its conversation, when startup selects it, announces it again.
            self._announce(root, session_id)
            # Roots also open on worker threads, such as for a UI record, so the timer is set from the loop.
            loop.call_soon_threadsafe(self._schedule_rotation)
        return self._roots[session_id]

    def _schedule_rotation(self) -> None:
        if self._loop is not None and self._rotation is None:
            self._rotation = self._loop.call_later(ROOT_LIFETIME, self._rotate)

    def _rotate(self) -> None:
        """End every open root while Logfire still accepts it, even in a session left idle until exit."""
        self._rotation = None
        # A snapshot: a worker thread can open a root meanwhile.
        for span in tuple(self._roots.values()):
            if span.is_recording():
                span.end()

    def _tags(self) -> list[str]:
        return [self._email] if self._email else []

    def _user(self) -> dict[str, str]:
        return {'user.email': self._email} if self._email else {}

    def _announce(self, root: Span, session_id: str) -> None:
        """Log the root's identity now: a span is exported only when it ends, at exit or after `ROOT_LIFETIME`."""
        with parent_span(root):
            # `tags=`, not a `logfire.tags` attribute, which `log` would serialize to a JSON string.
            self.instance.log(
                'info',
                'CLAI session opened',
                attributes={'agent_session_id': session_id, **self._user()},
                tags=self._tags(),
            )

    def _bind_identity(self) -> str:
        session_id = self.session_id() or self._fallback_id
        if (
            session_id != self._fallback_id
            and (pending := self._roots.pop(self._fallback_id, None)) is not None
            and pending.is_recording()
        ):
            # Startup UI records can precede --resume selection; keep their parent and bind it once known.
            pending.set_attribute('agent_session_id', session_id)
            self._roots[session_id] = pending
            self._announce(pending, session_id)
        return session_id

    def end(self, reason: SessionEndReason) -> None:
        self._bind_identity()
        self._loop = None
        if self._rotation is not None:
            self._rotation.cancel()
            self._rotation = None
        _TRACINGS[:] = [weak for weak in _TRACINGS if weak() is not self]
        for span in self._roots.values():
            if span.is_recording():
                span.set_attribute('reason', reason)
                span.end()
        self._roots.clear()

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost', wraps=(Instrumentation,))

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[None]]) -> AbstractCapability[None]:
        """Match instrumentation's last-instance precedence, preserving that instance's live roots."""
        return capabilities[-1]

    def raised_in_run(self, error: BaseException) -> bool:
        """Whether `error` left an agent run, so the `Instrumentation` this wraps recorded it on the run's span.

        A match is forgotten, so the same exception raised again outside a run is not mistaken for this one.
        """
        for index, raised in enumerate(self._run_errors):
            if raised is error:
                del self._run_errors[index]
                return True
        return False

    async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[object]:
        with parent_span(self.root()):
            try:
                return await handler()
            except Exception as error:
                # Every enabled copy of the plugin looks the error up at turn end, but `combine` keeps only the last
                # copy's `wrap_run`, so this one tells them all.
                for tracing in _live_tracings():
                    tracing._run_errors.append(error)
                raise


async def git_email() -> str | None:
    """Read the configured Git author email, without making telemetry depend on Git."""
    with move_on_after(2):
        try:
            result = await run_process(['git', 'config', '--get', 'user.email'], check=False)
        except OSError:
            return None
        if result.returncode == 0:
            return result.stdout.decode(errors='replace').strip() or None
    return None
