"""A plugin-owned session root, shared by UI events and agent instrumentation."""

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from uuid import uuid4
from weakref import ref

import logfire
from anyio import move_on_after, run_process
from opentelemetry.context import Context
from opentelemetry.trace import Span

from pydantic_ai import AgentRunResult, RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, Instrumentation, WrapRunHandler
from pydantic_clai2.plugins import SessionEndReason
from pydantic_clai2.ui.telemetry import SCOPE, parent_span

_TRACINGS: list[ref['SessionTracing']] = []
"""Every started session tracing, weakly: each enabled copy of the `observability` plugin has one."""


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
    _active: bool = field(default=False, init=False)
    _roots: dict[str, Span] = field(default_factory=dict[str, Span], init=False)
    _fallback_id: str = field(default_factory=lambda: str(uuid4()), init=False)
    # Bounded: a nested run's error that its parent handled never reaches a turn's end to be looked up.
    _run_errors: deque[Exception] = field(default_factory=lambda: deque[Exception](maxlen=16), init=False)

    def start(self, email: str | None) -> None:
        """Open the current conversation's root; `email`, when known, identifies the user on roots only."""
        if self not in _live_tracings():
            _TRACINGS.append(ref(self))
        self._email = email
        self._active = True
        self.root()

    def root(self) -> Span | None:
        """Reuse a saved conversation's root, including when it is resumed later in this shell."""
        if not self._active:
            return None
        session_id = self._bind_identity()
        if session_id not in self._roots:
            self._roots[session_id] = (
                self.instance.config.get_tracer_provider()
                .get_tracer(SCOPE)
                .start_span(
                    'CLAI session',
                    context=Context(),
                    attributes={
                        'agent_session_id': session_id,
                        'logfire.msg': 'CLAI session',
                        'logfire.tags': [self._email] if self._email else [],
                        **({'user.email': self._email} if self._email else {}),
                    },
                )
            )
        return self._roots[session_id]

    def _bind_identity(self) -> str:
        session_id = self.session_id() or self._fallback_id
        if session_id != self._fallback_id and (pending := self._roots.pop(self._fallback_id, None)) is not None:
            # Startup UI records can precede --resume selection; keep their parent and bind it once known.
            pending.set_attribute('agent_session_id', session_id)
            self._roots[session_id] = pending
        return session_id

    def end(self, reason: SessionEndReason) -> None:
        self._bind_identity()
        self._active = False
        _TRACINGS[:] = [weak for weak in _TRACINGS if weak() is not self]
        for span in self._roots.values():
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
