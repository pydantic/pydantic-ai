"""A plugin-owned session root, shared by UI events and agent instrumentation."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from uuid import uuid4

import logfire
from anyio import move_on_after, run_process
from opentelemetry.context import Context
from opentelemetry.trace import Span

from pydantic_ai import AgentRunResult, RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, Instrumentation, WrapRunHandler
from pydantic_clai2.plugins import SessionEndReason
from pydantic_clai2.ui.telemetry import SCOPE, parent_span


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

    def start(self, email: str | None) -> None:
        """Open the current conversation's root; `email`, when known, identifies the user on roots only."""
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

    async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[object]:
        with parent_span(self.root()):
            return await handler()


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
