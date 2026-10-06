"""A plugin-owned session root, shared by UI events and agent instrumentation."""

import os
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from uuid import uuid4

import logfire
from anyio import move_on_after, run_process
from opentelemetry.context import Context
from opentelemetry.trace import Span

from pydantic_ai import AgentRunResult, RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, Instrumentation, WrapRunHandler
from pydantic_clai2.plugins import SessionEndReason
from pydantic_clai2.ui.telemetry import PROMPT_SOURCE, PROMPT_SOURCE_ATTRIBUTE, SCOPE, parent_span

_IN_RUN: ContextVar[bool] = ContextVar('clai2_in_run', default=False)
"""Set inside a run, so a run started within it (a sub-agent's) is attributed as one."""


@dataclass(kw_only=True)
class SessionTracing(AbstractCapability[None]):
    """Keep agent runs under the session without replacing a nested run's parent."""

    instance: logfire.Logfire
    session_id: Callable[[], str | None]
    id: str | None = 'clai2_session_tracing'
    team: str | None = None
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

    @property
    def email(self) -> str | None:
        """The user's email once the session started, when `user_tag` names one."""
        return self._email

    def identity(self) -> dict[str, str]:
        """Who is running: baggage on every span of a run, and the attributes fleet targeting matches on."""
        session_id = self.session_id()
        return {
            **({'user.email': self._email} if self._email else {}),
            **({'clai2.team': self.team} if self.team else {}),
            **({'agent_session_id': session_id} if session_id else {}),
            # Hackathon: test sessions say so, so fleet analysis (the miner) can leave them out.
            **({'clai2.test': 'true'} if os.getenv('CLAI2_TEST') else {}),
        }

    async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[object]:
        # Hackathon: the fleet control plane groups traces by user and team, so identity rides on every span.
        source = 'subagent' if _IN_RUN.get() else PROMPT_SOURCE.get()
        nested = _IN_RUN.set(True)
        try:
            with parent_span(self.root()), logfire.set_baggage(**self.identity(), **{PROMPT_SOURCE_ATTRIBUTE: source}):
                return await handler()
        finally:
            _IN_RUN.reset(nested)

    def ui_identity(self) -> dict[str, str]:
        """What every UI record carries: the user, the team, and the session, so no join with the root is needed."""
        return self.identity()


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
