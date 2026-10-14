"""Keep the `run_code` REPL across agent runs in a store the host provides."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, Protocol

import anyio
from pydantic_monty import AsyncMontySession, MontyError

from pydantic_ai import RunContext
from pydantic_ai.tools import AgentDepsT
from pydantic_ai_harness._monty_exec import MontyRunState, call_monty

_logger = logging.getLogger(__name__)


class CodeModeReplStore(Protocol[AgentDepsT]):
    """Where `CodeMode` keeps the `run_code` REPL between agent runs.

    Each agent run starts its own REPL. With a store, a run that calls `run_code` first loads the
    state saved for it, and saves the REPL again when the run ends, so variables, imports, and
    function definitions carry over to the next run. Key the state by what identifies the
    conversation, usually `ctx.conversation_id`.

    The state is Monty's opaque dump of the REPL. Its size grows with the values the REPL holds;
    a store that limits it can drop an oversized state and forget the old one.
    """

    async def load(self, ctx: RunContext[AgentDepsT]) -> bytes | None:
        """Return the state saved for this run, or `None` to start a fresh REPL."""
        ...

    async def save(self, ctx: RunContext[AgentDepsT], state: bytes | None) -> None:
        """Keep `state` for the next run; `None` means `restart: true` cleared the REPL, so forget it."""
        ...


@dataclass
class PersistentRepl(Generic[AgentDepsT]):
    """One run's link between its REPL and a `CodeModeReplStore`, shared by the run's step copies."""

    store: CodeModeReplStore[AgentDepsT]
    ctx: RunContext[AgentDepsT] | None = None
    """The context of the run's first `run_code` call. Until it is set, nothing is loaded or saved."""
    pending: bytes | None = None
    """Loaded from the store and not yet restored into a session."""
    restarted: bool = False

    async def begin(self, ctx: RunContext[AgentDepsT]) -> None:
        """Load the saved state on the run's first `run_code` call."""
        if self.ctx is None:
            self.ctx = ctx
            self.pending = await self.store.load(ctx)

    async def restore(
        self,
        run_state: MontyRunState,
        session: AsyncMontySession,
        checkout: Callable[[], Awaitable[AsyncMontySession]],
    ) -> AsyncMontySession:
        """Load the pending state into `session`, the run's first, and return the session to feed.

        A state Monty cannot load, such as a truncated dump or one from an incompatible Monty
        version, is dropped, and the run continues in a fresh session from `checkout`.
        """
        state, self.pending = self.pending, None
        if state is None:
            return session
        try:
            await call_monty(run_state.portal, session.load_session, state)
        except MontyError as error:
            _logger.warning('CodeMode could not restore the saved REPL state, so this run starts fresh: %s', error)
            # A session that failed to load refuses another load, so check out a new one.
            await run_state.reset()
            return await checkout()
        # Later snippets are not type-checked; the checker would reject the restored names.
        run_state.has_executed_feed = True
        return session

    def restart(self, ctx: RunContext[AgentDepsT]) -> None:
        """Record `restart: true`: the saved state is not restored, and is forgotten if nothing replaces it."""
        self.ctx = ctx
        self.pending = None
        self.restarted = True

    async def save(self, run_state: MontyRunState) -> None:
        """Save the REPL as the run leaves it, before the run state closes.

        Shielded, so a cancelled run still saves. A REPL another failure reset (a timeout, a
        crash, a cancelled snippet) keeps the state saved before this run.
        """
        ctx = self.ctx
        if ctx is None:
            return
        with anyio.CancelScope(shield=True):
            session = run_state.session
            if session is None:
                if self.restarted:
                    await self.store.save(ctx, None)
                return
            try:
                state = await call_monty(run_state.portal, session.dump)
            # Monty dumps idle and suspended sessions alike; this guards a worker lost after the last snippet.
            except MontyError as error:  # pragma: no cover
                _logger.warning('CodeMode could not save the REPL state, so the next run keeps the last one: %s', error)
                return
            await self.store.save(ctx, state)
