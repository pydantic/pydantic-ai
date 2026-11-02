"""Keep the `run_code` REPL across agent runs in a store the host provides."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Generic, Protocol

import anyio
from pydantic_monty import MontyError, ResourceLimits

from pydantic_ai import RunContext
from pydantic_ai.tools import AgentDepsT
from pydantic_ai_harness._monty_exec import MontyRunState, call_monty

_logger = logging.getLogger(__name__)

_STATE_HEADER = b'pydantic-ai-harness code-mode REPL v1\n'
"""Starts every saved state, ahead of the JSON `resource_limits` line and the Monty dump."""

SAVE_TIMEOUT_SECONDS = 10.0
"""How long the end of a run waits for the REPL dump and `CodeModeReplStore.save`.

The save is shielded so a cancelled run still keeps its REPL, and this bounds how long that
shield can hold up the cancellation, for a remote worker or a store that stops responding.
"""


class CodeModeReplStore(Protocol[AgentDepsT]):
    """Where `CodeMode` keeps the `run_code` REPL between agent runs.

    Each agent run starts its own REPL. With a store, a run that calls `run_code` first loads the
    state saved for it, and saves the REPL again when the run ends, so variables, imports, and
    function definitions carry over to the next run. Key the state by what identifies the
    conversation, usually `ctx.conversation_id`.

    The state is opaque: Monty's dump of the REPL, behind a header recording the `resource_limits`
    it was saved under, since a restored REPL keeps the limits of the session it came from. A state
    saved under other limits is dropped and the run starts fresh. Its size grows with the values the
    REPL holds; a store that limits it can drop an oversized state and forget the old one.

    Monty does not authenticate a dump, so return only states this application saved, from storage
    no one else can write to.

    A store error does not fail the run: a failed `load` starts a fresh REPL, and a failed `save`
    leaves what the store held before. A run whose `load` failed does not save, so the state it
    could not read is still there for the next run, unless it restarted the REPL with `restart: true`.
    A `save` still running after `SAVE_TIMEOUT_SECONDS` is cancelled; work it handed to a thread may
    still finish, so write atomically. Each is logged as a warning.
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
    limits: ResourceLimits
    """The run's resolved resource limits, which a restored REPL must have been saved under."""
    ctx: RunContext[AgentDepsT] | None = None
    """The context the REPL is saved under at the end of the run. While it is unset, nothing is saved."""
    began: bool = False
    """Set by the run's first `run_code` call, so the store is loaded at most once per run."""
    pending: bytes | None = None
    """Loaded from the store and not yet restored into a session."""
    restarted: bool = False

    async def begin(self, ctx: RunContext[AgentDepsT]) -> None:
        """Load the saved state on the run's first `run_code` call."""
        if self.began:
            return
        self.began = True
        try:
            self.pending = await self.store.load(ctx)
        except Exception as error:
            # Leave `ctx` unset: the store may still hold a state this run could not read, which its
            # fresh REPL must not replace. Only `restart: true` sets it again, to discard that state.
            _logger.warning(
                'CodeMode could not load the saved REPL state, so this run starts fresh '
                'and saves over it only on a restart: %s',
                error,
            )
            return
        self.ctx = ctx

    async def restore(self, run_state: MontyRunState, *, limits: ResourceLimits, in_temporal_workflow: bool) -> bool:
        """Load the pending state into the run's first session; return whether it was restored.

        A state saved under other resource limits, or one Monty cannot load, such as a truncated dump
        or one from an incompatible Monty version, is dropped, and the run continues in a fresh REPL,
        type-checked like any other.
        """
        if self.pending is None:
            return False
        dump = self._decode(self.pending)
        if dump is None:
            self.pending = None
            _logger.warning(
                'CodeMode is not restoring the saved REPL state, which was saved in another format or under '
                'other resource limits, so this run starts fresh'
            )
            return False
        session = await run_state.get_session(
            type_check=False, type_check_stubs=None, limits=limits, in_temporal_workflow=in_temporal_workflow
        )
        self.pending = None
        try:
            await call_monty(run_state.portal, session.load_session, dump)
        except MontyError as error:
            _logger.warning('CodeMode could not restore the saved REPL state, so this run starts fresh: %s', error)
            # A session that failed to load refuses another load, so the next checkout starts a new one.
            await run_state.reset()
            return False
        run_state.has_executed_feed = True
        return True

    def restart(self, ctx: RunContext[AgentDepsT]) -> None:
        """Record `restart: true`: the saved state is not restored, and is forgotten if nothing replaces it."""
        self.began = True
        self.ctx = ctx
        self.pending = None
        self.restarted = True

    def skip_save(self) -> None:
        """Leave the stored state as it is, for a run whose REPL may still be mid-snippet at exit."""
        self.ctx = None

    async def save(self, run_state: MontyRunState) -> None:
        """Save the REPL as the run leaves it, before the run state closes.

        Shielded, so a cancelled run still saves, and bounded by `SAVE_TIMEOUT_SECONDS`. A REPL a
        failure reset (a timeout, a crash, a cancelled snippet) and no later snippet rebuilt keeps
        the state saved before this run, as does a run whose `load` failed and that did not restart.
        """
        ctx = self.ctx
        if ctx is None:
            return
        with anyio.move_on_after(SAVE_TIMEOUT_SECONDS, shield=True) as scope:
            try:
                await self._save(ctx, run_state)
            except Exception as error:
                _logger.warning('CodeMode could not save the REPL state, so the store keeps the last one: %s', error)
        if scope.cancelled_caught:
            _logger.warning(
                'CodeMode stopped saving the REPL state after %s seconds, so the store keeps the last one',
                SAVE_TIMEOUT_SECONDS,
            )

    async def _save(self, ctx: RunContext[AgentDepsT], run_state: MontyRunState) -> None:
        session = run_state.session
        if session is None:
            if self.restarted:
                await self.store.save(ctx, None)
            return
        dump = await call_monty(run_state.portal, session.dump)
        await self.store.save(ctx, _STATE_HEADER + self._limits_line() + dump)

    def _limits_line(self) -> bytes:
        return json.dumps(self.limits, sort_keys=True).encode() + b'\n'

    def _decode(self, state: bytes) -> bytes | None:
        """The Monty dump in `state`, or `None` when it was saved in another format or under other limits."""
        prefix = _STATE_HEADER + self._limits_line()
        return state[len(prefix) :] if state.startswith(prefix) else None
