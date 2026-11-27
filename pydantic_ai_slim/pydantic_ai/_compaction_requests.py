"""A run's pending compaction request, as [`RunContext.request_compaction`][pydantic_ai.tools.RunContext.request_compaction] makes it."""

from __future__ import annotations as _annotations

import dataclasses
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

from . import messages as _messages

__all__ = ('CompactionRequests', 'RequestedCompaction', 'claim_compaction_only_run', 'compaction_only_run')


@dataclasses.dataclass
class RequestedCompaction:
    """A compaction asked for with [`RunContext.request_compaction`][pydantic_ai.tools.RunContext.request_compaction]."""

    reason: _messages.CompactionReason
    focus: str | None = None


_COMPACTION_REASON_PRIORITY: tuple[str, ...] = (
    'manual',
    'model_requested',
    'context_overflow',
    'model_switch',
    'threshold',
    'cache_expiry',
)
"""The built-in reasons, most important first. A reason of your own ranks after all of them."""


def _reason_rank(reason: _messages.CompactionReason) -> int:
    try:
        return _COMPACTION_REASON_PRIORITY.index(reason)
    except ValueError:
        return len(_COMPACTION_REASON_PRIORITY)


@dataclasses.dataclass
class CompactionRequests:
    """A run's pending compaction request, shared by reference by every `RunContext` of the run."""

    requested: RequestedCompaction | None = None
    warned_unserved: bool = False
    """Whether the run already warned that a request was dropped because no capability compacts."""

    def request(self, requested: RequestedCompaction) -> None:
        """Make `requested` pending, merged into the request already pending, if any.

        The merged request has the more important reason, by `_COMPACTION_REASON_PRIORITY`, and the earlier one's on
        a tie. It keeps every focus: both, the earlier one first and separated by a blank line, when they differ.
        """
        pending = self.requested
        if pending is None:
            self.requested = requested
            return
        reason = requested.reason if _reason_rank(requested.reason) < _reason_rank(pending.reason) else pending.reason
        focuses = [focus for focus in (pending.focus, requested.focus) if focus]
        focus = '\n\n'.join(dict.fromkeys(focuses)) or None
        self.requested = RequestedCompaction(reason=reason, focus=focus)

    def settle(self, requested: RequestedCompaction) -> None:
        """Drop `requested` once it's been acted on, unless a later request was merged into it in the meantime."""
        if self.requested is requested:
            self.requested = None

    def drop_unserved(self, requested: RequestedCompaction) -> None:
        """Drop `requested`, which no capability of the run can serve, warning the first time in the run."""
        self.settle(requested)
        if self.warned_unserved:
            return
        self.warned_unserved = True
        warnings.warn(
            f'Compaction was requested (reason {requested.reason!r}), but no capability of this agent has a '
            '`compact_history` hook, so the request was dropped and the history is unchanged. Add `NativeCompaction` '
            'or a summarizing compaction capability to compact.',
            UserWarning,
            stacklevel=2,
        )


@dataclasses.dataclass
class CompactionOnlyRun:
    """A pending `compaction_only_run`, which records whether an agent run took it on."""

    claimed: bool = False
    """Whether an agent run started in the context claimed it, and so ended after its compaction step."""


_compaction_only: ContextVar[CompactionOnlyRun | None] = ContextVar('pydantic_ai.compaction_only_run', default=None)
"""The pending `compaction_only_run` for the next agent run started in this context, if any."""


@contextmanager
def compaction_only_run() -> Generator[CompactionOnlyRun]:
    """Make the next agent run started in this context end after its compaction step.

    A context variable rather than a run method argument, so it reaches the run through every agent wrapper's run
    methods, like a durable agent's, unchanged. The yielded holder says whether a run claimed it: one whose `run()`
    never reached an `Agent` run didn't, and ran however it runs.
    """
    pending = CompactionOnlyRun()
    token = _compaction_only.set(pending)
    try:
        yield pending
    finally:
        _compaction_only.reset(token)


@contextmanager
def claim_compaction_only_run() -> Generator[bool]:
    """Claim the pending `compaction_only_run` for the run started in this context, until that run ends.

    A run started inside that run isn't one too. The flag is restored when the run ends, rather than consumed, so
    an engine that retries the code that started the run in the same context, like a Prefect flow retry, starts a
    compaction-only run again.
    """
    if (pending := _compaction_only.get()) is None:
        yield False
        return
    pending.claimed = True
    token = _compaction_only.set(None)
    try:
        yield True
    finally:
        _compaction_only.reset(token)
