"""A run's pending compaction request, as [`RunContext.request_compaction`][pydantic_ai.tools.RunContext.request_compaction] makes it."""

from __future__ import annotations as _annotations

import dataclasses
import warnings

from . import messages as _messages

__all__ = ('CompactionRequests', 'RequestedCompaction')


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
            f'Compaction was requested (reason {requested.reason!r}), but no capability of this run can compact '
            "right now: none has a `compact_history` hook, or the one that does isn't loaded yet. The request was "
            'dropped and the history is unchanged. Add `NativeCompaction` or a summarizing compaction capability to '
            'compact.',
            UserWarning,
        )
