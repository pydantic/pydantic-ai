"""A run's pending compaction request, as [`RunContext.request_compaction`][pydantic_ai.tools.RunContext.request_compaction] makes it."""

from __future__ import annotations as _annotations

import dataclasses

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
