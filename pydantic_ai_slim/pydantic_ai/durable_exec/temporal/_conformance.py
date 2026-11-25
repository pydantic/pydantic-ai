"""How Temporal runs each capability case and scenario: the cells of its support table that aren't `supported`.

`tests/harness/test_durable_temporal_replay.py` checks every cell with `DurableEngineSuite`, and
`scripts/durable_exec_support.py` renders the docs table from it. Plain data: it imports nothing from Temporal.
"""

from __future__ import annotations

from collections.abc import Mapping

from .._support import Recovery, Support

__all__ = ('RECOVERY', 'SUPPORT')

_IN_UNIT = 'is not supported inside a durable'
_EMIT_REFUSED = 'Emitting events from a tool or event stream handler is not supported'
_ID_LESS = Support(
    'refused',
    'its toolset has no id, which Temporal names the activity of a function tool after',
    match='need to have a unique `id` in order to be used with Temporal',
    issue=9233,
)
_ENQUEUE_IN_UNIT = Support(
    'refused',
    'a durable unit replays from its record without running the code again, so an enqueued message would be lost',
    match=_IN_UNIT,
    issue=7971,
)

RECOVERY = Recovery()
"""Temporal recovers a workflow by replaying its event history."""

SUPPORT: Mapping[str, Support] = {
    'BrowserUse': _ID_LESS,
    'CompactOnRequest': Support(
        'broken',
        'its `for_run` copy builds a new toolset, which Temporal refuses as one added at run time',
        match='cannot be added at runtime with Temporal',
        issue=14403,
    ),
    'Planning': Support(
        'refused',
        'its tools emit plan events, which Temporal refuses from an activity',
        match=_EMIT_REFUSED,
        issue=7971,
    ),
    'PydanticAIDocs': _ID_LESS,
    'RunContext.emit_from_tool': Support(
        'refused',
        'Temporal runs a function tool in an activity, which refuses an emitted event',
        match=_EMIT_REFUSED,
        issue=7971,
    ),
    'RunContext.enqueue_from_tool': Support(
        'refused',
        'Temporal runs a function tool in an activity, which refuses an enqueued message',
        match=_IN_UNIT,
        issue=7971,
    ),
    'RunContext.emit_in_unit': Support(
        'refused',
        'Temporal refuses an event emitted in an activity',
        match=_EMIT_REFUSED,
        issue=7971,
    ),
    'RunContext.enqueue_in_unit': _ENQUEUE_IN_UNIT,
}
"""The cells that aren't `supported`, by case or scenario name."""
