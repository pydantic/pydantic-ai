"""How DBOS runs each capability case and scenario: the cells of its support table that aren't `supported`.

`tests/harness/test_durable_tool_io.py` checks every cell with `DurableEngineSuite`, and
`scripts/durable_exec_support.py` renders the docs table from it. Plain data: it imports nothing from DBOS.
"""

from __future__ import annotations

from collections.abc import Mapping

from .._support import Recovery, Support

__all__ = ('RECOVERY', 'SUPPORT')

_WAIT_REFUSED = 'does not support waiting for external events yet'
_ENQUEUE_IN_UNIT_MATCH = r'`ctx\.enqueue\(\)` is not supported inside a durable'

RECOVERY = Recovery()
"""DBOS recovers a workflow by replaying its recorded steps."""

SUPPORT: Mapping[str, Support] = {
    'BackgroundTools.survives_approval_wait': Support(
        'broken',
        'a mid-run approval wait raises `UserError` under the engine, so no background task can keep running across it',
        match=_WAIT_REFUSED,
        issue=14443,
    ),
    'IPythonCodeMode': Support(
        'refused',
        "it refuses durable execution, as an engine can't replay a kernel's state",
        match='`IPythonCodeMode` is not supported under durable execution',
        issue=14494,
    ),
    'RunContext.emit_in_unit': Support(
        'broken',
        'an event emitted in a step is not emitted again when recovery replays the step',
        match=r"recovery lost \['event:unit'\]",
        issue=7971,
    ),
    'RunContext.enqueue_in_unit': Support(
        'refused',
        'a durable unit replays from its record without running the code again, so an enqueued message would be lost',
        match=_ENQUEUE_IN_UNIT_MATCH,
        issue=7971,
    ),
}
"""The cells that aren't `supported`, by case or scenario name."""
