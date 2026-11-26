"""How Render runs each capability case and scenario: the cells of its support table measured so far.

Render doesn't run `DurableEngineSuite` yet (#12291), so it lists every cell measured elsewhere, `supported` ones
included: `BackgroundTools`' scenarios, which its own engine tests measure. `scripts/durable_exec_support.py` renders
the docs table from it, showing every other cell as untested. Plain data: it imports nothing from Render's SDK.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic_ai.durable_exec._support import Recovery, Support

__all__ = ('RECOVERY', 'SUPPORT')

_NO_DURABLE_WAIT = 'the engine has no durable wait for an outside answer, so a mid-run approval wait raises `UserError`'
_WAIT_REFUSED = 'does not support waiting for external events yet'

RECOVERY = Recovery(
    'restart',
    'retrying the entry task run restarts `agent.run(...)`, so finished model and tool calls run again',
    issue=14815,
)

SUPPORT: Mapping[str, Support] = {
    'BackgroundTools.runs_in_background': Support('supported'),
    'BackgroundTools.survives_approval_wait': Support('refused', _NO_DURABLE_WAIT, match=_WAIT_REFUSED),
}
"""The cells measured so far, by case or scenario name."""
