"""How pgtask runs each capability case and scenario: the cells of its support table measured so far.

pgtask doesn't run `DurableEngineSuite` yet (#12291), so it lists every cell measured elsewhere, `supported` ones
included: `BackgroundTools`' scenarios, which its own engine tests measure. `scripts/durable_exec_support.py` renders
the docs table from it, showing every other cell as untested. Plain data: it imports nothing from pgtask's SDK.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic_ai.durable_exec._support import Recovery, Support, ran_in_foreground

__all__ = ('RECOVERY', 'SUPPORT')

_FOREGROUND = (
    'the engine runs tool calls sequentially, so the tool runs in the foreground and returns its result inline'
)
_SEQUENTIAL_DEFAULT = f"{_FOREGROUND}; `parallel_execution_mode='sequential'` is the default"
_NO_DURABLE_WAIT = 'the engine has no durable wait for an outside answer, so a mid-run approval wait raises `UserError`'
_WAIT_REFUSED = 'does not support waiting for external events yet'

RECOVERY = Recovery()

SUPPORT: Mapping[str, Support] = {
    'BackgroundTools.runs_in_background': Support('limited', _SEQUENTIAL_DEFAULT, check=ran_in_foreground),
    'BackgroundTools.survives_approval_wait': Support('refused', _NO_DURABLE_WAIT, match=_WAIT_REFUSED),
}
"""The cells measured so far, by case or scenario name."""
