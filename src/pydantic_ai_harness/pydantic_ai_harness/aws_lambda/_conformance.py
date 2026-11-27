"""How AWS Lambda runs each capability case and scenario: the cells of its support table measured so far.

AWS Lambda doesn't run `DurableEngineSuite` yet (#12291), so it lists every cell measured elsewhere, `supported` ones
included: `BackgroundTools`' scenarios, which its own engine tests measure. `scripts/durable_exec_support.py` renders
the docs table from it, showing every other cell as untested. Plain data: it imports nothing from AWS Lambda's SDK.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic_ai.durable_exec._support import Recovery, Support, ran_in_foreground

__all__ = ('RECOVERY', 'SUPPORT')

_FOREGROUND = (
    'the engine runs tool calls sequentially, so the tool runs in the foreground and returns its result inline'
)

RECOVERY = Recovery()

SUPPORT: Mapping[str, Support] = {
    'BackgroundTools.runs_in_background': Support('limited', _FOREGROUND, check=ran_in_foreground, issue=13206),
    'BackgroundTools.survives_approval_wait': Support(
        'limited', f'{_FOREGROUND}, so no call is in flight across the wait', check=ran_in_foreground, issue=13206
    ),
}
"""The cells measured so far, by case or scenario name."""
