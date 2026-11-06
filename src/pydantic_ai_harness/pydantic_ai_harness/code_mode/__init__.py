"""Route tool calls through a sandboxed Python environment with `CodeMode` or `CodeModeToolset`.

Sandbox factories return native pools or compatible adapters: an async context manager with
`checkout` yielding native Monty sessions. Adapters do not need to inherit a native pool class.
"""

from pydantic_ai_harness._code_tools import FOLD_INTO_CODE_MODE
from pydantic_ai_harness.code_mode._capability import CodeMode
from pydantic_ai_harness.code_mode._events import (
    CODE_MODE_EVENTS,
    SpeculativeCallClaimedEvent,
    SpeculativeCallEvictedEvent,
    SpeculativeCallLaunchedEvent,
    SpeculativeCallMissedEvent,
    SpeculativeCallSettledEvent,
    SpeculativeCodeUpdateEvent,
)
from pydantic_ai_harness.code_mode._parts import RunCodeArgs, RunCodeCallPart
from pydantic_ai_harness.code_mode._repl_store import CodeModeReplStore
from pydantic_ai_harness.code_mode._speculation import SpeculationStats
from pydantic_ai_harness.code_mode._toolset import (
    CodeModeMount,
    CodeModeOS,
    CodeModeOSCallback,
    CodeModeResourceLimits,
    CodeModeReturnSchemaWarning,
    CodeModeToolset,
)

__all__ = [
    'CODE_MODE_EVENTS',
    'FOLD_INTO_CODE_MODE',
    'CodeMode',
    'CodeModeMount',
    'CodeModeOS',
    'CodeModeOSCallback',
    'CodeModeReplStore',
    'CodeModeResourceLimits',
    'CodeModeReturnSchemaWarning',
    'CodeModeToolset',
    'RunCodeArgs',
    'RunCodeCallPart',
    'SpeculationStats',
    'SpeculativeCallClaimedEvent',
    'SpeculativeCallEvictedEvent',
    'SpeculativeCallLaunchedEvent',
    'SpeculativeCallMissedEvent',
    'SpeculativeCallSettledEvent',
    'SpeculativeCodeUpdateEvent',
]
