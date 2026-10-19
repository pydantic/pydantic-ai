"""Tool-call error recovery for Pydantic AI agents.

A failed tool call is classified into one of four outcomes: call the tool again, report a
terminal failure to the model, return a substitute result, or propagate the exception. Every
recovery is logged, so a failure the model no longer sees stays visible to the operator.

Public API:

- `ToolErrorRecovery` -- the capability.
- `RecoveryPolicy` -- how failures are classified, rendered for the model, and logged.
- `RecoveryOutcome` -- the verdict for one failed tool call.
- `make_default_classify` / `default_classify` -- the conservative built-in classifier.
"""

from ._capability import ToolErrorRecovery
from ._outcome import DEFAULT_BUG_TYPES, DEFAULT_TRANSIENT_TYPES, RecoveryOutcome
from ._policy import (
    ErrorFormatter,
    RecoveryClassifier,
    RecoveryPolicy,
    default_classify,
    make_default_classify,
)

__all__ = [
    'DEFAULT_BUG_TYPES',
    'DEFAULT_TRANSIENT_TYPES',
    'ErrorFormatter',
    'RecoveryClassifier',
    'RecoveryOutcome',
    'RecoveryPolicy',
    'ToolErrorRecovery',
    'default_classify',
    'make_default_classify',
]
