"""Input, output, and tool guardrails for Pydantic AI agents.

Ready-made checks to plug into a guardrail live in
[`detectors`][pydantic_ai_harness.guardrails.detectors].
"""

from pydantic_ai_harness._warn import warn_class_renamed
from pydantic_ai_harness.guardrails import detectors
from pydantic_ai_harness.guardrails._capability import (
    InputGuardrail,
    InputGuardrailFunc,
    OutputGuardrail,
    OutputGuardrailFunc,
)
from pydantic_ai_harness.guardrails._exceptions import (
    GuardrailError,
    InputBlocked,
    OutputBlocked,
    ToolBlocked,
)
from pydantic_ai_harness.guardrails._shared import GuardrailResult
from pydantic_ai_harness.guardrails._tool_guardrail import (
    ToolCallInfo,
    ToolGuardrail,
    ToolGuardrailFunc,
    ToolResultGuardrailFunc,
    ToolResultInfo,
)

__all__ = [
    'GuardrailError',
    'GuardrailResult',
    'InputBlocked',
    'InputGuardrail',
    'InputGuardrailFunc',
    'OutputBlocked',
    'OutputGuardrail',
    'OutputGuardrailFunc',
    'ToolBlocked',
    'ToolCallInfo',
    'ToolGuardrail',
    'ToolGuardrailFunc',
    'ToolResultGuardrailFunc',
    'ToolResultInfo',
    'detectors',
]

_RENAMED: dict[str, object] = {
    'GuardResult': GuardrailResult,
    'InputGuard': InputGuardrail,
    'InputGuardFunc': InputGuardrailFunc,
    'OutputGuard': OutputGuardrail,
    'OutputGuardFunc': OutputGuardrailFunc,
}


def __getattr__(name: str) -> object:
    renamed = _RENAMED.get(name)
    if renamed is not None:
        warn_class_renamed(name, name.replace('Guard', 'Guardrail'), 'pydantic_ai_harness.guardrails')
        return renamed
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
