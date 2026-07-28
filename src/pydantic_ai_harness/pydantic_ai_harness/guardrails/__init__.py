"""Input and output guardrails for Pydantic AI agents."""

from pydantic_ai_harness._warn import warn_class_renamed
from pydantic_ai_harness.guardrails._capability import (
    GuardrailResult,
    InputGuardrail,
    InputGuardrailFunc,
    OutputGuardrail,
    OutputGuardrailFunc,
)
from pydantic_ai_harness.guardrails._exceptions import (
    GuardrailError,
    InputBlocked,
    OutputBlocked,
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
