"""The batteries for your Pydantic AI agent -- the official capability library."""

from typing import TYPE_CHECKING

from ._warn import HarnessDeprecationWarning

if TYPE_CHECKING:
    from .code_mode import CodeMode
    from .filesystem import FileSystem
    from .guardrails import (
        GuardrailError,
        GuardrailResult,
        InputBlocked,
        InputGuardrail,
        InputGuardrailFunc,
        OutputBlocked,
        OutputGuardrail,
        OutputGuardrailFunc,
    )
    from .logfire import ManagedPrompt
    from .shell import LLM_API_KEY_ENV_PATTERNS, Shell

__all__ = [
    'CodeMode',
    'FileSystem',
    'GuardrailError',
    'GuardrailResult',
    'HarnessDeprecationWarning',
    'InputBlocked',
    'InputGuardrail',
    'InputGuardrailFunc',
    'LLM_API_KEY_ENV_PATTERNS',
    'ManagedPrompt',
    'OutputBlocked',
    'OutputGuardrail',
    'OutputGuardrailFunc',
    'Shell',
]

_GUARDRAIL_EXPORTS = {
    'GuardrailError',
    'GuardrailResult',
    'InputBlocked',
    'InputGuardrail',
    'InputGuardrailFunc',
    'OutputBlocked',
    'OutputGuardrail',
    'OutputGuardrailFunc',
    # Pre-rename names; `pydantic_ai_harness.guardrails.__getattr__` emits the
    # deprecation warning when these resolve.
    'GuardResult',
    'InputGuard',
    'InputGuardFunc',
    'OutputGuard',
    'OutputGuardFunc',
}


def __getattr__(name: str) -> object:
    if name == 'CodeMode':
        from .code_mode import CodeMode

        return CodeMode
    if name in _GUARDRAIL_EXPORTS:
        from . import guardrails

        return getattr(guardrails, name)
    if name == 'FileSystem':
        from .filesystem import FileSystem

        return FileSystem
    if name == 'ManagedPrompt':
        from .logfire import ManagedPrompt

        return ManagedPrompt
    if name == 'Shell':
        from .shell import Shell

        return Shell
    if name == 'LLM_API_KEY_ENV_PATTERNS':
        from .shell import LLM_API_KEY_ENV_PATTERNS

        return LLM_API_KEY_ENV_PATTERNS
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
