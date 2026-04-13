"""Deprecated: pydantic-harness has been renamed to pydantic-ai-harness."""

import warnings
from typing import TYPE_CHECKING

warnings.warn(
    'pydantic-harness has been renamed to pydantic-ai-harness. '
    'Please update your dependencies and imports:\n'
    '    uv remove pydantic-harness\n'
    '    uv add pydantic-ai-harness\n'
    'Then change `from pydantic_harness import ...` to `from pydantic_ai_harness import ...`. '
    'This shim re-exports from pydantic-ai-harness 0.1.x and will stop resolving with 0.2.0.',
    DeprecationWarning,
    stacklevel=2,
)

if TYPE_CHECKING:
    from pydantic_ai_harness import CodeMode

__all__ = ['CodeMode']


def __getattr__(name: str) -> object:
    if name == 'CodeMode':
        from pydantic_ai_harness import CodeMode

        return CodeMode
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
