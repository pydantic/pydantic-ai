"""Agent harness for composable, reusable AI agent capabilities, for Pydantic AI."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .code_mode import CodeMode

__all__ = ['CodeMode']


def __getattr__(name: str) -> object:
    if name == 'CodeMode':
        from .code_mode import CodeMode

        return CodeMode
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
