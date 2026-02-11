from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._transport import DriverBasedRuntime, DriverTransport
from .abstract import (
    CodeExecutionError,
    CodeInterruptedError,
    CodeRuntime,
    CodeRuntimeError,
    CodeSyntaxError,
    CodeTypingError,
    FunctionCall,
    InterruptedToolCall,
    ToolCallback,
)
from .docker import DockerRuntime
from .modal import ModalRuntime

if TYPE_CHECKING:
    from .monty import MontyRuntime

__all__ = (
    'CodeExecutionError',
    'CodeInterruptedError',
    'CodeRuntime',
    'CodeRuntimeError',
    'CodeSyntaxError',
    'CodeTypingError',
    'DockerRuntime',
    'DriverBasedRuntime',
    'DriverTransport',
    'FunctionCall',
    'InterruptedToolCall',
    'ModalRuntime',
    'MontyRuntime',
    'ToolCallback',
)


def __getattr__(name: str) -> Any:
    if name == 'MontyRuntime':
        from .monty import MontyRuntime

        return MontyRuntime
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
