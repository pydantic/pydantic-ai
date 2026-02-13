from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from ._transport import DriverBasedRuntime, DriverTransport
from .abstract import (
    CodeExecutionError,
    CodeExecutionTimeout,
    CodeInterruptedError,
    CodeRuntime,
    CodeRuntimeError,
    CodeSyntaxError,
    CodeTypingError,
    FunctionCall,
    InterruptedToolCall,
    ToolCallback,
)
from .docker import DockerRuntime, DockerSecuritySettings

if TYPE_CHECKING:
    from .modal import ModalRuntime
    from .monty import MontyRuntime

__all__ = (
    'CodeExecutionError',
    'CodeExecutionTimeout',
    'CodeInterruptedError',
    'CodeRuntime',
    'CodeRuntimeError',
    'CodeSyntaxError',
    'CodeTypingError',
    'DockerRuntime',
    'DockerSecuritySettings',
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
    if name == 'ModalRuntime':
        from .modal import ModalRuntime

        return ModalRuntime
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


RuntimeName = Literal['monty', 'docker', 'modal']


def get_runtime(name: RuntimeName) -> CodeRuntime:
    if name == 'monty':
        from .monty import MontyRuntime

        return MontyRuntime()
    elif name == 'docker':
        from .docker import DockerRuntime

        return DockerRuntime()
    elif name == 'modal':
        from .modal import ModalRuntime

        return ModalRuntime()
    else:
        raise ValueError(f'Invalid runtime: {name}')
