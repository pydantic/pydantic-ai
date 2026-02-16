from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from typing_extensions import assert_never

from ._transport import DriverBasedRuntime, DriverTransport
from .abstract import (
    CodeExecutionError,
    CodeExecutionTimeout,
    CodeRuntime,
    CodeRuntimeError,
    CodeSyntaxError,
    CodeTypingError,
    FunctionCall,
    ToolCallback,
)
from .docker import DockerRuntime, DockerSecuritySettings

try:
    from .monty import MontyRuntime
except ImportError:
    pass

if TYPE_CHECKING:
    from .monty import MontyRuntime

__all__ = (
    'CodeExecutionError',
    'CodeExecutionTimeout',
    'CodeRuntime',
    'CodeRuntimeError',
    'CodeSyntaxError',
    'CodeTypingError',
    'DockerRuntime',
    'DockerSecuritySettings',
    'DriverBasedRuntime',
    'DriverTransport',
    'FunctionCall',
    'MontyRuntime',
    'ToolCallback',
)


RuntimeName = Literal['monty', 'docker']


def get_runtime(name: RuntimeName) -> CodeRuntime:
    if name == 'monty':
        from .monty import MontyRuntime

        return MontyRuntime()
    elif name == 'docker':
        return DockerRuntime()
    else:
        assert_never(name)
