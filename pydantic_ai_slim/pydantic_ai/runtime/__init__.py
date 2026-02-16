from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

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

if TYPE_CHECKING:
    from .docker import DockerRuntime, DockerSecuritySettings
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


def __getattr__(name: str) -> Any:
    if name == 'MontyRuntime':
        from .monty import MontyRuntime

        return MontyRuntime
    if name in ('DockerRuntime', 'DockerSecuritySettings'):
        from . import docker

        return getattr(docker, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


RuntimeName = Literal['monty', 'docker']


def get_runtime(name: RuntimeName) -> CodeRuntime:
    if name == 'monty':
        from .monty import MontyRuntime

        return MontyRuntime()
    elif name == 'docker':
        from .docker import DockerRuntime

        return DockerRuntime()
    else:
        assert_never(name)
