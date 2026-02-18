"""Backward-compatible re-exports — use `pydantic_ai.environments` and `pydantic_ai.toolsets.code_execution` instead."""

from __future__ import annotations

from pydantic_ai.environments._base import ExecutionEnvironment
from pydantic_ai.environments._driver import DriverBasedEnvironment, DriverTransport
from pydantic_ai.toolsets.code_execution import (
    CodeExecutionError,
    CodeExecutionTimeout,
    CodeRuntimeError,
    CodeSyntaxError,
    CodeTypingError,
    EnvironmentName,
    FunctionCall,
    FunctionCallback,
    get_environment,
)

try:
    from pydantic_ai.environments.monty import MontyEnvironment
except ImportError:
    pass

try:
    from pydantic_ai.environments.docker import DockerEnvironment
except ImportError:
    pass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai.environments.docker import DockerEnvironment
    from pydantic_ai.environments.monty import MontyEnvironment

__all__ = (
    'CodeExecutionError',
    'CodeExecutionTimeout',
    'CodeRuntimeError',
    'CodeSyntaxError',
    'CodeTypingError',
    'DockerEnvironment',
    'DriverBasedEnvironment',
    'DriverTransport',
    'EnvironmentName',
    'ExecutionEnvironment',
    'FunctionCall',
    'MontyEnvironment',
    'FunctionCallback',
    'get_environment',
)
