"""Backward-compatible re-exports — use `pydantic_ai.toolsets.code_execution` instead."""

from __future__ import annotations

from pydantic_ai.toolsets.code_execution import (
    CodeExecutionError,
    CodeExecutionTimeout,
    CodeRuntime,
    CodeRuntimeError,
    CodeSyntaxError,
    CodeTypingError,
    DockerRuntime,
    DockerSecuritySettings,
    DriverBasedRuntime,
    DriverTransport,
    FunctionCall,
    RuntimeName,
    ToolCallback,
    get_runtime,
)

try:
    from pydantic_ai.toolsets.code_execution.monty import MontyRuntime
except ImportError:
    pass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai.toolsets.code_execution.monty import MontyRuntime

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
    'RuntimeName',
    'ToolCallback',
    'get_runtime',
)
