"""Data types for the code execution layer.

This module defines error types, the ``FunctionCall`` dataclass, and the
``FunctionCallback`` type alias used by ``CodeExecutionToolset`` and execution
environments.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeAlias


@dataclass(frozen=True)
class FunctionCall:
    """Represents a call to an external function made by executing code."""

    call_id: str
    function_name: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict[str, Any])


class CodeExecutionError(Exception):
    """Base for all code execution errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class CodeSyntaxError(CodeExecutionError):
    """The generated code has a syntax error."""


class CodeTypingError(CodeExecutionError):
    """The generated code has a type error."""


class CodeRuntimeError(CodeExecutionError):
    """The generated code raised an exception at runtime."""


class CodeExecutionTimeout(CodeRuntimeError):
    """The code execution exceeded the configured timeout."""


FunctionCallback: TypeAlias = Callable[[FunctionCall], Awaitable[Any]]
