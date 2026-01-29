"""Abstract base classes and data types for the code execution runtime layer.

This module defines the vendor-agnostic protocol that all code runtimes must
implement. Runtimes execute LLM-generated code and invoke a caller-provided
callback whenever the code calls an external function (tool).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeAlias


@dataclass(frozen=True)
class FunctionCall:
    """Represents a call to an external function made by executing code."""

    function_name: str
    args: tuple[Any, ...] = ()  # Positional args
    kwargs: dict[str, Any] = field(default_factory=dict)  # keyword args


# Exception Hierarchy
#


class CodeExecutionError(Exception):
    """Base for all code execution errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class CodeSyntaxError(CodeExecutionError):
    """Code has a syntax error."""

    pass


class CodeTypeError(CodeExecutionError):
    """Code has a type error."""

    pass


class CodeRuntimeError(CodeExecutionError):
    """Code raised an exception at runtime."""

    pass


ToolCallback: TypeAlias = Callable[[FunctionCall], Awaitable[Any]]


class CodeRuntime(ABC):
    """Abstract base for code execution runtimes. Subclass per runtime provider."""

    @abstractmethod
    async def run(
        self,
        code: str,
        functions: list[str],
        call_tool: ToolCallback,
    ) -> Any:
        """Execute code, invoking call_tool for each external function call.

        Args:
            code: The LLM-generated Python code to execute.
            functions: List of external function names the code may call.
            call_tool: Callback invoked each time the code calls an external
                function. Receives a FunctionCall and returns the tool result.

        Returns:
            The final output of the code execution.

        Raises:
            CodeSyntaxError: If the code can't be parsed.
            CodeRuntimeError: If execution fails.
        """
        ...

    async def resume_with_tools(
        self,
        checkpoint: bytes,
        call_tool: ToolCallback,
    ) -> Any:
        """Resume execution from a checkpoint.

        Used to continue an execution that was interrupted mid-flight, e.g.
        after an approval request.

        Args:
            checkpoint: Bytes previously obtained from a runtime checkpoint.
            call_tool: Callback invoked for each remaining external function call.

        Returns:
            The final output of the code execution.

        Raises:
            CodeRuntimeError: If this runtime does not support checkpoint resumption.
        """
        raise CodeRuntimeError('This runtime does not support checkpoint resumption')

    async def type_check(self, code: str, signatures: list[str]) -> None:
        """Optional pre-execution type checking. Default: no-op.

        Args:
            code: The LLM-generated Python code to type check.
            signatures: List of Python function signature strings for available tools.

        Raises:
            CodeTypeError: If type checking finds errors.
            CodeSyntaxError: If the code can't be parsed.
        """
        pass
