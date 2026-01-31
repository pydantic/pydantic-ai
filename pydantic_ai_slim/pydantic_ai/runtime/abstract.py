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
    kwargs: dict[str, Any] = field(default_factory=lambda: {})  # keyword args


class CodeExecutionError(Exception):
    """Base for all code execution errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class CodeSyntaxError(CodeExecutionError):
    """Code has a syntax error."""

    pass


class CodeTypingError(CodeExecutionError):
    """Code has a typing error."""

    pass


class CodeRuntimeError(CodeExecutionError):
    """Code raised an exception at runtime."""

    pass


# TODO: Consider whether this should be Coroutine[Any, Any, Any] instead of Awaitable[Any].
# Awaitable is broader (covers Coroutine, Task, Future, __await__ objects), but
# Coroutine would allow asyncio.create_task() directly without ensure_future().
# Check what callers actually pass and whether the extra flexibility is needed.
ToolCallback: TypeAlias = Callable[[FunctionCall], Awaitable[Any]]


class CodeRuntime(ABC):
    """Abstract base for code execution runtimes. Subclass per runtime provider."""

    @abstractmethod
    async def run(self, code: str, functions: list[str], call_tool: ToolCallback, signatures: list[str]) -> Any:
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

    @abstractmethod
    async def resume(self, checkpoint: bytes, call_tool: ToolCallback) -> Any: ...
