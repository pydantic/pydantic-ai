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

from typing_extensions import Self

from pydantic_ai._python_signature import FunctionSignature, TypeSignature


@dataclass(frozen=True)
class FunctionCall:
    """Represents a call to an external function made by executing code."""

    call_id: str
    function_name: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=lambda: {})


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


ToolCallback: TypeAlias = Callable[[FunctionCall], Awaitable[Any]]


class CodeRuntime(ABC):
    """Abstract base for code execution runtimes. Subclass per runtime provider."""

    execution_timeout: float | None = None
    """Optional timeout in seconds for code execution. None means no timeout."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        pass

    @abstractmethod
    async def run(
        self,
        code: str,
        call_tool: ToolCallback,
        *,
        functions: dict[str, FunctionSignature],
        referenced_types: list[TypeSignature],
    ) -> Any:
        """Execute code in the runtime.

        Args:
            code: The LLM-generated Python code to execute.
            call_tool: Callback invoked each time the code calls an external
                function. Receives a FunctionCall and returns the tool result.
            functions: Mapping of function name to signature, for type checking
                and declaring external functions.
            referenced_types: Unique type definitions referenced by the signatures.

        Returns:
            The final output of the code execution.

        Raises:
            CodeSyntaxError: If the code can't be parsed.
            CodeTypingError: If the code has type errors.
            CodeRuntimeError: If execution fails.
            CodeExecutionTimeout: If execution exceeds ``execution_timeout``.
        """
        ...

    @property
    def instructions(self) -> str | None:
        """Return runtime-specific text to include in the LLM prompt.

        If non-empty, the returned string is inserted verbatim into the code
        mode prompt between the execution model section and the coding
        guidelines. Return an empty string (the default) to add nothing.
        """
        return None
