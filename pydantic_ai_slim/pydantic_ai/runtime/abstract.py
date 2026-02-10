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

from pydantic_ai.exceptions import ApprovalRequired, CallDeferred


@dataclass(frozen=True)
class FunctionCall:
    """Represents a call to an external function made by executing code."""

    call_id: str
    function_name: str
    args: tuple[Any, ...] = ()  # Positional args
    kwargs: dict[str, Any] = field(default_factory=dict)  # keyword args


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


@dataclass
class InterruptedToolCall:
    """A tool call that was interrupted during code execution.

    Wraps a [`FunctionCall`][pydantic_ai.runtime.abstract.FunctionCall] together with the
    reason the call did not complete — either because it requires human approval
    ([`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired]) or because it was
    deferred ([`CallDeferred`][pydantic_ai.exceptions.CallDeferred]).

    Attributes:
        type: The exception that caused the interruption.
        call: The function call that was interrupted.
    """

    type: ApprovalRequired | CallDeferred
    call: FunctionCall


@dataclass
class CodeInterruptedError(Exception):
    """Raised when code execution is interrupted by one or more tool calls that require approval or were deferred.

    Contains the runtime checkpoint needed to resume execution once the
    interrupted calls have been resolved.

    Attributes:
        interrupted_calls: The tool calls that caused the interruption.
        checkpoint: Opaque runtime-specific state from a previous interrupted run.
            Bundles everything the runtime needs to resume, including any
            completed results from calls that succeeded before the interruption.
    """

    interrupted_calls: list[InterruptedToolCall]
    checkpoint: bytes


ToolCallback: TypeAlias = Callable[[FunctionCall], Awaitable[Any]]


class CodeRuntime(ABC):
    """Abstract base for code execution runtimes. Subclass per runtime provider."""

    @abstractmethod
    async def run(
        self,
        code: str,
        functions: list[str],
        call_tool: ToolCallback,
        *,
        signatures: list[str],
        checkpoint: bytes | None = None,
    ) -> Any:
        """Execute code, or resume from checkpoint if provided.

        Args:
            code: The LLM-generated Python code to execute.
            functions: List of external function names the code may call.
            call_tool: Callback invoked each time the code calls an external
                function. Receives a FunctionCall and returns the tool result.
                On resume, may contain pre-resolved results for interrupted calls.
            signatures: Function signatures for type checking.
            checkpoint: Opaque runtime-specific state from a previous interrupted run.
                When None, execute from scratch. When provided, resume.

        Returns:
            The final output of the code execution.

        Raises:
            CodeSyntaxError: If the code can't be parsed.
            CodeTypingError: If the code has type errors.
            CodeRuntimeError: If execution fails.
            CodeInterruptedError: When execution is interrupted by tool calls
                requiring approval or deferral.
        """
        ...

    def prompt_hints(self) -> str:
        """Return runtime-specific text to include in the LLM prompt.

        If non-empty, the returned string is inserted verbatim into the code
        mode prompt between the execution model section and the coding
        guidelines. Return an empty string (the default) to add nothing.
        """
        return ''
