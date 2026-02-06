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
        checkpoint: Serialized runtime state that can be passed to
            [`CodeRuntime.resume`][pydantic_ai.runtime.abstract.CodeRuntime.resume]
            to continue execution.
        completed_results: Results from tool calls that completed successfully
            before the interruption, keyed by call index.
    """

    interrupted_calls: list[InterruptedToolCall]
    checkpoint: bytes
    completed_results: dict[int, Any] = field(default_factory=lambda: {})


ToolCallback: TypeAlias = Callable[[FunctionCall], Awaitable[Any]]


class CodeRuntime(ABC):
    """Abstract base for code execution runtimes. Subclass per runtime provider."""

    @abstractmethod
    async def run(self, code: str, functions: list[str], call_tool: ToolCallback, *, signatures: list[str]) -> Any:
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
    async def resume(
        self,
        checkpoint: bytes,
        call_tool: ToolCallback,
        interrupted_calls: list[dict[str, Any]],
        completed_results: dict[int, Any] | None = None,
    ) -> Any:
        """Resume execution from a checkpoint with resolved results.

        Args:
            checkpoint: The serialized checkpoint state from a previous interrupted run.
            call_tool: Callback invoked for each pending function call. The callback
                should have pre-resolved results available (via results_map) so that
                pending calls return immediately.
            interrupted_calls: List of call details from the interrupted execution.
                Each dict contains: call_id, tool_name, args, kwargs, type.
                These are needed to reconstruct FunctionCall objects for the callback.
            completed_results: Results from calls that already succeeded before the
                interruption. Keyed by integer call_id, values in Monty result format
                (e.g. ``{'return_value': ...}``). These are fed to the runtime first
                so already-completed calls are not re-executed.

        Returns:
            The final output of the resumed code execution.
        """
        ...
