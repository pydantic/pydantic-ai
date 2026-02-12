"""Abstract base classes and data types for the code execution runtime layer.

This module defines the vendor-agnostic protocol that all code runtimes must
implement. Runtimes execute LLM-generated code and invoke a caller-provided
callback whenever the code calls an external function (tool).
"""

from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import pydantic

from pydantic_ai.exceptions import ApprovalRequired, CallDeferred
from pydantic_ai.messages import ToolReturnContent, tool_return_ta

# Shared TypeAdapter for deserializing checkpoint values with proper type reconstruction.
# Uses ToolReturnContent (which includes the discriminated MultiModalContent union)
# so that rich types like BinaryContent and ImageUrl subclasses are reconstructed
# from their JSON representations via the `kind` discriminator, rather than being
# left as plain dicts after round-tripping through JSON.
#
# All runtimes normalize tool results to JSON-compatible form (via
# tool_return_ta.dump_python(mode='json')) before the executing code sees them.
# This means Pydantic BaseModel instances become plain dicts, bytes become base64
# strings, etc. — ensuring consistent behavior between fresh runs and checkpoint
# resumes, and across runtime implementations.
checkpoint_result_ta: pydantic.TypeAdapter[Any] = pydantic.TypeAdapter(
    ToolReturnContent,
    config=pydantic.ConfigDict(defer_build=True, ser_json_bytes='base64', val_json_bytes='base64'),
)


@dataclass(frozen=True)
class FunctionCall:
    """Represents a call to an external function made by executing code."""

    call_id: str
    function_name: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=lambda: {})


@dataclass(frozen=True)
class DeserializedCheckpoint:
    """Deserialized checkpoint data shared across runtimes."""

    completed_results: dict[str, Any]
    pending_calls: dict[str, dict[str, Any]]
    interpreter_state: bytes | None = None


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


@dataclass
class InterruptedToolCall:
    """A tool call that was interrupted during code execution.

    Wraps a [`FunctionCall`][pydantic_ai.runtime.abstract.FunctionCall] together with the
    reason the call did not complete — either because it requires human approval
    ([`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired]) or because it was
    deferred ([`CallDeferred`][pydantic_ai.exceptions.CallDeferred]).

    Attributes:
        reason: The exception that caused the interruption.
        call: The function call that was interrupted.
    """

    reason: ApprovalRequired | CallDeferred
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


def serialize_checkpoint_results(
    completed_results: dict[int, Any],
    interrupted_calls: list[InterruptedToolCall],
    *,
    interpreter_state: bytes | None = None,
) -> bytes:
    """Serialize checkpoint results into an opaque bytes blob.

    Each completed result is individually serialized to JSON bytes via Pydantic's
    ``tool_return_ta``, then base64-encoded for embedding in the outer JSON payload.
    On deserialization, ``checkpoint_result_ta`` (backed by ``ToolReturnContent``)
    reconstructs rich types (``BinaryContent``, ``ImageUrl``, etc.) from their JSON
    via the ``kind`` discriminator — plain dicts/strings/numbers pass through unchanged.

    Args:
        completed_results: Results from tool calls that completed before the interruption,
            keyed by internal int call IDs.
        interrupted_calls: The tool calls that were interrupted, with full FunctionCall details.
        interpreter_state: Optional raw interpreter state bytes (used by Monty runtime).
    """
    raw_results = {
        str(k): base64.b64encode(tool_return_ta.dump_json(v)).decode('ascii') for k, v in completed_results.items()
    }
    pending_calls = {
        ic.call.call_id: {
            'function_name': ic.call.function_name,
            'args': list(ic.call.args),
            'kwargs': ic.call.kwargs,
        }
        for ic in interrupted_calls
    }
    payload: dict[str, Any] = {
        'completed_results': raw_results,
        'pending_calls': pending_calls,
    }
    if interpreter_state is not None:
        payload['interpreter_state'] = base64.b64encode(interpreter_state).decode('ascii')
    return json.dumps(payload).encode('utf-8')


def decode_checkpoint_results(raw_results: dict[str, str]) -> dict[str, Any]:
    """Decode base64-encoded checkpoint results back to JSON-compatible values.

    Reverses the encoding done by ``serialize_checkpoint_results``: base64
    decode, reconstruct rich types (``BinaryContent``, ``ImageUrl``, etc.)
    via ``ToolReturnContent`` validation, then normalize to JSON-compatible
    form.
    """
    return {
        k: tool_return_ta.dump_python(
            checkpoint_result_ta.validate_json(base64.b64decode(v)),
            mode='json',
        )
        for k, v in raw_results.items()
    }


def deserialize_checkpoint(checkpoint: bytes) -> DeserializedCheckpoint:
    """Deserialize a checkpoint back into its components."""
    payload = json.loads(checkpoint)
    interpreter_state = None
    if 'interpreter_state' in payload:
        interpreter_state = base64.b64decode(payload['interpreter_state'])
    return DeserializedCheckpoint(
        completed_results=payload.get('completed_results', {}),
        pending_calls=payload.get('pending_calls', {}),
        interpreter_state=interpreter_state,
    )
