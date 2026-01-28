"""Abstract base classes and data types for the code execution runtime layer.

This module defines the vendor-agnostic protocol that all code runtimes must
implement. The design is still evolving — several parts of the current API
are shaped heavily by Monty's execution model (snapshot-based pause/resume
with opaque bytes checkpoints). As we add runtimes with different paradigms
(e.g. container-based, WASM, remote sandbox), the abstractions here will
likely need to be revisited. Known areas that need more thought:

- **Checkpoint format**: ``dump()`` / ``restore()`` currently use opaque
  ``bytes``, which works for Monty's in-process snapshots but may not suit
  runtimes where "checkpoint" means a session ID, a container image, or a
  remote reference. A more flexible checkpoint representation (or a
  runtime-specific opaque handle) may be needed.

- **Restore semantics**: ``restore()`` lives on ``CodeRuntime`` and takes raw
  bytes, which couples the caller to the snapshot serialization format. A
  runtime that keeps state server-side (e.g. a long-lived sandbox container)
  might not need byte-level serialization at all — it just needs a session
  token to reconnect.

- **Type checking coupling**: ``type_check()`` accepts Python signature
  strings and typing imports, which assumes the runtime understands Python
  type annotations. Runtimes that execute non-Python code or use a different
  type system would need a different interface.

- **Execution lifecycle**: The current ``next()`` / ``provide_result()`` loop
  assumes synchronous pause-and-resume. Runtimes with async callbacks,
  streaming output, or parallel tool calls may need a richer protocol.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeAlias


@dataclass(frozen=True)
class FunctionCall:
    """Emitted by CodeExecution.next() when running code calls a tool."""

    function_name: str
    args: tuple[Any, ...] = ()  # Positional args
    kwargs: dict[str, Any] = field(default_factory=dict)  # keyword args


@dataclass(frozen=True)
class ExecutionResult:
    """Emitted by CodeExecution.next() when running code finishes."""

    output: Any


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


# Callback type aliases injected into CodeExecution by concrete runtimes.
NextFn: TypeAlias = Callable[[], Awaitable[FunctionCall | ExecutionResult]]
ProvideResultFn: TypeAlias = Callable[[Any], Awaitable[None]]
# TODO: bytes works for Monty's in-process snapshots but may be too narrow for
# runtimes that checkpoint via session tokens or remote references. Consider a
# generic checkpoint handle type or runtime-specific opaque wrapper.
DumpFn: TypeAlias = Callable[[], bytes | None]


class CodeExecution:
    """Handle for one running code execution. Created by CodeRuntime.execute() or .restore().

    This is a concrete class, not abstract. Runtimes inject behavior through
    the private callable fields. Consumers only use the public methods.
    """

    # Injected by the runtime at construction time — these are the actual behavior.
    # Public methods below delegate to these callables.
    _next: NextFn
    _provide_result: ProvideResultFn
    _dump: DumpFn = lambda: None

    def __init__(self, next: NextFn, provide_result: ProvideResultFn, dump: DumpFn) -> None:
        self._next = next
        self._provide_result = provide_result
        self._dump = dump

    async def next(self) -> FunctionCall | ExecutionResult:
        """Pull the next event: either a FunctionCall (code wants a tool) or ExecutionResult (code finished).

        Call this first to start iteration. After providing a result via provide_result(),
        call this again to advance to the next pause point or completion.

        Raises:
            CodeRuntimeError: If the code raises an exception during execution.
        """
        return await self._next()

    async def provide_result(self, result: Any) -> None:
        """Feed a tool's return value back into the execution.

        Call this after next() returns a FunctionCall and you've run the real tool.
        The next call to next() will resume the code from where it paused, with
        this value as the return value of the function call.
        """
        await self._provide_result(result)

    def dump(self) -> bytes | None:
        """Serialize execution state for checkpointing (approval flow).

        Returns bytes if checkpointing is supported, None otherwise.

        .. note:: Design consideration
            Returning raw ``bytes`` assumes the checkpoint is a self-contained
            serialized blob (as with Monty snapshots). Runtimes that maintain
            server-side state may prefer returning a lightweight handle (e.g.
            session ID string) instead. This return type may need to become a
            union or a generic ``Checkpoint`` wrapper as more runtimes are added.
        """
        return self._dump()


class CodeRuntime(ABC):
    """Abstract base for code execution runtimes. Subclass per runtime provider."""

    async def execute(self, code: str, functions: list[str]) -> CodeExecution:
        """Start running code in the runtime.

        Args:
            code: The LLM-generated Python code to execute.
            functions: List of external function names the code may call.
                       These are the sanitized names (valid Python identifiers).

        Returns:
            A CodeExecution handle. Call .next() to begin iteration.

        Raises:
            CodeSyntaxError: If the code can't be parsed.
            CodeRuntimeError: If execution fails immediately.
        """
        ...

    async def type_check(self, code: str, signatures: list[str]) -> None:
        """Optional pre-execution type checking.

        Default is a no-op. Runtimes with a built-in type checker can override this.

        Args:
            code: The LLM-generated Python code to type check.
            signatures: List of Python function signature strings for available tools.

        Raises:
            CodeTypeError: If type checking finds errors.
            CodeSyntaxError: If the code can't be parsed.

        .. note:: Design consideration
            The ``signatures`` parameter currently expects Python function
            signature strings with standard ``typing`` annotations. This works
            for Monty and any Python-aware runtime, but runtimes executing
            non-Python code or using a different type system would need a
            different representation. Consider whether signatures should be
            more structured (e.g. JSON Schema) or if type_check should be an
            optional mixin rather than part of the base class.
        """
        pass

    async def restore(self, checkpoint: bytes) -> CodeExecution | None:
        """Restore execution from a checkpoint.

        Used to resume an execution that was serialized mid-flight, e.g. after
        yielding control back to the host for tool execution or approval.

        Args:
            checkpoint: Bytes previously returned by CodeExecution.dump().

        Returns:
            A CodeExecution positioned mid-execution, ready for provide_result() + next().
            Returns None if restoration failed (data corrupted, session expired, not supported).

        .. note:: Design consideration
            This method currently accepts raw ``bytes``, which maps directly to
            Monty's snapshot serialization. Runtimes that maintain state
            server-side (e.g. a persistent sandbox container or remote session)
            may not serialize state to bytes at all — they might only need a
            session token or URL to reconnect. A future revision could introduce
            a generic ``Checkpoint`` type or make restore a no-op for stateful
            runtimes that never actually disconnect.
        """
        return None
