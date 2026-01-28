from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, field
from typing import Any
from typing_extensions import Awaitable, Callable, TypeAlias


@dataclass(frozen=True)
class FunctionCall:
    """Emitted by CodeExecution.next() when running code calls a tool."""
    function_name: str
    args: tuple[Any, ...] = () # Positional args
    kwargs: dict[str, Any] = field(default_factory = dict) # keyword args


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

NextFn: TypeAlias = Callable[[], Awaitable[FunctionCall | ExecutionResult]]
ProvideResultFn: TypeAlias = Callable[[Any], Awaitable[None]]
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
        """
        return None
