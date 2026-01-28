"""CodeRuntime implementation backed by the Monty sandboxed Python interpreter.

Monty executes LLM-generated code in a restricted environment and pauses
whenever the code calls an external function (tool). This pause-and-resume
model maps directly onto the CodeExecution protocol: each pause yields a
FunctionCall, and the caller feeds back a result to continue execution.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.runtime.abstract import (
    CodeExecution,
    CodeRuntime,
    CodeRuntimeError,
    CodeSyntaxError,
    CodeTypeError,
    ExecutionResult,
    FunctionCall,
)

try:
    import pydantic_monty as monty
except ImportError:
    raise ImportError("MontyRuntime requires 'monty'. Install with: pip install 'pydantic-ai-slim[monty]'")


class MontyRuntime(CodeRuntime):
    """CodeRuntime that delegates to the Monty sandboxed interpreter.

    Monty provides sandboxed execution with built-in type checking and
    snapshot-based checkpointing. Code is executed in isolation and pauses
    at every external function call, returning a MontySnapshot that can be
    resumed once the host has computed the function's return value.
    """

    async def execute(self, code: str, functions: list[str]) -> CodeExecution:
        """Start executing code in the Monty sandbox.

        Args:
            code: LLM-generated Python source code.
            functions: Names of external functions the code is allowed to call.
                Monty will pause execution whenever one of these is invoked.

        Returns:
            A CodeExecution handle for iterating through function calls and
            the final result.

        Raises:
            CodeSyntaxError: If Monty cannot parse the code.
            CodeRuntimeError: If execution fails immediately (e.g. top-level error).
        """
        m = monty.Monty(code, external_functions=functions)
        try:
            state = m.start()
        except monty.MontySyntaxError as e:
            raise CodeSyntaxError(e.display())
        except monty.MontyRuntimeError as e:
            raise CodeRuntimeError(e.display())

        return _build_execution(state)

    async def type_check(self, code: str, signatures: list[str]) -> None:
        """Type check code using Monty's built-in type checker.

        Prepends standard typing imports and the provided tool signatures as
        prefix code so that Monty's checker can resolve external function types
        without needing the actual implementations.

        Args:
            code: LLM-generated Python source code to type check.
            signatures: Python function signature strings (e.g.
                ``def get_weather(city: str) -> str: ...``) for available tools.

        Raises:
            CodeTypeError: If Monty's type checker finds type errors.
            CodeSyntaxError: If the code cannot be parsed.
        """
        # Build a preamble containing typing imports and tool signatures so
        # Monty can resolve types for external function calls.
        imports = 'from typing import Any, TypedDict, NotRequired, Literal\n\n'
        prefix_code = imports + '\n\n'.join(signatures)
        m = monty.Monty(code, external_functions=[])
        try:
            m.type_check(prefix_code=prefix_code)
        except monty.MontyTypingError as e:
            raise CodeTypeError(e.display())
        except monty.MontySyntaxError as e:
            raise CodeSyntaxError(e.display())

    async def restore(self, checkpoint: bytes) -> CodeExecution:
        """Restore a paused execution from a serialized checkpoint.

        Args:
            checkpoint: Bytes previously returned by ``CodeExecution.dump()``,
                containing a serialized MontySnapshot.

        Returns:
            A CodeExecution positioned at the point where the snapshot was
            taken, ready to receive a result and continue.
        """
        result = monty.MontySnapshot.load(checkpoint)
        return _build_execution(result)


def _build_execution(initial_state: monty.MontySnapshot | monty.MontyComplete) -> CodeExecution:
    """Wrap a Monty state object in the CodeExecution protocol.

    Uses closure-based state to track the current Monty snapshot and any
    pending tool result. This avoids subclassing CodeExecution while still
    providing the three callbacks it requires (next, provide_result, dump).

    Args:
        initial_state: Either a MontySnapshot (execution paused at a function
            call) or MontyComplete (execution already finished).

    Returns:
        A fully wired CodeExecution instance.
    """
    # Mutable containers used as closure state — lists allow mutation from
    # inner functions without `nonlocal`.
    state = [initial_state]
    pending_result: list[Any] = []

    async def _next() -> FunctionCall | ExecutionResult:
        # If a result was provided since the last call, resume execution
        # with that value before inspecting the new state.
        if pending_result:
            if not isinstance(state[0], monty.MontySnapshot):
                raise CodeRuntimeError('Cannot resume a completed execution')
            try:
                state[0] = state[0].resume(return_value=pending_result.pop())
            except monty.MontyRuntimeError as e:
                raise CodeRuntimeError(e.display())

        s = state[0]
        if isinstance(s, monty.MontySnapshot):
            # Execution is paused at an external function call — surface it
            # so the host can run the real tool and feed back a result.
            return FunctionCall(
                function_name=s.function_name,
                args=tuple(s.args) if s.args else (),
                kwargs=dict(s.kwargs) if s.kwargs else {},
            )

        # MontyComplete — execution finished, return the final output.
        return ExecutionResult(output=s.output)

    async def _provide_result(value: Any) -> None:
        # Buffer the tool return value; it will be consumed on the next
        # call to _next() which resumes the snapshot.
        pending_result.append(value)

    def _dump() -> bytes | None:
        # Only snapshots (paused executions) can be serialized. Completed
        # executions have no state to checkpoint.
        if isinstance(state[0], monty.MontySnapshot):
            return state[0].dump()
        return None

    return CodeExecution(_next, _provide_result, _dump)
