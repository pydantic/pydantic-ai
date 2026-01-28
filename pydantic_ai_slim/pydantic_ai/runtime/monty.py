from __future__ import annotations

from pydantic_ai.runtime.abstract import CodeRuntime, CodeExecution, CodeRuntimeError, CodeSyntaxError, CodeTypeError, ExecutionResult, FunctionCall
from typing import Any

try:
    import monty
except ImportError:
    raise ImportError(
        "MontyRuntime requires 'monty'. "
        "Install with: pip install 'pydantic-ai[monty]'"
    )

class MontyRuntime(CodeRuntime):

    async def execute(self, code: str, functions: list[str]) -> CodeExecution:
        m = monty.Monty(code, external_functions=functions)
        try:
            state = m.start()
        except monty.MontySyntaxError as e:
            raise CodeSyntaxError(e.display())
        except monty.MontyRuntimeError as e:
            raise CodeRuntimeError(e.display())

        return _build_execution(state)

    async def type_check(self, code: str, signatures: list[str]) -> None:
        """Type check code using Monty's built-in type checker."""
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
        result = monty.MontySnapshot.load(checkpoint)
        return _build_execution(result)

def _build_execution(initial_state: monty.MontySnapshot | monty.MontyComplete) -> CodeExecution:
    state = [initial_state]
    pending_result = []

    async def _next() -> FunctionCall | ExecutionResult:
        if pending_result:
            if not isinstance(state[0], monty.MontySnapshot):
                raise CodeRuntimeError('Cannot resume a completed execution')
            try:
                state[0] = state[0].resume(return_value=pending_result.pop())
            except monty.MontyRuntimeError as e:
                raise CodeRuntimeError(e.display())

        s = state[0]
        if isinstance(s, monty.MontySnapshot):
            return FunctionCall(
                function_name=s.function_name,
                args=tuple(s.args) if s.args else (),
                kwargs=dict(s.kwargs) if s.kwargs else {}
            )

        return ExecutionResult(output=s.output)

    async def _provide_result(value: Any) -> None:
        pending_result.append(value)

    def _dump() -> bytes | None:
        if isinstance(state[0], monty.MontySnapshot):
            return state[0].dump()
        return None

    return CodeExecution(
        _next,
        _provide_result,
        _dump
    )
