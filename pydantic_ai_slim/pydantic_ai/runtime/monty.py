from __future__ import annotations

import asyncio
from typing import Any

from pydantic_ai._python_signature import FunctionSignature, collect_unique_referenced_types
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred
from pydantic_ai.messages import tool_return_ta
from pydantic_ai.runtime.abstract import (
    CodeExecutionTimeout,
    CodeInterruptedError,
    CodeRuntime,
    CodeRuntimeError,
    CodeSyntaxError,
    CodeTypingError,
    FunctionCall,
    InterruptedToolCall,
    ToolCallback,
)

_TYPING_IMPORTS = 'from typing import Any, TypedDict, NotRequired, Literal'

try:
    from pydantic_monty import (
        Monty,
        MontyComplete,
        MontyFutureSnapshot,
        MontyRuntimeError,
        MontySnapshot,
        MontySyntaxError,
        MontyTypingError,
        ResourceLimits,
    )
except ImportError as _import_error:
    raise ImportError(
        'Please install `pydantic-monty` to use the Monty runtime, '
        'you can use the `monty` optional group — `pip install "pydantic-ai-slim[monty]"`'
    ) from _import_error


def _build_type_check_prefix(signatures: list[FunctionSignature]) -> str:
    """Build the prefix code used for Monty type checking.

    Combines standard typing imports with tool signatures to create the
    prefix that Monty uses for type-checking LLM-generated code.

    Note: Signatures use `...` as the body by default, but ty/Monty requires
    `raise NotImplementedError()` for valid function stubs. See:
    https://github.com/astral-sh/ty/issues/1922

    Args:
        signatures: List of Python function signatures for available tools.

    Returns:
        Complete prefix code string with imports and signatures.
    """
    # TODO (Douwe): Move to better place — moved to _TYPING_IMPORTS module constant for now
    parts = [_TYPING_IMPORTS]
    parts.extend(t.render() for t in collect_unique_referenced_types(signatures))
    parts.extend(sig.render('raise NotImplementedError()') for sig in signatures)

    return '\n\n'.join(parts)


class MontyRuntime(CodeRuntime):
    """CodeRuntime that delegates to the Monty sandboxed interpreter.

    Monty provides sandboxed execution with built-in type checking. Code is
    executed in isolation and pauses at every external function call, returning
    a MontySnapshot that can be resumed once the host has computed the
    function's return value.

    Tool calls are executed concurrently via asyncio tasks. When Monty
    discovers an external call, the runtime fires a task and defers the
    result. Monty continues discovering more calls until it needs a value,
    at which point the runtime awaits completed tasks and provides results.
    """

    async def run(
        self,
        code: str,
        functions: list[str],
        call_tool: ToolCallback,
        *,
        signatures: list[FunctionSignature],
    ) -> Any:
        """Execute code in the Monty sandbox.

        Args:
            code: LLM-generated Python source code.
            functions: Names of external functions (tools) the code uses.
            call_tool: Callback invoked for each external function call.
            signatures: Function signatures for type checking.

        Returns:
            The final output of the code execution.
        """
        try:
            monty = Monty(code=code, external_functions=functions)
            monty.type_check(_build_type_check_prefix(signatures))
        except MontyTypingError as e:
            raise CodeTypingError(e.display(format='concise')) from e
        except MontyRuntimeError as e:
            raise CodeRuntimeError(e.display()) from e
        except MontySyntaxError as e:
            raise CodeSyntaxError(e.display()) from e

        monty_state: MontyComplete | MontyFutureSnapshot | MontySnapshot | None = None
        tasks: dict[int, asyncio.Task[Any]] = {}

        try:
            limits = self._build_resource_limits()
            monty_state = monty.start(limits=limits)
            monty_state = await MontyRuntime._execution_loop(monty_state, tasks, call_tool)

        except MontyRuntimeError as e:
            self._raise_if_timeout(e)
            raise CodeRuntimeError(e.display()) from e

        return monty_state.output

    @property
    def instructions(self) -> str | None:
        return (
            'Syntax note: the runtime uses a restricted Python subset.\n'
            '- Imports are not available — use the provided functions and builtins (len, sum, str, etc.) or define your own helpers.'
        )

    def _build_resource_limits(self) -> ResourceLimits | None:
        """Build Monty ResourceLimits from execution_timeout, or None if unset."""
        if self.execution_timeout is not None:
            return ResourceLimits(max_duration_secs=self.execution_timeout)
        return None

    def _raise_if_timeout(self, e: MontyRuntimeError) -> None:
        """Raise CodeExecutionTimeout if the MontyRuntimeError is a time limit violation."""
        # Coupling: Monty surfaces time limit violations as RuntimeErrors containing
        # 'time limit exceeded' in the display string. There is no structured error type
        # for this yet — if Monty changes the wording, this detection will break.
        if self.execution_timeout is not None and 'time limit exceeded' in e.display():
            raise CodeExecutionTimeout(f'Code execution timed out after {self.execution_timeout} seconds') from e

    @staticmethod
    async def _execution_loop(
        monty_state: MontyComplete | MontyFutureSnapshot | MontySnapshot,
        tasks: dict[int, asyncio.Task[Any]],
        call_tool: ToolCallback,
    ) -> MontyComplete:
        tool_call_id_to_call: dict[int, FunctionCall] = {}

        try:
            while not isinstance(monty_state, MontyComplete):
                if isinstance(monty_state, MontySnapshot):
                    call = FunctionCall(
                        call_id=str(monty_state.call_id),
                        function_name=monty_state.function_name,
                        args=monty_state.args,
                        kwargs=monty_state.kwargs,
                    )
                    tasks[monty_state.call_id] = asyncio.ensure_future(call_tool(call))
                    tool_call_id_to_call[monty_state.call_id] = call

                    monty_state = monty_state.resume(future=...)
                elif isinstance(monty_state, MontyFutureSnapshot):
                    pending_ids = monty_state.pending_call_ids or []
                    missing = [cid for cid in pending_ids if cid not in tasks]
                    if missing:
                        raise CodeRuntimeError(f'Monty expects results for call IDs {missing} but no tasks exist')
                    pending = [tasks[cid] for cid in pending_ids]
                    if not pending:
                        # No pending tasks - this can happen if all results are already available
                        # Just provide empty results and let Monty continue
                        monty_state = monty_state.resume(results={})
                        continue
                    done, _ = await asyncio.wait(pending, return_when=asyncio.ALL_COMPLETED)
                    task_to_cid = {id(t): cid for cid, t in tasks.items()}
                    results: dict[int, Any] = {}
                    interrupted_calls: list[InterruptedToolCall] = []
                    for task in done:
                        cid = task_to_cid[id(task)]
                        try:
                            raw = task.result()
                            # Normalize to JSON-compatible form (dicts, lists, strings, numbers)
                            # so that Monty's restricted interpreter can handle the value and
                            # behavior is consistent with driver-based runtimes
                            # (which serialize results over the JSON protocol).
                            results[cid] = {'return_value': tool_return_ta.dump_python(raw, mode='json')}
                        except (CallDeferred, ApprovalRequired) as e:
                            interrupted_calls.append(InterruptedToolCall(reason=e, call=tool_call_id_to_call[cid]))
                        del tasks[cid]

                    if interrupted_calls:
                        raise CodeInterruptedError(
                            interrupted_calls=interrupted_calls,
                        )

                    monty_state = monty_state.resume(results=results)

                    if isinstance(monty_state, MontyComplete):
                        break
        finally:
            for t in tasks.values():
                t.cancel()

        return monty_state
