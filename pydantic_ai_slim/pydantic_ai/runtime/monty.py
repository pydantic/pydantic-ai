from __future__ import annotations

import asyncio
from typing import Any

from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry
from pydantic_ai.runtime.abstract import (
    CodeInterruptedError,
    CodeRuntime,
    CodeRuntimeError,
    CodeSyntaxError,
    CodeTypingError,
    FunctionCall,
    InterruptedToolCall,
    ToolCallback,
)

try:
    from pydantic_monty import (
        Monty,
        MontyComplete,
        MontyFutureSnapshot,
        MontyRuntimeError,
        MontySnapshot,
        MontySyntaxError,
        MontyTypingError,
    )
except ImportError:
    raise ImportError("MontyRuntime requires 'monty'. Install with: pip install 'pydantic-ai-slim[monty]'")


def _build_type_check_prefix(signatures: list[str]) -> str:
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
    imports = 'from typing import Any, TypedDict, NotRequired, Literal\n\n'
    # Convert `...` body to `raise NotImplementedError()` for ty/Monty compatibility
    # The body is always on its own line with 4-space indent at the end of the function
    converted = [sig.replace('\n    ...', '\n    raise NotImplementedError()') for sig in signatures]
    return imports + '\n\n'.join(converted)


class MontyRuntime(CodeRuntime):
    """CodeRuntime that delegates to the Monty sandboxed interpreter.

    Monty provides sandboxed execution with built-in type checking and
    snapshot-based checkpointing. Code is executed in isolation and pauses
    at every external function call, returning a MontySnapshot that can be
    resumed once the host has computed the function's return value.

    Tool calls are executed concurrently via asyncio tasks. When Monty
    discovers an external call, the runtime fires a task and defers the
    result. Monty continues discovering more calls until it needs a value,
    at which point the runtime awaits completed tasks and provides results.
    """

    async def run(self, code: str, functions: list[str], call_tool: ToolCallback, *, signatures: list[str]) -> Any:
        """Start executing code in the Monty sandbox.

        Args:
            code: LLM-generated Python source code.
            functions: Names of external functions(tools) the code uses.
            call_tool: Callback invoked for each external function call.

        Returns:
            The final output of the code execution.

        Raises:
            CodeSyntaxError: If Monty cannot parse the code.
            CodeRuntimeError: If execution fails.
        """
        try:
            monty = Monty(code=code, external_functions=functions)
            await self._type_check(code, signatures)
        except MontyTypingError as e:
            raise CodeTypingError(e.display(format='concise'))
        except MontyRuntimeError as e:
            raise CodeRuntimeError(e.display())
        except MontySyntaxError as e:
            raise CodeSyntaxError(e.display())

        monty_state: MontyComplete | MontyFutureSnapshot | MontySnapshot | None = None
        tasks: dict[int, asyncio.Task[Any]] = {}

        try:
            monty_state = monty.start()
            monty_state = await MontyRuntime._execution_loop(monty_state, tasks, call_tool)

        except MontyRuntimeError as e:
            raise CodeRuntimeError(e.display())

        return monty_state.output

    async def _type_check(self, code: str, signatures: list[str]):
        prefix_code = _build_type_check_prefix(signatures)
        monty = Monty(code=code)
        monty.type_check(prefix_code=prefix_code)

    @staticmethod
    async def _execution_loop(
        monty_state: MontyComplete | MontyFutureSnapshot | MontySnapshot,
        tasks: dict[int, asyncio.Task[Any]],
        call_tool: ToolCallback,
    ):
        tool_call_id_to_call: dict[int, FunctionCall] = {}

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
                pending = [tasks[cid] for cid in pending_ids if cid in tasks]
                if not pending:
                    # No pending tasks - this can happen if all results are already available
                    # Just provide empty results and let Monty continue
                    monty_state = monty_state.resume(results={})
                    continue
                done, _ = await asyncio.wait(pending, return_when=asyncio.ALL_COMPLETED)
                results: dict[int, Any] = {}
                interrupted_calls: list[InterruptedToolCall] = []
                for task in done:
                    for cid, t in list(tasks.items()):
                        if t is task:
                            try:
                                results[cid] = {'return_value': task.result()}

                            except (CallDeferred, ApprovalRequired) as e:
                                interrupted_calls.append(InterruptedToolCall(type=e, call=tool_call_id_to_call[cid]))
                            except ModelRetry as e:
                                for remaining in tasks.values():
                                    remaining.cancel()

                                raise
                            except Exception as e:
                                for remaining in tasks.values():
                                    remaining.cancel()

                                raise ModelRetry(str(e))

                            del tasks[cid]
                            break

                # Save checkpoint BEFORE advancing if there are interrupted calls.
                # The checkpoint captures the state where Monty is waiting for these results.
                if interrupted_calls:
                    checkpoint = monty_state.dump()
                    raise CodeInterruptedError(interrupted_calls=interrupted_calls, checkpoint=checkpoint)

                monty_state = monty_state.resume(results=results)

                if isinstance(monty_state, MontyComplete):
                    break

        return monty_state

    async def resume(self, checkpoint: bytes, call_tool: ToolCallback, interrupted_calls: list[dict[str, Any]]) -> Any:
        try:
            tasks: dict[int, asyncio.Task[Any]] = {}
            monty_state = MontyFutureSnapshot.load(checkpoint)

            # Fire tasks for each pending call using the interrupted_calls details.
            # The callback has results_map populated, so these will resolve immediately
            # (either returning a value directly, or executing approved tools).
            pending_ids = monty_state.pending_call_ids or []
            for ic in interrupted_calls:
                call_id_int = int(ic['call_id'])
                if call_id_int in pending_ids:
                    call = FunctionCall(
                        call_id=ic['call_id'],
                        function_name=ic['tool_name'],
                        args=tuple(ic.get('args', ())),
                        kwargs=ic.get('kwargs', {}),
                    )
                    tasks[call_id_int] = asyncio.ensure_future(call_tool(call))

            # Fail fast if checkpoint state doesn't match interrupted calls
            if interrupted_calls and not tasks:
                interrupted_ids = [ic['call_id'] for ic in interrupted_calls]
                raise CodeRuntimeError(
                    f'Checkpoint mismatch: interrupted_calls={interrupted_ids} but pending_call_ids={pending_ids}'
                )

            monty_state = await MontyRuntime._execution_loop(monty_state, tasks, call_tool)

        except MontyRuntimeError as e:
            raise CodeRuntimeError(e.display())

        return monty_state.output
