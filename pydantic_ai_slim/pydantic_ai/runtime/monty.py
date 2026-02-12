from __future__ import annotations

import asyncio
from typing import Any

from pydantic_ai._python_signature import Signature
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
    decode_checkpoint_results,
    deserialize_checkpoint,
    serialize_checkpoint_results,
)

try:
    from pydantic_monty import (
        ExternalResult,
        Monty,
        MontyComplete,
        MontyFutureSnapshot,
        MontyRuntimeError,
        MontySnapshot,
        MontySyntaxError,
        MontyTypingError,
        ResourceLimits,
    )
except ImportError:
    raise ImportError("MontyRuntime requires 'monty'. Install with: pip install 'pydantic-ai-slim[monty]'")


def _build_type_check_prefix(signatures: list[Signature]) -> str:
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
    parts = ['from typing import Any, TypedDict, NotRequired, Literal']  # TODO (Douwe): Move to better place
    parts.extend(sig.with_typeddicts('raise NotImplementedError()') for sig in signatures)

    return '\n\n'.join(parts)


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

    async def run(
        self,
        code: str,
        functions: list[str],
        call_tool: ToolCallback,
        *,
        signatures: list[Signature],
        checkpoint: bytes | None = None,
    ) -> Any:
        """Execute code in the Monty sandbox, or resume from a checkpoint.

        When `checkpoint` is None, starts a fresh execution (type check, create
        Monty, execute). When `checkpoint` is provided, deserializes the saved
        interpreter state, feeds completed results, fires interrupted tasks,
        and continues execution.

        Args:
            code: LLM-generated Python source code.
            functions: Names of external functions (tools) the code uses.
            call_tool: Callback invoked for each external function call.
            signatures: Function signatures for type checking.
            checkpoint: Opaque checkpoint from a previous CodeInterruptedError.

        Returns:
            The final output of the code execution.
        """
        if checkpoint is not None:
            # Monty's checkpoint contains the full interpreter state, so code/functions/signatures
            # are not needed for resume. Other runtimes (stdio-based) will use these parameters
            # for re-execution with a result cache.
            return await self._resume_from_checkpoint(checkpoint, call_tool)

        try:
            monty = Monty(code=code, external_functions=functions)
            self._type_check(monty, code, signatures, functions)
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

    async def _resume_from_checkpoint(self, checkpoint: bytes, call_tool: ToolCallback) -> Any:
        """Resume execution from a serialized checkpoint.

        Args:
            checkpoint: Opaque checkpoint bytes containing interpreter state,
                completed results, and pending call details.
            call_tool: Callback with pre-resolved results for interrupted calls.

        Returns:
            The final output of the resumed code execution.
        """
        try:
            ckpt = deserialize_checkpoint(checkpoint)
            if ckpt.interpreter_state is None:
                raise ValueError("Checkpoint missing required 'interpreter_state' key")
            monty_state: MontyComplete | MontyFutureSnapshot | MontySnapshot = MontyFutureSnapshot.load(
                ckpt.interpreter_state
            )

            # Feed completed results (calls that succeeded before the interruption).
            # decode_checkpoint_results handles base64 → validate_json → dump_python(mode='json').
            if ckpt.completed_results:
                decoded = decode_checkpoint_results(ckpt.completed_results)
                monty_results: dict[int, ExternalResult] = {int(k): {'return_value': v} for k, v in decoded.items()}
                monty_state = monty_state.resume(results=monty_results)
                if isinstance(monty_state, MontyComplete):
                    return monty_state.output

            # Fire tasks for each pending call, reconstructing FunctionCall from
            # the details stored in the checkpoint.
            tasks: dict[int, asyncio.Task[Any]] = {}
            initial_calls: dict[int, FunctionCall] = {}
            pending_ids = monty_state.pending_call_ids if isinstance(monty_state, MontyFutureSnapshot) else []

            # Validate that checkpoint data matches interpreter state
            checkpoint_ids = set(ckpt.pending_calls.keys())
            missing = {str(cid) for cid in pending_ids} - checkpoint_ids
            if missing:
                raise CodeRuntimeError(f'Checkpoint corrupt: pending call IDs {missing} not found in checkpoint data')

            for cid in pending_ids:
                call_id_str = str(cid)
                details = ckpt.pending_calls[call_id_str]
                call = FunctionCall(
                    call_id=call_id_str,
                    function_name=details['function_name'],
                    args=tuple(details.get('args', ())),
                    kwargs=details.get('kwargs', {}),
                )
                tasks[cid] = asyncio.ensure_future(call_tool(call))
                initial_calls[cid] = call

            # Note: Monty's resume() does not accept resource limits, so execution_timeout
            # only applies to the initial start(). Infinite loops after resume are a known
            # limitation of Monty's synchronous interpreter.
            monty_state = await MontyRuntime._execution_loop(monty_state, tasks, call_tool, initial_calls=initial_calls)
        except MontyRuntimeError as e:
            self._raise_if_timeout(e)
            raise CodeRuntimeError(e.display()) from e
        except MontySyntaxError as e:
            raise CodeSyntaxError(e.display()) from e
        except MontyTypingError as e:
            raise CodeTypingError(e.display(format='concise')) from e
        except (KeyError, ValueError) as e:
            raise CodeRuntimeError(f'Invalid checkpoint data: {e}') from e

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
        if self.execution_timeout is not None and 'time limit exceeded' in e.display():
            raise CodeExecutionTimeout(f'Code execution timed out after {self.execution_timeout} seconds') from e

    def _type_check(self, monty: Monty, code: str, signatures: list[Signature], external_functions: list[str]) -> None:
        prefix_code = _build_type_check_prefix(signatures)
        monty.type_check(prefix_code=prefix_code)

    @staticmethod
    async def _execution_loop(
        monty_state: MontyComplete | MontyFutureSnapshot | MontySnapshot,
        tasks: dict[int, asyncio.Task[Any]],
        call_tool: ToolCallback,
        initial_calls: dict[int, FunctionCall] | None = None,
    ) -> MontyComplete:
        # Seed with any calls created outside the loop (resume path).
        # Without this, tasks pre-created in _resume_from_checkpoint would not have
        # entries in tool_call_id_to_call, causing a KeyError if they raise
        # ApprovalRequired/CallDeferred during the MontyFutureSnapshot branch.
        tool_call_id_to_call: dict[int, FunctionCall] = dict(initial_calls) if initial_calls else {}

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
                            # behavior is consistent with checkpoint resume (which also
                            # round-trips through JSON) and with driver-based runtimes
                            # (which serialize results over the JSON protocol).
                            results[cid] = {'return_value': tool_return_ta.dump_python(raw, mode='json')}
                        except (CallDeferred, ApprovalRequired) as e:
                            interrupted_calls.append(InterruptedToolCall(reason=e, call=tool_call_id_to_call[cid]))
                        del tasks[cid]

                    # Save checkpoint BEFORE advancing if there are interrupted calls.
                    # The checkpoint captures the state where Monty is waiting for these results.
                    # Completed results are bundled into the checkpoint so they can be replayed
                    # on resume without re-executing those calls (avoiding double execution / side effects).
                    if interrupted_calls:
                        monty_dump = monty_state.dump()
                        unwrapped = {k: v['return_value'] for k, v in results.items()}
                        checkpoint = serialize_checkpoint_results(
                            unwrapped, interrupted_calls, interpreter_state=monty_dump
                        )
                        raise CodeInterruptedError(
                            interrupted_calls=interrupted_calls,
                            checkpoint=checkpoint,
                        )

                    monty_state = monty_state.resume(results=results)

                    if isinstance(monty_state, MontyComplete):
                        break
        finally:
            for t in tasks.values():
                t.cancel()

        return monty_state
