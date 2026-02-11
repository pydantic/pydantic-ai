from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from typing import Any

import pydantic

from pydantic_ai.exceptions import ApprovalRequired, CallDeferred
from pydantic_ai.messages import ToolReturnContent, tool_return_ta
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
        ExternalResult,
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

# TypeAdapter for deserializing checkpoint values with proper type reconstruction.
# Uses ToolReturnContent (which includes the discriminated MultiModalContent union)
# so that rich types like BinaryContent and FileUrl subclasses are reconstructed
# from their JSON representations via the `kind` discriminator, rather than being
# left as plain dicts after round-tripping through JSON.
_checkpoint_result_ta: pydantic.TypeAdapter[Any] = pydantic.TypeAdapter(
    ToolReturnContent,
    config=pydantic.ConfigDict(defer_build=True, ser_json_bytes='base64', val_json_bytes='base64'),
)


@dataclass
class _DeserializedCheckpoint:
    monty_dump: bytes
    completed_results: dict[str, Any]
    pending_calls: dict[str, dict[str, Any]]


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


def _serialize_checkpoint(
    monty_dump: bytes,
    completed_results: dict[int, Any],
    interrupted_calls: list[InterruptedToolCall],
) -> bytes:
    """Serialize a Monty checkpoint into an opaque bytes blob.

    Bundles interpreter state, completed results, and interrupted call details
    so the runtime can fully resume without any external data.

    Each completed result is individually serialized to JSON bytes via Pydantic's
    ``tool_return_ta``, then base64-encoded for embedding in the outer JSON payload.
    On deserialization, ``_checkpoint_result_ta`` (backed by ``ToolReturnContent``)
    reconstructs rich types (``BinaryContent``, ``ImageUrl``, etc.) from their JSON
    via the ``kind`` discriminator — plain dicts/strings/numbers pass through unchanged.

    Args:
        monty_dump: Raw Monty interpreter state bytes from MontyFutureSnapshot.dump().
        completed_results: Results from tool calls that completed before the interruption,
            keyed by Monty's internal int call IDs, values in Monty's {'return_value': v} format.
        interrupted_calls: The tool calls that were interrupted, with full FunctionCall details.
    """
    # Each value is serialized to JSON bytes individually so that deserialization can
    # use validate_json (which respects val_json_bytes='base64') for proper round-tripping.
    raw_results = {
        str(k): base64.b64encode(tool_return_ta.dump_json(v['return_value'])).decode('ascii')
        for k, v in completed_results.items()
    }
    pending_calls = {
        ic.call.call_id: {
            'function_name': ic.call.function_name,
            'args': list(ic.call.args),
            'kwargs': ic.call.kwargs,
        }
        for ic in interrupted_calls
    }
    payload = {
        'interpreter_state': base64.b64encode(monty_dump).decode('ascii'),
        'completed_results': raw_results,
        'pending_calls': pending_calls,
    }
    return json.dumps(payload).encode('utf-8')


def _deserialize_checkpoint(checkpoint: bytes) -> _DeserializedCheckpoint:
    """Deserialize a Monty checkpoint back into its components."""
    payload = json.loads(checkpoint)
    if 'interpreter_state' not in payload:
        raise ValueError("Checkpoint missing required 'interpreter_state' key")
    return _DeserializedCheckpoint(
        monty_dump=base64.b64decode(payload['interpreter_state']),
        completed_results=payload.get('completed_results', {}),
        pending_calls=payload.get('pending_calls', {}),
    )


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
        signatures: list[str],
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
            await self._type_check(monty, code, signatures, functions)
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
            ckpt = _deserialize_checkpoint(checkpoint)
            monty_state: MontyComplete | MontyFutureSnapshot | MontySnapshot = MontyFutureSnapshot.load(ckpt.monty_dump)

            # Feed completed results (calls that succeeded before the interruption).
            # Values are deserialized through _checkpoint_result_ta (ToolReturnContent)
            # which reconstructs rich types (BinaryContent, etc.) via validate_json.
            if ckpt.completed_results:
                monty_results: dict[int, ExternalResult] = {}
                for k, v in ckpt.completed_results.items():
                    reconstructed = _checkpoint_result_ta.validate_json(base64.b64decode(v))
                    monty_results[int(k)] = {'return_value': reconstructed}
                monty_state = monty_state.resume(results=monty_results)
                if isinstance(monty_state, MontyComplete):
                    return monty_state.output

            # Fire tasks for each pending call, reconstructing FunctionCall from
            # the details stored in the checkpoint.
            tasks: dict[int, asyncio.Task[Any]] = {}
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

            monty_state = await MontyRuntime._execution_loop(monty_state, tasks, call_tool)
        except MontyRuntimeError as e:
            raise CodeRuntimeError(e.display())
        except (MontySyntaxError, MontyTypingError) as e:
            raise CodeRuntimeError(e.display())
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise CodeRuntimeError(f'Invalid checkpoint data: {e}')

        return monty_state.output

    def prompt_hints(self) -> str:
        return (
            'CRITICAL Syntax restrictions (the runtime uses a restricted Python subset):\n'
            '- No imports - use only the provided functions and builtins (len, sum, str, etc.) or write your own functions.'
        )

    async def _type_check(self, monty: Monty, code: str, signatures: list[str], external_functions: list[str]):
        prefix_code = _build_type_check_prefix(signatures)
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
                        results[cid] = {'return_value': task.result()}
                    except (CallDeferred, ApprovalRequired) as e:
                        interrupted_calls.append(InterruptedToolCall(type=e, call=tool_call_id_to_call[cid]))
                    except Exception as e:
                        for remaining in tasks.values():
                            remaining.cancel()
                        raise CodeRuntimeError(f'Tool execution error: {e}') from e
                    del tasks[cid]

                # Save checkpoint BEFORE advancing if there are interrupted calls.
                # The checkpoint captures the state where Monty is waiting for these results.
                # Completed results are bundled into the checkpoint so they can be replayed
                # on resume without re-executing those calls (avoiding double execution / side effects).
                if interrupted_calls:
                    for remaining in tasks.values():
                        remaining.cancel()
                    monty_dump = monty_state.dump()
                    checkpoint = _serialize_checkpoint(monty_dump, results, interrupted_calls)
                    raise CodeInterruptedError(
                        interrupted_calls=interrupted_calls,
                        checkpoint=checkpoint,
                    )

                monty_state = monty_state.resume(results=results)

                if isinstance(monty_state, MontyComplete):
                    break

        return monty_state
