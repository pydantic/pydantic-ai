"""CodeRuntime implementation backed by the Monty sandboxed Python interpreter.

Monty executes LLM-generated code in a restricted environment and pauses
whenever the code calls an external function (tool). This module implements
the callback-based CodeRuntime protocol: the runtime drives a loop over
Monty's pause points, invoking the caller's callback for each tool call.

Concurrency model:
    When Monty pauses at an external function call (MontySnapshot), the runtime
    fires an asyncio.Task for the callback and tells Monty to defer the result
    (``resume(future=...)``). Monty continues discovering independent calls
    until it actually needs a value — at which point it returns a
    MontyFutureSnapshot with the pending call IDs. The runtime then awaits
    completed tasks and feeds partial results back to Monty. This allows
    independent tool calls to execute concurrently.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

from pydantic_ai.exceptions import ApprovalRequired
from pydantic_ai.runtime.abstract import (
    CodeRuntime,
    CodeRuntimeError,
    CodeSyntaxError,
    CodeTypeError,
    FunctionCall,
    ToolCallback,
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

    Tool calls are executed concurrently via asyncio tasks. When Monty
    discovers an external call, the runtime fires a task and defers the
    result. Monty continues discovering more calls until it needs a value,
    at which point the runtime awaits completed tasks and provides results.
    """

    async def run_with_tools(self, code: str, functions: list[str], call_tool: ToolCallback) -> Any:
        """Start executing code in the Monty sandbox.

        Args:
            code: LLM-generated Python source code.
            functions: Names of external functions the code is allowed to call.
            call_tool: Callback invoked for each external function call.

        Returns:
            The final output of the code execution.

        Raises:
            CodeSyntaxError: If Monty cannot parse the code.
            CodeRuntimeError: If execution fails.
        """
        m = monty.Monty(code, external_functions=functions)
        try:
            state = m.start()
        except monty.MontySyntaxError as e:
            raise CodeSyntaxError(e.display())
        except monty.MontyRuntimeError as e:
            raise CodeRuntimeError(e.display())

        return await self._execution_loop(state, call_tool)

    async def resume_with_tools(self, checkpoint: bytes, call_tool: ToolCallback) -> Any:
        """Resume a paused execution from a serialized checkpoint.

        Supports both plain MontySnapshot checkpoints (legacy/simple) and
        enriched checkpoints that include MontyFutureSnapshot state plus
        pending call info for concurrent execution.

        Args:
            checkpoint: Bytes previously obtained from ``_build_checkpoint``.
            call_tool: Callback invoked for each remaining external function call.

        Returns:
            The final output of the code execution.
        """
        state, pending_calls = self._load_checkpoint(checkpoint)
        return await self._execution_loop(state, call_tool, pending_calls)

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
        imports = 'from typing import Any, TypedDict, NotRequired, Literal\n\n'
        prefix_code = imports + '\n\n'.join(signatures)
        m = monty.Monty(code, external_functions=[])
        try:
            m.type_check(prefix_code=prefix_code)
        except monty.MontyTypingError as e:
            raise CodeTypeError(e.display())
        except monty.MontySyntaxError as e:
            raise CodeSyntaxError(e.display())

    async def _execution_loop(
        self,
        state: monty.MontySnapshot | monty.MontyFutureSnapshot | monty.MontyComplete,
        call_tool: ToolCallback,
        pending_calls: dict[int, FunctionCall] | None = None,
    ) -> Any:
        """Drive Monty's concurrent pause/resume loop.

        The loop handles three Monty state types:

        **Discovery phase** (``MontySnapshot``): Each external function call is
        launched as an ``asyncio.Task`` via ``call_tool``, and the result is
        deferred (``resume(future=...)``). Monty continues discovering the next
        independent call until it needs a value.

        **Resolution phase** (``MontyFutureSnapshot``): Monty needs deferred
        values. The runtime awaits completed tasks and feeds results back.
        If any task raises ``ApprovalRequired``, completed results are provided
        as partial results, and the approval is surfaced with a checkpoint.

        **Complete** (``MontyComplete``): Execution finished. Any remaining
        side-effect-only tasks are awaited before returning.

        On resume (via ``resume_with_tools``), previously-pending calls are
        re-launched from the checkpoint's ``pending_calls`` info. The consumer
        builds the callback with approval matching by tool name, so the
        previously-blocked tool call succeeds on retry.

        Args:
            state: The current Monty state.
            call_tool: Callback provided by ``CodeModeToolset._make_tool_callback()``.
            pending_calls: Optional dict of call_id -> FunctionCall for resumed
                concurrent executions. Tasks are re-launched for these calls.

        Returns:
            The final output once Monty completes.
        """
        tasks: dict[int, asyncio.Task[Any]] = {}
        call_info: dict[int, FunctionCall] = {}

        # On resume with pending calls: re-launch tasks for deferred calls.
        if pending_calls:
            for cid, call in pending_calls.items():
                task = asyncio.create_task(call_tool(call))
                tasks[cid] = task
                call_info[cid] = call

        try:
            while True:
                if isinstance(state, monty.MontySnapshot):
                    # === DISCOVERY PHASE ===
                    # Fire-and-forget: launch a task for each call, defer the result,
                    # and let Monty discover the next independent call.
                    while isinstance(state, monty.MontySnapshot):
                        call = FunctionCall(
                            function_name=state.function_name,
                            args=tuple(state.args) if state.args else (),
                            kwargs=dict(state.kwargs) if state.kwargs else {},
                        )
                        task = asyncio.create_task(call_tool(call))
                        tasks[state.call_id] = task
                        call_info[state.call_id] = call
                        try:
                            state = state.resume(future=...)
                        except monty.MontyRuntimeError as e:
                            await self._cancel_tasks(tasks)
                            raise CodeRuntimeError(e.display())

                elif isinstance(state, monty.MontyFutureSnapshot):
                    # === RESOLUTION PHASE ===
                    state = await self._resolve_futures(state, tasks, call_info)

                else:
                    # === COMPLETE ===
                    assert isinstance(state, monty.MontyComplete)
                    await self._await_remaining_tasks(tasks)
                    return state.output
        except BaseException:
            await self._cancel_tasks(tasks)
            raise

    async def _resolve_futures(
        self,
        future_state: monty.MontyFutureSnapshot,
        tasks: dict[int, asyncio.Task[Any]],
        call_info: dict[int, FunctionCall],
    ) -> monty.MontySnapshot | monty.MontyFutureSnapshot | monty.MontyComplete:
        """Await tasks for pending call IDs and provide results to Monty.

        Waits for all pending tasks to complete. If any raise
        ``ApprovalRequired``, completed results are provided as a partial
        resume, and the approval is re-raised with checkpoint metadata.

        Args:
            future_state: The Monty state waiting for future results.
            tasks: Mapping of call_id to running asyncio.Task.
            call_info: Mapping of call_id to FunctionCall (for checkpoint).

        Returns:
            The next Monty state after providing results.
        """
        pending_ids = future_state.pending_call_ids
        pending_tasks = {cid: tasks[cid] for cid in pending_ids if cid in tasks}
        task_to_cid: dict[asyncio.Task[Any], int] = {task: cid for cid, task in pending_tasks.items()}

        completed: dict[int, monty.ExternalReturnValue] = {}
        approval_needed: dict[int, ApprovalRequired] = {}

        remaining = set(pending_tasks.values())
        while remaining:
            done, remaining = await asyncio.wait(remaining, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                cid = task_to_cid[task]
                try:
                    completed[cid] = monty.ExternalReturnValue(return_value=task.result())
                except ApprovalRequired as e:
                    approval_needed[cid] = e
                except Exception:
                    # Non-approval exceptions propagate up; cleanup handled
                    # by the outer try/except in _execution_loop.
                    raise

        if approval_needed:
            # Provide partial results for completed calls before checkpointing.
            if completed:
                try:
                    future_state = future_state.resume(results=completed)  # type: ignore[assignment]
                except monty.MontyRuntimeError as e:
                    raise CodeRuntimeError(e.display())

            # TODO: Currently we surface one approval at a time (sequential rounds).
            # Each round is efficient -- completed results are already provided to Monty
            # via partial resume, so no work is repeated. But the user sees N approval
            # dialogs for N concurrent tools needing approval.
            #
            # To batch all approvals into one user interaction:
            # 1. Include ALL approval-needed calls in the ApprovalRequired metadata
            #    (not just the first), e.g. `'approval_calls': [...]`
            # 2. Update code_mode consumer to emit multiple `_approval_call` entries
            # 3. Update _agent_graph.py `_populate_deferred_calls()` to expand one
            #    outer run_code call into N synthetic deferred approval entries
            # 4. Update DeferredToolResults handling to pass back N approval results
            # 5. Update the callback to mark multiple calls as approved on resume
            #
            # Discussion pending on whether this UX improvement warrants the agent
            # graph complexity.

            # Pick the first deferred call for approval.
            first_cid = min(approval_needed.keys())
            first_exc = approval_needed[first_cid]

            # Build enriched checkpoint: Monty state + pending call info.
            remaining_calls = {cid: call_info[cid] for cid in approval_needed}
            checkpoint = self._build_checkpoint(future_state, remaining_calls)

            raise ApprovalRequired(
                metadata={
                    **(first_exc.metadata or {}),
                    'runtime_checkpoint': checkpoint,
                }
            )

        # All resolved -- provide all results and continue.
        # Remove resolved tasks from tracking dicts.
        for cid in pending_ids:
            tasks.pop(cid, None)
            call_info.pop(cid, None)

        try:
            return future_state.resume(results=completed)
        except monty.MontyRuntimeError as e:
            raise CodeRuntimeError(e.display())

    # ------------------------------------------------------------------
    # Checkpoint serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _build_checkpoint(
        state: monty.MontySnapshot | monty.MontyFutureSnapshot,
        pending_calls: dict[int, FunctionCall] | None = None,
    ) -> bytes:
        """Serialize Monty state to a checkpoint.

        For a ``MontyFutureSnapshot`` with pending calls, the checkpoint is a
        JSON envelope containing the Monty dump (base64-encoded) plus the
        ``FunctionCall`` info for each pending call_id. This is necessary
        because ``MontyFutureSnapshot.pending_call_ids`` only gives IDs, not
        the function name/args/kwargs needed to re-launch callbacks on resume.

        For a plain ``MontySnapshot`` (no pending calls), the checkpoint is
        just the raw Monty dump bytes.

        Args:
            state: The Monty snapshot to serialize.
            pending_calls: Optional mapping of call_id -> FunctionCall for
                concurrent calls that still need to be resolved.

        Returns:
            Opaque bytes suitable for ``_load_checkpoint``.
        """
        if pending_calls:
            data = {
                'monty': base64.b64encode(state.dump()).decode(),
                'pending_calls': {
                    str(cid): {
                        'function_name': call.function_name,
                        'args': list(call.args),
                        'kwargs': call.kwargs,
                    }
                    for cid, call in pending_calls.items()
                },
            }
            return json.dumps(data).encode()
        else:
            return state.dump()

    @staticmethod
    def _load_checkpoint(
        checkpoint: bytes,
    ) -> tuple[monty.MontySnapshot | monty.MontyFutureSnapshot, dict[int, FunctionCall] | None]:
        """Deserialize a checkpoint produced by ``_build_checkpoint``.

        Handles both enriched (JSON envelope) and plain (raw bytes)
        checkpoint formats.

        Args:
            checkpoint: Bytes from ``_build_checkpoint``.

        Returns:
            A tuple of (Monty state, optional pending calls dict).
        """
        try:
            data = json.loads(checkpoint.decode())
            if isinstance(data, dict) and 'monty' in data and 'pending_calls' in data:
                monty_bytes = base64.b64decode(data['monty'])
                # Try MontyFutureSnapshot first, fall back to MontySnapshot.
                try:
                    state: monty.MontySnapshot | monty.MontyFutureSnapshot = monty.MontyFutureSnapshot.load(
                        monty_bytes
                    )
                except Exception:
                    state = monty.MontySnapshot.load(monty_bytes)
                pending_calls = {
                    int(cid): FunctionCall(
                        function_name=info['function_name'],
                        args=tuple(info['args']),
                        kwargs=info['kwargs'],
                    )
                    for cid, info in data['pending_calls'].items()
                }
                return state, pending_calls
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError):
            pass
        # Legacy/simple: plain MontySnapshot bytes.
        return monty.MontySnapshot.load(checkpoint), None

    # ------------------------------------------------------------------
    # Task helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _cancel_tasks(tasks: dict[int, asyncio.Task[Any]]) -> None:
        """Cancel all running tasks and suppress cancellation errors."""
        for task in tasks.values():
            task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        tasks.clear()

    @staticmethod
    async def _await_remaining_tasks(tasks: dict[int, asyncio.Task[Any]]) -> None:
        """Await any still-running tasks (for side-effect-only calls)."""
        pending = [t for t in tasks.values() if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
