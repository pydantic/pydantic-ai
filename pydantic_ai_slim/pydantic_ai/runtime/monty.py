"""CodeRuntime implementation backed by the Monty sandboxed Python interpreter.

Monty executes LLM-generated code in a restricted environment and pauses
whenever the code calls an external function (tool). This module implements
the callback-based CodeRuntime protocol: the runtime drives a loop over
Monty's pause points, invoking the caller's callback for each tool call.
"""

from __future__ import annotations

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

        Args:
            checkpoint: Bytes previously obtained from a MontySnapshot dump.
            call_tool: Callback invoked for each remaining external function call.

        Returns:
            The final output of the code execution.
        """
        state = monty.MontySnapshot.load(checkpoint)
        return await self._execution_loop(state, call_tool)

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
        state: monty.MontySnapshot | monty.MontyComplete,
        call_tool: ToolCallback,
    ) -> Any:
        """Drive Monty's pause/resume loop, invoking call_tool at each pause point.

        The runtime owns this loop — not the consumer. Each iteration:
        1. Monty is paused at a tool call (MontySnapshot) — package it as a FunctionCall.
        2. Invoke the consumer's callback, which handles name mapping, tracing,
           approval context, and the actual inner tool call.
        3. Feed the callback's return value back to Monty to resume execution.

        If the callback raises ApprovalRequired (because the inner tool needs
        human approval), the runtime is the right place to handle it: we have
        direct access to `state.dump()` here, so we serialize the Monty VM
        state and attach it as `runtime_checkpoint` in the exception metadata.
        The consumer (CodeModeToolset.call_tool) will wrap this further with
        its own `code_mode` metadata before re-raising to the agent graph.

        On resume (via resume_with_tools), the same loop runs again from the
        restored snapshot. The consumer builds the callback with
        next_call_approved=True for the first call, so the previously-blocked
        tool call succeeds this time, and execution continues normally.

        Args:
            state: The current Monty state (snapshot or complete).
            call_tool: Callback provided by CodeModeToolset._make_tool_callback().

        Returns:
            The final output once Monty completes.
        """
        while isinstance(state, monty.MontySnapshot):
            # Package Monty's pause point into a vendor-agnostic FunctionCall.
            call = FunctionCall(
                function_name=state.function_name,
                args=tuple(state.args) if state.args else (),
                kwargs=dict(state.kwargs) if state.kwargs else {},
            )
            try:
                # Invoke the consumer's callback. Inside, this maps the sanitized
                # function name back to the original tool name, builds kwargs,
                # applies approval context, opens a tracing span, and calls the
                # real tool via WrapperToolset.call_tool().
                result = await call_tool(call)
            except ApprovalRequired as e:
                # The inner tool needs human approval. Since we own the Monty
                # state, we can serialize it here — the consumer callback can't
                # do this because it doesn't have access to the Monty snapshot.
                # We merge our checkpoint into the exception's existing metadata
                # (which contains tool_name, tool_args, etc. from the callback).
                checkpoint_bytes = state.dump()
                raise ApprovalRequired(
                    metadata={
                        **(e.metadata or {}),
                        'runtime_checkpoint': checkpoint_bytes,
                    }
                )
            try:
                # Feed the tool result back to Monty. This resumes the sandboxed
                # code from where it paused, with `result` as the return value
                # of the function call. The new state is either another
                # MontySnapshot (next tool call) or MontyComplete (code finished).
                state = state.resume(return_value=result)
            except monty.MontyRuntimeError as e:
                raise CodeRuntimeError(e.display())

        assert isinstance(state, monty.MontyComplete)
        return state.output
