from __future__ import annotations

import asyncio
from typing import Any

from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry
from pydantic_ai.runtime.abstract import (
    CodeRuntime,
    CodeRuntimeError,
    CodeSyntaxError,
    CodeTypingError,
    FunctionCall,
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

    Args:
        signatures: List of Python function signatures for available tools.

    Returns:
        Complete prefix code string with imports and signatures.
    """
    imports = 'from typing import Any, TypedDict, NotRequired, Literal\n\n'
    return imports + '\n\n'.join(signatures)


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
            # Well first of all let us type check because Monty allows that
            await self._type_check(code, signatures)

        # Consider adding raise from None to keep these errors short and helpful for the LLMs to fix the code?

        except MontyTypingError as e:
            raise CodeTypingError(e.display(format='concise'))
        except MontyRuntimeError as e:
            raise CodeRuntimeError(e.display())
        except MontySyntaxError as e:
            raise CodeSyntaxError(e.display())

        # Code type checking worked so let us try to run it
        monty_state: MontyComplete | MontyFutureSnapshot | MontySnapshot | None = None
        tasks: dict[int, asyncio.Task[Any]] = {}

        try:
            monty_state = monty.start()
            # Seperated in case we end up adding resume after all
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
        while not isinstance(monty_state, MontyComplete):
            if isinstance(monty_state, MontySnapshot):
                # Do an external tool call
                tasks[monty_state.call_id] = asyncio.ensure_future(
                    call_tool(
                        FunctionCall(
                            function_name=monty_state.function_name,
                            args=monty_state.args,
                            kwargs=monty_state.kwargs,
                        )
                    )
                )
                monty_state = monty_state.resume(future=...)
            elif isinstance(monty_state, MontyFutureSnapshot):
                pending = [tasks[cid] for cid in monty_state.pending_call_ids if cid in tasks]
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                results: dict[int, Any] = {}
                for task in done:
                    for cid, t in list(tasks.items()):
                        if t is task:
                            try:
                                results[cid] = {'return_value': task.result()}

                            except ModelRetry:
                                # Cancel all the tool calls which are in flight
                                # It makes sense to cancel the calls though because if in the chain something does not work most probably nothing works?

                                for remaining in tasks.values():
                                    remaining.cancel()

                                raise

                            except (CallDeferred, ApprovalRequired):
                                # This is the tricky bit, once this works most of it shold be fine
                                # What should I do with the tasks that are currently in flight here
                                # Do I dump them, take this approval and come back because tasks anyway is not going to be retained across runs(I think)?
                                # If I finish them can I make it a part of the snapshot before I dump it?
                                # So I would hope to do everything and then dump it for when it resumes?

                                # I could have waited out the tool calls, saved the snapshot and resumed? Maybe something we should allow in Monty

                                monty_state.dump()  # I wish I could store my results before I dumped it? :(

                                raise
                            except Exception as e:
                                results[cid] = {'exception': e}
                            del tasks[cid]
                            break
                monty_state = monty_state.resume(results=results)

        return monty_state

    @staticmethod
    def dump(monty_state: MontySnapshot | MontyFutureSnapshot) -> bytes:
        return monty_state.dump()

    @staticmethod
    def load(dumped_monty_state: bytes) -> MontySnapshot:
        return MontySnapshot.load(dumped_monty_state)
