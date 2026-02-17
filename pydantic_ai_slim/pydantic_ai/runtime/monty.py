from __future__ import annotations

import asyncio
import textwrap
from typing import Any

from pydantic_ai._python_signature import FunctionSignature, TypeSignature
from pydantic_ai.runtime.abstract import (
    CodeExecutionTimeout,
    CodeRuntime,
    CodeRuntimeError,
    CodeSyntaxError,
    CodeTypingError,
    FunctionCall,
    ToolCallback,
)

try:
    from pydantic_monty import (
        ExternalReturnValue,
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


_TYPING_IMPORTS = 'from typing import Any, TypedDict, NotRequired, Literal'


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
        call_tool: ToolCallback,
        *,
        functions: dict[str, FunctionSignature],
        referenced_types: list[TypeSignature],
    ) -> Any:
        """Execute code in the Monty sandbox.

        Args:
            code: LLM-generated Python source code.
            call_tool: Callback invoked for each external function call.
            functions: Mapping of function name to signature, for type checking
                and declaring external functions.
            referenced_types: Unique type definitions referenced by the signatures.

        Returns:
            The final output of the code execution.
        """
        try:
            monty = Monty(code=code, external_functions=list(functions))
            monty.type_check(self._build_type_check_prefix(list(functions.values()), referenced_types))
        except MontyTypingError as e:
            raise CodeTypingError(e.display(format='concise')) from e
        except MontyRuntimeError as e:  # pragma: no cover
            raise CodeRuntimeError(e.display()) from e
        except MontySyntaxError as e:
            raise CodeSyntaxError(e.display()) from e

        monty_state: MontyComplete | MontyFutureSnapshot | MontySnapshot | None = None
        try:
            limits = (
                ResourceLimits(max_duration_secs=self.execution_timeout) if self.execution_timeout is not None else None
            )
            monty_state = monty.start(limits=limits)
            monty_state = await self._execution_loop(monty_state, call_tool, functions=functions)
        except MontyRuntimeError as e:
            self._raise_if_timeout(e)
            raise CodeRuntimeError(e.display()) from e

        return monty_state.output

    @property
    def instructions(self) -> str | None:
        # f-string with `{""}` breaks up the triple backticks so pytest-examples
        # doesn't pick up the code block as a testable example.
        return textwrap.dedent(
            f"""
            The runtime uses a restricted Python subset:
            - you cannot use the standard library except builtin functions and the following modules: `sys`, `typing`, `asyncio`
            - this means `collections`, `json`, `re`, `math`, `datetime`, `itertools`, `functools`, etc. are NOT available — use plain dicts, lists, and builtins instead
            - you cannot use third party libraries
            - you cannot define classes
            - `sorted()` and `.sort()` do not support keyword arguments (`key=`, `reverse=`) and cannot sort lists of tuples — only sort flat lists of numbers or strings. If you need a custom sort order, build the output list manually (e.g. find max in a loop)
            - chained subscript assignment like `x[a][b] = val` is NOT supported — read into a local variable, modify it, then assign back: `inner = x[a]; inner[b] = val; x[a] = inner`
            - set operators (`|`, `&`, `-`, `^`) are not supported — use `set.update()`, `set.add()`, or loop to combine sets

            The last expression evaluated is the return value.

            To run independent calls concurrently, fire them first, then `await`, or use `asyncio.gather`:
            `{''}``python
            # starts immediately:
            items_future = get_items()
            users_future = get_users()

            # wait for results:
            items = await items_future
            users = await users_future

            # or equivalently:
            import asyncio
            items, users = await asyncio.gather(items_future, users_future)
            `{''}``
            """
        )

    def _build_type_check_prefix(
        self, signatures: list[FunctionSignature], referenced_types: list[TypeSignature]
    ) -> str:
        """Build the prefix code used for Monty type checking.

        Combines standard typing imports with tool signatures to create the
        prefix that Monty uses for type-checking LLM-generated code.

        Note: Signatures use `...` as the body by default, but ty/Monty requires
        `raise NotImplementedError()` for valid function stubs. See:
        https://github.com/astral-sh/ty/issues/1922

        Args:
            signatures: List of Python function signatures for available tools.
            referenced_types: Unique type definitions referenced by the signatures.

        Returns:
            Complete prefix code string with imports and signatures.
        """
        parts = [_TYPING_IMPORTS]
        parts.extend(str(t) for t in referenced_types)
        parts.extend(sig.render('raise NotImplementedError()') for sig in signatures)

        return '\n\n'.join(parts)

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
        call_tool: ToolCallback,
        *,
        functions: dict[str, FunctionSignature],
    ) -> MontyComplete:
        tasks: dict[int, asyncio.Task[Any]] = {}
        try:
            while not isinstance(monty_state, MontyComplete):
                if isinstance(monty_state, MontySnapshot):
                    call = FunctionCall(
                        call_id=f'monty_{monty_state.call_id}',
                        function_name=monty_state.function_name,
                        args=monty_state.args,
                        kwargs=monty_state.kwargs,
                    )
                    sig = functions.get(monty_state.function_name)

                    if sig and not sig.is_async:
                        # Sequential: drain pending async tasks, then call synchronously
                        if tasks:
                            await asyncio.gather(*tasks.values())
                        result = await call_tool(call)
                        monty_state = monty_state.resume(return_value=result)
                    else:
                        # Async: fire and defer (existing behavior)
                        tasks[monty_state.call_id] = asyncio.ensure_future(call_tool(call))
                        monty_state = monty_state.resume(future=...)
                elif isinstance(monty_state, MontyFutureSnapshot):
                    pending_call_ids = monty_state.pending_call_ids
                    if not pending_call_ids:  # pragma: no cover
                        monty_state = monty_state.resume(results={})
                        continue

                    try:
                        pending_tasks = [tasks[call_id] for call_id in pending_call_ids]
                    except KeyError as e:  # pragma: no cover
                        raise CodeRuntimeError(
                            f'Monty expects results for call IDs {pending_call_ids} but no tasks exist'
                        ) from e

                    try:
                        task_results = await asyncio.gather(*pending_tasks)
                    except BaseException:
                        for t in pending_tasks:
                            t.cancel()
                        raise
                    finally:
                        for call_id in pending_call_ids:
                            del tasks[call_id]

                    monty_state = monty_state.resume(
                        results={
                            call_id: ExternalReturnValue(return_value=result)
                            for call_id, result in zip(pending_call_ids, task_results)
                        }
                    )
        finally:
            for t in tasks.values():
                t.cancel()

        return monty_state
