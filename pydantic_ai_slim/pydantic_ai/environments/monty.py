"""Monty sandboxed interpreter environment for code execution.

Requires the `pydantic-monty` package: `pip install pydantic-ai-slim[monty]`
"""

from __future__ import annotations

import asyncio
import textwrap
from typing import TYPE_CHECKING, Any

from typing_extensions import Self, assert_never

from pydantic_ai.toolsets.code_execution._abstract import (
    CodeExecutionTimeout,
    CodeRuntimeError,
    CodeSyntaxError,
    FunctionCall,
)

from ._base import ExecutionEnvironment

if TYPE_CHECKING:
    from pydantic_ai._python_signature import FunctionSignature, TypeSignature
    from pydantic_ai.toolsets.code_execution._abstract import FunctionCallback

    from ._base import Capability

try:
    from pydantic_monty import (
        ExternalReturnValue,
        FunctionSnapshot,
        FutureSnapshot,
        MontyComplete,
        MontyRepl,
        MontyRuntimeError,
        MontySyntaxError,
        ResourceLimits,
    )
except ImportError as _import_error:
    raise ImportError(
        'Please install `pydantic-monty` to use MontyEnvironment, '
        'you can use the `monty` optional group — `pip install "pydantic-ai-slim[monty]"`'
    ) from _import_error


class MontyEnvironment(ExecutionEnvironment):
    """Execution environment using the Monty sandboxed REPL interpreter.

    Monty provides sandboxed execution with state persistence across calls.
    Code is executed incrementally via `MontyRepl.feed_start()`, which pauses
    at every external function call, returning a snapshot that can be resumed
    once the host has computed the function's return value.

    Print output from executed code is captured and returned alongside the
    result value.

    This environment only supports code execution (`run_code` capability).
    It does not provide shell, file, or search operations.
    """

    execution_timeout: float | None = None
    """Optional timeout in seconds for code execution. None means no timeout."""

    _repl: MontyRepl | None = None

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset({'run_python', 'run_python_with_functions'})

    def instructions(self, capability: Capability) -> str | None:
        if capability in ('run_python', 'run_python_with_functions'):
            return textwrap.dedent(
                """
                The runtime uses a restricted Python subset:
                - you cannot use the standard library except builtin functions and the following modules: `sys`, `typing`, `asyncio`
                - this means `collections`, `json`, `re`, `math`, `datetime`, `itertools`, `functools`, etc. are NOT available — use plain dicts, lists, and builtins instead
                - you cannot use third party libraries
                - you cannot define classes
                - `sorted()` and `.sort()` do not support keyword arguments (`key=`, `reverse=`) and cannot sort lists of tuples — only sort flat lists of numbers or strings. If you need a custom sort order, build the output list manually (e.g. find max in a loop)
                - chained subscript assignment like `x[a][b] = val` is NOT supported — read into a local variable, modify it, then assign back: `inner = x[a]; inner[b] = val; x[a] = inner`
                - set operators (`|`, `&`, `-`, `^`) are not supported — use `set.update()`, `set.add()`, or loop to combine sets

                State persists across calls — variables and functions defined in previous calls are available in subsequent calls.

                The last expression evaluated is the return value.

                Parallelism: use `asyncio.gather` to fire multiple calls at the same time instead of awaiting each one sequentially:

                    # GOOD — parallel (all calls fire at once):
                    results = await asyncio.gather(
                        get_data(id=1),
                        get_data(id=2),
                        get_data(id=3),
                    )

                    # BAD — sequential (each call waits before the next starts):
                    r1 = await get_data(id=1)
                    r2 = await get_data(id=2)
                    r3 = await get_data(id=3)
                """
            )
        return None  # pragma: no cover

    async def __aenter__(self) -> Self:
        self._ensure_repl()
        return self

    async def __aexit__(self, *args: Any) -> None:
        self._repl = None

    # TODO: Concurrent agent runs sharing a MontyEnvironment will fail because
    # MontyRepl has an internal mutex — only one snippet can execute at a time.
    # Once PR #4688 (for_run/for_run_step lifecycle hooks) lands,
    # CodeExecutionToolset.for_run() should return a new instance with a fresh
    # MontyEnvironment so each agent run gets its own REPL with isolated state.

    def _ensure_repl(self) -> MontyRepl:
        """Return the active REPL, creating it lazily if needed."""
        if self._repl is None:
            limits = (
                ResourceLimits(max_duration_secs=self.execution_timeout)
                if self.execution_timeout is not None
                else None
            )
            self._repl = MontyRepl(limits=limits)
            self._repl.feed_start('import asyncio')
        return self._repl

    async def run_python(self, code: str) -> Any:
        """Execute code in the Monty REPL sandbox without external functions."""
        # TODO: Re-enable type checking with accumulated stubs.
        # MontyRepl does not support type_check. We previously used a throwaway
        # Monty(code, type_check=True, type_check_stubs=...) to catch type errors
        # before execution. With REPL state persistence, the stubs would need to
        # include declarations for all variables/functions from prior snippets,
        # which requires tracking the REPL's type context across calls.
        prints: list[str] = []
        try:
            monty_state = self._ensure_repl().feed_start(
                code, print_callback=lambda _stream, text: prints.append(text)
            )
            if not isinstance(monty_state, MontyComplete):
                raise CodeRuntimeError(
                    'Unexpected external function call in code without functions.'
                )  # pragma: no cover
        except MontySyntaxError as e:
            raise CodeSyntaxError(e.display()) from e
        except MontyRuntimeError as e:
            self._raise_if_timeout(e)
            raise CodeRuntimeError(e.display()) from e

        return self._build_result(monty_state.output, prints)

    async def run_python_with_functions(
        self,
        code: str,
        *,
        function_callback: FunctionCallback,
        functions: dict[str, FunctionSignature] | None = None,
        referenced_types: list[TypeSignature] | None = None,
    ) -> Any:
        """Execute code in the Monty REPL sandbox with external function support."""
        if functions is None:
            functions = {}

        # TODO: Re-enable type checking — see comment in run_python.
        prints: list[str] = []
        try:
            monty_state = self._ensure_repl().feed_start(
                code, print_callback=lambda _stream, text: prints.append(text)
            )
            monty_state = await self._execution_loop(monty_state, function_callback, functions=functions)
        except MontySyntaxError as e:
            raise CodeSyntaxError(e.display()) from e
        except MontyRuntimeError as e:
            self._raise_if_timeout(e)
            raise CodeRuntimeError(e.display()) from e

        return self._build_result(monty_state.output, prints)

    @staticmethod
    def _build_result(output: Any, prints: list[str]) -> Any:
        """Combine the expression result with any captured print output."""
        printed_text = ''.join(prints).rstrip('\n')
        if not printed_text:
            return output

        if output is None:
            return printed_text

        return f'{printed_text}\n{output}'

    def _build_type_check_prefix(
        self, signatures: list[FunctionSignature], referenced_types: list[TypeSignature]
    ) -> str:
        """Build the prefix code used for Monty type checking.

        Currently unused while type checking is disabled for REPL mode,
        but retained for when type checking is re-enabled.
        """
        parts = ['import asyncio\nfrom typing import Any, TypedDict, NotRequired, Literal']
        parts.extend(str(t) for t in referenced_types)
        parts.extend(sig.render('raise NotImplementedError()') for sig in signatures)

        return '\n\n'.join(parts)

    def _raise_if_timeout(self, e: MontyRuntimeError) -> None:
        """Raise CodeExecutionTimeout if the MontyRuntimeError is a time limit violation."""
        if self.execution_timeout is not None and 'time limit exceeded' in e.display():
            raise CodeExecutionTimeout(f'Code execution timed out after {self.execution_timeout} seconds') from e

    @staticmethod
    async def _execution_loop(
        monty_state: MontyComplete | FutureSnapshot | FunctionSnapshot,
        function_callback: FunctionCallback,
        *,
        functions: dict[str, FunctionSignature],
    ) -> MontyComplete:
        tasks: dict[int, asyncio.Task[Any]] = {}
        try:
            while not isinstance(monty_state, MontyComplete):
                if isinstance(monty_state, FunctionSnapshot):
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
                        result = await function_callback(call)
                        monty_state = monty_state.resume(return_value=result)
                    else:
                        # Async: fire and defer (existing behavior)
                        tasks[monty_state.call_id] = asyncio.ensure_future(function_callback(call))
                        monty_state = monty_state.resume(future=...)
                elif isinstance(monty_state, FutureSnapshot):
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
                else:
                    assert_never(monty_state)
        finally:
            for t in tasks.values():
                t.cancel()

        return monty_state
