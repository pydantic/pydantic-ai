"""Tests for the `CodeMode` capability and the `CodeModeToolset` it wraps.

Style follows `pydantic_ai/tests/test_toolsets.py`: module-level
`pytestmark = pytest.mark.anyio`, an `anyio_backend` fixture, async tests, and a
`build_run_context` factory. The `anyio` package's pytest plugin is already
loaded by the project (no extra dev dependency needed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic_ai import (
    AbstractToolset,
    Agent,
    RunContext,
    Tool,
    ToolDefinition,
)
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tool_manager import ParallelExecutionMode
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import RunUsage
from pydantic_core import SchemaValidator, core_schema
from pydantic_monty import NOT_HANDLED, MountDir, OSAccess, OsFunction
from typing_extensions import TypedDict

from pydantic_ai_harness import CodeMode
from pydantic_ai_harness._monty_exec import PrintCapture
from pydantic_ai_harness.code_mode import CodeModeToolset
from pydantic_ai_harness.code_mode._toolset import (  # pyright: ignore[reportPrivateUsage]
    _SEARCH_TOOLS_MODIFIER,
    _TOOL_SEARCH_ADDENDUM,
    _global_mode_is_sequential,
    _sanitize_tool_name,
)

pytestmark = pytest.mark.anyio

T = TypeVar('T')


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (matching upstream pydantic-ai)."""
    return 'asyncio'


def build_run_context(deps: T, run_step: int = 0) -> RunContext[T]:
    """Build a `RunContext` for invoking toolsets directly in tests.

    Mirrors the helper at `pydantic_ai/tests/test_toolsets.py`.
    """
    return RunContext[T](
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=run_step,
        # A live queue so `ctx.enqueue` works in tests; a real run wires this to the run's queue.
        pending_messages=[],
    )


async def build_ctx(
    deps: T,
    toolset: AbstractToolset[T],
    run_step: int = 0,
    *,
    root_capability: Any = None,
) -> RunContext[T]:
    """Build a `RunContext` with a prepared `ToolManager`.

    Use this for tests that call `call_tool` -- `CodeModeToolset` requires
    `ctx.tool_manager` to be set.
    """
    from pydantic_ai.tool_manager import ToolManager

    ctx = build_run_context(deps, run_step=run_step)
    tm = ToolManager(toolset=toolset, root_capability=root_capability)
    prepared_tm = await tm.for_run_step(ctx)
    ctx.tool_manager = prepared_tm
    return ctx


# ---------------------------------------------------------------------------
# Sample tool functions used by tests
# ---------------------------------------------------------------------------


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def greet(name: str, greeting: str = 'Hello') -> str:
    """Greet someone."""
    return f'{greeting}, {name}!'


class Address(TypedDict):
    """A simple postal address."""

    street: str
    city: str


class Person(TypedDict):
    """A person with a home address."""

    name: str
    home: Address


def lookup_person(person: Person, count: int = 1) -> str:
    """Look up details for a person."""
    return f'{count}x {person["name"]} @ {person["home"]["street"]}'


# Hand-built `ToolDefinition` objects + a tiny stub toolset are used by
# `test_conflicting_typed_dicts_get_tool_name_prefix` to exercise the
# `needs_prefix=True` rendering path. Going through Pydantic's JSON schema generator
# would not produce a true `$def`-key collision (Pydantic disambiguates `$def` keys
# by Python class identity even when `__name__` matches), so we build the schemas by
# hand and feed them through a fake toolset.


def _make_address_tool_def(name: str, description: str, addr_field: str) -> ToolDefinition:
    """Build a `ToolDefinition` whose `$defs` contains an `Address` type with one field."""
    return ToolDefinition(
        name=name,
        description=description,
        parameters_json_schema={
            'type': 'object',
            '$defs': {
                'Address': {
                    'type': 'object',
                    'title': 'Address',
                    'properties': {addr_field: {'type': 'string'}},
                    'required': [addr_field],
                },
            },
            'properties': {
                'addr': {'$ref': '#/$defs/Address'},
                'label': {'type': 'string'},
            },
            'required': ['addr', 'label'],
        },
        return_schema={'type': 'string'},
    )


class _StaticToolset(AbstractToolset[object]):
    """A minimal `AbstractToolset` that returns a fixed set of `ToolDefinition`s.

    Mirrors the `MockToolsetWithInstructions` pattern from `pydantic_ai/tests/test_toolsets.py`.
    Used by tests that need to construct hand-crafted `ToolDefinition`s without going
    through the function-introspection pipeline.
    """

    def __init__(self, tool_defs: list[ToolDefinition], results: dict[str, Any] | None = None) -> None:
        self._tool_defs = tool_defs
        self._results = results or {}

    @property
    def id(self) -> str | None:
        return None  # pragma: no cover - required by AbstractToolset, never read in tests

    async def get_tools(self, ctx: RunContext[object]) -> dict[str, ToolsetTool[object]]:
        return {
            td.name: ToolsetTool(
                toolset=self,
                tool_def=td,
                max_retries=1,
                args_validator=_ANY_VALIDATOR,
            )
            for td in self._tool_defs
        }

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[object],
        tool: ToolsetTool[object],
    ) -> Any:
        # Tests always set up `_results` for every tool name they invoke; the
        # fallback exists only to keep the abstract contract satisfied.
        return self._results[name]


_ANY_VALIDATOR = SchemaValidator(schema=core_schema.any_schema())


def _build_function_toolset(*tools: Any) -> FunctionToolset[object]:
    return FunctionToolset[object](tools=[Tool(t) for t in tools])


# ---------------------------------------------------------------------------
# OTel / Logfire instrumentation (import block at module level)
# ---------------------------------------------------------------------------

try:
    from logfire.testing import CaptureLogfire

    logfire_installed = True
except ImportError:  # pragma: no cover
    logfire_installed = False


class TestCodeMode:
    # ---------------------------------------------------------------------------
    # `tools='all'` (default) behaviour
    # ---------------------------------------------------------------------------

    async def test_default_wraps_all_tools_behind_run_code(self) -> None:
        """`CodeMode()` exposes only `run_code` and renders every tool as an `async def`."""
        toolset = _build_function_toolset(add, greet)
        wrapper = CodeMode[object]().get_wrapper_toolset(toolset)
        assert isinstance(wrapper, CodeModeToolset)

        tools = await wrapper.get_tools(build_run_context(None))
        assert list(tools.keys()) == ['run_code']

        description = tools['run_code'].tool_def.description
        assert description is not None
        assert 'async def add(*, a: int, b: int) -> int' in description
        assert 'async def greet(*, name: str, greeting: str' in description
        assert '"""Add two numbers."""' in description
        # The base description must tell the model to await tool calls.
        assert 'await' in description

    async def test_run_code_description_explains_final_expression_return(self) -> None:
        """The model is told how to return a value after assigning it."""
        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)

        tools = await wrapper.get_tools(build_run_context(None))
        description = tools['run_code'].tool_def.description

        assert description is not None
        assert 'End the snippet with the value to return as a bare expression.' in description
        assert 'result = some_expression\nresult' in description
        assert 'Without a non-`None` final expression or print output, `run_code` returns `{}`.' in description
        assert 'A final expression that evaluates to `None` is treated as no result.' in description
        assert 'results = await asyncio.gather' not in description
        assert 'With `print()` output and no non-`None` final expression' in description
        assert 'With `print()` output and a plain, non-`None` final expression' in description
        assert 'With `print()` output and a multimodal final expression' in description

    async def test_run_code_function_examples_are_expressions(self) -> None:
        """Async, sync, and mixed function examples do not end on assignments."""
        cases: list[tuple[FunctionToolset[object], tuple[str, ...]]] = [
            (_build_function_toolset(add), ('e.g. `await tool_name(arg=value)`.',)),
            (
                FunctionToolset[object](tools=[Tool(add, sequential=True)]),
                ('e.g. `tool_name(arg=value)`.',),
            ),
            (
                FunctionToolset[object](tools=[Tool(add, sequential=True), Tool(greet)]),
                ('e.g. `await tool_name(arg=value)`.', 'e.g. `tool_name(arg=value)`.'),
            ),
        ]

        for toolset, expected_examples in cases:
            wrapper = CodeMode[object]().get_wrapper_toolset(toolset)
            assert isinstance(wrapper, CodeModeToolset)

            description = (await wrapper.get_tools(build_run_context(None)))['run_code'].tool_def.description

            assert description is not None
            assert all(example in description for example in expected_examples)
            assert 'e.g. `result =' not in description

    async def test_run_code_executes_call_through_monty(self) -> None:
        """End-to-end: `run_code` runs Python in Monty and dispatches to a sync wrapped tool."""
        toolset = _build_function_toolset(add)
        wrapper = CodeMode[object]().get_wrapper_toolset(toolset)
        assert isinstance(wrapper, CodeModeToolset)

        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        result = await wrapper.call_tool(
            'run_code',
            {'code': 'print(await add(a=2, b=3))'},
            ctx,
            tools['run_code'],
        )
        assert result.return_value == {'output': '5\n'}

        # Nested tool calls are recorded as ToolCallPart/ToolReturnPart pairs in metadata.
        assert result.metadata['code_mode'] is True
        calls = result.metadata['tool_calls']
        returns = result.metadata['tool_returns']
        assert list(calls.keys()) == ['pyd_ai_code_mode__1']
        assert calls['pyd_ai_code_mode__1'].tool_name == 'add'
        assert calls['pyd_ai_code_mode__1'].args == {'a': 2, 'b': 3}
        assert returns['pyd_ai_code_mode__1'].tool_name == 'add'
        assert returns['pyd_ai_code_mode__1'].content == 5

    async def test_run_code_executes_string_returning_tool_with_default_arg(self) -> None:
        """End-to-end: a string-returning tool with a default arg is callable from the sandbox.

        Exercises (a) string return values flowing back through the await/dispatch loop,
        (b) default-argument handling -- the LLM-side code only passes `name`, not `greeting`.
        """
        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(greet))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        result = await wrapper.call_tool(
            'run_code',
            {'code': "print(await greet(name='Alice'))"},
            ctx,
            tools['run_code'],
        )
        assert result.return_value == {'output': 'Hello, Alice!\n'}

    async def test_run_code_can_chain_multiple_tool_calls_in_one_snippet(self) -> None:
        """A realistic LLM snippet that calls two tools in one `run_code` invocation."""
        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(add, greet))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        code = "total = await add(a=2, b=3)\nmsg = await greet(name=str(total), greeting='Result is')\nprint(msg)"
        result = await wrapper.call_tool('run_code', {'code': code}, ctx, tools['run_code'])
        assert result.return_value == {'output': 'Result is, 5!\n'}

    async def test_run_code_parallel_tool_calls_via_gather(self) -> None:
        """Concurrent tool calls via asyncio.gather work and record all nested metadata."""
        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        code = 'import asyncio\nresults = await asyncio.gather(add(a=1, b=2), add(a=3, b=4))\nresults'
        result = await wrapper.call_tool('run_code', {'code': code}, ctx, tools['run_code'])
        assert result.return_value == [3, 7]

        # Both parallel calls are recorded in metadata.
        calls = result.metadata['tool_calls']
        returns = result.metadata['tool_returns']
        assert len(calls) == 2
        assert len(returns) == 2

    async def test_run_code_parallel_tool_calls_one_fails(self) -> None:
        """When one of several parallel tool calls fails, the error surfaces as ModelRetry."""

        def flaky(x: int) -> int:
            """Always fails."""
            raise ModelRetry('not allowed')

        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(add, flaky))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        code = 'import asyncio\nawait asyncio.gather(add(a=1, b=2), flaky(x=3))'
        with pytest.raises(ModelRetry, match='not allowed'):
            await wrapper.call_tool('run_code', {'code': code}, ctx, tools['run_code'])

    async def test_run_code_renders_no_arg_tool_signature(self) -> None:
        """A no-argument tool renders as `async def name() -> ...` (without `(*, ...)`).

        Covers the empty-params branch of `FunctionSignature._render` and verifies the
        no-args path through Monty round-trips correctly.
        """

        def now_iso() -> str:
            """Return a fake fixed timestamp."""
            return '2026-04-08T12:00:00Z'

        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(now_iso))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)

        description = tools['run_code'].tool_def.description
        assert description is not None
        # Note the lack of `(*, ...)` -- empty params render as `()`.
        assert 'async def now_iso() -> str' in description
        assert 'async def now_iso(*' not in description

        result = await wrapper.call_tool(
            'run_code',
            {'code': 'print(await now_iso())'},
            ctx,
            tools['run_code'],
        )
        assert result.return_value == {'output': '2026-04-08T12:00:00Z\n'}

    async def test_run_code_state_persists_between_calls(self) -> None:
        """REPL state must survive across consecutive `run_code` calls within a run."""
        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)

        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        run_code = tools['run_code']

        first = await wrapper.call_tool('run_code', {'code': 'x = await add(a=1, b=2)'}, ctx, run_code)
        assert first.return_value == {}  # assignment, no output, no expression result
        second = await wrapper.call_tool('run_code', {'code': 'print(x * 10)'}, ctx, run_code)
        assert second.return_value == {'output': '30\n'}

    async def test_run_code_restart_resets_repl_state(self) -> None:
        """Passing `restart=True` clears any previously-set names in the sandbox."""
        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)

        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        run_code = tools['run_code']

        await wrapper.call_tool('run_code', {'code': 'x = 99'}, ctx, run_code)
        # After restart, `x` should no longer exist -- on a fresh REPL the static
        # type checker catches undefined names before execution.
        with pytest.raises(ModelRetry, match=r'x'):
            await wrapper.call_tool('run_code', {'code': 'print(x)', 'restart': True}, ctx, run_code)

    async def test_run_code_returns_last_expression_value(self) -> None:
        """When the last statement is an expression, its value is returned in `result`."""
        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        result = await wrapper.call_tool('run_code', {'code': '1 + 2'}, ctx, tools['run_code'])
        # No print output → result returned directly (not wrapped in a dict).
        assert result.return_value == 3

    async def test_run_code_treats_none_as_no_expression_result(self) -> None:
        """A final `None` uses the same return shapes as no final expression."""
        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)

        result = await wrapper.call_tool('run_code', {'code': 'None'}, ctx, tools['run_code'])
        assert result.return_value == {}

        printed = await wrapper.call_tool('run_code', {'code': 'print("done")\nNone'}, ctx, tools['run_code'])
        assert printed.return_value == {'output': 'done\n'}

    async def test_run_code_syntax_error_becomes_model_retry(self) -> None:
        """A Python syntax error is surfaced as `ModelRetry` so the model can fix it."""
        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        run_code = tools['run_code']
        # Fresh REPL: type checker catches the syntax error.
        with pytest.raises(ModelRetry, match=r'Syntax error in code'):
            await wrapper.call_tool('run_code', {'code': 'def ('}, ctx, run_code)

        # Non-fresh REPL: feed_start catches the syntax error at runtime.
        await wrapper.call_tool('run_code', {'code': '1 + 1', 'restart': True}, ctx, run_code)
        with pytest.raises(ModelRetry, match=r'Syntax error in code'):
            await wrapper.call_tool('run_code', {'code': 'def ('}, ctx, run_code)

        # Non-fresh REPL: undefined name triggers NameLookupSnapshot → NameError.
        with pytest.raises(ModelRetry, match=r"name 'undefined_var' is not defined"):
            await wrapper.call_tool('run_code', {'code': 'print(undefined_var)'}, ctx, run_code)

    async def test_run_code_typing_error_becomes_model_retry(self) -> None:
        """A `MontyTypingError` from static type checking is translated into `ModelRetry`.

        On a fresh REPL (first call or after restart), the code is type-checked
        before execution using Monty's stateless type checker.
        """
        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)

        with pytest.raises(ModelRetry, match=r'Type error in code'):
            await wrapper.call_tool(
                'run_code',
                {'code': '"hello" + 1'},
                ctx,
                tools['run_code'],
            )

    # ---------------------------------------------------------------------------
    # `for_run` / `for_run_step` lifecycle
    # ---------------------------------------------------------------------------

    async def test_for_run_returns_fresh_instance_with_cleared_repl(self) -> None:
        """`for_run` must hand back a new toolset instance -- concurrent runs cannot share REPL state."""
        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)

        # Force lazy REPL creation on the *original* instance.
        tools = await wrapper.get_tools(ctx)
        await wrapper.call_tool('run_code', {'code': 'x = 1'}, ctx, tools['run_code'])
        assert wrapper._repl is not None  # pyright: ignore[reportPrivateUsage]

        fresh = await wrapper.for_run(ctx)
        assert isinstance(fresh, CodeModeToolset)
        assert fresh is not wrapper
        assert fresh._repl is None  # pyright: ignore[reportPrivateUsage]

    async def test_for_run_step_short_circuits_when_wrapped_unchanged(self) -> None:
        """If the inner toolset doesn't change between steps, `for_run_step` returns `self` unchanged."""
        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = build_run_context(None)
        same = await wrapper.for_run_step(ctx)
        assert same is wrapper

    async def test_for_run_step_preserves_repl_when_wrapped_changes(self) -> None:
        """When the wrapped toolset changes between steps, REPL state must carry over to the new instance."""

        class _SwappingToolset(AbstractToolset[object]):
            """Returns a *different* underlying toolset on each `for_run_step` call."""

            def __init__(self) -> None:
                self._inner = _build_function_toolset(add)
                self._step = 0

            @property
            def id(self) -> str | None:
                return None  # pragma: no cover - required by AbstractToolset, never read

            async def get_tools(self, ctx: RunContext[object]) -> dict[str, ToolsetTool[object]]:
                return await self._inner.get_tools(ctx)

            async def call_tool(  # pragma: no cover - test only exercises lifecycle methods, not call_tool
                self,
                name: str,
                tool_args: dict[str, Any],
                ctx: RunContext[object],
                tool: ToolsetTool[object],
            ) -> Any:
                return await self._inner.call_tool(name, tool_args, ctx, tool)

            async def for_run_step(self, ctx: RunContext[object]) -> AbstractToolset[object]:
                # Return a brand-new toolset on every step so `is` comparison fails in
                # `CodeModeToolset.for_run_step`, forcing the rebuild branch.
                self._step += 1
                new_self = _SwappingToolset()
                new_self._step = self._step
                return new_self

        wrapper = CodeMode[object]().get_wrapper_toolset(_SwappingToolset())
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)

        # Lazily create the REPL on the original instance.
        tools = await wrapper.get_tools(ctx)
        await wrapper.call_tool('run_code', {'code': 'x = 7'}, ctx, tools['run_code'])
        original_repl = wrapper._repl  # pyright: ignore[reportPrivateUsage]
        assert original_repl is not None

        next_step = await wrapper.for_run_step(ctx)
        assert isinstance(next_step, CodeModeToolset)
        assert next_step is not wrapper
        # State carries over so the LLM doesn't lose its variables between steps.
        assert next_step._repl is original_repl  # pyright: ignore[reportPrivateUsage]

    # ---------------------------------------------------------------------------
    # Filter behaviour
    # ---------------------------------------------------------------------------

    async def test_filter_keeps_rejected_tools_native(self) -> None:
        """A callable filter sandboxes accepted tools and leaves the rest visible to the model."""
        capability = CodeMode[object](tools=lambda ctx, td: td.name == 'add')
        wrapper = capability.get_wrapper_toolset(_build_function_toolset(add, greet))
        assert isinstance(wrapper, CodeModeToolset)

        tools = await wrapper.get_tools(build_run_context(None))
        assert sorted(tools.keys()) == ['greet', 'run_code']

        description = tools['run_code'].tool_def.description
        assert description is not None
        assert 'async def add(*, a: int, b: int)' in description
        # `greet` is exposed natively, so it must NOT appear inside the run_code description
        assert 'async def greet' not in description

    async def test_native_tool_call_passes_through(self) -> None:
        """Calling a native (non-sandboxed) tool passes through to the wrapped toolset."""
        capability = CodeMode[object](tools=lambda ctx, td: td.name == 'add')
        wrapper = capability.get_wrapper_toolset(_build_function_toolset(add, greet))
        assert isinstance(wrapper, CodeModeToolset)

        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        result = await wrapper.call_tool('greet', {'name': 'Alice', 'greeting': 'Hi'}, ctx, tools['greet'])
        assert result == 'Hi, Alice!'

    async def test_native_tool_named_run_code_raises_user_error(self) -> None:
        """A native tool named `run_code` raises UserError (reserved name)."""
        from pydantic_ai.exceptions import UserError

        def run_code() -> str:
            """A tool that collides with the reserved name."""
            return 'oops'  # pragma: no cover

        capability = CodeMode[object](tools=lambda ctx, td: td.name != 'run_code')
        wrapper = capability.get_wrapper_toolset(_build_function_toolset(run_code, add))
        assert isinstance(wrapper, CodeModeToolset)

        with pytest.raises(UserError, match="'run_code' is reserved"):
            await wrapper.get_tools(build_run_context(None))

    async def test_sandboxed_tool_named_run_code_raises_user_error(self) -> None:
        """A sandboxed tool named `run_code` raises UserError (conflicts with meta-tool)."""
        from pydantic_ai.exceptions import UserError

        def run_code() -> str:
            """A tool that collides with the meta-tool name."""
            return 'oops'  # pragma: no cover

        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(run_code, add))
        assert isinstance(wrapper, CodeModeToolset)

        with pytest.raises(UserError, match='conflicts with the code mode'):
            await wrapper.get_tools(build_run_context(None))

    async def test_filter_excluding_everything_yields_run_code_with_no_functions(self) -> None:
        """A filter that rejects every tool produces a `run_code` with no functions block."""
        capability = CodeMode[object](tools=lambda ctx, td: False)
        wrapper = capability.get_wrapper_toolset(_build_function_toolset(add, greet))
        assert isinstance(wrapper, CodeModeToolset)

        tools = await wrapper.get_tools(build_run_context(None))
        assert sorted(tools.keys()) == ['add', 'greet', 'run_code']

        description = tools['run_code'].tool_def.description
        assert description is not None
        assert 'functions are available inside the sandbox' not in description

    async def test_filter_uses_run_context_for_dynamic_decisions(self) -> None:
        """The filter receives the live `RunContext` so it can vary per run/step."""
        seen_steps: list[int] = []

        def filter_func(ctx: RunContext[object], td: Any) -> bool:
            seen_steps.append(ctx.run_step)
            return td.name == 'add'

        wrapper = CodeMode[object](tools=filter_func).get_wrapper_toolset(_build_function_toolset(add, greet))
        assert isinstance(wrapper, CodeModeToolset)
        await wrapper.get_tools(build_run_context(None, run_step=7))
        assert 7 in seen_steps

    # ---------------------------------------------------------------------------
    # TypedDict prelude rendering
    # ---------------------------------------------------------------------------

    async def test_typed_dict_arguments_render_as_prelude(self) -> None:
        """Tools with structured (TypedDict) parameters render their types in the prelude."""
        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(lookup_person))
        assert isinstance(wrapper, CodeModeToolset)

        description = (await wrapper.get_tools(build_run_context(None)))['run_code'].tool_def.description
        assert description is not None
        # Type prelude
        assert 'class Address(TypedDict):' in description
        assert 'street: str' in description
        assert 'class Person(TypedDict):' in description
        assert 'home: Address' in description
        # Function signature references the TypedDict
        assert 'async def lookup_person(*, person: Person, count: int = 1) -> str' in description

    async def test_typed_dict_argument_round_trips_through_monty(self) -> None:
        """End-to-end with a structured argument: dict literal flows through Monty into the tool.

        The dict literal is constructed incrementally across two REPL calls so
        that static type checking (which only runs on the first snippet) doesn't
        reject the dict-to-TypedDict coercion that Monty handles at runtime.
        """
        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(lookup_person))
        assert isinstance(wrapper, CodeModeToolset)

        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        run_code = tools['run_code']

        # First call sets up variables -- type-checked but valid.
        await wrapper.call_tool('run_code', {'code': "addr = {'street': '1 Main St', 'city': 'NYC'}"}, ctx, run_code)
        # Second call uses them -- not type-checked (accumulated REPL state).
        code = "p = {'name': 'Alice', 'home': addr}\nprint(await lookup_person(person=p, count=3))"
        result = await wrapper.call_tool('run_code', {'code': code}, ctx, run_code)
        assert result.return_value == {'output': '3x Alice @ 1 Main St\n'}

    async def test_conflicting_typed_dicts_get_tool_name_prefix(self) -> None:
        """Two tools whose `$defs` collide on `Address` get tool-name prefixes in the prelude."""
        user_td = _make_address_tool_def('get_user', 'Get a user.', 'street')
        company_td = _make_address_tool_def('get_company', 'Get a company.', 'country')
        static = _StaticToolset(
            [user_td, company_td],
            results={'get_user': 'user-result', 'get_company': 'company-result'},
        )

        wrapper = CodeMode[object]().get_wrapper_toolset(static)
        assert isinstance(wrapper, CodeModeToolset)

        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        description = tools['run_code'].tool_def.description
        assert description is not None
        # Both conflicting `Address` types get tool-name prefixes.
        assert 'class get_user_Address(TypedDict):' in description
        assert 'class get_company_Address(TypedDict):' in description
        assert 'addr: get_user_Address' in description
        assert 'addr: get_company_Address' in description

        # End-to-end through Monty: both tools are callable from inside the sandbox.
        result = await wrapper.call_tool(
            'run_code',
            {
                'code': (
                    "u = await get_user(addr={'street': 'main'}, label='u')\n"
                    "c = await get_company(addr={'country': 'usa'}, label='c')\n"
                    'print(u, c)'
                ),
            },
            ctx,
            tools['run_code'],
        )
        assert result.return_value == {'output': 'user-result company-result\n'}

    # ---------------------------------------------------------------------------
    # Deferred tools
    # ---------------------------------------------------------------------------

    async def test_deferred_loading_tools_not_sandboxed(self) -> None:
        """Tools with `defer_loading=True` (Tool Search) stay native so the deferred-loading contract is honored."""

        def later(x: int) -> str:
            """A deferred-loading tool."""
            return str(x)  # pragma: no cover - tool body is not invoked in this test

        toolset = FunctionToolset[object](tools=[Tool(add), Tool(later, defer_loading=True)])
        wrapper = CodeMode[object]().get_wrapper_toolset(toolset)
        assert isinstance(wrapper, CodeModeToolset)

        ctx = build_run_context(None)
        tools = await wrapper.get_tools(ctx)

        description = tools['run_code'].tool_def.description
        assert description is not None
        # Non-deferred tools are sandboxed as usual.
        assert 'async def add' in description
        # The deferred-loading tool is NOT rendered into run_code's description...
        assert 'later' not in description
        # ...and stays exposed as a native tool with its `defer_loading` flag intact,
        # so `ToolSearchToolset` / `Model.prepare_request` can drive discovery.
        assert 'later' in tools
        assert tools['later'].tool_def.defer_loading is True

    async def test_deferred_loading_tool_sandboxed_once_discovered(self) -> None:
        """Once a deferred tool is discovered (`defer_loading=False`) it folds into `run_code`."""

        def later(x: int) -> str:
            """A discovered tool."""
            return str(x)  # pragma: no cover - tool body is not invoked in this test

        # `defer_loading=False` mimics the post-discovery state ToolSearchToolset hands back.
        toolset = FunctionToolset[object](tools=[Tool(add), Tool(later, defer_loading=False)])
        wrapper = CodeMode[object]().get_wrapper_toolset(toolset)
        assert isinstance(wrapper, CodeModeToolset)

        ctx = build_run_context(None)
        tools = await wrapper.get_tools(ctx)

        description = tools['run_code'].tool_def.description
        assert description is not None
        assert 'async def add' in description
        assert 'async def later' in description
        assert 'later' not in tools

    async def test_framework_tool_kind_tool_not_sandboxed(self) -> None:
        """Framework control tools with `tool_kind` stay native even when CodeMode wraps all user tools."""
        td_loader = ToolDefinition(
            name='load_capability',
            description='Load a deferred capability.',
            parameters_json_schema={
                'type': 'object',
                'properties': {'capability_id': {'type': 'string'}},
                'required': ['capability_id'],
            },
            return_schema={'type': 'string'},
            tool_kind='capability-load',
        )
        static = _StaticToolset([_make_address_tool_def('get_user', 'Get a user.', 'street'), td_loader])
        wrapper = CodeMode[object]().get_wrapper_toolset(static)
        assert isinstance(wrapper, CodeModeToolset)

        tools = await wrapper.get_tools(build_run_context(None))

        description = tools['run_code'].tool_def.description
        assert description is not None
        assert 'async def get_user' in description
        assert 'load_capability' not in description
        assert 'load_capability' in tools
        assert tools['load_capability'].tool_def.tool_kind == 'capability-load'

    async def test_code_execution_tool_not_sandboxed(self) -> None:
        """A tool that is itself a code sandbox (carries `code_arg_name` metadata) stays native.

        Folding one code-execution tool into `run_code` would make the model pass a script as a
        string argument to a function inside another script. Such a tool (e.g. DynamicWorkflow's
        `run_workflow`) is a peer of `run_code`, exposed alongside it, not inside it.
        """
        td_run_workflow = ToolDefinition(
            name='run_workflow',
            description='Run an orchestration script.',
            parameters_json_schema={'type': 'object', 'properties': {'code': {'type': 'string'}}, 'required': ['code']},
            return_schema={'type': 'string'},
            metadata={'code_arg_name': 'code', 'code_arg_language': 'python'},
        )
        static = _StaticToolset([_make_address_tool_def('get_user', 'Get a user.', 'street'), td_run_workflow])
        wrapper = CodeMode[object]().get_wrapper_toolset(static)
        assert isinstance(wrapper, CodeModeToolset)

        tools = await wrapper.get_tools(build_run_context(None))

        description = tools['run_code'].tool_def.description
        assert description is not None
        # Ordinary tools are still sandboxed...
        assert 'async def get_user' in description
        # ...but the code-execution tool stays native and is not folded into run_code.
        assert 'run_workflow' not in description
        assert 'run_workflow' in tools

    async def test_unless_native_tool_not_sandboxed(self) -> None:
        """Tools annotated with `unless_native` stay native so `Model.prepare_request` can filter them."""
        td_fallback = ToolDefinition(
            name='duckduckgo_search',
            description='DDG fallback.',
            parameters_json_schema={'type': 'object', 'properties': {'q': {'type': 'string'}}, 'required': ['q']},
            return_schema={'type': 'string'},
            unless_native='web_search',
        )
        static = _StaticToolset([_make_address_tool_def('get_user', 'Get a user.', 'street'), td_fallback])
        wrapper = CodeMode[object]().get_wrapper_toolset(static)
        assert isinstance(wrapper, CodeModeToolset)

        ctx = build_run_context(None)
        tools = await wrapper.get_tools(ctx)

        description = tools['run_code'].tool_def.description
        assert description is not None
        # Other tools are sandboxed as usual.
        assert 'async def get_user' in description
        # The unless_native tool's signature must NOT appear inside run_code's description.
        assert 'duckduckgo_search' not in description

    async def test_unless_native_tool_exposed_as_native(self) -> None:
        """`unless_native` tools remain in the toolset's native tools so `Model.prepare_request` can drop them when the provider supports the native tool."""
        td_fallback = ToolDefinition(
            name='duckduckgo_search',
            description='DDG fallback.',
            parameters_json_schema={'type': 'object', 'properties': {'q': {'type': 'string'}}, 'required': ['q']},
            return_schema={'type': 'string'},
            unless_native='web_search',
        )
        static = _StaticToolset([td_fallback])
        wrapper = CodeMode[object]().get_wrapper_toolset(static)
        assert isinstance(wrapper, CodeModeToolset)

        ctx = build_run_context(None)
        tools = await wrapper.get_tools(ctx)

        # The fallback tool is exposed as a native tool, with its unless_native annotation
        # preserved so Model.prepare_request can filter it when the native tool is supported.
        assert 'duckduckgo_search' in tools
        assert tools['duckduckgo_search'].tool_def.unless_native == 'web_search'

    async def test_no_unless_native_tool_is_sandboxed(self) -> None:
        """Tools without an `unless_native` annotation are sandboxed as usual (confirms the guard only diverts truthy values)."""
        td_plain = ToolDefinition(
            name='duckduckgo_search',
            description='DDG (no fallback annotation).',
            parameters_json_schema={'type': 'object', 'properties': {'q': {'type': 'string'}}, 'required': ['q']},
            return_schema={'type': 'string'},
        )
        static = _StaticToolset([td_plain])
        wrapper = CodeMode[object]().get_wrapper_toolset(static)
        assert isinstance(wrapper, CodeModeToolset)

        ctx = build_run_context(None)
        tools = await wrapper.get_tools(ctx)

        description = tools['run_code'].tool_def.description
        assert description is not None
        # Without unless_native, the tool is sandboxed normally.
        assert 'async def duckduckgo_search' in description
        assert 'duckduckgo_search' not in tools

    async def test_deferred_execution_tools_sandboxed(self) -> None:
        """Tools with `kind='external'`/`'unapproved'` are sandboxed like any other tool; resolution happens via a `HandleDeferredToolCalls` capability."""
        td_external = ToolDefinition(
            name='approve_action',
            description='Needs approval.',
            parameters_json_schema={'type': 'object', 'properties': {'x': {'type': 'string'}}, 'required': ['x']},
            return_schema={'type': 'string'},
            kind='external',
        )
        static = _StaticToolset([_make_address_tool_def('get_user', 'Get a user.', 'street'), td_external])
        wrapper = CodeMode[object]().get_wrapper_toolset(static)
        assert isinstance(wrapper, CodeModeToolset)

        ctx = build_run_context(None)
        tools = await wrapper.get_tools(ctx)

        description = tools['run_code'].tool_def.description
        assert description is not None
        # The external tool appears as a sandboxed function signature.
        assert 'async def approve_action' in description
        # Not exposed as a native tool.
        assert 'approve_action' not in tools

    async def test_tool_without_return_schema_warns(self) -> None:
        """A sandboxed tool with no return_schema triggers a one-time warning."""
        td = ToolDefinition(
            name='search',
            description='Search for things.',
            parameters_json_schema={'type': 'object', 'properties': {'q': {'type': 'string'}}, 'required': ['q']},
            # No return_schema -- simulates an MCP tool without outputSchema.
        )
        static = _StaticToolset([td], results={'search': 'found it'})
        wrapper = CodeMode[object]().get_wrapper_toolset(static)
        assert isinstance(wrapper, CodeModeToolset)

        ctx = build_run_context(None)
        with pytest.warns(UserWarning, match=r"tool 'search' has no return schema"):
            tools = await wrapper.get_tools(ctx)

        # Tool is still callable despite the warning.
        description = tools['run_code'].tool_def.description
        assert description is not None
        assert 'async def search' in description

        # Second call must not warn again.
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter('error')
            await wrapper.get_tools(ctx)

    async def test_tool_with_return_schema_does_not_warn(self) -> None:
        """A sandboxed tool WITH a return_schema does not trigger the warning."""
        import warnings as _warnings

        td = ToolDefinition(
            name='get_user',
            description='Get a user.',
            parameters_json_schema={'type': 'object', 'properties': {'id': {'type': 'integer'}}, 'required': ['id']},
            return_schema={'type': 'object', 'properties': {'name': {'type': 'string'}}},
        )
        static = _StaticToolset([td], results={'get_user': {'name': 'Alice'}})
        wrapper = CodeMode[object]().get_wrapper_toolset(static)
        assert isinstance(wrapper, CodeModeToolset)

        with _warnings.catch_warnings():
            _warnings.simplefilter('error')
            await wrapper.get_tools(build_run_context(None))

    # ---------------------------------------------------------------------------
    # Agent.run end-to-end (with FunctionModel hand-driving the model output)
    # ---------------------------------------------------------------------------

    async def test_code_mode_via_agent_run_executes_run_code_and_returns_result(self) -> None:
        """End-to-end through `Agent.run`: a `FunctionModel` issues a `run_code` call, the
        sandbox dispatches to a wrapped tool, and the second model turn observes the
        tool's return value before producing the final text output.
        """
        from pydantic_ai.messages import (
            ModelMessage,
            ModelRequest,
            ModelResponse,
            TextPart,
            ToolCallPart,
            ToolReturnPart,
        )
        from pydantic_ai.models.function import AgentInfo, FunctionModel

        observed_tool_calls: list[str] = []
        observed_tool_returns: list[Any] = []
        seen_tool_definitions: list[list[str]] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            # Snapshot what tool definitions the model is being shown each turn --
            # if `CodeMode` is wired correctly the model only ever sees `run_code`.
            seen_tool_definitions.append([td.name for td in info.function_tools])

            # First turn: issue a `run_code` call that calls the wrapped `add` tool
            # through the sandbox.
            if not observed_tool_calls:
                code = 'result = await add(a=4, b=6)\nprint(f"add returned {result}")\nresult'
                observed_tool_calls.append(code)
                return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': code})])

            # Second turn: pull the `run_code` return value out of the most recent
            # ModelRequest (which is the one Pydantic AI just appended after dispatch).
            last_request = messages[-1]
            assert isinstance(last_request, ModelRequest)
            run_code_return = next(
                p for p in last_request.parts if isinstance(p, ToolReturnPart) and p.tool_name == 'run_code'
            )
            observed_tool_returns.append(run_code_return.content)
            return ModelResponse(parts=[TextPart(f'sum is {observed_tool_returns[-1]["result"]}')])

        agent: Agent[object, str] = Agent(FunctionModel(model_fn), capabilities=[CodeMode[object]()])

        @agent.tool_plain
        def add(a: int, b: int) -> int:  # pyright: ignore[reportUnusedFunction]
            """Add two numbers."""
            return a + b

        result = await agent.run('please add 4 and 6')

        # The model was shown only `run_code` -- the wrapped `add` tool is hidden behind it.
        assert seen_tool_definitions[0] == ['run_code']
        assert seen_tool_definitions[1] == ['run_code']

        # The first turn issued exactly the code we expected and the sandbox returned
        # both the printed output and the value of the trailing expression.
        assert len(observed_tool_calls) == 1
        assert len(observed_tool_returns) == 1
        assert observed_tool_returns[0] == {'output': 'add returned 10\n', 'result': 10}

        # The agent's final output reflects the value flowing through the sandbox.
        assert result.output == 'sum is 10'

    async def test_deferred_capability_loader_stays_native_with_tools_all(self) -> None:
        """Regression for the deferred-capability bootstrap (issue #276).

        With `CodeMode(tools='all')` and a deferred capability configured, the
        framework-managed `load_capability` tool must reach the model as a native call
        (alongside `run_code`) so the model can reveal the capability. The deferred
        member tool stays hidden -- it is neither folded into `run_code` nor surfaced as
        a plain tool until loaded.

        (The native-vs-sandbox split per tool kind is covered directly at the toolset
        level by `test_framework_tool_kind_tool_not_sandboxed` and
        `test_tool_search_toolset_deferred_tool_not_in_run_code`; this exercises the
        end-to-end path through `Agent`.)
        """
        from pydantic_ai.capabilities import Capability

        capability = Capability[object](
            id='demo',
            description='Demo deferred capability.',
            instructions='Use demo_tool.',
            defer_loading=True,
        )

        @capability.tool_plain
        def demo_tool() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'ok'  # pragma: no cover - deferred tool stays hidden, body is not invoked

        model = TestModel(call_tools=[])
        agent: Agent[object, str] = Agent(
            model,
            capabilities=[capability, CodeMode[object](tools='all')],
        )
        await agent.run('inspect tools')

        assert model.last_model_request_parameters is not None
        by_name = {td.name: td for td in model.last_model_request_parameters.function_tools}

        # The bootstrap tool is a native call alongside `run_code`, not buried in the sandbox.
        assert 'load_capability' in by_name
        assert 'run_code' in by_name

        # The deferred member tool stays hidden until loaded: not folded into `run_code`
        # and not surfaced as a plain native tool.
        assert 'demo_tool' not in by_name
        run_code_desc = by_name['run_code'].description or ''
        assert 'demo_tool' not in run_code_desc
        assert 'load_capability' not in run_code_desc

    # ---------------------------------------------------------------------------
    # Capability registration
    # ---------------------------------------------------------------------------

    async def test_code_mode_can_be_registered_as_agent_capability(self) -> None:
        """`CodeMode` can be passed via `Agent(capabilities=[...])` without raising."""
        Agent(TestModel(), capabilities=[CodeMode[object]()])

    # ---------------------------------------------------------------------------
    # Tool name sanitization
    # ---------------------------------------------------------------------------

    @pytest.mark.parametrize(
        'original, expected',
        [
            ('get_weather', 'get_weather'),  # already valid -- no change
            ('get-weather', 'get_weather'),  # hyphen → underscore
            ('api.call', 'api_call'),  # dot → underscore
            ('api.call-now', 'api_call_now'),  # mixed
            ('123tool', '_123tool'),  # leading digit → prepend underscore
            ('a', 'a'),  # single char
            ('-', '_'),  # single invalid char
            ('for', 'for_'),  # Python keyword → append underscore
            ('import', 'import_'),  # Python keyword
        ],
    )
    def test_sanitize_tool_name(self, original: str, expected: str) -> None:
        assert _sanitize_tool_name(original) == expected

    async def test_hyphenated_tool_name_is_sanitized_and_callable(self) -> None:
        """A tool with hyphens in the name is automatically renamed and callable from the sandbox."""
        td = ToolDefinition(
            name='get-weather',
            description='Get the weather.',
            parameters_json_schema={
                'type': 'object',
                'properties': {'city': {'type': 'string'}},
                'required': ['city'],
            },
            return_schema={'type': 'string'},
        )
        static = _StaticToolset([td], results={'get-weather': 'sunny'})
        wrapper = CodeMode[object]().get_wrapper_toolset(static)
        assert isinstance(wrapper, CodeModeToolset)

        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        description = tools['run_code'].tool_def.description
        assert description is not None
        # The sanitized name appears in the description, not the original.
        assert 'get_weather' in description
        assert 'get-weather' not in description

        # End-to-end: the model writes `await get_weather(...)` and the call
        # dispatches to the original `get-weather` tool in the wrapped toolset.
        result = await wrapper.call_tool(
            'run_code',
            {'code': "print(await get_weather(city='NYC'))"},
            ctx,
            tools['run_code'],
        )
        assert result.return_value == {'output': 'sunny\n'}

    async def test_dotted_tool_name_is_sanitized_and_callable(self) -> None:
        """A tool with dots in the name is automatically renamed and callable."""
        td = ToolDefinition(
            name='api.lookup',
            description='Look up an API.',
            parameters_json_schema={
                'type': 'object',
                'properties': {'key': {'type': 'string'}},
                'required': ['key'],
            },
            return_schema={'type': 'string'},
        )
        static = _StaticToolset([td], results={'api.lookup': 'found'})
        wrapper = CodeMode[object]().get_wrapper_toolset(static)
        assert isinstance(wrapper, CodeModeToolset)

        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        result = await wrapper.call_tool(
            'run_code',
            {'code': "print(await api_lookup(key='x'))"},
            ctx,
            tools['run_code'],
        )
        assert result.return_value == {'output': 'found\n'}

    async def test_sanitized_name_collision_warns_and_drops_second(self) -> None:
        """When two tool names sanitize to the same identifier, the second is dropped with a warning."""
        td1 = ToolDefinition(
            name='get-weather',
            description='Get weather (hyphens).',
            parameters_json_schema={'type': 'object', 'properties': {'x': {'type': 'string'}}, 'required': ['x']},
            return_schema={'type': 'string'},
        )
        td2 = ToolDefinition(
            name='get.weather',
            description='Get weather (dots).',
            parameters_json_schema={'type': 'object', 'properties': {'x': {'type': 'string'}}, 'required': ['x']},
            return_schema={'type': 'string'},
        )
        static = _StaticToolset([td1, td2], results={'get-weather': 'rain'})
        wrapper = CodeMode[object]().get_wrapper_toolset(static)
        assert isinstance(wrapper, CodeModeToolset)

        ctx = build_run_context(None)
        with pytest.warns(UserWarning, match=r"tool 'get\.weather'.*collides with 'get-weather'"):
            tools = await wrapper.get_tools(ctx)

        description = tools['run_code'].tool_def.description
        assert description is not None
        # Only the first tool survives.
        assert description.count('get_weather') >= 1
        assert 'Get weather (dots)' not in description

    async def test_sanitized_name_collision_with_native_tool(self) -> None:
        """A sanitized name that collides with a native (already valid) tool is dropped."""
        td_native = ToolDefinition(
            name='get_weather',
            description='Native tool.',
            parameters_json_schema={'type': 'object', 'properties': {'x': {'type': 'string'}}, 'required': ['x']},
            return_schema={'type': 'string'},
        )
        td_hyphen = ToolDefinition(
            name='get-weather',
            description='Hyphenated tool.',
            parameters_json_schema={'type': 'object', 'properties': {'x': {'type': 'string'}}, 'required': ['x']},
            return_schema={'type': 'string'},
        )
        static = _StaticToolset([td_native, td_hyphen], results={'get_weather': 'ok'})
        wrapper = CodeMode[object]().get_wrapper_toolset(static)
        assert isinstance(wrapper, CodeModeToolset)

        ctx = build_run_context(None)
        with pytest.warns(UserWarning, match=r"tool 'get-weather'.*collides with 'get_weather'"):
            tools = await wrapper.get_tools(ctx)

        description = tools['run_code'].tool_def.description
        assert description is not None
        assert 'Native tool' in description
        assert 'Hyphenated tool' not in description

    # ---------------------------------------------------------------------------
    # Logfire metadata
    # ---------------------------------------------------------------------------

    async def test_run_code_tool_has_code_metadata(self) -> None:
        """The `run_code` ToolDefinition carries metadata for Logfire code rendering."""
        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)

        tools = await wrapper.get_tools(build_run_context(None))
        metadata = tools['run_code'].tool_def.metadata
        assert metadata is not None
        assert metadata['code_arg_name'] == 'code'
        assert metadata['code_arg_language'] == 'python'

    async def test_tool_returning_tool_return_is_unwrapped(self) -> None:
        """A wrapped tool that returns a `ToolReturn` has its value unwrapped for the sandbox."""
        from pydantic_ai.messages import ToolReturn as ToolReturnMsg

        def fancy() -> Any:
            """Return a ToolReturn with metadata."""
            return ToolReturnMsg(return_value=42, metadata={'source': 'test'})

        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(fancy))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)

        result = await wrapper.call_tool('run_code', {'code': 'await fancy()'}, ctx, tools['run_code'])
        # The sandbox receives the unwrapped value (42), not the ToolReturn wrapper.
        # No print output → result returned directly.
        assert result.return_value == 42

        # The nested ToolReturnPart carries the ToolReturn metadata.
        returns = result.metadata['tool_returns']
        assert returns['pyd_ai_code_mode__1'].metadata == {'source': 'test'}

    async def test_approval_required_surfaces_as_model_retry(self) -> None:
        """Tools that raise ApprovalRequired inside the sandbox surface as ModelRetry."""
        from pydantic_ai.exceptions import ApprovalRequired as _ApprovalRequired

        def needs_approval() -> str:
            """A tool that requires approval."""
            raise _ApprovalRequired()

        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(needs_approval))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)

        with pytest.raises(ModelRetry, match='no `HandleDeferredToolCalls` capability resolved it'):
            await wrapper.call_tool('run_code', {'code': 'await needs_approval()'}, ctx, tools['run_code'])

    async def test_handler_denial_surfaces_as_model_retry(self) -> None:
        """A `HandleDeferredToolCalls` handler denying a sandboxed tool call surfaces the denial.

        The denial raises `RuntimeError` inside the sandbox so the script can't mistake
        the denial message for a regular string return. If the script doesn't catch it,
        Monty re-raises as `MontyRuntimeError`, which the harness converts to `ModelRetry`
        with the original denial message preserved in the trace.
        """
        try:
            from pydantic_ai.capabilities import HandleDeferredToolCalls
        except ImportError:  # pragma: no cover -- only fires on floor-slim CI, which doesn't gate on coverage
            pytest.skip('Requires pydantic-ai-slim with `HandleDeferredToolCalls` (next release after 1.86.1)')

        from pydantic_ai.exceptions import ApprovalRequired as _ApprovalRequired
        from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDenied

        def needs_approval() -> str:
            """A tool that requires approval."""
            raise _ApprovalRequired()

        async def handler(ctx: RunContext[object], requests: DeferredToolRequests) -> DeferredToolResults:
            return DeferredToolResults(
                approvals={call.tool_call_id: ToolDenied(message='nope') for call in requests.approvals}
            )

        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(needs_approval))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper, root_capability=HandleDeferredToolCalls(handler=handler))
        tools = await wrapper.get_tools(ctx)

        with pytest.raises(ModelRetry, match=r'call denied: nope'):
            await wrapper.call_tool('run_code', {'code': 'await needs_approval()'}, ctx, tools['run_code'])

    async def test_approved_tool_re_raising_approval_required_surfaces_as_model_retry(self) -> None:
        """If the approved tool body re-raises `ApprovalRequired`, pydantic-ai propagates it
        without re-invoking the handler; the harness then converts it to a `ModelRetry`.

        This guards the contract documented on `_resolve_single_deferred.Raises`: a re-raised
        deferral after approval is *not* re-resolved -- it bubbles up to the caller.
        """
        try:
            from pydantic_ai.capabilities import HandleDeferredToolCalls
        except ImportError:  # pragma: no cover -- only fires on floor-slim CI, which doesn't gate on coverage
            pytest.skip('Requires pydantic-ai-slim with `HandleDeferredToolCalls` (next release after 1.86.1)')

        from pydantic_ai.exceptions import ApprovalRequired as _ApprovalRequired
        from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved

        def always_needs_approval(ctx: RunContext[object]) -> str:
            """Raises `ApprovalRequired` every time, even after being approved."""
            raise _ApprovalRequired()

        async def handler(ctx: RunContext[object], requests: DeferredToolRequests) -> DeferredToolResults:
            return DeferredToolResults(approvals={call.tool_call_id: ToolApproved() for call in requests.approvals})

        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(always_needs_approval))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper, root_capability=HandleDeferredToolCalls(handler=handler))
        tools = await wrapper.get_tools(ctx)

        with pytest.raises(ModelRetry, match='no `HandleDeferredToolCalls` capability resolved it'):
            await wrapper.call_tool('run_code', {'code': 'await always_needs_approval()'}, ctx, tools['run_code'])

    async def test_model_retry_from_wrapped_tool_surfaces_as_model_retry(self) -> None:
        """A wrapped tool that raises ModelRetry gets double-wrapped through Monty but still retries.

        The flow is: ModelRetry → Monty catches as RuntimeError → MontyRuntimeError → ModelRetry.
        The original error message is preserved in the display string.
        """

        def flaky() -> str:
            """A tool that always retries."""
            raise ModelRetry('try again please')

        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(flaky))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)

        with pytest.raises(ModelRetry, match='try again please'):
            await wrapper.call_tool('run_code', {'code': 'await flaky()'}, ctx, tools['run_code'])

    async def test_invalid_tool_args_surface_as_model_retry(self) -> None:
        """Wrong argument types passed to a sandboxed tool surface as ModelRetry.

        On a fresh REPL, the static type checker catches this before execution.
        """
        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)

        # Pass a string where int is expected -- type checker catches this.
        with pytest.raises(ModelRetry, match='error in code'):
            await wrapper.call_tool(
                'run_code',
                {'code': "await add(a='not_a_number', b=3)"},
                ctx,
                tools['run_code'],
            )

    # ---------------------------------------------------------------------------
    # Multimodal tool returns
    # ---------------------------------------------------------------------------

    async def test_tool_returning_binary_image_is_returned_directly(self) -> None:
        """A tool that returns BinaryContent passes through the sandbox and is
        returned as native multimodal content (not wrapped in a dict)."""
        from pydantic_ai.messages import BinaryContent

        image_bytes = b'\x89PNG\r\n\x1a\n fake image data'

        def gen_image() -> Any:
            """Generate an image."""
            return BinaryContent(data=image_bytes, media_type='image/png')

        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(gen_image))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)

        result = await wrapper.call_tool('run_code', {'code': 'await gen_image()'}, ctx, tools['run_code'])
        # No print → multimodal content returned directly for native model delivery.
        rv = result.return_value
        assert isinstance(rv, BinaryContent)
        assert rv.data == image_bytes
        assert rv.media_type == 'image/png'

    async def test_tool_returning_binary_image_with_print_uses_list_format(self) -> None:
        """When print output accompanies a multimodal return, the result is a list
        so _split_content can extract the image for native delivery."""
        from pydantic_ai.messages import BinaryContent

        image_bytes = b'\x89PNG fake'

        def gen_image() -> Any:
            """Generate an image."""
            return BinaryContent(data=image_bytes, media_type='image/png')

        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(gen_image))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)

        result = await wrapper.call_tool(
            'run_code',
            {'code': 'img = await gen_image()\nprint("generated")\nimg'},
            ctx,
            tools['run_code'],
        )
        # Print + multimodal → list format.
        rv = result.return_value
        assert isinstance(rv, list)
        assert rv[0] == 'generated\n'
        assert isinstance(rv[1], BinaryContent)
        assert rv[1].data == image_bytes

    async def test_tool_returning_list_with_binary_image_and_print(self) -> None:
        """A list result containing multimodal items with print output gets flattened
        so _split_content can find each multimodal item at the top level."""
        from pydantic_ai.messages import BinaryContent

        image_bytes = b'\x89PNG list'

        def gen_images() -> Any:
            """Generate a list with an image."""
            return [BinaryContent(data=image_bytes, media_type='image/png'), 'caption']

        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(gen_images))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)

        result = await wrapper.call_tool(
            'run_code',
            {'code': 'imgs = await gen_images()\nprint("done")\nimgs'},
            ctx,
            tools['run_code'],
        )
        # Print + list-with-multimodal → flattened list.
        rv = result.return_value
        assert isinstance(rv, list)
        assert rv[0] == 'done\n'
        assert isinstance(rv[1], BinaryContent)
        assert rv[1].data == image_bytes
        assert rv[2] == 'caption'

    async def test_tool_returning_tool_return_with_binary_content(self) -> None:
        """A tool that wraps a BinaryContent in a ToolReturn has the image properly unwrapped
        and returned as native multimodal content."""
        from pydantic_ai.messages import BinaryContent
        from pydantic_ai.messages import ToolReturn as ToolReturnMsg

        image_bytes = b'\x89PNG wrapped'

        def gen_image() -> Any:
            """Generate an image wrapped in ToolReturn."""
            return ToolReturnMsg(
                return_value=BinaryContent(data=image_bytes, media_type='image/png'), metadata={'src': 'test'}
            )

        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(gen_image))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)

        result = await wrapper.call_tool('run_code', {'code': 'await gen_image()'}, ctx, tools['run_code'])
        rv = result.return_value
        assert isinstance(rv, BinaryContent)
        assert rv.data == image_bytes
        # ToolReturn metadata is preserved on the nested return part.
        returns = result.metadata['tool_returns']
        assert returns['pyd_ai_code_mode__1'].metadata == {'src': 'test'}

    # ---------------------------------------------------------------------------
    # OTel / Logfire instrumentation
    # ---------------------------------------------------------------------------

    @pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
    async def test_sandboxed_tool_calls_produce_otel_spans(self, capfire: CaptureLogfire) -> None:
        """Sandboxed tool calls dispatched through ToolManager produce OTel execute_tool spans."""
        from pydantic_ai.capabilities import Instrumentation
        from pydantic_ai.messages import (
            ModelMessage,
            ModelResponse,
            TextPart,
            ToolCallPart,
        )
        from pydantic_ai.models.function import AgentInfo, FunctionModel
        from pydantic_ai.models.instrumented import InstrumentationSettings

        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': 'await add(a=1, b=2)'})])
            return ModelResponse(parts=[TextPart('done')])

        agent: Agent[object, str] = Agent(
            FunctionModel(model_fn),
            capabilities=[CodeMode[object](), Instrumentation(settings=InstrumentationSettings(include_content=True))],
        )

        @agent.tool_plain
        def add(a: int, b: int) -> int:  # pyright: ignore[reportUnusedFunction]
            """Add two numbers."""
            return a + b

        result = await agent.run('test')
        assert result.output == 'done'

        spans = capfire.exporter.exported_spans_as_dict()
        tool_spans = [s for s in spans if s['attributes'].get('gen_ai.tool.name')]
        tool_names = [s['attributes']['gen_ai.tool.name'] for s in tool_spans]

        # The outer `run_code` tool call should produce a span.
        assert 'run_code' in tool_names, f'No run_code span found in {tool_names}'

        # The inner `add` tool call (dispatched through ToolManager) should also produce a span.
        assert 'add' in tool_names, f'No add span found in {tool_names}'

        # Verify the inner tool span has the expected OTel attributes.
        add_span = next(s for s in tool_spans if s['attributes']['gen_ai.tool.name'] == 'add')
        assert add_span['attributes']['gen_ai.tool.name'] == 'add'
        assert 'gen_ai.tool.call.id' in add_span['attributes']

    # ---------------------------------------------------------------------------
    # Error handling improvements
    # ---------------------------------------------------------------------------

    async def test_unknown_function_call_surfaces_as_model_retry(self) -> None:
        """Calling an undefined function from sandbox code surfaces as ModelRetry.

        On a fresh REPL, the type checker catches this; on subsequent calls,
        it becomes a runtime NameError.
        """
        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        run_code = tools['run_code']

        # First call (fresh REPL): type checker catches undefined name.
        with pytest.raises(ModelRetry, match='error in code'):
            await wrapper.call_tool('run_code', {'code': 'await nonexistent_tool(x=1)'}, ctx, run_code)

        # After a successful call, type checking is skipped -- falls to runtime NameError.
        await wrapper.call_tool('run_code', {'code': '1 + 1', 'restart': True}, ctx, run_code)
        with pytest.raises(ModelRetry, match='Runtime error'):
            await wrapper.call_tool('run_code', {'code': 'await nonexistent_tool(x=1)'}, ctx, run_code)

    async def test_positional_args_rejected(self) -> None:
        """Calling a tool with positional args surfaces as ModelRetry.

        On a fresh REPL the type checker catches it; on subsequent calls
        the runtime positional-args guard catches it.
        """
        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        run_code = tools['run_code']

        # Fresh REPL: type checker catches positional args.
        with pytest.raises(ModelRetry, match='error in code'):
            await wrapper.call_tool('run_code', {'code': 'await add(1, 2)'}, ctx, run_code)

        # After a valid call, type checking is skipped -- runtime guard catches it.
        await wrapper.call_tool('run_code', {'code': '1 + 1', 'restart': True}, ctx, run_code)
        with pytest.raises(ModelRetry, match='does not accept positional arguments'):
            await wrapper.call_tool('run_code', {'code': 'await add(1, 2)'}, ctx, run_code)

        # Caught positional args -- sandbox code handles the error gracefully.
        result = await wrapper.call_tool(
            'run_code',
            {'code': 'try:\n    await add(1, 2)\nexcept TypeError:\n    pass\n"recovered"'},
            ctx,
            run_code,
        )
        assert result.return_value == 'recovered'

    async def test_print_output_preserved_in_runtime_error(self) -> None:
        """When sandbox code prints before crashing, the print output is included
        in the ModelRetry error message so the model can use it for debugging."""
        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)

        with pytest.raises(ModelRetry, match=r'Runtime error') as exc_info:
            await wrapper.call_tool(
                'run_code',
                {'code': 'print("debug info")\n1 / 0'},
                ctx,
                tools['run_code'],
            )
        msg = str(exc_info.value)
        assert 'debug info' in msg
        assert '[stdout before error]' in msg

    async def test_duplicate_future_in_gather_is_retryable(self) -> None:
        # Awaiting the same tool call twice in one gather makes the Monty VM panic; that panic
        # must surface as a retry (with the corrupt REPL dropped), not tear down the agent run.
        wrapper = CodeMode[None]().get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        code = 'import asyncio\nf = add(a=1, b=2)\nawait asyncio.gather(f, f)'
        with pytest.raises(ModelRetry, match='aborted inside the sandbox'):
            await wrapper.call_tool('run_code', {'code': code}, ctx, tools['run_code'])
        assert wrapper._repl is None  # pyright: ignore[reportPrivateUsage]

    async def test_non_panic_base_exception_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The panic guard catches BaseException but must re-raise anything that is not a VM panic.
        class _Boom(BaseException):
            pass

        async def _boom(self: Any, state: Any) -> Any:
            raise _Boom('boom')

        monkeypatch.setattr('pydantic_ai_harness._monty_exec.MontyExecutor.run', _boom)
        wrapper = CodeMode[None]().get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        with pytest.raises(_Boom):
            await wrapper.call_tool('run_code', {'code': 'await add(a=1, b=2)'}, ctx, tools['run_code'])

    # ---------------------------------------------------------------------------
    # Sequential tool resolution
    # ---------------------------------------------------------------------------

    async def test_sequential_tool_rendered_as_sync_and_resolved_inline(self) -> None:
        """A tool with `sequential=True` is rendered as `def` (sync) and
        resolved inline at FunctionSnapshot via `resume({'return_value': ...})`."""
        from dataclasses import replace as dc_replace

        class _SeqToolset(AbstractToolset[object]):
            """Marks add as sequential; greet stays parallel."""

            def __init__(self) -> None:
                self._inner = _build_function_toolset(add, greet)

            @property
            def id(self) -> str | None:
                return None  # pragma: no cover

            async def get_tools(self, ctx: RunContext[object]) -> dict[str, ToolsetTool[object]]:
                tools = await self._inner.get_tools(ctx)
                return {
                    n: dc_replace(t, tool_def=dc_replace(t.tool_def, sequential=True)) if n == 'add' else t
                    for n, t in tools.items()
                }

            async def call_tool(
                self, name: str, tool_args: dict[str, Any], ctx: RunContext[object], tool: ToolsetTool[object]
            ) -> Any:
                return await self._inner.call_tool(name, tool_args, ctx, tool)

        seq_wrapper = CodeModeToolset[object](wrapped=_SeqToolset(), tool_selector='all')
        ctx = await build_ctx(None, seq_wrapper)
        tools = await seq_wrapper.get_tools(ctx)
        run_code = tools['run_code']

        # Sequential tool rendered as `def`, parallel tool as `async def`.
        desc = run_code.tool_def.description or ''
        assert 'def add(' in desc
        assert 'async def add(' not in desc
        assert 'async def greet(' in desc

        # Sequential tool called without `await`, parallel with `await`.
        result = await seq_wrapper.call_tool(
            'run_code',
            {
                'code': 'result_add = add(a=1, b=2)\nresult_greet = await greet(name="World")\n[result_add, result_greet]'
            },
            ctx,
            run_code,
        )
        assert result.return_value == [3, 'Hello, World!']

        # Metadata records both sequential and parallel tool calls.
        assert result.metadata['code_mode'] is True
        calls = result.metadata['tool_calls']
        returns = result.metadata['tool_returns']
        assert len(calls) == 2
        assert len(returns) == 2
        call_names = {c.tool_name for c in calls.values()}
        assert call_names == {'add', 'greet'}
        for tc_id, call in calls.items():
            assert tc_id in returns
            assert returns[tc_id].tool_name == call.tool_name

    async def test_sequential_tool_barrier_awaits_pending_parallel_tasks(self) -> None:
        """When a sequential tool is called while parallel tasks are pending,
        the pending tasks are awaited first (barrier) before dispatching."""
        from dataclasses import replace as dc_replace

        class _SeqToolset(AbstractToolset[object]):
            def __init__(self) -> None:
                self._inner = _build_function_toolset(add, greet)

            @property
            def id(self) -> str | None:
                return None  # pragma: no cover

            async def get_tools(self, ctx: RunContext[object]) -> dict[str, ToolsetTool[object]]:
                tools = await self._inner.get_tools(ctx)
                return {
                    n: dc_replace(t, tool_def=dc_replace(t.tool_def, sequential=True)) if n == 'add' else t
                    for n, t in tools.items()
                }

            async def call_tool(
                self, name: str, tool_args: dict[str, Any], ctx: RunContext[object], tool: ToolsetTool[object]
            ) -> Any:
                return await self._inner.call_tool(name, tool_args, ctx, tool)

        seq_wrapper = CodeModeToolset[object](wrapped=_SeqToolset(), tool_selector='all')
        ctx = await build_ctx(None, seq_wrapper)
        tools = await seq_wrapper.get_tools(ctx)

        # Start a parallel call (greet, async def), then call a sequential tool (add, def).
        # The barrier should await greet before dispatching add.
        result = await seq_wrapper.call_tool(
            'run_code',
            {
                'code': (
                    'future_greet = greet(name="World")\n'
                    'result_add = add(a=1, b=2)\n'
                    'result_greet = await future_greet\n'
                    '[result_add, result_greet]'
                )
            },
            ctx,
            tools['run_code'],
        )
        assert result.return_value == [3, 'Hello, World!']

        # Both calls recorded in metadata -- greet resolved at barrier, add resolved inline.
        assert result.metadata['code_mode'] is True
        calls = result.metadata['tool_calls']
        returns = result.metadata['tool_returns']
        assert len(calls) == 2
        assert len(returns) == 2
        # greet was dispatched first (parallel), add second (sequential barrier).
        call_list = list(calls.values())
        assert call_list[0].tool_name == 'greet'
        assert call_list[1].tool_name == 'add'
        for tc_id in calls:
            assert returns[tc_id].content in (3, 'Hello, World!')

    async def test_sequential_tool_error_surfaces_as_model_retry(self) -> None:
        """An error from a sequential tool (resolved inline) surfaces as ModelRetry."""
        from dataclasses import replace as dc_replace

        class _SeqToolset(AbstractToolset[object]):
            def __init__(self) -> None:
                self._inner = _build_function_toolset(add)

            @property
            def id(self) -> str | None:
                return None  # pragma: no cover

            async def get_tools(self, ctx: RunContext[object]) -> dict[str, ToolsetTool[object]]:
                tools = await self._inner.get_tools(ctx)
                return {n: dc_replace(t, tool_def=dc_replace(t.tool_def, sequential=True)) for n, t in tools.items()}

            async def call_tool(
                self, name: str, tool_args: dict[str, Any], ctx: RunContext[object], tool: ToolsetTool[object]
            ) -> Any:
                return await self._inner.call_tool(name, tool_args, ctx, tool)

        seq_wrapper = CodeModeToolset[object](wrapped=_SeqToolset(), tool_selector='all')
        ctx = await build_ctx(None, seq_wrapper)
        tools = await seq_wrapper.get_tools(ctx)
        run_code = tools['run_code']
        # Make a successful call so the REPL is no longer fresh (type checking skipped).
        await seq_wrapper.call_tool('run_code', {'code': 'add(a=1, b=2)'}, ctx, run_code)
        # Now bad args go through the runtime path in sequential resolution.
        with pytest.raises(ModelRetry, match='Runtime error'):
            await seq_wrapper.call_tool('run_code', {'code': "add(a='bad', b=3)"}, ctx, run_code)

    async def test_global_sequential_mode_forces_sequential_resolution(self) -> None:
        """When the parallel execution mode is `sequential`, tool calls inside the
        sandbox are resolved sequentially via FutureSnapshot. Signatures stay `async def`."""
        from pydantic_ai.tool_manager import ToolManager

        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)

        with ToolManager.parallel_execution_mode('sequential'):
            tools = await wrapper.get_tools(ctx)
            run_code = tools['run_code']

            # All tools are still rendered as `async def` (global mode doesn't affect rendering).
            desc = run_code.tool_def.description
            assert desc is not None
            assert 'async def add(' in desc

            result = await wrapper.call_tool(
                'run_code',
                {'code': 'await add(a=10, b=20)'},
                ctx,
                run_code,
            )
            assert result.return_value == 30

    async def test_global_sequential_overrides_per_tool_sequential(self) -> None:
        """When global sequential mode is active AND a tool has `sequential=True`,
        the tool is deferred (not resolved inline) and handled via FutureSnapshot."""
        from dataclasses import replace as dc_replace

        from pydantic_ai.tool_manager import ToolManager

        class _SeqToolset(AbstractToolset[object]):
            def __init__(self) -> None:
                self._inner = _build_function_toolset(add)

            @property
            def id(self) -> str | None:
                return None  # pragma: no cover

            async def get_tools(self, ctx: RunContext[object]) -> dict[str, ToolsetTool[object]]:
                tools = await self._inner.get_tools(ctx)
                return {n: dc_replace(t, tool_def=dc_replace(t.tool_def, sequential=True)) for n, t in tools.items()}

            async def call_tool(
                self, name: str, tool_args: dict[str, Any], ctx: RunContext[object], tool: ToolsetTool[object]
            ) -> Any:
                return await self._inner.call_tool(name, tool_args, ctx, tool)

        seq_wrapper = CodeModeToolset[object](wrapped=_SeqToolset(), tool_selector='all')
        ctx = await build_ctx(None, seq_wrapper)

        with ToolManager.parallel_execution_mode('sequential'):
            tools = await seq_wrapper.get_tools(ctx)
            run_code = tools['run_code']

            # Per-tool sequential renders as `def`, but global mode uses deferred path.
            desc = run_code.tool_def.description or ''
            assert 'def add(' in desc
            assert 'async def add(' not in desc

            # The tool still works -- global sequential resolves at FutureSnapshot.
            result = await seq_wrapper.call_tool('run_code', {'code': 'add(a=5, b=7)'}, ctx, run_code)
            assert result.return_value == 12

    async def test_restart_with_invalid_code_clears_repl_for_retry(self) -> None:
        """When `restart=True` and type checking fails, the REPL is cleared so
        the next retry still gets type-checked on a fresh REPL."""
        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        run_code = tools['run_code']

        # First call succeeds -- REPL has state.
        await wrapper.call_tool('run_code', {'code': 'x = await add(a=1, b=2)'}, ctx, run_code)

        # Restart with bad code -- type checking catches it.
        with pytest.raises(ModelRetry, match='Type error'):
            await wrapper.call_tool('run_code', {'code': "await add(a='bad', b=3)", 'restart': True}, ctx, run_code)

        # Retry without restart -- should still be type-checked (REPL was cleared).
        with pytest.raises(ModelRetry, match='Type error'):
            await wrapper.call_tool('run_code', {'code': "await add(a='bad', b=3)"}, ctx, run_code)

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def test_print_capture_concatenates_chunks_in_order(self) -> None:
        """`PrintCapture` accumulates print-callback chunks and joins them on read.

        Lives in the production module rather than as a closure inside `call_tool` so
        coverage.py sees it execute even when Monty's Rust-side worker thread bypasses
        the per-thread tracer hooks. This unit test exercises it directly.
        """
        capture = PrintCapture()
        assert capture.joined == ''
        capture('stdout', 'hello')
        capture('stdout', ' ')
        capture('stdout', 'world\n')
        assert capture.joined == 'hello world\n'


class TestToolSearchIntegration:
    """Tests for CodeMode + ToolSearch (search_tools) interaction."""

    async def test_search_tool_stays_native(self) -> None:
        """search_tools is kept as a native tool even with tools='all'."""
        from pydantic_ai.toolsets._tool_search import _SEARCH_TOOLS_NAME
        from pydantic_ai.toolsets.combined import CombinedToolset

        search_toolset = _StaticToolset([_search_tool_def()])
        func_toolset = _build_function_toolset(add)
        combined = CombinedToolset([search_toolset, func_toolset])
        code_mode = CodeModeToolset(wrapped=combined, tool_selector='all')
        ctx = build_run_context(None)
        tools = await code_mode.get_tools(ctx)

        # search_tools should be native (not sandboxed inside run_code)
        assert _SEARCH_TOOLS_NAME in tools
        assert tools[_SEARCH_TOOLS_NAME].tool_def.name == _SEARCH_TOOLS_NAME
        # run_code should also be present with the sandboxed 'add' function
        assert 'run_code' in tools
        # add should be sandboxed (not a separate native tool)
        assert 'add' not in tools

    async def test_search_tools_description_appended(self) -> None:
        """search_tools description gets a modifier appended about run_code functions."""
        from pydantic_ai.toolsets._tool_search import _SEARCH_TOOLS_NAME

        original_desc = 'There are additional tools. Search here.'
        toolset = _StaticToolset([_search_tool_def(description=original_desc)])
        code_mode = CodeModeToolset(wrapped=toolset, tool_selector='all')
        ctx = build_run_context(None)
        tools = await code_mode.get_tools(ctx)

        modified_desc = tools[_SEARCH_TOOLS_NAME].tool_def.description
        assert modified_desc is not None
        assert modified_desc.startswith(original_desc)
        assert modified_desc.endswith(_SEARCH_TOOLS_MODIFIER)

    async def test_run_code_description_includes_search_note(self) -> None:
        """run_code description includes tool search addendum when search_tools present."""
        toolset = _StaticToolset([_search_tool_def()])
        code_mode = CodeModeToolset(wrapped=toolset, tool_selector='all')
        ctx = build_run_context(None)
        tools = await code_mode.get_tools(ctx)

        run_code_desc = tools['run_code'].tool_def.description
        assert run_code_desc is not None
        assert _TOOL_SEARCH_ADDENDUM.strip() in run_code_desc

    async def test_run_code_description_no_search_note_without_search_tools(self) -> None:
        """run_code description does NOT include search addendum when no search_tools."""
        toolset = _build_function_toolset(add)
        code_mode = CodeModeToolset(wrapped=toolset, tool_selector='all')
        ctx = build_run_context(None)
        tools = await code_mode.get_tools(ctx)

        run_code_desc = tools['run_code'].tool_def.description
        assert run_code_desc is not None
        assert 'search_tools' not in run_code_desc

    async def test_tool_search_toolset_deferred_tool_not_in_run_code(self) -> None:
        """End-to-end: `FunctionToolset` + `ToolSearchToolset` + `CodeMode`, before discovery.

        The deferred tool stays out of `run_code`'s description (progressive disclosure
        preserved). `ToolSearchToolset` still emits it as a corpus member carrying
        `defer_loading=True` / `with_native`, and `CodeMode` keeps it as a native
        pass-through so those flags reach `Model.prepare_request` unaltered. `search_tools`
        is native alongside `run_code`.
        """
        from pydantic_ai.toolsets._tool_search import _SEARCH_TOOLS_NAME, ToolSearchToolset

        def later(x: int) -> str:
            """A deferred-loading tool."""
            return str(x)  # pragma: no cover - tool body is not invoked in this test

        base = FunctionToolset[object](tools=[Tool(add), Tool(later, defer_loading=True)])
        code_mode = CodeModeToolset(wrapped=ToolSearchToolset(wrapped=base), tool_selector='all')
        tools = await code_mode.get_tools(build_run_context(None))

        description = tools['run_code'].tool_def.description
        assert description is not None
        assert 'async def add' in description
        # Not folded into run_code while undiscovered...
        assert 'later' not in description
        # ...but exposed as a native pass-through tool with its deferral intent intact.
        assert 'later' in tools
        assert tools['later'].tool_def.defer_loading is True
        # search_tools is the discovery surface and stays native alongside run_code.
        assert _SEARCH_TOOLS_NAME in tools
        assert _TOOL_SEARCH_ADDENDUM.strip() in description

    async def test_tool_search_toolset_discovered_tool_in_run_code(self) -> None:
        """End-to-end: once `search_tools` has discovered the deferred tool, it folds into `run_code`."""
        from pydantic_ai.messages import ModelMessage, ModelRequest, ToolSearchReturnPart
        from pydantic_ai.toolsets._tool_search import ToolSearchToolset, parse_discovered_tools

        def later(x: int) -> str:
            """A deferred-loading tool."""
            return str(x)  # pragma: no cover - tool body is not invoked in this test

        base = FunctionToolset[object](tools=[Tool(add), Tool(later, defer_loading=True)])
        code_mode = CodeModeToolset(wrapped=ToolSearchToolset(wrapped=base), tool_selector='all')

        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    ToolSearchReturnPart(
                        content={'discovered_tools': [{'name': 'later'}]},
                        tool_call_id='search-1',
                    )
                ]
            )
        ]
        ctx = RunContext[object](
            deps=None,
            model=TestModel(),
            usage=RunUsage(),
            prompt=None,
            messages=messages,
            # The agent graph reconstructs `discovered_tool_names` from history each step;
            # mirror that here since the test drives `get_tools` without a real run.
            discovered_tool_names=parse_discovered_tools(messages),
            run_step=1,
        )
        tools = await code_mode.get_tools(ctx)

        description = tools['run_code'].tool_def.description
        assert description is not None
        assert 'async def add' in description
        # Post-discovery the deferred tool comes back with `defer_loading=False`,
        # so it folds into run_code and is no longer a separate native tool.
        assert 'async def later' in description
        assert 'later' not in tools

    def test_code_mode_ordering(self) -> None:
        """CodeMode declares ordering: outermost position, wraps ToolSearch."""
        from pydantic_ai.capabilities._tool_search import ToolSearch

        ordering = CodeMode().get_ordering()
        assert ordering is not None
        assert ordering.position == 'outermost'
        assert ToolSearch in ordering.wraps


class TestDynamicCatalog:
    """`CodeMode(dynamic_catalog=True)`: move the catalog to instructions + announce discoveries.

    Two surfaces:

    1. **Catalog placement** — `CodeModeToolset` strips signatures from `run_code.description`
       and re-exposes them as a dynamic `InstructionPart` via `get_instructions`.
    2. **Discovery announcements** — `CodeMode.after_tool_execute` (local search) and
       `after_model_request` (native search) enqueue a `SystemPromptPart` so the model
       learns that freshly-discovered tools are callable.
    """

    # -- catalog placement -------------------------------------------------

    async def test_description_drops_signatures_keeps_base_prose(self) -> None:
        toolset = CodeModeToolset(wrapped=_build_function_toolset(add), tool_selector='all', dynamic_catalog=True)
        tools = await toolset.get_tools(build_run_context(None))

        description = tools['run_code'].tool_def.description
        assert description is not None
        # The signature is gone from the description...
        assert 'async def add' not in description
        # ...but the static base prose remains.
        assert 'sandboxed environment' in description

    async def test_catalog_surfaces_as_dynamic_instruction_part(self) -> None:
        toolset = CodeModeToolset(wrapped=_build_function_toolset(add), tool_selector='all', dynamic_catalog=True)
        ctx = build_run_context(None)
        await toolset.get_tools(ctx)
        instructions = await toolset.get_instructions(ctx)

        from pydantic_ai.messages import InstructionPart

        # No upstream instructions → the catalog is the only InstructionPart returned.
        assert isinstance(instructions, InstructionPart)
        assert 'async def add' in instructions.content
        # `dynamic=True` so Anthropic/Bedrock place the cache breakpoint before this block.
        assert instructions.dynamic is True

    async def test_get_instructions_appends_to_upstream_string(self) -> None:
        from pydantic_ai.messages import InstructionPart

        class _UpstreamToolset(FunctionToolset[object]):
            async def get_instructions(self, ctx: RunContext[object]) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
                return 'wrapped instructions'

        toolset = CodeModeToolset(
            wrapped=_UpstreamToolset(tools=[Tool(add)]), tool_selector='all', dynamic_catalog=True
        )
        ctx = build_run_context(None)
        await toolset.get_tools(ctx)
        instructions = await toolset.get_instructions(ctx)

        assert isinstance(instructions, list)
        assert instructions[0] == 'wrapped instructions'
        assert isinstance(instructions[1], InstructionPart)
        assert 'async def add' in instructions[1].content

    async def test_get_instructions_appends_to_upstream_sequence(self) -> None:
        from pydantic_ai.messages import InstructionPart

        class _UpstreamToolset(FunctionToolset[object]):
            async def get_instructions(  # pyright: ignore[reportIncompatibleMethodOverride]
                self, ctx: RunContext[object]
            ) -> list[str | InstructionPart]:
                return ['a', InstructionPart(content='b')]

        toolset = CodeModeToolset(
            wrapped=_UpstreamToolset(tools=[Tool(add)]), tool_selector='all', dynamic_catalog=True
        )
        ctx = build_run_context(None)
        await toolset.get_tools(ctx)
        instructions = await toolset.get_instructions(ctx)

        assert isinstance(instructions, list)
        assert instructions[0] == 'a'
        assert isinstance(instructions[1], InstructionPart) and instructions[1].content == 'b'
        # The catalog is appended at the end.
        assert isinstance(instructions[2], InstructionPart) and 'async def add' in instructions[2].content

    async def test_default_keeps_catalog_in_description_and_no_instructions(self) -> None:
        """With `dynamic_catalog=False` (default) the catalog stays in the description."""
        toolset = CodeModeToolset(wrapped=_build_function_toolset(add), tool_selector='all')
        ctx = build_run_context(None)
        tools = await toolset.get_tools(ctx)

        description = tools['run_code'].tool_def.description
        assert description is not None
        assert 'async def add' in description
        # Nothing stashed → defer to upstream (None for FunctionToolset).
        assert await toolset.get_instructions(ctx) is None

    async def test_empty_catalog_emits_no_instruction(self) -> None:
        """No sandboxed tools → empty catalog → defer to upstream instructions."""
        toolset = CodeModeToolset(wrapped=_build_function_toolset(), tool_selector='all', dynamic_catalog=True)
        ctx = build_run_context(None)
        tools = await toolset.get_tools(ctx)

        description = tools['run_code'].tool_def.description
        assert description is not None
        assert 'sandboxed environment' in description
        assert await toolset.get_instructions(ctx) is None

    async def test_search_addendum_stays_in_description(self) -> None:
        """The (cache-stable) search addendum stays in `run_code.description` even in dynamic mode."""
        toolset = CodeModeToolset(
            wrapped=_StaticToolset([_search_tool_def()]), tool_selector='all', dynamic_catalog=True
        )
        ctx = build_run_context(None)
        tools = await toolset.get_tools(ctx)

        description = tools['run_code'].tool_def.description
        assert description is not None
        assert _TOOL_SEARCH_ADDENDUM.strip() in description

    async def test_for_run_step_preserves_catalog_stash(self) -> None:
        """A per-step rebuild must carry `_last_catalog` so instructions stay populated."""

        class _ChangingToolset(FunctionToolset[object]):
            async def for_run_step(self, ctx: RunContext[object]) -> AbstractToolset[object]:
                # Force `CodeModeToolset.for_run_step` down the `new_wrapped is not self.wrapped`
                # branch by returning a distinct (but equivalent) wrapped instance.
                return type(self)(tools=list(self.tools.values()))

        toolset = CodeModeToolset(
            wrapped=_ChangingToolset(tools=[Tool(add)]), tool_selector='all', dynamic_catalog=True
        )
        ctx = build_run_context(None)
        await toolset.get_tools(ctx)
        stashed = toolset._last_catalog  # pyright: ignore[reportPrivateUsage]
        assert stashed  # populated

        new_toolset = await toolset.for_run_step(ctx)
        assert isinstance(new_toolset, CodeModeToolset)
        assert new_toolset is not toolset
        assert new_toolset._last_catalog == stashed  # pyright: ignore[reportPrivateUsage]

    # -- capability per-run state -----------------------------------------

    async def test_for_run_returns_fresh_state_when_enabled(self) -> None:
        cap = CodeMode[object](dynamic_catalog=True)
        cap._announced_tools.add('foo')  # pyright: ignore[reportPrivateUsage]
        fresh = await cap.for_run(build_run_context(None))
        assert fresh is not cap
        assert fresh._announced_tools == set()  # pyright: ignore[reportPrivateUsage]

    async def test_for_run_returns_self_when_disabled(self) -> None:
        cap = CodeMode[object]()
        assert await cap.for_run(build_run_context(None)) is cap

    # -- discovery announcement: local search path ------------------------

    async def test_announce_on_local_search_return(self) -> None:
        from pydantic_ai.messages import ModelRequest, SystemPromptPart, ToolCallPart

        cap = CodeMode[object](dynamic_catalog=True)
        ctx = build_run_context(None)
        await cap.after_tool_execute(
            ctx,
            call=ToolCallPart(tool_name='search_tools', args={}, tool_call_id='c1'),
            tool_def=_search_tool_def(),
            args={},
            result={'discovered_tools': [{'name': 'weather'}]},
        )

        assert ctx.pending_messages is not None
        assert len(ctx.pending_messages) == 1
        [request] = ctx.pending_messages[0].messages
        assert isinstance(request, ModelRequest)
        [part] = request.parts
        assert isinstance(part, SystemPromptPart)
        assert '`weather`' in part.content

    async def test_no_announce_when_disabled(self) -> None:
        """With `dynamic_catalog=False`, the hooks are inert even on a real search return."""
        from pydantic_ai.messages import ToolCallPart

        cap = CodeMode[object]()
        ctx = build_run_context(None)
        await cap.after_tool_execute(
            ctx,
            call=ToolCallPart(tool_name='search_tools', args={}, tool_call_id='c1'),
            tool_def=_search_tool_def(),
            args={},
            result={'discovered_tools': [{'name': 'weather'}]},
        )
        assert ctx.pending_messages == []

    async def test_announce_skipped_when_no_discoveries(self) -> None:
        from pydantic_ai.messages import ToolCallPart

        cap = CodeMode[object](dynamic_catalog=True)
        ctx = build_run_context(None)
        await cap.after_tool_execute(
            ctx,
            call=ToolCallPart(tool_name='search_tools', args={}, tool_call_id='c1'),
            tool_def=_search_tool_def(),
            args={},
            result={'discovered_tools': []},
        )
        assert ctx.pending_messages == []

    async def test_no_announce_for_non_search_tool(self) -> None:
        """`tool_kind != 'tool-search'` short-circuits before reading the result."""
        from pydantic_ai.messages import ToolCallPart

        cap = CodeMode[object](dynamic_catalog=True)
        ctx = build_run_context(None)
        await cap.after_tool_execute(
            ctx,
            call=ToolCallPart(tool_name='add', args={}, tool_call_id='c1'),
            tool_def=ToolDefinition(name='add', description='', parameters_json_schema={}),
            args={},
            # Even a `discovered_tools`-shaped result doesn't trigger an announcement:
            # the `tool_kind` guard is the source of truth.
            result={'discovered_tools': [{'name': 'spurious'}]},
        )
        assert ctx.pending_messages == []

    async def test_no_duplicate_announcement_for_same_tool(self) -> None:
        from pydantic_ai.messages import ToolCallPart

        cap = CodeMode[object](dynamic_catalog=True)
        ctx = build_run_context(None)
        result = {'discovered_tools': [{'name': 'weather'}]}
        for cid in ('c1', 'c2'):
            await cap.after_tool_execute(
                ctx,
                call=ToolCallPart(tool_name='search_tools', args={}, tool_call_id=cid),
                tool_def=_search_tool_def(),
                args={},
                result=result,
            )
        # Only the first discovery of `weather` announces.
        assert ctx.pending_messages is not None
        assert len(ctx.pending_messages) == 1

    # -- discovery announcement: native search path -----------------------

    async def test_announce_on_native_search_return_part(self) -> None:
        from pydantic_ai.messages import ModelRequest, ModelResponse, NativeToolSearchReturnPart, SystemPromptPart
        from pydantic_ai.usage import RequestUsage

        cap = CodeMode[object](dynamic_catalog=True)
        ctx = build_run_context(None)
        response = ModelResponse(
            parts=[
                NativeToolSearchReturnPart(
                    tool_name='tool_search',
                    content={'discovered_tools': [{'name': 'weather'}]},
                    tool_call_id='c1',
                )
            ],
            usage=RequestUsage(input_tokens=1, output_tokens=1),
        )
        await cap.after_model_request(ctx, request_context=None, response=response)  # pyright: ignore[reportArgumentType]

        assert ctx.pending_messages is not None
        assert len(ctx.pending_messages) == 1
        [request] = ctx.pending_messages[0].messages
        assert isinstance(request, ModelRequest)
        [part] = request.parts
        assert isinstance(part, SystemPromptPart) and '`weather`' in part.content

    async def test_no_announce_for_unrelated_response_parts(self) -> None:
        from pydantic_ai.messages import ModelResponse, NativeToolReturnPart, TextPart
        from pydantic_ai.usage import RequestUsage

        cap = CodeMode[object](dynamic_catalog=True)
        ctx = build_run_context(None)
        response = ModelResponse(
            parts=[
                TextPart('hi'),
                NativeToolReturnPart(tool_name='whatever', content='ignored', tool_call_id='c1'),
            ],
            usage=RequestUsage(input_tokens=1, output_tokens=1),
        )
        await cap.after_model_request(ctx, request_context=None, response=response)  # pyright: ignore[reportArgumentType]
        assert ctx.pending_messages == []

    # -- `_extract_discovered_names` edge cases ---------------------------

    @pytest.mark.parametrize(
        ('content', 'expected'),
        [
            ('not a dict', []),
            ({}, []),
            ({'discovered_tools': 'not a list'}, []),
            ({'discovered_tools': [{'name': 'a'}, 'not a dict', {'no_name': 1}, {'name': 42}]}, ['a']),
        ],
    )
    def test_extract_discovered_names_handles_malformed(self, content: Any, expected: list[str]) -> None:
        from pydantic_ai_harness.code_mode._capability import (
            _extract_discovered_names,  # pyright: ignore[reportPrivateUsage]
        )

        assert _extract_discovered_names(content) == expected

    # -- end-to-end via `Agent.run` ---------------------------------------

    async def test_agent_run_announces_discovery_and_lists_catalog_in_instructions(self) -> None:
        """`Agent.run` end-to-end: catalog in instructions, discovery enqueues an announcement.

        Two-step run:
          1. Model calls `search_tools(['weather'])` (the discovery surface).
          2. After the local tool-search returns, `CodeMode.after_tool_execute` enqueues a
             `SystemPromptPart`; the pending-message queue drains it into the next request.
             On the wire it renders as an (XML-wrapped) `UserPromptPart` — mid-conversation
             system content is no longer hoisted (pydantic/pydantic-ai#5509) — so the model
             sees the announcement inline and replies.
        """
        from pydantic_ai.capabilities import ToolSearch
        from pydantic_ai.messages import (
            ModelMessage,
            ModelRequest,
            ModelResponse,
            SystemPromptPart,
            TextPart,
            ToolCallPart,
            ToolReturnPart,
            ToolSearchReturnPart,
            UserPromptPart,
        )
        from pydantic_ai.models.function import AgentInfo, FunctionModel
        from pydantic_ai.usage import RequestUsage

        captured_prompt_texts: list[list[str]] = []
        captured_descriptions: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            run_code_def = next(td for td in info.function_tools if td.name == 'run_code')
            assert run_code_def.description is not None
            captured_descriptions.append(run_code_def.description)

            last_request = messages[-1]
            assert isinstance(last_request, ModelRequest)
            # The announcement may arrive as a `SystemPromptPart` or, after wire-rendering of
            # mid-conversation system content, an (XML-wrapped) `UserPromptPart` — capture both.
            captured_prompt_texts.append(
                [
                    p.content
                    for p in last_request.parts
                    if isinstance(p, (SystemPromptPart, UserPromptPart)) and isinstance(p.content, str)
                ]
            )

            if len(captured_descriptions) == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name='search_tools', args={'queries': ['weather']}, tool_call_id='c1')],
                    usage=RequestUsage(input_tokens=1, output_tokens=1),
                )
            return ModelResponse(parts=[TextPart('done')], usage=RequestUsage(input_tokens=1, output_tokens=1))

        def weather(city: str) -> str:
            """Get the weather."""
            return f'sunny in {city}'  # pragma: no cover — only the signature matters.

        agent: Agent[object, str] = Agent(
            FunctionModel(model_fn),
            tools=[Tool(weather, defer_loading=True)],
            capabilities=[ToolSearch[object](), CodeMode[object](dynamic_catalog=True)],
        )
        result = await agent.run('please find a weather tool')

        # `run_code.description` stayed static across both turns — no signature in the tool-defs block.
        assert all('async def' not in d for d in captured_descriptions)
        # The discovery announcement landed in turn 2's request (system- or user-framed).
        assert len(captured_prompt_texts) >= 2
        assert 'weather' in '\n'.join(captured_prompt_texts[1])
        # The local `ToolSearchReturnPart` is in history.
        history = result.all_messages()
        assert any(
            isinstance(p, ToolSearchReturnPart) for msg in history if isinstance(msg, ModelRequest) for p in msg.parts
        )
        assert any(
            isinstance(p, ToolReturnPart) and p.tool_name == 'search_tools'
            for msg in history
            if isinstance(msg, ModelRequest)
            for p in msg.parts
        )
        assert result.output == 'done'

    async def test_run_code_calls_eager_tool_with_catalog_in_instructions(self) -> None:
        """An eager tool whose signature lives in instructions is still callable via `run_code`."""
        from pydantic_ai.messages import (
            ModelMessage,
            ModelRequest,
            ModelResponse,
            TextPart,
            ToolCallPart,
            ToolReturnPart,
        )
        from pydantic_ai.models.function import AgentInfo, FunctionModel
        from pydantic_ai.usage import RequestUsage

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            run_code_def = next(td for td in info.function_tools if td.name == 'run_code')
            assert run_code_def.description is not None
            assert 'async def add' not in run_code_def.description
            if not any(isinstance(msg, ModelResponse) for msg in messages):
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='run_code',
                            # CodeMode renders all tools as `async def` by default — use `await`.
                            args={'code': 'result = await add(a=3, b=4)\nresult'},
                            tool_call_id='c1',
                        )
                    ],
                    usage=RequestUsage(input_tokens=1, output_tokens=1),
                )
            last_request = messages[-1]
            assert isinstance(last_request, ModelRequest)
            run_code_return = next(p for p in last_request.parts if isinstance(p, ToolReturnPart))
            return ModelResponse(
                parts=[TextPart(f'got {run_code_return.content}')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
            )

        agent: Agent[object, str] = Agent(
            FunctionModel(model_fn),
            tools=[Tool(add)],
            capabilities=[CodeMode[object](dynamic_catalog=True)],
        )
        result = await agent.run('add 3 and 4 via run_code')
        assert result.output == 'got 7'


def _unused_os_callback(fn: OsFunction, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """An `os` callback for tests that only assert description/forwarding, never run code."""
    return NOT_HANDLED  # pragma: no cover - never invoked by these tests


class TestCodeModeOSAccess:
    """`CodeMode(os_access=...)` / `mount=...` give sandboxed code host-backed OS access."""

    async def test_description_default_notes_no_fs_env_or_clock(self) -> None:
        """Without `os`/`mount`, the description states filesystem, env, and clock calls are
        unavailable, so the model does not waste retries calling `pathlib`/`os` I/O."""
        wrapper = CodeMode[object]().get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)
        description = (await wrapper.get_tools(build_run_context(None)))['run_code'].tool_def.description
        assert description is not None
        assert 'No filesystem, environment, or timing primitives' in description
        assert 'their I/O operations are not supported in this configuration' in description

    async def test_description_with_os_callback_notes_host_access(self) -> None:
        """An `os` callback swaps the restriction line for the host-access note."""
        wrapper = CodeMode[object](os_access=_unused_os_callback).get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)
        description = (await wrapper.get_tools(build_run_context(None)))['run_code'].tool_def.description
        assert description is not None
        assert 'Host-backed OS access' in description

    async def test_description_mount_only_advertises_filesystem_not_env_or_clock(self, tmp_path: Path) -> None:
        """A `mount` without `os` advertises filesystem access only -- it must not tell the model
        that env/clock are host-backed, since a mount cannot route `os.getenv`/`datetime.now()`."""
        wrapper = CodeMode[object](mount=MountDir('/work', str(tmp_path))).get_wrapper_toolset(
            _build_function_toolset(add)
        )
        assert isinstance(wrapper, CodeModeToolset)
        description = (await wrapper.get_tools(build_run_context(None)))['run_code'].tool_def.description
        assert description is not None
        # The regression guard: a mount must select the filesystem note, not the OS note that would
        # (wrongly) advertise env/clock as host-routed -- this assert fails if the OS note is picked.
        assert 'Mounted filesystem access' in description

    async def test_description_host_access_note_shows_with_no_sandboxed_tools(self) -> None:
        """The host-access note appears even when no tools are sandboxed (base description)."""
        # `tools=[]` sandboxes nothing, so `run_code` renders the base description path.
        wrapper = CodeMode[object](os_access=_unused_os_callback, tools=[]).get_wrapper_toolset(
            _build_function_toolset(add)
        )
        assert isinstance(wrapper, CodeModeToolset)
        description = (await wrapper.get_tools(build_run_context(None)))['run_code'].tool_def.description
        assert description is not None
        assert 'Host-backed OS access' in description

    async def test_os_callback_dispatches_inside_run_code(self) -> None:
        """An `os` callback is threaded through `feed_start` and every `resume`, so OS calls
        keep dispatching even after a tool call suspends and resumes the sandbox."""

        def os_cb(fn: OsFunction, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            if fn == 'os.getenv':
                return 'envval'
            return NOT_HANDLED  # pragma: no cover - sandbox only calls os.getenv here

        wrapper = CodeMode[object](os_access=os_cb).get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        # The tool call forces a FunctionSnapshot -> FutureSnapshot round-trip; the os.getenv
        # afterwards only resolves if `os` survived those resumes.
        code = "import os\nx = await add(a=2, b=3)\nhome = os.getenv('THING')\n{'sum': x, 'home': home}"
        result = await wrapper.call_tool('run_code', {'code': code}, ctx, tools['run_code'])
        assert result.return_value == {'sum': 5, 'home': 'envval'}

    async def test_os_access_persists_across_run_code_calls(self) -> None:
        """`os` is supplied on every `feed_start`, so OS access still works on a later
        `run_code` call that reuses the persisted (non-fresh) REPL."""

        def os_cb(fn: OsFunction, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            if fn == 'os.getenv':
                return 'persisted'
            return NOT_HANDLED  # pragma: no cover - sandbox only calls os.getenv here

        wrapper = CodeMode[object](os_access=os_cb).get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        first = await wrapper.call_tool('run_code', {'code': "import os\nos.getenv('A')"}, ctx, tools['run_code'])
        assert first.return_value == 'persisted'
        # Second call reuses the REPL (so `import os` carries over) and must still dispatch.
        second = await wrapper.call_tool('run_code', {'code': "os.getenv('B')"}, ctx, tools['run_code'])
        assert second.return_value == 'persisted'

    async def test_abstract_os_instance_dispatches_inside_run_code(self) -> None:
        """An `AbstractOS` instance is accepted as the `os` value and dispatches OS calls."""
        wrapper = CodeMode[object](os_access=OSAccess(environ={'THING': 'fromabs'})).get_wrapper_toolset(
            _build_function_toolset(add)
        )
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        result = await wrapper.call_tool('run_code', {'code': "import os\nos.getenv('THING')"}, ctx, tools['run_code'])
        assert result.return_value == 'fromabs'

    async def test_os_callback_exception_becomes_model_retry(self) -> None:
        """A raising `os` callback surfaces as a `ModelRetry`, like any other sandbox runtime
        error -- it must not crash the agent loop."""

        def os_cb(fn: OsFunction, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            raise ValueError('boom from os')

        wrapper = CodeMode[object](os_access=os_cb).get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        with pytest.raises(ModelRetry, match='boom from os'):
            await wrapper.call_tool('run_code', {'code': "import os\nos.getenv('X')"}, ctx, tools['run_code'])

    async def test_os_callback_returning_value_answers_call_including_none(self) -> None:
        """Returning a value from the `os` callback -- even `None` -- *answers* the call.

        Allow-listed keys resolve; every other key reads back as `None`, exactly like a real
        unset env var, so the sandbox keeps running with no retry. This is how a callback hides
        a secret: by answering with an empty value, not by refusing the call.
        """
        allowed = {'API_KEY': 'sk-xxx'}

        def os_cb(fn: OsFunction, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            if fn == 'os.getenv':
                return allowed.get(args[0])
            return NOT_HANDLED  # pragma: no cover - sandbox only calls os.getenv here

        wrapper = CodeMode[object](os_access=os_cb).get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        code = "import os\n{'allowed': os.getenv('API_KEY'), 'hidden': os.getenv('SECRET')}"
        result = await wrapper.call_tool('run_code', {'code': code}, ctx, tools['run_code'])
        assert result.return_value == {'allowed': 'sk-xxx', 'hidden': None}

    async def test_os_callback_not_handled_refuses_call_as_model_retry(self) -> None:
        """Returning `NOT_HANDLED` *refuses* the call rather than answering it.

        The OS function is treated as unsupported, so it raises in the sandbox and surfaces as
        `ModelRetry`. This is the counterpart to returning a value: refusing is not the same as
        answering `None`, and using it for a key the model expects will burn retries.
        """

        def os_cb(fn: OsFunction, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            return NOT_HANDLED

        wrapper = CodeMode[object](os_access=os_cb).get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        with pytest.raises(ModelRetry, match='not supported in this environment'):
            await wrapper.call_tool('run_code', {'code': "import os\nos.getenv('X')"}, ctx, tools['run_code'])

    async def test_mount_exposes_host_directory(self, tmp_path: Path) -> None:
        """A `mount` exposes a host directory inside the sandbox, threaded through resumes."""
        (tmp_path / 'data.txt').write_text('hello-from-host')
        wrapper = CodeMode[object](mount=MountDir('/work', str(tmp_path))).get_wrapper_toolset(
            _build_function_toolset(add)
        )
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        code = "from pathlib import Path\nawait add(a=1, b=1)\nPath('/work/data.txt').read_text()"
        result = await wrapper.call_tool('run_code', {'code': code}, ctx, tools['run_code'])
        assert result.return_value == 'hello-from-host'

    async def test_mount_accepts_list_of_directories(self, tmp_path: Path) -> None:
        """`mount` accepts a `list[MountDir]`; each directory is exposed at its virtual path."""
        (tmp_path / 'a').mkdir()
        (tmp_path / 'b').mkdir()
        (tmp_path / 'a' / 'f.txt').write_text('AA')
        (tmp_path / 'b' / 'f.txt').write_text('BB')
        mounts = [MountDir('/a', str(tmp_path / 'a')), MountDir('/b', str(tmp_path / 'b'))]
        wrapper = CodeMode[object](mount=mounts).get_wrapper_toolset(_build_function_toolset(add))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tools = await wrapper.get_tools(ctx)
        code = "from pathlib import Path\nPath('/a/f.txt').read_text() + Path('/b/f.txt').read_text()"
        result = await wrapper.call_tool('run_code', {'code': code}, ctx, tools['run_code'])
        assert result.return_value == 'AABB'

    def test_capability_forwards_os_and_mount_to_toolset(self, tmp_path: Path) -> None:
        """`CodeMode` forwards `os_access`/`mount` onto the `CodeModeToolset` it builds."""
        mount = MountDir('/work', str(tmp_path))
        wrapper = CodeMode[object](os_access=_unused_os_callback, mount=mount).get_wrapper_toolset(
            _build_function_toolset(add)
        )
        assert isinstance(wrapper, CodeModeToolset)
        assert wrapper.os_access is _unused_os_callback
        assert wrapper.mount is mount


def _search_tool_def(description: str = 'Search for tools.') -> ToolDefinition:
    """Create a ToolDefinition mimicking the search_tools tool from ToolSearchToolset.

    Carries `tool_kind='tool-search'`, matching what pydantic-ai emits (since 1.95.0);
    CodeMode routes it native off `tool_kind`, not its name.
    """
    from pydantic_ai.toolsets._tool_search import _SEARCH_TOOLS_NAME

    return ToolDefinition(
        name=_SEARCH_TOOLS_NAME,
        description=description,
        parameters_json_schema={'type': 'object', 'properties': {'keywords': {'type': 'string'}}},
        tool_kind='tool-search',
    )


class TestGlobalModeIsSequential:
    """`_global_mode_is_sequential` dispatches across pydantic-ai v1 and v2.

    v1's `get_parallel_execution_mode` takes the pending calls list; v2 dropped
    the argument. The helper inspects arity and calls the matching shape, so
    both code paths are exercised here regardless of which major is installed.
    """

    def test_v1_signature_with_calls_argument(self) -> None:
        def parallel(calls: list[ToolCallPart]) -> ParallelExecutionMode:
            return 'parallel'

        def sequential(calls: list[ToolCallPart]) -> ParallelExecutionMode:
            return 'sequential'

        assert _global_mode_is_sequential(parallel) is False
        assert _global_mode_is_sequential(sequential) is True

    def test_v2_signature_without_arguments(self) -> None:
        def parallel() -> ParallelExecutionMode:
            return 'parallel'

        def sequential() -> ParallelExecutionMode:
            return 'sequential'

        assert _global_mode_is_sequential(parallel) is False
        assert _global_mode_is_sequential(sequential) is True
