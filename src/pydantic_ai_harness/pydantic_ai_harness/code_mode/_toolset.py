"""Code mode toolset that runs LLM-generated Python in a Monty sandbox."""

from __future__ import annotations

import asyncio
import keyword
import re
import warnings
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field, replace
from typing import Annotated, Any

from pydantic import Field, TypeAdapter
from pydantic_ai import AbstractToolset, RunContext, ToolDefinition, WrapperToolset
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.function_signature import FunctionSignature
from pydantic_ai.messages import ToolCallPart, ToolReturn, ToolReturnContent, ToolReturnPart, is_multi_modal_content
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import AgentDepsT, ToolDenied, ToolSelector, matches_tool_selector
from pydantic_ai.toolsets.abstract import SchemaValidatorProt, ToolsetTool

try:
    from pydantic_ai.toolsets._tool_search import _SEARCH_TOOLS_NAME  # pyright: ignore[reportPrivateUsage]
except ImportError:  # pragma: no cover
    _SEARCH_TOOLS_NAME = 'search_tools'  # pyright: ignore[reportConstantRedefinition]

try:
    from pydantic_monty import (
        ExternalException,
        ExternalResult,
        ExternalReturnValue,
        FunctionSnapshot,
        FutureSnapshot,
        Monty,
        MontyComplete,
        MontyRepl,
        MontyRuntimeError,
        MontySyntaxError,
        MontyTypingError,
        NameLookupSnapshot,
    )
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'pydantic-monty is required for CodeMode. Install it with: pip install "pydantic-ai-harness[code-mode]"'
    ) from _import_error
from typing_extensions import NotRequired, TypedDict

# Type alias for the dispatch callback passed to _execution_loop.
_DispatchFn = Callable[[str, dict[str, Any]], Coroutine[Any, Any, Any]]


class _RunCodeArguments(TypedDict):
    code: Annotated[str, Field(description='The Python code to execute in the sandbox.')]
    restart: NotRequired[
        Annotated[
            bool,
            Field(
                description='Set to true to reset REPL state. When false (default), state is preserved between calls.'
            ),
        ]
    ]


_RUN_CODE_TOOL_NAME = 'run_code'
_RUN_CODE_ADAPTER = TypeAdapter(_RunCodeArguments)
_RUN_CODE_JSON_SCHEMA = _RUN_CODE_ADAPTER.json_schema()
_RUN_CODE_ARGS_VALIDATOR: SchemaValidatorProt = _RUN_CODE_ADAPTER.validator  # pyright: ignore[reportAssignmentType]
# Used to serialize tool return values before sending into Monty (dump_python)
# and to reconstruct multimodal types (e.g. BinaryContent) from Monty results (validate_python).
_TOOL_RETURN_CONTENT_TA: TypeAdapter[Any] = TypeAdapter(ToolReturnContent)

_RUN_CODE_BASE_DESCRIPTION = """\
Write and run Python code in a sandboxed environment.

The sandbox uses Monty, a subset of Python. Key restrictions:
- **No classes**: class definitions are not supported
- **No third-party libraries**: only the standard library modules listed below can be used
- **Importable standard library modules**: `sys`, `typing`, `asyncio`, `math`, `json`, `re`, `datetime`, `os`, `pathlib`. These must be imported at the top of your snippet before use, just like in regular Python. For example: `import asyncio` then `results = await asyncio.gather(tool_one(...), tool_two(...))`.
- **No wall-clock or timing primitives**: `asyncio.sleep`, `datetime.datetime.now()`, `datetime.date.today()`, and the `time` module are unavailable.
- **No `import *`**: wildcard imports are not supported

State is preserved between calls (REPL-style). Set `restart: true` to reset state.

The last expression's value is automatically captured as the return value -- you do **not** need to \
`print()` it. Avoid `print()` for return values as it produces Python string representations, not \
structured data. Use `print()` only for supplementary logging or debug output.

Returns the last expression's value directly. If `print()` was also called, returns \
`{"output": "<printed text>", "result": <last expression>}`.\
"""


def _functions_header(*, has_sync: bool, has_async: bool) -> str:
    """Build the functions-header paragraph for the `run_code` tool description."""
    base = (
        '\nThe following functions are available inside the sandbox. Call them directly '
        '(do **not** redefine or import them). All parameters are keyword-only.'
    )
    if has_async and not has_sync:
        return base + (
            ' All tool functions are async: invoke them with `await`,'
            ' e.g. `result = await tool_name(arg=value)`.'
            ' Calling without `await` returns an unresolved future, not the value.'
        )
    if has_sync and not has_async:
        return base + (' All tool functions are synchronous: call them directly, e.g. `result = tool_name(arg=value)`.')
    return base + (
        ' Async functions (`async def`) must be invoked with `await`,'
        ' e.g. `result = await tool_name(arg=value)`.'
        ' Sync functions (`def`) are called directly, e.g. `result = tool_name(arg=value)`.'
    )


_SEARCH_TOOLS_MODIFIER = (
    ' Note: discovered tools become callable as functions inside the run_code sandbox in subsequent invocations.'
)

_TOOL_SEARCH_ADDENDUM = (
    f'\n\nNot all functions may be available initially.'
    f' Use the `{_SEARCH_TOOLS_NAME}` tool to discover additional functions'
    f' that will become callable in subsequent `run_code` invocations.'
)

_INVALID_IDENT_CHARS = re.compile(r'[^a-zA-Z0-9_]')


def _sanitize_tool_name(name: str) -> str:
    """Turn a tool name into a valid Python identifier.

    Replaces hyphens, dots, and other non-identifier characters with underscores,
    prepends `_` if the result starts with a digit, appends `_` if it is a Python keyword.
    """
    sanitized = _INVALID_IDENT_CHARS.sub('_', name)
    if sanitized and sanitized[0].isdigit():
        sanitized = f'_{sanitized}'
    if keyword.iskeyword(sanitized):
        sanitized = f'{sanitized}_'
    return sanitized or '_'


@dataclass(kw_only=True)
class _RunCodeTool(ToolsetTool[AgentDepsT]):
    """ToolsetTool subclass that caches data computed during `get_tools`.

    Avoids a redundant `get_tools` call in `call_tool` by storing the
    callable tool definitions and name mapping on the tool instance itself.
    Follows the same pattern as `_SearchTool` in pydantic-ai's
    `ToolSearchToolset`.
    """

    callable_defs: dict[str, ToolDefinition]
    """Tool definitions callable from inside the sandbox, keyed by (possibly sanitized) name."""

    sanitized_to_original: dict[str, str]
    """Maps sanitized Python-safe names back to original tool names (only for renamed tools)."""

    wrapped_tools: dict[str, ToolsetTool[AgentDepsT]]
    """The wrapped toolset's tools, keyed by original name."""


@dataclass
class CodeModeToolset(WrapperToolset[AgentDepsT]):
    """Implementation toolset for the `CodeMode` capability.

    Exposes a single `run_code` tool alongside any native (non-sandboxed) tools.
    Tools selected by `tool_selector` are presented to the model as Python
    function signatures inside the `run_code` tool description and become
    callable from the sandbox at runtime. Non-selected tools remain visible
    to the model as normal tool calls.

    Tools with `defer_loading=True` (Tool Search) are never sandboxed: they stay
    native pass-through so the deferred-loading contract is honored, and only get
    folded into `run_code` once they've been discovered (`defer_loading=False`).
    Tools annotated with `unless_native` likewise stay native so
    `Model.prepare_request` can drop them when the provider supports the native tool.
    To keep a Tool Search corpus native even after discovery (e.g. for prompt-cache
    stability), pass a `tool_selector` that excludes tools with `with_native` set.
    """

    tool_selector: ToolSelector[AgentDepsT] = 'all'
    """Which wrapped tools to sandbox inside `run_code`. Non-matching tools
    are exposed as native tools."""

    max_retries: int = 3
    """Maximum number of retries for the `run_code` tool (syntax errors count as retries)."""

    # init=False so `replace()` in `for_run` produces a fresh instance with _repl=None,
    # giving each agent run isolated REPL state. Lazy-initialized on first call_tool.
    _repl: MontyRepl | None = field(default=None, init=False, repr=False)

    # Tracks deferred-tool names we've already warned about so we don't spam the
    # logs every step. Reset on `for_run` because each run gets a fresh instance.
    _warned_deferred: set[str] = field(default_factory=set[str], init=False, repr=False)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Return a fresh toolset instance with isolated REPL state for this agent run."""
        wrapped = await self.wrapped.for_run(ctx)
        return replace(self, wrapped=wrapped)

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Update the wrapped toolset for this step while preserving REPL state."""
        new_wrapped = await self.wrapped.for_run_step(ctx)
        if new_wrapped is self.wrapped:
            return self
        new_self = replace(self, wrapped=new_wrapped)
        new_self._repl = self._repl
        new_self._warned_deferred = self._warned_deferred
        return new_self

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Return the `run_code` tool plus any native (non-sandboxed) tools."""
        wrapped_tools = await self.wrapped.get_tools(ctx)

        # Split tools into sandboxed vs native based on the selector.
        # The search_tools tool (from ToolSearchToolset) is always kept native
        # so the model can discover deferred tools alongside run_code.
        sandboxed_tools: dict[str, ToolsetTool[AgentDepsT]] = {}
        native_tools: dict[str, ToolsetTool[AgentDepsT]] = {}
        for name, tool in wrapped_tools.items():
            if name == _SEARCH_TOOLS_NAME:
                native_tools[name] = tool
            elif tool.tool_def.defer_loading:
                # Tool Search keeps these out of the model's initial context until discovered.
                # Stay native pass-through so `ToolSearchToolset`'s `defer_loading` /
                # `with_native` flags reach `Model.prepare_request` unaltered; once a tool is
                # discovered it comes back with `defer_loading=False` and is sandboxed from then on.
                native_tools[name] = tool
            elif tool.tool_def.unless_native:
                # Defer to `Model.prepare_request`'s `unless_native` filtering: keep the local
                # fallback native so it can be dropped when the provider supports the native tool.
                native_tools[name] = tool
            elif await matches_tool_selector(self.tool_selector, ctx, tool.tool_def):
                sandboxed_tools[name] = tool
            else:
                native_tools[name] = tool

        callable_defs, sanitized_to_original = self._partition_callable_tools(sandboxed_tools)

        description = self._build_description(callable_defs)

        if _RUN_CODE_TOOL_NAME in native_tools:
            raise UserError(
                f"Tool name '{_RUN_CODE_TOOL_NAME}' is reserved for code mode. Rename your tool to avoid conflicts."
            )

        # When search_tools is present, append context about run_code to its
        # description and add a discovery note to the run_code description.
        has_search_tools = _SEARCH_TOOLS_NAME in native_tools
        if has_search_tools:
            search_tool = native_tools[_SEARCH_TOOLS_NAME]
            native_tools[_SEARCH_TOOLS_NAME] = replace(
                search_tool,
                tool_def=replace(
                    search_tool.tool_def,
                    description=(search_tool.tool_def.description or '') + _SEARCH_TOOLS_MODIFIER,
                ),
            )
            description += _TOOL_SEARCH_ADDENDUM

        result: dict[str, ToolsetTool[AgentDepsT]] = dict(native_tools)
        result[_RUN_CODE_TOOL_NAME] = _RunCodeTool(
            toolset=self,
            tool_def=ToolDefinition(
                name=_RUN_CODE_TOOL_NAME,
                description=description,
                parameters_json_schema=_RUN_CODE_JSON_SCHEMA,
                metadata={'code_arg_name': 'code', 'code_arg_language': 'python'},
                sequential=True,
            ),
            max_retries=self.max_retries,
            args_validator=_RUN_CODE_ARGS_VALIDATOR,
            callable_defs=callable_defs,
            sanitized_to_original=sanitized_to_original,
            wrapped_tools=wrapped_tools,
        )
        return result

    async def call_tool(  # noqa: C901
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        """Execute Python code in the sandbox, or pass through to a native tool."""
        if not isinstance(tool, _RunCodeTool):
            # Native (non-sandboxed) tool -- pass through to the wrapped toolset.
            return await self.wrapped.call_tool(name, tool_args, ctx, tool)

        code = tool_args['code']
        restart = tool_args.get('restart', False)

        # Clear the REPL on restart so that if type checking fails, the
        # next retry still gets fresh_repl=True and is type-checked again.
        if restart:
            self._repl = None
        fresh_repl = self._repl is None

        callable_defs = tool.callable_defs
        sanitized_to_original = tool.sanitized_to_original

        # Build a ToolManager for the sandbox's inner tools so that sandboxed
        # tool calls go through the standard validation/execution path. We
        # inherit `root_capability` from the agent's ToolManager (for capability
        # hooks) but use the *wrapped* toolset and its tools.
        # See https://github.com/pydantic/pydantic-ai/pull/4307
        parent_tm = ctx.tool_manager
        assert parent_tm is not None, 'CodeModeToolset requires ctx.tool_manager to be set'
        tool_manager = ToolManager(
            toolset=self.wrapped,
            root_capability=parent_tm.root_capability,
            ctx=ctx,
            tools=tool.wrapped_tools,
        )

        # Determine execution mode for sandbox tool calls:
        # - global_sequential: forced by durable execution engines (DBOS/Temporal)
        #   via the parallel execution mode context var. Checked with empty calls
        #   to isolate the context var from per-tool flags.
        # - sequential_tools: per-tool `sequential` flags on ToolDefinition.
        #   These tools are rendered as `def` (sync) and resolved inline.
        global_sequential = tool_manager.get_parallel_execution_mode([]) != 'parallel'
        sequential_tools = {name for name, td in callable_defs.items() if td.sequential}

        # Collect nested tool calls and returns keyed by tool_call_id so they
        # can be attached as metadata on the run_code ToolReturnPart.
        nested_calls: dict[str, ToolCallPart] = {}
        nested_returns: dict[str, ToolReturnPart] = {}
        call_counter = 0

        async def dispatch_tool_call(original_name: str, kwargs: dict[str, Any]) -> Any:
            """Dispatch a single tool call from inside the sandbox.

            Returns the serialized tool result on success. On failure, the
            exception propagates -- the execution loop passes it back into
            Monty via `ExternalException` so the sandbox sees it at the
            `await` site.
            """
            nonlocal call_counter
            call_counter += 1
            parent_id = ctx.tool_call_id or 'pyd_ai_code_mode'
            tool_call_id = f'{parent_id}__{call_counter}'
            call_part = ToolCallPart(tool_name=original_name, args=kwargs, tool_call_id=tool_call_id)
            nested_calls[tool_call_id] = call_part

            try:
                result = await tool_manager.handle_call(call_part, wrap_validation_errors=False)
            except (CallDeferred, ApprovalRequired) as e:
                # No handler resolved the deferral. The sandbox can't round-trip to the
                # caller, so we convert it to a UserError that propagates through
                # Monty → MontyRuntimeError → ModelRetry.
                raise UserError(
                    f'Tool {original_name!r} raised {type(e).__name__} inside code mode, '
                    'but no `HandleDeferredToolCalls` capability resolved it. Add a handler '
                    'capability on the agent so deferred and approval-required calls can '
                    'be resolved inline.'
                ) from e

            if isinstance(result, ToolDenied):
                # Handler denied the call. Record the denial with outcome='denied' so
                # message history reflects it, then raise inside the sandbox: surfacing
                # `ToolDenied` to the user's script would let it masquerade as a string
                # tool result, and the script has no way to introspect the marker class
                # since `ToolDenied` isn't exposed inside Monty.
                nested_returns[tool_call_id] = ToolReturnPart(
                    tool_name=original_name,
                    content=result.message,
                    tool_call_id=tool_call_id,
                    outcome='denied',
                )
                raise RuntimeError(f'Tool {original_name!r} call denied: {result.message}')

            # Unwrap ToolReturn to get the plain value for the sandbox,
            # preserving the full ToolReturn metadata on the return part.
            return_metadata: Any = None
            if isinstance(result, ToolReturn):
                return_metadata = result.metadata
                result = result.return_value

            nested_returns[tool_call_id] = ToolReturnPart(
                tool_name=original_name,
                content=result,
                tool_call_id=tool_call_id,
                metadata=return_metadata,
            )

            # Serialize to JSON-compatible form so Monty receives only plain data.
            return _TOOL_RETURN_CONTENT_TA.dump_python(result)

        # Static type checking on fresh REPL sessions (first call or after
        # restart). Skipped on subsequent calls because accumulated REPL state
        # (variables from prior snippets) is invisible to the stateless checker.
        # Runs before REPL creation so that if this raises ModelRetry, the REPL
        # stays None and the next retry still gets type-checked.
        if fresh_repl and callable_defs:
            self._type_check(code, callable_defs=callable_defs)

        # Create the REPL after type checking passes.
        if fresh_repl:
            self._repl = MontyRepl()
        assert self._repl is not None

        capture = _PrintCapture()

        try:
            monty_state = self._repl.feed_start(code, print_callback=capture)
            completed = await _execution_loop(
                monty_state,
                dispatch=dispatch_tool_call,
                callable_defs=callable_defs,
                sanitized_to_original=sanitized_to_original,
                sequential_tools=sequential_tools,
                global_sequential=global_sequential,
            )
        except MontySyntaxError as e:
            raise ModelRetry(f'Syntax error in code:\n{_prepend_prints(e.display(), capture)}') from e
        except MontyTypingError as e:  # pragma: no cover -- MontyRepl.feed_start doesn't raise this
            raise ModelRetry(f'Type error in code:\n{_prepend_prints(e.display(), capture)}') from e
        except MontyRuntimeError as e:
            # Exceptions raised inside dispatch_tool_call (e.g. UserError from
            # ApprovalRequired, or ModelRetry from a wrapped tool) are passed
            # back into Monty via ExternalException. Monty re-raises them at the
            # await site; if the sandbox code doesn't catch them, they bubble up
            # as MontyRuntimeError. The original exception message is preserved
            # in the display string, so the model sees a useful error. This means
            # ModelRetry from a wrapped tool gets double-wrapped
            # (ModelRetry → MontyRuntimeError → ModelRetry), but the retry
            # semantics are the same -- the model gets another chance.
            raise ModelRetry(f'Runtime error:\n{_prepend_prints(e.display(), capture)}') from e

        result = completed.output
        printed = capture.joined

        # Validate result to reconstruct multimodal types (e.g. BinaryContent from
        # serialized dicts) so they flow through to the model natively.
        if result is not None:
            result = _TOOL_RETURN_CONTENT_TA.validate_python(result)

        # Build return value:
        # - No print → return result directly (multimodal content stays top-level
        #   so _split_content can extract it for native model delivery)
        # - Print + multimodal result → list format so _split_content can extract files
        # - Print + plain result → dict with output/result keys
        if not printed:
            return_value: Any = result if result is not None else {}
        elif result is None:
            return_value = {'output': printed}
        elif _contains_multimodal(result):
            # Flatten lists so _split_content can find each multimodal item at top level.
            return_value = [printed, *result] if isinstance(result, list) else [printed, result]
        else:
            return_value = {'output': printed, 'result': result}

        return ToolReturn(
            return_value=return_value,
            metadata={'code_mode': True, 'tool_calls': nested_calls, 'tool_returns': nested_returns},
        )

    def _partition_callable_tools(
        self, wrapped_tools: dict[str, ToolsetTool[AgentDepsT]]
    ) -> tuple[dict[str, ToolDefinition], dict[str, str]]:
        """Return tool definitions that can be called from inside the sandbox.

        Tool names that are not valid Python identifiers (e.g. MCP tools with
        hyphens or dots like `get-weather`, `api.call`) are sanitized to
        underscored forms and mapped back to their original names for dispatch.

        Returns:
            A tuple of `(callable_defs, sanitized_to_original)`.
        """
        callable_defs: dict[str, ToolDefinition] = {}
        sanitized_to_original: dict[str, str] = {}
        for name, tool in wrapped_tools.items():
            td = tool.tool_def

            safe_name = _sanitize_tool_name(name)
            if safe_name == _RUN_CODE_TOOL_NAME:
                raise UserError(
                    f"Tool name '{name}' (sanitized to '{safe_name}') conflicts with the code mode "
                    f'meta-tool. Rename your tool to avoid conflicts.'
                )
            if safe_name in callable_defs:
                existing = sanitized_to_original.get(safe_name, safe_name)
                warnings.warn(
                    f'CodeMode: tool {name!r} (sanitized to {safe_name!r}) collides '
                    f'with {existing!r}; {name!r} will be hidden from the sandbox.',
                    UserWarning,
                    stacklevel=2,
                )
                continue
            # Warn when a sandboxed tool has no return schema -- the generated
            # signature will show `-> Any`, giving the model no type information
            # about the return shape, which limits code mode effectiveness.
            if td.return_schema is None and name not in self._warned_deferred:
                self._warned_deferred.add(name)
                warnings.warn(
                    f'CodeMode: tool {name!r} has no return schema; '
                    f'its signature will show `-> Any`, which may reduce code mode effectiveness.',
                    UserWarning,
                    stacklevel=2,
                )

            if safe_name != name:
                sanitized_to_original[safe_name] = name
                td = replace(td, name=safe_name)

            callable_defs[safe_name] = td
        return callable_defs, sanitized_to_original

    @staticmethod
    def _build_description(callable_defs: dict[str, ToolDefinition]) -> str:
        """Render the `run_code` description: base prose + TypedDicts + function signatures."""
        if not callable_defs:
            return _RUN_CODE_BASE_DESCRIPTION

        sigs, conflicting = _get_sigs_and_conflicting(callable_defs)
        type_blocks = FunctionSignature.render_type_definitions(sigs, conflicting)
        function_blocks = [
            td.render_signature('...', is_async=not td.sequential, conflicting_type_names=conflicting)
            for td in callable_defs.values()
        ]

        has_sync = any(td.sequential for td in callable_defs.values())
        has_async = any(not td.sequential for td in callable_defs.values())
        header = _functions_header(has_sync=has_sync, has_async=has_async)

        sections = [_RUN_CODE_BASE_DESCRIPTION, header]
        if type_blocks:
            sections.append('```python\n' + '\n\n'.join(type_blocks) + '\n```')
        sections.append('```python\n' + '\n\n'.join(function_blocks) + '\n```')
        return '\n\n'.join(sections)

    @staticmethod
    def _build_type_check_stubs(callable_defs: dict[str, ToolDefinition]) -> str:
        """Build Python stubs for Monty's static type checker."""
        sigs, conflicting = _get_sigs_and_conflicting(callable_defs)
        parts = ['import asyncio\nfrom typing import Any, TypedDict, NotRequired, Literal']
        type_blocks = FunctionSignature.render_type_definitions(sigs, conflicting)
        parts.extend(type_blocks)
        parts.extend(
            td.render_signature(
                'raise NotImplementedError()', is_async=not td.sequential, conflicting_type_names=conflicting
            )
            for td in callable_defs.values()
        )
        return '\n\n'.join(parts)

    @staticmethod
    def _type_check(code: str, *, callable_defs: dict[str, ToolDefinition]) -> None:
        """Type-check a code snippet against tool signatures before execution.

        Uses Monty's stateless type checker with function stubs. Only sound
        when the REPL has no accumulated state (first call or after restart).

        Raises:
            ModelRetry: If the code has type errors or syntax errors.
        """
        stubs = CodeModeToolset._build_type_check_stubs(callable_defs)
        try:
            Monty(code, type_check=True, type_check_stubs=stubs)
        except MontyTypingError as e:
            raise ModelRetry(f'Type error in code:\n{e.display()}') from e
        except MontySyntaxError as e:
            raise ModelRetry(f'Syntax error in code:\n{e.display()}') from e


def _get_sigs_and_conflicting(
    callable_defs: dict[str, ToolDefinition],
) -> tuple[list[FunctionSignature], frozenset[str]]:
    """Extract FunctionSignatures and conflicting type names from tool definitions."""
    sigs: list[FunctionSignature] = []
    for td in callable_defs.values():
        assert td.function_signature is not None, f'function_signature missing for tool {td.name!r}'
        sigs.append(td.function_signature)
    return sigs, FunctionSignature.get_conflicting_type_names(sigs)


async def _execution_loop(
    monty_state: FunctionSnapshot | FutureSnapshot | NameLookupSnapshot | MontyComplete,
    *,
    dispatch: _DispatchFn,
    callable_defs: dict[str, ToolDefinition],
    sanitized_to_original: dict[str, str],
    sequential_tools: set[str],
    global_sequential: bool,
) -> MontyComplete:
    """Drive the Monty REPL via the synchronous snapshot API until completion.

    Uses Monty's `feed_start`/`resume` snapshot API instead of `feed_run_async`
    to avoid background threads and `call_soon_threadsafe`. This makes it safe
    to run inside restricted event loops like Temporal's workflow sandbox.

    Tool calls are handled based on their execution mode:

    - **Parallel tools** (`async def`): deferred via `resume({'future': ...})`
      and eagerly scheduled as `asyncio.Task`s for concurrent execution.
      Resolved at `FutureSnapshot` via `asyncio.gather`.
    - **Sequential tools** (`def`): resolved inline at `FunctionSnapshot`
      via `resume({'return_value': ...})` or `resume({'exception': ...})`. Before
      dispatching, any pending parallel tasks are awaited to maintain ordering.
    - **Global sequential mode** (DBOS/Temporal): all tools are deferred via
      `resume({'future': ...})` but stored as bare coroutines and awaited
      one-at-a-time at `FutureSnapshot` to prevent interleaving.
    """
    pending: dict[int, asyncio.Task[Any] | Coroutine[Any, Any, Any]] = {}
    # Results from parallel tasks that were awaited early (at a sequential-tool
    # barrier) but whose FutureSnapshot hasn't been reached yet.
    pre_resolved: dict[int, ExternalResult] = {}
    try:
        while not isinstance(monty_state, MontyComplete):
            if isinstance(monty_state, NameLookupSnapshot):
                monty_state = monty_state.resume()
            elif isinstance(monty_state, FunctionSnapshot):
                monty_state = await _handle_function_snapshot(
                    monty_state,
                    dispatch,
                    callable_defs,
                    sanitized_to_original,
                    sequential_tools=sequential_tools,
                    global_sequential=global_sequential,
                    pending=pending,
                    pre_resolved=pre_resolved,
                )
            else:
                monty_state = await _resolve_future_snapshot(
                    monty_state,
                    pending=pending,
                    pre_resolved=pre_resolved,
                    global_sequential=global_sequential,
                )
    finally:
        for item in pending.values():  # pragma: no cover
            if isinstance(item, asyncio.Task):
                item.cancel()
            else:
                item.close()

    return monty_state


async def _handle_function_snapshot(
    snapshot: FunctionSnapshot,
    dispatch: _DispatchFn,
    callable_defs: dict[str, ToolDefinition],
    sanitized_to_original: dict[str, str],
    *,
    sequential_tools: set[str],
    global_sequential: bool,
    pending: dict[int, asyncio.Task[Any] | Coroutine[Any, Any, Any]],
    pre_resolved: dict[int, ExternalResult],
) -> FunctionSnapshot | FutureSnapshot | NameLookupSnapshot | MontyComplete:
    """Handle a single FunctionSnapshot from the Monty execution loop."""
    fn_name = snapshot.function_name

    if fn_name not in callable_defs:
        return snapshot.resume({'exception': NameError(f'Unknown function: {fn_name}')})

    if snapshot.args:
        return snapshot.resume(
            {'exception': TypeError(f'{fn_name}() does not accept positional arguments; use keyword arguments')}
        )

    original_name = sanitized_to_original.get(fn_name, fn_name)

    if fn_name in sequential_tools:
        # Per-tool sequential: rendered as `def` (sync), so must resolve inline --
        # the sandbox code doesn't `await` the result. Await pending parallel
        # tasks first (barrier) to maintain ordering.
        for cid in list(pending):
            pre_resolved[cid] = await _resolve_coro(pending.pop(cid))
        outcome = await _resolve_coro(dispatch(original_name, snapshot.kwargs))
        if 'return_value' in outcome:
            return snapshot.resume({'return_value': outcome['return_value']})
        return snapshot.resume({'exception': outcome['exception']})

    # Deferred execution -- store for later resolution at FutureSnapshot.
    if global_sequential:
        # Bare coroutine -- don't schedule on the event loop yet.
        pending[snapshot.call_id] = dispatch(original_name, snapshot.kwargs)
    else:
        # Eagerly schedule as a Task for concurrent execution.
        pending[snapshot.call_id] = asyncio.ensure_future(dispatch(original_name, snapshot.kwargs))
    return snapshot.resume({'future': ...})


async def _resolve_future_snapshot(
    snapshot: FutureSnapshot,
    *,
    pending: dict[int, asyncio.Task[Any] | Coroutine[Any, Any, Any]],
    pre_resolved: dict[int, ExternalResult],
    global_sequential: bool,
) -> FunctionSnapshot | FutureSnapshot | NameLookupSnapshot | MontyComplete:
    """Resolve pending tool calls at a FutureSnapshot."""
    pending_ids = snapshot.pending_call_ids
    if not pending_ids:  # pragma: no cover
        return snapshot.resume(results={})

    results: dict[int, ExternalResult] = {}
    for cid in pending_ids:
        if cid in pre_resolved:
            results[cid] = pre_resolved.pop(cid)
        elif global_sequential:
            results[cid] = await _resolve_coro(pending.pop(cid))

    # Gather remaining parallel tasks.
    gather_ids = [cid for cid in pending_ids if cid not in results]
    if gather_ids:
        tasks = [pending[cid] for cid in gather_ids]
        settled = await asyncio.gather(*tasks, return_exceptions=True)
        for cid in gather_ids:
            del pending[cid]
        for cid, outcome in zip(gather_ids, settled):
            results[cid] = _settle_outcome(outcome)

    return snapshot.resume(results=results)


async def _resolve_coro(
    coro: Coroutine[Any, Any, Any] | asyncio.Task[Any],
) -> ExternalReturnValue | ExternalException:
    """Await a single coroutine/task and wrap the result for Monty."""
    try:
        result = await coro
    except Exception as exc:
        return ExternalException(exception=exc)
    else:
        return ExternalReturnValue(return_value=result)


def _settle_outcome(outcome: Any) -> ExternalReturnValue | ExternalException:
    """Wrap an `asyncio.gather(return_exceptions=True)` outcome for Monty."""
    if isinstance(outcome, Exception):
        return ExternalException(exception=outcome)
    if isinstance(outcome, BaseException):  # pragma: no cover
        raise outcome
    return ExternalReturnValue(return_value=outcome)


def _prepend_prints(error_message: str, capture: _PrintCapture) -> str:
    """Prepend any captured print output to an error message.

    When sandbox code prints debug output before crashing, this preserves
    that output in the error so the model can use it for debugging.
    """
    printed = capture.joined.rstrip('\n')
    if not printed:
        return error_message
    return f'[stdout before error]\n{printed}\n[/stdout before error]\n{error_message}'


def _contains_multimodal(value: Any) -> bool:
    """Check if a value is or directly contains multimodal content (images, audio, etc.)."""
    if is_multi_modal_content(value):
        return True
    if isinstance(value, list):
        return any(is_multi_modal_content(item) for item in value)  # pyright: ignore[reportUnknownVariableType]
    return False


class _PrintCapture:
    """Accumulates print-callback chunks from the Monty REPL.

    Pulled out to module scope (rather than a closure inside `call_tool`) so
    the callback path is testable in isolation and visible to coverage.py.
    """

    def __init__(self) -> None:
        self._chunks: list[str] = []

    def __call__(self, _stream: str, text: str) -> None:
        self._chunks.append(text)

    @property
    def joined(self) -> str:
        return ''.join(self._chunks)
