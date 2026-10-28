"""Toolset behind `IPythonCodeMode`: one `run_code` tool that runs cells in a persistent IPython kernel."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, NotRequired, Self

from pydantic import Field, TypeAdapter, ValidationError
from typing_extensions import TypedDict

from pydantic_ai import AbstractToolset, RunContext, ToolDefinition, WrapperToolset
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry, UsageLimitExceeded, UserError
from pydantic_ai.function_signature import FunctionSignature
from pydantic_ai.messages import ToolCallPart, ToolReturn, ToolReturnContent, ToolReturnPart, UserContent
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import AgentDepsT, ToolSelector, matches_tool_selector
from pydantic_ai.toolsets.abstract import SchemaValidatorProt, ToolsetTool
from pydantic_ai_harness._code_tools import (
    can_be_sandboxed,
    contains_multimodal,
    partition_callable_tools,
    record_nested_call,
    run_nested_call,
    signatures_and_conflicts,
)
from pydantic_ai_harness.ipython_code_mode._kernel import CellResult, IPythonKernel

RUN_CODE_TOOL_NAME = 'run_code'


class _RunCodeArguments(TypedDict):
    code: Annotated[str, Field(description='The Python code to run in the IPython kernel.')]
    restart: NotRequired[
        Annotated[
            bool,
            Field(description='Set to true to restart the kernel, clearing its state, before running the code.'),
        ]
    ]


class _FunctionCall(TypedDict):
    name: str
    args: dict[str, Any]


_RUN_CODE_ADAPTER = TypeAdapter(_RunCodeArguments)
_RUN_CODE_JSON_SCHEMA = _RUN_CODE_ADAPTER.json_schema()
_RUN_CODE_ARGS_VALIDATOR: SchemaValidatorProt = _RUN_CODE_ADAPTER.validator  # pyright: ignore[reportAssignmentType]
_FUNCTION_CALL_ADAPTER = TypeAdapter(_FunctionCall)
_TOOL_RETURN_CONTENT_TA: TypeAdapter[Any] = TypeAdapter(ToolReturnContent)

_DESCRIPTION_HEAD = """\
Run Python code in a persistent IPython kernel.

The kernel is a full CPython interpreter: any installed package can be imported, and code runs with \
the host process's permissions, including its filesystem and network access. Variables, imports, \
functions, and loaded data persist between calls. IPython syntax works, including magics such as \
`%timeit`, shell escapes such as `!ls`, and top-level `await`. `input()` returns an empty string. \
Set `restart: true` to restart the kernel and clear its state before the code runs."""

_DESCRIPTION_TAIL = """\
The last expression's value is automatically captured as the return value -- you do **not** need to \
`print()` it. A final expression that evaluates to `None` is treated as no result. Values with a JSON \
form are returned as data; other values are returned as their `repr`, and images (for example \
`matplotlib` figures or PIL images) are returned as images. Without a result, printed output, or \
displayed images, `run_code` returns `{}`.

Returns a non-`None` last expression's value directly when nothing is printed or displayed. With \
printed output and no result, returns `{"output": "<printed text>"}`. With printed output and a plain \
result, returns `{"output": "<printed text>", "result": <last expression>}`. With displayed images, \
returns a list: the printed text if any, then the images, then the result if any. With printed output \
and an image result, returns a list of the printed text and the image.\
"""

_FUNCTIONS_HEADER = (
    'The following functions are already defined in the kernel; call them by name without redefining '
    'or importing them. All parameters are keyword-only. They are synchronous, so call them directly, '
    'e.g. `tool_name(arg=value)`, without `await`. Calls run one at a time. A failed call raises '
    "`ToolCallError` with the tool's error message."
)


def _description(callable_defs: dict[str, ToolDefinition], timeout: float | None) -> str:
    sections = [_DESCRIPTION_HEAD]
    if timeout is not None:
        sections.append(
            f'Each call may run for {timeout:g} seconds, not counting time spent in the functions below; '
            'code still running then is interrupted with `KeyboardInterrupt`.'
        )
    sections.append(_DESCRIPTION_TAIL)
    if callable_defs:
        sigs, conflicting = signatures_and_conflicts(callable_defs)
        type_blocks = FunctionSignature.render_type_definitions(sigs, conflicting)
        function_blocks = [
            td.render_signature('...', is_async=False, conflicting_type_names=conflicting)
            for td in callable_defs.values()
        ]
        sections.append(_FUNCTIONS_HEADER)
        if type_blocks:
            sections.append('```python\n' + '\n\n'.join(type_blocks) + '\n```')
        sections.append('```python\n' + '\n\n'.join(function_blocks) + '\n```')
    return '\n\n'.join(sections)


@dataclass(kw_only=True)
class _RunCodeTool(ToolsetTool[AgentDepsT]):
    """The `run_code` tool, with the step's function catalog computed in `get_tools`."""

    callable_defs: dict[str, ToolDefinition]
    sanitized_to_original: dict[str, str]
    wrapped_tools: dict[str, ToolsetTool[AgentDepsT]]


@dataclass
class _Cell:
    """The nested calls one `run_code` call makes, recorded the way `CodeMode` records them."""

    parent_tool_call_id: str
    tool: _RunCodeTool[Any]
    tool_manager: ToolManager[Any]
    max_tool_calls: int
    call_count: int = 0
    calls: dict[str, ToolCallPart] = field(default_factory=dict[str, ToolCallPart])
    returns: dict[str, ToolReturnPart] = field(default_factory=dict[str, ToolReturnPart])
    user_content: dict[str, list[UserContent]] = field(default_factory=dict[str, list[UserContent]])
    usage_limit_exceeded: UsageLimitExceeded | None = None

    async def call_function(self, request: str) -> str:
        """Run the tool behind one function call from the kernel and return the JSON reply."""
        try:
            call = _FUNCTION_CALL_ADAPTER.validate_json(request)
        # Only code that forges a request with the kernel's private token gets here.
        except ValidationError:  # pragma: no cover
            return json.dumps({'error': 'Malformed function call.'})
        name = call['name']
        if name not in self.tool.callable_defs:
            return json.dumps({'error': f'{name!r} is no longer an available function.'})
        if self.call_count >= self.max_tool_calls:
            return json.dumps(
                {
                    'error': f'Code mode allows {self.max_tool_calls} nested tool calls per `run_code` call '
                    'and this code asked for more. Call fewer tools, for example by filtering the inputs '
                    'first, or split the work across several `run_code` calls.'
                }
            )
        self.call_count += 1
        tool_call_id = f'{self.parent_tool_call_id}__{self.call_count}'
        # Reserved before the call runs, so `ToolReturn.content` reaches the model in call order.
        self.user_content[tool_call_id] = []
        call_part = ToolCallPart(
            tool_name=self.tool.sanitized_to_original.get(name, name), args=call['args'], tool_call_id=tool_call_id
        )
        outcome = await run_nested_call(self.tool_manager, call_part)
        record_nested_call(call_part, outcome, calls=self.calls, returns=self.returns, user_content=self.user_content)
        if outcome.error is not None:
            if outcome.usage_limit_exceeded is not None:
                self.usage_limit_exceeded = outcome.usage_limit_exceeded
            return json.dumps({'error': f'{type(outcome.error).__name__}: {outcome.error}'})
        return '{"value": ' + _TOOL_RETURN_CONTENT_TA.dump_json(outcome.content, warnings=False).decode() + '}'

    def tool_return(self, cell: CellResult) -> ToolReturn[Any]:
        """Build the `run_code` return from the cell's output, mirroring `CodeMode`'s shape."""
        output, result, images = cell.output, cell.result, cell.images
        return_value: Any
        if images:
            return_value = [*([output] if output else []), *images, *([] if result is None else [result])]
        elif not output:
            return_value = {} if result is None else result
        elif result is None:
            return_value = {'output': output}
        elif contains_multimodal(result):
            return_value = [output, result]
        else:
            return_value = {'output': output, 'result': result}
        return ToolReturn(
            return_value=return_value,
            content=[item for items in self.user_content.values() for item in items] or None,
            metadata={'code_mode': True, 'tool_calls': self.calls, 'tool_returns': self.returns},
        )


@dataclass
class IPythonCodeModeToolset(WrapperToolset[AgentDepsT]):
    """Implementation toolset for the `IPythonCodeMode` capability.

    Exposes a single `run_code` tool alongside the tools that stay native. Tools picked by
    `tool_selector` are defined as functions in the kernel and listed in `run_code`'s description.
    """

    tool_selector: ToolSelector[AgentDepsT] = 'all'
    """Which wrapped tools to define as kernel functions. Non-matching tools are exposed as native tools."""

    max_retries: int = 3
    """Maximum number of retries for the `run_code` tool; a cell that raises counts as one."""

    max_tool_calls: int = field(default=100, kw_only=True)
    """Maximum function calls one `run_code` call may make; calls past it raise in the kernel."""

    timeout: float | None = field(default=60.0, kw_only=True)
    """Seconds one `run_code` call may run, not counting time spent in tool calls; `None` for no limit."""

    kernel: IPythonKernel | None = field(default=None, kw_only=True)
    """A kernel to keep across runs; unset, each run starts its own kernel on first use and stops it at the end."""

    capability: AbstractCapability[AgentDepsT] | None = field(default=None, kw_only=True, repr=False)
    """The run's `IPythonCodeMode` instance, so `run_code` is attributed to it."""

    _run_kernel: IPythonKernel | None = field(default=None, init=False, repr=False, compare=False)
    _owns_kernel: bool = field(default=False, init=False, repr=False, compare=False)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Return a fresh toolset instance for this agent run, so concurrent runs don't share an owned kernel.

        Refuses durable execution here, before any code runs: an engine replays a workflow's code on
        recovery, and a kernel's state can't be rebuilt that way.
        """
        if ctx.in_durable_context:
            raise UserError(
                "`IPythonCodeMode` is not supported under durable execution: a kernel's state cannot be replayed."
            )
        return replace(self, wrapped=await self.wrapped.for_run(ctx))

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Update the wrapped toolset for this step while keeping the run's kernel."""
        new_wrapped = await self.wrapped.for_run_step(ctx)
        if new_wrapped is self.wrapped:
            return self
        new_self = replace(self, wrapped=new_wrapped)
        new_self._run_kernel = self._run_kernel
        return new_self

    async def __aenter__(self) -> Self:
        """Enter the wrapped toolset and pick the run's kernel; a kernel this run owns starts on first use."""
        if self.max_tool_calls < 1:
            raise UserError('`max_tool_calls` must be at least 1')
        if self.timeout is not None and not self.timeout > 0:
            raise UserError('`timeout` must be a positive number of seconds, or `None` for no limit')
        await self.wrapped.__aenter__()
        self._owns_kernel = self.kernel is None
        self._run_kernel = IPythonKernel() if self.kernel is None else self.kernel
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        """Exit the wrapped toolset, then shut down the kernel if this run started it."""
        kernel, self._run_kernel = self._run_kernel, None
        try:
            return await self.wrapped.__aexit__(*args)
        finally:
            if kernel is not None and self._owns_kernel:
                await kernel.shutdown()

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Return the `run_code` tool plus the tools that stay native."""
        wrapped_tools = await self.wrapped.get_tools(ctx)
        in_kernel: dict[str, ToolsetTool[AgentDepsT]] = {}
        native: dict[str, ToolsetTool[AgentDepsT]] = {}
        for name, tool in wrapped_tools.items():
            tool_def = tool.tool_def
            if (
                can_be_sandboxed(tool_def)
                and ctx.is_tool_available(tool_def)
                and await matches_tool_selector(self.tool_selector, ctx, tool_def)
            ):
                in_kernel[name] = tool
            else:
                native[name] = tool
        if RUN_CODE_TOOL_NAME in native:
            raise UserError(
                f"Tool name '{RUN_CODE_TOOL_NAME}' is reserved for code mode. Rename your tool to avoid conflicts."
            )
        callable_defs, sanitized_to_original = partition_callable_tools(
            in_kernel, owner='IPythonCodeMode', reserved_name=RUN_CODE_TOOL_NAME
        )
        capability_id = (
            None
            if self.capability is None
            else next((run_id for run_id, cap in ctx.capabilities.items() if cap is self.capability), None)
        )
        result: dict[str, ToolsetTool[AgentDepsT]] = dict(native)
        result[RUN_CODE_TOOL_NAME] = _RunCodeTool(
            toolset=self,
            tool_def=ToolDefinition(
                name=RUN_CODE_TOOL_NAME,
                description=_description(callable_defs, self.timeout),
                parameters_json_schema=_RUN_CODE_JSON_SCHEMA,
                metadata={'code_arg_name': 'code', 'code_arg_language': 'python'},
                sequential=True,
                capability_id=capability_id,
            ),
            max_retries=self.max_retries,
            args_validator=_RUN_CODE_ARGS_VALIDATOR,
            callable_defs=callable_defs,
            sanitized_to_original=sanitized_to_original,
            wrapped_tools=wrapped_tools,
        )
        return result

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        """Run the code in the kernel, or pass the call through to a native tool."""
        if not isinstance(tool, _RunCodeTool):
            return await self.wrapped.call_tool(name, tool_args, ctx, tool)
        kernel = self._run_kernel
        assert kernel is not None, '`IPythonCodeModeToolset` must be entered before calling `run_code`'
        if tool_args.get('restart', False):
            await kernel.restart()
        parent_tm = ctx.tool_manager
        assert parent_tm is not None, '`IPythonCodeModeToolset` requires `ctx.tool_manager` to be set'
        cell = _Cell(
            parent_tool_call_id=ctx.tool_call_id or 'pyd_ai_ipython_code_mode',
            tool=tool,
            # Nested calls go through the standard validation and execution path, with the agent's
            # capability hooks, against the wrapped toolset's tools.
            tool_manager=ToolManager(
                toolset=self.wrapped, root_capability=parent_tm.root_capability, ctx=ctx, tools=tool.wrapped_tools
            ),
            max_tool_calls=self.max_tool_calls,
        )
        result = await kernel._execute(  # pyright: ignore[reportPrivateUsage]
            tool_args['code'],
            functions=list(tool.callable_defs),
            call_function=cell.call_function,
            timeout=self.timeout,
        )
        if cell.usage_limit_exceeded is not None:
            raise cell.usage_limit_exceeded
        if result.outcome == 'ok':
            return cell.tool_return(result)
        raise ModelRetry(self._failure_message(result))

    def _failure_message(self, cell: CellResult) -> str:
        output = f'{cell.output}\n' if cell.output else ''
        if cell.outcome == 'error':
            return f'Runtime error:\n{output}{cell.error}'
        if cell.outcome == 'interrupted':
            details = '\n'.join(part for part in (cell.output.rstrip('\n'), cell.error) if part)
            return (f'{details}\n\n' if details else '') + (
                f'The code ran longer than the {self.timeout:g}-second timeout and was interrupted. '
                'The kernel and its state are kept. Make the code do less work per call.'
            )

        if cell.outcome == 'restarted':
            return (
                f'The code ran longer than the {self.timeout:g}-second timeout and did not stop when interrupted, '
                'so the kernel was shut down and its state is gone. The next call starts a new kernel: re-run any '
                'imports, recreate any state you need, and make the code do less work per call.'
            )
        return (
            'The kernel process exited, for example because the code crashed it or called `os._exit()`, '
            'so its state is gone. The next call starts a new kernel: re-run any imports and recreate any '
            'state you need.'
        )
