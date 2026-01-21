"""Code mode toolset that wraps tools as Python functions callable from generated code."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, cast

import monty
import asyncio
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from .._run_context import AgentDepsT, RunContext
from .._signature_from_schema import signature_from_function, signature_from_schema
from ..exceptions import ModelRetry
from ..tools import ToolDefinition
from .abstract import SchemaValidatorProt, ToolsetTool
from .function import FunctionToolset, FunctionToolsetTool
from .wrapper import WrapperToolset

__all__ = (
    'CodeModeToolset',
    'build_code_mode_prompt',
    'signature_from_function',
    'signature_from_schema',
)


class _CodeToolArguments(TypedDict):
    code: str


_CODE_ADAPTER = TypeAdapter(_CodeToolArguments)
_CODE_MODE_TOOL_NAME = 'run_code'


def build_code_mode_prompt(*, signatures: list[str]) -> str:
    """Build the default code mode prompt with the given tool signatures.

    This is the default prompt builder used by CodeModeToolset. Users can provide
    their own prompt_builder callback to customize the prompt entirely.

    Args:
        signatures: List of Python function signatures for available tools.

    Returns:
        The complete prompt describing code mode capabilities and available functions.
    """
    functions_block = '\n\n'.join(signatures)
    # TODO: The first line of the prompt should be customizeable by the user using Prompt Templates
    return f"""\
You should consider writing Python code to accomplish multiple tasks in one go instead of using multiple tools one by one.

CRITICAL:
- Use for loops to handle multiple items (e.g., for each user, fetch their orders and aggregate)
- The last expression evaluated becomes the return value - make it the final answer

Syntax restrictions (the runtime uses a restricted Python subset):
- No imports - use only the provided functions and builtins (len, sum, str, etc.)
- No while loops - use for loops instead
- No comprehensions (list/dict/set) or generator expressions - use explicit for loops
- No lambdas - define logic inline
- No tuple unpacking (e.g., `a, b = 1, 2`) - assign variables separately
- No list index assignment (e.g., `lst[0] = x`) - use list.append() to build lists
- No string methods (.join, .split, .upper, etc.) - return data structures, not formatted strings

What DOES work:
- Dict assignment: `d["key"] = value`
- Dict methods: `.get()`, `.keys()`, `.values()`, `.items()`
- List methods: `.append()`
- F-strings: `f"value is {{x}}"`
- Builtins: `len()`, `sum()`, `str()`, `list()`, `range()`

Available functions:

```python
{functions_block}
```

Example - completing a full aggregation task in one execution:
```python
items = get_items(category="electronics")
results = []
total = 0

for item in items:
    details = get_item_details(id=item["id"])
    if details["status"] == "active":
        total += details["price"]
        results.append({{"name": item["name"], "price": details["price"]}})

{{"total": total, "count": len(results), "items": results}}
```"""


def _build_type_check_prefix(signatures: list[str]) -> str:
    """Build prefix code with imports and tool signatures for Monty type checking."""
    imports = 'from typing import Any, TypedDict, NotRequired, Literal\n\n'
    return imports + '\n\n'.join(signatures)


@dataclass(kw_only=True)
class _CodeModeTool(ToolsetTool[AgentDepsT]):
    original_tools: dict[str, ToolsetTool[AgentDepsT]]


def _get_tool_signature(tool: ToolsetTool[Any]) -> str:
    """Get a Python signature string for a tool.

    For native function tools, uses the original function's signature (including return type).
    For external tools (MCP, etc.), converts the JSON schema to a signature.

    Note: Code mode always includes return types because the model needs to know
    what structure each function returns to write correct code.
    """
    # Code mode MUST show return types - without them the model can't know
    # that get_weather() returns a dict with 'temperature' key vs just a number.
    # We ignore tool.include_return_schema here because code mode has different needs
    # than traditional tool calling.
    # TODO: For native function tools, we call signature_from_function which uses
    # inspect.signature() and get_type_hints() every time get_tools() is called.
    # This re-inspection is needed to filter out RunContext params and format the signature,
    # but could be cached/precomputed when the tool is registered. This becomes non-optimal
    # when: (1) get_tools() is called frequently during an agent run, (2) there are many
    # tools in the toolset, (3) the toolset is reused across multiple agent runs.
    # Consider storing the formatted signature string on FunctionToolsetTool at registration time.
    if isinstance(tool, FunctionToolsetTool) and isinstance(tool.toolset, FunctionToolset):
        tool_name = tool.tool_def.name
        if tool_name in tool.toolset.tools:
            original_tool = tool.toolset.tools[tool_name]
            result = signature_from_function(
                original_tool.function,
                name=tool_name,
                description=tool.tool_def.description,
                include_return_type=True,  # Always show return types in code mode
            )
            if result.typeddict_defs:
                return '\n\n'.join(result.typeddict_defs) + '\n\n' + result.signature
            return result.signature

    # For external tools (MCP, etc.), convert JSON schema to signature
    result = signature_from_schema(
        name=tool.tool_def.name,
        parameters_json_schema=tool.tool_def.parameters_json_schema,
        description=tool.tool_def.description,
        return_json_schema=tool.tool_def.return_schema,  # Always include if available
        namespace_defs=True,
    )

    if result.typeddict_defs:
        return '\n\n'.join(result.typeddict_defs) + '\n\n' + result.signature
    return result.signature


@dataclass(kw_only=True)
class CodeModeToolset(WrapperToolset[AgentDepsT]):
    """A toolset that exposes wrapped tools as callable Python functions in a code execution context.

    Args:
        wrapped: The underlying toolset to wrap.
        prompt_builder: Optional callback to build a custom prompt. If not provided,
            uses `build_code_mode_prompt`. The callback receives `signatures` as a
            keyword argument containing the list of Python function signatures.
    """

    prompt_builder: Callable[..., str] = build_code_mode_prompt
    _cached_signatures: list[str] = field(default_factory=list, init=False, repr=False)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        wrapped_tools = await super().get_tools(ctx)
        # TODO: MCP tool names can include characters that aren't valid Python identifiers.
        # We should sanitize names and maintain a mapping back to the original tool name.
        available_functions = [_get_tool_signature(tool) for tool in wrapped_tools.values()]
        self._cached_signatures = available_functions

        # TODO: This dumps all tool signatures up-front, which can bloat context for large toolsets.
        # Example: hundreds of MCP tools can push tens of thousands of tokens into the prompt,
        # defeating the progressive-disclosure approach described in code-mode references.
        # Consider: progressive discovery (list tool names first, fetch signatures on demand).
        description = self.prompt_builder(signatures=available_functions)
        # TODO: Ideally we'd use kind='output' to make the code result be the final answer
        # without a second LLM call. However, output tools are treated differently by models -
        # they expect to provide structured output directly, not execute code. We need a way
        # to have a function tool whose result becomes the final output without another LLM call.
        return {
            _CODE_MODE_TOOL_NAME: _CodeModeTool(
                toolset=self,
                original_tools=wrapped_tools,
                tool_def=ToolDefinition(
                    name=_CODE_MODE_TOOL_NAME,
                    parameters_json_schema=_CODE_ADAPTER.json_schema(),
                    description=description,
                ),
                max_retries=3, # -> Should allow to be overrideable? 3 tries are plenty but not sure if a specially dumb model might need more attempts to get the code right?
                args_validator=cast(SchemaValidatorProt, _CODE_ADAPTER.validator),
            )
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        code = tool_args['code']
        assert name == _CODE_MODE_TOOL_NAME
        assert isinstance(tool, _CodeModeTool)
        assert isinstance(code, str)


        #
        #
        # Adding this to mitigate potential infinite loops or resource exhaustion
        # Or maybe just something dubious by the model which could hang this
        #
        monty_limits = monty.ResourceLimits(
            max_duration_secs=60
        )

        try:
            m = monty.Monty(code, external_functions=list(tool.original_tools.keys()))
            # Type check the code before execution
            prefix = _build_type_check_prefix(self._cached_signatures)
            m.type_check(prefix_code=prefix)
            result = m.start(limits=monty_limits)
        except monty.MontyTypingError as e:
            error_msg = e.display('concise')
            raise ModelRetry(f'Type error in generated code:\n{error_msg}')
        except monty.MontySyntaxError as e:
            error_msg = e.display()
            raise ModelRetry(f'Syntax error in generated code:\n{error_msg}')
        except monty.MontyRuntimeError as e:
            error_msg = e.display('traceback')
            raise ModelRetry(f'Runtime error in generated code:\n{error_msg}')

        while isinstance(result, monty.MontySnapshot):
            tool_name = result.function_name
            original_tool = tool.original_tools[tool_name]

            tool_kwargs = dict(result.kwargs)
            if result.args:
                # TODO: Positional args currently map using JSON schema property order, which may
                # not match the tool's intended parameter order (especially for MCP tools). This
                # can silently swap arguments; consider enforcing keyword-only or deriving a stable
                # ordering from the tool source.
                param_names = list(original_tool.tool_def.parameters_json_schema.get('properties', {}).keys())
                for i, arg in enumerate(result.args):
                    if i < len(param_names):
                        tool_kwargs[param_names[i]] = arg
            inner_ctx = replace(ctx, tool_name=tool_name)

            # TODO: Approval/defer flows are handled by the outer tool call, so inner tool approvals
            # (ApprovalRequired/CallDeferred) currently fail the entire run_code call instead of
            # surfacing a per-tool approval step. Cloudflare-style proxying would allow pausing after
            # each tool call. Example:
            #   run_code: "result = dangerous_tool(action='delete')"
            # Ideally yields: a pending approval for dangerous_tool, rather than a failed run_code call.
            # Create span for inner tool call with code_mode.inner_tool attribute
            span_attributes = {
                'gen_ai.tool.name': tool_name,
                'code_mode.inner_tool': True,
                'logfire.msg': f'code mode calling: {tool_name}',
            }

            # TODO: Consider moving this to tool manager(Discussion with Douwe)?
            span_name = f'code_mode_tool:{tool_name}'
            with ctx.tracer.start_as_current_span(span_name, attributes=span_attributes):
                tool_return_value = await super().call_tool(tool_name, tool_kwargs, inner_ctx, original_tool)

            try:
                result = result.resume(return_value=tool_return_value)
            except monty.MontyRuntimeError as e:
                error_msg = e.display('traceback')
                raise ModelRetry(f'Runtime error in generated code:\n{error_msg}')

        return result.output
