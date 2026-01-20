"""Code mode toolset that wraps tools as Python functions callable from generated code."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, cast

import monty
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from .._run_context import AgentDepsT, RunContext
from .._signature_from_schema import signature_from_function, signature_from_schema
from ..tools import ToolDefinition

from .abstract import SchemaValidatorProt, ToolsetTool
from .function import FunctionToolset, FunctionToolsetTool
from .wrapper import WrapperToolset


class _CodeToolArguments(TypedDict):
    code: str


_CODE_ADAPTER = TypeAdapter(_CodeToolArguments)
_CODE_MODE_TOOL_NAME = 'run_code'


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
            return signature_from_function(
                original_tool.function,
                name=tool_name,
                description=tool.tool_def.description,
                include_return_type=True,  # Always show return types in code mode
            )

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
    """A toolset that exposes wrapped tools as callable Python functions in a code execution context."""

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        wrapped_tools = await super().get_tools(ctx)
        # TODO: MCP tool names can include characters that aren't valid Python identifiers.
        # We should sanitize names and maintain a mapping back to the original tool name.
        available_functions = [_get_tool_signature(tool) for tool in wrapped_tools.values()]

        # Debug: show what signatures the model will see
        print(f'\n{"="*60}\nCODE MODE - Tool Signatures shown to model:\n{"="*60}')
        for sig in available_functions:
            print(sig)
        print(f'{"="*60}\n')

        # TODO: This dumps all tool signatures up-front, which can bloat context for large toolsets.
        # Example: hundreds of MCP tools can push tens of thousands of tokens into the prompt,
        # defeating the progressive-disclosure approach described in code-mode references.
        # Consider: progressive discovery (list tool names first, fetch signatures on demand).
        description = (
            """\
Write Python code to accomplish the ENTIRE task in a SINGLE code block.

CRITICAL:
- Solve the complete task in ONE execution - do not break it into multiple steps
- Do not write exploratory code that just fetches data and returns - process it fully
- Use loops to handle multiple items (e.g., for each user, fetch their orders and aggregate)
- The last expression evaluated becomes the return value - make it the final answer

Syntax restrictions:
- No imports allowed - use only the functions provided below
- No generator expressions (e.g., `sum(x for x in items)`) - use explicit for loops
- Initialize numeric accumulators with 0.0 (float) not 0 (int)
- Access dict fields with brackets: result["field"], not result.field

Available functions:

```python
"""
            + '\n\n'.join(available_functions)
            + """
```

Example - completing a full aggregation task in one execution:
```python
# Get all items and process them completely in one go
items = get_items(category="electronics")

# Aggregate data using a loop
total = 0.0
for item in items:
    details = get_item_details(id=item["id"])
    if details["status"] == "active":
        total += details["price"]

f"Total value of active items: ${total}"
```"""
        )
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
                max_retries=3,
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

        # Log the generated code for debugging visibility
        print(f'\n{"="*60}\nCODE MODE - Generated Code:\n{"="*60}\n{code}\n{"="*60}\n')

        # TODO: There is no execution timeout or step budget for Monty runs here.
        # Example: `for _ in range(10**9): pass` can hang the agent run indefinitely.
        # TODO: Monty supports a limited Python subset (no while loops, list comprehensions, lambdas).
        # Consider documenting supported syntax so we do not gen incorrect code.
        # TODO: Monty errors should be caught and returned to the model for retry, allowing the model
        # to fix syntax errors (e.g., unsupported generator expressions) in a subsequent attempt.
        try:
            m = monty.Monty(code, external_functions=list(tool.original_tools.keys()))
            result = m.start()
        except monty.MontyRuntimeError as e:
            print(f'\n{"!"*60}\nCODE MODE - Monty Parse/Start Error:\n{e}\n{"!"*60}\n')
            raise

        call_count = 0
        while isinstance(result, monty.MontySnapshot):
            call_count += 1
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

            # Log inner tool call
            print(f'  [{call_count}] Calling: {tool_name}({tool_kwargs})')

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

            span_name = f'code_mode_tool:{tool_name}'
            with ctx.tracer.start_as_current_span(span_name, attributes=span_attributes):
                tool_return_value = await super().call_tool(tool_name, tool_kwargs, inner_ctx, original_tool)

            # Log return value
            print(f'      -> {tool_return_value}')

            try:
                result = result.resume(return_value=tool_return_value)
            except monty.MontyRuntimeError as e:
                print(f'\n{"!"*60}\nCODE MODE - Monty Runtime Error after {tool_name} returned:\n')
                print(f'  Return value was: {tool_return_value}')
                print(f'  Error: {e}\n{"!"*60}\n')
                raise

        # Log final output
        print(f'\n{"="*60}\nCODE MODE - Final Output:\n{"="*60}\n{result.output}\n{"="*60}\n')

        return result.output
