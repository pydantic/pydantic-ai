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
    """
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
            )

    result = signature_from_schema(
        name=tool.tool_def.name,
        parameters_json_schema=tool.tool_def.parameters_json_schema,
        description=tool.tool_def.description,
        return_json_schema=tool.tool_def.return_schema,
    )

    if result.typeddict_defs:
        return '\n\n'.join(result.typeddict_defs) + '\n\n' + result.signature
    return result.signature


@dataclass(kw_only=True)
class CodeModeToolset(WrapperToolset[AgentDepsT]):
    """A toolset that exposes wrapped tools as callable Python functions in a code execution context."""

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        wrapped_tools = await super().get_tools(ctx)
        available_functions = [_get_tool_signature(tool) for tool in wrapped_tools.values()]
        # TODO: This dumps all tool signatures up-front, which can bloat context for large toolsets.
        # Example: hundreds of MCP tools can push tens of thousands of tokens into the prompt,
        # defeating the progressive-disclosure approach described in code-mode references.
        # Consider: progressive discovery (list tool names first, fetch signatures on demand).
        description = (
            """\
You can generate arbitrary python code that doesn't use imports.
The last evaluated expression will be the return value of the tool.

The following functions are available as local variables in the interpreter:

```python
"""
            + '\n\n'.join(available_functions)
            + """
```

Example:
```python
# Fetch data, process, return result
result1 = get_data("item1")
result2 = get_data("item2")
combined = [result1, result2]
f"Found {len(combined)} items"
```"""
        )
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

        # TODO: There is no execution timeout or step budget for Monty runs here.
        # Example: `while True: pass` will hang the agent run indefinitely.
        m = monty.Monty(code, external_functions=list(tool.original_tools.keys()))
        result = m.start()
        while isinstance(result, monty.MontySnapshot):
            tool_name = result.function_name
            original_tool = tool.original_tools[tool_name]

            tool_kwargs = dict(result.kwargs)
            if result.args:
                # TODO: Positional args rely on JSON schema property order, which may not match
                # actual tool parameter ordering for external tools.
                # Example: schema properties are {'b': ..., 'a': ...} but tool expects (a, b),
                # so calling tool(1, 2) maps b=1, a=2 and silently swaps inputs.
                param_names = list(original_tool.tool_def.parameters_json_schema.get('properties', {}).keys())
                for i, arg in enumerate(result.args):
                    if i < len(param_names):
                        tool_kwargs[param_names[i]] = arg

            inner_ctx = replace(ctx, tool_name=tool_name)

            # Create span for inner tool call with code_mode.inner_tool attribute
            span_attributes = {
                'gen_ai.tool.name': tool_name,
                'code_mode.inner_tool': True,
                'logfire.msg': f'code mode calling: {tool_name}',
            }
            span_name = f'code_mode_tool:{tool_name}'
            with ctx.tracer.start_as_current_span(span_name, attributes=span_attributes):
                tool_return_value = await super().call_tool(tool_name, tool_kwargs, inner_ctx, original_tool)

            result = result.resume(return_value=tool_return_value)
        return result.output
