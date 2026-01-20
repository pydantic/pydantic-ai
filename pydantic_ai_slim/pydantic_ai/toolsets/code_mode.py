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
from .function import FunctionToolset, FunctionToolsetTool  # circular import?
from .wrapper import WrapperToolset


class _CodeToolArguments(TypedDict):
    code: str


_CODE_ADAPTER = TypeAdapter(_CodeToolArguments)
_CODE_MODE_TOOL_NAME = 'run_code'  # TODO: Does this need to be configurable?


@dataclass(kw_only=True)
class _CodeModeTool(ToolsetTool[AgentDepsT]):
    original_tools: dict[str, ToolsetTool[AgentDepsT]]


def _get_tool_signature(tool: ToolsetTool[Any]) -> str:
    """Get a Python signature string for a tool.

    For native function tools, uses the original function's signature.
    For external tools (MCP, etc.), converts the JSON schema to a signature.
    """
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
        description = (
            """\
You can generate arbitrary python code that doesn't use imports.
The last evaluated expression will be the return value of the tool.

The following functions are available as local variables in the interpreter:

```python
"""
            + '\n\n'.join(available_functions)
            + '\n```'
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
                max_retries=3,  # TODO: ?? Should this be user-controlled in some way?
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

        m = monty.Monty(code, external_functions=list(tool.original_tools.keys()))
        result = m.start()
        while isinstance(result, monty.MontySnapshot):
            tool_name = result.function_name
            original_tool = tool.original_tools[tool_name]

            tool_kwargs = dict(result.kwargs)
            if result.args:
                param_names = list(original_tool.tool_def.parameters_json_schema.get('properties', {}).keys())
                for i, arg in enumerate(result.args):
                    if i < len(param_names):
                        tool_kwargs[param_names[i]] = arg

            ctx = replace(ctx, tool_name=tool_name)
            tool_return_value = await super().call_tool(tool_name, tool_kwargs, ctx, original_tool)
            result = result.resume(return_value=tool_return_value)
        return result.output
