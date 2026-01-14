"""TODO:
* There's no way to distinguish between dataclasses and toolsets in what we include in the tool description
* Probably want to show the model actual python function signatures in the code tool description, rather than JSON schemas etc.
* Might want/need to do something closer to calling the underlying tool when possible, rather than using the tool calling infrastructure (though that is necessary for MCP/etc.)
* Need to expose the annotated return type of the tool to the model; apparently Aditya has some work in progress for this, though it may be unnecessary if we can retain a reference to the underlying function and reference its signature
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, TypedDict

import monty
from pydantic import TypeAdapter

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from .abstract import ToolsetTool
from .wrapper import WrapperToolset


class _CodeToolArguments(TypedDict):
    code: str


_CODE_ADAPTER = TypeAdapter(_CodeToolArguments)
_CODE_MODE_TOOL_NAME = 'run_code'  # TODO: Does this need to be configurable?


@dataclass(kw_only=True)
class _CodeModeTool(ToolsetTool[AgentDepsT]):
    original_tools: dict[str, ToolsetTool[AgentDepsT]]


@dataclass(kw_only=True)
class CodeModeToolset(WrapperToolset[AgentDepsT]):
    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        wrapped_tools = await super().get_tools(ctx)
        available_functions = [
            f'name={x.tool_def.name!r} description={x.tool_def.description!r} parameters_json_schema={x.tool_def.parameters_json_schema!r}'
            for x in wrapped_tools.values()
        ]
        description = """\
You can generate arbitrary python code that doesn't use imports. 
The last evaluated expression will be the return value of the tool.

The following functions are available as local variables in the interpreter. Even though I'm giving you a JSON schema for the parameters, the top-level properties should be treated as keyword arguments.
""" + '\n\n'.join(available_functions)
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
                args_validator=_CODE_ADAPTER.validator,
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
            assert not result.args
            tool_name = result.function_name
            original_tool = tool.original_tools[tool_name]
            ctx = replace(ctx, tool_name=tool_name)
            tool_return_value = await super().call_tool(tool_name, result.kwargs, ctx, original_tool)
            result = result.resume(return_value=tool_return_value)
        return result.output
