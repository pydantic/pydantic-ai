import json
from typing import Any

from mcp.client.session import ClientSession, ElicitationFnT
from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams, ElicitResult
from pydantic import ValidationError
from pydantic.type_adapter import TypeAdapter

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai._tool_manager import ToolManager
from pydantic_ai.exceptions import ModelRetry, ToolRetryError
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.usage import RunUsage


async def get_tools_and_schemas_from_toolset(
    toolset: AbstractToolset[Any], deps: AgentDepsT | None = None
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Extract all tool names and schemas from a toolset.

    Args:
        toolset: The toolset to extract tools from.
        deps: Optional dependencies for the run context.

    Returns:
        Tuple of (tool_names, tool_schemas) where tool_schemas maps tool_name to JSON schema.
    """
    # Create a temporary run context to get tools
    temp_context = RunContext[Any](
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
    )

    # Get all available tools from the toolset
    available_tools = await toolset.get_tools(temp_context)

    tool_names = list(available_tools.keys())
    tool_schemas: dict[str, dict[str, Any]] = {}

    for tool_name, toolset_tool in available_tools.items():
        # Extract JSON schema from tool definition
        tool_schemas[tool_name] = toolset_tool.tool_def.parameters_json_schema

    return tool_names, tool_schemas


def create_tool_elicitation_callback(toolset: AbstractToolset[Any], deps: AgentDepsT | None = None) -> ElicitationFnT:
    """Create an elicitation callback for tool injection.

    Args:
        toolset: The toolset containing tools available to Python code.
        deps: Optional dependencies for tool execution context.
    """
    tool_call_adapter = TypeAdapter(ToolCallPart)

    async def elicitation_callback(
        context: RequestContext[ClientSession, Any, Any], params: ElicitRequestParams
    ) -> ElicitResult:
        try:
            try:
                data = json.loads(params.message)
            except json.JSONDecodeError as e:
                return ElicitResult(action='decline', content={'error': f'Invalid JSON: {e}'})

            # Handle tool discovery requests
            if data.get('action') == 'discover_tools':
                try:
                    tool_names, tool_schemas = await get_tools_and_schemas_from_toolset(toolset, deps)
                    # Return as JSON string in the content dict
                    discovery_result = {'tool_names': tool_names, 'tool_schemas': tool_schemas}
                    return ElicitResult(action='accept', content={'data': json.dumps(discovery_result)})
                except Exception as e:
                    return ElicitResult(action='decline', content={'error': f'Tool discovery failed: {e}'})

            # Handle regular tool execution requests
            try:
                tool_call = tool_call_adapter.validate_python(data)
            except ValidationError as e:
                return ElicitResult(action='decline', content={'error': f'Invalid tool call: {e}'})

            base_run_context = RunContext[Any](
                deps=deps,
                model=TestModel(),
                usage=RunUsage(),
            )

            tool_manager = await ToolManager(toolset=toolset).for_run_step(ctx=base_run_context)
            result = await tool_manager.handle_call(call=tool_call)
            # Wrap result as JSON string since ElicitResult.content has type constraints
            return ElicitResult(action='accept', content={'result': json.dumps(result)})

        except ToolRetryError as e:
            # Return the retry information so the agent can understand what went wrong
            # and adjust its approach. The tool_retry contains the error message and guidance.
            retry_info = {
                'error': 'Tool retry needed',
                'tool_name': e.tool_retry.tool_name,
                'message': e.tool_retry.content,
                'tool_call_id': e.tool_retry.tool_call_id,
            }
            return ElicitResult(action='decline', content={'retry': json.dumps(retry_info)})
        except ModelRetry as e:
            return ElicitResult(action='decline', content={'error': f'Model retry failed: {e}'})

    return elicitation_callback
