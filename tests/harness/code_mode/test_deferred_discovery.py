"""Tool Search guidance for deferred tools that CodeMode keeps native."""

from typing import Literal

import pytest

from pydantic_ai import AbstractToolset, Agent, Tool, ToolDefinition
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolSearchCallPart,
    NativeToolSearchReturnPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults
from pydantic_ai.toolsets import ExternalToolset, FunctionToolset
from pydantic_ai_harness import CodeMode


@pytest.mark.parametrize('kind', ['unapproved', 'external'])
@pytest.mark.parametrize('dynamic_catalog', [False, True])
@pytest.mark.parametrize('native_search', [False, True])
async def test_discovered_deferred_tool_is_called_natively(
    kind: Literal['unapproved', 'external'], dynamic_catalog: bool, native_search: bool
) -> None:
    executions: list[str] = []

    def action(value: str) -> str:
        """Perform an action."""
        executions.append(value)
        return value

    toolset: AbstractToolset[object]
    if kind == 'unapproved':
        toolset = FunctionToolset[object](tools=[Tool(action, requires_approval=True, defer_loading=True)])
    else:
        toolset = ExternalToolset[object](
            [
                ToolDefinition(
                    name='action',
                    description='Perform an action.',
                    parameters_json_schema={
                        'type': 'object',
                        'properties': {'value': {'type': 'string'}},
                        'required': ['value'],
                    },
                    return_schema={'type': 'string'},
                    defer_loading=True,
                )
            ]
        )

    guidance = (
        'Only tools in the available-functions catalog can be called inside `run_code`; '
        'call separately exposed tools directly.'
    )
    step = 0

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal step
        step += 1
        tools = {tool.name: tool for tool in info.function_tools}
        assert 'async def action' not in (tools['run_code'].description or '')
        assert 'async def action' not in (info.instructions or '')
        if step == 1:
            if native_search:
                return ModelResponse(
                    parts=[
                        NativeToolSearchCallPart(args={'queries': ['action']}, tool_call_id='search-1'),
                        NativeToolSearchReturnPart(
                            content={'discovered_tools': [{'name': 'action'}]}, tool_call_id='search-1'
                        ),
                        ToolCallPart('run_code', {'code': 'None'}, tool_call_id='code-1'),
                    ]
                )
            assert guidance in (tools['search_tools'].description or '')
            assert guidance in (tools['run_code'].description or '')
            return ModelResponse(parts=[ToolCallPart('search_tools', {'queries': ['action']}, tool_call_id='search-1')])
        if step == 2:
            assert tools['action'].kind == kind
            announcements = [
                part.content
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, (SystemPromptPart, UserPromptPart))
                and isinstance(part.content, str)
                and guidance in part.content
            ]
            assert bool(announcements) is dynamic_catalog
            if dynamic_catalog:
                assert 'Newly available tools: `action`.' in announcements[0]
            return ModelResponse(parts=[ToolCallPart('action', {'value': 'executed'}, tool_call_id='action-1')])
        assert any(
            isinstance(part, ToolReturnPart) and part.tool_name == 'action' and part.content == 'executed'
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        )
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(
        FunctionModel(model, profile=None if native_search else {'supported_native_tools': frozenset()}),
        toolsets=[toolset],
        capabilities=[ToolSearch(), CodeMode(dynamic_catalog=dynamic_catalog)],
        output_type=[str, DeferredToolRequests],
    )
    paused = await agent.run('find and perform the action')
    assert isinstance(paused.output, DeferredToolRequests)
    assert executions == []
    if kind == 'unapproved':
        assert [call.tool_name for call in paused.output.approvals] == ['action']
        results = DeferredToolResults(approvals={'action-1': True})
    else:
        assert [call.tool_name for call in paused.output.calls] == ['action']
        results = DeferredToolResults(calls={'action-1': 'executed'})

    resumed = await agent.run(message_history=paused.all_messages(), deferred_tool_results=results)
    assert resumed.output == 'done'
    assert executions == (['executed'] if kind == 'unapproved' else [])
