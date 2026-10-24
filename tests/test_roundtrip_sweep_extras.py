"""Preserve typed tool payloads at the public JSON history boundary."""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot
from pydantic import TypeAdapter, ValidationError

from pydantic_ai import Agent, ModelMessagesTypeAdapter, RunContext, _utils
from pydantic_ai._deferred_capabilities import (
    LoadCapabilityArgs,
    LoadCapabilityCallPart,
    LoadCapabilityReturn,
    LoadCapabilityReturnPart,
)
from pydantic_ai._tool_search import ToolSearchArgs, ToolSearchReturnContent
from pydantic_ai.capabilities import Capability, Hooks, ToolSearch
from pydantic_ai.capabilities.abstract import ValidatedToolArgs
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    NativeToolSearchCallPart,
    NativeToolSearchReturnPart,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    ToolSearchCallPart,
    ToolSearchReturnPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import ExternalToolset


@pytest.mark.parametrize(
    ('message', 'part_type'),
    [
        pytest.param(
            ModelResponse(
                parts=[
                    NativeToolCallPart(
                        tool_name='tool_search', tool_kind='tool-search', args={'queries': ['x'], 'extra': 'KEEP_ME'}
                    )
                ]
            ),
            NativeToolSearchCallPart,
            id='native-search-call',
        ),
        pytest.param(
            ModelResponse(
                parts=[
                    NativeToolReturnPart(
                        tool_name='tool_search',
                        tool_kind='tool-search',
                        content={'discovered_tools': [{'name': 't1'}], 'extra': 'KEEP_ME'},
                    )
                ]
            ),
            NativeToolSearchReturnPart,
            id='native-search-return',
        ),
        pytest.param(
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='search_tools', tool_kind='tool-search', args={'queries': ['x'], 'extra': 'KEEP_ME'}
                    )
                ]
            ),
            ToolSearchCallPart,
            id='local-search-call',
        ),
        pytest.param(
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='search_tools',
                        tool_kind='tool-search',
                        content={'discovered_tools': [{'name': 't1'}], 'message': 'found', 'extra': 'KEEP_ME'},
                    )
                ]
            ),
            ToolSearchReturnPart,
            id='local-search-return',
        ),
        pytest.param(
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='load_capability',
                        tool_kind='capability-load',
                        args={'id': 'sample', 'extra': 'KEEP_ME'},
                    )
                ]
            ),
            LoadCapabilityCallPart,
            id='capability-call',
        ),
        pytest.param(
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='load_capability',
                        tool_kind='capability-load',
                        content={'instructions': 'Do x', 'version': 2, 'extra': 'KEEP_ME'},
                    )
                ]
            ),
            LoadCapabilityReturnPart,
            id='capability-return',
        ),
    ],
)
def test_typed_tool_kind_round_trip_preserves_extras(message: ModelMessage, part_type: type[object]) -> None:
    """Serialization fidelity is independent of which provider produced the history."""
    assert isinstance(message.parts[0], part_type)
    restored = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json([message]))
    assert isinstance(restored[0].parts[0], part_type)
    assert restored == [message]


@pytest.mark.parametrize(
    ('payload_type', 'payload'),
    [
        (ToolSearchArgs, {'extra': 'KEEP_ME'}),
        (ToolSearchArgs, {'queries': 1, 'extra': 'KEEP_ME'}),
        (ToolSearchReturnContent, {'extra': 'KEEP_ME'}),
        (ToolSearchReturnContent, {'discovered_tools': 'invalid', 'extra': 'KEEP_ME'}),
        (ToolSearchReturnContent, {'discovered_tools': [], 'message': 1, 'extra': 'KEEP_ME'}),
        (LoadCapabilityArgs, {'extra': 'KEEP_ME'}),
        (LoadCapabilityArgs, {'id': 1, 'extra': 'KEEP_ME'}),
        (LoadCapabilityReturn, {'instructions': [], 'extra': 'KEEP_ME'}),
    ],
)
def test_extra_keys_do_not_relax_known_field_validation(payload_type: type[object], payload: object) -> None:
    """Pin the payload validators directly: malformed history deliberately falls back to base parts."""
    adapter = TypeAdapter(payload_type)
    with pytest.raises(ValidationError):
        adapter.validate_python(payload)


async def test_agent_emitted_extra_args_survive_live_history_and_round_trip() -> None:
    """Live search calls retain original args, including keys rejected by tool execution."""
    toolset = ExternalToolset[object]([ToolDefinition(name='get_payment', defer_loading=True)])

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='search_tools',
                        args={'queries': ['mortgage'], 'extra': 'KEEP_ME'},
                    )
                ]
            )
        return ModelResponse(parts=[TextPart('done')])

    result = await Agent(FunctionModel(model_function), toolsets=[toolset], capabilities=[ToolSearch()]).run(
        'find a tool'
    )
    live_calls = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolSearchCallPart)]
    assert len(live_calls) == 1
    assert live_calls[0].args == {'queries': ['mortgage'], 'extra': 'KEEP_ME'}
    assert ModelMessagesTypeAdapter.validate_json(result.all_messages_json()) == result.all_messages()


async def test_after_tool_hook_return_extras_survive_history() -> None:
    hooks = Hooks()

    @hooks.on.after_tool_execute
    def add_version(
        ctx: RunContext[object],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: object,
    ) -> object:
        assert isinstance(result, ToolReturn)
        assert _utils.is_str_dict(result.return_value)
        return ToolReturn(return_value={**result.return_value, 'version': 2}, tools=result.tools)

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='load_capability', args={'id': 'sample'})])
        return ModelResponse(parts=[TextPart('done')])

    capability = Capability[object](id='sample', description='Sample', instructions='Do x', defer_loading=True)
    result = await Agent(FunctionModel(model_function), capabilities=[capability, hooks]).run('load it')
    live_returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, LoadCapabilityReturnPart)]
    assert len(live_returns) == 1
    assert live_returns[0].content == {'instructions': 'Do x', 'version': 2}
    assert ModelMessagesTypeAdapter.validate_json(result.all_messages_json()) == result.all_messages()


async def test_load_capability_schema_keeps_openai_strict_compatibility() -> None:
    """Pin the model-facing schema separately from permissive history payload validation."""

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert len(info.function_tools) == 1
        tool = info.function_tools[0]
        assert tool.parameters_json_schema == snapshot(
            {
                'description': 'Typed arguments for a `load_capability` tool call.',
                'properties': {
                    'id': {'description': 'The id of the capability to load.', 'title': 'Id', 'type': 'string'}
                },
                'required': ['id'],
                'title': 'LoadCapabilityArgs',
                'type': 'object',
            }
        )
        model = FunctionModel(model_function, profile=openai_model_profile('gpt-5'))
        customized = model.customize_request_parameters(ModelRequestParameters(function_tools=[tool]))
        assert customized.function_tools[0].strict is True
        assert customized.function_tools[0].parameters_json_schema == snapshot(
            {
                'description': 'Typed arguments for a `load_capability` tool call.',
                'properties': {'id': {'description': 'The id of the capability to load.', 'type': 'string'}},
                'required': ['id'],
                'type': 'object',
                'additionalProperties': False,
            }
        )
        return ModelResponse(parts=[TextPart('done')])

    capability = Capability[object](id='sample', description='Sample', defer_loading=True)
    await Agent(FunctionModel(model_function), capabilities=[capability]).run('hello')
