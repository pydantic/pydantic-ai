"""Unit tests pinning the run-result serialization contract that cassette matching cannot cover."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from inline_snapshot import snapshot
from pydantic import BaseModel, TypeAdapter

from pydantic_ai import (
    Agent,
    AgentRunResult,
    AgentRunResultEvent,
    ModelMessage,
    ModelResponse,
    RequestUsage,
    RunUsage,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel


class StringResultEnvelope(BaseModel):
    result: AgentRunResult[str]


class Profile(BaseModel):
    name: str
    score: int


class ProfileResultEnvelope(BaseModel):
    result: AgentRunResult[Profile]


def assert_same_result(actual: AgentRunResult[Any], expected: AgentRunResult[Any]) -> None:
    assert actual.output == expected.output
    assert actual.all_messages() == expected.all_messages()
    assert actual.new_messages() == expected.new_messages()
    assert actual.usage == expected.usage
    assert actual.run_id == expected.run_id
    assert actual.conversation_id == expected.conversation_id
    assert actual.metadata == expected.metadata
    assert actual.response == expected.response
    assert actual.timestamp == expected.timestamp
    assert actual._traceparent(required=False) == expected._traceparent(required=False)  # pyright: ignore[reportPrivateUsage]


def test_plain_result_round_trip_and_serialized_shape() -> None:
    result = Agent(TestModel(custom_output_text='stored')).run_sync('Save this result', metadata={'tenant': 'example'})
    result._traceparent_value = '00-0123456789abcdef0123456789abcdef-0123456789abcdef-01'  # pyright: ignore[reportPrivateUsage]
    envelope = StringResultEnvelope(result=result)
    assert envelope.result is result

    python_data = envelope.model_dump(mode='python')
    result_data = python_data['result']
    assert set(result_data) == snapshot(
        {
            'conversation_id',
            'messages',
            'metadata',
            'new_message_index',
            'output',
            'output_tool_name',
            'run_id',
            'traceparent',
            'usage',
        }
    )
    assert not {
        'last_model_request_parameters',
        'event_stream_buffer',
        'mcp_tool_defs_cache',
        'pending_messages',
        'last_max_tokens',
        'output_retries_used',
        'run_step',
    } & set(result_data)

    from_python = StringResultEnvelope.model_validate(python_data).result
    from_json = StringResultEnvelope.model_validate_json(envelope.model_dump_json()).result
    assert_same_result(from_python, result)
    assert_same_result(from_json, result)


def test_structured_result_round_trip_and_reuse_as_history() -> None:
    def return_profile(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {'name': 'Ada', 'score': 10})],
            usage=RequestUsage(input_tokens=12, output_tokens=5),
        )

    agent = Agent(FunctionModel(return_profile), output_type=Profile)
    first_result = agent.run_sync('Create a profile')
    result = agent.run_sync('Create another profile', message_history=first_result.all_messages())
    result.usage.requests = 7
    result.usage.tool_calls = 3

    envelope = ProfileResultEnvelope(result=result)
    from_python = ProfileResultEnvelope.model_validate(envelope.model_dump(mode='python')).result
    from_json = ProfileResultEnvelope.model_validate_json(envelope.model_dump_json()).result

    assert isinstance(from_json.output, Profile)
    assert_same_result(from_python, result)
    assert_same_result(from_json, result)
    assert from_json.usage.requests == 7
    assert from_json.usage.tool_calls == 3

    messages = from_json.all_messages(output_tool_return_content='Profile stored')
    assert isinstance(messages[-1].parts[0], ToolReturnPart)
    assert messages[-1].parts[0].content == 'Profile stored'

    continued = agent.run_sync('Continue', message_history=from_json.all_messages())
    assert continued.output == Profile(name='Ada', score=10)


def test_unparameterized_result_follows_the_output_type_default() -> None:
    """`OutputDataT` defaults to `str`, so a bare `AgentRunResult` is `AgentRunResult[str]`."""
    adapter = TypeAdapter(AgentRunResult)
    result = AgentRunResult(output='plain')

    assert adapter.validate_json(adapter.dump_json(result)).output == 'plain'
    assert adapter.validate_python(result) is result


def test_any_output_round_trip() -> None:
    adapter = TypeAdapter(AgentRunResult[Any])
    result = AgentRunResult(output={'nested': ['value']})

    assert adapter.validate_python(adapter.dump_python(result, mode='python')).output == {'nested': ['value']}
    assert adapter.validate_json(adapter.dump_json(result)).output == {'nested': ['value']}


def test_missing_optional_fields_use_fresh_defaults() -> None:
    adapter = TypeAdapter(AgentRunResult[str])

    first = adapter.validate_python({'output': 'one', 'messages': []})
    second = adapter.validate_python({'output': 'two', 'messages': []})

    assert first.new_messages() == []
    assert first.usage == RunUsage()
    assert first.usage is not second.usage
    assert first.metadata is None
    assert first._traceparent(required=False) is None  # pyright: ignore[reportPrivateUsage]
    assert UUID(first.run_id).version == 7
    assert UUID(first.conversation_id).version == 7
    assert first.run_id != second.run_id
    assert first.conversation_id != second.conversation_id


def test_legacy_result_shape_round_trip() -> None:
    result = Agent(TestModel(custom_output_text='legacy')).run_sync('Load an old result', metadata={'source': 'old'})
    result.usage.requests = 4
    result.usage.input_tokens = 123
    public_data: dict[str, Any] = StringResultEnvelope(result=result).model_dump(mode='json')['result']
    legacy_data: dict[str, Any] = {
        'output': public_data['output'],
        '_output_tool_name': public_data['output_tool_name'],
        '_state': {
            'message_history': public_data['messages'],
            'usage': public_data['usage'],
            'output_retries_used': 2,
            'run_step': 9,
            'run_id': public_data['run_id'],
            'conversation_id': public_data['conversation_id'],
            'metadata': public_data['metadata'],
            'last_max_tokens': 100,
            'last_model_request_parameters': None,
            'pending_messages': [],
            'event_stream_buffer': [],
            'mcp_tool_defs_cache': {},
        },
        '_new_message_index': public_data['new_message_index'],
        '_traceparent_value': public_data['traceparent'],
    }

    reloaded = StringResultEnvelope.model_validate({'result': legacy_data}).result
    assert_same_result(reloaded, result)
    assert reloaded.usage.requests == 4
    assert reloaded.usage.input_tokens == 123

    preferred = StringResultEnvelope.model_validate(
        {'result': legacy_data | {'run_id': 'public-run-id', 'metadata': {'source': 'public'}}}
    ).result
    assert preferred.run_id == 'public-run-id'
    assert preferred.metadata == {'source': 'public'}


def test_run_result_event_round_trip() -> None:
    result = Agent(TestModel(custom_output_text='event output')).run_sync('Stream this result')
    adapter = TypeAdapter(AgentRunResultEvent[str])

    reloaded = adapter.validate_json(adapter.dump_json(AgentRunResultEvent(result))).result

    assert_same_result(reloaded, result)
