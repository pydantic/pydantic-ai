"""Unit tests pinning the run-result serialization contract that cassette matching cannot cover."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Literal
from uuid import UUID

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, TypeAdapter, ValidationError

from pydantic_ai import (
    Agent,
    AgentRunResult,
    AgentRunResultEvent,
    DeferredToolRequests,
    ModelMessage,
    ModelResponse,
    ModelRetry,
    RequestUsage,
    RunCancelled,
    RunContext,
    RunUsage,
    ToolCallPart,
    ToolReturnPart,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.conversation_store import InMemoryConversationStore
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.result import StreamedRunResult

from .function_model_prefix import COMPACTION_REWRITES_HISTORY


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


async def test_streamed_run_result_settles_into_a_serializable_result() -> None:
    agent = Agent(TestModel(custom_output_text='streamed'), instructions='Be helpful.')

    async with agent.run_stream('Stream this') as streamed:
        with pytest.raises(UserError, match='still streaming'):
            streamed.result

        await streamed.get_output()
        result = streamed.result

        assert isinstance(result, AgentRunResult)
        assert result.output == 'streamed'
        assert result.all_messages() == streamed.all_messages()
        assert result.new_messages() == streamed.new_messages()
        assert result.usage == streamed.usage
        assert result.run_id == streamed.run_id
        assert result.conversation_id == streamed.conversation_id
        assert result.metadata == streamed.metadata

    adapter = TypeAdapter(AgentRunResult[str])
    reloaded = adapter.validate_json(adapter.dump_json(result))
    assert_same_result(reloaded, result)


async def test_streamed_structured_output_keeps_the_output_tool_name() -> None:
    agent = Agent(TestModel(), instructions='Be helpful.', output_type=Profile)

    async with agent.run_stream('Create a profile') as streamed:
        await streamed.get_output()
        result = streamed.result

    assert result.output == Profile(name='a', score=0)
    messages = result.all_messages(output_tool_return_content='Profile stored')
    assert isinstance(messages[-1].parts[0], ToolReturnPart)
    assert messages[-1].parts[0].content == 'Profile stored'

    adapter = TypeAdapter(AgentRunResult[Profile])
    assert adapter.validate_json(adapter.dump_json(result)).output == Profile(name='a', score=0)


async def test_streamed_deferred_pause_settles_into_its_requests() -> None:
    """A `run_stream` that pauses on an approval settles into a result carrying the pending requests."""
    agent = Agent(
        TestModel(call_tools=['delete_file']),
        instructions='Be helpful.',
        output_type=[str, DeferredToolRequests],
    )

    @agent.tool_plain(requires_approval=True)
    def delete_file(path: str) -> str:
        raise AssertionError('should not execute')  # pragma: no cover

    async with agent.run_stream('Delete a file') as streamed:
        await streamed.get_output()
        result = streamed.result

    assert isinstance(result.output, DeferredToolRequests)
    assert [call.tool_name for call in result.output.approvals] == ['delete_file']
    assert result.all_messages() == streamed.all_messages()


async def test_streamed_result_can_be_stored_and_replayed_as_history() -> None:
    agent = Agent(TestModel(custom_output_text='first'), instructions='Be helpful.')

    async with agent.run_stream('Start') as streamed:
        await streamed.get_output()
        stored = StringResultEnvelope(result=streamed.result).model_dump_json()

    loaded = StringResultEnvelope.model_validate_json(stored).result
    continued = await agent.run('Continue', message_history=loaded.all_messages())
    assert continued.all_messages()[: len(loaded.all_messages())] == loaded.all_messages()


async def test_settling_a_cancelled_stream_is_refused() -> None:
    """Cancelling completes the stream without producing an output, so there is nothing to settle."""
    agent = Agent(TestModel(custom_output_text='a much longer streamed response'), instructions='Be helpful.')

    async with agent.run_stream('Stream this') as streamed:
        await anext(streamed.stream_text(delta=True))
        await streamed.cancel()

        assert streamed.is_complete
        with pytest.raises(UserError, match='cancelled before it produced an output'):
            streamed.result

        # The partial history is still there; only the settled result is refused.
        assert streamed.all_messages()


async def test_cancelling_after_the_output_arrived_still_settles() -> None:
    """A stream consumed to the end has its output cached, so a later cancel changes nothing."""
    agent = Agent(TestModel(custom_output_text='settled'), instructions='Be helpful.')

    async with agent.run_stream('Stream this') as streamed:
        await streamed.get_output()
        await streamed.cancel()

        assert streamed.result.output == 'settled'


@pytest.mark.parametrize('method', ['stream_text', 'stream_text_delta', 'stream_response'])
async def test_settling_a_run_whose_final_output_failed_validation_is_refused(
    method: Literal['stream_text', 'stream_text_delta', 'stream_response'],
) -> None:
    """These methods validate the final output as they finish the stream; when it fails, nothing is left to settle."""
    agent = Agent(TestModel(custom_output_text='rejected'), instructions='Be helpful.')

    @agent.output_validator
    def reject_the_final_output(ctx: RunContext, output: str) -> str:
        if ctx.partial_output:
            return output
        raise ModelRetry('Not good enough.')

    async with agent.run_stream('Stream this') as streamed:
        stream = (
            streamed.stream_response()
            if method == 'stream_response'
            else streamed.stream_text(delta=method == 'stream_text_delta')
        )
        with pytest.raises(UnexpectedModelBehavior, match='retries are not supported'):
            async for _ in stream:
                pass

        assert streamed.is_complete
        with pytest.raises(UserError, match='stopped before it finished'):
            streamed.result

        assert isinstance(streamed.all_messages()[-1], ModelResponse)


async def stream_text_then_a_tool_call(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    yield 'The answer'
    yield {0: DeltaToolCall(name='record_answer', json_args='{}', tool_call_id='call_1')}


async def test_settling_a_run_whose_tool_call_failed_after_the_output_is_refused() -> None:
    """With `end_strategy='exhaustive'`, tool calls next to the output run as the stream finishes, after validation."""
    agent = Agent(
        FunctionModel(stream_function=stream_text_then_a_tool_call),
        instructions='Be helpful.',
        end_strategy='exhaustive',
    )

    @agent.tool_plain
    def record_answer() -> str:
        raise RuntimeError('Storage is down.')

    async with agent.run_stream('Answer') as streamed:
        with pytest.raises(RuntimeError, match='Storage is down'):
            await streamed.get_output()

        assert streamed.is_complete
        with pytest.raises(UserError, match='stopped before it finished'):
            streamed.result


async def test_settling_a_run_cancelled_while_it_finishes_is_refused() -> None:
    """A run cancelled while the tool calls next to its output run never finished, though its output was validated."""
    agent = Agent(
        FunctionModel(stream_function=stream_text_then_a_tool_call),
        instructions='Be helpful.',
        end_strategy='exhaustive',
    )

    @agent.tool
    async def record_answer(ctx: RunContext) -> str:
        ctx.cancel()
        return 'recorded'

    streams: list[StreamedRunResult[object, str]] = []
    with pytest.raises(RunCancelled):
        async with agent.run_stream('Answer') as streamed:
            streams.append(streamed)
            await streamed.get_output()

    [streamed] = streams
    assert streamed.is_complete
    with pytest.raises(UserError, match='stopped before it finished'):
        streamed.result


def test_streamed_result_hands_back_a_run_result_it_already_holds() -> None:
    """`run_stream` yields a pre-built result when a `wrap_run` capability short-circuits the run."""
    held = Agent(TestModel(custom_output_text='short-circuited')).run_sync('Go')
    streamed: StreamedRunResult[None, str] = StreamedRunResult(held.all_messages(), 0, run_result=held)

    assert streamed.result is held


def test_serialization_honors_the_callers_filters() -> None:
    """`include`/`exclude` name the public keys, which the serializer synthesizes rather than owns."""
    result = Agent(TestModel(custom_output_text='filtered')).run_sync('Filter this')
    adapter = TypeAdapter(AgentRunResult[str])

    assert 'output' not in adapter.dump_python(result, exclude={'output'})
    assert 'messages' not in adapter.dump_python(result, exclude={'messages'})
    assert set(adapter.dump_python(result, include={'output', 'usage'})) == {'output', 'usage'}
    assert 'messages' not in StringResultEnvelope(result=result).model_dump(exclude={'result': {'messages'}})['result']

    assert 'messages' not in adapter.dump_python(result, exclude={'messages': True})

    # A spec reaching *into* a key is dropped by Pydantic before the serializer is handed its
    # mapping, so it has to be applied here too.
    assert len(adapter.dump_python(result, exclude={'messages': {0}})['messages']) == len(result.all_messages()) - 1
    assert len(adapter.dump_python(result, include={'messages': {0}})['messages']) == 1


def test_a_nested_spec_it_cannot_apply_leaves_the_value_whole() -> None:
    """Filtering the container bounds what a nested spec can reach: one level, mapping or sequence."""
    result = Agent(TestModel(custom_output_text='filtered')).run_sync('Filter this')
    adapter = TypeAdapter(AgentRunResult[str])
    whole = adapter.dump_python(result)

    # Deeper than one level.
    deep = adapter.dump_python(result, exclude={'messages': {'__all__': {'parts'}}})
    assert len(deep['messages']) == len(whole['messages'])

    # Aimed at a key whose value is neither a mapping nor a sequence.
    assert adapter.dump_python(result, exclude={'usage': {'requests'}})['usage'] == whole['usage']


def test_serialization_honors_a_redaction_inside_metadata() -> None:
    """A nested `exclude` must not dump in full the value it was asked to redact."""
    result = Agent(TestModel(custom_output_text='filtered')).run_sync(
        'Filter this', metadata={'api_key': 'secret', 'tenant': 'acme'}
    )
    adapter = TypeAdapter(AgentRunResult[str])

    assert adapter.dump_python(result, exclude={'metadata': {'api_key'}})['metadata'] == {'tenant': 'acme'}
    assert b'secret' not in adapter.dump_json(result, exclude={'metadata': {'api_key'}})
    assert adapter.dump_python(result)['metadata'] == {'api_key': 'secret', 'tenant': 'acme'}


def test_serialization_honors_a_one_level_spec_written_as_a_mapping() -> None:
    """Pydantic takes `{'api_key': True}` as the same one-level spec as `{'api_key'}`."""
    result = Agent(TestModel(custom_output_text='filtered')).run_sync(
        'Filter this', metadata={'api_key': 'secret', 'tenant': 'acme'}
    )
    adapter = TypeAdapter(AgentRunResult[str])

    assert adapter.dump_python(result, exclude={'metadata': {'api_key': True}})['metadata'] == {'tenant': 'acme'}
    assert b'secret' not in adapter.dump_json(result, exclude={'metadata': {'api_key': True}})
    assert adapter.dump_python(result, include={'metadata': {'tenant': True}}) == {'metadata': {'tenant': 'acme'}}
    assert (
        len(adapter.dump_python(result, exclude={'messages': {0: True}})['messages']) == len(result.all_messages()) - 1
    )

    # An empty mapping names no keys, and `True` names the whole key, so either way the value stays whole.
    assert adapter.dump_python(result, include={'metadata': {}})['metadata'] == {'api_key': 'secret', 'tenant': 'acme'}
    assert adapter.dump_python(result, include={'metadata': True})['metadata'] == {
        'api_key': 'secret',
        'tenant': 'acme',
    }


def test_a_filtered_out_output_is_left_out_rather_than_failing_the_dump() -> None:
    """The wrap handler drops `output` when the caller filters it, and the serializer follows."""
    adapter = TypeAdapter(AgentRunResult[Any])
    result = AgentRunResult[Any](output=None)

    assert 'output' not in adapter.dump_python(result, exclude_none=True)
    assert b'"output"' not in adapter.dump_json(result, exclude_none=True)
    assert adapter.dump_python(result)['output'] is None


def test_validator_leaves_non_mapping_input_to_the_dataclass_schema() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(AgentRunResult[str]).validate_python(['not', 'a', 'mapping'])


def test_only_output_is_required() -> None:
    only_output = TypeAdapter(AgentRunResult[str]).validate_python({'output': 'alone'})
    assert only_output.output == 'alone'
    assert only_output.all_messages() == []

    with pytest.raises(ValidationError, match='output'):
        TypeAdapter(AgentRunResult[str]).validate_python({'messages': []})


def test_legacy_shape_tolerates_a_sparse_state() -> None:
    """An old payload whose `_state` carries none of the keys worth keeping still loads."""
    sparse = TypeAdapter(AgentRunResult[str]).validate_python({'output': 'sparse', '_state': {}})

    assert sparse.output == 'sparse'
    assert sparse.all_messages() == []
    assert sparse.usage == RunUsage()
    assert UUID(sparse.run_id).version == 7


def _resolve(schema: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    """Follow `node` to its definition in `schema` when it is a `$ref`."""
    if '$ref' in node:
        return schema['$defs'][node['$ref'].removeprefix('#/$defs/')]
    return node


def test_validation_schema_publishes_the_shape_the_validator_accepts() -> None:
    """The published input schema describes the public keys, not the private dataclass fields.

    The JSON schema is generated from the dataclass, but the validator replaces its fields with the
    public ones, so without the override an `AgentRunResult` in an API request model advertised
    `_state` and pulled `GraphAgentState`, tool schemas included, into the published `$defs`.
    """
    adapter = TypeAdapter(AgentRunResult[str])
    validation = adapter.json_schema(mode='validation')
    serialization = adapter.json_schema(mode='serialization')
    accepted = _resolve(validation, validation)
    produced = _resolve(serialization, serialization)

    assert set(accepted['properties']) == set(produced['properties'])
    assert {k: v for k, v in accepted.items() if k != '$defs'} == snapshot(
        {
            'description': 'The final result of an agent run.',
            'properties': {
                'output': {'title': 'Output', 'type': 'string'},
                'messages': {
                    'items': {
                        'discriminator': {
                            'mapping': {'request': '#/$defs/ModelRequest', 'response': '#/$defs/ModelResponse'},
                            'propertyName': 'kind',
                        },
                        'oneOf': [{'$ref': '#/$defs/ModelRequest'}, {'$ref': '#/$defs/ModelResponse'}],
                    },
                    'title': 'Messages',
                    'type': 'array',
                },
                'new_message_index': {'title': 'New Message Index', 'type': 'integer'},
                'output_tool_name': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Output Tool Name'},
                'usage': {'$ref': '#/$defs/RunUsage'},
                'run_id': {'title': 'Run Id', 'type': 'string'},
                'conversation_id': {'title': 'Conversation Id', 'type': 'string'},
                'conversation_revision': {
                    'anyOf': [{'type': 'string'}, {'type': 'null'}],
                    'title': 'Conversation Revision',
                },
                'conversation_unsaved': {
                    'items': {
                        'discriminator': {
                            'mapping': {'request': '#/$defs/ModelRequest', 'response': '#/$defs/ModelResponse'},
                            'propertyName': 'kind',
                        },
                        'oneOf': [{'$ref': '#/$defs/ModelRequest'}, {'$ref': '#/$defs/ModelResponse'}],
                    },
                    'title': 'Conversation Unsaved',
                    'type': 'array',
                },
                'conversation_unchecked': {'title': 'Conversation Unchecked', 'type': 'boolean'},
                'conversation_saved': {'title': 'Conversation Saved', 'type': 'boolean'},
                'conversation_merged_tail': {'$ref': '#/$defs/ModelRequest'},
                'conversation_messages': {
                    'items': {
                        'anyOf': [
                            {
                                'discriminator': {
                                    'mapping': {'request': '#/$defs/ModelRequest', 'response': '#/$defs/ModelResponse'},
                                    'propertyName': 'kind',
                                },
                                'oneOf': [{'$ref': '#/$defs/ModelRequest'}, {'$ref': '#/$defs/ModelResponse'}],
                            },
                            {'type': 'integer'},
                        ]
                    },
                    'title': 'Conversation Messages',
                    'type': 'array',
                },
                'conversation_run_start': {'title': 'Conversation Run Start', 'type': 'integer'},
                'metadata': {
                    'anyOf': [{'additionalProperties': True, 'type': 'object'}, {'type': 'null'}],
                    'title': 'Metadata',
                },
                'traceparent': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Traceparent'},
            },
            'required': ['output'],
            'title': 'AgentRunResult',
            'type': 'object',
        }
    )
    assert not [
        name
        for name in ('_state', '_new_message_index', '_AgentRunResult', 'GraphAgentState', 'ModelRequestParameters')
        if name in json.dumps(validation)
    ]
    assert adapter.validate_python({'output': 'hi'}).output == 'hi'

    bare = TypeAdapter(AgentRunResult).json_schema(mode='validation')
    assert set(bare['properties']) == set(accepted['properties'])
    assert bare['required'] == ['output']


def test_each_output_type_keeps_its_own_validation_schema() -> None:
    """Two parameterizations in one API each publish their own `output`, not one shared definition."""

    class City(BaseModel):
        name: str

    class Request(BaseModel):
        text: AgentRunResult[str]
        city: AgentRunResult[City]

    schema = Request.model_json_schema(mode='validation')
    text = _resolve(schema, schema['properties']['text'])
    city = _resolve(schema, schema['properties']['city'])

    assert text['properties']['output'] == {'title': 'Output', 'type': 'string'}
    assert city['properties']['output'] == {'$ref': '#/$defs/City'}


TYPE_ADAPTER: TypeAdapter[AgentRunResult[str]] = TypeAdapter(AgentRunResult[str])


def _trimming_agent() -> Agent[None, str]:
    """An agent that calls a tool three times while a capability shows the model only the last two messages."""
    calls: list[int] = []

    def look_up_three_times(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(calls) == 3:
            return ModelResponse(parts=[TextPart('done')])
        calls.append(len(calls))
        return ModelResponse(parts=[ToolCallPart('lookup', {'n': len(calls)}, tool_call_id=f'call-{len(calls)}')])

    agent = Agent(FunctionModel(look_up_three_times), capabilities=[ProcessHistory(lambda messages: messages[-2:])])

    @agent.tool_plain
    def lookup(n: int) -> str:
        return f'fact {n}'

    return agent


@pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY)
async def test_a_trimmed_run_result_keeps_its_whole_conversation_across_serialization() -> None:
    store = InMemoryConversationStore()
    stored = (await Agent(TestModel()).run('hello')).conversation
    await store.save(stored)
    loaded = await store.load(stored.conversation_id)
    assert loaded is not None
    result = await _trimming_agent().run('look it up', conversation=loaded)
    assert len(result.all_messages()) < len(result.conversation.messages)

    dumped = TYPE_ADAPTER.dump_python(result)
    # The messages the trimmed history still holds are referred to by their index in it.
    assert [item for item in dumped['conversation_messages'] if isinstance(item, int)] == list(
        range(len(result.all_messages()))
    )
    restored = TYPE_ADAPTER.validate_json(TYPE_ADAPTER.dump_json(result))

    assert restored.all_messages() == result.all_messages()
    assert restored.conversation.messages == result.conversation.messages
    assert restored.conversation.new_message_index == result.conversation.new_message_index == len(stored.messages)
    await store.save(restored)
    reloaded = await store.load(stored.conversation_id)
    assert reloaded is not None
    assert reloaded.messages == result.conversation.messages


@pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY)
async def test_a_trimmed_run_result_dumped_without_its_messages_leaves_out_its_whole_conversation_too() -> None:
    result = await _trimming_agent().run('look it up')
    assert 'conversation_messages' in TYPE_ADAPTER.dump_python(result)

    for exclude in ({'messages'}, {'messages': {0}}):
        dumped = TYPE_ADAPTER.dump_python(result, exclude=exclude)
        assert 'conversation_messages' not in dumped
        assert 'conversation_run_start' not in dumped
        TYPE_ADAPTER.validate_json(TYPE_ADAPTER.dump_json(result, exclude=exclude))


def test_a_run_result_whose_whole_conversation_refers_past_its_messages_fails_validation() -> None:
    with pytest.raises(ValidationError, match='`conversation_messages` refers to message 1, but `messages` holds 1'):
        TYPE_ADAPTER.validate_python(
            {
                'output': 'done',
                'messages': [{'kind': 'request', 'parts': []}],
                'conversation_messages': [0, 1],
                'conversation_run_start': 1,
            }
        )


@pytest.mark.parametrize('run_start', [-1, 2, 2.0])
def test_a_run_result_whose_whole_conversation_starts_outside_it_fails_validation(run_start: float) -> None:
    with pytest.raises(
        ValidationError, match=f'`conversation_run_start` is {run_start}, but `conversation_messages` holds 1'
    ):
        TYPE_ADAPTER.validate_python(
            {
                'output': 'done',
                'messages': [{'kind': 'request', 'parts': []}],
                'conversation_messages': [0],
                'conversation_run_start': run_start,
            }
        )


def test_a_run_result_whose_whole_conversation_holds_a_bool_does_not_read_it_as_an_index() -> None:
    with pytest.raises(ValidationError):
        TYPE_ADAPTER.validate_python(
            {
                'output': 'done',
                'messages': [{'kind': 'request', 'parts': []}, {'kind': 'request', 'parts': []}],
                'conversation_messages': [True],
                'conversation_run_start': 0,
            }
        )


def test_a_run_result_whose_whole_conversation_is_not_a_list_fails_validation() -> None:
    with pytest.raises(ValidationError, match='`conversation_messages` must be a list'):
        TYPE_ADAPTER.validate_python(
            {
                'output': 'done',
                'messages': [{'kind': 'request', 'parts': []}],
                'conversation_messages': {'0': 0},
                'conversation_run_start': 0,
            }
        )
