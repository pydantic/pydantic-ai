"""Tool call IDs that are only unique within one response, per `ModelProfile.tool_call_ids_are_response_scoped`."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, Literal, cast

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, ModelHTTPError
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters, infer_model
from pydantic_ai.models._tool_call_ids import qualify_tool_call_id, tool_call_id_scope, unqualify_tool_call_id
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.tools import ToolDefinition

from ..conftest import try_import
from .mock_openai import MockOpenAI, get_mock_chat_completion_kwargs

with try_import() as imports_successful:
    from openai import AsyncOpenAI
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_chunk import (
        Choice as ChunkChoice,
        ChoiceDelta,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
    from openai.types.chat.chat_completion_message_tool_call import Function

    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.bedrock import BedrockProvider
    from pydantic_ai.providers.bedrock_mantle import BedrockMantleProvider
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai or bedrock not installed')

# Anthropic's `tool_use.id` pattern, which Claude on Bedrock enforces for IDs from any provider.
ANTHROPIC_TOOL_USE_ID = re.compile(r'[a-zA-Z0-9_-]+')
CREATED = 1704067200  # 2024-01-01


def test_qualified_ids_restore_and_stay_valid():
    """Unit: the qualified form is a pure function of the ID and its response, and restoring it is exact.

    Not a VCR test: these are the invariants every adapter relies on, including inputs no provider sends today.
    """
    qualified = qualify_tool_call_id('call_0', 'resp_a')
    assert qualified == snapshot('call_0_MshzREZh')
    assert ANTHROPIC_TOOL_USE_ID.fullmatch(qualified)
    assert qualify_tool_call_id('call_0', 'resp_b') != qualified
    assert unqualify_tool_call_id(qualified, 'resp_a') == 'call_0'
    # Only the scope it was qualified with restores an ID, so an ID from another response is left alone.
    assert unqualify_tool_call_id(qualified, 'resp_b') == qualified
    assert unqualify_tool_call_id('call_0', 'resp_a') == 'call_0'
    # `OpenAIResponsesModel` stored `{response_id}:{call_id}` before the flag moved to `ModelProfile`.
    assert unqualify_tool_call_id('resp_a:call_0', 'resp_a') == 'call_0'


def test_scope_falls_back_to_the_timestamp():
    """Unit: without a provider response ID the scope is the response's timestamp, whatever its time zone.

    Not a VCR test: a stored history can carry a naive or non-UTC timestamp no recorded response produces.
    """
    timestamp = datetime(2024, 1, 1, 12, tzinfo=UTC)
    assert tool_call_id_scope('resp_a', timestamp) == 'resp_a'
    assert tool_call_id_scope(None, timestamp) == snapshot('2024-01-01T12:00:00+00:00')
    assert tool_call_id_scope('', timestamp) == tool_call_id_scope(None, timestamp)
    assert tool_call_id_scope(None, timestamp.replace(tzinfo=None)) == tool_call_id_scope(None, timestamp)
    assert tool_call_id_scope(None, timestamp.astimezone(timezone(timedelta(hours=2)))) == tool_call_id_scope(
        None, timestamp
    )


def _completion(response_id: str, tool_name: str | None) -> chat.ChatCompletion:
    if tool_name:
        tool_call = ChatCompletionMessageFunctionToolCall(
            id='call_0', type='function', function=Function(name=tool_name, arguments='{}')
        )
        message = ChatCompletionMessage(role='assistant', tool_calls=[tool_call])
    else:
        message = ChatCompletionMessage(role='assistant', content='done')
    return chat.ChatCompletion(
        id=response_id,
        choices=[Choice(finish_reason='tool_calls' if tool_name else 'stop', index=0, message=message)],
        created=CREATED,
        model='gpt-test',
        object='chat.completion',
    )


def _chunks(response_id: str, tool_name: str | None) -> list[chat.ChatCompletionChunk]:
    if tool_name:
        tool_call = ChoiceDeltaToolCall(
            index=0, id='call_0', type='function', function=ChoiceDeltaToolCallFunction(name=tool_name, arguments='{}')
        )
        deltas: list[tuple[ChoiceDelta, Literal['tool_calls', 'stop'] | None]] = [
            (ChoiceDelta(tool_calls=[tool_call]), None),
            (ChoiceDelta(), 'tool_calls'),
        ]
    else:
        deltas = [(ChoiceDelta(content='done'), None), (ChoiceDelta(), 'stop')]
    return [
        chat.ChatCompletionChunk(
            id=response_id,
            choices=[ChunkChoice(index=0, delta=delta, finish_reason=finish_reason)],
            created=CREATED,
            model='gpt-test',
            object='chat.completion.chunk',
        )
        for delta, finish_reason in deltas
    ]


def _chat_model(
    *, stream: bool, response_ids: list[str], scoped: bool, base_url: str = 'https://api.openai.com/v1'
) -> tuple[OpenAIChatModel, AsyncOpenAI]:
    """A Chat Completions model whose responses call `first_tool`, then `second_tool`, both as `call_0`, then answer.

    With a single response ID, it only answers.
    """
    tool_names = ['first_tool', 'second_tool', None] if len(response_ids) == 3 else [None]
    turns = list(zip(response_ids, tool_names))
    if stream:
        client = MockOpenAI(stream=[_chunks(response_id, tool_name) for response_id, tool_name in turns])
    else:
        client = MockOpenAI(completions=[_completion(response_id, tool_name) for response_id, tool_name in turns])
    client.base_url = base_url
    sdk_client = cast(AsyncOpenAI, client)
    model = OpenAIChatModel(
        'gpt-test',
        provider=OpenAIProvider(openai_client=sdk_client),
        profile=ModelProfile(tool_call_ids_are_response_scoped=scoped),
    )
    return model, sdk_client


def _agent() -> Agent[None, str]:
    agent = Agent(instructions='Call first_tool. After receiving its result, call second_tool in a new model response.')

    @agent.tool_plain
    def first_tool() -> str:
        return 'first result'

    @agent.tool_plain
    def second_tool() -> str:
        return 'second result'

    return agent


async def _run(agent: Agent[None, str], model: Model, *, stream: bool) -> list[ModelMessage]:
    if stream:
        async with agent.run_stream('Follow the tool instructions.', model=model) as result:
            await result.get_output()
        return result.all_messages()
    return (await agent.run('Follow the tool instructions.', model=model)).all_messages()


def _history_ids(messages: list[ModelMessage]) -> list[str]:
    return [
        part.tool_call_id
        for message in messages
        for part in message.parts
        if isinstance(part, ToolCallPart | ToolReturnPart)
    ]


def _wire_ids(client: AsyncOpenAI) -> list[list[str]]:
    """The tool call IDs each Chat Completions request sent, calls and results in message order."""
    return [
        [
            tool_call_id
            for message in kwargs['messages']
            for tool_call_id in (
                [message['tool_call_id']]
                if message['role'] == 'tool'
                else [tool_call['id'] for tool_call in message.get('tool_calls', [])]
            )
        ]
        for kwargs in get_mock_chat_completion_kwargs(client)
    ]


@pytest.mark.parametrize('stream', [False, True], ids=['request', 'stream'])
@pytest.mark.parametrize('scoped', [True, False], ids=['scoped', 'unscoped'])
async def test_chat_ids_are_qualified_in_history_and_restored_on_the_wire(
    allow_model_requests: None, stream: bool, scoped: bool
):
    """A Chat Completions API that sends `call_0` in every response gets unique IDs in history, and its own IDs back.

    Mocked rather than recorded: no Chat Completions API we can record reuses IDs (Ollama, llama.cpp, vLLM, Groq,
    Mistral and Mantle's Chat endpoint all send unique ones), and the mock lets one test pin both sides of the flag.
    """
    model, client = _chat_model(stream=stream, response_ids=['chatcmpl-a', 'chatcmpl-b', 'chatcmpl-c'], scoped=scoped)

    messages = await _run(_agent(), model, stream=stream)

    if scoped:
        first, second = qualify_tool_call_id('call_0', 'chatcmpl-a'), qualify_tool_call_id('call_0', 'chatcmpl-b')
        assert _history_ids(messages) == [first, first, second, second]
        assert first != second
    else:
        assert _history_ids(messages) == ['call_0', 'call_0', 'call_0', 'call_0']
    assert _wire_ids(client) == [[], ['call_0', 'call_0'], ['call_0', 'call_0', 'call_0', 'call_0']]


async def test_chat_ids_stay_qualified_for_another_endpoint(allow_model_requests: None):
    """Only the endpoint that issued an ID gets it back unqualified: another one gets the unique stored ID."""
    model, _ = _chat_model(stream=False, response_ids=['chatcmpl-a', 'chatcmpl-b', 'chatcmpl-c'], scoped=True)
    messages = await _run(_agent(), model, stream=False)
    other_model, other_client = _chat_model(
        stream=False, response_ids=['chatcmpl-d'], scoped=True, base_url='https://other.example/v1'
    )

    await Agent(other_model).run('Again.', message_history=messages)

    first, second = qualify_tool_call_id('call_0', 'chatcmpl-a'), qualify_tool_call_id('call_0', 'chatcmpl-b')
    assert _wire_ids(other_client) == [[first, first, second, second]]


@pytest.mark.parametrize('stream', [False, True], ids=['request', 'stream'])
async def test_chat_ids_without_response_id_survive_a_stored_history(allow_model_requests: None, stream: bool):
    """A response without an ID is scoped by its timestamp, so a history stored and loaded again, as a durable
    execution engine does between steps, restores the same IDs.
    """
    model, _ = _chat_model(stream=stream, response_ids=['', '', ''], scoped=True)
    messages = await _run(_agent(), model, stream=stream)
    responses = [message for message in messages if isinstance(message, ModelResponse) and message.tool_calls]
    assert [response.provider_response_id for response in responses] == [None, None]
    assert [call.tool_call_id for response in responses for call in response.tool_calls] == [
        qualify_tool_call_id('call_0', tool_call_id_scope(None, response.timestamp)) for response in responses
    ]

    loaded = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages))
    replay_model, replay_client = _chat_model(stream=False, response_ids=['chatcmpl-d'], scoped=True)
    await Agent(replay_model).run('Again.', message_history=loaded)

    assert _wire_ids(replay_client) == [['call_0', 'call_0', 'call_0', 'call_0']]


@pytest.mark.vcr
async def test_qualified_ids_replay_to_anthropic_on_bedrock(
    allow_model_requests: None, bedrock_provider: BedrockProvider
):
    """Mantle's Responses IDs, qualified per response, are accepted by Claude on Bedrock, which enforces Anthropic's
    `^[a-zA-Z0-9_-]+$` pattern on the IDs of every provider's history.
    """
    mantle = infer_model(
        'bedrock-mantle:openai.gpt-5.6-luna',
        lambda _: BedrockMantleProvider(
            region_name='us-east-1', api_key=os.getenv('AWS_BEARER_TOKEN_BEDROCK', 'mock-api-key')
        ),
    )
    agent = _agent()
    messages = (await agent.run('Follow the tool instructions.', model=mantle)).all_messages()

    ids = _history_ids(messages)
    # Mantle's GPT-5 profile is flagged because it has sent `call_0` in every response (`test_bedrock_mantle.py`
    # records it); in this recording it sent unique IDs, which get their response's suffix all the same.
    assert ids == snapshot(
        [
            'call_90eec98086dd5fc8ba30da13ebb9b8de_XKHtVqYh',
            'call_90eec98086dd5fc8ba30da13ebb9b8de_XKHtVqYh',
            'call_8b70c794677d532d8856b610cd7a6ff8_C6Tnln0q',
            'call_8b70c794677d532d8856b610cd7a6ff8_C6Tnln0q',
        ]
    )
    assert all(ANTHROPIC_TOOL_USE_ID.fullmatch(tool_call_id) for tool_call_id in ids)

    claude = BedrockConverseModel('us.anthropic.claude-haiku-4-5-20251001-v1:0', provider=bedrock_provider)
    bodies: list[dict[str, Any]] = []

    def capture(request: Any, **_: Any) -> None:
        bodies.append(json.loads(request.body))

    events = ['before-send.bedrock-runtime.Converse', 'before-send.bedrock-runtime.ConverseStream']
    for event in events:
        claude.client.meta.events.register_last(event, capture)
    try:
        result = await agent.run('Reply with exactly OK.', message_history=messages, model=claude)
    finally:
        for event in events:
            claude.client.meta.events.unregister(event, capture)

    assert result.output == snapshot('OK')
    sent_ids = [
        block[kind]['toolUseId']
        for message in bodies[0]['messages']
        for block in message['content']
        for kind in ('toolUse', 'toolResult')
        if kind in block
    ]
    assert sent_ids == ids


@pytest.mark.vcr
async def test_colon_qualified_ids_are_rejected_by_anthropic_on_bedrock(
    allow_model_requests: None, bedrock_provider: BedrockProvider
):
    """The `{response_id}:{call_id}` form `OpenAIResponsesModel` stored before is rejected by Claude on Bedrock,
    which is why the qualified form doesn't use a colon.
    """
    tool_call_id = 'resp_43amfn3g3uar3i4sa5b7sz35cukuufaisldkwel5v6o47xzpr5va:call_0'
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Look it up, then say done.')]),
        ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id)]),
        ModelRequest(parts=[ToolReturnPart('lookup', 'a', tool_call_id)]),
    ]
    lookup = ToolDefinition(name='lookup', parameters_json_schema={'type': 'object', 'properties': {}})
    claude = BedrockConverseModel('us.anthropic.claude-haiku-4-5-20251001-v1:0', provider=bedrock_provider)

    with pytest.raises(ModelHTTPError) as exc_info:
        await claude.request(history, None, ModelRequestParameters(function_tools=[lookup]))

    assert exc_info.value.status_code == 400
    assert "tool_use.id: String should match pattern '^[a-zA-Z0-9_-]+$'" in str(exc_info.value)
