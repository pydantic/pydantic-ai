from __future__ import annotations as _annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from typing import Any, Literal, cast

import pytest
from inline_snapshot import snapshot
from typing_extensions import TypedDict

from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import Usage

from ..conftest import IsNow, try_import
from .mock_async_stream import MockAsyncStream

with try_import() as imports_successful:
    import cohere
    from cohere import (
        AssistantChatMessageV2,
        AssistantMessageResponse,
        AsyncClientV2,
        ChatMessageV2,
        ChatResponse,
        SystemChatMessageV2,
        TextAssistantMessageResponseContentItem,
        ToolCallV2,
        ToolCallV2Function,
        ToolChatMessageV2,
        ToolV2,
        ToolV2Function,
        UserChatMessageV2,
    )

    from pydantic_ai.models.cohere import CohereModel

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='cohere not installed'),
    pytest.mark.anyio,
]


def test_init():
    m = CohereModel('command-r7b-12-2024', api_key='foobar')
    assert m.name() == 'cohere:command-r7b-12-2024'


@dataclass
class MockAsyncClientV2:
    completions: ChatResponse | list[ChatResponse] | None = None
    index = 0

    # @cached_property
    # def chat(self) -> Any:
    #     chat_completions = type('Completions', (), {'create': self.chat_completions_create})
    #     return type('Chat', (), {'completions': chat_completions})

    @classmethod
    def create_mock(cls, completions: ChatResponse | list[ChatResponse]) -> AsyncClientV2:
        return cast(AsyncClientV2, cls(completions=completions))

    async def chat(  # pragma: no cover
        self, *_args: Any, **_kwargs: Any
    ) -> ChatResponse:
        assert self.completions is not None
        if isinstance(self.completions, list):
            response = self.completions[self.index]
        else:
            response = self.completions
        self.index += 1
        return response


def completion_message(message: AssistantMessageResponse, *, usage: cohere.Usage | None = None) -> ChatResponse:
    return ChatResponse(
        id='123',
        finish_reason='COMPLETE',
        message=message,
        usage=usage,
    )


async def test_request_simple_success(allow_model_requests: None):
    c = completion_message(
        AssistantMessageResponse(
            content=[
                TextAssistantMessageResponseContentItem(text='world'),
            ],
        )
    )
    mock_client = MockAsyncClientV2.create_mock(c)
    m = CohereModel('command-r7b-12-2024', cohere_client=mock_client)
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.data == 'world'
    assert result.usage() == snapshot(Usage(requests=1))

    # reset the index so we get the same response again
    mock_client.index = 0  # type: ignore

    result = await agent.run('hello', message_history=result.new_messages())
    assert result.data == 'world'
    assert result.usage() == snapshot(Usage(requests=1))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse.from_text(content='world', timestamp=IsNow(tz=timezone.utc)),
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse.from_text(content='world', timestamp=IsNow(tz=timezone.utc)),
        ]
    )


async def test_request_simple_usage(allow_model_requests: None):
    c = completion_message(
        AssistantMessageResponse(
            content=[TextAssistantMessageResponseContentItem(text='world')],
            role='assistant',
        ),
        usage=cohere.Usage(
            tokens=cohere.UsageTokens(input_tokens=1, output_tokens=1),
            billed_units=cohere.UsageBilledUnits(input_tokens=1, output_tokens=1),
        ),
    )
    mock_client = MockAsyncClientV2.create_mock(c)
    m = CohereModel('command-r7b-12-2024', cohere_client=mock_client)
    agent = Agent(m)

    result = await agent.run('Hello')
    assert result.data == 'world'
    assert result.usage() == snapshot(
        Usage(
            requests=1,
            request_tokens=1,
            response_tokens=1,
            total_tokens=2,
            details={
                'input_tokens': 1,
                'output_tokens': 1,
            },
        )
    )


# async def test_request_structured_response(allow_model_requests: None):
#     c = completion_message(
#         ChatCompletionMessage(
#             content=None,
#             role='assistant',
#             tool_calls=[
#                 chat.ChatCompletionMessageToolCall(
#                     id='123',
#                     function=Function(arguments='{"response": [1, 2, 123]}', name='final_result'),
#                     type='function',
#                 )
#             ],
#         )
#     )
#     mock_client = MockOpenAI.create_mock(c)
#     m = OpenAIModel('gpt-4', openai_client=mock_client)
#     agent = Agent(m, result_type=list[int])
#
#     result = await agent.run('Hello')
#     assert result.data == [1, 2, 123]
#     assert result.all_messages() == snapshot(
#         [
#             ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
#             ModelResponse(
#                 parts=[
#                     ToolCallPart.from_raw_args(
#                         tool_name='final_result',
#                         args='{"response": [1, 2, 123]}',
#                         tool_call_id='123',
#                     )
#                 ],
#                 timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
#             ),
#             ModelRequest(
#                 parts=[
#                     ToolReturnPart(
#                         tool_name='final_result',
#                         content='Final result processed.',
#                         tool_call_id='123',
#                         timestamp=IsNow(tz=timezone.utc),
#                     )
#                 ]
#             ),
#         ]
#     )
#
#
# async def test_request_tool_call(allow_model_requests: None):
#     responses = [
#         completion_message(
#             ChatCompletionMessage(
#                 content=None,
#                 role='assistant',
#                 tool_calls=[
#                     chat.ChatCompletionMessageToolCall(
#                         id='1',
#                         function=Function(arguments='{"loc_name": "San Fransisco"}', name='get_location'),
#                         type='function',
#                     )
#                 ],
#             ),
#             usage=CompletionUsage(
#                 completion_tokens=1,
#                 prompt_tokens=2,
#                 total_tokens=3,
#                 prompt_tokens_details=PromptTokensDetails(cached_tokens=1),
#             ),
#         ),
#         completion_message(
#             ChatCompletionMessage(
#                 content=None,
#                 role='assistant',
#                 tool_calls=[
#                     chat.ChatCompletionMessageToolCall(
#                         id='2',
#                         function=Function(arguments='{"loc_name": "London"}', name='get_location'),
#                         type='function',
#                     )
#                 ],
#             ),
#             usage=CompletionUsage(
#                 completion_tokens=2,
#                 prompt_tokens=3,
#                 total_tokens=6,
#                 prompt_tokens_details=PromptTokensDetails(cached_tokens=2),
#             ),
#         ),
#         completion_message(ChatCompletionMessage(content='final response', role='assistant')),
#     ]
#     mock_client = MockOpenAI.create_mock(responses)
#     m = OpenAIModel('gpt-4', openai_client=mock_client)
#     agent = Agent(m, system_prompt='this is the system prompt')
#
#     @agent.tool_plain
#     async def get_location(loc_name: str) -> str:
#         if loc_name == 'London':
#             return json.dumps({'lat': 51, 'lng': 0})
#         else:
#             raise ModelRetry('Wrong location, please try again')
#
#     result = await agent.run('Hello')
#     assert result.data == 'final response'
#     assert result.all_messages() == snapshot(
#         [
#             ModelRequest(
#                 parts=[
#                     SystemPromptPart(content='this is the system prompt'),
#                     UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
#                 ]
#             ),
#             ModelResponse(
#                 parts=[
#                     ToolCallPart.from_raw_args(
#                         tool_name='get_location',
#                         args='{"loc_name": "San Fransisco"}',
#                         tool_call_id='1',
#                     )
#                 ],
#                 timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
#             ),
#             ModelRequest(
#                 parts=[
#                     RetryPromptPart(
#                         content='Wrong location, please try again',
#                         tool_name='get_location',
#                         tool_call_id='1',
#                         timestamp=IsNow(tz=timezone.utc),
#                     )
#                 ]
#             ),
#             ModelResponse(
#                 parts=[
#                     ToolCallPart.from_raw_args(
#                         tool_name='get_location',
#                         args='{"loc_name": "London"}',
#                         tool_call_id='2',
#                     )
#                 ],
#                 timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
#             ),
#             ModelRequest(
#                 parts=[
#                     ToolReturnPart(
#                         tool_name='get_location',
#                         content='{"lat": 51, "lng": 0}',
#                         tool_call_id='2',
#                         timestamp=IsNow(tz=timezone.utc),
#                     )
#                 ]
#             ),
#             ModelResponse.from_text(content='final response', timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc)),
#         ]
#     )
#     assert result.usage() == snapshot(
#         Usage(
#             requests=3,
#             request_tokens=5,
#             response_tokens=3,
#             total_tokens=9,
#             details={'cached_tokens': 3},
#         )
#     )
#
#
# FinishReason = Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']
#
#
# def chunk(delta: list[ChoiceDelta], finish_reason: FinishReason | None = None) -> chat.ChatCompletionChunk:
#     return chat.ChatCompletionChunk(
#         id='x',
#         choices=[
#             ChunkChoice(index=index, delta=delta, finish_reason=finish_reason) for index, delta in enumerate(delta)
#         ],
#         created=1704067200,  # 2024-01-01
#         model='gpt-4',
#         object='chat.completion.chunk',
#         usage=CompletionUsage(completion_tokens=1, prompt_tokens=2, total_tokens=3),
#     )
#
#
# def text_chunk(text: str, finish_reason: FinishReason | None = None) -> chat.ChatCompletionChunk:
#     return chunk([ChoiceDelta(content=text, role='assistant')], finish_reason=finish_reason)
#
#
# def struc_chunk(
#     tool_name: str | None, tool_arguments: str | None, finish_reason: FinishReason | None = None
# ) -> chat.ChatCompletionChunk:
#     return chunk(
#         [
#             ChoiceDelta(
#                 tool_calls=[
#                     ChoiceDeltaToolCall(
#                         index=0, function=ChoiceDeltaToolCallFunction(name=tool_name, arguments=tool_arguments)
#                     )
#                 ]
#             ),
#         ],
#         finish_reason=finish_reason,
#     )
#
#
# class MyTypedDict(TypedDict, total=False):
#     first: str
#     second: str
#
