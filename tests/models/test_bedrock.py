from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timezone
from typing import TYPE_CHECKING, Any, cast

import pytest
from dirty_equals import IsNow
from inline_snapshot import snapshot

from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import (
    ArgsDict,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import Usage

from ..conftest import IsNow, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.bedrock import BedrockModel


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='boto3 not installed'),
    pytest.mark.anyio,
]

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.type_defs import (
        ContentBlockOutputTypeDef,
        ConverseResponseTypeDef,
        ConverseStreamOutputTypeDef,
        ConverseStreamResponseTypeDef,
        TokenUsageTypeDef,
    )


def test_init():
    m = BedrockModel(
        'us.amazon.nova-micro-v1:0',
        aws_access_key_id='foo',
        aws_secret_access_key='bar',
        aws_region='us-east-1',
    )
    assert m.name() == 'us.amazon.nova-micro-v1:0'


@dataclass
class MockBedrockClient:
    completions: ConverseResponseTypeDef | list[ConverseResponseTypeDef] | None = None
    stream: list[ConverseStreamOutputTypeDef] | list[list[ConverseStreamOutputTypeDef]] | None = None
    index = 0

    def converse(self, *_args: Any, **_kwargs: Any) -> ConverseResponseTypeDef:
        if isinstance(self.completions, list):
            response = self.completions[self.index]
        else:
            response = self.completions
        self.index += 1
        return response

    def converse_stream(self, *_args: Any, **_kwargs: Any) -> ConverseStreamResponseTypeDef:
        raise NotImplementedError
        if isinstance(self.stream, list):
            response = self.stream[self.index]
        else:
            response = self.stream
        self.index += 1
        return response

    @classmethod
    def create_mock(cls, completions: ConverseResponseTypeDef | list[ConverseResponseTypeDef]) -> MockBedrockClient:
        return cast(MockBedrockClient, cls(completions))


def completion_message(content: list[ContentBlockOutputTypeDef], usage: TokenUsageTypeDef) -> ConverseResponseTypeDef:
    return {
        'ResponseMetadata': {
            'RequestId': '5f3335f9-edc7-4506-b899-33de742d7e90',
            'HTTPStatusCode': 200,
            'HTTPHeaders': {},
            'RetryAttempts': 0,
        },
        'output': {
            'message': {
                'role': 'assistant',
                'content': content,
            }
        },
        'stopReason': 'max_tokens',
        'usage': usage,
        'metrics': {'latencyMs': 1},
    }


async def test_sync_request_text_response(allow_model_requests: None):
    c = completion_message(
        [{'text': 'world'}],
        {'inputTokens': 5, 'outputTokens': 10, 'totalTokens': 15},
    )
    mock_client = MockBedrockClient.create_mock(c)
    m = BedrockModel('us.amazon.nova-micro-v1:0', bedrock_client=mock_client)
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.data == 'world'
    assert result.usage() == snapshot(Usage(requests=1, request_tokens=5, response_tokens=10, total_tokens=15))

    # reset the index so we get the same response again
    mock_client.index = 0  # type: ignore

    result = await agent.run('hello', message_history=result.new_messages())
    assert result.data == 'world'
    assert result.usage() == snapshot(Usage(requests=1, request_tokens=5, response_tokens=10, total_tokens=15))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(parts=[TextPart(content='world')], timestamp=IsNow(tz=timezone.utc)),
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(parts=[TextPart(content='world')], timestamp=IsNow(tz=timezone.utc)),
        ]
    )


async def test_async_request_text_response(allow_model_requests: None):
    c = completion_message(
        [{'text': 'world'}],
        {'inputTokens': 3, 'outputTokens': 5, 'totalTokens': 8},
    )
    mock_client = MockBedrockClient.create_mock(c)
    m = BedrockModel('claude-3-5-haiku-latest', bedrock_client=mock_client)
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.data == 'world'
    assert result.usage() == snapshot(Usage(requests=1, request_tokens=3, response_tokens=5, total_tokens=8))


async def test_request_structured_response(allow_model_requests: None):
    c = completion_message(
        [
            {
                'toolUse': {
                    'toolUseId': '123',
                    'name': 'final_result',
                    'input': {'response': [1, 2, 3]},
                }
            }
        ],
        {'inputTokens': 3, 'outputTokens': 5, 'totalTokens': 8},
    )

    mock_client = MockBedrockClient.create_mock(c)
    m = BedrockModel('claude-3-5-haiku-latest', bedrock_client=mock_client)
    agent = Agent(m, result_type=list[int])

    result = await agent.run('hello')
    assert result.data == [1, 2, 3]
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='hello',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args=ArgsDict(args_dict={'response': [1, 2, 3]}),
                        tool_call_id='123',
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='123',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
        ]
    )


async def test_request_tool_call(allow_model_requests: None):
    responses = [
        completion_message(
            [
                {
                    'toolUse': {
                        'toolUseId': '1',
                        'name': 'get_location',
                        'input': {'loc_name': 'San Francisco'},
                    }
                }
            ],
            usage={'inputTokens': 2, 'outputTokens': 1, 'totalTokens': 3},
        ),
        completion_message(
            [
                {
                    'toolUse': {
                        'toolUseId': '2',
                        'name': 'get_location',
                        'input': {'loc_name': 'London'},
                    }
                }
            ],
            usage={'inputTokens': 3, 'outputTokens': 2, 'totalTokens': 6},
        ),
        completion_message(
            [{'text': 'final response'}],
            usage={'inputTokens': 3, 'outputTokens': 5, 'totalTokens': 8},
        ),
    ]

    mock_client = MockBedrockClient.create_mock(responses)
    m = BedrockModel('us.amazon.nova-micro-v1:0', bedrock_client=mock_client)
    agent = Agent(m, system_prompt='this is the system prompt')

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')

    result = await agent.run('hello')
    assert result.data == 'final response'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='this is the system prompt'),
                    UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc)),
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args=ArgsDict(args_dict={'loc_name': 'San Francisco'}),
                        tool_call_id='1',
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Wrong location, please try again',
                        tool_name='get_location',
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args=ArgsDict(args_dict={'loc_name': 'London'}),
                        tool_call_id='2',
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_location',
                        content='{"lat": 51, "lng": 0}',
                        tool_call_id='2',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='final response')],
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )
