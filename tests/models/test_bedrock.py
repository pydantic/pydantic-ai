from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pydantic_ai
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

from pydantic_ai_bedrock.bedrock import BedrockModel

if TYPE_CHECKING:
    from botocore.eventstream import EventStream
    from mypy_boto3_bedrock_runtime.type_defs import (
        ContentBlockOutputTypeDef,
        ConverseResponseTypeDef,
        ConverseStreamOutputTypeDef,
        ConverseStreamResponseTypeDef,
        TokenUsageTypeDef,
    )

_HERE = Path(__file__).parent


def test_init():
    m = BedrockModel(
        "test",
        aws_access_key_id="foo",
        aws_secret_access_key="bar",
        region_name="us-east-1",
    )
    assert m.name() == "test"


@dataclass
class MockBedrockClient:
    completions: ConverseResponseTypeDef | list[ConverseResponseTypeDef] | None = None
    stream: list[ConverseStreamOutputTypeDef] | None = None
    index = 0

    def converse(self, *_args: Any, **_kwargs: Any) -> ConverseResponseTypeDef:
        if isinstance(self.completions, list):
            response = self.completions[self.index]
        else:
            response = self.completions
        self.index += 1
        return response

    def converse_stream(self, *_args: Any, **_kwargs: Any) -> ConverseStreamResponseTypeDef:
        @dataclass
        class MockEventStream:
            stream: list[ConverseStreamOutputTypeDef]

            def __iter__(self):
                for item in self.stream:
                    yield item

        return {
            "stream": MockEventStream(self.stream),
        }

    @classmethod
    def create_mock(
        cls, completions: ConverseResponseTypeDef | list[ConverseResponseTypeDef]
    ) -> MockBedrockClient:
        return cast(MockBedrockClient, cls(completions=completions))

    @classmethod
    def create_mock_stream(
        cls,
        stream: list[ConverseStreamOutputTypeDef],
    ) -> MockBedrockClient:
        return cast(MockBedrockClient, cls(stream=stream))


@pytest.fixture
def allow_model_requests():
    with pydantic_ai.models.override_allow_model_requests(True):
        yield


def completion_message(
    content: list[ContentBlockOutputTypeDef], usage: TokenUsageTypeDef
) -> ConverseResponseTypeDef:
    return {
        "ResponseMetadata": {
            "RequestId": "5f3335f9-edc7-4506-b899-33de742d7e90",
            "HTTPStatusCode": 200,
            "HTTPHeaders": {},
            "RetryAttempts": 0,
        },
        "output": {
            "message": {
                "role": "assistant",
                "content": content,
            }
        },
        "stopReason": "max_tokens",
        "usage": usage,
        "metrics": {"latencyMs": 1},
    }


async def test_sync_request_text_response(allow_model_requests: None):
    c = completion_message(
        [{"text": "world"}],
        {"inputTokens": 5, "outputTokens": 10, "totalTokens": 15},
    )
    mock_client = MockBedrockClient.create_mock(c)
    m = BedrockModel("test", bedrock_client=mock_client)
    agent = Agent(m)

    result = await agent.run("hello")
    assert result.data == "world"
    assert result.usage() == snapshot(
        Usage(requests=1, request_tokens=5, response_tokens=10, total_tokens=15)
    )

    # reset the index so we get the same response again
    mock_client.index = 0  # type: ignore

    result = await agent.run("hello", message_history=result.new_messages())
    assert result.data == "world"
    assert result.usage() == snapshot(
        Usage(requests=1, request_tokens=5, response_tokens=10, total_tokens=15)
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content="hello", timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(parts=[TextPart(content="world")], timestamp=IsNow(tz=timezone.utc)),
            ModelRequest(parts=[UserPromptPart(content="hello", timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(parts=[TextPart(content="world")], timestamp=IsNow(tz=timezone.utc)),
        ]
    )


async def test_async_request_text_response(allow_model_requests: None):
    c = completion_message(
        [{"text": "world"}],
        {"inputTokens": 3, "outputTokens": 5, "totalTokens": 8},
    )
    mock_client = MockBedrockClient.create_mock(c)
    m = BedrockModel("claude-3-5-haiku-latest", bedrock_client=mock_client)
    agent = Agent(m)

    result = await agent.run("hello")
    assert result.data == "world"
    assert result.usage() == snapshot(
        Usage(requests=1, request_tokens=3, response_tokens=5, total_tokens=8)
    )


async def test_request_structured_response(allow_model_requests: None):
    c = completion_message(
        [
            {
                "toolUse": {
                    "toolUseId": "123",
                    "name": "final_result",
                    "input": {"response": [1, 2, 3]},
                }
            }
        ],
        {"inputTokens": 3, "outputTokens": 5, "totalTokens": 8},
    )

    mock_client = MockBedrockClient.create_mock(c)
    m = BedrockModel("claude-3-5-haiku-latest", bedrock_client=mock_client)
    agent = Agent(m, result_type=list[int])

    result = await agent.run("hello")
    assert result.data == [1, 2, 3]
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content="hello",
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="final_result",
                        args=ArgsDict(args_dict={"response": [1, 2, 3]}),
                        tool_call_id="123",
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="final_result",
                        content="Final result processed.",
                        tool_call_id="123",
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
                    "toolUse": {
                        "toolUseId": "1",
                        "name": "get_location",
                        "input": {"loc_name": "San Francisco"},
                    }
                }
            ],
            usage={"inputTokens": 2, "outputTokens": 1, "totalTokens": 3},
        ),
        completion_message(
            [
                {
                    "toolUse": {
                        "toolUseId": "2",
                        "name": "get_location",
                        "input": {"loc_name": "London"},
                    }
                }
            ],
            usage={"inputTokens": 3, "outputTokens": 2, "totalTokens": 6},
        ),
        completion_message(
            [{"text": "final response"}],
            usage={"inputTokens": 3, "outputTokens": 5, "totalTokens": 8},
        ),
    ]

    mock_client = MockBedrockClient.create_mock(responses)
    m = BedrockModel("test", bedrock_client=mock_client)
    agent = Agent(m, system_prompt="this is the system prompt")

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == "London":
            return json.dumps({"lat": 51, "lng": 0})
        else:
            raise ModelRetry("Wrong location, please try again")

    result = await agent.run("hello")
    assert result.data == "final response"
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content="this is the system prompt"),
                    UserPromptPart(content="hello", timestamp=IsNow(tz=timezone.utc)),
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="get_location",
                        args=ArgsDict(args_dict={"loc_name": "San Francisco"}),
                        tool_call_id="1",
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content="Wrong location, please try again",
                        tool_name="get_location",
                        tool_call_id="1",
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="get_location",
                        args=ArgsDict(args_dict={"loc_name": "London"}),
                        tool_call_id="2",
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="get_location",
                        content='{"lat": 51, "lng": 0}',
                        tool_call_id="2",
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content="final response")],
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )


from pydantic import BaseModel


class QuestionResponse(BaseModel):
    answer: str


async def test_stream_tool(allow_model_requests: None):
    response_file = _HERE / "stream_tool_responses.jsons"
    message = []
    for line in response_file.read_text().splitlines():
        message.append(json.loads(line))

    mock_client = MockBedrockClient.create_mock_stream(message)
    m = BedrockModel("test", bedrock_client=mock_client)
    agent = Agent(m, result_type=QuestionResponse)

    async with agent.run_stream("What is the capital of the UK?") as result:
        assert not result.is_complete
        assert [c async for c in result.stream(debounce_by=None)] == snapshot(
            [QuestionResponse(answer="London"), QuestionResponse(answer="London")]
        )
        assert result.is_complete
        assert result.usage() == snapshot(
            Usage(requests=1, request_tokens=417, response_tokens=76, total_tokens=493)
        )


async def test_stream_text(allow_model_requests: None):
    response_file = _HERE / "stream_text_responses.jsons"
    message = []
    for line in response_file.read_text().splitlines():
        message.append(json.loads(line))

    mock_client = MockBedrockClient.create_mock_stream(message)
    m = BedrockModel("test", bedrock_client=mock_client)
    agent = Agent(m)

    async with agent.run_stream("What is the capital of the UK?") as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(
            [
                "Hello",
                "Hello!",
                "Hello! How",
                "Hello! How can",
                "Hello! How can I",
                "Hello! How can I assist",
                "Hello! How can I assist you",
                "Hello! How can I assist you today",
                "Hello! How can I assist you today?",
                "Hello! How can I assist you today? If",
                "Hello! How can I assist you today? If you",
                "Hello! How can I assist you today? If you have",
                "Hello! How can I assist you today? If you have any",
                "Hello! How can I assist you today? If you have any questions",
                "Hello! How can I assist you today? If you have any questions or",
                "Hello! How can I assist you today? If you have any questions or need",
                "Hello! How can I assist you today? If you have any questions or need help",
                "Hello! How can I assist you today? If you have any questions or need help with",
                "Hello! How can I assist you today? If you have any questions or need help with something",
                "Hello! How can I assist you today? If you have any questions or need help with something,",
                "Hello! How can I assist you today? If you have any questions or need help with something, feel",
                "Hello! How can I assist you today? If you have any questions or need help with something, feel free",
                "Hello! How can I assist you today? If you have any questions or need help with something, feel free to",
                "Hello! How can I assist you today? If you have any questions or need help with something, feel free to let",
                "Hello! How can I assist you today? If you have any questions or need help with something, feel free to let me",
                "Hello! How can I assist you today? If you have any questions or need help with something, feel free to let me know",
                "Hello! How can I assist you today? If you have any questions or need help with something, feel free to let me know.",
            ]
        )
        assert result.is_complete
        assert result.usage() == snapshot(
            Usage(requests=1, request_tokens=7, response_tokens=27, total_tokens=34)
        )
