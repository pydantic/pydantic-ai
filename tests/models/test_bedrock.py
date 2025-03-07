from __future__ import annotations as _annotations

import datetime
import os
from typing import TypedDict

import pytest
from inline_snapshot import snapshot

from pydantic_ai.agent import Agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import Usage

from ..conftest import IsDatetime, try_import

with try_import() as imports_successful:
    import boto3

    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='bedrock not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


@pytest.fixture
def bedrock_provider():
    bedrock_client = boto3.client(  # type: ignore[reportUnknownMemberType]
        'bedrock-runtime',
        region_name=os.getenv('AWS_REGION', 'us-east-1'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'AKIA6666666666666666'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', '6666666666666666666666666666666666666666'),
    )
    yield BedrockProvider(bedrock_client=bedrock_client)
    bedrock_client.close()


async def test_bedrock_model(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(model=model, system_prompt='You are a chatbot.')

    result = await agent.run('Hello!')
    assert result.data == snapshot(
        "Hello! How can I assist you today? Whether you have questions, need information, or just want to chat, I'm here to help."
    )
    assert result.usage() == snapshot(Usage(requests=1, request_tokens=7, response_tokens=30, total_tokens=37))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='You are a chatbot.'),
                    UserPromptPart(
                        content='Hello!',
                        timestamp=IsDatetime(),
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="Hello! How can I assist you today? Whether you have questions, need information, or just want to chat, I'm here to help."
                    )
                ],
                model_name='us.amazon.nova-micro-v1:0',
                timestamp=IsDatetime(),
            ),
        ]
    )


async def test_bedrock_model_structured_response(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', retries=5)

    class Response(TypedDict):
        temperature: str
        date: datetime.date
        city: str

    @agent.tool_plain
    async def temperature(city: str, date: datetime.date) -> str:
        """Get the temperature in a city on a specific date.

        Args:
            city: The city name.
            date: The date.

        Returns:
            The temperature in degrees Celsius.
        """
        return '30°C'

    result = await agent.run('What was the temperature in London 1st January 2022?', result_type=Response)
    assert result.data == snapshot({'temperature': '30°C', 'date': datetime.date(2022, 1, 1), 'city': 'London'})
    assert result.usage() == snapshot(Usage(requests=2, request_tokens=1238, response_tokens=311, total_tokens=1549))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='You are a helpful chatbot.'),
                    UserPromptPart(
                        content='What was the temperature in London 1st January 2022?',
                        timestamp=IsDatetime(),
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='<thinking> To find the temperature in London on 1st January 2022, I need to use the "temperature" tool with the appropriate arguments. The required arguments are "date" and "city". The date provided is "1st January 2022" and the city is "London". I will use these to call the "temperature" tool.</thinking>\n'
                    ),
                    ToolCallPart(
                        tool_name='temperature',
                        args={'date': '2022-01-01', 'city': 'London'},
                        tool_call_id='tooluse_caKIVK5IT_OcfrKtKLgM4w',
                    ),
                ],
                model_name='us.amazon.nova-micro-v1:0',
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='temperature',
                        content='30°C',
                        tool_call_id='tooluse_caKIVK5IT_OcfrKtKLgM4w',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='<thinking> The "temperature" tool has provided the temperature information for the specified date and city. The temperature in London on 1st January 2022 was 30°C. Now, I will use the "final_result" tool to present this information to the user in the required format.</thinking> '
                    ),
                    ToolCallPart(
                        tool_name='final_result',
                        args={'date': '2022-01-01', 'city': 'London', 'temperature': '30°C'},
                        tool_call_id='tooluse_orpaL-zsTGCl0L818gkMRQ',
                    ),
                ],
                model_name='us.amazon.nova-micro-v1:0',
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='tooluse_orpaL-zsTGCl0L818gkMRQ',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
        ]
    )
