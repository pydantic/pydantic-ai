from __future__ import annotations as _annotations

import datetime
import os

import pytest
from inline_snapshot import snapshot
from typing_extensions import TypedDict

from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import (
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
    assert result.usage() == snapshot(Usage(requests=2, request_tokens=1236, response_tokens=298, total_tokens=1534))
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
                        content='<thinking> To find the temperature in London on 1st January 2022, I will use the "temperature" tool. I need to provide the date and the city name. The date is already provided as "1st January 2022" and the city name is "London". I will call the "temperature" tool with these parameters.</thinking>\n'
                    ),
                    ToolCallPart(
                        tool_name='temperature',
                        args={'date': '2022-01-01', 'city': 'London'},
                        tool_call_id='tooluse_5WEci1UmQ8ifMFkUcy2gHQ',
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
                        tool_call_id='tooluse_5WEci1UmQ8ifMFkUcy2gHQ',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='<thinking> I have received the result from the "temperature" tool. The temperature in London on 1st January 2022 was 30°C. Now, I will use the "final_result" tool to provide this information to the user.</thinking> '
                    ),
                    ToolCallPart(
                        tool_name='final_result',
                        args={'date': '2022-01-01', 'city': 'London', 'temperature': '30°C'},
                        tool_call_id='tooluse_9AjloJSaQDKmpPFff-2Clg',
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
                        tool_call_id='tooluse_9AjloJSaQDKmpPFff-2Clg',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
        ]
    )


async def test_bedrock_model_stream(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings={'temperature': 0.0})
    async with agent.run_stream('What is the capital of France?') as result:
        data = await result.get_data()
    assert data == snapshot(
        'The capital of France is Paris. Paris is not only the capital city but also the most populous city in France, known for its significant cultural, political, and economic influence. It is famous for landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral, among many other attractions.'
    )


async def test_bedrock_model_anthropic_model_with_tools(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('anthropic.claude-v2', provider=bedrock_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings={'temperature': 0.0})

    @agent.tool_plain
    async def get_current_temperature(city: str) -> str:
        """Get the current temperature in a city.

        Args:
            city: The city name.

        Returns:
            The current temperature in degrees Celsius.
        """
        return '30°C'

    # TODO(Marcelo): Anthropic models don't support tools on the Bedrock Converse Interface.
    # I'm unsure what to do, so for the time being I'm just documenting the test. Let's see if someone complains.
    with pytest.raises(Exception):
        await agent.run('What is the current temperature in London?')


async def test_bedrock_model_anthropic_model_without_tools(
    allow_model_requests: None, bedrock_provider: BedrockProvider
):
    model = BedrockConverseModel('anthropic.claude-v2', provider=bedrock_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings={'temperature': 0.0})
    result = await agent.run('What is the capital of France?')
    assert result.data == snapshot('Paris is the capital of France.')


async def test_bedrock_model_retry(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(
        model=model, system_prompt='You are a helpful chatbot.', model_settings={'temperature': 0.0}, retries=2
    )

    @agent.tool_plain
    async def get_capital(country: str) -> str:
        """Get the capital of a country.

        Args:
            country: The country name.
        """
        raise ModelRetry('The country is not supported.')

    result = await agent.run('What is the capital of France?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='You are a helpful chatbot.'),
                    UserPromptPart(
                        content='What is the capital of France?',
                        timestamp=datetime.datetime(2025, 3, 7, 9, 18, 24, 818226, tzinfo=datetime.timezone.utc),
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='<thinking> To find the capital of France, I will use the available tool "get_capital". I will input the country name "France" into the tool. </thinking>\n'
                    ),
                    ToolCallPart(
                        tool_name='get_capital',
                        args={'country': 'France'},
                        tool_call_id='tooluse_F8LnaCMtQ0-chKTnPhNH2g',
                    ),
                ],
                model_name='us.amazon.nova-micro-v1:0',
                timestamp=datetime.datetime(2025, 3, 7, 9, 18, 25, 614733, tzinfo=datetime.timezone.utc),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='The country is not supported.',
                        tool_name='get_capital',
                        tool_call_id='tooluse_F8LnaCMtQ0-chKTnPhNH2g',
                        timestamp=datetime.datetime(2025, 3, 7, 9, 18, 25, 615589, tzinfo=datetime.timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
<thinking> It seems there was an error in retrieving the capital of France. The tool returned a message saying "The country is not supported." This indicates that the tool does not support the country France. I will inform the user about this limitation and suggest alternative ways to find the information. </thinking>

I'm sorry, but the tool I have does not support retrieving the capital of France. However, I can tell you that the capital of France is Paris. If you need information on a different country, please let me know!\
"""
                    )
                ],
                model_name='us.amazon.nova-micro-v1:0',
                timestamp=datetime.datetime(2025, 3, 7, 9, 18, 26, 642657, tzinfo=datetime.timezone.utc),
            ),
        ]
    )
