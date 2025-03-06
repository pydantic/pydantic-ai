import datetime
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
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='bedrock not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


async def test_bedrock_model(allow_model_requests: None):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=BedrockProvider())
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


async def test_bedrock_model_structured_response(allow_model_requests: None):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0')
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
    assert result.usage() == snapshot(Usage(requests=2, request_tokens=1237, response_tokens=304, total_tokens=1541))
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
                        content='<thinking> To find the temperature in London on 1st January 2022, I will use the "temperature" tool. I will need to provide the date and the city name. The date is already provided as "1st January 2022" and the city name is "London". I will call the "temperature" tool with these parameters.</thinking>\n'
                    ),
                    ToolCallPart(
                        tool_name='temperature',
                        args={'date': '2022-01-01', 'city': 'London'},
                        tool_call_id='tooluse_72V1-POFTp2399ewBfduMQ',
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
                        tool_call_id='tooluse_72V1-POFTp2399ewBfduMQ',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='<thinking> The "temperature" tool has returned the temperature for the specified date and city. The temperature in London on 1st January 2022 was 30°C. Now I will use the "final_result" tool to provide this information to the user.</thinking> '
                    ),
                    ToolCallPart(
                        tool_name='final_result',
                        args={'date': '2022-01-01', 'city': 'London', 'temperature': '30°C'},
                        tool_call_id='tooluse_TPjEx1BESIqmrElHJ0b3sQ',
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
                        tool_call_id='tooluse_TPjEx1BESIqmrElHJ0b3sQ',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
        ]
    )
