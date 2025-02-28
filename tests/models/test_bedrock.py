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


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='bedrock not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


async def test_bedrock_model():
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0')
    agent = Agent(model=model, system_prompt='You are a chatbot.')

    result = await agent.run('Hello!')
    assert result.data == snapshot(
        "Hello! How can I assist you today? Whether you have questions, need information, or just want to chat, I'm here for you."
    )
    assert result.usage() == snapshot(Usage(requests=1, request_tokens=7, response_tokens=30, total_tokens=37))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='You are a chatbot.'),
                    UserPromptPart(content='Hello!', timestamp=IsDatetime()),
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="Hello! How can I assist you today? Whether you have questions, need information, or just want to chat, I'm here for you."
                    )
                ],
                model_name='us.amazon.nova-micro-v1:0',
                timestamp=IsDatetime(),
            ),
        ]
    )


async def test_bedrock_model_structured_response():
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
    assert result.usage() == snapshot(Usage(requests=2, request_tokens=1202, response_tokens=297, total_tokens=1499))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='You are a helpful chatbot.'),
                    UserPromptPart(
                        content='What was the temperature in London 1st January 2022?', timestamp=IsDatetime()
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='<thinking> To find the temperature in London on 1st January 2022, I will use the "temperature" tool. I need to provide the city name and the specific date. </thinking>\n'
                    ),
                    ToolCallPart(
                        tool_name='temperature',
                        args={'date': '2022-01-01', 'city': 'London'},
                        tool_call_id='tooluse_SVRG-CVsTVScs_3kz6VpJA',
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
                        tool_call_id='tooluse_SVRG-CVsTVScs_3kz6VpJA',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='<thinking> I have retrieved the temperature information using the "temperature" tool. Now, I will present the information to the user in the format required by the "final_result" tool. </thinking> '
                    ),
                    ToolCallPart(
                        tool_name='final_result',
                        args={'date': '2022-01-01', 'city': 'London', 'temperature': '30°C'},
                        tool_call_id='tooluse_Gbzm_o2nSACp037rxte_tA',
                    ),
                    TextPart(
                        content="""\


The temperature in London on 1st January 2022 was **30°C**.\
"""
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
                        tool_call_id='tooluse_Gbzm_o2nSACp037rxte_tA',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
        ]
    )
