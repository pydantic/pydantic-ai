"""Tests for OpenAI Responses model: thinking/reasoning capabilities and related features."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, cast

import pytest
from inline_snapshot import snapshot

from pydantic_ai import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ModelRequest,
    ModelResponse,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.agent import Agent
from pydantic_ai.builtin_tools import CodeExecutionTool
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    BuiltinToolCallEvent,  # pyright: ignore[reportDeprecated]
    BuiltinToolResultEvent,  # pyright: ignore[reportDeprecated]
    FinalResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.usage import RequestUsage

from ...conftest import IsDatetime, IsNow, IsStr, try_import
from ..mock_openai import MockOpenAIResponses, response_message

with try_import() as imports_successful:
    from openai.types.responses.response_output_message import Content, ResponseOutputMessage, ResponseOutputText
    from openai.types.responses.response_reasoning_item import ResponseReasoningItem, Summary

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.models.openai import (
        OpenAIResponsesModel,
        OpenAIResponsesModelSettings,
    )
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


async def test_openai_responses_reasoning_effort(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('o3-mini', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, model_settings=OpenAIResponsesModelSettings(openai_reasoning_effort='low'))
    result = await agent.run(
        'Explain me how to cook uruguayan alfajor. Do not send whitespaces at the end of the lines.'
    )
    assert [line.strip() for line in result.output.splitlines()] == snapshot(
        [
            'Ingredients for the dough:',
            '• 300 g cornstarch',
            '• 200 g flour',
            '• 150 g powdered sugar',
            '• 200 g unsalted butter',
            '• 3 egg yolks',
            '• Zest of 1 lemon',
            '• 1 teaspoon vanilla extract',
            '• A pinch of salt',
            '',
            'Ingredients for the filling (dulce de leche):',
            '• 400 g dulce de leche',
            '',
            'Optional coating:',
            '• Powdered sugar for dusting',
            '• Grated coconut',
            '• Crushed peanuts or walnuts',
            '• Melted chocolate',
            '',
            'Steps:',
            '1. In a bowl, mix together the cornstarch, flour, powdered sugar, and salt.',
            '2. Add the unsalted butter cut into small pieces. Work it into the dry ingredients until the mixture resembles coarse breadcrumbs.',
            '3. Incorporate the egg yolks, lemon zest, and vanilla extract. Mix until you obtain a smooth and homogeneous dough.',
            '4. Wrap the dough in plastic wrap and let it rest in the refrigerator for at least one hour.',
            '5. Meanwhile, prepare a clean workspace by lightly dusting it with flour.',
            '6. Roll out the dough on the working surface until it is about 0.5 cm thick.',
            '7. Use a round cutter (approximately 3-4 cm in diameter) to cut out circles. Re-roll any scraps to maximize the number of cookies.',
            '8. Arrange the circles on a baking sheet lined with parchment paper.',
            '9. Preheat the oven to 180°C (350°F) and bake the cookies for about 10-12 minutes until they are lightly golden at the edges. They should remain soft.',
            '10. Remove the cookies from the oven and allow them to cool completely on a rack.',
            '11. Once the cookies are cool, spread dulce de leche on the flat side of one cookie and sandwich it with another.',
            '12. If desired, roll the edges of the alfajores in powdered sugar, grated coconut, crushed nuts, or dip them in melted chocolate.',
            '13. Allow any coatings to set before serving.',
            '',
            'Enjoy your homemade Uruguayan alfajores!',
        ]
    )


async def test_openai_responses_reasoning_generate_summary(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('computer-use-preview', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(
        model=model,
        model_settings=OpenAIResponsesModelSettings(
            openai_reasoning_summary='concise',
            openai_truncation='auto',
        ),
    )
    result = await agent.run('What should I do to cross the street?')
    assert result.output == snapshot("""\
To cross the street safely, follow these steps:

1. **Use a Crosswalk**: Always use a designated crosswalk or pedestrian crossing whenever available.
2. **Press the Button**: If there is a pedestrian signal button, press it and wait for the signal.
3. **Look Both Ways**: Look left, right, and left again before stepping off the curb.
4. **Wait for the Signal**: Cross only when the pedestrian signal indicates it is safe to do so or when there is a clear gap in traffic.
5. **Stay Alert**: Be mindful of turning vehicles and stay attentive while crossing.
6. **Walk, Don't Run**: Walk across the street; running can increase the risk of falling or not noticing an oncoming vehicle.

Always follow local traffic rules and be cautious, even when crossing at a crosswalk. Safety is the priority.\
""")


async def test_reasoning_model_with_temperature(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('o3-mini', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m, model_settings=OpenAIResponsesModelSettings(temperature=0.5))
    result = await agent.run('What is the capital of Mexico?')
    assert result.output == snapshot(
        'The capital of Mexico is Mexico City. It serves as the political, cultural, and economic heart of the country and is one of the largest metropolitan areas in the world.'
    )


@pytest.mark.vcr()
async def test_gpt5_pro(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-5-pro', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)
    result = await agent.run('What is the capital of Mexico?')
    assert result.output == snapshot('Mexico City (Ciudad de México).')


async def test_openai_responses_model_thinking_part(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    settings = OpenAIResponsesModelSettings(openai_reasoning_effort='high', openai_reasoning_summary='detailed')
    agent = Agent(m, model_settings=settings)

    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='How do I cross the street?', timestamp=IsDatetime())],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42c90b950819c9e32c46d4f8326ca07460311b0c8d3de',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ThinkingPart(content=IsStr(), id='rs_68c42c90b950819c9e32c46d4f8326ca07460311b0c8d3de'),
                    ThinkingPart(content=IsStr(), id='rs_68c42c90b950819c9e32c46d4f8326ca07460311b0c8d3de'),
                    ThinkingPart(content=IsStr(), id='rs_68c42c90b950819c9e32c46d4f8326ca07460311b0c8d3de'),
                    ThinkingPart(content=IsStr(), id='rs_68c42c90b950819c9e32c46d4f8326ca07460311b0c8d3de'),
                    ThinkingPart(content=IsStr(), id='rs_68c42c90b950819c9e32c46d4f8326ca07460311b0c8d3de'),
                    TextPart(
                        content=IsStr(),
                        id='msg_68c42cb1aaec819cb992bd92a8c7766007460311b0c8d3de',
                    ),
                ],
                usage=RequestUsage(input_tokens=13, output_tokens=2199, details={'reasoning_tokens': 1920}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 12, 14, 22, 8, tzinfo=timezone.utc),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run(
        'Considering the way to cross the street, analogously, how do I cross the river?',
        message_history=result.all_messages(),
    )
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Considering the way to cross the street, analogously, how do I cross the river?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42cb43d3c819caf078978cc2514ea07460311b0c8d3de',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ThinkingPart(content=IsStr(), id='rs_68c42cb43d3c819caf078978cc2514ea07460311b0c8d3de'),
                    ThinkingPart(content=IsStr(), id='rs_68c42cb43d3c819caf078978cc2514ea07460311b0c8d3de'),
                    ThinkingPart(content=IsStr(), id='rs_68c42cb43d3c819caf078978cc2514ea07460311b0c8d3de'),
                    ThinkingPart(content=IsStr(), id='rs_68c42cb43d3c819caf078978cc2514ea07460311b0c8d3de'),
                    TextPart(
                        content=IsStr(),
                        id='msg_68c42cd36134819c800463490961f7df07460311b0c8d3de',
                    ),
                ],
                usage=RequestUsage(input_tokens=314, output_tokens=2737, details={'reasoning_tokens': 2112}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 12, 14, 22, 43, tzinfo=timezone.utc),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_thinking_part_from_other_model(
    allow_model_requests: None, anthropic_api_key: str, openai_api_key: str
):
    m = AnthropicModel(
        'claude-sonnet-4-0',
        provider=AnthropicProvider(api_key=anthropic_api_key),
        settings=AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024}),
    )
    agent = Agent(m)

    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='How do I cross the street?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        signature=IsStr(),
                        provider_name='anthropic',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=42,
                    output_tokens=291,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 42,
                        'output_tokens': 291,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run(
        'Considering the way to cross the street, analogously, how do I cross the river?',
        model=OpenAIResponsesModel(
            'gpt-5',
            provider=OpenAIProvider(api_key=openai_api_key),
            settings=OpenAIResponsesModelSettings(openai_reasoning_effort='high', openai_reasoning_summary='detailed'),
        ),
        message_history=result.all_messages(),
    )
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Considering the way to cross the street, analogously, how do I cross the river?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42ce323d48193bcf88db6278980cf0ad492c7955fc6fc',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ThinkingPart(content=IsStr(), id='rs_68c42ce323d48193bcf88db6278980cf0ad492c7955fc6fc'),
                    ThinkingPart(content=IsStr(), id='rs_68c42ce323d48193bcf88db6278980cf0ad492c7955fc6fc'),
                    ThinkingPart(content=IsStr(), id='rs_68c42ce323d48193bcf88db6278980cf0ad492c7955fc6fc'),
                    ThinkingPart(content=IsStr(), id='rs_68c42ce323d48193bcf88db6278980cf0ad492c7955fc6fc'),
                    ThinkingPart(content=IsStr(), id='rs_68c42ce323d48193bcf88db6278980cf0ad492c7955fc6fc'),
                    TextPart(content=IsStr(), id='msg_68c42d0b5e5c819385352dde1f447d910ad492c7955fc6fc'),
                ],
                usage=RequestUsage(input_tokens=306, output_tokens=3134, details={'reasoning_tokens': 2496}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 12, 14, 23, 30, tzinfo=timezone.utc),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_thinking_part_iter(allow_model_requests: None, openai_api_key: str):
    provider = OpenAIProvider(api_key=openai_api_key)
    responses_model = OpenAIResponsesModel('o3-mini', provider=provider)
    settings = OpenAIResponsesModelSettings(openai_reasoning_effort='high', openai_reasoning_summary='detailed')
    agent = Agent(responses_model, model_settings=settings)

    async with agent.iter(user_prompt='How do I cross the street?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for _ in request_stream:
                        pass

    assert agent_run.result is not None
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='How do I cross the street?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42d1d0878819d8266007cd3d1402c08fbf9b1584184ff',
                        signature='gAAAAABoxC0m_QWpOlSt8wyPk_gtnjiI4mNLOryYlNXO-6rrVeIqBYDDAyMVg2_ldboZvfhW8baVbpki29gkTAyNygTr7L8gF1XK0hFovoa23ZYJKvuOnyLIJF-rXCsbDG7YdMYhi3bm82pMFVQxNK4r5muWCQcHmyJ2S1YtBoJtF_D1Ah7GpW2ACvJWsGikb3neAOnI-RsmUxCRu-cew7rVWfSj8jFKs8RGNQRvDaUzVniaMXJxVW9T5C7Ytzi852MF1PfVq0U-aNBzZBtAdwQcbn5KZtGkYLYTChmCi2hMrh5-lg9CgS8pqqY9-jv2EQvKHIumdv6oLiW8K59Zvo8zGxYoqT--osfjfS0vPZhTHiSX4qCkK30YNJrWHKJ95Hpe23fnPBL0nEQE5l6XdhsyY7TwMom016P3dgWwgP5AtWmQ30zeXDs=',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42d1d0878819d8266007cd3d1402c08fbf9b1584184ff',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42d1d0878819d8266007cd3d1402c08fbf9b1584184ff',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42d1d0878819d8266007cd3d1402c08fbf9b1584184ff',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_68c42d26866c819da8d5c606621c911608fbf9b1584184ff',
                    ),
                ],
                usage=RequestUsage(input_tokens=13, output_tokens=1680, details={'reasoning_tokens': 1408}),
                model_name='o3-mini-2025-01-31',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 12, 14, 24, 15, tzinfo=timezone.utc),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_thinking_with_tool_calls(allow_model_requests: None, openai_api_key: str):
    provider = OpenAIProvider(api_key=openai_api_key)
    m = OpenAIResponsesModel(
        model_name='gpt-5',
        provider=provider,
        settings=OpenAIResponsesModelSettings(openai_reasoning_summary='detailed', openai_reasoning_effort='low'),
    )
    agent = Agent(model=m)

    @agent.instructions
    def system_prompt():
        return (
            'You are a helpful assistant that uses planning. You MUST use the update_plan tool and continually '
            "update it as you make progress against the user's prompt"
        )

    @agent.tool_plain
    def update_plan(plan: str) -> str:
        return 'plan updated'

    prompt = (
        'Compose a 12-line poem where the first letters of the odd-numbered lines form the name "SAMIRA" '
        'and the first letters of the even-numbered lines spell out "DAWOOD." Additionally, the first letter '
        'of each word in every line should create the capital of a country'
    )

    result = await agent.run(prompt)

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Compose a 12-line poem where the first letters of the odd-numbered lines form the name "SAMIRA" and the first letters of the even-numbered lines spell out "DAWOOD." Additionally, the first letter of each word in every line should create the capital of a country',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions="You are a helpful assistant that uses planning. You MUST use the update_plan tool and continually update it as you make progress against the user's prompt",
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42d29124881968e24c1ca8c1fc7860e8bc41441c948f6',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ThinkingPart(content=IsStr(), id='rs_68c42d29124881968e24c1ca8c1fc7860e8bc41441c948f6'),
                    ThinkingPart(content=IsStr(), id='rs_68c42d29124881968e24c1ca8c1fc7860e8bc41441c948f6'),
                    ThinkingPart(content=IsStr(), id='rs_68c42d29124881968e24c1ca8c1fc7860e8bc41441c948f6'),
                    ThinkingPart(content=IsStr(), id='rs_68c42d29124881968e24c1ca8c1fc7860e8bc41441c948f6'),
                    ToolCallPart(
                        tool_name='update_plan',
                        args=IsStr(),
                        tool_call_id='call_gL7JE6GDeGGsFubqO2XGytyO',
                        id='fc_68c42d3e9e4881968b15fbb8253f58540e8bc41441c948f6',
                    ),
                ],
                usage=RequestUsage(input_tokens=124, output_tokens=1926, details={'reasoning_tokens': 1792}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 12, 14, 24, 40, tzinfo=timezone.utc),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='update_plan',
                        content='plan updated',
                        tool_call_id='call_gL7JE6GDeGGsFubqO2XGytyO',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions="You are a helpful assistant that uses planning. You MUST use the update_plan tool and continually update it as you make progress against the user's prompt",
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content=IsStr(), id='msg_68c42d408eec8196ae1c5883e07c093e0e8bc41441c948f6')],
                usage=RequestUsage(
                    input_tokens=2087, cache_read_tokens=2048, output_tokens=124, details={'reasoning_tokens': 0}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 12, 14, 25, 3, tzinfo=timezone.utc),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_thinking_without_summary(allow_model_requests: None):
    c = response_message(
        [
            ResponseReasoningItem(
                id='rs_123',
                summary=[],
                type='reasoning',
                encrypted_content='123',
            ),
            ResponseOutputMessage(
                id='msg_123',
                content=cast(list[Content], [ResponseOutputText(text='4', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            ),
        ],
    )
    mock_client = MockOpenAIResponses.create_mock(c)
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(openai_client=mock_client))

    agent = Agent(model=model)
    result = await agent.run('What is 2+2?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='', id='rs_123', signature='123', provider_name='openai'),
                    TextPart(content='4', id='msg_123'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
                provider_response_id=IsStr(),
                run_id=IsStr(),
            ),
        ]
    )

    _, openai_messages = await model._map_messages(  # type: ignore[reportPrivateUsage]
        result.all_messages(),
        model_settings=cast(OpenAIResponsesModelSettings, model.settings or {}),
        model_request_parameters=ModelRequestParameters(),
    )
    assert openai_messages == snapshot(
        [
            {'role': 'user', 'content': 'What is 2+2?'},
            {'id': 'rs_123', 'summary': [], 'encrypted_content': '123', 'type': 'reasoning'},
            {
                'role': 'assistant',
                'id': 'msg_123',
                'content': [{'text': '4', 'type': 'output_text', 'annotations': []}],
                'type': 'message',
                'status': 'completed',
            },
        ]
    )


async def test_openai_responses_thinking_with_multiple_summaries(allow_model_requests: None):
    c = response_message(
        [
            ResponseReasoningItem(
                id='rs_123',
                summary=[
                    Summary(text='1', type='summary_text'),
                    Summary(text='2', type='summary_text'),
                    Summary(text='3', type='summary_text'),
                    Summary(text='4', type='summary_text'),
                ],
                type='reasoning',
                encrypted_content='123',
            ),
            ResponseOutputMessage(
                id='msg_123',
                content=cast(list[Content], [ResponseOutputText(text='4', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            ),
        ],
    )
    mock_client = MockOpenAIResponses.create_mock(c)
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(openai_client=mock_client))

    agent = Agent(model=model)
    result = await agent.run('What is 2+2?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='1', id='rs_123', signature='123', provider_name='openai'),
                    ThinkingPart(content='2', id='rs_123'),
                    ThinkingPart(content='3', id='rs_123'),
                    ThinkingPart(content='4', id='rs_123'),
                    TextPart(content='4', id='msg_123'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
                provider_response_id=IsStr(),
                run_id=IsStr(),
            ),
        ]
    )

    _, openai_messages = await model._map_messages(  # type: ignore[reportPrivateUsage]
        result.all_messages(),
        model_settings=cast(OpenAIResponsesModelSettings, model.settings or {}),
        model_request_parameters=ModelRequestParameters(),
    )
    assert openai_messages == snapshot(
        [
            {'role': 'user', 'content': 'What is 2+2?'},
            {
                'id': 'rs_123',
                'summary': [
                    {'text': '1', 'type': 'summary_text'},
                    {'text': '2', 'type': 'summary_text'},
                    {'text': '3', 'type': 'summary_text'},
                    {'text': '4', 'type': 'summary_text'},
                ],
                'encrypted_content': '123',
                'type': 'reasoning',
            },
            {
                'role': 'assistant',
                'id': 'msg_123',
                'content': [{'text': '4', 'type': 'output_text', 'annotations': []}],
                'type': 'message',
                'status': 'completed',
            },
        ]
    )


async def test_openai_responses_thinking_with_modified_history(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    settings = OpenAIResponsesModelSettings(openai_reasoning_effort='low', openai_reasoning_summary='detailed')
    agent = Agent(m, model_settings=settings)

    result = await agent.run('What is the meaning of life?')
    messages = result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the meaning of life?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42de022c881948db7ed1cc2529f2e0202c9ad459e0d23',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    TextPart(content=IsStr(), id='msg_68c42de31d348194a251b43ad913ef140202c9ad459e0d23'),
                ],
                usage=RequestUsage(input_tokens=13, output_tokens=248, details={'reasoning_tokens': 64}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 12, 14, 27, 43, tzinfo=timezone.utc),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    response = messages[-1]
    assert isinstance(response, ModelResponse)
    assert isinstance(response.parts, list)
    response.parts[1] = TextPart(content='The meaning of life is 42')

    with pytest.raises(
        ModelHTTPError,
        match=r"Item '.*' of type 'reasoning' was provided without its required following item\.",
    ):
        await agent.run('Anything to add?', message_history=messages)

    result = await agent.run(
        'Anything to add?',
        message_history=messages,
        model_settings=OpenAIResponsesModelSettings(
            openai_reasoning_effort='low',
            openai_reasoning_summary='detailed',
            openai_send_reasoning_ids=False,
        ),
    )
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Anything to add?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42de4f63c819fb31b6019a4eaf67c051f82c608a83beb',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    TextPart(content=IsStr(), id='msg_68c42de8a410819faf7a9cbebd2b4bc4051f82c608a83beb'),
                ],
                usage=RequestUsage(input_tokens=142, output_tokens=355, details={'reasoning_tokens': 128}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 12, 14, 27, 48, tzinfo=timezone.utc),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_thinking_with_code_execution_tool(allow_model_requests: None, openai_api_key: str):
    provider = OpenAIProvider(api_key=openai_api_key)
    m = OpenAIResponsesModel(
        model_name='gpt-5',
        provider=provider,
        settings=OpenAIResponsesModelSettings(
            openai_reasoning_summary='detailed',
            openai_reasoning_effort='low',
            openai_include_code_execution_outputs=True,
        ),
    )
    agent = Agent(model=m, builtin_tools=[CodeExecutionTool()])

    result = await agent.run(user_prompt='what is 65465-6544 * 65464-6+1.02255')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='what is 65465-6544 * 65464-6+1.02255',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68cdba57390881a3b7ef1d2de5c8499709b7445677780c8f',
                        signature='gAAAAABozbpoKwjspVdWvC2skgCFSKx1Fiw9QGDrOxixFaC8O5gPVmC35FfE2jaedsn0zsHctrsl2LvPt7ELnOB3N20bvDGcDHkYzjSOLpf1jl2IAtQrkPWuLPOb6h8mIPL-Z1wNrngsmuoaKP0rrAcGwDwKzq8hxpLQbjvpRib-bbaVQ0SX7KHDpbOuEam3bIEiNSCNsA1Ot54R091vvwInnCCDMWVj-9u2fn7xtNzRGjHorkAt9mOhOBIVgZNZHnWb4RQ-PaYccgi44-gtwOK_2rhI9Qo0JiCBJ9PDdblms0EzBE7vfAWrCvnb_jKiEmKf2x9BBv3GMydsgnTCJdbBf6UVaMUnth1GvnDuJBdV12ecNT2LhOF2JNs3QjlbdDx661cnNoCDpNhXpdH3bL0Gncl7VApVY3iT2vRw4AJCU9U4xVdHeWb5GYz-sgkTgjbgEGg_RiU42taKsdm6B2gvc5_Pqf4g6WTdq-BNCwOjXQ4DatQBiJkgV5kyg4PqUqr35AD05wiSwz6reIsdnxDEqtWv4gBJWfGj4I96YqkL9YEuIBKORJ7ArZnjE5PSv6TIhqW-X9mmQTGkXl8emxpbdsNfow3QEd_l8rQEo4fHiFOGwU-uuPCikx7v6vDsE-w_fiZTFkM0X4iwFb6NXvOxKSdigfUgDfeCySwfmxtMx67QuoRA4xbfSHI9cctr-guZwMIIsMmKnTT-qGp-0F4UiyRQdgz2pF1bRUjkPml2rsleHQISztdSsiOGC2jozXNHwmf1b5z6KxymO8gvlImvLZ4tgseYpnAP8p_QZzMjIU7Y7Z2NQMDASr9hvv3tVjVCphqz1RH-h4gifjZJexwK9BR9O98u63X03f01NqgimS_dZHZUeC9voUb7_khNizA9-dS-fpYUduqvxZt-KZ7Q9gx7kFIH3wJvF-Gef55lwy4JNb8svu1wSna3EaQWTBeZOPHD3qbMXWVT5Yf5yrz7KvSemiWKqofYIInNaRLTtXLAOqq4VXP3dmgyEmAZIUfbh3IZtQ1uYwaV2hQoF-0YgM7JLPNDBwX8cRZtlyzFstnDsL_QLArf0bA8FMFNPuqPfyKFvXcGTgzquaUzngzNaoGo7k6kPHWLoSsWbvY3WvzYg4CO04sphuuSHh9TZRBy6LXCdxaMHIZDY_qVB1Cf-_dmDW6Eqr9_xodcTMBqs6RHlttLwFMMiul4aE_hUgNFlzOX7oVbisIS2Sm36GTuKE4zrbkvsA==',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'container_id': 'cntr_68cdba56addc81918f656db25fd0a6800d6da575ea4fee9b',
                            'code': """\
# compute the value
65465 - 6544 * 65464 - 6 + 1.02255
""",
                        },
                        tool_call_id='ci_68cdba5af39881a393a01eebb253854e09b7445677780c8f',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'status': 'completed', 'logs': ['-428330955.97745']},
                        tool_call_id='ci_68cdba5af39881a393a01eebb253854e09b7445677780c8f',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68cdba63843881a3a9c585d83e4df9f309b7445677780c8f',
                        signature='gAAAAABozbpoJefk0Fp1xqQzY6ego00t7KnH2ohbIw-rR9ZgaEAQs3n0Fubka6xbgRxzb1og6Xup1BuT8hQKMS-NHFxYsYXw4b6KeSbCd5oySVO53bsITEVk0A6tgjGssDJc1xSct1ORo-nCNV24MCNZvL9MKFeGQHP-jRypOZ9Vhepje87kFWTpw9lP9j54fZJdRIBGA9G_goI9m1cPztFUufcUxtLsgorsM053oxh8yWiEccAbvBaGXRlPWSoZYktbKrWeBVwiRt2ul-jRV43Z3chB32bEM1l9sIWG1xnvLE3OY6HuAy5s3bB-bnk78dibx5yx_iA36zGOvRkfiF0okXZoYiMNzJz3U7rTSsKlYoMtCKgnYGFdrh0D8RPj4VtxnRr-zAMJSSZQCm7ZipNSMS0PpN1wri14KktSkIGZGLhPBJpzPf9AjzaBBi2ZcUM347BtOfEohPdLBn8R6Cz-WxmoA-jH9qsyO-bPzwtRkv28H5G6836IxU2a402Hl0ZQ0Q-kPb5iqhvNmyvEQr6sEY_FN6ogkxwS-UEdDs0QlvJmgGfOfhMpdxfi5hr-PtElPg7j5_OwA7pXtuEI8mADy2VEqicuZzIpo6d-P72-Wd8sapjo-bC3DLcJVudFF09bJA0UirrxwC-zJZlmOLZKG8OqXKBE4GLfiLn48bYa5FC8a_QznrX8iAV6qPoqyqXANXuBtBClmzTHQU5A3lUgwSgtJo6X_0wZqw0O4lQ1iQQrkt7ZLeT7Ef6QVLyh9ZVaMZqVGrmHbphZK5N1u8b4woZYJKe0J57SrNihO8Slu8jZ71dmXjB4NAPjm0ZN6pVaZNLUajSxolJfmkBuF1BCcMYMVJyvV7Kk9guTCtntLZjN4XVOJWRU8Db5BjL17ciWWHGPlQBMxMdYFZOinwCHLIRrtdVxz4Na2BODjl0-taYJHbKd-_5up5nysUPc4imgNawbN2mNwjhdc1Qv919Q9Cz-he9i3j6lKYnEkgJvKF2RDY6-XAI=',
                        provider_name='openai',
                    ),
                    TextPart(
                        content="""\
Using standard order of operations (multiplication before addition/subtraction):

65465 - 6544 * 65464 - 6 + 1.02255 = -428,330,955.97745

If you intended different grouping with parentheses, let me know.\
""",
                        id='msg_68cdba6652ac81a3a58625883261465809b7445677780c8f',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=1493, cache_read_tokens=1280, output_tokens=125, details={'reasoning_tokens': 64}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 19, 20, 17, 21, tzinfo=timezone.utc),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    messages = result.all_messages()
    result = await agent.run(user_prompt='how about 2 to the power of 8?', message_history=messages)
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='how about 2 to the power of 8?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68cdba6c100481a394047de63f3e175009b7445677780c8f',
                        signature='gAAAAABozbpuOXVfjIYw7Gw6uSeadpkyaqMU1Frav7mTaf9LP8p8YuC8CWR9fYa02yZ5oYr1mqmYraD8ViOE33zqO2HBCdiWpOkVdNX-s4SGuPPB7ewyM7bDD4XbaSzo-Q5I6MgZmvVGWDGodqa3MfSKKNcGyD4aEfryQRLi4ObvHE5yuOqRo8FzGXMqe_pFdnvJXXD7njyfUofhWNvQPsLVLQFA_g_e7WKXtJJf_2JY183oi7-jNQ6rD9wGhM81HWSv0sTSBIHMpcE44rvlVQMFuh_rOPVUHUhT7vED7fYtrMoaPl46yDBc148T3MfXTnS-zm163zBOa34Yy_VXjyXw04a8Ig32y72bJY7-PRpZdBaeqD3BLvXfMuY4C911Z7FSxVze36mUxVO62g0uqV4PRw9qFA9mG37KF2j0ZsRzfyAClK1tu5omrYpenVKuRlrOO6JFtgyyE9OtLJxqvRNRKgULe2-cOQlo5S74t9lSMgcSGQFqF4JKG0A4XbzlliIcvC3puEzObHz-jArn_2BVUL_OPqx9ohJ9ZxAkXYgf0IRNYiKF4fOwKufYa5scL1kx2VAmsmEv5Yp5YcWlriB9L9Mpg3IguNBmq9DeJPiEQBtlnuOpSNEaNMTZQl4jTHVLgA5eRoCSbDdqGtQWgQB5wa7eH085HktejdxFeG7g-Fc1neHocRoGARxwhwcTT0U-re2ooJp99c0ujZtym-LiflSQUICi59VMAO8dNBE3CqXhG6S_ZicUmAvguo1iGKaKElMBv1Tv5qWcs41eAQkhRPBXQXoBD6MtBLBK1M-7jhidVrco0uTFhHBUTqx3jTGzE15YUJAwR69WvIOuZOvJdcBNObYWF9k84j0bZjJfRRbJG0C7XbU=',
                        provider_name='openai',
                    ),
                    TextPart(content='256', id='msg_68cdba6e02c881a3802ed88715e0be4709b7445677780c8f'),
                ],
                usage=RequestUsage(input_tokens=793, output_tokens=7, details={'reasoning_tokens': 0}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 19, 20, 17, 46, tzinfo=timezone.utc),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_thinking_with_code_execution_tool_stream(
    allow_model_requests: None, openai_api_key: str
):
    provider = OpenAIProvider(api_key=openai_api_key)
    m = OpenAIResponsesModel(
        model_name='gpt-5',
        provider=provider,
        settings=OpenAIResponsesModelSettings(openai_reasoning_summary='detailed', openai_reasoning_effort='low'),
    )
    agent = Agent(model=m, builtin_tools=[CodeExecutionTool()])

    event_parts: list[Any] = []
    async with agent.iter(user_prompt="what's 123456 to the power of 123?") as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content="what's 123456 to the power of 123?",
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c3509b2ee0819eba32735182d275ad0f2d670b80edc507',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"n = pow(123456, 123)\\nlen(str(n))"}',
                        tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'status': 'completed'},
                        tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"str(n)[:100], str(n)[-100:]"}',
                        tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'status': 'completed'},
                        tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"n"}',
                        tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'status': 'completed'},
                        tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_68c350a75ddc819ea5406470460be7850f2d670b80edc507',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=3727, cache_read_tokens=3200, output_tokens=347, details={'reasoning_tokens': 128}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 11, 22, 43, 36, tzinfo=timezone.utc),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0, part=ThinkingPart(content='', id='rs_68c3509b2ee0819eba32735182d275ad0f2d670b80edc507')
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='**Calcul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' large')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' integer')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
**

I\
"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' need')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' compute')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' 123')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='456')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' raised')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' power')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' 123')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' That')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' an')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' enormous')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' integer')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' user')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' probably')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' wants')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' exact')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' value')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Python')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ability')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' handle')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' big')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' integers')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' output')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' will')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' likely')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' extremely')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' long')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' —')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' potentially')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' hundreds')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' digits')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' should')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' consider')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' prepare')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' return')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' result')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' as')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' plain')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' text')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' even')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ends')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' up')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' being')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' around')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' 627')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' digits')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' So')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' go')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ahead')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' compute')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='!')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    signature_delta='gAAAAABow1CfwMTF6GjgPzWVr8oKbF3qM2qnldMGM_sXMoJ2SSXHrcL4lsIK69rnKn43STNM_YZ3f5AcwxF4oThzCOPl1g9-u4GGFd5sISVWJYruCukTVDPaEEzdmJqCU1JMSIZvlvqo7b5PsUGyQU5ldX4KXDq8zs4NmRyLIJe-34SCmDG3BYVWR_O-CtcjH0tF9e3XnJ5T9TvxioDEGbASqXMKx5XB9P_b1ser8P9WIQk6hxZ8YX-FAmWSt-sad-zScdeTmyPcakDb7Z4NVcXmL_I-hoQYH_lu-HPFVwcXU8R7yeXU-7YF3vZBE84cmFuv25lftyojbdGq2A7uxGJZBPMCoUBDGBNG2_7mVvKyGz_ZZ6vXIO0GVDhHdW4Y012pkoDfLp6B-B9CGvANOH3ORlcbhB8aT9qN5bY773wW44JIxRU3umkmNzwF7lkbmuMCbGybHYSzqtkOrMIRgqxaXOx3bGbsreM4kGwgD3EXWqQ1PVye_K7gRkToVQpfpID5iuH4jJZDkvNjjJI09JR2yqlR6QkQayVg2x1y8VHXoMYjNdQdZeP62AguqYbgrlBRcjaUnw78KcWscQHaNsg0MfxL_5Q-pZR1OPVsFppHRTzrVK8458d05yEhDmun345oI9ScBrtXFRdHXPy0dQaayfjxM9H0grPrIogMw_zz4jAcFqWxE_C7GPMnNIJ_uEAhkPOetpNb-izd-iY4pGYKs8pmCB5czrAlKC1MXTnowrlWcwf5_kuD5SzWlzlWOoKWCeBDOZuKTDVJKXh_QCtQfftomQazDFCiCSgaQMuP7GaPcDuS1jdQoMQBcFfKuWoq-3eQBOCiEOAERH81zR4hz1x02T_910jGreSpfgxSqt4Td0pDDSmlEV6CwaUDQvrPc67d8_Wtx8YKv4eBH544_p1k9T8tHo3Q7xvgE37ZCdd_AVhC2ed1b5oUI95tM570HAVugFilcHJICa1RbFzIlRkNgI4k2JvsVWtD5_h3x6ZaEFTomwIXlochYgsegh8RJIRRCNKO9ebsvTrkdl8n1mb3hLrz7puwCkRFyUkxYBGT9zUjuKrjp_IjTvvov29v6pwYHg2Xd0nAfLP4WWWPBLNx3oV1-yOfXStRGHMZTB6iN9d0Bxi2QS7dk-rPPXml5HxrSo1TG06EdBXQ1VgrkWIxG1TF97-gK9oWWT9S5aaYKZAOdaqDvi7qO8I-4VwExtIq4Do3BHnWrgKNHfyuAobQK4H_CFMElYibJHwA9t-UGujMic07AxS-2XjXaCtjf7LnW_aXE2rQDqzHiTiLmTqT6jYHP0WHGSqFTOFkNmzqy6uVfU-TbdT91zDBeesc8XpzCXWBVKqxEzuQGdJrYk6ieZaxL76Kjs4jyo838LMJCXzhcF8enukz_llnoxAV59hTDAn0MUQvstGlDX0ToI7C8Oc0NZfZU5Pi4gs8u0He_Nw5UsoV7sA-jk4M45sFt6g3u00kJFP3gIcdvOzHcRK5z3Sfb9JF0bnvIYSbUFUidEJxSOAcRlxofOJPnkPtWCYiiv3zSVxZXX77-wtc8yrOYFzH1k_8P6CDpcfzOW7Yl1Tajgcm20nygmPlFtXF3RNFPztW1V5GwQHc99FvT4ZAex3fQ_UBDKyXnyGoySgpZbHQIvhzUhDEGm77EiYw5FoF6JgnHGGUCbfXr2EudtpbGW8MRHop2ytonb8Hq7w10yQSginBbH_w3bwtd7cwgDKcp6wIPotjpEC-N1YDsRqhPuqxVA==',
                    provider_name='openai',
                ),
            ),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content=IsStr(),
                    id='rs_68c3509b2ee0819eba32735182d275ad0f2d670b80edc507',
                    signature='gAAAAABow1CfwMTF6GjgPzWVr8oKbF3qM2qnldMGM_sXMoJ2SSXHrcL4lsIK69rnKn43STNM_YZ3f5AcwxF4oThzCOPl1g9-u4GGFd5sISVWJYruCukTVDPaEEzdmJqCU1JMSIZvlvqo7b5PsUGyQU5ldX4KXDq8zs4NmRyLIJe-34SCmDG3BYVWR_O-CtcjH0tF9e3XnJ5T9TvxioDEGbASqXMKx5XB9P_b1ser8P9WIQk6hxZ8YX-FAmWSt-sad-zScdeTmyPcakDb7Z4NVcXmL_I-hoQYH_lu-HPFVwcXU8R7yeXU-7YF3vZBE84cmFuv25lftyojbdGq2A7uxGJZBPMCoUBDGBNG2_7mVvKyGz_ZZ6vXIO0GVDhHdW4Y012pkoDfLp6B-B9CGvANOH3ORlcbhB8aT9qN5bY773wW44JIxRU3umkmNzwF7lkbmuMCbGybHYSzqtkOrMIRgqxaXOx3bGbsreM4kGwgD3EXWqQ1PVye_K7gRkToVQpfpID5iuH4jJZDkvNjjJI09JR2yqlR6QkQayVg2x1y8VHXoMYjNdQdZeP62AguqYbgrlBRcjaUnw78KcWscQHaNsg0MfxL_5Q-pZR1OPVsFppHRTzrVK8458d05yEhDmun345oI9ScBrtXFRdHXPy0dQaayfjxM9H0grPrIogMw_zz4jAcFqWxE_C7GPMnNIJ_uEAhkPOetpNb-izd-iY4pGYKs8pmCB5czrAlKC1MXTnowrlWcwf5_kuD5SzWlzlWOoKWCeBDOZuKTDVJKXh_QCtQfftomQazDFCiCSgaQMuP7GaPcDuS1jdQoMQBcFfKuWoq-3eQBOCiEOAERH81zR4hz1x02T_910jGreSpfgxSqt4Td0pDDSmlEV6CwaUDQvrPc67d8_Wtx8YKv4eBH544_p1k9T8tHo3Q7xvgE37ZCdd_AVhC2ed1b5oUI95tM570HAVugFilcHJICa1RbFzIlRkNgI4k2JvsVWtD5_h3x6ZaEFTomwIXlochYgsegh8RJIRRCNKO9ebsvTrkdl8n1mb3hLrz7puwCkRFyUkxYBGT9zUjuKrjp_IjTvvov29v6pwYHg2Xd0nAfLP4WWWPBLNx3oV1-yOfXStRGHMZTB6iN9d0Bxi2QS7dk-rPPXml5HxrSo1TG06EdBXQ1VgrkWIxG1TF97-gK9oWWT9S5aaYKZAOdaqDvi7qO8I-4VwExtIq4Do3BHnWrgKNHfyuAobQK4H_CFMElYibJHwA9t-UGujMic07AxS-2XjXaCtjf7LnW_aXE2rQDqzHiTiLmTqT6jYHP0WHGSqFTOFkNmzqy6uVfU-TbdT91zDBeesc8XpzCXWBVKqxEzuQGdJrYk6ieZaxL76Kjs4jyo838LMJCXzhcF8enukz_llnoxAV59hTDAn0MUQvstGlDX0ToI7C8Oc0NZfZU5Pi4gs8u0He_Nw5UsoV7sA-jk4M45sFt6g3u00kJFP3gIcdvOzHcRK5z3Sfb9JF0bnvIYSbUFUidEJxSOAcRlxofOJPnkPtWCYiiv3zSVxZXX77-wtc8yrOYFzH1k_8P6CDpcfzOW7Yl1Tajgcm20nygmPlFtXF3RNFPztW1V5GwQHc99FvT4ZAex3fQ_UBDKyXnyGoySgpZbHQIvhzUhDEGm77EiYw5FoF6JgnHGGUCbfXr2EudtpbGW8MRHop2ytonb8Hq7w10yQSginBbH_w3bwtd7cwgDKcp6wIPotjpEC-N1YDsRqhPuqxVA==',
                    provider_name='openai',
                ),
                next_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                    provider_name='openai',
                ),
                previous_part_kind='thinking',
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"',
                    tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='n', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' =', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' pow', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='123', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='456', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' ', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='123', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=')\\n', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='len', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(str', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(n', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='))', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='"}', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartEndEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"n = pow(123456, 123)\\nlen(str(n))"}',
                    tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                    provider_name='openai',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=2,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'status': 'completed'},
                    tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=3,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                    provider_name='openai',
                ),
                previous_part_kind='builtin-tool-return',
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"',
                    tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='str', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='(n', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta=')', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='[:', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='100', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='],', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta=' str', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='(n', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta=')[', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='-', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='100', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta=':]', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='"}', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartEndEvent(
                index=3,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"str(n)[:100], str(n)[-100:]"}',
                    tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                    provider_name='openai',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=4,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'status': 'completed'},
                    tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=5,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                    provider_name='openai',
                ),
                previous_part_kind='builtin-tool-return',
            ),
            PartDeltaEvent(
                index=5,
                delta=ToolCallPartDelta(
                    args_delta='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"',
                    tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                ),
            ),
            PartDeltaEvent(
                index=5,
                delta=ToolCallPartDelta(
                    args_delta='n', tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=5,
                delta=ToolCallPartDelta(
                    args_delta='"}', tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507'
                ),
            ),
            PartEndEvent(
                index=5,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"n"}',
                    tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                    provider_name='openai',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=6,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'status': 'completed'},
                    tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=7,
                part=TextPart(content='123', id='msg_68c350a75ddc819ea5406470460be7850f2d670b80edc507'),
                previous_part_kind='builtin-tool-return',
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='456')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='^')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='123')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta=' equals')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta=':\n')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='180')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='302')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='106')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='304')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='044')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='807')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='508')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='140')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='927')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='865')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='938')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='572')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='807')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='342')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='688')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='638')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='559')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='680')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='488')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='440')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='159')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='857')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='958')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='502')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='360')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='813')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='732')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='502')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='197')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='826')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='969')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='863')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='225')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='730')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='871')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='630')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='436')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='419')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='794')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='758')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='932')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='074')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='350')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='380')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='367')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='697')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='649')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='814')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='626')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='542')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='926')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='602')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='664')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='707')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='275')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='874')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='269')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='201')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='777')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='743')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='912')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='313')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='197')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='516')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='323')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='690')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='221')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='274')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='713')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='845')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='895')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='457')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='748')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='735')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='309')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='484')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='337')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='191')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='373')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='255')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='527')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='928')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='271')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='785')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='206')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='382')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='967')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='998')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='984')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='330')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='482')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='105')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='350')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='942')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='229')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='970')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='677')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='054')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='940')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='838')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='210')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='936')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='952')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='303')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='939')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='401')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='656')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='756')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='127')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='607')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='778')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='599')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='667')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='243')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='702')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='814')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='072')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='746')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='219')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='431')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='942')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='293')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='005')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='416')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='411')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='635')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='076')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='021')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='296')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='045')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='493')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='305')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='133')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='645')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='615')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='566')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='590')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='735')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='965')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='652')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='587')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='934')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='290')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='425')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='473')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='827')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='719')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='935')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='012')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='870')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='093')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='575')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='987')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='789')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='431')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='818')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='047')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='013')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='404')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='691')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='795')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='773')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='170')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='405')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='764')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='614')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='646')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='054')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='949')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='298')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='846')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='184')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='678')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='296')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='813')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='625')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='595')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='333')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='311')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='611')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='385')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='251')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='735')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='244')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='505')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='448')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='443')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='050')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='050')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='547')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='161')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='779')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='229')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='749')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='134')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='489')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='643')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='622')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='579')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='100')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='908')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='331')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='839')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='817')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='426')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='366')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='854')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='332')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='416')),
            PartEndEvent(
                index=7,
                part=TextPart(content=IsStr(), id='msg_68c350a75ddc819ea5406470460be7850f2d670b80edc507'),
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"n = pow(123456, 123)\\nlen(str(n))"}',
                    tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'status': 'completed'},
                    tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"str(n)[:100], str(n)[-100:]"}',
                    tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'status': 'completed'},
                    tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"n"}',
                    tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'status': 'completed'},
                    tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
        ]
    )


async def test_reasoning_summary_auto(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-5.2', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model, instructions='You are a helpful coding assistant.')
    settings = OpenAIResponsesModelSettings(openai_reasoning_effort='high', openai_reasoning_summary='auto')

    result = await agent.run(
        'Write a Python function that calculates the factorial of a number. Think step by step.',
        model_settings=settings,
    )
    assert result.response.thinking == snapshot("""\
**Generating factorial function**

I need to respond with a Python function for calculating the factorial. The user wants me to think step-by-step, but I need to keep my reasoning brief. I'll provide a brief explanation of how the function works and include some input validation. I could choose either an iterative or recursive approach. I'll keep the details high-level, showing only the essential steps before presenting the final code to the user.\
""")
