from __future__ import annotations as _annotations

import datetime
from typing import Any

import pytest
from inline_snapshot import snapshot
from typing_extensions import TypedDict

from pydantic_ai import (
    BinaryContent,
    CachePoint,
    DocumentUrl,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import ModelHTTPError, ModelRetry, UsageLimitExceeded
from pydantic_ai.messages import AgentStreamEvent
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.bedrock import BedrockProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

from ..conftest import IsDatetime, IsInstance, IsStr, try_import

with try_import() as imports_successful:
    pass

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='bedrock not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


async def test_bedrock_model(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    assert model.base_url == 'https://bedrock-runtime.us-east-1.amazonaws.com'
    agent = Agent(model=model, system_prompt='You are a chatbot.')

    result = await agent.run('Hello!')
    assert result.output == snapshot(
        "Hello! How can I assist you today? Whether you have questions, need information, or just want to chat, I'm here to help."
    )
    assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=7, output_tokens=30))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content='You are a chatbot.',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='Hello!',
                        timestamp=IsDatetime(),
                    ),
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="Hello! How can I assist you today? Whether you have questions, need information, or just want to chat, I'm here to help."
                    )
                ],
                usage=RequestUsage(input_tokens=7, output_tokens=30),
                model_name='us.amazon.nova-micro-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_details={'finish_reason': 'end_turn'},
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_bedrock_model_usage_limit_exceeded(
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-20250514-v1:0', provider=bedrock_provider)
    agent = Agent(model=model)

    with pytest.raises(
        UsageLimitExceeded,
        match='The next request would exceed the input_tokens_limit of 18 \\(input_tokens=19\\)',
    ):
        await agent.run(
            'The quick brown fox jumps over the lazydog.',
            usage_limits=UsageLimits(input_tokens_limit=18, count_tokens_before_request=True),
        )


async def test_bedrock_model_usage_limit_not_exceeded(
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-20250514-v1:0', provider=bedrock_provider)
    agent = Agent(model=model)

    result = await agent.run(
        'The quick brown fox jumps over the lazydog.',
        usage_limits=UsageLimits(input_tokens_limit=25, count_tokens_before_request=True),
    )

    assert result.output == snapshot(
        'I notice there\'s a small typo in your message - it should be "lazy dog" (two words) rather than '
        '"lazydog."\n\nThe corrected version is: "The quick brown fox jumps over the lazy dog."\n\n'
        'This is a famous pangram - a sentence that contains every letter of the English alphabet at least once. '
        "It's commonly used for testing typewriters, keyboards, fonts, and other applications where you want to "
        "display all the letters.\n\nIs there something specific you'd like to know about this phrase, or were you "
        'perhaps testing something?'
    )


@pytest.mark.vcr()
async def test_bedrock_count_tokens_error(allow_model_requests: None, bedrock_provider: BedrockProvider):
    """Test that errors convert to ModelHTTPError."""
    model_id = 'us.does-not-exist-model-v1:0'
    model = BedrockConverseModel(model_id, provider=bedrock_provider)
    agent = Agent(model)

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('hello', usage_limits=UsageLimits(input_tokens_limit=20, count_tokens_before_request=True))

    assert exc_info.value.status_code == 400
    assert exc_info.value.model_name == model_id
    assert exc_info.value.body.get('Error', {}).get('Message') == 'The provided model identifier is invalid.'  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ('model_name', 'expected'),
    [
        ('us.anthropic.claude-sonnet-4-20250514-v1:0', 'anthropic.claude-sonnet-4-20250514-v1:0'),
        ('eu.amazon.nova-micro-v1:0', 'amazon.nova-micro-v1:0'),
        ('apac.meta.llama3-8b-instruct-v1:0', 'meta.llama3-8b-instruct-v1:0'),
        ('anthropic.claude-3-7-sonnet-20250219-v1:0', 'anthropic.claude-3-7-sonnet-20250219-v1:0'),
    ],
)
def test_remove_inference_geo_prefix(model_name: str, expected: str):
    assert BedrockConverseModel._remove_inference_geo_prefix(model_name) == expected  # pyright: ignore[reportPrivateUsage]


async def test_bedrock_model_structured_output(allow_model_requests: None, bedrock_provider: BedrockProvider):
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

    result = await agent.run('What was the temperature in London 1st January 2022?', output_type=Response)
    assert result.output == snapshot({'temperature': '30°C', 'date': datetime.date(2022, 1, 1), 'city': 'London'})
    assert result.usage() == snapshot(RunUsage(requests=2, input_tokens=1236, output_tokens=298, tool_calls=1))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content='You are a helpful chatbot.',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='What was the temperature in London 1st January 2022?',
                        timestamp=IsDatetime(),
                    ),
                ],
                run_id=IsStr(),
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
                usage=RequestUsage(input_tokens=551, output_tokens=132),
                model_name='us.amazon.nova-micro-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_details={'finish_reason': 'tool_use'},
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='temperature',
                        content='30°C',
                        tool_call_id='tooluse_5WEci1UmQ8ifMFkUcy2gHQ',
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
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
                usage=RequestUsage(input_tokens=685, output_tokens=166),
                model_name='us.amazon.nova-micro-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_details={'finish_reason': 'tool_use'},
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='tooluse_9AjloJSaQDKmpPFff-2Clg',
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
        ]
    )


async def test_bedrock_model_stream(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings={'temperature': 0.0})
    async with agent.run_stream('What is the capital of France?') as result:
        data = await result.get_output()
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
        return '30°C'  # pragma: no cover

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
    assert result.output == snapshot('Paris is the capital of France.')


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
                    SystemPromptPart(
                        content='You are a helpful chatbot.',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='What is the capital of France?',
                        timestamp=IsDatetime(),
                    ),
                ],
                run_id=IsStr(),
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
                usage=RequestUsage(input_tokens=417, output_tokens=69),
                model_name='us.amazon.nova-micro-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_details={'finish_reason': 'tool_use'},
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='The country is not supported.',
                        tool_name='get_capital',
                        tool_call_id='tooluse_F8LnaCMtQ0-chKTnPhNH2g',
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
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
                usage=RequestUsage(input_tokens=509, output_tokens=108),
                model_name='us.amazon.nova-micro-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_details={'finish_reason': 'end_turn'},
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_bedrock_model_max_tokens(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings={'max_tokens': 5})
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is')


async def test_bedrock_model_top_p(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings={'top_p': 0.5})
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        'The capital of France is Paris. Paris is not only the capital city but also the most populous city in France, known for its significant cultural, political, and economic influence both within the country and globally. It is famous for landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, among many other historical and architectural treasures.'
    )


async def test_bedrock_model_performance_config(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    model_settings = BedrockModelSettings(bedrock_performance_configuration={'latency': 'optimized'})
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings=model_settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        'The capital of France is Paris. Paris is not only the capital city but also the most populous city in France, known for its significant cultural, political, and economic influence both within the country and globally. It is famous for landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, among many other historical and architectural treasures.'
    )


async def test_bedrock_model_guardrail_config(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    model_settings = BedrockModelSettings(
        bedrock_guardrail_config={'guardrailIdentifier': 'guardrailv1', 'guardrailVersion': 'v1', 'trace': 'enabled'}
    )
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings=model_settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        'The capital of France is Paris. Paris is not only the capital city but also the most populous city in France, known for its significant cultural, political, and economic influence both within the country and globally. It is famous for landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, among many other historical and architectural treasures.'
    )


async def test_bedrock_model_other_parameters(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    model_settings = BedrockModelSettings(
        bedrock_prompt_variables={'leo': {'text': 'aaaa'}},
        bedrock_additional_model_requests_fields={'test': 'test'},
        bedrock_request_metadata={'test': 'test'},
        bedrock_additional_model_response_fields_paths=['test'],
    )
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings=model_settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        'The capital of France is Paris. Paris is not only the capital city but also the most populous city in France, known for its significant cultural, political, and economic influence both within the country and globally. It is famous for landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, among many other historical and architectural treasures.'
    )


async def test_bedrock_model_iter_stream(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings={'top_p': 0.5})

    @agent.tool_plain
    async def get_capital(country: str) -> str:
        """Get the capital of a country.

        Args:
            country: The country name.
        """
        return 'Paris'  # pragma: no cover

    @agent.tool_plain
    async def get_temperature(city: str) -> str:
        """Get the temperature in a city.

        Args:
            city: The city name.
        """
        return '30°C'

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='What is the temperature of the capital of France?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert event_parts == snapshot(
        [
            PartStartEvent(index=0, part=TextPart(content='<thinking')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='> To find')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' the temperature')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' of the capital of France,')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' I need to first')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' determine the capital')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' of France and')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' then get')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' the current')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' temperature in')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' that city. The')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' capital of France is Paris')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='. I')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' will use')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' the "get_temperature"')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' tool to find the current temperature')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' in Paris.</')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='thinking')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='>\n')),
            PartEndEvent(
                index=0,
                part=TextPart(
                    content='<thinking> To find the temperature of the capital of France, I need to first determine the capital of France and then get the current temperature in that city. The capital of France is Paris. I will use the "get_temperature" tool to find the current temperature in Paris.</thinking>\n'
                ),
                next_part_kind='tool-call',
            ),
            PartStartEvent(
                index=1,
                part=ToolCallPart(tool_name='get_temperature', tool_call_id='tooluse_lAG_zP8QRHmSYOwZzzaCqA'),
                previous_part_kind='text',
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='{"city":"Paris"}', tool_call_id='tooluse_lAG_zP8QRHmSYOwZzzaCqA'),
            ),
            PartEndEvent(
                index=1,
                part=ToolCallPart(
                    tool_name='get_temperature', args='{"city":"Paris"}', tool_call_id='tooluse_lAG_zP8QRHmSYOwZzzaCqA'
                ),
            ),
            IsInstance(FunctionToolCallEvent),
            FunctionToolResultEvent(
                result=ToolReturnPart(
                    tool_name='get_temperature',
                    content='30°C',
                    tool_call_id='tooluse_lAG_zP8QRHmSYOwZzzaCqA',
                    timestamp=IsDatetime(),
                )
            ),
            PartStartEvent(index=0, part=TextPart(content='The')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' current temperature in Paris, the')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' capital of France,')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' is 30°C')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='.')),
            PartEndEvent(
                index=0, part=TextPart(content='The current temperature in Paris, the capital of France, is 30°C.')
            ),
        ]
    )


@pytest.mark.vcr()
async def test_image_as_binary_content_input(
    allow_model_requests: None, image_content: BinaryContent, bedrock_provider: BedrockProvider
):
    m = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    result = await agent.run(['What fruit is in the image?', image_content])
    assert result.output == snapshot(
        'The image features a fruit that is round and has a green skin with brown dots. The fruit is cut in half, revealing its interior, which is also green. Based on the appearance and characteristics, the fruit in the image is a kiwi.'
    )


@pytest.mark.vcr()
async def test_video_as_binary_content_input(
    allow_model_requests: None, video_content: BinaryContent, bedrock_provider: BedrockProvider
):
    m = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    result = await agent.run(['Explain me this video', video_content])
    assert result.output == snapshot(
        'The video shows a camera set up on a tripod, pointed at a scenic view of a rocky landscape under a clear sky. The camera remains stationary throughout the video, capturing the same view without any changes.'
    )


@pytest.mark.vcr()
async def test_image_url_input(allow_model_requests: None, bedrock_provider: BedrockProvider):
    m = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    result = await agent.run(
        [
            'What is this vegetable?',
            ImageUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'),
        ]
    )
    assert result.output == snapshot(
        'The image shows a potato. It is oval in shape and has a yellow skin with numerous dark brown patches. These patches are known as lenticels, which are pores that allow the potato to breathe. The potato is a root vegetable that is widely cultivated and consumed around the world. It is a versatile ingredient that can be used in a variety of dishes, including mashed potatoes, fries, and potato salad.'
    )


@pytest.mark.vcr()
async def test_video_url_input(allow_model_requests: None, bedrock_provider: BedrockProvider):
    m = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    result = await agent.run(
        [
            'Explain me this video',
            VideoUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/small_video.mp4'),
        ]
    )
    assert result.output == snapshot(
        'The video shows a camera set up on a tripod, pointed at a scenic view of a rocky landscape under a clear sky. The camera remains stationary throughout the video, capturing the same view without any changes.'
    )


@pytest.mark.vcr()
async def test_document_url_input(allow_model_requests: None, bedrock_provider: BedrockProvider):
    m = BedrockConverseModel('anthropic.claude-v2', provider=bedrock_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    document_url = DocumentUrl(url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf')

    result = await agent.run(['What is the main content on this document?', document_url])
    assert result.output == snapshot(
        'Based on the provided XML data, the main content of the document is "Dummy PDF file". This is contained in the <document_content> tag for the document with index="1".'
    )


@pytest.mark.vcr()
async def test_text_document_url_input(allow_model_requests: None, bedrock_provider: BedrockProvider):
    m = BedrockConverseModel('anthropic.claude-v2', provider=bedrock_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    text_document_url = DocumentUrl(url='https://example-files.online-convert.com/document/txt/example.txt')

    result = await agent.run(['What is the main content on this document?', text_document_url])
    assert result.output == snapshot("""\
Based on the text in the <document_content> tag, the main content of this document appears to be:

An example text describing the use of "John Doe" as a placeholder name in legal cases, hospitals, and other contexts where a party's real identity is unknown or needs to be withheld. It provides background on how "John Doe" and "Jane Doe" are commonly used in the United States and Canada for this purpose, in contrast to other English speaking countries that use names like "Joe Bloggs". The text gives examples of using John/Jane Doe for legal cases, unidentified corpses, and as generic names on forms. It also mentions how "Baby Doe" and "Precious Doe" are used for unidentified children.\
""")


@pytest.mark.vcr()
async def test_text_as_binary_content_input(allow_model_requests: None, bedrock_provider: BedrockProvider):
    m = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    text_content = BinaryContent(data=b'This is a test document.', media_type='text/plain')

    result = await agent.run(['What is the main content on this document?', text_content])
    assert result.output == snapshot("""\
The document you're referring to appears to be a test document, which means its primary purpose is likely to serve as an example or a placeholder rather than containing substantive content. Test documents are commonly used for various purposes such as:

1. **Software Testing**: To verify that a system can correctly handle, display, or process documents.
2. **Design Mockups**: To illustrate how a document might look in a particular format or style.
3. **Training Materials**: To provide examples for instructional purposes.
4. **Placeholders**: To fill space in a system or application where real content will eventually be placed.

Since this is a test document, it probably doesn't contain any meaningful or specific information beyond what is necessary to serve its testing purpose. If you have specific questions about the format, structure, or any particular element within the document, feel free to ask!\
""")


@pytest.mark.vcr()
async def test_bedrock_model_instructions(allow_model_requests: None, bedrock_provider: BedrockProvider):
    m = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)

    def instructions() -> str:
        return 'You are a helpful assistant.'

    agent = Agent(m, instructions=instructions)

    result = await agent.run('What is the capital of France?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the capital of France?', timestamp=IsDatetime())],
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='The capital of France is Paris. Paris is not only the political and economic hub of the country but also a major center for culture, fashion, art, and tourism. It is renowned for its rich history, iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, as well as its influence on global culture and cuisine.'
                    )
                ],
                usage=RequestUsage(input_tokens=13, output_tokens=71),
                model_name='us.amazon.nova-pro-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_details={'finish_reason': 'end_turn'},
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.vcr()
async def test_bedrock_empty_system_prompt(allow_model_requests: None, bedrock_provider: BedrockProvider):
    m = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(m)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        'The capital of France is Paris. Paris, officially known as "Ville de Paris," is not only the capital city but also the most populous city in France. It is located in the northern central part of the country along the Seine River. Paris is a major global city, renowned for its cultural, political, economic, and social influence. It is famous for its landmarks such as the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and the Champs-Élysées, among many other historic and modern attractions. The city has played a significant role in the history of art, fashion, gastronomy, and science.'
    )


@pytest.mark.vcr()
async def test_bedrock_multiple_documents_in_history(
    allow_model_requests: None, bedrock_provider: BedrockProvider, document_content: BinaryContent
):
    m = BedrockConverseModel(model_name='us.anthropic.claude-3-7-sonnet-20250219-v1:0', provider=bedrock_provider)
    agent = Agent(model=m)

    result = await agent.run(
        'What is in the documents?',
        message_history=[
            ModelRequest(parts=[UserPromptPart(content=['Here is a PDF document: ', document_content])]),
            ModelResponse(parts=[TextPart(content='foo bar')]),
            ModelRequest(parts=[UserPromptPart(content=['Here is another PDF document: ', document_content])]),
            ModelResponse(parts=[TextPart(content='foo bar 2')]),
        ],
    )

    assert result.output == snapshot(
        'Based on the documents you\'ve shared, both Document 1.pdf and Document 2.pdf contain the text "Dummy PDF file". These appear to be placeholder or sample PDF documents rather than files with substantial content.'
    )


async def test_bedrock_model_thinking_part_deepseek(allow_model_requests: None, bedrock_provider: BedrockProvider):
    m = BedrockConverseModel('us.deepseek.r1-v1:0', provider=bedrock_provider)
    agent = Agent(m)

    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='How do I cross the street?', timestamp=IsDatetime())],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content=IsStr()), ThinkingPart(content=IsStr())],
                usage=RequestUsage(input_tokens=12, output_tokens=693),
                model_name='us.deepseek.r1-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_details={'finish_reason': 'end_turn'},
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
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content=IsStr()), ThinkingPart(content=IsStr())],
                usage=RequestUsage(input_tokens=33, output_tokens=907),
                model_name='us.deepseek.r1-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_details={'finish_reason': 'end_turn'},
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_bedrock_model_thinking_part_anthropic(allow_model_requests: None, bedrock_provider: BedrockProvider):
    m = BedrockConverseModel(
        'us.anthropic.claude-sonnet-4-20250514-v1:0',
        provider=bedrock_provider,
        settings=BedrockModelSettings(
            bedrock_additional_model_requests_fields={
                'thinking': {'type': 'enabled', 'budget_tokens': 1024},
            }
        ),
    )
    agent = Agent(m)

    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='How do I cross the street?', timestamp=IsDatetime())],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        signature=IsStr(),
                        provider_name='bedrock',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(input_tokens=42, output_tokens=313),
                model_name='us.anthropic.claude-sonnet-4-20250514-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_details={'finish_reason': 'end_turn'},
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
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        signature=IsStr(),
                        provider_name='bedrock',
                    ),
                    IsInstance(TextPart),
                ],
                usage=RequestUsage(input_tokens=334, output_tokens=432),
                model_name='us.anthropic.claude-sonnet-4-20250514-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_details={'finish_reason': 'end_turn'},
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_bedrock_model_thinking_part_redacted(allow_model_requests: None, bedrock_provider: BedrockProvider):
    m = BedrockConverseModel(
        'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
        provider=bedrock_provider,
        settings=BedrockModelSettings(
            bedrock_additional_model_requests_fields={
                'thinking': {'type': 'enabled', 'budget_tokens': 1024},
            }
        ),
    )
    agent = Agent(m)

    result = await agent.run(
        'ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB'
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB',
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='redacted_content',
                        signature=IsStr(),
                        provider_name='bedrock',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(input_tokens=92, output_tokens=176),
                model_name='us.anthropic.claude-3-7-sonnet-20250219-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_details={'finish_reason': 'end_turn'},
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run(
        'What was that?',
        message_history=result.all_messages(),
    )
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What was that?',
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='redacted_content',
                        signature=IsStr(),
                        provider_name='bedrock',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(input_tokens=182, output_tokens=258),
                model_name='us.anthropic.claude-3-7-sonnet-20250219-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_details={'finish_reason': 'end_turn'},
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_bedrock_model_thinking_part_redacted_stream(
    allow_model_requests: None, bedrock_provider: BedrockProvider
):
    m = BedrockConverseModel(
        'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
        provider=bedrock_provider,
        settings=BedrockModelSettings(
            bedrock_additional_model_requests_fields={
                'thinking': {'type': 'enabled', 'budget_tokens': 1024},
            }
        ),
    )
    agent = Agent(m)

    event_parts: list[Any] = []
    async with agent.iter(
        user_prompt='ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB'
    ) as agent_run:
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
                        content='ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB',
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='redacted_content',
                        signature=IsStr(),
                        provider_name='bedrock',
                    ),
                    ThinkingPart(
                        content='',
                        id='redacted_content',
                        signature=IsStr(),
                        provider_name='bedrock',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(input_tokens=92, output_tokens=253),
                model_name='us.anthropic.claude-3-7-sonnet-20250219-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_details={'finish_reason': 'end_turn'},
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ThinkingPart(
                    content='',
                    id='redacted_content',
                    signature=IsStr(),
                    provider_name='bedrock',
                ),
            ),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content='',
                    id='redacted_content',
                    signature='EtkECkgIBxABGAIqQJTfqS/PYuAFZeOls6R8uGN014YNT7YDIFuhNyywoX1Cjf9oIYThX1ucUFJ1cfckdN55jozmXi1PEgMfufPmD44SDHBw8Yp6gJ8Ys/Gt3BoMYdLaNUOqr7k/MAeYIjBhPIc9z85HrJAbeS8Hz/69R+vKHpRanI0n/B69dnv2nebRe7LKZgHs2AlVPEtNyyoqvgP463qJ7/KvDrAPSnhHQqZ8TH8JBC4eYb4Qow5eX7dI3UXY/DrQ2IOWLADJqshcXBg7zbN78H4l6fTP97Ztzz0qw4fadTzTb36dRR7p8rs2zA/pHWhK+75xvUGh8IdLPvMikKccHssHKdceru4JLG1cMVtq1Ci7ZPAbHRU8/XsjFtLWPHeYLfKGJN33C1MpWX5nQU2BjYICs5Hn+8Z9Smxhp06rZXTjZARiExrd1dgLn5/5PbEzMLJv/Q3c6XJH7kx7iUO4NAonTT1Q3WY1cGa38UNGYuTUae3CNFEZWjS21tWRmjX4t5w8L0BtQ5DSaW/ZzGf0yzUKUaS/fkVjr2xztQBvysFFbb7UrX+/lNw26CHXKUIXFcZzV9l0HrA6z3oQrqSpnwem/pt/Cxdh5YQlXq6DSdzstqwJA53n9Hj3osjT/viH4Y6N5dWLLBTQBvhUEy24FhlytD3scYrvAqCdxW9aDSW+e+Foj5vsjVA9VFrXqZeNSO77Qp5dLw5XcA8CH6YFTE6EWeFTki5vfTfSIw+m4inZGVzIRi8Qk90IzW2EnrxGtx3wsEn5XImQr1vg1Npq2jN6uiOPOp8nsBgB',
                    provider_name='bedrock',
                ),
                next_part_kind='thinking',
            ),
            PartStartEvent(
                index=1,
                part=ThinkingPart(
                    content='',
                    id='redacted_content',
                    signature='EqADCkgIBxABGAIqQB3h5GyHJD4hocRchUq2I40ChRLdpxjVl0xZkyVZrrk6JIJWeInuRQfJG5nJymmQBjH9VDeV53H/D3W9xjIJvPUSDLv7jRCF9b6Tx1Z5EBoMSv3CBw4zUjjSDaqlIjDBpH7V3YQB5twUmulAycDyZRvP3loupy6o2eqrfKAZZjq3rwkApWD9qOqJD3OEfd4qhQJZfOcHs9bt5zCqzYjoaIkxE3raXnhUHOlwq1Jq60bTQt2SQiHqoZTEht/DeDEEgpFy9Z32Zz3/Az0ORgTi3QE56K15OXo6GWMPYq/CTJ/xzPXfH0/yoQ4EP103VfVqvymEpXUru6RQGkou41LKRI92fRsqCK+jPOpxeED4kz7CFhQYMHttk7cOAF85SE3nCcpliARrLDvsApjgMFAYnineZQMLwawmnIm6EB61C20dB1Ft7vLG1TS6fn27EB8JZjr/jeC8O4ZysKv5iUxpMlDZib8jFszfzxCXdFX7NVKO9+dH8cW3RsJ80kzBp6xyoQhXSFx72jFllwDy8e+QlI3OIhweJ8IYAQ==',
                    provider_name='bedrock',
                ),
                previous_part_kind='thinking',
            ),
            PartEndEvent(
                index=1,
                part=ThinkingPart(
                    content='',
                    id='redacted_content',
                    signature='EqADCkgIBxABGAIqQB3h5GyHJD4hocRchUq2I40ChRLdpxjVl0xZkyVZrrk6JIJWeInuRQfJG5nJymmQBjH9VDeV53H/D3W9xjIJvPUSDLv7jRCF9b6Tx1Z5EBoMSv3CBw4zUjjSDaqlIjDBpH7V3YQB5twUmulAycDyZRvP3loupy6o2eqrfKAZZjq3rwkApWD9qOqJD3OEfd4qhQJZfOcHs9bt5zCqzYjoaIkxE3raXnhUHOlwq1Jq60bTQt2SQiHqoZTEht/DeDEEgpFy9Z32Zz3/Az0ORgTi3QE56K15OXo6GWMPYq/CTJ/xzPXfH0/yoQ4EP103VfVqvymEpXUru6RQGkou41LKRI92fRsqCK+jPOpxeED4kz7CFhQYMHttk7cOAF85SE3nCcpliARrLDvsApjgMFAYnineZQMLwawmnIm6EB61C20dB1Ft7vLG1TS6fn27EB8JZjr/jeC8O4ZysKv5iUxpMlDZib8jFszfzxCXdFX7NVKO9+dH8cW3RsJ80kzBp6xyoQhXSFx72jFllwDy8e+QlI3OIhweJ8IYAQ==',
                    provider_name='bedrock',
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=2,
                part=TextPart(content="I notice you've sent what appears to be some"),
                previous_part_kind='thinking',
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' kind of command or trigger string, but I don')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta="'t respond to special codes or")),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=" triggers. That string doesn't have")),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' any special meaning to me.')),
            PartDeltaEvent(
                index=2,
                delta=TextPartDelta(
                    content_delta="""\


If you have a question you\
"""
                ),
            ),
            PartDeltaEvent(
                index=2, delta=TextPartDelta(content_delta="'d like to discuss or need assistance with something")
            ),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=", I'd be happy to help in")),
            PartDeltaEvent(
                index=2, delta=TextPartDelta(content_delta=' a straightforward conversation. What would you like to')
            ),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' talk about today?')),
            PartEndEvent(
                index=2,
                part=TextPart(
                    content="""\
I notice you've sent what appears to be some kind of command or trigger string, but I don't respond to special codes or triggers. That string doesn't have any special meaning to me.

If you have a question you'd like to discuss or need assistance with something, I'd be happy to help in a straightforward conversation. What would you like to talk about today?\
"""
                ),
            ),
        ]
    )


async def test_bedrock_model_thinking_part_from_other_model(
    allow_model_requests: None, bedrock_provider: BedrockProvider, openai_api_key: str
):
    provider = OpenAIProvider(api_key=openai_api_key)
    m = OpenAIResponsesModel('gpt-5', provider=provider)
    settings = OpenAIResponsesModelSettings(openai_reasoning_effort='high', openai_reasoning_summary='detailed')
    agent = Agent(m, system_prompt='You are a helpful assistant.', model_settings=settings)

    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content='You are a helpful assistant.',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='How do I cross the street?',
                        timestamp=IsDatetime(),
                    ),
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1ffe148588191812b659c6dc35ce60003919771fccd27',
                        signature='gAAAAABowgAKxFTo-oXVZ9WpxX1o2XmQkqXqGTeqSbHjr1hsNXhe0QDBXDnKBMrBVbYympkJVMbAIsYJuZ8P3-DmXZVwYJR_F1cfpCbt97TxVSbG7WIbUp-H1vYpN3oA2-hlP-G76YzOGJzHQy1bWWluUC4GsPP194NpVANRnTUBQakfwhOgk9WE2Op7SyzfdHxYV5vpRPcrXRMrLZYZFUXM6D6ROZljjaZKNj9KaluIOdiTZydQnKVyZs0ffjIpNe6Cn9jJNAUH-cxKfOJ3fmUVN213tTr-PveUkAdlYwCRdtq_IlrFrr1gp6hiMgtdQXxSdtjPuoMfQEZTsI-FiAGFipYDrN5Gu_YXlqX1Lmzbb2famCXTYp6bWljYT14pCSMA-OZrJWsgj4tSahyZIgNq_E_cvHnQ-iJo1ACH0Jt22soOFBhAhSG8rLOG8O5ZkmF7sGUr1MbP56LLkz29NPgh98Zsyxp4tM33QH5XPrMC7MOfTvzj8TyhRH31CWHScQl3AJq1o3z2K3qgl6spkmWIwWLjbo4DBzFz6-wRPBm5Fv60hct1oFuYjXL-ntOBASLOAES7U3Cvb56VPex7JdmTyzb-XP7jNhYzWK-69HgGZaMhOJJmLGZhu8Xp9P6GPnXiQpyL5LvcX_FEiR6CzpkhhS54IryQx2UW7VadUMnpvwEUwtT2c9xoh6WEwt2kTDj65DyzRwFdcms3WG_B1cSe5iwBN1JAQm3ay04dSG-a5JNVqFyaW7r1NcVts3HWC2c-S9Z_Xjse548XftM_aD97KTqoiR5GxU95geXvrWI8szDSYSueSGCTI8L7bCDO-iKE4RQEmyS8ZbqMSWyQgClVQOR5CF3jPKb6hP7ofoQlPRuMyMY8AqyWGeY9bbWb-LjrSDpRTAR6af8Ip5JYr4rlcG1YqEWYT-MqiCPw3ZJqBXUICSpz9ZHQNTrYIzkJZqPg-hCqvFkOCUtvOYSDtGkAe9x1ekPqlV0IuWLxAmjqbkGH0QCaYAF90wVQUgWPkVWfQ6ULRz2sveQDZf0P8rVZw6ATEvZVnkml6VDbaH69lMyvzls7suvEZJxS5osyjrGfkt6L4nsvhZS7Nuxj2TcRxSEXxo5kULEqAO85Ivsm4j7R1Cxb2h8I4ZZZ_-DnkbWsgd7DELMI-CYtpAWLFl4K4VaMBT6mNAuud545BemUlWnQgmrde4aS7Q_W5GP11iQea9_JcJr6DMf4Y40NDr_fPVU5p7q1bnc1xtwkIpyx0uEeXHEZDR8k-5apBXScJtmelzpiy-25oJdSU5xtgVPrb77kVyJofPtujplZoqMh6MOqTdIhIMm_Goy_Wne4W39hVI01b2vwduBaCCaX6M8uACX96s454WPitX4MYAVc65UHF0BTFskEcbY5bFZpzcWb39VTfra-Ru2URvdo_66qmUd-03XzLKiMsqJHGclhaU6XBqaIo9qD8FjLVT9DOx56eh3GFvYA1dxvgbp6gyOg7bOBL0KDarT9Vmo40vGvwyCT_a2S_6Oki6uBU_3bf-jGvtum4tkN--wZkBrhOj7L8onItPoAZQXjYXrcXfVC1KR_xA0IOxYZD59G1rBxTDlvatIFwhvoISinkU-zPkKMpralHlxDicmJrBsKsy-mZWCF5qHeWF36pjE35dE9GxR28xw1Ed0pA_kOIgMKSKCiRWUYY8D1jAHKzimnj_4VTKR05kTp30pasr0IUMl2celsQMDv1D2atEJ_65CeRio5cnNGUR_Z73LJ-fqLkSjSxlE2YvtcKX7bdF6bSq3EqDtOdLVUjYl_pxRaUNMRmahQUJXGsDx7X-W9xUgQmAq09qT3lh1fhVUgdtUuuaoNY_M1s5V0E5ePuu_C6Duuz8WCcecbhrcbI3FDQSJn_XHK6ImLMYBowGRYVkBE_Rf7q7Hj4zdF-3bVE_QDce3syZNshCYK5kO8mvADptgdNVG7lEiZ9TIQPBd-XWRUrZ3XvIfGVJFVMjh_Laq8RTDyvvId7iQDuvq89hQ86hlfWteEl8HzuwpakWnogg3CCStX5CMGpYUWWkOCUu2LCH2H4EBaeCcAPLCmEoxcpKS182kYLm8-4ShRz-YOMIEmE9TL2za15I6BCBi9OhQGcLSl4BquhfBVHyxmkEN7_g102yI1Ocucux8q_HLMo5UZz0KALRQy4qmNpnLg9f4Yetj6msezjuU17Ji1ofIcadglOYy2J3Aswf58M9fCwCfB6hAHRYM2XkYzJ3nc0VosWA0er90zqKOeM1-erWC-skbupO-8nw9DA5OtnJTZOLnhGRjzXqna0E5R69wOHi3yvb3zzv2K9fLMKi11bCM_cnel9ItcFM-AYQ0AhBTZ3sTn-tpIf3IVNCvnCxMWvbO-MBmoexQnPorA0SL6n_nL49Y9Zb7UgwCyNGmhsFjIlSXu-YG-yCV1lVXBYoEPDwa2eCaMwph0QneXPHHMUs_i9PuFVI-nwfEiwU0b4tk8x3tWdkltvtzhjB8fxQxJNrk-ykNhuEYfQMQ0_MCqIRD097_gjO8q-eFUjnuiVqqoQ9_rH9QCxABdA8afoNt0hFxBwR6d57P81_XKOnApyrPx0DjsuKVTBFoCWccKX4DZuQT_PhmsFtPquNp6OPWQM5a8HzKntjz_HgFYnyS5p6n0hBGZVC_GDtFEm8JELcwuVoSLSXhI_XKnck2FIhHA5YQ4vLGOhCEEZoINkDdq3oNgm-NiP-DpG2LYetLl4ljlUpRBUizmWn4Fr3jhIt8rmQwqmFj6aMDSEM0Sgen9DsUH7H3uGK2NipvFv2Uxic5aXAKQ37EFjxPFqvKXlDl-hLnUXtkXLXBbmgCJJw6nBvm-SeIxU_eKnWHkhtdnkNZrmNFaq0OYZKk-moYSxEgzxasQNYGtkN89LqAhRTS6dIbb4nXa8ArvuHTJ_qpLFjGF3SSX98Y53cgtSdGTTmHQ6_v0BmeKCWhRd83vPrmFosif57AXyBVk0HJ5YdeueitsBCyXcJmeCntrT4zDlujwuMWK7wDO4vGMj3nIIyuJMJjtpD_auuDLmpYHqmKTHm8Ob8R2jJIwDhJIupkTldX5kHZmo6Nyh8tjeMgeEbp4Tp05CfyUTWWM16gaGkwW2Gto3sJtv0AiA_PzSN_dDziD5fRSH2Q2JTW4g03Uc9SBelL2fFiQifPSc3-mI4i8QHIswd_qPnSAnHxBW6SLJFqY-qIG6soLzt2VnH5hpVvakMfO27A82DQrcoFDFsqRb8KgLEoL5u-6NbgwKSNFjfIrLFg9IzrQI7oktylkFrc_EWL_smmL6iuT5WEYt4jBwtMvyDD6nVHzzx7jd8J3XQqjXfWuH_uTAX6cOHprzaPn05QRAluZgcBL-FSQJ3Qw7PjpoiLyd3DGL77nfl_m9cpAnpz3ojtajP7Gb-aq_xa_JIqxbnuBDBkeyN8pOQp--ZD7T2BOAgS7poVoqPFXRYIJOwKtOcrj6UdPN2yrx-44ZMTJYzwcGELnFRs32PKx8TiiF1pKSwo4NB5Z97_0k_WbyBwyNajMtRUPmEuTr9VoO7CBwe1r3U3iIZbBKCfJjiG5FQToqzku31_YAs5OIIaV4B9ifLt5PwUA4mO-7XqgO1VQQjt2cUQo3Ui3EKWEJ-ov7F3wf_byGsguBwv2qMuAQiLBqs5jxrJUxyYIJAM7B_TtUjpQnNERvHEkt9TxCN8Kc6L-MejMOfu3VPdArf38naQjvBjBAZDznV639bkIRED7-soJbGMcGEyGWUqAVs9vkFleO9S4YLNvFShwo3ujBd7SMMdAyvi851CXT5uN5SDtaxmQnUGzAXmPJ9-UoJF23lSGB26eMdnIerzFoYMCgWPHyvt949IrsUKnpjuxebqQYVSrppmhIIrD8R255bJGSscVwdbrd9iA9-gHoB3UzCr5pd3gfW9Z6ynT4dQVILqtj0KgrDOHw4AIBqmwaecTBi5BeyXJx2oF1ClqS_7AanfqNToLcAwaKXnrK4RGyrX_mXHUFX9cT-o-eGqhi0lifCcJixwb3kG2AhP1USNNsCz31m40_c7cm7JcqLbzCnz4hvbivUvON5rf6kQ8PrfrjNrZA73VVIKhgZBDHxsHa3skwQvq-JH_3QulELy1-6vL5Kq84bg3ZPQxOUtxBRuyjxEJkpgG-sED2pYsKrUPqo0Ku_ggMTQjvoGGYRBt5uMlVX4pdB1zhOe1ZjcvPb8IwnL_BdLX4NvLpN97KH9Ot45bLeVTCGpv5UH8Nnm5CzQ53wqsOUD-9u5hqrSwx89sF7h8TlN9non95r7b_oHkU1R_czZ-ZjL6EubsUx4w-rWKwVU7GYde-ie62v8jcaLhkM72O4B0UvCfY2t3GtruZ4OirX44hWfOPujFr5L6bOkVSMKONJFooIJ2RIwCw64Mczkle2zQZ1P3u1DrMS5s65h-gNTwSGw3qyQBwF58-um9ycDis6f6O0ggqubsCDlsW7Vdnk_GlETHLDQ7lR_lRG1g3kRQEhKz2iwzxQan01X021EJd4TlocJYafpp8HU_rgcJdUmcvPFgB2xysE6F1vYdUAdovDztLftb5Bad4aKueUfDs8haq9TBgosHQinvKFfazE2StHUaEAVK_BiOYrH1XsrFQlXuMwhQlRgA9L3Q663gMrnhnfcQPSNd7P5EhqbadtddoVrLOKhMD5yBJj9RiC0vamCGVr2LA7hStIPBGysTBanE3u4bT-TKe2qCOskvfR2xU8NSlai9b8d57zkuxklf7LaDnMi-xu9TOqduYFfXOn87uqjaN3_emcq0NExYcQ1fMUMcbOuGoW6qeWlWmMtANjI3VaJCa_v2JYJ4cyl4gUoboC42d2esKg_Em2XfqUkKQh4XTG673LC1ebToWGPRvFtTQM3gZ4Wh5JY4pL58VeSsf1jhINWsytNpgGckHCK11BzUUx4MABT2BuMWf-a_5DV4KYdmXHn_AKAqoZWHgE2hC2Q6DUEaKTm7AV56Cm5vo-NibALDGH1zG8ih5C3dmHvQmES7vUOVM1jPS6k7paHXEwnPFE9M-zg6XmjKjdvSZ04lauZEeCjSJPb4E_v-uWlwkdHsDcTxfj9oTjfEpX0mZxIuT_Ex7Mx2I7DUHDUQgKgZT9n1TQym9patiPO8VYzYuoXrsEeLS1Mk5N3AmQXeB89x85_Xj2plBbDOqqMpAD2uMBXwHI4kut10unkHhl3S0JtA1tE0ukxTRaitpDQveHfao0tQC8gy4JEA6M5AD7iyWOm_iuW9baElC-R_g_6s_X1t2qv4mWwd8P-h7yFm4XEZg_oJEIA40hGwSPKD1d-b9QRz7Kl734V5RvMw1ekdsvZ9dVKNcPffkGX0inTp8RgkOWFUnS0hZpxuNbte3-rGWEt6Syy4x2jaH-Zr6o667kigSt1Q3cQO_eqQtq4VWuFmYIbDzkEbIKmIHY52gh-rB5k-FMQqCs-ay5Blj_IpvfcImMtrZBrbhL89gzGNRonBZEa-9kJeu4jr2_DLzw14KJR5zVNwiGLub3jJkgYqOZZ5ee_oNchx3v68S3wHyFnZA9IIaXRZjYLMrjD699h9SZvkTHdGAwICpyOjrfYbgX_7woRp1ZWBslOamnw6mDqJAk22nb1a8cpdGNP2IjXVRtuqIB8y36bHEFjChDTxERZ2dsz7a2mp5qM2Xz75OGBM77DAjnGpU7GFXDnolAnAsU5T3dd-LLnVlVhvzyuZWg7ZdH-0WsVVCezyIsQnm3WMpdPrlUcHtT6fyY2fhJVIm1QJEES5wEiEPMRrmGQ68V-q8TWlrPan6LU5Kr8Ak0nJKhE-r5bcaemeUbIsY4a9n2YDZck9CI6VGumMccelQ61Bhs5vgQ0W4AID90TXnUtJjWrVcgdhrLCWV_kv2_YSqDDoI6TM0oJKNaoNeG2HXCxXpHy8izUvfMwHvdniW3c4BPnvMpQW83bXrMPteKk-CFXdwQ6bB2PzzXAzWTp5q6D5cLWAyPJjju4AmopBUJmRwp0tjulMCClWqMiB08y8DIWDDLAAaG7Q-de-_Q-T6tZy4LRk_c0sYOtAaNCA1HgTDSLvP4j-xeuu8DrKv5SqefP2J7LLFM_JAi1gRh_84NUvUDvBdexr9wZI8eXjnnoDvP6KTosKCLmSC_ErmtzRXfUg1mz5fNVtlKSm03tqzmfL46iKDATVuEejDtlo34djj7uBV5DUw4lDIpQY1VsO1Ozgpoz9i8sNcRKQ-K3Of-vDL6R28gLBUq0Xo3nm1hAJgjc68C57jrMlJhD8GM6AeoGnnhDTfJ2xuxsdnH6i06qFUKcuTmA8l23Ek-A3ryx8DHAIaRX40d3e5MwaUqbglufHWBGId7KBiaiFuD3LhJC0CLl23XyHf225Rd4lir9LpltmuaRLnyS0FwIGZMaRmxQ-SWB2fDVzj81SJpo9lPDsuLu_ji7AA1cx-PnTj5fVp3APeRmy9E0A2v8hCKm4C6tPuvgC7Xp6MV8epxYIsGRiTy5wlHQE0FUuOdBtBH0rmGJDf4HQJoZHjhDhOJZqkvlDtEowB1mtndHgRz-0lpQurRm-RwKvl4n0quBfWZ1GL_PmiZIO36Iyyw4BRt3c1a5Zc5ilweQcle_-ZxawS1aAXXOaknt2c6AGB5JnmrTz2dXS7A8M20uNp7Cv8RoeiCYjPa1Co3Nr_6BuQL7HFxNsyk1AXDbG2qUJljSeWG3YFkaPHxgTw7aAefXrFFL_GNPi0YtageYJq3WN6lrdQ2CB0g7QLoj9dsHlAGhm8PtUESBUBbSyJVOm1lCuGGbB7psYxOLLO3BSqnXHb0--sDiyCTKMi-80rtMiHttXC3zAxXUFQjTre3a8KNohgPWx1PTAbxf96enJ33rhBV-2ewMIROT9j-K_Esee0eWUcTmt9v0yHW-V5ij0Hopx7oaXadNQLdgBJwUDf6R9xEktHhzUkyJ0g73gjrKQz2EidorhljD9LSFMAlUuRTkUhG35crMduH9TAAEgOHXZI24CD5Fz3n2KgXKoxWHlpaLlTwBXK1xLHVCrqCqvsBo60w5FV7cmdNTBjFbDU1EKSHLopt_aMgtT_6Fg1ZT6H2p0CAvvbinLkTLop3pSVU1_itnzRHOf3ayHzMrmSN_pI_03Of_63ZuHJmRWRCd7s1PviAo-B1LcG52VTanJz0JCF1RAlPj9-2DIgJLxDgNcPI96cTqZBbLk-rwKlebrmX6d5CBg3V5pmJKkgLIj5FpTmhiXhqDHHJvu-BxfzDQl2c8QtQYF6aygihfCCluN5biEv51XKRDpC-S3sU3USofDTgcg1pznwUvVv2eL8nWywckhIHWnip7z_ptCTmyn7BEzzgRgGLA_pLG17SPRJP6laoXHG_dprfpRM7gcLJZQ2zk29W2zVEpFwWePGpnQbpPjPqcOBiQfewxwnLHEuV8yGBR7Y-SEKrc6M6v8AHYk9oLXaRu1qBKkLUKSzKQhNFtfl-h-J8Adf0W9hxYSt6QNzf1YUuE8H_w2SrUGcVnsCnIQY_xu11sJ-0d-T2oFelzeEoasMeeCDamuFQye14ps0k4cM8vXpk_7ZrVE7rQmEpW40_n1iNHwB4UINg9CnQGXH98DzBBCoGPZpA1SELOwGTcJGcBZVQ5Tfey1SRFwXWJO0QFHfDb5-_tQUj9o30MhJBGxOftnwLaFROLgq3FuSBRM9dYsdlpHe1SILQXKVIwjXcOVMFgmbDq_hMSNFlMvblX9LLBduT9cXk6JhBVcxb8-oKbvbjL7zqQHOgke3ZC6oDEvcew2YzLMiNLiyGxJcthsyDfrWbhbq9DSRE7lYq9AVeh_Zc2wZq0RFh4CJGhXtW8WobIOY8JPIkyQKD4W_mKRxchykWyrCRliFId1Tzbgzu1NKxdZLiGZchs7MRgd-c_Kk0mDAvcVqyCSw5ZnlG8qWxmgwods9KD80tww2Bvp87a9Jwf-S8_PhqqG3ggGuLLm2CH71h7v6uA7f9-aCJKnlPiyb43OU2IK-rRgJf_U6VNAs1n0-RwWlaMttgA5wcecqRUlkneFkWpJOKDXpuAR9vwfoArMnPnp0jGQDN3-OPymX4xsYY6L4k0zC6j3zz9K2wgcGFD9kliVy2qwbeAqWL37Qdnr6sEbkxusF6IiYh-POUU_8rCQX03_uw0XHroHwK4mFajchjXmOY8ykOBQCIGPwCNI446xFhqWFDytDTXq9Eu651PlEqDELIcRwQz6KYWNJNlEFi4_f4GYS8sn0wpwte5R9QuaaLjc38obGBswmh15l9PrMvrWklBnnEZpV3NWmxQViKWcuey_QG_hRfQ-8Kjhv0f4D4L-d52x89yVXeVu0wbN_GstklEGCCecqvmQi1vXDf2FKr69Md-TE-mAh9pA-72vepP3guNcHz6PqzzOQX9Sj1uNZCkB0heHrXuCunn_Elv3ZvHZ-9AE26ybqtRVxaHtYrbtX9AKVk7ud_YdFPxSq-HeavXCXOBDGxEVleN03Q01jj7xoz5MjhKrVDF7XOobW0xMLtPfJLLmEGkBtSrLFCDGo1T7T3DnEiFQzXZutM50_l0k_3DxzDKhI4s5rOeeTMjSXDaxjM52LLgwAanVnMtKEsEXFVF4b5xvu_xn5CzqW5T0TTDOFXm2Gdxj-t59bgRGmnO56K85rTGgeJyXBroTz8cS4hkgfm2fQKiDAQZ5iMJeY4iqKZJTrOYb0IueB_ez-I8XW_dibgUd-WcJNKYKf4KnZR9_Z8o4OofbCdVj2mcgunpgjbTCORNWj7IpYmkHcbIQFtXnnts_2WNf-TtE6xr-iIVkwGABYE7ugHl1BUO5yKuDmeTOijSxWQGO22dzPnGVQ4O7AuXUYBFRa6FKVEIIVyk49ggvgRFFerncqEW1s8LR9gCzMIsxH2jCOyOSqjWGdZncRqDWhF6NYgFsqs3BDGYspC1vd9KFYppnH5W7MRYb2Duoi9yb7SQhNarto9KaqqgiTdEWeOw3kSkTZxa1moEh8F3ueFWhjQXNW4I3_inDPUdw0Xcf703y7uitnAsi-235tGC36JkWMR9M9Dx1cQSnS0NWhOYjUPPrKSHW8QCY-ZAfEUSJfixJeXEEUI0YmuGlFCIrLFvtlqFjxzqJW4JPCfnB0jCC9Z07d7rwHznYBSkr_cis4gNwnPOa11060WODyso6zRSJ7Q57bPhULvgnMZHZq2hl5dygeAz-elG8XYIUmr8jwXKuVGT_hl13cNI5QHaxshgdJuTzE362jxI4c0usFIVIzwhX6KqDFtWIZ5skj8iGioS6pDkY5tTj91aRu1ZL9eQ7KSLBbPeqhZCjQJGuudUr4u7HGuz8lQR0KvuZqKGGaybbPYwzJSx9qkGwqr_RNT7RW7oDxNiPlUHEf1qvED5M5FBFt_YlTmVtLQDJHRxvx3jv-Nc9pm6tew-et17Z0lMcXypXhr138RTXZYHSwJXsHMTNNGZFHCuZsyrq-PywrzCm-i6tXstJXx79s9os_dAaYgMtYEjPNRCb29LjaNw6OL60MKAl0Fung52DEDjnxFCTp9ygM_IkmLw95r9nhdq3smfsasefn6cp3YnEG3skKDswqS2Ul8Pilfqz3JI7mVucw4zA08ICIXAxB_L8_MPXUPPrVrdcf2HHicjjFs5L7mabPyv6blX2uB0BJ8Pcsdr_qdm-JbxmEEZZnxmtaG0VPgo23-DaHHIdMnNa-4cElpS64Tqcanin5QIsd1e1jIBJcjLmGOjV0eJpawOICK6dIhgdAsgLyXT-ItiUkVc_7NrPdpe0Fag7jMtvqXlvi-JljdILhGfbT7o-rNPY2iJ32jKUIDVZTSADQRf7Psnt40y3m1Ccx6aN3JVhNrgihrfjMF4rhZkqrh7Rlzs350VVOar8RblBoycjjBh9-xyXXSp4OWebr4rK6w76HQqKoOdQZvFrBG0Y3Qfkq1tNnJyy7QA3ZZwhnVPzmvi7GeCLIZMNQLQ2A3mUvZXcmmcI2NmLBJuTHoQ5IBhmtMA9_b1qVTt-8iy0jIklazgzzUa0Zdl3IAuptdmJT7AGneTDhrR60WBnxVbbjJa-_LvOyVdEVimw6wNuUO0HIuyLo7s5MkR1D1SNShzV7PUtM2YKxUxbE1zEHkqiTIF1P5RxIhh85XAaIaJMlIxjhvtIUy--jiuzLh9HDDjDCuSMRrqOk958lSAkZnProYbHuRI12ViZ561Z4-whNCQwctuoP68FvRWLByoO2NtNSaPBC9aqNx6OWHcTTGdaip8MZLmD_xPjoq6O04HNxBsaCQeo2xqMkeoB74m_8HtZQIPHyEgW2cAnDDOPDRFspt8KN9TMgAWTf_Pa7eI1ZvWo1vZtjOUi9E9SARkpmtFNtaQP_NRLp_76h9B_piPJCdzuIl9QXbwscJOaDHIlYfeauN1j1zMGmSY1jS2UPNPF7Qfy1wUcdxLFuzGy_1YPe6i8DoMimj_c995kmHFKi9jIdBHrTz5p-pX_E01O95Wd1mzgCeQCo643zzQ10c93MASc5dgHgCjyTfT4RATHXhVrhhjnamu0xnLxIHt0qA43qDfQd23xzzp5gA0KLoQ-b9fYpo5tjD3z-A6BuVES9k9W60WN3nwxJiil6rjHSHxzq_rmoDj--EOBsKv5TcismcMk4IdBgoKWcsHGW43c5t5gmaA6c1QZPDHZWTnkHPZIsH2U1kMcsNHoWG-H-xQ65cz6cceu_ATRu26etMuEZo4ecqNSENhCQq7NlkEnWVacuW7qybowkEr2uIU-BB_wI1oHPKVupH-0ZOHsVZOgktQ5g1DWiXUVloBabeRIZJt7fDYFs5oNgXxggElnN9fK-fb8BQb9j2BraENpRQonC68YsbFQLoyefvK3WnO1GFQQg7qDqzhU9PgMU6CIYfMfuHAoFiXtaTsnykAIv7m0nckJn8nldATLqakn72ObT_rzRQXi_cKoksBvKel4sqg7FtoM9no5s9a3wT1OwRXNUZ5Jg5iYyFW9mlRV4-Pwo67XhiipGG-iXsqxlhmDQjmeJoBfOKfm3MWJccFO9hMReoCp1DDqP5wxG_1gFMhl4mHPgxQW24pRrYOO00YYdR9VVrsBdjalyjo4mK5PuWqP0O3BKTZ7-Al2P5_VyQ2MxMZAZCkHSE5tRIkq0k29sZLPM58yUwN5FkIrzop2PR_VNYNa2eY2jK-mVv7eYvwcq9LcF6JbJN79K9YyPI-dqKutPoFzFXQEijdF77VbVDQYN5v33gKMYWIyXUb_ZgBFZ9wwZkGkzK7aRR22QVhUMk-M6dZrVH365Cmnboiq_7ZqSIa49uF1qlWbCljkpXMDxF8i0YGRdx4CUSU6vfyKyMUtCb-c6ZxGztojxz72-u3SwPOEJeRNUjpgH25LHo21ORRGuDHM0p04CyxXYe6YH-qYyINgouQ58GorDnhZJfLssqXFDEV_HeQfuZp-KsEnHSMDgX7ibItCu_ETXE2ano2M0XnOjdmSPRHl1aFyQAWkHsgTsrlzucRFcDhkK1BNIGPgC4eWce4bsaf_DHP0OJW8qVEnd15Oj1r9Om2K-vL5pYCLySkxA85DSgMKNOXPsPV3wGkiJjLJqn250v5aiwAziMHrcY5ik4Fm2AvDlRXPvGqXOuQG-zJsFc05J-1TBLgT1wZ1b2mw_qihmlJt71mthNKfgjmCMtx6WVKgRGM2lhdZ6gXt_9AkBcf3Rax9inuLnPgfaOZSCNa-MMR5yVa7ql7i-NwvuupwuuTuuKGkXv_-T3EK-Ky418dDDOMTgpW8nHiUM6Y5uBu6v__N8NMYvnJmujw6dUTNMR-R6vgaXdDtzs6a4KAccwIgqQ43uhgDexj9x4OB4304dKb5PJ2HpgIlnXlhjB-JGmnQAbAIaLrEcW9V0S0PX4H_Mz4NGqaAtDTeeiw=',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1ffe148588191812b659c6dc35ce60003919771fccd27',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1ffe148588191812b659c6dc35ce60003919771fccd27',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1ffe148588191812b659c6dc35ce60003919771fccd27',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1ffe148588191812b659c6dc35ce60003919771fccd27',
                    ),
                    TextPart(content=IsStr(), id='msg_68c200091ccc8191b38e07ea231e862d0003919771fccd27'),
                ],
                usage=RequestUsage(input_tokens=23, output_tokens=2030, details={'reasoning_tokens': 1728}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68c1ffe0f9a48191894c46b63c1a4f440003919771fccd27',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run(
        'Considering the way to cross the street, analogously, how do I cross the river?',
        model=BedrockConverseModel(
            'us.anthropic.claude-sonnet-4-20250514-v1:0',
            provider=bedrock_provider,
            settings=BedrockModelSettings(
                bedrock_additional_model_requests_fields={'thinking': {'type': 'enabled', 'budget_tokens': 1024}}
            ),
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
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        signature=IsStr(),
                        provider_name='bedrock',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(input_tokens=1241, output_tokens=495),
                model_name='us.anthropic.claude-sonnet-4-20250514-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_details={'finish_reason': 'end_turn'},
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_bedrock_anthropic_tool_with_thinking(allow_model_requests: None, bedrock_provider: BedrockProvider):
    """When using thinking with tool calls in Anthropic, we need to send the thinking part back to the provider.

    This tests the issue raised in https://github.com/pydantic/pydantic-ai/issues/2453.
    """
    m = BedrockConverseModel('us.anthropic.claude-3-7-sonnet-20250219-v1:0', provider=bedrock_provider)
    settings = BedrockModelSettings(
        bedrock_additional_model_requests_fields={'thinking': {'type': 'enabled', 'budget_tokens': 1024}},
    )
    agent = Agent(m, model_settings=settings)

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert result.output == snapshot("""\
Based on your location in Mexico, the largest city is Mexico City (Ciudad de México). It's not only the capital but also the most populous city in Mexico with a metropolitan area population of over 21 million people, making it one of the largest urban agglomerations in the world.

Mexico City is an important cultural, financial, and political center for the country and has a rich history dating back to the Aztec empire when it was known as Tenochtitlán.\
""")


async def test_bedrock_group_consecutive_tool_return_parts(bedrock_provider: BedrockProvider):
    """
    Test that consecutive ToolReturnPart objects are grouped into a single user message for Bedrock.
    """
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    now = datetime.datetime.now()
    # Create a ModelRequest with 3 consecutive ToolReturnParts
    req = [
        ModelRequest(parts=[UserPromptPart(content=['Hello'])]),
        ModelResponse(parts=[TextPart(content='Hi')]),
        ModelRequest(parts=[UserPromptPart(content=['How are you?'])]),
        ModelResponse(parts=[TextPart(content='Cloudy')]),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='tool1', content='result1', tool_call_id='id1', timestamp=now),
                ToolReturnPart(tool_name='tool2', content='result2', tool_call_id='id2', timestamp=now),
                ToolReturnPart(tool_name='tool3', content='result3', tool_call_id='id3', timestamp=now),
            ]
        ),
    ]

    # Call the mapping function directly
    _, bedrock_messages = await model._map_messages(req, ModelRequestParameters())  # type: ignore[reportPrivateUsage]

    assert bedrock_messages == snapshot(
        [
            {'role': 'user', 'content': [{'text': 'Hello'}]},
            {'role': 'assistant', 'content': [{'text': 'Hi'}]},
            {'role': 'user', 'content': [{'text': 'How are you?'}]},
            {'role': 'assistant', 'content': [{'text': 'Cloudy'}]},
            {
                'role': 'user',
                'content': [
                    {'toolResult': {'toolUseId': 'id1', 'content': [{'text': 'result1'}], 'status': 'success'}},
                    {'toolResult': {'toolUseId': 'id2', 'content': [{'text': 'result2'}], 'status': 'success'}},
                    {'toolResult': {'toolUseId': 'id3', 'content': [{'text': 'result3'}], 'status': 'success'}},
                ],
            },
        ]
    )


async def test_bedrock_model_thinking_part_stream(allow_model_requests: None, bedrock_provider: BedrockProvider):
    m = BedrockConverseModel(
        'us.anthropic.claude-sonnet-4-20250514-v1:0',
        provider=bedrock_provider,
        settings=BedrockModelSettings(
            bedrock_additional_model_requests_fields={
                'thinking': {'type': 'enabled', 'budget_tokens': 1024},
            }
        ),
    )
    agent = Agent(m)

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='Hello') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert event_parts == snapshot(
        [
            PartStartEvent(index=0, part=ThinkingPart(content='The')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' user has')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' greeted me with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a simple "Hello".')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I should respond in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a friendly and wel')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='coming manner.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' This is a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' straightforward greeting')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=" so I'll respond warm")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ly and ask')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' how I can help')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' them today.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(signature_delta=IsStr(), provider_name='bedrock')),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content='The user has greeted me with a simple "Hello". I should respond in a friendly and welcoming manner. This is a straightforward greeting, so I\'ll respond warmly and ask how I can help them today.',
                    signature='Eu0CCkgIBxABGAIqQJDccbDQkr81n7QjZ0Fi43umSvw0YvnGkMPEpaGAa2btYHyWw06KhwckvsnKzpKcxiRJT35meoG4/pdrTUiy2UISDPDaEWfOl3+HlRVsCxoMzfiqBp252RMvpmEyIjCbQ97Ac9Epkr5mgxeu1vGtJg+fDWIg0UnpMM8NYknhhvJmsXpYrfquwGL1ZnlBslUq0gHtbAAPwlWPmiQXU7gDQCDW9IdMVyw42b4f5MrAlpWkPWOJc9H+yYv0TpP/jY72SD1opqwkWnBgkzbi7A2jPmEFzIMQSO1KDXha5ADqQ3cLYMmVdNTSH9wlM7G7/JJ2/cqowqkwD6/q1AnYzcPte9iC67fY1LYN0NMCOSABFojP1rmkv9YBEulx5Y6eQpeVXBQiIqcGoCmWSumpGBskS1KxGerUmzUB0JmJnTENv4x3fSGSUSEPqMiz6Ebao8sVkb1wCWuZEXWJGtiQLMIm1o471iEYAQ==',
                    provider_name='bedrock',
                ),
                next_part_kind='text',
            ),
            PartStartEvent(index=1, part=TextPart(content='Hello! It'), previous_part_kind='thinking'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta="'s nice")),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to meet you.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' How can I help')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' you today?')),
            PartEndEvent(index=1, part=TextPart(content="Hello! It's nice to meet you. How can I help you today?")),
        ]
    )
    assert agent_run.result is not None
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Hello',
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user has greeted me with a simple "Hello". I should respond in a friendly and welcoming manner. This is a straightforward greeting, so I\'ll respond warmly and ask how I can help them today.',
                        signature=IsStr(),
                        provider_name='bedrock',
                    ),
                    TextPart(content="Hello! It's nice to meet you. How can I help you today?"),
                ],
                usage=RequestUsage(input_tokens=36, output_tokens=73),
                model_name='us.anthropic.claude-sonnet-4-20250514-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_details={'finish_reason': 'end_turn'},
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_bedrock_mistral_tool_result_format(bedrock_provider: BedrockProvider):
    now = datetime.datetime.now()
    req = [
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='tool1', content={'foo': 'bar'}, tool_call_id='id1', timestamp=now),
            ]
        ),
    ]

    # Models other than Mistral support toolResult.content with text, not json
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    # Call the mapping function directly
    _, bedrock_messages = await model._map_messages(req, ModelRequestParameters())  # type: ignore[reportPrivateUsage]

    assert bedrock_messages == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {'toolResult': {'toolUseId': 'id1', 'content': [{'text': '{"foo":"bar"}'}], 'status': 'success'}},
                ],
            },
        ]
    )

    # Mistral requires toolResult.content to hold json, not text
    model = BedrockConverseModel('mistral.mistral-7b-instruct-v0:2', provider=bedrock_provider)
    # Call the mapping function directly
    _, bedrock_messages = await model._map_messages(req, ModelRequestParameters())  # type: ignore[reportPrivateUsage]

    assert bedrock_messages == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {'toolResult': {'toolUseId': 'id1', 'content': [{'json': {'foo': 'bar'}}], 'status': 'success'}},
                ],
            },
        ]
    )


async def test_bedrock_no_tool_choice(bedrock_provider: BedrockProvider):
    my_tool = ToolDefinition(
        name='my_tool',
        description='This is my tool',
        parameters_json_schema={'type': 'object', 'title': 'Result', 'properties': {'spam': {'type': 'number'}}},
    )
    mrp = ModelRequestParameters(output_mode='tool', function_tools=[my_tool], allow_text_output=False, output_tools=[])

    # Amazon Nova supports tool_choice
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    tool_config = model._map_tool_config(mrp)  # type: ignore[reportPrivateUsage]

    assert tool_config == snapshot(
        {
            'tools': [
                {
                    'toolSpec': {
                        'name': 'my_tool',
                        'description': 'This is my tool',
                        'inputSchema': {
                            'json': {'type': 'object', 'title': 'Result', 'properties': {'spam': {'type': 'number'}}}
                        },
                    }
                }
            ],
            'toolChoice': {'any': {}},
        }
    )

    # Anthropic supports tool_choice
    model = BedrockConverseModel('us.anthropic.claude-3-7-sonnet-20250219-v1:0', provider=bedrock_provider)
    tool_config = model._map_tool_config(mrp)  # type: ignore[reportPrivateUsage]

    assert tool_config == snapshot(
        {
            'tools': [
                {
                    'toolSpec': {
                        'name': 'my_tool',
                        'description': 'This is my tool',
                        'inputSchema': {
                            'json': {'type': 'object', 'title': 'Result', 'properties': {'spam': {'type': 'number'}}}
                        },
                    }
                }
            ],
            'toolChoice': {'any': {}},
        }
    )

    # Other models don't support tool_choice
    model = BedrockConverseModel('us.meta.llama4-maverick-17b-instruct-v1:0', provider=bedrock_provider)
    tool_config = model._map_tool_config(mrp)  # type: ignore[reportPrivateUsage]

    assert tool_config == snapshot(
        {
            'tools': [
                {
                    'toolSpec': {
                        'name': 'my_tool',
                        'description': 'This is my tool',
                        'inputSchema': {
                            'json': {'type': 'object', 'title': 'Result', 'properties': {'spam': {'type': 'number'}}}
                        },
                    }
                }
            ]
        }
    )


async def test_bedrock_model_stream_empty_text_delta(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel(model_name='openai.gpt-oss-120b-1:0', provider=bedrock_provider)
    agent = Agent(model)

    result: AgentRunResult | None = None
    events: list[AgentStreamEvent] = []
    async for event in agent.run_stream_events('Hi'):
        if isinstance(event, AgentRunResultEvent):
            result = event.result
        else:
            events.append(event)

    assert result is not None
    # The response stream contains `{'contentBlockDelta': {'delta': {'text': ''}, 'contentBlockIndex': 0}}`, but our response should not have any empty text parts.
    assert not any(part.content == '' for part in result.response.parts if isinstance(part, TextPart))
    assert events == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ThinkingPart(
                    content='The user just says "Hi". We need to respond appropriately, friendly greeting. No special instructions. Should be short.'
                ),
            ),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content='The user just says "Hi". We need to respond appropriately, friendly greeting. No special instructions. Should be short.'
                ),
                next_part_kind='text',
            ),
            PartStartEvent(index=1, part=TextPart(content='Hello! How can I help'), previous_part_kind='thinking'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' you today?')),
            PartEndEvent(index=1, part=TextPart(content='Hello! How can I help you today?')),
        ]
    )


@pytest.mark.vcr()
async def test_bedrock_error(allow_model_requests: None, bedrock_provider: BedrockProvider):
    """Test that errors convert to ModelHTTPError."""
    model_id = 'us.does-not-exist-model-v1:0'
    model = BedrockConverseModel(model_id, provider=bedrock_provider)
    agent = Agent(model)

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('hello')

    assert exc_info.value.status_code == 400
    assert exc_info.value.model_name == model_id
    assert exc_info.value.body.get('Error', {}).get('Message') == 'The provided model identifier is invalid.'  # type: ignore[union-attr]


@pytest.mark.vcr()
async def test_bedrock_streaming_error(allow_model_requests: None, bedrock_provider: BedrockProvider):
    """Test that errors during streaming convert to ModelHTTPError."""
    model_id = 'us.does-not-exist-model-v1:0'
    model = BedrockConverseModel(model_id, provider=bedrock_provider)
    agent = Agent(model)

    with pytest.raises(ModelHTTPError) as exc_info:
        async with agent.run_stream('hello'):
            pass

    assert exc_info.value.status_code == 400
    assert exc_info.value.model_name == model_id
    assert exc_info.value.body.get('Error', {}).get('Message') == 'The provided model identifier is invalid.'  # type: ignore[union-attr]


async def test_cache_point_filtering():
    """Test that CachePoint is filtered out in Bedrock message mapping."""
    from itertools import count

    # Test the static method directly
    messages = await BedrockConverseModel._map_user_prompt(UserPromptPart(content=['text', CachePoint()]), count())  # pyright: ignore[reportPrivateUsage]
    # CachePoint should be filtered out, message should still be valid
    assert len(messages) == 1
    assert messages[0]['role'] == 'user'
