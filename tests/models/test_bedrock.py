from __future__ import annotations as _annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from inline_snapshot import snapshot
from typing_extensions import TypedDict

from pydantic_ai import (
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    CachePoint,
    DocumentUrl,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ImageUrl,
    ModelMessage,
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
from pydantic_ai.builtin_tools import CodeExecutionTool
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, ModelRetry, UsageLimitExceeded, UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    BuiltinToolCallEvent,  # pyright: ignore[reportDeprecated]
    BuiltinToolResultEvent,  # pyright: ignore[reportDeprecated]
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.profiles import DEFAULT_PROFILE
from pydantic_ai.providers import Provider
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

from ..conftest import IsDatetime, IsInstance, IsNow, IsStr, try_import

with try_import() as imports_successful:
    from botocore.exceptions import ClientError
    from mypy_boto3_bedrock_runtime.type_defs import MessageUnionTypeDef, SystemContentBlockTypeDef, ToolTypeDef

    from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelName, BedrockModelSettings
    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.providers.bedrock import BedrockProvider
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='bedrock not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolCallEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolCallPart` instead.:DeprecationWarning'
    ),
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolResultEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolReturnPart` instead.:DeprecationWarning'
    ),
]


class _StubBedrockClient:
    """Minimal Bedrock client that always raises the provided error."""

    def __init__(self, error: ClientError):
        self._error = error
        self.meta = SimpleNamespace(endpoint_url='https://bedrock.stub')

    def converse(self, **_: Any) -> None:
        raise self._error

    def converse_stream(self, **_: Any) -> None:
        raise self._error

    def count_tokens(self, **_: Any) -> None:
        raise self._error


class _StubBedrockProvider(Provider[Any]):
    """Provider implementation backed by the stub client."""

    def __init__(self, client: _StubBedrockClient):
        self._client = client

    @property
    def name(self) -> str:
        return 'bedrock-stub'

    @property
    def base_url(self) -> str:
        return 'https://bedrock.stub'

    @property
    def client(self) -> _StubBedrockClient:
        return self._client

    def model_profile(self, model_name: str):
        return DEFAULT_PROFILE


def _bedrock_model_with_client_error(error: ClientError) -> BedrockConverseModel:
    """Instantiate a BedrockConverseModel wired to always raise the given error."""
    return BedrockConverseModel(
        'us.amazon.nova-micro-v1:0',
        provider=_StubBedrockProvider(_StubBedrockClient(error)),
    )


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
                timestamp=IsDatetime(),
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
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                provider_details={'finish_reason': 'end_turn'},
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.vcr()
async def test_bedrock_model_usage_limit_exceeded(
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-20250514-v1:0', provider=bedrock_provider)
    agent = Agent(model=model)

    with pytest.raises(
        UsageLimitExceeded,
        match='The next request would exceed the input_tokens_limit of 18 \\(input_tokens=23\\)',
    ):
        await agent.run(
            ['The quick brown fox jumps over the lazydog.', CachePoint(), 'What was next?'],
            usage_limits=UsageLimits(input_tokens_limit=18, count_tokens_before_request=True),
        )


@pytest.mark.vcr()
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


async def test_bedrock_request_non_http_error():
    error = ClientError({'Error': {'Code': 'TestException', 'Message': 'broken connection'}}, 'converse')
    model = _bedrock_model_with_client_error(error)
    params = ModelRequestParameters()

    with pytest.raises(ModelAPIError) as exc_info:
        await model.request([ModelRequest.user_text_prompt('hi')], None, params)

    assert exc_info.value.message == snapshot(
        'An error occurred (TestException) when calling the converse operation: broken connection'
    )


async def test_bedrock_count_tokens_non_http_error():
    error = ClientError({'Error': {'Code': 'TestException', 'Message': 'broken connection'}}, 'count_tokens')
    model = _bedrock_model_with_client_error(error)
    params = ModelRequestParameters()

    with pytest.raises(ModelAPIError) as exc_info:
        await model.count_tokens([ModelRequest.user_text_prompt('hi')], None, params)

    assert exc_info.value.message == snapshot(
        'An error occurred (TestException) when calling the count_tokens operation: broken connection'
    )


async def test_bedrock_stream_non_http_error():
    error = ClientError({'Error': {'Code': 'TestException', 'Message': 'broken connection'}}, 'converse_stream')
    model = _bedrock_model_with_client_error(error)
    params = ModelRequestParameters()

    with pytest.raises(ModelAPIError) as exc_info:
        async with model.request_stream([ModelRequest.user_text_prompt('hi')], None, params) as stream:
            async for _ in stream:
                pass

    assert 'broken connection' in exc_info.value.message


async def test_stub_provider_properties():
    # tests the test utility itself...
    error = ClientError({'Error': {'Code': 'TestException', 'Message': 'test'}}, 'converse')
    model = _bedrock_model_with_client_error(error)
    provider = model._provider  # pyright: ignore[reportPrivateUsage]

    assert provider.name == 'bedrock-stub'
    assert provider.base_url == 'https://bedrock.stub'


async def test_bedrock_model_structured_output(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(model=model, instructions='You are a helpful chatbot.', retries=5)

    class Response(TypedDict):
        temperature: str
        date: date
        city: str

    @agent.tool_plain
    async def temperature(city: str, date: date) -> str:
        """Get the temperature in a city on a specific date.

        Args:
            city: The city name.
            date: The date.

        Returns:
            The temperature in degrees Celsius.
        """
        return '30°C'

    result = await agent.run('What was the temperature in London 1st January 2022?', output_type=Response)
    assert result.output == snapshot({'temperature': '30°C', 'date': date(2022, 1, 1), 'city': 'London'})
    assert result.usage() == snapshot(RunUsage(requests=3, input_tokens=2019, output_tokens=120, tool_calls=1))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What was the temperature in London 1st January 2022?',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful chatbot.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='temperature',
                        args={'date': '2022-01-01', 'city': 'London'},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=571, output_tokens=22),
                model_name='us.amazon.nova-micro-v1:0',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='bedrock',
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                provider_details={'finish_reason': 'tool_use'},
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='temperature',
                        content='30°C',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful chatbot.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\

<thinking> The tool has provided the temperature for London on 1st January 2022, which was 30°C. I will now provide this information to the user.</thinking>
The temperature in London on 1st January 2022 was 30°C.\
"""
                    )
                ],
                usage=RequestUsage(input_tokens=627, output_tokens=67),
                model_name='us.amazon.nova-micro-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                provider_details={'finish_reason': 'end_turn'},
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content=[
                            {
                                'type': 'json_invalid',
                                'loc': (),
                                'msg': 'Invalid JSON: expected value at line 2 column 1',
                                'input': """\

<thinking> The tool has provided the temperature for London on 1st January 2022, which was 30°C. I will now provide this information to the user.</thinking>
The temperature in London on 1st January 2022 was 30°C.\
""",
                                'ctx': {'error': 'expected value at line 2 column 1'},
                            }
                        ],
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='You are a helpful chatbot.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'date': '2022-01-01', 'city': 'London', 'temperature': '30°C'},
                        tool_call_id='tooluse_qVHAm8Q9QMGoJRkk06_TVA',
                    )
                ],
                usage=RequestUsage(input_tokens=821, output_tokens=31),
                model_name='us.amazon.nova-micro-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                provider_details={'finish_reason': 'tool_use'},
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


async def test_bedrock_model_stream(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings={'temperature': 0.0})
    async with agent.run_stream('What is the capital of France?') as result:
        data = await result.get_output()
    assert data == snapshot(
        'The capital of France is Paris. Paris is not only the capital city but also the most populous city in France, and it is a major center for culture, commerce, fashion, and international diplomacy. Known for its historical landmarks, such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral, Paris is often referred to as "The City of Light" or "The City of Love."'
    )
    assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=13, output_tokens=82))


async def test_bedrock_model_anthropic_model_with_tools(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('anthropic.claude-v2', provider=bedrock_provider)
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings={'temperature': 0.0})

    @agent.tool_plain
    async def get_current_temperature(city: str) -> str:
        """Get the current temperature in a city.

        Args:
            city: The city name.

        Returns:
            The current temperature in degrees Celsius.
        """
        return '30°C'  # pragma: no cover

    # dated March 2025, update when no longer the case
    # TODO(Marcelo): Anthropic models don't support tools on the Bedrock Converse Interface.
    # I'm unsure what to do, so for the time being I'm just documenting the test. Let's see if someone complains.
    with pytest.raises(Exception):
        await agent.run('What is the current temperature in London?')


async def test_bedrock_model_anthropic_model_without_tools(
    allow_model_requests: None, bedrock_provider: BedrockProvider
):
    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=bedrock_provider)
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings={'temperature': 0.0})
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        "The capital of France is **Paris**. It's the largest city in France and has been the country's capital since the 12th century. Paris is known for its iconic landmarks like the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and its rich history, culture, and cuisine."
    )


async def test_bedrock_model_retry(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(
        model=model, instructions='You are a helpful chatbot.', model_settings={'temperature': 0.0}, retries=2
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
                    UserPromptPart(
                        content='What is the capital of France?',
                        timestamp=IsDatetime(),
                    ),
                ],
                instructions='You are a helpful chatbot.',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='<thinking> To determine the capital of France, I can use the provided tool that returns the capital of a given country. Since the country in question is France, I will use the tool with the country parameter set to "France". </thinking>\n'
                    ),
                    ToolCallPart(
                        tool_name='get_capital',
                        args={'country': 'France'},
                        tool_call_id=IsStr(),
                    ),
                ],
                usage=RequestUsage(input_tokens=426, output_tokens=66),
                model_name='us.amazon.nova-micro-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                provider_details={'finish_reason': 'tool_use'},
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='The country is not supported.',
                        tool_name='get_capital',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a helpful chatbot.',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
<thinking> It appears that there was an error in retrieving the capital of France as the tool indicated that the country is not supported. Since the tool is not able to provide the requested information, I will respond to the User with the information I have access to. </thinking> \n\

The capital of France is Paris. If you need any further information, feel free to ask!\
"""
                    )
                ],
                usage=RequestUsage(input_tokens=531, output_tokens=76),
                model_name='us.amazon.nova-micro-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                provider_details={'finish_reason': 'end_turn'},
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_bedrock_model_max_tokens(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings={'max_tokens': 5})
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is')


async def test_bedrock_model_top_p(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings={'top_p': 0.5})
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        'The capital of France is Paris. Paris is not only the capital city but also the most populous city in France, and it is a major center for culture, fashion, gastronomy, and international diplomacy.'
    )


async def test_bedrock_model_performance_config(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)
    model_settings = BedrockModelSettings(bedrock_performance_configuration={'latency': 'optimized'})
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings=model_settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        'The capital of France is Paris. It is one of the most visited cities in the world and is known for its rich history, culture, and iconic landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. Paris is also a major center for finance, diplomacy, commerce, fashion, science, and arts.'
    )


async def test_bedrock_model_guardrail_config(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    model_settings = BedrockModelSettings(
        bedrock_guardrail_config={
            'guardrailIdentifier': 'xbgw7g293v7o',
            'guardrailVersion': 'DRAFT',
            'trace': 'enabled',
        }
    )
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings=model_settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        "The capital of France is Paris. Paris is not only the capital city but also the most populous city in France, serving as the center of French government, culture, and commerce. It's known for its historical and cultural landmarks such as the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and many charming neighborhoods like Montmartre."
    )


async def test_bedrock_model_other_parameters(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    model_settings = BedrockModelSettings(
        bedrock_prompt_variables={'leo': {'text': 'aaaa'}},
        bedrock_additional_model_requests_fields={'test': 'test'},
        bedrock_request_metadata={'test': 'test'},
        bedrock_additional_model_response_fields_paths=['test'],
    )
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings=model_settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        'The capital of France is Paris. Paris is not only the capital city but also the most populous city in France, known for its significant cultural, political, and economic influence both within the country and globally. It is famous for landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, among many other historical and architectural treasures.'
    )


async def test_bedrock_model_service_tier(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    model_settings = BedrockModelSettings(bedrock_service_tier={'type': 'flex'})
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings=model_settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        'The capital of France is Paris. Paris is not only the capital city but also the most populous city in France, known for its significant cultural, political, and economic influence both within the country and globally. It is famous for landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, among many other historical and architectural treasures.'
    )


async def test_bedrock_model_iter_stream(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings={'top_p': 0.5})

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
                part=ToolCallPart(tool_name='get_temperature', tool_call_id=IsStr()),
                previous_part_kind='text',
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='{"city":"Paris"}', tool_call_id=IsStr()),
            ),
            PartEndEvent(
                index=1,
                part=ToolCallPart(tool_name='get_temperature', args='{"city":"Paris"}', tool_call_id=IsStr()),
            ),
            IsInstance(FunctionToolCallEvent),
            FunctionToolResultEvent(
                result=ToolReturnPart(
                    tool_name='get_temperature',
                    content='30°C',
                    tool_call_id=IsStr(),
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
    agent = Agent(m, instructions='You are a helpful chatbot.')

    result = await agent.run(['What fruit is in the image?', image_content])
    assert result.output == snapshot(
        'The image features a fruit that is round and has a green skin with brown dots. The fruit is cut in half, revealing its interior, which is also green. Based on the appearance and characteristics, the fruit in the image is a kiwi.'
    )


@pytest.mark.vcr()
async def test_video_as_binary_content_input(
    allow_model_requests: None, video_content: BinaryContent, bedrock_provider: BedrockProvider
):
    m = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

    result = await agent.run(['Explain me this video', video_content])
    assert result.output == snapshot(
        'The video shows a camera set up on a tripod, pointed at a scenic view of a rocky landscape under a clear sky. The camera remains stationary throughout the video, capturing the same view without any changes.'
    )


@pytest.mark.vcr()
async def test_image_url_input(
    allow_model_requests: None, bedrock_provider: BedrockProvider, disable_ssrf_protection_for_vcr: None
):
    m = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

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
async def test_video_url_input(
    allow_model_requests: None, bedrock_provider: BedrockProvider, disable_ssrf_protection_for_vcr: None
):
    m = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

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
async def test_document_url_input(
    allow_model_requests: None, bedrock_provider: BedrockProvider, disable_ssrf_protection_for_vcr: None
):
    m = BedrockConverseModel('anthropic.claude-v2', provider=bedrock_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

    document_url = DocumentUrl(url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf')

    result = await agent.run(['What is the main content on this document?', document_url])
    assert result.output == snapshot(
        'Based on the provided XML data, the main content of the document is "Dummy PDF file". This is contained in the <document_content> tag for the document with index="1".'
    )


@pytest.mark.vcr()
async def test_text_document_url_input(
    allow_model_requests: None, bedrock_provider: BedrockProvider, disable_ssrf_protection_for_vcr: None
):
    m = BedrockConverseModel('anthropic.claude-v2', provider=bedrock_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

    text_document_url = DocumentUrl(url='https://example-files.online-convert.com/document/txt/example.txt')

    result = await agent.run(['What is the main content on this document?', text_document_url])
    assert result.output == snapshot(
        """\
Based on the text in the <document_content> tag, the main content of this document appears to be:

An example text describing the use of "John Doe" as a placeholder name in legal cases, hospitals, and other contexts where a party's real identity is unknown or needs to be withheld. It provides background on how "John Doe" and "Jane Doe" are commonly used in the United States and Canada for this purpose, in contrast to other English speaking countries that use names like "Joe Bloggs". The text gives examples of using John/Jane Doe for legal cases, unidentified corpses, and as generic names on forms. It also mentions how "Baby Doe" and "Precious Doe" are used for unidentified children.\
"""
    )


async def test_s3_image_url_input(bedrock_provider: BedrockProvider):
    """Test that s3:// image URLs are passed directly to Bedrock API without downloading."""
    model = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)
    image_url = ImageUrl(url='s3://my-bucket/images/test-image.jpg', media_type='image/jpeg')

    req = [
        ModelRequest(parts=[UserPromptPart(content=['What is in this image?', image_url])]),
    ]

    _, bedrock_messages = await model._map_messages(req, ModelRequestParameters(), None)  # type: ignore[reportPrivateUsage]

    assert bedrock_messages == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {'text': 'What is in this image?'},
                    {
                        'image': {
                            'format': 'jpeg',
                            'source': {'s3Location': {'uri': 's3://my-bucket/images/test-image.jpg'}},
                        }
                    },
                ],
            }
        ]
    )


async def test_s3_video_url_input(bedrock_provider: BedrockProvider):
    """Test that s3:// video URLs are passed directly to Bedrock API."""
    model = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)
    video_url = VideoUrl(url='s3://my-bucket/videos/test-video.mp4', media_type='video/mp4')

    req = [
        ModelRequest(parts=[UserPromptPart(content=['Describe this video', video_url])]),
    ]

    _, bedrock_messages = await model._map_messages(req, ModelRequestParameters(), None)  # type: ignore[reportPrivateUsage]

    assert bedrock_messages == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {'text': 'Describe this video'},
                    {
                        'video': {
                            'format': 'mp4',
                            'source': {'s3Location': {'uri': 's3://my-bucket/videos/test-video.mp4'}},
                        }
                    },
                ],
            }
        ]
    )


async def test_s3_document_url_input(bedrock_provider: BedrockProvider):
    """Test that s3:// document URLs are passed directly to Bedrock API."""
    model = BedrockConverseModel('anthropic.claude-v2', provider=bedrock_provider)
    document_url = DocumentUrl(url='s3://my-bucket/documents/test-doc.pdf', media_type='application/pdf')

    req = [
        ModelRequest(parts=[UserPromptPart(content=['What is the main content on this document?', document_url])]),
    ]

    _, bedrock_messages = await model._map_messages(req, ModelRequestParameters(), None)  # type: ignore[reportPrivateUsage]

    assert bedrock_messages == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {'text': 'What is the main content on this document?'},
                    {
                        'document': {
                            'format': 'pdf',
                            'name': 'Document 1',
                            'source': {'s3Location': {'uri': 's3://my-bucket/documents/test-doc.pdf'}},
                        }
                    },
                ],
            }
        ]
    )


async def test_s3_url_with_bucket_owner(bedrock_provider: BedrockProvider):
    """Test that s3:// URLs with bucketOwner parameter are parsed correctly."""
    model = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)
    image_url = ImageUrl(url='s3://my-bucket/images/test-image.jpg?bucketOwner=123456789012', media_type='image/jpeg')

    req = [
        ModelRequest(parts=[UserPromptPart(content=['What is in this image?', image_url])]),
    ]

    _, bedrock_messages = await model._map_messages(req, ModelRequestParameters(), None)  # type: ignore[reportPrivateUsage]

    assert bedrock_messages == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {'text': 'What is in this image?'},
                    {
                        'image': {
                            'format': 'jpeg',
                            'source': {
                                's3Location': {
                                    'uri': 's3://my-bucket/images/test-image.jpg',
                                    'bucketOwner': '123456789012',
                                }
                            },
                        }
                    },
                ],
            }
        ]
    )


@pytest.mark.vcr()
async def test_text_as_binary_content_input(allow_model_requests: None, bedrock_provider: BedrockProvider):
    m = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

    text_content = BinaryContent(data=b'This is a test document.', media_type='text/plain')

    result = await agent.run(['What is the main content on this document?', text_content])
    assert result.output == snapshot(
        """\
The document you're referring to appears to be a test document, which means its primary purpose is likely to serve as an example or a placeholder rather than containing substantive content. Test documents are commonly used for various purposes such as:

1. **Software Testing**: To verify that a system can correctly handle, display, or process documents.
2. **Design Mockups**: To illustrate how a document might look in a particular format or style.
3. **Training Materials**: To provide examples for instructional purposes.
4. **Placeholders**: To fill space in a system or application where real content will eventually be placed.

Since this is a test document, it probably doesn't contain any meaningful or specific information beyond what is necessary to serve its testing purpose. If you have specific questions about the format, structure, or any particular element within the document, feel free to ask!\
"""
    )


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
                timestamp=IsDatetime(),
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
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
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
            ModelRequest(
                parts=[UserPromptPart(content=['Here is a PDF document: ', document_content])], timestamp=IsDatetime()
            ),
            ModelResponse(parts=[TextPart(content='foo bar')]),
            ModelRequest(
                parts=[UserPromptPart(content=['Here is another PDF document: ', document_content])],
                timestamp=IsDatetime(),
            ),
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
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content=IsStr()), ThinkingPart(content=IsStr())],
                usage=RequestUsage(input_tokens=12, output_tokens=693),
                model_name='us.deepseek.r1-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
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
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content=IsStr()), ThinkingPart(content=IsStr())],
                usage=RequestUsage(input_tokens=33, output_tokens=907),
                model_name='us.deepseek.r1-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
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
                timestamp=IsDatetime(),
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
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
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
                timestamp=IsDatetime(),
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
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
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
                timestamp=IsDatetime(),
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
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
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
                timestamp=IsDatetime(),
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
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
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
                timestamp=IsDatetime(),
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
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
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
    agent = Agent(m, instructions='You are a helpful assistant.', model_settings=settings)

    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='How do I cross the street?',
                        timestamp=IsDatetime(),
                    ),
                ],
                instructions='You are a helpful assistant.',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1ffe148588191812b659c6dc35ce60003919771fccd27',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1ffe148588191812b659c6dc35ce60003919771fccd27',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1ffe148588191812b659c6dc35ce60003919771fccd27',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1ffe148588191812b659c6dc35ce60003919771fccd27',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1ffe148588191812b659c6dc35ce60003919771fccd27',
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_68c200091ccc8191b38e07ea231e862d0003919771fccd27',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(input_tokens=23, output_tokens=2030, details={'reasoning_tokens': 1728}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 10, 22, 46, 57, tzinfo=timezone.utc),
                },
                provider_response_id=IsStr(),
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
                instructions='You are a helpful assistant.',
                timestamp=IsDatetime(),
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
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
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
    assert result.output == snapshot(
        """\
Based on your location in Mexico, the largest city is Mexico City (Ciudad de México). It's not only the capital but also the most populous city in Mexico with a metropolitan area population of over 21 million people, making it one of the largest urban agglomerations in the world.

Mexico City is an important cultural, financial, and political center for the country and has a rich history dating back to the Aztec empire when it was known as Tenochtitlán.\
"""
    )


async def test_bedrock_group_consecutive_tool_return_parts(bedrock_provider: BedrockProvider):
    """
    Test that consecutive ToolReturnPart objects are grouped into a single user message for Bedrock.
    """
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    now = datetime.now()
    # Create a ModelRequest with 3 consecutive ToolReturnParts
    req = [
        ModelRequest(parts=[UserPromptPart(content=['Hello'])], timestamp=IsDatetime()),
        ModelResponse(parts=[TextPart(content='Hi')]),
        ModelRequest(parts=[UserPromptPart(content=['How are you?'])], timestamp=IsDatetime()),
        ModelResponse(parts=[TextPart(content='Cloudy')]),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='tool1', content='result1', tool_call_id='id1', timestamp=now),
                ToolReturnPart(tool_name='tool2', content='result2', tool_call_id='id2', timestamp=now),
                ToolReturnPart(tool_name='tool3', content='result3', tool_call_id='id3', timestamp=now),
            ],
            timestamp=IsDatetime(),
        ),
    ]

    # Call the mapping function directly
    _, bedrock_messages = await model._map_messages(req, ModelRequestParameters(), BedrockModelSettings())  # type: ignore[reportPrivateUsage]

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
                timestamp=IsDatetime(),
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
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                provider_details={'finish_reason': 'end_turn'},
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_bedrock_mistral_tool_result_format(bedrock_provider: BedrockProvider):
    now = datetime.now()
    req = [
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='tool1', content={'foo': 'bar'}, tool_call_id='id1', timestamp=now),
            ],
            timestamp=IsDatetime(),
        ),
    ]

    # Models other than Mistral support toolResult.content with text, not json
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    # Call the mapping function directly
    _, bedrock_messages = await model._map_messages(req, ModelRequestParameters(), BedrockModelSettings())  # type: ignore[reportPrivateUsage]

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
    _, bedrock_messages = await model._map_messages(req, ModelRequestParameters(), BedrockModelSettings())  # type: ignore[reportPrivateUsage]

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
    tool_config = model._map_tool_config(mrp, BedrockModelSettings())  # type: ignore[reportPrivateUsage]

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
    tool_config = model._map_tool_config(mrp, BedrockModelSettings())  # type: ignore[reportPrivateUsage]

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
    tool_config = model._map_tool_config(mrp, BedrockModelSettings())  # type: ignore[reportPrivateUsage]

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


@pytest.mark.vcr()
@pytest.mark.parametrize(
    'model_name',
    [
        pytest.param('us.anthropic.claude-sonnet-4-5-20250929-v1:0', id='claude-sonnet-4-5'),
        pytest.param('us.amazon.nova-lite-v1:0', id='nova-lite'),
    ],
)
async def test_bedrock_cache_point_adds_cache_control(
    allow_model_requests: None, bedrock_provider: BedrockProvider, model_name: BedrockModelName
):
    """Record a real Bedrock call to confirm cache points reach AWS (requires ~1k tokens)."""
    model = BedrockConverseModel(model_name, provider=bedrock_provider)
    agent = Agent(
        model,
        system_prompt='YOU MUST RESPONSE ONLY WITH SINGLE NUMBER\n' * 50,  # More tokens to activate a cache
        model_settings=BedrockModelSettings(bedrock_cache_instructions=True),
    )
    long_context = 'ONLY SINGLE NUMBER IN RESPONSE\n' * 100  # More tokens to activate a cache

    result = await agent.run([long_context, CachePoint(), 'Response only number What is 2 + 3'])
    assert result.output == snapshot('5')
    # Different tokens usage depending on a model - could be written or read depending on the cassette read/write
    usage = result.usage()
    assert usage.cache_write_tokens >= 1000 or usage.cache_read_tokens >= 1000
    assert usage.input_tokens >= usage.cache_write_tokens + usage.cache_read_tokens


async def test_bedrock_cache_usage_includes_cache_tokens(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=bedrock_provider)
    agent = Agent(
        model,
        system_prompt='YOU MUST RESPONSE ONLY WITH SINGLE NUMBER\n' * 50,  # More tokens to activate a cache
        model_settings=BedrockModelSettings(bedrock_cache_instructions=True),
    )
    long_context = 'ONLY SINGLE NUMBER IN RESPONSE\n' * 100  # More tokens to activate a cache

    result = await agent.run([long_context, CachePoint(), 'Response only number What is 2 + 3'])
    assert result.output == snapshot('5')
    assert result.usage() == snapshot(RunUsage(input_tokens=1517, cache_read_tokens=1504, output_tokens=5, requests=1))


@pytest.mark.vcr()
async def test_bedrock_cache_write_and_read(allow_model_requests: None, bedrock_provider: BedrockProvider):
    """Integration test covering all cache settings using a recorded cassette.

    This test enables all 3 cache settings plus 2 manual CachePoints (5 total),
    which triggers the _limit_cache_points logic to strip the oldest one (limit is 4).
    """
    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=bedrock_provider)
    agent = Agent(
        model,
        system_prompt='YOU MUST RESPONSE ONLY WITH SINGLE NUMBER\n' * 50,  # More tokens to activate a cache
        model_settings=BedrockModelSettings(
            bedrock_cache_instructions=True,  # 1 cache point
            bedrock_cache_tool_definitions=True,  # 1 cache point
            bedrock_cache_messages=True,  # 1 cache point (on last user message)
        ),
    )

    @agent.tool_plain
    def catalog_lookup() -> str:  # pragma: no cover - exercised via agent call
        return 'catalog-ok'

    @agent.tool_plain
    def diagnostics() -> str:  # pragma: no cover - exercised via agent call
        return 'diagnostics-ok'

    long_context = 'Newer response with something except single number\n' * 10
    document = BinaryContent(data=b'You are a great mathematician', media_type='text/plain')
    # 2 CachePoints, more that maximum allowed, so will be stripped.
    run_args = [long_context, CachePoint(), document, CachePoint(), 'What is 10 + 11?']

    first = await agent.run(run_args)
    assert first.output == snapshot('21')
    first_usage = first.usage()
    assert first_usage == snapshot(RunUsage(input_tokens=1324, cache_write_tokens=1322, output_tokens=5, requests=1))

    second = await agent.run(run_args)
    assert second.output == snapshot('21')
    second_usage = second.usage()
    assert second_usage == snapshot(RunUsage(input_tokens=1324, output_tokens=5, cache_read_tokens=1322, requests=1))


@pytest.mark.vcr()
async def test_bedrock_cache_messages_with_document_as_last_content(
    allow_model_requests: None, bedrock_provider: BedrockProvider
):
    """Test the workaround for the AWS bug where cache points cannot be added after documents, so we insert them before the documents."""
    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=bedrock_provider)
    agent = Agent(
        model,
        system_prompt='YOU ARE A HELPFUL ASSISTANT THAT ANALYZES DOCUMENTS.\n' * 50,  # More tokens to activate a cache
        model_settings=BedrockModelSettings(
            bedrock_cache_messages=True,  # This should add a cache point to the last user message
        ),
    )

    # Create a document as the last piece of content in the user message
    document = BinaryContent(data=b'This is a test document with important analysis data.', media_type='text/plain')
    document2 = BinaryContent(data=b'This is a test document with unimportant data.', media_type='text/plain')
    run_args = [
        'YOU ARE A HELPFUL ASSISTANT THAT ANALYZES DOCUMENTS.\n' * 50,  # More tokens to activate a cache
        'Please analyze this document:',
        document,
        'And this document:',
        document2,
    ]

    result = await agent.run(run_args)
    assert result.output == snapshot("""\
I'll analyze the documents you've provided:

## Document 1 (Document 1.txt)
- **Content**: Contains important analysis data
- **Key characteristic**: Explicitly marked as containing "important" information
- **Purpose**: Appears to be a test document designed to hold significant analytical data

## Document 2 (Document 2.txt)
- **Content**: Contains unimportant data
- **Key characteristic**: Explicitly marked as containing "unimportant" information
- **Purpose**: Also a test document, but with data of lesser significance

## Summary
Both documents appear to be test files with minimal content. The main distinction between them is the stated importance level of their data - Document 1 contains information designated as important for analysis, while Document 2's data is marked as unimportant. Without more specific content or a particular analysis goal, these seem to be placeholder documents for testing document analysis capabilities.

Is there a specific aspect of these documents you'd like me to focus on or a particular type of analysis you need?\
""")

    usage = result.usage()
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0
    assert usage.cache_write_tokens > 0
    assert usage.cache_read_tokens == 0

    messages = result.all_messages()

    result = await agent.run('How long is the doc?', message_history=messages)
    assert result.output == snapshot("""\
Based on the documents provided:

**Document 1 (Document 1.txt)**: 10 words
- "This is a test document with important analysis data."

**Document 2 (Document 2.txt)**: 8 words
- "This is a test document with unimportant data."

Both documents are very short - just single sentences. If you're asking about character count instead:

- Document 1: 55 characters (including spaces)
- Document 2: 49 characters (including spaces)\
""")

    usage = result.usage()
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0
    assert usage.cache_write_tokens > 0
    assert usage.cache_read_tokens > 0


@pytest.mark.vcr()
async def test_bedrock_cache_messages_with_image_as_last_content(
    allow_model_requests: None, bedrock_provider: BedrockProvider, image_content: BinaryContent
):
    """Test that cache points can be added after images without the workaround necessary for documents."""
    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=bedrock_provider)
    agent = Agent(
        model,
        system_prompt='YOU ARE A HELPFUL ASSISTANT THAT ANALYZES IMAGES.\n' * 50,  # More tokens to activate a cache
        model_settings=BedrockModelSettings(
            bedrock_cache_messages=True,  # This should add a cache point to the last user message
        ),
    )

    # Create a document as the last piece of content in the user message
    run_args = [
        'YOU ARE A HELPFUL ASSISTANT THAT ANALYZES IMAGES.\n' * 50,  # More tokens to activate a cache
        'Please analyze the following image:',
        image_content,
    ]

    result = await agent.run(run_args)
    assert result.output == snapshot("""\
I'd be happy to analyze this image for you!

This is a close-up photograph of a **kiwi fruit cross-section**. Here are the key details:

## Visual Characteristics:
- **Color Palette**: Vibrant green flesh with a pale cream/white center
- **Seeds**: Multiple small, black, teardrop-shaped seeds arranged in a radial pattern around the center
- **Texture**: The flesh appears juicy and translucent with a gradient from bright green at the edges to lighter green near the center
- **Skin**: Brown fuzzy skin visible around the perimeter of the slice
- **Pattern**: Natural starburst or sunburst pattern created by the seed arrangement

## Composition:
- The slice is photographed from directly above against a white background
- The fruit is cut perpendicular to its length, showing a perfect circular cross-section
- The lighting is bright and even, highlighting the fruit's natural moisture and color variations

## Notable Features:
- The radial symmetry creates an aesthetically pleasing natural pattern
- Tiny fine hairs (trichomes) are visible on the brown skin edge
- The flesh shows subtle striations radiating outward from the center

This type of image is commonly used in food photography, nutritional content, or botanical documentation.\
""")

    usage = result.usage()
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0
    assert usage.cache_write_tokens > 0
    assert usage.cache_read_tokens == 0

    messages = result.all_messages()

    result = await agent.run('How large is the image?', message_history=messages)
    assert result.output == snapshot("""\
The image dimensions are **597 × 597 pixels** (a perfect square).

This is a relatively small to medium-sized image by modern standards, suitable for web use, thumbnails, or social media posts, but not high-resolution enough for large-format printing.\
""")

    usage = result.usage()
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0
    assert usage.cache_write_tokens > 0
    assert usage.cache_read_tokens > 0


@pytest.mark.vcr()
async def test_bedrock_cache_messages_with_video_as_last_content(
    allow_model_requests: None, bedrock_provider: BedrockProvider, video_content: BinaryContent
):
    """Test that cache points can be added after videos without the workaround necessary for documents."""
    model = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)
    agent = Agent(
        model,
        system_prompt='YOU ARE A HELPFUL ASSISTANT THAT ANALYZES VIDEOS.\n' * 50,  # More tokens to activate a cache
        model_settings=BedrockModelSettings(
            bedrock_cache_messages=True,  # This should add a cache point to the last user message
        ),
    )

    # Create a document as the last piece of content in the user message
    run_args = [
        'YOU ARE A HELPFUL ASSISTANT THAT ANALYZES VIDEOS.\n' * 50,  # More tokens to activate a cache
        'Please analyze this video:',
        video_content,
    ]

    result = await agent.run(run_args)
    assert result.output == snapshot(
        'The video depicts a camera mounted on a tripod, capturing a scenic view of a landscape featuring mountains and a road. The camera remains stationary throughout the video, focusing on the picturesque scenery.'
    )

    usage = result.usage()
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0
    assert usage.cache_write_tokens > 0
    assert usage.cache_read_tokens == 0


async def test_bedrock_cache_point_as_first_content_raises_error(
    allow_model_requests: None, bedrock_provider: BedrockProvider
):
    """CachePoint should raise a UserError if it appears before any other content."""
    model = BedrockConverseModel('anthropic.claude-3-7-sonnet-20250219-v1:0', provider=bedrock_provider)
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=[CachePoint(), 'This should fail'])])]
    with pytest.raises(UserError, match='CachePoint cannot be the first content in a user message'):
        await model._map_messages(messages, ModelRequestParameters(), BedrockModelSettings())  # pyright: ignore[reportPrivateUsage]


async def test_bedrock_cache_point_with_only_document_raises_error(
    allow_model_requests: None, bedrock_provider: BedrockProvider
):
    """CachePoint should raise a UserError if the message contains only a document/video with no text."""
    model = BedrockConverseModel('anthropic.claude-3-7-sonnet-20250219-v1:0', provider=bedrock_provider)
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        BinaryContent(data=b'Document content', media_type='text/plain'),
                        CachePoint(),
                    ]
                )
            ]
        )
    ]
    with pytest.raises(
        UserError, match='CachePoint cannot be placed when the user message contains only a document or video'
    ):
        await model._map_messages(messages, ModelRequestParameters(), BedrockModelSettings())  # pyright: ignore[reportPrivateUsage]


async def test_bedrock_cache_messages_no_duplicate_with_explicit_cache_point(
    allow_model_requests: None, bedrock_provider: BedrockProvider
):
    """bedrock_cache_messages should not add a duplicate cache point when one already exists before multi-modal content."""
    model = BedrockConverseModel('us.anthropic.claude-haiku-4-5-20251001-v1:0', provider=bedrock_provider)
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'Process this document:',
                        CachePoint(),
                        BinaryContent(data=b'Document content', media_type='text/plain'),
                    ]
                )
            ]
        )
    ]
    # With bedrock_cache_messages=True, the explicit CachePoint is moved before the document.
    # The auto-caching logic should not add another cache point (which would be back-to-back).
    _, bedrock_messages = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, ModelRequestParameters(), BedrockModelSettings(bedrock_cache_messages=True)
    )
    assert bedrock_messages[0]['content'] == snapshot(
        [
            {'text': 'Process this document:'},
            {'cachePoint': {'type': 'default'}},
            {
                'document': {
                    'name': 'Document 1',
                    'format': 'txt',
                    'source': {'bytes': b'Document content'},
                }
            },
        ]
    )


async def test_bedrock_cache_messages_no_duplicate_when_text_ends_with_cache_point(
    allow_model_requests: None, bedrock_provider: BedrockProvider
):
    """bedrock_cache_messages should not add a duplicate cache point when text content already ends with one."""
    model = BedrockConverseModel('us.anthropic.claude-haiku-4-5-20251001-v1:0', provider=bedrock_provider)
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'Some text content',
                        CachePoint(),
                    ]
                )
            ]
        )
    ]
    # With bedrock_cache_messages=True, the explicit CachePoint is already at the end.
    # The auto-caching logic should not add another cache point.
    _, bedrock_messages = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, ModelRequestParameters(), BedrockModelSettings(bedrock_cache_messages=True)
    )
    assert bedrock_messages[0]['content'] == snapshot(
        [
            {'text': 'Some text content'},
            {'cachePoint': {'type': 'default'}},
        ]
    )


# Bedrock currently errors if a cache point immediately follows documents/videos, so we insert it before them.
async def test_bedrock_cache_point_before_binary_content(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.anthropic.claude-haiku-4-5-20251001-v1:0', provider=bedrock_provider)
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'Process the attached text file. Return the answer only.',
                        BinaryContent(data=b'What is 2+2? Provide the answer only.', media_type='text/plain'),
                        CachePoint(),
                    ]
                )
            ]
        )
    ]
    _, bedrock_messages = await model._map_messages(messages, ModelRequestParameters(), BedrockModelSettings())  # pyright: ignore[reportPrivateUsage]
    assert bedrock_messages[0]['content'] == snapshot(
        [
            {'text': 'Process the attached text file. Return the answer only.'},
            {'cachePoint': {'type': 'default'}},
            {
                'document': {
                    'name': 'Document 1',
                    'format': 'txt',
                    'source': {'bytes': b'What is 2+2? Provide the answer only.'},
                }
            },
        ]
    )


async def test_bedrock_cache_point_with_multiple_trailing_documents(
    allow_model_requests: None, bedrock_provider: BedrockProvider
):
    """CachePoint should be placed before the entire trailing group of documents/videos, not just the last one."""
    model = BedrockConverseModel('us.anthropic.claude-haiku-4-5-20251001-v1:0', provider=bedrock_provider)
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'Process these documents.',
                        BinaryContent(data=b'Document 1 content', media_type='text/plain'),
                        BinaryContent(data=b'Document 2 content', media_type='text/plain'),
                        CachePoint(),
                    ]
                )
            ]
        )
    ]
    _, bedrock_messages = await model._map_messages(messages, ModelRequestParameters(), BedrockModelSettings())  # pyright: ignore[reportPrivateUsage]
    # CachePoint should be inserted BEFORE both documents, not between them
    assert bedrock_messages[0]['content'] == snapshot(
        [
            {'text': 'Process these documents.'},
            {'cachePoint': {'type': 'default'}},
            {
                'document': {
                    'name': 'Document 1',
                    'format': 'txt',
                    'source': {'bytes': b'Document 1 content'},
                }
            },
            {
                'document': {
                    'name': 'Document 2',
                    'format': 'txt',
                    'source': {'bytes': b'Document 2 content'},
                }
            },
        ]
    )


async def test_bedrock_cache_point_with_mixed_content_and_trailing_documents(
    allow_model_requests: None, bedrock_provider: BedrockProvider
):
    """CachePoint should only move before the trailing contiguous group, not all documents."""
    model = BedrockConverseModel('us.anthropic.claude-haiku-4-5-20251001-v1:0', provider=bedrock_provider)
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'First instruction.',
                        BinaryContent(data=b'Doc 1', media_type='text/plain'),
                        # Image breaks the trailing document group (images don't have the cache restriction)
                        BinaryContent(data=b'\x89PNG\r\n\x1a\n', media_type='image/png'),
                        BinaryContent(data=b'Doc 2', media_type='text/plain'),
                        BinaryContent(data=b'Doc 3', media_type='text/plain'),
                        CachePoint(),
                    ]
                )
            ]
        )
    ]
    _, bedrock_messages = await model._map_messages(messages, ModelRequestParameters(), BedrockModelSettings())  # pyright: ignore[reportPrivateUsage]
    # CachePoint should be inserted after the image (non-document/video) and before Doc 2 and Doc 3
    assert bedrock_messages[0]['content'] == snapshot(
        [
            {'text': 'First instruction.'},
            {
                'document': {
                    'name': 'Document 1',
                    'format': 'txt',
                    'source': {'bytes': b'Doc 1'},
                }
            },
            {
                'image': {
                    'format': 'png',
                    'source': {'bytes': b'\x89PNG\r\n\x1a\n'},
                }
            },
            {'cachePoint': {'type': 'default'}},
            {
                'document': {
                    'name': 'Document 2',
                    'format': 'txt',
                    'source': {'bytes': b'Doc 2'},
                }
            },
            {
                'document': {
                    'name': 'Document 3',
                    'format': 'txt',
                    'source': {'bytes': b'Doc 3'},
                }
            },
        ]
    )


async def test_bedrock_cache_messages_with_multiple_trailing_documents(
    allow_model_requests: None, bedrock_provider: BedrockProvider
):
    """bedrock_cache_messages should place cache point before the entire trailing document group."""
    model = BedrockConverseModel('us.anthropic.claude-haiku-4-5-20251001-v1:0', provider=bedrock_provider)
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'Analyze these files.',
                        BinaryContent(data=b'File 1', media_type='text/plain'),
                        BinaryContent(data=b'File 2', media_type='text/plain'),
                    ]
                )
            ]
        )
    ]
    _, bedrock_messages = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, ModelRequestParameters(), BedrockModelSettings(bedrock_cache_messages=True)
    )
    # Cache point should be before both documents
    assert bedrock_messages[0]['content'] == snapshot(
        [
            {'text': 'Analyze these files.'},
            {'cachePoint': {'type': 'default'}},
            {
                'document': {
                    'name': 'Document 1',
                    'format': 'txt',
                    'source': {'bytes': b'File 1'},
                }
            },
            {
                'document': {
                    'name': 'Document 2',
                    'format': 'txt',
                    'source': {'bytes': b'File 2'},
                }
            },
        ]
    )


async def test_bedrock_cache_point_multiple_markers_with_documents_no_back_to_back(
    allow_model_requests: None, bedrock_provider: BedrockProvider
):
    """Multiple CachePoints with trailing documents should not create back-to-back cache points.

    When processing ['text', doc1, CachePoint(), doc2, CachePoint()], both documents form
    a single trailing group. The first CachePoint is placed before the group, and the second
    CachePoint is skipped to avoid back-to-back cachePoints.
    """
    model = BedrockConverseModel('us.anthropic.claude-haiku-4-5-20251001-v1:0', provider=bedrock_provider)
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'Analyze these:',
                        BinaryContent(data=b'Doc 1', media_type='text/plain'),
                        CachePoint(),
                        BinaryContent(data=b'Doc 2', media_type='text/plain'),
                        CachePoint(),
                    ]
                )
            ]
        )
    ]
    _, bedrock_messages = await model._map_messages(messages, ModelRequestParameters(), BedrockModelSettings())  # pyright: ignore[reportPrivateUsage]
    # Both docs are trailing, so first CachePoint goes before both.
    # Second CachePoint is skipped to avoid back-to-back cachePoints.
    assert bedrock_messages[0]['content'] == snapshot(
        [
            {'text': 'Analyze these:'},
            {'cachePoint': {'type': 'default'}},
            {'document': {'name': 'Document 1', 'format': 'txt', 'source': {'bytes': b'Doc 1'}}},
            {'document': {'name': 'Document 2', 'format': 'txt', 'source': {'bytes': b'Doc 2'}}},
        ]
    )


async def test_bedrock_cache_point_multiple_markers(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.anthropic.claude-3-5-haiku-20241022-v1:0', provider=bedrock_provider)
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[UserPromptPart(content=['First chunk', CachePoint(), 'Second chunk', CachePoint(), 'Question'])]
        )
    ]
    _, bedrock_messages = await model._map_messages(messages, ModelRequestParameters(), BedrockModelSettings())  # pyright: ignore[reportPrivateUsage]
    assert bedrock_messages[0]['content'] == snapshot(
        [
            {'text': 'First chunk'},
            {'cachePoint': {'type': 'default'}},
            {'text': 'Second chunk'},
            {'cachePoint': {'type': 'default'}},
            {'text': 'Question'},
        ]
    )


async def test_bedrock_cache_skipped_for_unsupported_models(
    allow_model_requests: None, bedrock_provider: BedrockProvider
):
    """All cache settings should be silently skipped for models that don't support prompt caching."""
    # Meta models don't support prompt caching
    model = BedrockConverseModel('meta.llama3-70b-instruct-v1:0', provider=bedrock_provider)

    # Test CachePoint markers are skipped
    messages_with_cache_points: list[ModelMessage] = [
        ModelRequest(
            parts=[UserPromptPart(content=['First chunk', CachePoint(), 'Second chunk', CachePoint(), 'Question'])]
        )
    ]
    _, bedrock_messages = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages_with_cache_points, ModelRequestParameters(), BedrockModelSettings()
    )
    assert bedrock_messages[0]['content'] == snapshot(
        [{'text': 'First chunk'}, {'text': 'Second chunk'}, {'text': 'Question'}]
    )

    # Test bedrock_cache_instructions is skipped
    messages_with_system: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart(content='System instructions.'), UserPromptPart(content='Hi!')])
    ]
    system_prompt, _ = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages_with_system, ModelRequestParameters(), BedrockModelSettings(bedrock_cache_instructions=True)
    )
    assert system_prompt == snapshot([{'text': 'System instructions.'}])

    # Test bedrock_cache_messages is skipped
    messages_user: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='User message.')])]
    _, bedrock_messages = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages_user, ModelRequestParameters(), BedrockModelSettings(bedrock_cache_messages=True)
    )
    assert bedrock_messages[0]['content'] == snapshot([{'text': 'User message.'}])


async def test_bedrock_cache_tool_definitions_skipped_for_nova(
    allow_model_requests: None, bedrock_provider: BedrockProvider
):
    """Tool caching should be skipped for Nova models (they only support system/messages caching, not tools)."""
    model = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)
    params = ModelRequestParameters(
        function_tools=[
            ToolDefinition(name='tool_one'),
            ToolDefinition(name='tool_two'),
        ]
    )
    params = model.customize_request_parameters(params)
    tool_config = model._map_tool_config(  # pyright: ignore[reportPrivateUsage]
        params,
        BedrockModelSettings(bedrock_cache_tool_definitions=True),
    )
    # Nova doesn't support tool caching, so no cachePoint should be added
    assert tool_config and len(tool_config['tools']) == 2
    assert all('cachePoint' not in tool for tool in tool_config['tools'])


async def test_bedrock_cache_tool_definitions(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('anthropic.claude-sonnet-4-20250514-v1:0', provider=bedrock_provider)
    params = ModelRequestParameters(
        function_tools=[
            ToolDefinition(name='tool_one'),
            ToolDefinition(name='tool_two'),
        ]
    )
    params = model.customize_request_parameters(params)
    tool_config = model._map_tool_config(  # pyright: ignore[reportPrivateUsage]
        params,
        BedrockModelSettings(bedrock_cache_tool_definitions=True),
    )
    assert tool_config and len(tool_config['tools']) == 3
    assert tool_config['tools'][-1] == {'cachePoint': {'type': 'default'}}


async def test_bedrock_cache_instructions(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.anthropic.claude-3-5-sonnet-20240620-v1:0', provider=bedrock_provider)
    messages: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart(content='System instructions to cache.'), UserPromptPart(content='Hi!')])
    ]
    system_prompt, _ = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages,
        ModelRequestParameters(),
        BedrockModelSettings(bedrock_cache_instructions=True),
    )
    assert system_prompt == snapshot(
        [
            {'text': 'System instructions to cache.'},
            {'cachePoint': {'type': 'default'}},
        ]
    )


async def test_bedrock_cache_messages(allow_model_requests: None, bedrock_provider: BedrockProvider):
    """Test that bedrock_cache_messages adds cache point to the last user message."""
    model = BedrockConverseModel('us.anthropic.claude-3-5-sonnet-20240620-v1:0', provider=bedrock_provider)
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='User message to cache.')])]
    _, bedrock_messages = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages,
        ModelRequestParameters(),
        BedrockModelSettings(bedrock_cache_messages=True),
    )
    assert bedrock_messages == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {'text': 'User message to cache.'},
                    {'cachePoint': {'type': 'default'}},
                ],
            }
        ]
    )


async def test_bedrock_cache_messages_with_binary_content(
    allow_model_requests: None, bedrock_provider: BedrockProvider
):
    """Test that bedrock_cache_messages does add cache point for document content."""
    model = BedrockConverseModel('us.anthropic.claude-haiku-4-5-20251001-v1:0', provider=bedrock_provider)
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        BinaryContent(data=b'Test document content', media_type='text/plain'),
                    ]
                )
            ]
        )
    ]
    _, bedrock_messages = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages,
        ModelRequestParameters(),
        BedrockModelSettings(bedrock_cache_messages=True),
    )
    # Should not add cache point for document content
    assert bedrock_messages[0]['content'] == snapshot(
        [
            {
                'document': {
                    'name': 'Document 1',
                    'format': 'txt',
                    'source': {'bytes': b'Test document content'},
                }
            }
        ]
    )


async def test_bedrock_cache_messages_with_tool_result(allow_model_requests: None, bedrock_provider: BedrockProvider):
    """Test that bedrock_cache_messages does add cache point for tool call content."""
    model = BedrockConverseModel('us.anthropic.claude-haiku-4-5-20251001-v1:0', provider=bedrock_provider)
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='final_result',
                    content='Final result processed.',
                    tool_call_id='tooluse_DaRsVjwcShCI_3pOsIsWqg',
                    timestamp=IsDatetime(),
                )
            ],
        )
    ]
    _, bedrock_messages = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages,
        ModelRequestParameters(),
        BedrockModelSettings(bedrock_cache_messages=True),
    )
    # Should add cache point for tool call content
    assert bedrock_messages[0]['content'] == snapshot(
        [
            {
                'toolResult': {
                    'toolUseId': 'tooluse_DaRsVjwcShCI_3pOsIsWqg',
                    'content': [{'text': 'Final result processed.'}],
                    'status': 'success',
                }
            },
            {'cachePoint': {'type': 'default'}},
        ]
    )


async def test_bedrock_cache_messages_does_not_duplicate(allow_model_requests: None, bedrock_provider: BedrockProvider):
    """Test that bedrock_cache_messages does not add duplicate cache point if already present."""
    model = BedrockConverseModel('us.anthropic.claude-3-5-sonnet-20240620-v1:0', provider=bedrock_provider)
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=['User message', CachePoint()])])]
    _, bedrock_messages = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages,
        ModelRequestParameters(),
        BedrockModelSettings(bedrock_cache_messages=True),
    )
    # Should not add another cache point since one already exists
    cache_point_count = sum(1 for block in bedrock_messages[0]['content'] if 'cachePoint' in block)
    assert cache_point_count == 1


async def test_bedrock_cache_messages_no_user_messages(allow_model_requests: None, bedrock_provider: BedrockProvider):
    """Test that bedrock_cache_messages handles case with no user messages."""
    model = BedrockConverseModel('us.anthropic.claude-3-5-sonnet-20240620-v1:0', provider=bedrock_provider)
    # Only assistant message, no user message
    messages: list[ModelMessage] = [ModelResponse(parts=[TextPart(content='Assistant response')])]
    _, bedrock_messages = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages,
        ModelRequestParameters(),
        BedrockModelSettings(bedrock_cache_messages=True),
    )
    # Should not crash, no cache point added since no user message
    assert len(bedrock_messages) == 1
    assert bedrock_messages[0]['role'] == 'assistant'


async def test_get_last_user_message_content_non_dict_block(
    allow_model_requests: None, bedrock_provider: BedrockProvider
):
    """Test _get_last_user_message_content returns None when last block is not a dict."""
    model = BedrockConverseModel('us.anthropic.claude-3-5-sonnet-20240620-v1:0', provider=bedrock_provider)
    # Directly test the helper with a message that has non-dict content
    messages: list[MessageUnionTypeDef] = [{'role': 'user', 'content': ['string content']}]  # type: ignore[list-item]
    result = model._get_last_user_message_content(messages)  # pyright: ignore[reportPrivateUsage]
    assert result is None


async def test_get_last_user_message_content_empty_content(
    allow_model_requests: None, bedrock_provider: BedrockProvider
):
    """Test _get_last_user_message_content returns None when content is empty or not a list."""
    model = BedrockConverseModel('us.anthropic.claude-3-5-sonnet-20240620-v1:0', provider=bedrock_provider)
    # Test with empty content list
    messages: list[MessageUnionTypeDef] = [{'role': 'user', 'content': []}]
    result = model._get_last_user_message_content(messages)  # pyright: ignore[reportPrivateUsage]
    assert result is None


def test_limit_cache_points_filters_excess_cache_points(bedrock_provider: BedrockProvider):
    """Test that _limit_cache_points filters out excess cache points beyond the limit of 4."""
    model = BedrockConverseModel('us.anthropic.claude-3-5-sonnet-20240620-v1:0', provider=bedrock_provider)

    # Create system prompt (no cache points)
    system_prompt: list[SystemContentBlockTypeDef] = [{'text': 'System prompt'}]

    # Create messages with 5 standalone cachePoint blocks (limit is 4)
    bedrock_messages: list[MessageUnionTypeDef] = [
        {
            'role': 'user',
            'content': [
                {'text': 'Context 1'},
                {'cachePoint': {'type': 'default'}},  # Will be filtered (oldest, over limit)
                {'text': 'Context 2'},
                {'cachePoint': {'type': 'default'}},  # Will be kept (4th newest)
                {'text': 'Context 3'},
                {'cachePoint': {'type': 'default'}},  # Will be kept (3rd newest)
                {'text': 'Context 4'},
                {'cachePoint': {'type': 'default'}},  # Will be kept (2nd newest)
                {'text': 'Question'},
                {'cachePoint': {'type': 'default'}},  # Will be kept (newest)
            ],
        },
    ]

    # Apply limit with no tools (max 4 cache points, we have 5)
    model._limit_cache_points(system_prompt, bedrock_messages, [])  # pyright: ignore[reportPrivateUsage]

    # Verify only 4 cache points remain (the newest ones)
    content = bedrock_messages[0]['content']
    assert isinstance(content, list)

    # Count remaining cache points
    cache_points = [b for b in content if isinstance(b, dict) and 'cachePoint' in b]
    assert len(cache_points) == 4  # Only 4 kept (the limit)

    # Verify no empty blocks exist
    empty_blocks = [b for b in content if isinstance(b, dict) and not b]
    assert len(empty_blocks) == 0


async def test_limit_cache_points_with_cache_messages(allow_model_requests: None, bedrock_provider: BedrockProvider):
    """Test that cache points are limited when using bedrock_cache_messages + CachePoint markers."""
    model = BedrockConverseModel('us.anthropic.claude-3-5-sonnet-20240620-v1:0', provider=bedrock_provider)
    # Create messages with 4 CachePoint markers + 1 from bedrock_cache_messages = 5 total
    # Only 4 should be kept (limit)
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'Context 1',
                        CachePoint(),  # Oldest, should be removed
                        'Context 2',
                        CachePoint(),  # Should be kept
                        'Context 3',
                        CachePoint(),  # Should be kept
                        'Context 4',
                        CachePoint(),  # Should be kept
                        'Question',
                    ]
                )
            ]
        )
    ]
    system_prompt, bedrock_messages = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages,
        ModelRequestParameters(),
        BedrockModelSettings(bedrock_cache_messages=True),
    )
    # Apply limit (this is normally called in _messages_create)
    model._limit_cache_points(system_prompt, bedrock_messages, [])  # pyright: ignore[reportPrivateUsage]

    # Count cache points in messages
    cache_count = 0
    for msg in bedrock_messages:
        for block in msg['content']:
            if 'cachePoint' in block:
                cache_count += 1

    # Should have exactly 4 cache points (the limit)
    assert cache_count == 4


async def test_limit_cache_points_all_settings(allow_model_requests: None, bedrock_provider: BedrockProvider):
    """Test cache point limiting with all cache settings enabled."""
    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-20250514-v1:0', provider=bedrock_provider)

    # Create messages with 3 CachePoint markers
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='System instructions.'),
                UserPromptPart(
                    content=[
                        'Context 1',
                        CachePoint(),  # Oldest, should be removed
                        'Context 2',
                        CachePoint(),  # Should be kept
                        'Context 3',
                        CachePoint(),  # Should be kept
                        'Question',
                    ]
                ),
            ]
        )
    ]

    # Map messages with cache_instructions enabled (uses 1 cache point)
    system_prompt, bedrock_messages = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages,
        ModelRequestParameters(),
        BedrockModelSettings(bedrock_cache_instructions=True),
    )

    # Create tools with cache point (uses 1 cache point)
    tools: list[ToolTypeDef] = [
        {'toolSpec': {'name': 'tool_one', 'inputSchema': {'json': {}}}},
        {'cachePoint': {'type': 'default'}},
    ]

    # Apply limit: 1 (system) + 1 (tools) = 2 used, 2 remaining for messages
    model._limit_cache_points(system_prompt, bedrock_messages, tools)  # pyright: ignore[reportPrivateUsage]

    # Count cache points in messages only
    cache_count = 0
    for msg in bedrock_messages:
        for block in msg['content']:
            if 'cachePoint' in block:
                cache_count += 1

    # Should have exactly 2 cache points in messages (4 total - 1 system - 1 tool = 2)
    assert cache_count == 2


async def test_bedrock_empty_model_response_skipped(bedrock_provider: BedrockProvider):
    """Test that ModelResponse with empty parts (e.g. content_filtered) is skipped in message mapping."""
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)

    # Create a message history that includes a ModelResponse with empty parts
    req = [
        ModelRequest(parts=[UserPromptPart(content='Hello')]),
        ModelResponse(
            parts=[],
            usage=RequestUsage(input_tokens=100, output_tokens=1),
            model_name='us.amazon.nova-micro-v1:0',
            provider_name='bedrock',
            provider_details={'finish_reason': 'content_filtered'},
            finish_reason='content_filter',
        ),
        ModelRequest(parts=[UserPromptPart(content='Follow up question')]),
    ]

    # Call the mapping function directly
    _, bedrock_messages = await model._map_messages(req, ModelRequestParameters(), BedrockModelSettings())  # type: ignore[reportPrivateUsage]

    # The empty ModelResponse should be skipped, so we should only have 2 user messages
    # that get merged into one since they're consecutive after the empty response is skipped
    assert bedrock_messages == snapshot(
        [
            {'role': 'user', 'content': [{'text': 'Hello'}, {'text': 'Follow up question'}]},
        ]
    )


async def test_bedrock_map_messages_builtin_tool_provider_filtering(
    allow_model_requests: None, bedrock_provider: BedrockProvider
):
    model = BedrockConverseModel('us.amazon.nova-2-lite-v1:0', provider=bedrock_provider)

    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[
                # BuiltinToolCallPart (w/dict) for bedrock (should be included)
                BuiltinToolCallPart(
                    provider_name='bedrock',
                    tool_name=CodeExecutionTool.kind,
                    args={'snippet': 'print("hello")'},
                    tool_call_id='call_1',
                ),
                # BuiltinToolReturnPart for bedrock with empty provider_details (should be included)
                BuiltinToolReturnPart(
                    provider_name='bedrock',
                    tool_name=CodeExecutionTool.kind,
                    content={'stdOut': 'hello', 'stdErr': '', 'exitCode': 0, 'isError': False},
                    tool_call_id='call_1',
                    provider_details={},
                ),
                # BuiltinToolCallPart for the other provider (should NOT be included)
                BuiltinToolCallPart(
                    provider_name='anthropic',
                    tool_name=CodeExecutionTool.kind,
                    args={'code': 'print("other")'},
                    tool_call_id='call_2',
                ),
                # BuiltinToolReturnPart for the other provider (should NOT be included)
                BuiltinToolReturnPart(
                    provider_name='anthropic',
                    tool_name=CodeExecutionTool.kind,
                    content={'stdOut': 'other', 'stdErr': '', 'exitCode': 0, 'isError': False},
                    tool_call_id='call_2',
                ),
                # BuiltinToolCallPart (w/str) for bedrock (should be included)
                BuiltinToolCallPart(
                    provider_name='bedrock',
                    tool_name=CodeExecutionTool.kind,
                    args='{"snippet": "10*5"}',
                    tool_call_id='call_3',
                ),
                # BuiltinToolReturnPart for the bedrock provider with status (should be included)
                BuiltinToolReturnPart(
                    provider_name='bedrock',
                    tool_name=CodeExecutionTool.kind,
                    content={'stdOut': '50', 'stdErr': '', 'exitCode': 0, 'isError': False},
                    tool_call_id='call_3',
                    provider_details={'status': 'success'},
                ),
                # BuiltinToolCallPart for the bedrock provider but unmapped tool (should NOT be included)
                BuiltinToolCallPart(
                    provider_name='bedrock',
                    tool_name='foo',
                    args={'snippet': 'print("unknown")'},
                    tool_call_id='call_4',
                ),
                # BuiltinToolReturnPart for the bedrock provider but unmapped tool (should NOT be included)
                BuiltinToolReturnPart(
                    provider_name='bedrock',
                    tool_name='foo',
                    content={'other': 'content'},
                    tool_call_id='call_4',
                    provider_details={'status': 'success'},
                ),
            ]
        )
    ]

    _, bedrock_messages = await model._map_messages(  # type: ignore[reportPrivateUsage]
        messages,
        ModelRequestParameters(
            function_tools=[],
            allow_text_output=True,
            builtin_tools=[CodeExecutionTool()],
        ),
        None,
    )
    assert bedrock_messages == snapshot(
        [
            {
                'role': 'assistant',
                'content': [
                    {
                        'toolUse': {
                            'toolUseId': 'call_1',
                            'name': 'nova_code_interpreter',
                            'input': {'snippet': 'print("hello")'},
                            'type': 'server_tool_use',
                        }
                    },
                    {
                        'toolResult': {
                            'toolUseId': 'call_1',
                            'content': [{'json': {'stdOut': 'hello', 'stdErr': '', 'exitCode': 0, 'isError': False}}],
                            'type': 'nova_code_interpreter_result',
                        }
                    },
                    {
                        'toolUse': {
                            'toolUseId': 'call_3',
                            'name': 'nova_code_interpreter',
                            'input': {'snippet': '10*5'},
                            'type': 'server_tool_use',
                        }
                    },
                    {
                        'toolResult': {
                            'toolUseId': 'call_3',
                            'content': [{'json': {'stdOut': '50', 'stdErr': '', 'exitCode': 0, 'isError': False}}],
                            'type': 'nova_code_interpreter_result',
                            'status': 'success',
                        }
                    },
                ],
            }
        ]
    )


async def test_bedrock_model_with_code_execution_tool(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-2-lite-v1:0', provider=bedrock_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', builtin_tools=[CodeExecutionTool()])

    class Response(TypedDict):
        result: float

    # First turn
    result1 = await agent.run('What is 1234 * 5678?', output_type=Response)
    assert result1.output == snapshot({'result': 7006652.0})
    assert result1.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='You are a helpful chatbot.', timestamp=IsDatetime()),
                    UserPromptPart(content='What is 1234 * 5678?', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={'snippet': '1234 * 5678'},
                        tool_call_id='tooluse_dV5ehBNfl1hUE-UTM9cIww',
                        provider_name='bedrock',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'stdOut': '7006652', 'stdErr': '', 'exitCode': 0, 'isError': False},
                        tool_call_id='tooluse_dV5ehBNfl1hUE-UTM9cIww',
                        timestamp=IsDatetime(),
                        provider_name='bedrock',
                        provider_details={'status': 'success'},
                    ),
                    ToolCallPart(
                        tool_name='final_result',
                        args={'result': 7006652.0},
                        tool_call_id='tooluse_DaRsVjwcShCI_3pOsIsWqg',
                    ),
                ],
                usage=RequestUsage(input_tokens=1002, output_tokens=59),
                model_name='us.amazon.nova-2-lite-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                provider_details={'finish_reason': 'tool_use'},
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='tooluse_DaRsVjwcShCI_3pOsIsWqg',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )

    # Second turn
    result2 = await agent.run('Now multiply that by 2', message_history=result1.new_messages(), output_type=Response)
    assert result2.output == snapshot({'result': 14013304.0})
    assert result2.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Now multiply that by 2', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={'snippet': '7006652 * 2'},
                        tool_call_id='tooluse_VYEuMWAFChlHdy6-56IQ4g',
                        provider_name='bedrock',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'stdOut': '14013304', 'stdErr': '', 'exitCode': 0, 'isError': False},
                        tool_call_id='tooluse_VYEuMWAFChlHdy6-56IQ4g',
                        timestamp=IsDatetime(),
                        provider_name='bedrock',
                        provider_details={'status': 'success'},
                    ),
                    ToolCallPart(
                        tool_name='final_result',
                        args={'result': 14013304.0},
                        tool_call_id='tooluse_RyG7SphVTsuS_8GFmX9hIA',
                    ),
                ],
                usage=RequestUsage(input_tokens=1148, output_tokens=59),
                model_name='us.amazon.nova-2-lite-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                provider_details={'finish_reason': 'tool_use'},
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='tooluse_RyG7SphVTsuS_8GFmX9hIA',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )


async def test_bedrock_model_code_execution_tool_stream(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-2-lite-v1:0', provider=bedrock_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', builtin_tools=[CodeExecutionTool()])

    class Response(TypedDict):
        result: float

    event_parts: list[Any] = []
    async with agent.iter('What is 1234 * 5678?', output_type=Response) as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    assert agent_run.result.output == snapshot({'result': 7006652.0})
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='You are a helpful chatbot.', timestamp=IsDatetime()),
                    UserPromptPart(content='What is 1234 * 5678?', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args='{"snippet":"1234 * 5678"}',
                        tool_call_id='tooluse_VQNZJRUFMoqZzszVsRd4og',
                        provider_name='bedrock',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'stdOut': '7006652', 'stdErr': '', 'exitCode': 0, 'isError': False},
                        tool_call_id='tooluse_VQNZJRUFMoqZzszVsRd4og',
                        timestamp=IsDatetime(),
                        provider_name='bedrock',
                        provider_details={'status': 'success'},
                    ),
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"result":7006652.0}',
                        tool_call_id='tooluse_ptgCcZ0uQu-UUMz0abqoWw',
                    ),
                ],
                usage=RequestUsage(input_tokens=1002, output_tokens=59),
                model_name='us.amazon.nova-2-lite-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                provider_details={'finish_reason': 'tool_use'},
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='tooluse_ptgCcZ0uQu-UUMz0abqoWw',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )
    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=BuiltinToolCallPart(
                    tool_name='code_execution', tool_call_id='tooluse_VQNZJRUFMoqZzszVsRd4og', provider_name='bedrock'
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ToolCallPartDelta(
                    args_delta='{"snippet":"1234 * 5678"}', tool_call_id='tooluse_VQNZJRUFMoqZzszVsRd4og'
                ),
            ),
            PartEndEvent(
                index=0,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args='{"snippet":"1234 * 5678"}',
                    tool_call_id='tooluse_VQNZJRUFMoqZzszVsRd4og',
                    provider_name='bedrock',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'stdOut': '7006652', 'stdErr': '', 'exitCode': 0, 'isError': False},
                    tool_call_id='tooluse_VQNZJRUFMoqZzszVsRd4og',
                    timestamp=IsDatetime(),
                    provider_name='bedrock',
                    provider_details={'status': 'success'},
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=2,
                part=ToolCallPart(tool_name='final_result', tool_call_id='tooluse_ptgCcZ0uQu-UUMz0abqoWw'),
                previous_part_kind='builtin-tool-return',
            ),
            FinalResultEvent(tool_name='final_result', tool_call_id='tooluse_ptgCcZ0uQu-UUMz0abqoWw'),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='{"result":7006652.0}', tool_call_id='tooluse_ptgCcZ0uQu-UUMz0abqoWw'
                ),
            ),
            PartEndEvent(
                index=2,
                part=ToolCallPart(
                    tool_name='final_result', args='{"result":7006652.0}', tool_call_id='tooluse_ptgCcZ0uQu-UUMz0abqoWw'
                ),
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args='{"snippet":"1234 * 5678"}',
                    tool_call_id='tooluse_VQNZJRUFMoqZzszVsRd4og',
                    provider_name='bedrock',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'stdOut': '7006652', 'stdErr': '', 'exitCode': 0, 'isError': False},
                    tool_call_id='tooluse_VQNZJRUFMoqZzszVsRd4og',
                    timestamp=IsDatetime(),
                    provider_name='bedrock',
                    provider_details={'status': 'success'},
                )
            ),
        ]
    )
