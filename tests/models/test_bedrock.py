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
        'amazon.nova-micro-v1:0',
        provider=_StubBedrockProvider(_StubBedrockClient(error)),
    )


async def test_bedrock_model(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('amazon.nova-micro-v1:0', provider=bedrock_provider)
    assert model.base_url == 'https://bedrock-runtime.us-east-1.amazonaws.com'
    agent = Agent(model=model, system_prompt='You are a chatbot.')

    result = await agent.run('Hello!')
    assert result.output == snapshot(
        "Hello there! How can I assist you today? Whether you have questions, need information, or just want to chat, I'm here to help. What's on your mind?"
    )
    assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=7, output_tokens=39))
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
                        content="Hello there! How can I assist you today? Whether you have questions, need information, or just want to chat, I'm here to help. What's on your mind?"
                    )
                ],
                usage=RequestUsage(input_tokens=7, output_tokens=39),
                model_name='amazon.nova-micro-v1:0',
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
        """\
I notice there's a small typo in your message - it should be "lazy dog" (two words) rather than "lazydog" (one word).

The corrected pangram is: "The quick brown fox jumps over the lazy dog."

This is a famous sentence used to test fonts and keyboards because it contains every letter of the English alphabet at least once. Is there something specific you'd like to know about this sentence or were you testing something?\
"""
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
    model = BedrockConverseModel('amazon.nova-micro-v1:0', provider=bedrock_provider)
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
    assert result.usage() == snapshot(RunUsage(requests=2, input_tokens=1198, output_tokens=74, tool_calls=1))
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
                model_name='amazon.nova-micro-v1:0',
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
                    ToolCallPart(
                        tool_name='final_result',
                        args={'city': 'London', 'date': '2022-01-01', 'temperature': '30°C'},
                        tool_call_id='tooluse_0RUR37nJRKWprDhFgmYE5A',
                    ),
                    TextPart(
                        content="""\


The temperature in London on 1st January 2022 was 30°C.\
"""
                    ),
                ],
                usage=RequestUsage(input_tokens=627, output_tokens=52),
                model_name='amazon.nova-micro-v1:0',
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
    model = BedrockConverseModel('amazon.nova-micro-v1:0', provider=bedrock_provider)
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
    model = BedrockConverseModel('amazon.nova-micro-v1:0', provider=bedrock_provider)
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
                        content='<thinking> To determine the capital of France, I can use the provided tool "get_capital". I will need to call this tool with the country "France" as the argument.</thinking>\n'
                    ),
                    ToolCallPart(
                        tool_name='get_capital',
                        args={'country': 'France'},
                        tool_call_id=IsStr(),
                    ),
                ],
                usage=RequestUsage(input_tokens=426, output_tokens=54),
                model_name='amazon.nova-micro-v1:0',
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
<thinking> It seems there was an error with the tool, possibly because it's not set up to handle "France". Since I can't rely on the tool in this case, I will use my existing knowledge to provide the answer.</thinking> \n\

The capital of France is Paris.\
"""
                    )
                ],
                usage=RequestUsage(input_tokens=521, output_tokens=60),
                model_name='amazon.nova-micro-v1:0',
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
    model = BedrockConverseModel('amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings={'max_tokens': 5})
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is')


async def test_bedrock_model_top_p(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings={'top_p': 0.5})
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        "The capital of France is Paris. Paris is not only the capital city but also the most populous city in France, and it is a major center for culture, commerce, fashion, and international diplomacy. It's well-known for its historical landmarks, such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral, among many other attractions."
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
    model = BedrockConverseModel('amazon.nova-micro-v1:0', provider=bedrock_provider)
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
        'The capital of France is Paris. Paris is not only the political center of the country but also a major global city known for its historical landmarks, cultural significance, and as a hub for fashion, art, and gastronomy.'
    )


async def test_bedrock_model_other_parameters(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('amazon.nova-micro-v1:0', provider=bedrock_provider)
    model_settings = BedrockModelSettings(
        bedrock_request_metadata={'requestId': 'test-request-123'},
    )
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings=model_settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        'The capital of France is Paris. Paris is not only the capital city but also a major cultural, historical, and economic center of the country. It is known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, among many other attractions.'
    )


async def test_bedrock_model_service_tier(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('amazon.nova-pro-v1:0', provider=bedrock_provider)
    model_settings = BedrockModelSettings(bedrock_service_tier={'type': 'flex'})
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings=model_settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        'The capital of France is Paris. Paris is renowned for its significant influence in culture, arts, fashion, and cuisine. It is located in the northern central part of the country and is one of the most visited cities in the world. Paris is home to iconic landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.'
    )


async def test_bedrock_model_iter_stream(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('amazon.nova-micro-v1:0', provider=bedrock_provider)
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
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='>')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' To')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' find')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' temperature')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' capital')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' France')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' first')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' need')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' determine')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' capital')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' France')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' Then')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' can')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' temperature')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' tool')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' get')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' current')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' temperature')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' city')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' </')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='thinking')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='>\n')),
            PartEndEvent(
                index=0,
                part=TextPart(
                    content='<thinking> To find the temperature of the capital of France, I first need to determine the capital of France. Then, I can use the temperature tool to get the current temperature in that city. </thinking>\n'
                ),
                next_part_kind='tool-call',
            ),
            PartStartEvent(
                index=1,
                part=ToolCallPart(tool_name='get_capital', tool_call_id='tooluse_QjQX34XORLu0wJVplWWHGQ'),
                previous_part_kind='text',
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='{"country":"France"}', tool_call_id='tooluse_QjQX34XORLu0wJVplWWHGQ'
                ),
            ),
            PartEndEvent(
                index=1,
                part=ToolCallPart(
                    tool_name='get_capital', args='{"country":"France"}', tool_call_id='tooluse_QjQX34XORLu0wJVplWWHGQ'
                ),
            ),
            FunctionToolCallEvent(
                part=ToolCallPart(
                    tool_name='get_capital', args='{"country":"France"}', tool_call_id='tooluse_QjQX34XORLu0wJVplWWHGQ'
                )
            ),
            FunctionToolResultEvent(
                result=ToolReturnPart(
                    tool_name='get_capital',
                    content='Paris',
                    tool_call_id='tooluse_QjQX34XORLu0wJVplWWHGQ',
                    timestamp=IsDatetime(),
                )
            ),
            PartStartEvent(index=0, part=TextPart(content='<thinking')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='>')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' have')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' found')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' capital')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' France')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' Paris')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' Now')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' will')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' temperature')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' tool')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' get')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' current')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' temperature')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' Paris')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' </')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='thinking')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='>')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' ')),
            PartEndEvent(
                index=0,
                part=TextPart(
                    content='<thinking> I have found that the capital of France is Paris. Now, I will use the temperature tool to get the current temperature in Paris. </thinking> '
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
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' current')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' temperature')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' Paris')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' capital')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' France')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='3')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='0')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='°')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='C')),
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
        "The fruit in the image is a kiwi. It is a small, round fruit with a brown skin and a green flesh. The kiwi has a unique texture and a tangy taste, making it a popular ingredient in various dishes and desserts. The image shows a close-up view of the kiwi's skin, highlighting its distinctive appearance."
    )


@pytest.mark.vcr()
async def test_video_as_binary_content_input(
    allow_model_requests: None, video_content: BinaryContent, bedrock_provider: BedrockProvider
):
    m = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

    result = await agent.run(['Explain me this video', video_content])
    assert result.output == snapshot(
        "The video shows a camera on a tripod, filming a landscape with a rock formation and a dirt road. The camera's screen displays the scene being captured."
    )


@pytest.mark.vcr()
async def test_image_url_input(allow_model_requests: None, bedrock_provider: BedrockProvider):
    m = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

    result = await agent.run(
        [
            'What is this vegetable?',
            ImageUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'),
        ]
    )
    assert result.output == snapshot(
        'The vegetable shown in the image is a potato. It is a starchy tuber that is widely consumed around the world. Potatoes come in various shapes, sizes, and colors, but the one in the image appears to be a common yellow-skinned potato. The potato has a rough, textured surface with small eyes or buds scattered across it.'
    )


@pytest.mark.vcr()
async def test_video_url_input(allow_model_requests: None, bedrock_provider: BedrockProvider):
    m = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

    result = await agent.run(
        [
            'Explain me this video',
            VideoUrl(url='https://www.w3schools.com/html/mov_bbb.mp4'),
        ]
    )
    assert result.output == snapshot(
        'The rabbit is now standing on a field of grass, surrounded by trees and flowers. It is looking up at the sky.'
    )


@pytest.mark.vcr()
async def test_document_url_input(allow_model_requests: None, bedrock_provider: BedrockProvider):
    m = BedrockConverseModel('us.anthropic.claude-sonnet-4-20250514-v1:0', provider=bedrock_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

    document_url = DocumentUrl(url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf')

    result = await agent.run(['What is the main content on this document?', document_url])
    assert result.output == snapshot(
        """\
Based on the document you've shared, this appears to be a very simple PDF file with minimal content. The main (and only) content on this document is:

- A title that reads "Dummy PDF file"

This appears to be a test or placeholder document, as indicated by the word "Dummy" in the title. The rest of the page appears to be blank space. This type of file is commonly used for testing purposes, file format demonstrations, or as a template placeholder.\
"""
    )


@pytest.mark.vcr()
async def test_text_document_url_input(allow_model_requests: None, bedrock_provider: BedrockProvider):
    m = BedrockConverseModel('us.anthropic.claude-sonnet-4-20250514-v1:0', provider=bedrock_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

    text_document_url = DocumentUrl(url='https://example-files.online-convert.com/document/txt/example.txt')

    result = await agent.run(['What is the main content on this document?', text_document_url])
    assert result.output == snapshot(
        """\
Based on the document, the main content is an explanation of placeholder names used in legal and other contexts, specifically focusing on "John Doe" and related names. \n\

The document explains:

1. **Legal placeholder names**: How names like "John Doe" (for males), "Jane Doe/Jane Roe" (for females), and "Jonnie/Janie Doe" (for children) are used when someone's true identity is unknown or must be kept confidential in legal proceedings

2. **Medical/forensic use**: How these names are applied to unidentified corpses or hospital patients

3. **Geographic usage**: The practice is common in the US and Canada, but less common in other English-speaking countries like the UK, Australia, and New Zealand, which use alternatives like "Joe Bloggs" or "John Smith"

4. **Other contexts**: How "John Doe" is used as a typical male reference (similar to "John Q. Public") in forms, examples, and popular culture

5. **Variations and extensions**: Examples of related names like "Baby Doe" for unidentified children, and how multiple anonymous parties are handled in legal cases (e.g., "John Doe #1, #2" etc.)

The document appears to be a sample text file demonstrating the TXT file format, with content sourced from Wikipedia about the John Doe naming convention.\
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
It looks like the document you're referring to is a simple test document. Typically, a test document might contain placeholder text or basic information to demonstrate the format or functionality of a document template. \n\

In this case, the main content appears to be:

1. **Title**: "This is a test document."
2. **Content**: There isn't much additional content beyond the title. It's likely used to show how text appears in a document or to test printing, formatting, or other document-related functions.

If you have a specific document in mind and need more detailed analysis, please provide more context or specific sections from the document!\
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
                        content='The capital of France is Paris. It is one of the largest and most influential cities in Europe, known for its rich history, culture, art, fashion, gastronomy, and landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major center for diplomacy, commerce, and tourism.'
                    )
                ],
                usage=RequestUsage(input_tokens=13, output_tokens=67),
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
    m = BedrockConverseModel('amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(m)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        'The capital of France is Paris. Paris is not only the capital city but also the most populous city in France, with an extensive history that dates back over two millennia. It is located in the northern central part of the country and lies along the Seine River. Paris is renowned for its significant cultural, economic, political, and social influence both in France and globally. It is famous for landmarks such as the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and the Champs-Élysées, among many other historical and modern attractions.'
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
        """\
I apologize for my previous unhelpful responses. Looking at the two PDF documents you've shared:

Both documents appear to be identical, simple placeholder PDF files. Each contains just the text "Dummy PDF file" centered near the top of the page, with the rest of the page being blank. These are likely sample or test PDFs that are used as placeholders when testing document handling systems or demonstrating PDF functionality.\
"""
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
                usage=RequestUsage(input_tokens=12, output_tokens=1039),
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
                usage=RequestUsage(input_tokens=33, output_tokens=1260),
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
                usage=RequestUsage(input_tokens=43, output_tokens=303),
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
                usage=RequestUsage(input_tokens=337, output_tokens=408),
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
                usage=RequestUsage(input_tokens=92, output_tokens=198),
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
                usage=RequestUsage(input_tokens=171, output_tokens=299),
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
                    ThinkingPart(
                        content='',
                        id='redacted_content',
                        signature='EpgCCkgICxABGAIqQA6aZKBEab7hknq1R6KqDCz9DIdx57BAKFclmgabcUfY83XiuqeZIuy9prvQ96LWRg6pHsNjMNy8takaY2E+lg4SDPqsipqBp0AVsnivLhoMljaQ/RrDhNC7Jv7JIjCGDodyNx+Ug8ptjufk/AmKNqQYJjtqwnhyS0oZr9WjGDjyt74OWeZ6IvJzubDqo2Uqfvvdmyc7LB4CngqO4qmqJHzo1yuxNYRm7JSctRHP/Ou0c6J4qMR30VfzbAcvJB1/FPzncOdxS2kqDk97vpV/sZ5lyGritGXABnN+MpVzVKlshq/7mSCM+UC+ejaVHIcgiKoTdwp7NS9XR7fG30/gB/iWawHdnv0pGrxHaNFCCxgB',
                        provider_name='bedrock',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(input_tokens=92, output_tokens=173),
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
                    signature='EtwCCkgICxABGAIqQD0L9iotuc3utpYeikGx/iOavb7iiY/DZdo6hXbNpnCHzQJLR2X4PAIjDXhtyLAQE9Jn5hPA9ZYAvpUfQLEEJAYSDB+i6zhjowsbzO+AmBoMwi5Fv1h1OKFPZHNEIjBDRu7nETnTrDjL6lps7PN4nyUxuGTPU7Nl8UC3jOy8D66RhDEsqauOcKivzpnmHOIqwQGsUamUs+ik94QKQBnndHsoLbUqv587+KgkJIZ5WkpK+D64El58ujWwjtRHKVCezhrMdeWBBkSm0Nmc62MhHyRLYdy3w5sSxOTevTxwqTrZMXzNrg0nHgTNjehwxZK+NVEpSgZFG57JjoL73Yum05SaXOKC1WwkciADyndwOWDTmFPbokjdDXD3hfIuITF1Hblhq6B0GVM0H9T92O1q6QfEoIJL081YZNXKVpOoSbAR6FZ3sGMzo2D0crKFv19IIhfrGAE=',
                    provider_name='bedrock',
                ),
                next_part_kind='thinking',
            ),
            PartStartEvent(
                index=1,
                part=ThinkingPart(
                    content='',
                    id='redacted_content',
                    signature='EtkCCkgICxABGAIqQCpz2dpSmU8cEBNDrr8Oet6bpUi1VttzzDMarPlabYHDjePXYwFBwljkaYNKR58RDdJZ1MZuTiGu9uHOcCsYUooSDLUiPml31uQZESNN6RoMo8TLygFMH0EUS+QNIjDS1Xb/1BWWmvAMRcTLnC2Zpx0EFdqpsXmfL9SDqURkTEUcVEsYkTfwm7lbMaiuNr4qvgGH/xWdjigNNdc60m3nuq0g4U1KKFbeo2/P3ifnJyC3v368cQfU7UzGoWFkrLAnEGT4ItkNzCMaBQjVfaAAjzyODAPwGy443kQlpY0eHTvH4Svb59pA688opcq07jKBh3keizlATo+R3eP+o0Dg+LD+zIeGc8ZmHS5K+Ab9tot7I6EdpVfoa0yDvfbAb0HhhVW8rhFOp3MJttvndLFb6Kl4t15bf3fomZFvf7QF52HlTnsoeIs04F3lMlIUBbYeGAE=',
                    provider_name='bedrock',
                ),
                previous_part_kind='thinking',
            ),
            PartEndEvent(
                index=1,
                part=ThinkingPart(
                    content='',
                    id='redacted_content',
                    signature='EtkCCkgICxABGAIqQCpz2dpSmU8cEBNDrr8Oet6bpUi1VttzzDMarPlabYHDjePXYwFBwljkaYNKR58RDdJZ1MZuTiGu9uHOcCsYUooSDLUiPml31uQZESNN6RoMo8TLygFMH0EUS+QNIjDS1Xb/1BWWmvAMRcTLnC2Zpx0EFdqpsXmfL9SDqURkTEUcVEsYkTfwm7lbMaiuNr4qvgGH/xWdjigNNdc60m3nuq0g4U1KKFbeo2/P3ifnJyC3v368cQfU7UzGoWFkrLAnEGT4ItkNzCMaBQjVfaAAjzyODAPwGy443kQlpY0eHTvH4Svb59pA688opcq07jKBh3keizlATo+R3eP+o0Dg+LD+zIeGc8ZmHS5K+Ab9tot7I6EdpVfoa0yDvfbAb0HhhVW8rhFOp3MJttvndLFb6Kl4t15bf3fomZFvf7QF52HlTnsoeIs04F3lMlIUBbYeGAE=',
                    provider_name='bedrock',
                ),
                next_part_kind='thinking',
            ),
            PartStartEvent(
                index=2,
                part=ThinkingPart(
                    content='',
                    id='redacted_content',
                    signature='EpgCCkgICxABGAIqQA6aZKBEab7hknq1R6KqDCz9DIdx57BAKFclmgabcUfY83XiuqeZIuy9prvQ96LWRg6pHsNjMNy8takaY2E+lg4SDPqsipqBp0AVsnivLhoMljaQ/RrDhNC7Jv7JIjCGDodyNx+Ug8ptjufk/AmKNqQYJjtqwnhyS0oZr9WjGDjyt74OWeZ6IvJzubDqo2Uqfvvdmyc7LB4CngqO4qmqJHzo1yuxNYRm7JSctRHP/Ou0c6J4qMR30VfzbAcvJB1/FPzncOdxS2kqDk97vpV/sZ5lyGritGXABnN+MpVzVKlshq/7mSCM+UC+ejaVHIcgiKoTdwp7NS9XR7fG30/gB/iWawHdnv0pGrxHaNFCCxgB',
                    provider_name='bedrock',
                ),
                previous_part_kind='thinking',
            ),
            PartEndEvent(
                index=2,
                part=ThinkingPart(
                    content='',
                    id='redacted_content',
                    signature='EpgCCkgICxABGAIqQA6aZKBEab7hknq1R6KqDCz9DIdx57BAKFclmgabcUfY83XiuqeZIuy9prvQ96LWRg6pHsNjMNy8takaY2E+lg4SDPqsipqBp0AVsnivLhoMljaQ/RrDhNC7Jv7JIjCGDodyNx+Ug8ptjufk/AmKNqQYJjtqwnhyS0oZr9WjGDjyt74OWeZ6IvJzubDqo2Uqfvvdmyc7LB4CngqO4qmqJHzo1yuxNYRm7JSctRHP/Ou0c6J4qMR30VfzbAcvJB1/FPzncOdxS2kqDk97vpV/sZ5lyGritGXABnN+MpVzVKlshq/7mSCM+UC+ejaVHIcgiKoTdwp7NS9XR7fG30/gB/iWawHdnv0pGrxHaNFCCxgB',
                    provider_name='bedrock',
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=3, part=TextPart(content='I notice your message contains'), previous_part_kind='thinking'
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' what appears to be')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' a specific')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' trigger phrase or')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' comman')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta="d. I'm designe")),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='d to respon')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='d to normal')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' questions')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' and requests rather')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' than to special')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' commands')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' or triggers')),
            PartDeltaEvent(
                index=3,
                delta=TextPartDelta(
                    content_delta="""\
.

If\
"""
                ),
            ),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' you have a genuine')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' question or need assistance')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' with something, I')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta="'")),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='d be happy to help')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' please')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' just')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' ask directly')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' what')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=" you'd like to")),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' know or discuss')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='.')),
            PartEndEvent(
                index=3,
                part=TextPart(
                    content="""\
I notice your message contains what appears to be a specific trigger phrase or command. I'm designed to respond to normal questions and requests rather than to special commands or triggers.

If you have a genuine question or need assistance with something, I'd be happy to help - please just ask directly what you'd like to know or discuss.\
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
                        content="""\
**Providing street crossing advice**

I need to reply to the user's question about crossing the street. Since there's no image, it's important to give general safety advice without being too commanding. I'll keep it high-level and non-judgmental, which means I shouldn't use "must" or "should" language too much. I can include a step-by-step guide but avoid overly professional instructions and not mention calling emergency services, as this isn't a crisis. Just general tips will do!\
""",
                        id='rs_0f522e1dce1a173a0069814689807c819489cb71998380b944',
                        signature='gAAAAABpgUa5C9uo9uZozsur6hDZl3rW_dckS3A-AtNSV5uYScLJva6Q3boUwgMAGtN4vpmLhgJ-0MUUr95llqQ295UJNVuxIpJW0QYfQNgbvaBeZTk3BcAWzq86O9lhDJybr4Owlvbg5vF5mKLaPz5NXxC38Qwfbvk8WtXLm0qiJy88FXYlT_gaOZ_YSx8gsZ0EpkIITBjRpXi-86Ve6HZmpq6GDUFkqvadWSIs2fr4MVc99RDommqD4mAi7HrMFkFafUBaniq3ICOpDt-j519xucfAwoPriRHatSSvpRhnItxnf6j3y1GEoAu7MY8qlx_3OI5tulByTkWZzQEzDF5FcadSE00j0IKrQH0EOKHBTlqS7XU4161Plr-GKwYnQSIGgvJm8C70jUsRCUv364R7k68XqaS8KMK1AY9zVdUoz3eLMVrQ0yO7JweG5KUBNUVWM5k2UDCWeOgR7MYQWZbJKyE5gSPd9ZV6_Py3lb5wxoxkz80QuQU74RVrB6eh0FrFOLxVYM-BrltoUBcWyxPRH-sdbT6_PaC09yWcQQqMsR8f5b6gqkAgQYmoYoRe9mvhzzYHpJcBkE9Xvs9JWzvWvpx-ImaUn4XzeltJtfgQNPLKnh4jxF5CJqBuzfJJomW5MjKkKcEy4TNrz5X46UJFxo8o0rclT8q1UNIkMQc5AyKqN_tb73F9TsiJY_T136_vQhfudt_HORzJn1J55xv_yTX0gMZTemSjJdV4ELxam-L7VgX4qswoQDkCAhBJ8CQ6LqBplC4S0McuDxuzsKJn_0luyQag0N_ojrHkPTGaHom6a-4vvck0VmI9o2rOFanIKIi7B5wbPHOvAyyTTfWzZew8XdAT_bQ8Cd_ggW4gE_6KMbW7dpnIpF-CkbdZkx5UsSHmJtVdwJzslL95y0bjAvEyOWiZKCq4dtBMn1vaM5A5OCCYCW0ePDfEIIqfY28ArzoMReFJ8aWtOuMdHKjWO9lBfAuVhTAT-e6GArSNyjsxqnenhUhJcwdx1iQbVzAd7SgEfLq5ll5yxfC62MLe-sm1hljp5f8J_WfdYjBut6jyESBbbvLWVmMXCZt_u0loQseSInDrbwemYuXEh9egwa4_z15s4-6B8XuqOrY9QqoCpHtuJsl8mlRsua8LGILSCZrCvOwf-HrMYgC1AgCXle4_btTcclPnZJqfrvXB-RCBwb_NVVgyY-zyyABBNesy7aSVNGmgMjOB35QVNl-l2OoRVdZ8TxsMXarwfW73z5eF_mga-FIZjbD383_abvPZqYRFl-hGKbgc6-GNzenELI04OUndMERrnOQd09xyLujoTu5qtIWcgmzqnQq_dcEX-43sPLTk0SHN5V-9aXCaMD7l0-UkxemKez7_fVVEXEaeF1FZ_DV4gTusBbWgC6qOSEgo6A-WZYII-L73_L20ai8eZmjuBWQ1gUSnellhkEJz93FOGdqv-MrRwSBc6bERZdMChN7X78qfPS52IFkqJ-Gy66ZhRzLwAXjhpchxCSDwZK-gb-DZBa5Nzg4gOWRK5_MflPVYIXtEN-yexhVzWndH4TRyK748WKCGvVjYcI-2Hc4MFt9nHG2WKj6ZaL3rkTlw1oH_fJ9fd8Ntyf-kBj0hcFtKlggvxxLd06j_x6ZuYldlMvU1mEYws4yDHljBh1FxQKmD4hSUKgYLGygYkHURqfDZ65t9O6aZke63EGupWjslYUNBYFddbcwIRBz8V96k7bSUUPkYvWrDDrpWSndAEj9TTlc7FSp1iqAvl3O819D0FbrxLqAItuGtJdrbDmuPRP6Dy52cMPxl0GPiv_CZXf-XFKfEHIHh8auCbJh_auCzLBhCWYoYU_sKJDLt07j_g21i-ae9YMuafIsbmgzdrT_eTn-sMscZbDchrMp6IAKBzrQ04-eGZM0P29yPigkoiX8z-7cykstd4wNS-VwiuRa0Fq-QCArELCnSYAZNUD2OomWAQPYYAzu1zrsY6t0Ny9wABb1-YlTm52RW3jwNTr7MhZ-P9Z74k2QEpooy_28iY4RknSVswbPuEVflIYKuKdfSwh7T-CI3LEeT81fz4KAalq7a2zGJIkGk3-lsdIIlg-7ePBW7eGF-sEhr9t5GGkaPXr9O_lGV9SG2XZD4X4p6VuFeIzl7DnL6-wQIZkc89scdGruZDmJa_8z-WNWnnyBpj90ef9U_5Hb7TA4zN6c9EWkyNayXjO2neZT00IggT_uOyRALMlwX9pCslg4z8jXBXPqlkGMUcBK05aAU9kn2gprDOUqRd7y0ukal1IvPAx2RZnZEFAb3dHp5QqnDMfxH5431bN3CzJrsxu2ojlRHDIRCwenBn7nQdTnJ3a6WX9xqWQ3N0_NueIzYuzw9TTyw13K2ANKwReGikhuYL4SiC-tYHYsz2cfmdMFcllPM19HLsX5wY3Lsm9cKNSpKQ957Ex2-Ry_lJD5NJzLGKkYkHqyGmCxia2ULqtoeGbKsyL5J58PvhDKEykhylj4VHvwiCGCnrr0dEyJnYCl8-NGG0KkxMoBD6Ccqa2CsBdEbmMBAo0nREho-h5aNvlyNcHPg1Y6nlddMzCC6sXAD8xZqz2CmFkrP_G6HeyhjmUkVYc79bGclxLcqO570A-DPkRFbS1S5iF2gP9rzuLNUb2Bls6r_7ifVHgPH1zcWugeGLD30UpOEJFqMbj0mvT-nd94pFo0ojxHAIJEdyLsADZQkr9dzg26AjKijeSXdid-_cIRUdnKUk_ejZSKiGYtJCatXBeP2tNTaIYyOyt9kIMvmqKlLUh0mSBkpk91cbcW-TqLBN8HUpctW1Njx-IlTIVSOeRksWU3nFFJmrlaJ7G2TjAUdriLjucpAZSqbrQDPBrVSh_hSEDeefon1yRdyAbzPGpD1UHqMJ80zRHoPcyym6Wr7AtSLDgkt7TsaVfTnZ5BQa9xIQCpMWJoC3IiXLEch_GYx45cyiy974uJYzHrY_Yj_De0kDIEbcykiAL7vcQH5nuh-rlNSCfD2cGdbRemx9MYYoGn6XJSWvRsSvikUfBxhRQVyJ0r6xHWP5U4r2KTEXlTyBgJiYJijfV9R_rRfb-CWhDnNHfGAeKF7kylYH20AVYmuKUMKbGVUrUAvwhekWOqBe3eFHX-2KNWvL6krxQBkhyDJBjiBwt69xRICGjGxQlWd31mU3541pdfpKBLySCb-7Uv7nOaNiuKXBn5POeYk0h2uH3pjm6wOiOzTBlshbM_XydtHXwZl3_Kdh1e6RgdLMHvnOvf6yAImdPq3qkAUj_uvvezWq7o2ALhoBQN1NJdUTBlVrUuvGlTZkopBQ3LHtQIurdKGGebUs8uk0YMMX_gsnO4kwtnvuTxRvIXmJdLEq_dSzZAsT5HIxaIN1WwIjmrRh-PgaqqtI3PqsE8QUFQ3zQvuidbBGkQb_Y5IKbBxIOvZSd8i0TYinR-udl5JqvOaFPqp5kfPzAEM4Rdy8qCPpoDOummZAZgDJMlKhylrAZEIJD1nw8SFM_YYwwJWcJn4S9u3yqsXYcETrEVv9SsChRo1FnwzkBtrfXly-FU8FtqWopNyReIh6wgjjtOehpFlhViN1EqH2_WZbyvHD1SjOvFE2DSo4dEzpBkDmc8BqOGsuv3D-LmuwoCy9TR_4HSr1NVATth5uxM5-7HNkD_imQVIWTd93MxGlqqYmdMFOadzQFZlJgAZjDc8wUOeWd1bddoTAvOxmaoeb6ZWXyWXAtP1g09S167B7H2Jd0Tsmgk9CwPqr8OEJwJ8W4b28j80n7aknx_TYd8w9HURN7bnveQt10XLZjpJO2PfuAIB5Tq2Wa8OeLGhEVM7eTWakZ6tF3hZ66oqnDECPBKbd4cmMWpkNJio6bcfUpodVcxxvx3JYA9s13QwN-kLDeXCdqDoqB7oG8YMYnm5VJQap7T_eIGn3Epn5mx4cq-uYNcQVsZAhA9H-oLNIXRVAzhj0E3o_sanQ4iYY9PFqV3nVz3pgDoA0uFNgX6DnLRX63opsbchQGnKRo3aHemCh4Fo9odz5NwQusHW7g7oPzh9hKxQ8wx-jaqszgSQGBhxIPc_04Axb4-hmLkg97kMn64LVE2tQks1fCsIAewnJSJckr00N_rRHEukw-D0rJKUHRj4tItH2Ra8hq2HCrquLULRHXRq4hZmxjQEeHI0tVTryM3NTh2Ja87M-G68EcX8xe54fhkBkFUOjRfgZBEeVxfSeVE3-N9Y_D7nfq_jKpPy0mkRmW3dtH_--YwmjrgW1qGjRJ-37wmJwjNZ6O-kUY3n0LzArwJL9qQ6vqHoLp-ofKyouClP1R_K2-VGkBF-2oFPuyyW-2qQaYPKLrgOk1UQlhPQoChHYWviNxWKUAL40jotd3d63hhaVDT-tG2MptcG6fd4QjbJmnslVImkXKJvRV3_Db0HbPKAeaBTDhMTKnxMWK_b0g3qgkUWpGv2gK3pgwfvhSkhFQODWpv6Q3oDhpt2aZKA_IvbZQLYGBDcZX3lCYSokWP9FgiU70nZEitp8BSxNWy9vMegBPTtaKaWwnKhIp-n4iOe2Nmtub9jlhj0qQa3gTVcKl8R_zMjgcHnyj78kit1kjHpDmgShc9IKbWw6G-IcRfBQPfsyOZ4rRs8WMYPhxSzXaivuxyvCrh8mRud8NAOBCOFl3osLlikEplV_zw5SxwY4beUW9pqhqNdjo53qB2QGY_G2hBYn-Y2YRDRwqI45uyDhJHswnuGNeoMkKt9X3TGH5v-5-gaWMeG117uYoANvvbhcF7w-xFwo9tEXl34cipJy28WfgG3I2qI6E4TKZ9_lOJUQwbfAYwEBanGAu--wJQMTEztLBRbHIf6zpM8VmOj-ywYI0LFi854-RbqCP1XDcHw5tyJGnXgwVdcdL3Tg3fUNaD31YwQ1mqHJnyue9PIuVLocL7tLrxElIcWhmb2ir21YEvzaI0gx_ZiBimXULHOt1idIbExEdjAE6t247fu9Yi24Qd3XpXpT5Dza30PnOwR1R60MRDqsRXWYx-TkNh_EkvGUixAq_5e_LirE47Hjbw6K_mdlVdS41vYgir4dIxjXvX5XfZLuxGqLxyFtwTZ-SSopyDkLtPxuAmdcERjw0ldxqdhjHQStf7V_OKOL2PtaCIFus8JiY0UwXYSyBaXWP_te_zdwSQkGzMhJhXQ0F4l-28LosJB4sgPB72F4FzBkTrMCeFjcrOGCp5FhrTUEg_H7kCajbuPyQWIBLHh21VOL8Iy-d5Odq1nAr1CfH-BGCamzopEZGi9w6UI6UY3x1kvxB0sGorjKHg76TVBiSF6Xa62NXb6sefw-uf4W6iTB2stxULBenlYyNToMKFXbr_FrjfoC2LMy0G4DiUNSaB1csU7bBFJ4HGx6QuUfkvz5i_zhslm8aoPcDfu5WuU7Zq4_SpNmZVwtJhMC5svJJLfCdeFQrGssjNmWfG-1GwPaAHlQzjkLvJMsJBx3BkmCZYm4G8ZKPKzhCHCJQgeXiFqmXzEbyQlS0WdbUCRhV38uULBRrnxkocih5IdVRhfAkfykxKLHTTr1cTw6vkD-Vzgow0D6tCmYSqfgHY6fLr4lxVA8ElMNKN9ryeFwSKijKomSAEO8To3G5b6SoC5ChOWyKmud2ZbVeeZULgdaKqMdGKVJ8ItoKVxYsKCpydXg0qrHwYRYZJ4Xk-96t2BKxP8daVNqIkFeKvRMp1n_6ih4pssDo62DaMSvCgplacEww0-y97pzACle9f3X9FyL2fGrApQpH76D4Lf6708eW912kbAnvbSA-LJ56QCpMvpEq6tg50ReIrwTMLNsfvQkIA419lm-Ez5eEJXi3VUa-4Ohox3yQqc5ffJzqZgml4xkklWoZdIznfuHCmC3T2QJsfLAqvTDvy9XIPTmIwafJc8r9JhFTblVp9FJjiL0EaSvzjMjxGW2LrPRPvay5t54XDXh0oPxp161YkQpzijMK6FmhV6x9d7uoluTlNvOMCu_CLPUUzNtCsbGEuOOBhVvJ1MNN40kQlVmj42qBU2NJqpUu0NP_wA3np41CEDYpt69aE1U_tKSuyjDpx6re13xfTPufpuAK4LyNYDAeQCaWi6WoLxj-E3xJfUtfbpcO3TQByxEdwZM56yKJOrff0kUORmqpnSpfsmSpJZatUB7pVoSb93xmcvr6tYLK7hHzElj177-zd9pb1ct0vw8CO3_-GjBaf761fLZmlbm4o8zk4KnRvJ5mMhlYC6Kiqylhv7Msw-OL1W63eUy3sKQg5HOWtRNUyGQJdQ3iTG9nfnpkvjx1AMDQQ9C6rc9bMAEQBFL20Ugdc6Vh2FwZhyzTrJ4q_M_qtQF4SlXgBnFCGjfX6YqVQ3_kJh10X522AdGKSWCgmN2hEKfLmGYfJd1PT6MohKBqNot6nkJw8asF4Wpvj77fRKQy47xJ7u3x05xvjaSld-BJjZfr_xuAljbJYGHeOHEeYRpjALgOjiuYQNYa-0Wu266XGZOTJ6F12157mbtQUBjTMqXOOnrWZgjYyTvMxXVBmyayTGjZOJtmQKSY-HKh_o90CPBuO8-S7beYgPI5yrVSbCd3ScbFlOxyyRBf4Du8dC4-2-FsmjGbi4w3Z4F7b_4ZmAwG4V5FfgPznmM3xpwi3DwI5eXnOHS4-q-S3_Eprqz4J1_BqeCw3T3heIMPsxs-kYA-6m6K81iJPCvou-f0LmYOyPVfIVQu3XsoIXe1C_VCj7J1RUza7oLEYq1TqVpk9J2-jCO-a8KXQ-SwDx8s-BAfDzECS_ro-gHjhqgDLvhTKam0_2p9VKCLx5DkdeDpPNhrwCbcZaMFhQ5X79hSLOWxlKUgDdRWA7Ax59JCu-LYBtlAitcpsOo2cUCE-SQKqZzSF2bJMdws3ghJLeJNiwjRDLiZSLT7ITTcWyJ2E4xE7MQaa3C7kL4HMjvFCXCkIrMCKeQU554y5lLeuLg50pUQYVCVPt805y3UTNmH2JgvdeeNwkk3Gj9PmkZ_eeW2JXjFgCimiyuC8IYuChdEHjFlN_OyGknZb7PA7vbqfrtBgtXfGCJ_EeZSiZe_ZlWtY7N0sfJwXhT2m9ySzS_vLu0ICmANK520xiob2Y_Osh7a0YGTD4LaF1iDL9bVIjUhTSKTwj_t2GYNoewD-2R2OtGOFBYc0U8SuRJIU7pVX6tfPjDk7mK0CjG-bT1X5pWMJsoFqctx73Wz01Lxg97ikTWScly9uHCJZ7VVYV1y5-NbbVAP1zjwYiF73nlEUuhpuH8mf9vNkxWnDpKgfOCr180LyK4DChWjAgA6Ira1sfrcs_ibsBgd-5r7nGmjiF9JY9SrgbrqKIVvCUePQ06XIi0O8QMLY3CsLbVP1rDE0n5-LXyxfVIQcqkZEtsCZnHt_XnsMMYVDSuHF6FFNY3hUbQk6nTipJZAZrWddmquZ65zDR36mGLj5C-yAP5O8WeTxtLARo-E_HYLUCv4VvH8xAj4vYAfiF3PSr1s9mRw1VZvEVsXDtNE1XXGRe549fX6fjzywlOiAePUTgKRgC2FtNal0o5MQYf3zfDA0pEykFjLM2Nen9F49Yn4wFbjo0WsWrNAyYd08ME7mvRlAuKHd9KZco-FFDSY1SLkgZosl9kUe0LURMJrTBJ-twK9oyXgsYFs79nqlHE-UkNtVqEYExRXvQNEYEKm31aA9Znv8tT_Nh-o9M1Dd2tu355_aud4L2S23on0Avq8XVXJFPv2KoB6Anct_l7w7B331aiKrraIXzXwdbRDo99X3tz8fApX7XWckUXm4AGZ11ZazXJQY9Ui0R_Onz_Xmndf2KfeJ7ex3PDhNyOQz7gFf9GFf9w951egav330ObpuuWQk8jPVQUjkiIIP-hBF7oF16hrOILVqUmAgXomwmfB8N4-KbiQetqBTHmw8Po1ElTtR1rqGe-bEOirber56714CiiBSUIIQ4LdufGHs_siA0Ti8rQSF_MWuK4TIFciC6s98bTS3oYbpo8sXeLQ5w2krFcvsl8_GyLEKqrWLcYy0VdFEgJ6CSuB51rJ_gLCmaJniTWGXoIpJ7cplXHiMz4acHVBle0Uuz-MaQELaH_7PjdJQmJAGydGI-xSnzOqGctMY7CDxSFrE924M1aiC1ZoFsx97c-XczPAtqjowOdo36yIaonHWnWEF6R9L88fqSqYfyoxSPcLXV_T2xXJNbUn3QVrLAKRxGdrWVG13srgC20CKuHefQU_mluIoD5LMgXlgZBQpy6FU76OlShV3b5qFJe2ye1B8jBWSd7KGUp8COTo-ysFLZPuePCtyV-FbA2wHOAsWrv-0J5pElJJXdCaFOd_B97tRIsd692HwQQg1skSu6nj5UpXrFPJryWqLJx1nSt-IMf7Y4ar7q3HgxCTSPJ_WBSLfN32ME2ylZahyVt1zimraPxQ2ghCqohZHRnWIdEco-QwFOIRjjOz6s6PNXgHyfh50G0QRJd3401Jr6VY7PuA9yu1tOLvNQeSucdTs8jDFO7KN8pQXmSE-qQZvT98-MlP-zsClOTx9fX5vxcc85_S01WkWWS6oZJbRB4bERePIAdCLkumli2MJ7aDntoGmcVy3UKq34kYonEIFAkvB4y5ACG99uiAa5-fAhPsG4HgCW7kg5ApAn0ktoNYnXvd-vl7Wtc5I2MbL18Wdy5Gmd4Do7MkVIiSXEZbEsX9T9lKDsD39gpaK4NM7wzwUiiMYZoQRSF4bRr38n1oqKwjasFgSCsfT_P36UQlkIIsZPHbG_4shGeEvBZ8jhdiIkHkwGViNSDF8XWlGHh8-Erv-grC2-ZhtdEpF3Ide1ovXgV4xZifDzfQitG0I8T-T5FuxAz9yDM11mDK2fcFZ_UbmuNdK4oIO3UvlFeOs5zgybsx9Q3kYM-rbPs1ZK73RheKFRel8TxnBDLwCtJsA5AGAYBRM9S5f9Lr5o6v4-UFFDagJAYv0tMCn6GlUuqZdLCmXHMloX0buvjqsjgFX69J4wEaaCzSAzAHKUNpjO0uwTXOPPQsdETxwSZzMmgyuC-vOZnnytTA_owRQ5P2WLRSusyWOVqV4Z7iDllptvpJxxaekv17CK9bIl7XpriT3PEuH4YPQBrMn2kh8QNRsMGipzQzh_TLLAKgp9ALSiMc1ageRB6MLNaMijqKitez6kiZZNy8AnFX6n0anL59ce5p87VugBwYV3K8i9vqPxpFY3m25NyQgwRDpE_ePSk19ZevAqWe8yjdzfqRdVJc_waFgN3260laDm-_Gi-vWxPao7pTgFAsJlg_Sv0zuLtFaG3nNHhGp1hO0QRO-HTvOWsvG3c_-3o-igBWabxlPSgftnUKKRtO1k-nhJN3-eH03L52nZ3z9pdoaVYEVgp3tykC5QqaxyOEWS22vfTHmrlN6-dW-oDHzL1li06OSzkYu1pW7kpelfuisNHOXQ88T86_f2vid-QhqLNm3DETjZHh5SjtUL5aE3LW33TPRTPNSzBrdhAKnP3BszMnLZwCxkB9ALBCAPC67g62J0p3dCjjSgvGu21b1XswYEIOmJh_xNniMkS20Rqp5Mpw--4UdPuQCmni2_ccNH_fIyr729qKKSnqLxD9DGaR5w7gqnQEuWZykJKScOWWwrH5iQtfKL0hQRb3dvLj-Y6g6duflnFVLD2wlJ5VjnPfNPYy-B04fMzDjNl283nLEVvFhmkLjnSnzLY_yXGPRpvx875IN_uVG-o39JuXGRpr7BsAPWYcrmaAQfD60OWbW2i1OfnZHby5fSMfUCkHzT9ExR9co6YheUcXK-IlL4P9BH2U-EIZGzRTBsi47SIOyq74_roja682UeZQJVLbrfRc2GiT7j0WJv1BzaBNa64JDfM2g5MBYUFwCNgxB-0jHf5yzjhWqW2iDxc-Y-BnDXMuXAMK2KGBUnNVgrXhBFWt-3UiFzAhj3I2wYnvKiu2xMybSMG21VighyqIqUH4ftePT3qy_8XNOaw7vsjMCJAyDP4AR0FxzEtuOa9PbG4GINHmuUy4FXhEY0J9L4ia4JkxMuOicqIwBnc5wruXlUqEaiRAv4Mwjbalo8q7DYTFyPCyXrtI7CiZDP3sMNmAxbdhFlWlLKHkz0jwZ6BiDeW_9lIEPDhXP398wGR6DoK4VNEYoQ4IbGKcz5aWGzRaCsGMynJNWX-Xy8gAnhShJI9Hspkm2Ic7q_K8qM8yWrj0coAQDtTE4MHDB9Lc89RtCmsjo5SMfdGyrrehUhmJq6anmYEIvceEo9XbRXIN8aWgfAQf3gCzcE1FLescOVNR0IerGvYh2I4C8FEN0XeljA82m7luWbRLf_P-VDCMtEZsCnnOU4asvXjB7vMpm6JtlBq3zIswO1Ob21MdElM4wKfNsTaEGym5wJlpXdn6IPlpI4WS9TV7Dpa9ByvgJSAZEP-jEFYPv5IwsgNmzreUYQv2rMERPmQRbY7ww_j8MWCCA0TLOG2bS6knM88-AYk2yajr8CNiRWW-uY7oKG7CxEom7j9mEE3Ad67sSvagGP_DMR2R78TPp-G0CtQ7Nx05CVzIDpGYh2JJCOjaF7XHRVbLZ5AlrjuRO0G7nga2Ppc3XE6uMa2Lr0B84ARJ5Fg-T6VRGb8Bs0YQpjy1RTFNkjAbAMNeTuIunLogHLvbSxVcTf76YtEMqwIVufjgmLhPYRIsJGt0IsgY29HVrFA8Eak1o4VWco9ULcka6zba7SeBV6trWQ8yfusb8ur6Y3vRBYDLQ18wgUPi2P0p5ZkNpXevWGje3M09H-fzHoPk6cnRzz0WQvtrBQdd7mVwK2EnyLp1HDKIaM6JkA_jHMIUu27TDj1x8i5-z2me9NwyilR0joVBQ4KIW-zhxL54RYUMmD_pIRzQj6M62DPGZosTvcMaqrLuk32k6cFzQGZ-A9WMbmRCqTLMDDIfg1VKgSL0Cofjgh5cLcXZp2SWr8MNYtTYgx3OcNabPdBWBCGmavJRL3YekDa2g4yBrL0-_dnbpLGTkWe7SKiVJBIUSLXimlhpsuBXljpzAPf1-8JccY1vumGtsg3z4kDXjQaoV1JsHnKLNI96laQJSCLUD1uoFr1PU5pQkA6L-bfuCkqse5R7PHoElw3hrV1aZFkWLNbk5EUsQ4KGsayqRwXc7II-hBX63BFwdVZcjvVOgVSIwkQs9C_z1T0wWreXV3UQn8QPQUFw3NwsVCU2L4FwRufh42HeK0tSmCf36CZKu2eFpmM5jU90y-_ZOqTKyGUaQocNIJbbhALOmviTwnTSEE9gamJVUFOCaux-Mu2HlsZeWL_AW0NnmmKltAd1BzysTNtuFkLBDfk368COc3e5Dqg0xhntcF9dS-aD7kkZ1JYu6OcqSXDC3fIDy7DLvG2Tkb7iDMZjaXQWJRHi454pNpnwnx0IJF2GdraPKwbZwiYHgIOyHuSIET8WpOwYI2uyiKGWoRgqfa1nWY6U-DXPBGL7JHTCe5MqlVnlcQA5duH_QzuH8u849FUy52ZObMrsPGLjqDBjkKc7lW39OsCYTdnGUk-07nQVjj-T_-77qQPERwpKz2I1CS4bfm8XATvBZDHckegFweZfWgm50u2cZ0QSKPTlLQgfmx07ls7SzcoEQYotf8xDLANcylH087fR9xtDJ-3kgfGaXi3x4CsPMYs4QzzzBX_mdRRXYyspsDhAjkn0tgaYAYGFHEiaNMx89mA2QmQXWe2-7b6sEQ5uFUcuyKpISuyZ8956-xOKKdPX3KsdBOkYBD67lWW7ym7V-6SFTiYeAu65X0vaNbc2WFJ91twWpwBn9o1Ipuqz1lnE9CEKcMvMg0cMKwCzgyBFZXxnCFJXbak01YEPT34qyguxhQf19ir1XSTjwfM8H7sWldt80SWZJ_7dX6ERkavpGJb6h70-Vb_Cbo7CZLB3Z3RV2OQZAwLlEAdvr2LcZp55R_tls9_USo4k4926JOSlpz3GNSlh0QW77uW_KRnT2yj3uekdIl9Bm9_OVYkuB4NnCXmseEo6QIxDoh01FNlTuKgTZewestZf2QpVS76uu5Kqegv5_tuDSU_6aTJanbsceTMrb65a80NxSnGN-n4TZvB6LLWETEKJE9kqT24o5Brb9mET5DLUltyj4hNMnUd9m-XZ_PRQ1S5FN1C65xKwRnewUfMB2Ss3NdbmVTkWkJIUgeeXwNj1dUY18gzii3amVwc1XrRsRZh9ExuSgfaxWPkMtU0Nw5rTQul96NCq6QESDDJzwz7QQcnt-c_2YfD8r0gzvFT7ChUgCkwefEG7qnTD17QAcK5ESWcaeN3mCEgVnxizIlZeoDTkEN5wYi8BTeOR7uT9taBUWMcVaVvBmZlRPOZ3a0fR9zIueZarQGaWE8VQ7eWWt5yyl6MXng3BxWNlD5RzlSzWjathzD85WJAHsv5mAdBv7W534Sf_jqR5Dxm_SrQH72qV2wYNeiTMdtfeNwzf_TmbJ2OM09CG5Q5vMXBlHsSlb8MwsSLjbyNYGo7l5xXX8OD2doffPeZOiqOldOkLhGcbbRfJw2LObxxIgRTrZ9d7Pay3HWo3mRTE08P0CM1Tp5CsBqD1QOedbM06pP5KzHjLs4wc6w2IzrQFHJg4-qd-BntK8N-xjgO5IT18A8yn0erstdJLYeNxL_iJZtNvViBytFhF-Y7aQlkFg5p3LE2biFe6bZChFaOMae_9R5nLR696aq50NAO31c7hylyDxGLP8rl7ygQxBDzABX3rP4zCNu-Nau_xHkh1aDyjbvwcmMNDxy2JFYEMqu6oRI6EStsM_PDdXz_NI5B8VxIGk8p6weSiQcIkRJw07hp6RbaLXs-Hs-GWc3KP4J9br7gouSi0kC',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content="""\
**Advising on street crossing safety**

I can provide guidance on how to cross the street safely. First, find a crosswalk or intersection with signals. Stop at the curb and look for the signals; push the button and wait for the walk symbol. Always check left-right-left and over your shoulder for any vehicles, making eye contact with drivers. I'll also mention various contexts like crossing at night, with children, or on multi-lane roads, emphasizing the importance of obeying signals and staying visible.\
""",
                        id='rs_0f522e1dce1a173a0069814689807c819489cb71998380b944',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content="""\
**Giving crossing advice without a crosswalk**

If there's no crosswalk, I should suggest finding a well-lit spot with good visibility, avoiding curves or hills. The user needs to wait for a big enough gap in both directions, estimating speed and distance. They should cross one lane at a time and pause in the median if it's safe. I can note that it's illegal to cross mid-block in some places, reminding them to obey local laws. I should also keep it concise and consider the possibility that the user might be a child or visually impaired. Adding safety disclaimers like, "I can offer general tips, but I can't see your surroundings," will fit well.\
""",
                        id='rs_0f522e1dce1a173a0069814689807c819489cb71998380b944',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content="""\
**Outlining street crossing steps**

Even though there's no image, I can still provide steps for crossing at signalized and unsignalized crosswalks, and when it's safe to cross mid-block. I won't include professional disclaimers but can add a short one-liner. So here's a structure I could use:

- Opening: "Here's a safe way to cross a street."
- Basic steps:
  1) Choose a safe crossing point, like a crosswalk or signal, avoiding spots between parked cars. Use overpasses or underpasses if available.
  2) Stop at the curb, remove distractions, hold kids' hands, and keep pets leashed.
  3) For signalized crossings, press the button and wait for the WALK signal, always checking for turning vehicles.\
""",
                        id='rs_0f522e1dce1a173a0069814689807c819489cb71998380b944',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content="""\
**Providing street crossing guidance**

Here's a simple, safe way to cross a street:

1) Choose the safest crossing point, like a crosswalk or intersection; avoid curves, hills, and areas between parked cars.  \n\
2) Stop at the edge, remove distractions, and wait for a gap; keep kids close.  \n\
3) If there's a signal, press the button, wait for the WALK, and look for turning cars.  \n\
4) If no signal, look left-right-left, listen, and wait for a big gap in both directions.  \n\
5) When you cross, walk briskly but don't run; scan for vehicles and stay safe.  \n\
6) At night, wear reflective clothing, use a flashlight, and exercise extra caution.  \n\
7) If crossing with children, pets, or mobility aids, hold hands and allow more time.  \n\
8) Always obey local signals and laws; remember, visibility is key. \n\

Are you at a crosswalk or signal?\
""",
                        id='rs_0f522e1dce1a173a0069814689807c819489cb71998380b944',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content="""\
**Finalizing street crossing advice**

I need to make sure we address some edge cases in the crossing advice. For multi-lane roads, I'll remind users to watch for a "second-lane threat," where a car in one lane stops, but another lane may have cars passing. I'll also note that if the "Don't Walk" signal starts flashing while someone is already crossing, they should continue to the other side without stopping. If there isn't a sidewalk or shoulder, they should avoid crossing if it's unsafe. Alright, let's compile all this into a cohesive guide!\
""",
                        id='rs_0f522e1dce1a173a0069814689807c819489cb71998380b944',
                        provider_name='openai',
                    ),
                    TextPart(
                        content="""\
Here's a simple, safe way to cross a street:

1) Pick the safest place
- Use a crosswalk, pedestrian signal, or a well-lit intersection. Avoid crossing between parked cars, on curves, or just over hills.

2) Prepare at the curb
- Stop at the edge; remove distractions (phones, headphones volume). Keep kids close and pets leashed.

3) If there's a pedestrian signal
- Press the button (if there is one) and wait for the WALK symbol/green man.
- Even on WALK, look for turning vehicles and drivers who may not yield. Try to make eye contact.

4) If there's no signal
- Look left-right-left, then behind you for turning cars and across the street again. Listen for approaching traffic.
- Wait for a big gap in both directions. If a car stops for you, check that traffic in the other lanes is stopping too.

5) While crossing
- Walk, don't run; go straight across. Keep scanning for cars, especially turning vehicles and those coming out of driveways.
- If the signal changes while you're already in the street, continue to the far curb.
- If there's a safe median/refuge island, you can pause there if needed.

6) Low light or bad weather
- Wear bright/reflective clothing and consider a small light. Take extra time--drivers may see you later than you expect.

General tips
- Never assume a driver sees you; visibility and attention vary.
- Obey pedestrian signals and local laws.
- With children or anyone needing extra time, cross together and at signals when possible.

If you tell me whether you're at a signalized crosswalk, an unsignalized intersection, or midblock, I can tailor the steps.\
""",
                        id='msg_0f522e1dce1a173a00698146b4f36081948501ae14c42b727c',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(input_tokens=23, output_tokens=2256, details={'reasoning_tokens': 1856}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
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
                usage=RequestUsage(input_tokens=1421, output_tokens=590),
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
        "Based on your location, you're in Mexico. The largest city in Mexico is Mexico City (Ciudad de México), which is both the capital and the most populous city in the country. Mexico City is one of the largest urban areas in the world with a population of over 9 million people in the city proper, and over 21 million in its metropolitan area."
    )


async def test_bedrock_group_consecutive_tool_return_parts(bedrock_provider: BedrockProvider):
    """
    Test that consecutive ToolReturnPart objects are grouped into a single user message for Bedrock.
    """
    model = BedrockConverseModel('amazon.nova-micro-v1:0', provider=bedrock_provider)
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
            PartStartEvent(index=0, part=ThinkingPart(content='The human')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' has')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' just said "Hello" to me.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' This is a simple greeting,')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' so I should respond in a friendly an')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='d wel')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="coming way. I'll keep")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' my')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' response warm')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' brief since')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they haven')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'t asked anything")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' specific')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' yet.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(signature_delta=IsStr(), provider_name='bedrock')),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content='The human has just said "Hello" to me. This is a simple greeting, so I should respond in a friendly and welcoming way. I\'ll keep my response warm but brief since they haven\'t asked anything specific yet.',
                    signature='EvYCCkgICxABGAIqQH+vdAorxvHnWTPEdc8BK/BavcOAlYhx4WJGgsYDLAOnOlProhrcdoakElJ1TyNwuVoDmeufvBY0r1oL2vhd1goSDJPU6x/yFNQsgepyvBoMBDWVqEroDO1NxlW5IjAVu0TJ7SNljvQ6W2VwGVC5HiORuEoWIzQcAm8CEbVhXN+MiMJ9kPPqIzEyolBP91kq2wF5KP9Z5za6wDigTkqSbmFOYsqsugcW63mgtP7yuWSIQbiYkA+uUrf0qx8PsZjNZKi0VmqRJK9NRbDDhMoHt/AWqMLiYbEjz3aVYJGGU8UTNkVc4+Iqw9Ha/O9Q+7Y/wF2Ycrdt1n74u39t/0v/ZTQh5vgVpb24S7nqAG0Guim9WX00ALjRPBV2ugqaD/CE6haaLATMrTrBMGep6dhGnkypmO2loFPgHTU43UR17Iv9ltU0KJF6QFCZx7L67OzpgXkAQeh2eEuphYpJUMqN2rAHzByemarql73B96wYAQ==',
                    provider_name='bedrock',
                ),
                next_part_kind='text',
            ),
            PartStartEvent(index=1, part=TextPart(content='Hello! How'), previous_part_kind='thinking'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' are you doing today? Is')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' there anything I can help you with?')),
            PartEndEvent(
                index=1, part=TextPart(content='Hello! How are you doing today? Is there anything I can help you with?')
            ),
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
                        content='The human has just said "Hello" to me. This is a simple greeting, so I should respond in a friendly and welcoming way. I\'ll keep my response warm but brief since they haven\'t asked anything specific yet.',
                        signature=IsStr(),
                        provider_name='bedrock',
                    ),
                    TextPart(content='Hello! How are you doing today? Is there anything I can help you with?'),
                ],
                usage=RequestUsage(input_tokens=37, output_tokens=74),
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
    model = BedrockConverseModel('amazon.nova-micro-v1:0', provider=bedrock_provider)
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
    model = BedrockConverseModel('amazon.nova-micro-v1:0', provider=bedrock_provider)
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
                    content='We need to respond as ChatGPT. The user just said "Hi". Probably respond friendly and ask how can help.'
                ),
            ),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content='We need to respond as ChatGPT. The user just said "Hi". Probably respond friendly and ask how can help.'
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=1, part=TextPart(content='Hello! 👋 How can I assist you today?'), previous_part_kind='thinking'
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartEndEvent(index=1, part=TextPart(content='Hello! 👋 How can I assist you today?')),
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
    assert second_usage == snapshot(RunUsage(input_tokens=1324, cache_write_tokens=1322, output_tokens=5, requests=1))


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
    model = BedrockConverseModel('amazon.nova-micro-v1:0', provider=bedrock_provider)

    # Create a message history that includes a ModelResponse with empty parts
    req = [
        ModelRequest(parts=[UserPromptPart(content='Hello')]),
        ModelResponse(
            parts=[],
            usage=RequestUsage(input_tokens=100, output_tokens=1),
            model_name='amazon.nova-micro-v1:0',
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
                        tool_call_id='tooluse_EkXfr5c0K3VyZ09M6pxEFg',
                        provider_name='bedrock',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'stdOut': '7006652', 'stdErr': '', 'exitCode': 0, 'isError': False},
                        tool_call_id='tooluse_EkXfr5c0K3VyZ09M6pxEFg',
                        timestamp=IsDatetime(),
                        provider_name='bedrock',
                        provider_details={'status': 'success'},
                    ),
                    ToolCallPart(
                        tool_name='final_result',
                        args={'result': 7006652.0},
                        tool_call_id='tooluse_owdFRfkTTQabYjvP3GM_vQ',
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
                        tool_call_id='tooluse_owdFRfkTTQabYjvP3GM_vQ',
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
                        tool_call_id='tooluse_1f51iFYBVw5nfnBZuSWrsw',
                        provider_name='bedrock',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'stdOut': '14013304', 'stdErr': '', 'exitCode': 0, 'isError': False},
                        tool_call_id='tooluse_1f51iFYBVw5nfnBZuSWrsw',
                        timestamp=IsDatetime(),
                        provider_name='bedrock',
                        provider_details={'status': 'success'},
                    ),
                    ToolCallPart(
                        tool_name='final_result',
                        args={'result': 14013304.0},
                        tool_call_id='tooluse_gp3q9OOSSgqdIioRsfzs3w',
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
                        tool_call_id='tooluse_gp3q9OOSSgqdIioRsfzs3w',
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
                        tool_call_id='tooluse_3ioV80ZZClc5KdeIhGx32A',
                        provider_name='bedrock',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'stdOut': '7006652', 'stdErr': '', 'exitCode': 0, 'isError': False},
                        tool_call_id='tooluse_3ioV80ZZClc5KdeIhGx32A',
                        timestamp=IsDatetime(),
                        provider_name='bedrock',
                        provider_details={'status': 'success'},
                    ),
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"result":7006652.0}',
                        tool_call_id='tooluse_7hzWoeZES8ykBy2ZYAyUeA',
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
                        tool_call_id='tooluse_7hzWoeZES8ykBy2ZYAyUeA',
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
                    tool_name='code_execution', tool_call_id='tooluse_3ioV80ZZClc5KdeIhGx32A', provider_name='bedrock'
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ToolCallPartDelta(
                    args_delta='{"snippet":"1234 * 5678"}', tool_call_id='tooluse_3ioV80ZZClc5KdeIhGx32A'
                ),
            ),
            PartEndEvent(
                index=0,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args='{"snippet":"1234 * 5678"}',
                    tool_call_id='tooluse_3ioV80ZZClc5KdeIhGx32A',
                    provider_name='bedrock',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'stdOut': '7006652', 'stdErr': '', 'exitCode': 0, 'isError': False},
                    tool_call_id='tooluse_3ioV80ZZClc5KdeIhGx32A',
                    timestamp=IsDatetime(),
                    provider_name='bedrock',
                    provider_details={'status': 'success'},
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=2,
                part=ToolCallPart(tool_name='final_result', tool_call_id='tooluse_7hzWoeZES8ykBy2ZYAyUeA'),
                previous_part_kind='builtin-tool-return',
            ),
            FinalResultEvent(tool_name='final_result', tool_call_id='tooluse_7hzWoeZES8ykBy2ZYAyUeA'),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='{"result":7006652.0}', tool_call_id='tooluse_7hzWoeZES8ykBy2ZYAyUeA'
                ),
            ),
            PartEndEvent(
                index=2,
                part=ToolCallPart(
                    tool_name='final_result', args='{"result":7006652.0}', tool_call_id='tooluse_7hzWoeZES8ykBy2ZYAyUeA'
                ),
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args='{"snippet":"1234 * 5678"}',
                    tool_call_id='tooluse_3ioV80ZZClc5KdeIhGx32A',
                    provider_name='bedrock',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'stdOut': '7006652', 'stdErr': '', 'exitCode': 0, 'isError': False},
                    tool_call_id='tooluse_3ioV80ZZClc5KdeIhGx32A',
                    timestamp=IsDatetime(),
                    provider_name='bedrock',
                    provider_details={'status': 'success'},
                )
            ),
        ]
    )
