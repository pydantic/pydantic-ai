# pyright: reportPrivateUsage=false
from typing import Literal, cast

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    ModelHTTPError,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolDefinition,
)
from pydantic_ai.direct import model_request, model_request_stream
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.usage import RequestUsage

from ..conftest import IsDatetime, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.databricks import (
        DatabricksModel,
        DatabricksReasoningContent,
        DatabricksSummaryText,
    )
    from pydantic_ai.providers.databricks import DatabricksProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]

DATABRICKS_MODELS = [
    'databricks-gpt-5-2',
    'databricks-claude-opus-4-5',
    'databricks-gpt-oss-120b',
    'databricks-qwen3-next-80b-a3b-instruct',
]


@pytest.mark.vcr(match_on=['method', 'scheme', 'path', 'query'])
@pytest.mark.parametrize('model_name', DATABRICKS_MODELS)
class TestDatabricks:
    async def test_databricks_simple_request(
        self,
        allow_model_requests: None,
        databricks_api_key: str,
        databricks_base_url: str,
        model_name: str,
    ) -> None:
        provider = DatabricksProvider(api_key=databricks_api_key, base_url=databricks_base_url)
        model = DatabricksModel(model_name, provider=provider)

        response = await model_request(
            model,
            [
                ModelRequest.user_text_prompt(
                    'What is the capital of France? lowercase, one word reply, no punctuation.'
                )
            ],
        )

        text_part = cast(TextPart, response.parts[0])

        assert 'paris' in text_part.content

        assert response.provider_details is not None
        assert response.provider_details['finish_reason'] == 'stop'

    async def test_databricks_stream(
        self,
        allow_model_requests: None,
        databricks_api_key: str,
        databricks_base_url: str,
        model_name: str,
    ) -> None:
        provider = DatabricksProvider(api_key=databricks_api_key, base_url=databricks_base_url)
        model = DatabricksModel(model_name, provider=provider)

        async with model_request_stream(model, [ModelRequest.user_text_prompt('Count to 3')]) as stream:
            chunks = [chunk async for chunk in stream]

            assert len(chunks) > 0
            assert stream.provider_details is not None
            assert stream.provider_details['finish_reason'] == 'stop'

    async def test_databricks_tool_calling(
        self,
        allow_model_requests: None,
        databricks_api_key: str,
        databricks_base_url: str,
        model_name: str,
    ) -> None:
        provider = DatabricksProvider(api_key=databricks_api_key, base_url=databricks_base_url)

        class GetWeather(BaseModel):
            """Get the current weather."""

            location: str
            unit: Literal['celsius', 'fahrenheit'] = 'celsius'

        model = DatabricksModel(model_name, provider=provider)

        response = await model_request(
            model,
            [ModelRequest.user_text_prompt('What is the weather in London?')],
            model_request_parameters=ModelRequestParameters(
                function_tools=[
                    ToolDefinition(
                        name='get_weather',
                        description=GetWeather.__doc__,
                        parameters_json_schema=GetWeather.model_json_schema(),
                    )
                ]
            ),
        )

        assert len(response.parts) == 1
        tool_call_part = response.parts[0]
        assert isinstance(tool_call_part, ToolCallPart)
        assert tool_call_part.tool_name == 'get_weather'

        assert 'location' in tool_call_part.args_as_dict()
        assert 'London' in tool_call_part.args_as_dict()['location']

    async def test_databricks_structured_output(
        self,
        allow_model_requests: None,
        databricks_api_key: str,
        databricks_base_url: str,
        model_name: str,
    ) -> None:
        provider = DatabricksProvider(api_key=databricks_api_key, base_url=databricks_base_url)
        model = DatabricksModel(model_name, provider=provider)

        class CityInfo(BaseModel):
            city: str
            country: str
            population_approx: int

        agent = Agent(model, output_type=CityInfo)

        result = await agent.run('Tell me about Tokyo. Provide a structured response.')

        assert isinstance(result.output, CityInfo)
        assert result.output.city == 'Tokyo'
        assert result.output.country == 'Japan'
        assert result.output.population_approx > 1000000

    async def test_databricks_usage_tracking(
        self,
        allow_model_requests: None,
        databricks_api_key: str,
        databricks_base_url: str,
        model_name: str,
    ) -> None:
        provider = DatabricksProvider(api_key=databricks_api_key, base_url=databricks_base_url)
        model = DatabricksModel(model_name, provider=provider)
        agent = Agent(model)

        result = await agent.run('Hello')

        usage = result.usage()
        assert usage.requests == 1
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0

        last_msg = result.all_messages()[-1]
        assert isinstance(last_msg, ModelResponse)
        assert last_msg.provider_details is not None

    async def test_databricks_error_handling(
        self,
        allow_model_requests: None,
        databricks_base_url: str,
        model_name: str,
    ) -> None:
        """Test error handling using standard VCR or by verifying behavior on failure."""
        provider = DatabricksProvider(api_key='invalid-key', base_url=databricks_base_url)
        model = DatabricksModel(model_name, provider=provider)

        with pytest.raises(ModelHTTPError) as exc_info:
            await model_request(model, [ModelRequest.user_text_prompt('Hello')])

        assert exc_info.value.status_code in (400, 401, 403)


class TestDatabricksReasoningContent:
    """Tests for DatabricksReasoningContent.get_value() variants."""

    def test_get_value_text(self):
        block = DatabricksReasoningContent(type='reasoning', text='hello')
        assert block.get_value() == 'hello'

    def test_get_value_content(self):
        block = DatabricksReasoningContent(type='reasoning', content='from content field')
        assert block.get_value() == 'from content field'

    def test_get_value_text_preferred_over_content(self):
        block = DatabricksReasoningContent(type='reasoning', text='from text', content='from content')
        assert block.get_value() == 'from text'

    def test_get_value_summary(self):
        block = DatabricksReasoningContent(
            type='reasoning',
            summary=[
                DatabricksSummaryText(type='summary_text', text='a'),
                DatabricksSummaryText(type='summary_text', text='b'),
            ],
        )
        assert block.get_value() == 'ab'

    def test_get_value_empty(self):
        block = DatabricksReasoningContent(type='reasoning')
        assert block.get_value() == ''


async def test_databricks_list_content_with_reasoning(
    allow_model_requests: None,
    databricks_api_key: str,
    databricks_base_url: str,
) -> None:
    """Databricks list content with text+reasoning blocks is parsed correctly."""
    provider = DatabricksProvider(api_key=databricks_api_key, base_url=databricks_base_url)
    model = DatabricksModel('databricks-gpt-5-2', provider=provider)

    response = await model_request(
        model,
        [ModelRequest.user_text_prompt('What is 2+2? Think step by step.')],
    )

    assert len(response.parts) >= 2
    # Reasoning block becomes ThinkingPart, text block becomes TextPart
    text_part = cast(TextPart, response.parts[1])
    assert 'The answer is 4.' in text_part.content

    assert response == snapshot(
        ModelResponse(
            parts=[
                ThinkingPart(
                    content='I need to add 2 and 2 together.', id='reasoning_content', provider_name='databricks'
                ),
                TextPart(content='The answer is 4.'),
            ],
            usage=RequestUsage(input_tokens=12, output_tokens=15),
            model_name='databricks-gpt-5-2',
            timestamp=IsDatetime(),
            provider_name='databricks',
            provider_url='https://mock.databricks.com/serving-endpoints/',
            provider_details={'finish_reason': 'stop', 'timestamp': IsDatetime()},
            provider_response_id='chatcmpl-list-content-test',
            finish_reason='stop',
        )
    )


async def test_databricks_missing_id_placeholder(
    allow_model_requests: None,
    databricks_api_key: str,
    databricks_base_url: str,
) -> None:
    """Databricks responses with null id get a placeholder id."""
    provider = DatabricksProvider(api_key=databricks_api_key, base_url=databricks_base_url)
    model = DatabricksModel('databricks-gpt-5-2', provider=provider)

    response = await model_request(
        model,
        [ModelRequest.user_text_prompt('Say hello')],
    )

    assert response.provider_response_id == 'databricks-placeholder-id'
    text_part = cast(TextPart, response.parts[0])
    assert 'hello' in text_part.content

    assert response == snapshot(
        ModelResponse(
            parts=[TextPart(content='hello')],
            usage=RequestUsage(input_tokens=5, output_tokens=1),
            model_name='databricks-gpt-5-2',
            timestamp=IsDatetime(),
            provider_name='databricks',
            provider_url='https://mock.databricks.com/serving-endpoints/',
            provider_details={'finish_reason': 'stop', 'timestamp': IsDatetime()},
            provider_response_id='databricks-placeholder-id',
            finish_reason='stop',
        )
    )


async def test_databricks_safety_identifier(
    allow_model_requests: None,
    databricks_api_key: str,
    databricks_base_url: str,
) -> None:
    """safety_identifier from Databricks response is captured in provider_details."""
    provider = DatabricksProvider(api_key=databricks_api_key, base_url=databricks_base_url)
    model = DatabricksModel('databricks-gpt-5-2', provider=provider)

    response = await model_request(
        model,
        [ModelRequest.user_text_prompt('Hello')],
    )

    assert response.provider_details is not None
    assert response.provider_details.get('safety_identifier') == 'safety-abc-123'

    assert response == snapshot(
        ModelResponse(
            parts=[TextPart(content='Hi there!')],
            usage=RequestUsage(input_tokens=5, output_tokens=3),
            model_name='databricks-gpt-5-2',
            timestamp=IsDatetime(),
            provider_name='databricks',
            provider_url='https://mock.databricks.com/serving-endpoints/',
            provider_details={
                'finish_reason': 'stop',
                'safety_identifier': 'safety-abc-123',
                'timestamp': IsDatetime(),
            },
            provider_response_id='chatcmpl-safety-test',
            finish_reason='stop',
        )
    )


async def test_databricks_stream_structured_content(
    allow_model_requests: None,
    databricks_api_key: str,
    databricks_base_url: str,
) -> None:
    """Streaming with structured content blocks (reasoning + text) produces events."""
    provider = DatabricksProvider(api_key=databricks_api_key, base_url=databricks_base_url)
    model = DatabricksModel('databricks-gpt-5-2', provider=provider)

    async with model_request_stream(
        model,
        [ModelRequest.user_text_prompt('What is 2+2? Think step by step.')],
    ) as stream:
        chunks = [chunk async for chunk in stream]

        assert len(chunks) > 0
        assert stream.provider_details is not None
        assert stream.provider_details['finish_reason'] == 'stop'


async def test_databricks_list_content_text_only(
    allow_model_requests: None,
    databricks_api_key: str,
    databricks_base_url: str,
) -> None:
    """List content with only text blocks (no reasoning) is handled correctly."""
    provider = DatabricksProvider(api_key=databricks_api_key, base_url=databricks_base_url)
    model = DatabricksModel('databricks-gpt-5-2', provider=provider)

    response = await model_request(
        model,
        [ModelRequest.user_text_prompt('Say hello')],
    )

    assert len(response.parts) == 1
    text_part = cast(TextPart, response.parts[0])
    assert 'Hello' in text_part.content

    assert response == snapshot(
        ModelResponse(
            parts=[TextPart(content='Hello there!')],
            usage=RequestUsage(input_tokens=5, output_tokens=3),
            model_name='databricks-gpt-5-2',
            timestamp=IsDatetime(),
            provider_name='databricks',
            provider_url='https://mock.databricks.com/serving-endpoints/',
            provider_details={'finish_reason': 'stop', 'timestamp': IsDatetime()},
            provider_response_id='chatcmpl-text-only-list',
            finish_reason='stop',
        )
    )


async def test_databricks_stream_empty_reasoning(
    allow_model_requests: None,
    databricks_api_key: str,
    databricks_base_url: str,
) -> None:
    """Streaming with an empty reasoning block (no text) skips it and continues."""
    provider = DatabricksProvider(api_key=databricks_api_key, base_url=databricks_base_url)
    model = DatabricksModel('databricks-gpt-5-2', provider=provider)

    async with model_request_stream(
        model,
        [ModelRequest.user_text_prompt('Think about nothing.')],
    ) as stream:
        chunks = [chunk async for chunk in stream]

        assert len(chunks) > 0
        assert stream.provider_details is not None
        assert stream.provider_details['finish_reason'] == 'stop'
