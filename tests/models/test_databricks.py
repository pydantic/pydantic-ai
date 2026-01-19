import base64
from typing import Literal, cast
from unittest.mock import MagicMock, patch

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    BinaryContent,
    ModelHTTPError,
    ModelRequest,
    ModelResponse,
    RunUsage,
    TextPart,
    ToolCallPart,
    ToolDefinition,
)
from pydantic_ai.direct import model_request, model_request_stream
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.usage import RequestUsage

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.databricks import DatabricksModel, DatabricksModelSettings
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
        # Note: Snapshots may need updating as different models output different text
        assert text_part.content == snapshot('paris')

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
            assert stream.provider_details == snapshot({'usage': RequestUsage()})

            chunks = [chunk async for chunk in stream]

            assert len(chunks) > 0
            assert stream.provider_details is not None
            assert stream.provider_details['finish_reason'] == 'stop'
            assert 'usage' in stream.provider_details

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
        assert 'usage' in last_msg.provider_details

    async def test_databricks_multimodal_content_structure(
        self,
        allow_model_requests: None,
        databricks_api_key: str,
        databricks_base_url: str,
        model_name: str,
    ) -> None:
        """
        Test that multimodal inputs (Text + Image) are correctly mapped to the
        List[ContentItem] schema required by Databricks, using _map_messages inspection.
        """
        provider = DatabricksProvider(api_key=databricks_api_key, base_url=databricks_base_url)
        model = DatabricksModel(model_name, provider=provider)

        fake_image_data = b'fake_image_bytes'
        inputs = [
            'What is in this image?',
            BinaryContent(data=fake_image_data, media_type='image/png'),
        ]

        message = ModelRequest.user_text_prompt(inputs)

        mapped_messages = await model._map_messages([message], None)  # type: ignore[reportPrivateUsage]

        assert len(mapped_messages) == 1
        content = mapped_messages[0]['content']

        assert isinstance(content, list)
        assert len(content) == 2

        assert content[0] == {'type': 'text', 'text': 'What is in this image?'}

        b64_img = base64.b64encode(fake_image_data).decode('utf-8')
        assert content[1] == {
            'type': 'image_url',
            'image_url': {'url': f'data:image/png;base64,{b64_img}'},
        }

    async def test_databricks_reasoning_effort_setting(
        self,
        allow_model_requests: None,
        databricks_api_key: str,
        databricks_base_url: str,
        model_name: str,
    ) -> None:
        """
        Test that the 'reasoning_effort' parameter is correctly passed to the client.
        """
        provider = DatabricksProvider(api_key=databricks_api_key, base_url=databricks_base_url)
        model = DatabricksModel(model_name, provider=provider)
        settings = DatabricksModelSettings(openai_reasoning_effort='high')

        with patch.object(model.client.chat.completions, 'create', new_callable=MagicMock) as mock_create:

            async def async_return(*args, **kwargs):
                from openai.types.chat import ChatCompletion, ChatCompletionMessage
                from openai.types.chat.chat_completion import Choice

                return ChatCompletion(
                    id='test-id',
                    created=123,
                    model=model_name,
                    object='chat.completion',
                    choices=[
                        Choice(
                            index=0,
                            message=ChatCompletionMessage(role='assistant', content='Thoughtful answer'),
                            finish_reason='stop',
                        )
                    ],
                )

            mock_create.side_effect = async_return

            await model_request(model, [ModelRequest.user_text_prompt('Think hard')], model_settings=settings)

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs.get('reasoning_effort') == 'high'

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
