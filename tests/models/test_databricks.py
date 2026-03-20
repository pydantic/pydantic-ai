# pyright: reportPrivateUsage=false
import base64
from typing import Any, Literal, cast
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    BinaryContent,
    ModelHTTPError,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolDefinition,
    UserPromptPart,
)
from pydantic_ai.direct import model_request, model_request_stream
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.usage import RequestUsage

from ..conftest import try_import

with try_import() as imports_successful:
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice

    from pydantic_ai.models.databricks import (
        DatabricksModel,
        DatabricksModelSettings,
        DatabricksReasoningContent,
        DatabricksStreamedResponse,
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

        message = ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'What is in this image?',
                        BinaryContent(data=fake_image_data, media_type='image/png'),
                    ]
                )
            ]
        )

        mapped_messages = await model._map_messages([message], None)  # type: ignore[reportPrivateUsage]

        assert len(mapped_messages) == 1

        message_param = mapped_messages[0]
        assert 'content' in message_param
        content = message_param['content']

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
            # FIX: Add types for args and kwargs
            async def async_return(*args: Any, **kwargs: Any):
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


class TestValidateCompletion:
    """Tests for DatabricksModel._validate_completion edge cases."""

    def _make_model(self) -> DatabricksModel:
        provider = DatabricksProvider(api_key='test', base_url='https://test.com')
        return DatabricksModel('test-model', provider=provider)

    def test_empty_choices(self):
        """Response with no choices delegates to parent."""
        model = self._make_model()
        response = ChatCompletion(id='test', choices=[], created=0, model='test', object='chat.completion')
        result = model._validate_completion(response)
        assert result.id == 'test'

    def test_missing_id_gets_placeholder(self):
        """Response with null id gets a placeholder."""
        model = self._make_model()
        response = ChatCompletion(
            id='test-id',
            choices=[
                Choice(
                    index=0,
                    finish_reason='stop',
                    message=ChatCompletionMessage(role='assistant', content='hello'),
                )
            ],
            created=0,
            model='test',
            object='chat.completion',
        )
        data = response.model_dump(mode='json', warnings=False)
        data['id'] = None
        patched = ChatCompletion.model_construct(**data)
        result = model._validate_completion(patched)
        assert result.id == 'databricks-placeholder-id'

    def test_list_content_with_text_and_reasoning(self):
        """List content is parsed into text + reasoning_content."""
        model = self._make_model()
        response = ChatCompletion(
            id='test',
            choices=[
                Choice(
                    index=0,
                    finish_reason='stop',
                    message=ChatCompletionMessage(role='assistant', content='placeholder'),
                )
            ],
            created=0,
            model='test',
            object='chat.completion',
        )
        data = response.model_dump(mode='json', warnings=False)
        data['choices'][0]['message']['content'] = [
            {'type': 'text', 'text': 'The answer is 4.'},
            {'type': 'reasoning', 'text': 'I calculated 2+2.'},
        ]
        patched = ChatCompletion.model_construct(**data)
        result = model._validate_completion(patched)
        assert result.choices[0].message.content == 'The answer is 4.'
        assert getattr(result.choices[0].message, 'reasoning_content', None) == 'I calculated 2+2.'

    def test_list_content_invalid_blocks_raises(self):
        """Invalid list content: TypeAdapter ValidationError is caught but model_validate fails."""
        from pydantic import ValidationError as PydanticValidationError

        model = self._make_model()
        response = ChatCompletion(
            id='test',
            choices=[
                Choice(
                    index=0,
                    finish_reason='stop',
                    message=ChatCompletionMessage(role='assistant', content='placeholder'),
                )
            ],
            created=0,
            model='test',
            object='chat.completion',
        )
        data = response.model_dump(mode='json', warnings=False)
        data['choices'][0]['message']['content'] = [{'type': 'unknown_type', 'bad': True}]
        patched = ChatCompletion.model_construct(**data)
        with pytest.raises(PydanticValidationError):
            model._validate_completion(patched)

    def test_string_content_passes_through(self):
        """Normal string content is not modified."""
        model = self._make_model()
        response = ChatCompletion(
            id='test',
            choices=[
                Choice(
                    index=0,
                    finish_reason='stop',
                    message=ChatCompletionMessage(role='assistant', content='hello world'),
                )
            ],
            created=0,
            model='test',
            object='chat.completion',
        )
        result = model._validate_completion(response)
        assert result.choices[0].message.content == 'hello world'

    def test_list_content_text_only_no_reasoning(self):
        """List content with only text blocks — no reasoning_content set."""
        model = self._make_model()
        response = ChatCompletion(
            id='test',
            choices=[
                Choice(
                    index=0,
                    finish_reason='stop',
                    message=ChatCompletionMessage(role='assistant', content='placeholder'),
                )
            ],
            created=0,
            model='test',
            object='chat.completion',
        )
        data = response.model_dump(mode='json', warnings=False)
        data['choices'][0]['message']['content'] = [
            {'type': 'text', 'text': 'Just text.'},
        ]
        patched = ChatCompletion.model_construct(**data)
        result = model._validate_completion(patched)
        assert result.choices[0].message.content == 'Just text.'
        # No reasoning_content should be set
        assert not getattr(result.choices[0].message, 'reasoning_content', None)


class TestDatabricksMapUsage:
    """Tests for DatabricksModel._map_usage edge cases."""

    def _make_model(self) -> DatabricksModel:
        provider = DatabricksProvider(api_key='test', base_url='https://test.com')
        return DatabricksModel('test-model', provider=provider)

    def test_no_usage_returns_empty(self):
        model = self._make_model()
        response = ChatCompletion(id='test', choices=[], created=0, model='test', object='chat.completion')
        # response.usage is None by default
        result = model._map_usage(response)
        assert result == RequestUsage()

    def test_usage_with_reasoning_tokens_in_model_extra(self):
        """Reasoning tokens found in model_extra when not a direct attribute."""
        model = self._make_model()
        response = ChatCompletion(
            id='test',
            choices=[
                Choice(
                    index=0,
                    finish_reason='stop',
                    message=ChatCompletionMessage(role='assistant', content='hi'),
                )
            ],
            created=0,
            model='test',
            object='chat.completion',
            usage={'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30, 'reasoning_tokens': 5},  # type: ignore
        )
        result = model._map_usage(response)
        assert result.input_tokens == 10
        assert result.output_tokens == 20
        assert result.details.get('reasoning_tokens') == 5


class TestDatabricksProcessProviderDetailsSafety:
    """Test safety_identifier handling in provider details."""

    def _make_model(self) -> DatabricksModel:
        provider = DatabricksProvider(api_key='test', base_url='https://test.com')
        return DatabricksModel('test-model', provider=provider)

    def test_safety_identifier_captured(self):
        """safety_identifier is captured when present on the response."""
        model = self._make_model()
        response = ChatCompletion(
            id='test',
            choices=[
                Choice(
                    index=0,
                    finish_reason='stop',
                    message=ChatCompletionMessage(role='assistant', content='hi'),
                )
            ],
            created=0,
            model='test',
            object='chat.completion',
            usage={'prompt_tokens': 5, 'completion_tokens': 10, 'total_tokens': 15},  # type: ignore
        )
        # Monkey-patch safety_identifier onto the response
        response.safety_identifier = 'safety-123'  # type: ignore
        details = model._process_provider_details(response)
        assert details is not None
        assert details['safety_identifier'] == 'safety-123'


class TestDatabricksStreamedResponseMapPartDelta:
    """Test _map_part_delta with structured content."""

    def _make_stream_response(self) -> DatabricksStreamedResponse:
        from pydantic_ai.models import ModelResponsePartsManager

        sr = object.__new__(DatabricksStreamedResponse)
        sr._internal_provider_details = None
        sr._usage = RequestUsage()
        sr._parts_manager = ModelResponsePartsManager()
        sr._provider_name = 'databricks'
        return sr

    def test_map_part_delta_with_text_block(self):
        """Test _map_part_delta with list content containing a text block."""
        from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta

        sr = self._make_stream_response()
        # Create a choice with list content
        delta = ChoiceDelta(role='assistant', content=None)
        choice = ChunkChoice(index=0, delta=delta, finish_reason=None)
        # Monkey-patch content to be a list
        choice.delta.content = [{'type': 'text', 'text': 'hello'}]  # type: ignore
        events = list(sr._map_part_delta(choice))
        assert len(events) > 0

    def test_map_part_delta_with_reasoning_block(self):
        """Test _map_part_delta with list content containing a reasoning block."""
        from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta

        sr = self._make_stream_response()
        delta = ChoiceDelta(role='assistant', content=None)
        choice = ChunkChoice(index=0, delta=delta, finish_reason=None)
        choice.delta.content = [{'type': 'reasoning', 'text': 'thinking...'}]  # type: ignore
        events = list(sr._map_part_delta(choice))
        assert len(events) > 0

    def test_map_part_delta_with_mixed_blocks(self):
        """Test _map_part_delta with list content containing both text and reasoning."""
        from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta

        sr = self._make_stream_response()
        delta = ChoiceDelta(role='assistant', content=None)
        choice = ChunkChoice(index=0, delta=delta, finish_reason=None)
        choice.delta.content = [  # type: ignore
            {'type': 'reasoning', 'text': 'thinking...'},
            {'type': 'text', 'text': 'The answer is 4.'},
        ]
        events = list(sr._map_part_delta(choice))
        assert len(events) >= 2  # At least one for reasoning and one for text

    def test_map_part_delta_with_empty_reasoning_block(self):
        """Reasoning block with empty get_value() produces no events."""
        from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta

        sr = self._make_stream_response()
        delta = ChoiceDelta(role='assistant', content=None)
        choice = ChunkChoice(index=0, delta=delta, finish_reason=None)
        choice.delta.content = [{'type': 'reasoning'}]  # type: ignore
        events = list(sr._map_part_delta(choice))
        assert len(events) == 0
