import datetime
import json
from collections.abc import Sequence
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel
from vcr.cassette import Cassette

from pydantic_ai import (
    Agent,
    BinaryContent,
    CachePoint,
    DocumentUrl,
    ModelHTTPError,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartEndEvent,
    PartStartEvent,
    RunUsage,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolDefinition,
    UnexpectedModelBehavior,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai.direct import model_request, model_request_stream
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ImageUrl, InstructionPart, SystemPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.usage import RequestUsage

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime, IsStr, try_import

with try_import() as imports_successful:
    from openai.types import chat
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion import Choice

    from pydantic_ai.models.openrouter import (
        OpenRouterModel,
        OpenRouterModelSettings,
        _map_openrouter_provider_details,  # pyright: ignore[reportPrivateUsage]
        _openrouter_settings_to_openai_settings,  # pyright: ignore[reportPrivateUsage]
        _OpenRouterChatCompletion,  # pyright: ignore[reportPrivateUsage]
        _OpenRouterChatCompletionChunk,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers.openrouter import OpenRouterModelProfile, OpenRouterProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


async def test_openrouter_with_preset(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', provider=provider)
    settings = OpenRouterModelSettings(openrouter_preset='@preset/comedian')
    response = await model_request(model, [ModelRequest.user_text_prompt('Trains')], model_settings=settings)
    text_part = cast(TextPart, response.parts[0])
    assert text_part.content == snapshot(
        """\
Why did the train break up with the track?

Because it felt like their relationship was going nowhere.\
"""
    )


async def test_openrouter_with_native_options(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.0-flash-exp:free', provider=provider)
    # These specific settings will force OpenRouter to use the fallback model, since Gemini is not available via the xAI provider.
    settings = OpenRouterModelSettings(
        openrouter_models=['x-ai/grok-4'],
        openrouter_transforms=['middle-out'],
        openrouter_provider={'only': ['xai']},
    )
    response = await model_request(model, [ModelRequest.user_text_prompt('Who are you')], model_settings=settings)
    text_part = cast(TextPart, response.parts[0])
    assert text_part.content == snapshot(
        """\
I'm Grok, a helpful and maximally truthful AI built by xAI. I'm not based on any other companies' models—instead, I'm inspired by the Hitchhiker's Guide to the Galaxy and JARVIS from Iron Man. My goal is to assist with questions, provide information, and maybe crack a joke or two along the way.

What can I help you with today?\
"""
    )
    assert response.provider_details is not None
    assert response.provider_details['downstream_provider'] == 'xAI'
    assert response.provider_details['finish_reason'] == 'stop'


async def test_openrouter_stream_with_native_options(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.0-flash-exp:free', provider=provider)
    # These specific settings will force OpenRouter to use the fallback model, since Gemini is not available via the xAI provider.
    settings = OpenRouterModelSettings(
        openrouter_models=['x-ai/grok-4'],
        openrouter_transforms=['middle-out'],
        openrouter_provider={'only': ['xai']},
    )

    async with model_request_stream(
        model, [ModelRequest.user_text_prompt('Who are you')], model_settings=settings
    ) as stream:
        assert stream.provider_details == snapshot(None)
        assert stream.finish_reason == snapshot(None)

        _ = [chunk async for chunk in stream]

        assert stream.provider_details is not None
        assert stream.provider_details == snapshot(
            {
                'timestamp': datetime.datetime(2025, 11, 2, 6, 14, 57, tzinfo=datetime.timezone.utc),
                'finish_reason': 'completed',
                'cost': 0.00333825,
                'upstream_inference_cost': None,
                'is_byok': False,
                'downstream_provider': 'xAI',
            }
        )
        # Explicitly verify native_finish_reason is 'completed' and wasn't overwritten by the
        # final usage chunk (which has native_finish_reason: null, see cassette for details)
        assert stream.provider_details['finish_reason'] == 'completed'
        assert stream.finish_reason == snapshot('stop')


async def test_openrouter_stream_with_reasoning(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel(
        'openai/o3',
        provider=provider,
        settings=OpenRouterModelSettings(openrouter_reasoning={'effort': 'high'}),
    )

    async with model_request_stream(model, [ModelRequest.user_text_prompt('Who are you')]) as stream:
        chunks = [chunk async for chunk in stream]

        thinking_event_start = chunks[0]
        assert isinstance(thinking_event_start, PartStartEvent)
        thinking_part = thinking_event_start.part
        assert isinstance(thinking_part, ThinkingPart)
        assert thinking_part.id == 'rs_0aa4f2c435e6d1dc0169082486816c8193a029b5fc4ef1764f'
        assert thinking_part.content == ''
        assert thinking_part.provider_name == 'openrouter'
        # After fix: signature and provider_details are now properly preserved
        assert thinking_part.signature is not None
        assert thinking_part.provider_details is not None
        assert thinking_part.provider_details['type'] == 'reasoning.encrypted'
        assert thinking_part.provider_details['format'] == 'openai-responses-v1'

        thinking_event_end = chunks[1]
        assert isinstance(thinking_event_end, PartEndEvent)
        thinking_part_end = thinking_event_end.part
        assert isinstance(thinking_part_end, ThinkingPart)
        assert thinking_part_end.id == 'rs_0aa4f2c435e6d1dc0169082486816c8193a029b5fc4ef1764f'
        assert thinking_part_end.signature is not None


async def test_openrouter_stream_error(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('minimax/minimax-m2:free', provider=provider)
    settings = OpenRouterModelSettings(max_tokens=10)

    with pytest.raises(ModelHTTPError):
        async with model_request_stream(
            model, [ModelRequest.user_text_prompt('Hello there')], model_settings=settings
        ) as stream:
            _ = [chunk async for chunk in stream]


async def test_openrouter_tool_calling(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)

    class Divide(BaseModel):
        """Divide two numbers."""

        numerator: float
        denominator: float
        on_inf: Literal['error', 'infinity'] = 'infinity'

    model = OpenRouterModel('mistralai/mistral-small', provider=provider)
    response = await model_request(
        model,
        [ModelRequest.user_text_prompt('What is 123 / 456?')],
        model_request_parameters=ModelRequestParameters(
            function_tools=[
                ToolDefinition(
                    name=Divide.__name__.lower(),
                    description=Divide.__doc__,
                    parameters_json_schema=Divide.model_json_schema(),
                )
            ],
            allow_text_output=True,  # Allow model to either use tools or respond directly
        ),
    )

    assert len(response.parts) == 1

    tool_call_part = response.parts[0]
    assert isinstance(tool_call_part, ToolCallPart)
    assert tool_call_part.tool_call_id == snapshot('3sniiMddS')
    assert tool_call_part.tool_name == 'divide'
    assert tool_call_part.args == snapshot('{"numerator": 123, "denominator": 456, "on_inf": "infinity"}')

    mapped_messages = await model._map_messages([response], ModelRequestParameters())  # type: ignore[reportPrivateUsage]
    tool_call_message = mapped_messages[0]
    assert tool_call_message['role'] == 'assistant'
    assert tool_call_message.get('content') is None
    assert tool_call_message.get('tool_calls') == snapshot(
        [
            {
                'id': '3sniiMddS',
                'type': 'function',
                'function': {
                    'name': 'divide',
                    'arguments': '{"numerator": 123, "denominator": 456, "on_inf": "infinity"}',
                },
            }
        ]
    )


async def test_openrouter_with_reasoning(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    request = ModelRequest.user_text_prompt(
        "What was the impact of Voltaire's writings on modern french culture? Think about your answer."
    )

    model = OpenRouterModel('z-ai/glm-4.6', provider=provider)
    response = await model_request(model, [request])

    assert len(response.parts) == 2

    thinking_part = response.parts[0]
    assert isinstance(thinking_part, ThinkingPart)
    assert thinking_part.id == snapshot(None)
    assert thinking_part.content is not None
    assert thinking_part.signature is None


async def test_openrouter_preserve_reasoning_block(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('openai/gpt-5-mini', provider=provider)

    messages: Sequence[ModelMessage] = []
    messages.append(ModelRequest.user_text_prompt('Hello!'))
    messages.append(await model_request(model, messages))
    messages.append(
        ModelRequest.user_text_prompt("What was the impact of Voltaire's writings on modern french culture?")
    )
    messages.append(await model_request(model, messages))

    openai_messages = await model._map_messages(messages, ModelRequestParameters())  # type: ignore[reportPrivateUsage]

    assistant_message = openai_messages[1]
    assert assistant_message['role'] == 'assistant'
    assert 'reasoning_details' not in assistant_message

    assistant_message = openai_messages[3]
    assert assistant_message['role'] == 'assistant'
    assert 'reasoning_details' in assistant_message

    reasoning_details = assistant_message['reasoning_details']
    assert len(reasoning_details) == 2

    reasoning_summary = reasoning_details[0]

    assert 'summary' in reasoning_summary
    assert reasoning_summary['type'] == 'reasoning.summary'
    assert reasoning_summary['format'] == 'openai-responses-v1'

    reasoning_encrypted = reasoning_details[1]

    assert 'data' in reasoning_encrypted
    assert reasoning_encrypted['type'] == 'reasoning.encrypted'
    assert reasoning_encrypted['format'] == 'openai-responses-v1'


async def test_openrouter_video_url_mapping() -> None:
    provider = OpenRouterProvider(api_key='test-key')
    model = OpenRouterModel('google/gemini-3-flash-preview', provider=provider)

    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'Count the students.',
                        VideoUrl(url='https://example.com/video.mp4'),
                    ]
                )
            ]
        )
    ]

    mapped_messages = await model._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    content = mapped_messages[0].get('content')
    assert content is not None
    assert isinstance(content, list)

    assert content[0] == {'type': 'text', 'text': 'Count the students.'}
    assert content[1] == {'type': 'video_url', 'video_url': {'url': 'https://example.com/video.mp4'}}


async def test_openrouter_binary_content_video_mapping() -> None:
    """Test that `BinaryContent` with a video media type maps to a `video_url` part."""
    provider = OpenRouterProvider(api_key='test-key')
    model = OpenRouterModel('google/gemini-3-flash-preview', provider=provider)

    binary_video = BinaryContent(data=b'video-bytes', media_type='video/mp4')

    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'Count the students.',
                        binary_video,
                    ]
                )
            ]
        )
    ]

    mapped_messages = await model._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    content = mapped_messages[0].get('content')
    assert content is not None
    assert isinstance(content, list)

    assert content[0] == {'type': 'text', 'text': 'Count the students.'}
    assert content[1] == {
        'type': 'video_url',
        'video_url': {'url': binary_video.data_uri},
    }


async def test_openrouter_video_url_force_download() -> None:
    provider = OpenRouterProvider(api_key='test-key')
    model = OpenRouterModel('google/gemini-3-flash-preview', provider=provider)

    with patch('pydantic_ai.models.openrouter.download_item', new_callable=AsyncMock) as mock_download:
        mock_download.return_value = {
            'data': 'data:video/mp4;base64,AAAA',
            'data_type': 'mp4',
        }

        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Count the students.',
                            VideoUrl(url='https://example.com/video.mp4', force_download=True),
                        ]
                    )
                ]
            )
        ]

        mapped_messages = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
            messages, ModelRequestParameters()
        )
        content = mapped_messages[0].get('content')
        assert content is not None
        assert isinstance(content, list)

        assert content[1] == {'type': 'video_url', 'video_url': {'url': 'data:video/mp4;base64,AAAA'}}
        mock_download.assert_called_once()
        call_args = mock_download.call_args
        assert call_args[0][0].url == 'https://example.com/video.mp4'
        assert call_args[1]['data_format'] == 'base64_uri'
        assert call_args[1]['type_format'] == 'extension'


async def test_openrouter_video_url_no_force_download() -> None:
    """Test that `force_download=False` does not call `download_item` for `VideoUrl`."""
    provider = OpenRouterProvider(api_key='test-key')
    model = OpenRouterModel('google/gemini-3-flash-preview', provider=provider)

    with patch('pydantic_ai.models.openrouter.download_item', new_callable=AsyncMock) as mock_download:
        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Count the students.',
                            VideoUrl(url='https://example.com/video.mp4', force_download=False),
                        ]
                    )
                ]
            )
        ]

        mapped_messages = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
            messages, ModelRequestParameters()
        )
        content = mapped_messages[0].get('content')
        assert content is not None
        assert isinstance(content, list)

        assert content[1] == {'type': 'video_url', 'video_url': {'url': 'https://example.com/video.mp4'}}
        mock_download.assert_not_called()


async def test_openrouter_video_url_public_api(
    allow_model_requests: None, openrouter_api_key: str
) -> None:  # pragma: lax no cover
    """Test `VideoUrl` support through the public `Agent.run` API."""
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.5-flash', provider=provider)
    agent = Agent(model)

    result = await agent.run(
        [
            'What is in this video?',
            VideoUrl(url='https://upload.wikimedia.org/wikipedia/commons/8/8f/Panda_at_Smithsonian_zoo.webm'),
        ]
    )

    assert isinstance(result.output, str)
    assert result.output == snapshot("""\
This video features a giant panda in an enclosure designed to resemble its natural habitat. The enclosure includes:
- **Rocks and terrain:** Various sized rocks create a textured landscape.
- **Bamboo:** Fresh bamboo shoots are scattered around, which the panda is seen eating.
- **Background mural:** A painted mural on the back wall depicts a mountainous, green landscape, enhancing the immersive feel of the habitat.
- **Window:** A clear window is visible in the upper part of the background, likely part of the viewing area for visitors.
- **Enrichment toy:** A large, round, light brown object (possibly a ball or feeder) is seen on the rocks, likely an enrichment toy for the panda.
- **Panda:** The main subject is a black and white giant panda, which is actively eating bamboo at the bottom right of the frame, occasionally looking up.\
""")


async def test_openrouter_binary_content_video_public_api(
    allow_model_requests: None, openrouter_api_key: str, video_content: BinaryContent, vcr: Cassette
) -> None:  # pragma: lax no cover
    """Test `BinaryContent` video support through the public `Agent.run` API."""
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.5-flash', provider=provider)
    agent = Agent(model)

    result = await agent.run(['What is in this video? Answer in one short sentence.', video_content])
    assert isinstance(result.output, str)
    assert result.output == snapshot(
        "The video shows a camera on a tripod recording a scenic mountain landscape, with a preview of the shot visible on the camera's screen."
    )

    assert vcr is not None
    assert len(vcr.requests) == 1  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    request_body = json.loads(vcr.requests[0].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]

    video_content_part = request_body['messages'][0]['content'][1]
    assert video_content_part['type'] == 'video_url'
    assert video_content_part['video_url']['url'].startswith('data:video/mp4;base64,')


async def test_openrouter_errors_raised(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.0-flash-exp:free', provider=provider)
    agent = Agent(model, instructions='Be helpful.', retries=1)
    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Tell me a joke.')
    assert str(exc_info.value) == snapshot(
        "status_code: 429, model_name: google/gemini-2.0-flash-exp:free, body: {'code': 429, 'message': 'Provider returned error', 'metadata': {'provider_name': 'Google', 'raw': 'google/gemini-2.0-flash-exp:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations'}}"
    )


async def test_openrouter_usage(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('openai/gpt-5-mini', provider=provider)
    agent = Agent(model, instructions='Be helpful.', retries=1)

    result = await agent.run('Tell me about Venus')

    assert result.usage() == snapshot(
        RunUsage(input_tokens=17, output_tokens=1515, details={'reasoning_tokens': 704}, requests=1)
    )

    settings = OpenRouterModelSettings(openrouter_usage={'include': True})

    result = await agent.run('Tell me about Mars', model_settings=settings)

    assert result.usage() == snapshot(
        RunUsage(
            input_tokens=17,
            output_tokens=2177,
            details={'is_byok': 0, 'reasoning_tokens': 960, 'image_tokens': 0},
            requests=1,
        )
    )

    last_message = result.all_messages()[-1]

    assert isinstance(last_message, ModelResponse)
    assert last_message.provider_details is not None
    for key in ['cost', 'upstream_inference_cost', 'is_byok']:
        assert key in last_message.provider_details


async def test_openrouter_validate_non_json_response(openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.0-flash-exp:free', provider=provider)

    with pytest.raises(UnexpectedModelBehavior) as exc_info:
        model._process_response('This is not JSON!')  # type: ignore[reportPrivateUsage]

    assert str(exc_info.value) == snapshot(
        'Invalid response from openrouter chat completions endpoint, expected JSON data'
    )


async def test_openrouter_validate_error_response(openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.0-flash-exp:free', provider=provider)

    choice = Choice.model_construct(
        index=0, message={'role': 'assistant'}, finish_reason='error', native_finish_reason='stop'
    )
    response = ChatCompletion.model_construct(
        id='', choices=[choice], created=0, object='chat.completion', model='test', provider='test'
    )
    response.error = {'message': 'This response has an error attribute', 'code': 200}  # type: ignore[reportAttributeAccessIssue]

    with pytest.raises(ModelHTTPError) as exc_info:
        model._process_response(response)  # type: ignore[reportPrivateUsage]

    assert str(exc_info.value) == snapshot(
        'status_code: 200, model_name: test, body: This response has an error attribute'
    )


async def test_openrouter_with_provider_details_but_no_parent_details(openrouter_api_key: str) -> None:
    class TestOpenRouterModel(OpenRouterModel):
        def _process_provider_details(self, response: ChatCompletion) -> dict[str, Any] | None:
            assert isinstance(response, _OpenRouterChatCompletion)
            openrouter_details = _map_openrouter_provider_details(response)
            return openrouter_details or None

    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = TestOpenRouterModel('google/gemini-2.0-flash-exp:free', provider=provider)

    choice = Choice.model_construct(
        index=0, message={'role': 'assistant', 'content': 'test'}, finish_reason='stop', native_finish_reason='stop'
    )
    response = ChatCompletion.model_construct(
        id='test', choices=[choice], created=1704067200, object='chat.completion', model='test', provider='TestProvider'
    )
    result = model._process_response(response)  # type: ignore[reportPrivateUsage]

    assert result.provider_details == snapshot(
        {
            'downstream_provider': 'TestProvider',
            'finish_reason': 'stop',
            'timestamp': datetime.datetime(2024, 1, 1, 0, 0, tzinfo=datetime.timezone.utc),
        }
    )


async def test_openrouter_map_messages_reasoning(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('anthropic/claude-3.7-sonnet:thinking', provider=provider)

    user_message = ModelRequest.user_text_prompt('Who are you. Think about it.')
    response = await model_request(model, [user_message])

    mapped_messages = await model._map_messages([user_message, response], ModelRequestParameters())  # type: ignore[reportPrivateUsage]

    assert len(mapped_messages) == 2
    assert mapped_messages[1]['reasoning_details'] == snapshot(  # type: ignore[reportGeneralTypeIssues]
        [
            {
                'id': None,
                'type': 'reasoning.text',
                'text': """\
This question is asking me about my identity. Let me think about how to respond clearly and accurately.

I am Claude, an AI assistant created by Anthropic. I'm designed to be helpful, harmless, and honest in my interactions with humans. I don't have a physical form - I exist as a large language model running on computer hardware. I don't have consciousness, sentience, or feelings in the way humans do. I don't have personal experiences or a life outside of these conversations.

My capabilities include understanding and generating natural language text, reasoning about various topics, and attempting to be helpful to users in a wide range of contexts. I have been trained on a large corpus of text data, but my training data has a cutoff date, so I don't have knowledge of events that occurred after my training.

I have certain limitations - I don't have the ability to access the internet, run code, or interact with external systems unless given specific tools to do so. I don't have perfect knowledge and can make mistakes.

I'm designed to be conversational and to engage with users in a way that's helpful and informative, while respecting important ethical boundaries.\
""",
                'signature': 'ErcBCkgICBACGAIiQHtMxpqcMhnwgGUmSDWGoOL9ZHTbDKjWnhbFm0xKzFl0NmXFjQQxjFj5mieRYY718fINsJMGjycTVYeiu69npakSDDrsnKYAD/fdcpI57xoMHlQBxI93RMa5CSUZIjAFVCMQF5GfLLQCibyPbb7LhZ4kLIFxw/nqsTwDDt6bx3yipUcq7G7eGts8MZ6LxOYqHTlIDx0tfHRIlkkcNCdB2sUeMqP8e7kuQqIHoD52GAI=',
                'format': 'anthropic-claude-v1',
                'index': 0,
            }
        ]
    )


async def test_openrouter_tool_optional_parameters(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)

    class FindEducationContentFilters(BaseModel):
        title: str | None = None

    model = OpenRouterModel('anthropic/claude-sonnet-4.5', provider=provider)
    response = await model_request(
        model,
        [ModelRequest.user_text_prompt('Can you find me any education content?')],
        model_request_parameters=ModelRequestParameters(
            function_tools=[
                ToolDefinition(
                    name='find_education_content',
                    description='',
                    parameters_json_schema=FindEducationContentFilters.model_json_schema(),
                )
            ],
            allow_text_output=True,  # Allow model to either use tools or respond directly
        ),
    )

    assert len(response.parts) == 2

    tool_call_part = response.parts[1]
    assert isinstance(tool_call_part, ToolCallPart)
    assert tool_call_part.tool_call_id == snapshot('toolu_vrtx_015QAXScZzRDPttiPoc34AdD')
    assert tool_call_part.tool_name == 'find_education_content'
    assert tool_call_part.args == snapshot(None)

    mapped_messages = await model._map_messages([response], ModelRequestParameters())  # type: ignore[reportPrivateUsage]
    tool_call_message = mapped_messages[0]
    assert tool_call_message['role'] == 'assistant'
    assert tool_call_message.get('content') == snapshot("I'll search for education content for you.")
    assert tool_call_message.get('tool_calls') == snapshot(
        [
            {
                'id': 'toolu_vrtx_015QAXScZzRDPttiPoc34AdD',
                'type': 'function',
                'function': {
                    'name': 'find_education_content',
                    'arguments': '{}',
                },
            }
        ]
    )


async def test_openrouter_streaming_reasoning(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('anthropic/claude-sonnet-4.5', provider=provider)
    agent = Agent(
        model=model,
        model_settings=OpenRouterModelSettings(openrouter_reasoning={'enabled': True}),
    )

    async with agent.run_stream('What is 2+2?') as stream:
        _ = await stream.get_output()

        assert stream.response.parts == snapshot(
            [
                ThinkingPart(
                    content='This is a simple arithmetic question. 2+2 equals 4.',
                    signature='Et0BCkgIChACGAIqQA2s7h7tA7IG35fbwVkou9PM2hANVJNUwcEM4q12fTRDK6y3v6YoEvJ+7bko8wnW/GLsQFXadaJPAEMCpLkhI9ISDLjFkeR1aVUIvdCtyBoMrUTovh0jwk+wpnZWIjANV3e6VVdgbGSsEyyTHO6KMmVtqqs79f9blnVdJmmMIwMyTi6bEtG59+jTU7v1zlsqQ2IKGZILOlr6adh0Aam7zYttvisys+wjyZZXU1y/Srz0nmp1cFgVOJe1BLKQI3SSRrjsqQC0uAEUZy0GX0Rq1AXjvIcYAQ==',
                    provider_name='openrouter',
                    provider_details={'format': 'anthropic-claude-v1', 'index': 0, 'type': 'reasoning.text'},
                ),
                TextPart(content='2 + 2 = 4'),
            ]
        )


async def test_openrouter_no_openrouter_details(openrouter_api_key: str) -> None:
    """Test _process_provider_details when _map_openrouter_provider_details returns empty dict."""
    from unittest.mock import patch

    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.0-flash-exp:free', provider=provider)

    choice = Choice.model_construct(
        index=0, message={'role': 'assistant', 'content': 'test'}, finish_reason='stop', native_finish_reason='stop'
    )
    response = ChatCompletion.model_construct(
        id='test', choices=[choice], created=1704067200, object='chat.completion', model='test', provider='TestProvider'
    )

    with patch('pydantic_ai.models.openrouter._map_openrouter_provider_details', return_value={}):
        result = model._process_response(response)  # type: ignore[reportPrivateUsage]

    # With empty openrouter_details, we should still get the parent's provider_details (timestamp + finish_reason)
    assert result.provider_details == snapshot(
        {'finish_reason': 'stop', 'timestamp': datetime.datetime(2024, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)}
    )


async def test_openrouter_google_nested_schema(allow_model_requests: None, openrouter_api_key: str) -> None:
    """Test that nested schemas with $defs/$ref work correctly with OpenRouter + Gemini.

    This verifies the fix for https://github.com/pydantic/pydantic-ai/issues/3617
    where OpenRouter's translation layer didn't support modern JSON Schema features.
    """
    from enum import Enum

    provider = OpenRouterProvider(api_key=openrouter_api_key)

    class LevelType(str, Enum):
        ground = 'ground'
        basement = 'basement'
        floor = 'floor'
        attic = 'attic'

    class SpaceType(str, Enum):
        entryway = 'entryway'
        living_room = 'living-room'
        kitchen = 'kitchen'
        bedroom = 'bedroom'
        bathroom = 'bathroom'
        garage = 'garage'

    class InsertLevelArg(BaseModel):
        level_name: str
        level_type: LevelType

    class SpaceArg(BaseModel):
        space_name: str
        space_type: SpaceType

    class InsertedLevel(BaseModel):
        """Result of inserting a level."""

        level_name: str
        level_type: LevelType
        space_count: int

    model = OpenRouterModel('google/gemini-2.5-flash', provider=provider)
    agent: Agent[None, InsertedLevel] = Agent(model, output_type=InsertedLevel)

    @agent.tool_plain
    def insert_level_with_spaces(level: InsertLevelArg | None, spaces: list[SpaceArg]) -> str:
        """Insert a level with its spaces."""
        return f'Inserted level {level} with {len(spaces)} spaces'

    result = await agent.run("It's a house with a ground floor that has an entryway, a living room and a garage.")

    tool_call_message = result.all_messages()[1]
    assert tool_call_message.parts == snapshot(
        [
            ToolCallPart(
                tool_name='insert_level_with_spaces',
                args='{"spaces":[{"space_type":"entryway","space_name":"entryway"},{"space_name":"living_room","space_type":"living-room"},{"space_name":"garage","space_type":"garage"}],"level":{"level_type":"ground","level_name":"ground_floor"}}',
                tool_call_id='tool_insert_level_with_spaces_3ZiChYzj8xER8HixJe7W',
            )
        ]
    )

    assert result.output.level_type == LevelType.ground
    assert result.output.space_count == 3


async def test_openrouter_file_annotation(
    allow_model_requests: None, openrouter_api_key: str, document_content: BinaryContent
) -> None:
    """Test that file annotations from OpenRouter are handled correctly.

    When sending files (e.g., PDFs) to OpenRouter, the response can include
    annotations with type="file". This test ensures those annotations are
    parsed without validation errors.
    """
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('openai/gpt-5.1-codex-mini', provider=provider)
    agent = Agent(model)

    result = await agent.run(
        user_prompt=[
            'What does this PDF contain? Answer in one short sentence.',
            document_content,
        ]
    )

    # The response should contain text (model may or may not include file annotations)
    assert isinstance(result.output, str)
    assert len(result.output) > 0


async def test_openrouter_file_annotation_validation(openrouter_api_key: str) -> None:
    """Test that file annotations from OpenRouter are correctly validated.

    This unit test verifies that responses containing type="file" annotations
    are parsed without validation errors, which was failing before the fix.
    """
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('openai/gpt-4.1-mini', provider=provider)

    message = ChatCompletionMessage.model_construct(
        role='assistant',
        content='Here is the summary of your file.',
        annotations=[
            {'type': 'file', 'file': {'filename': 'test.pdf', 'file_id': 'file-123'}},
        ],
    )
    choice = Choice.model_construct(index=0, message=message, finish_reason='stop', native_finish_reason='stop')
    response = ChatCompletion.model_construct(
        id='test', choices=[choice], created=0, object='chat.completion', model='test', provider='test'
    )

    # This should not raise a validation error
    result = model._process_response(response)  # type: ignore[reportPrivateUsage]
    text_part = cast(TextPart, result.parts[0])
    assert text_part.content == 'Here is the summary of your file.'


async def test_openrouter_url_citation_annotation_validation(openrouter_api_key: str) -> None:
    """Test that url_citation annotations from OpenRouter are correctly validated."""
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('openai/gpt-4.1-mini', provider=provider)

    message = ChatCompletionMessage.model_construct(
        role='assistant',
        content='According to the source, this is the answer.',
        annotations=[
            {
                'type': 'url_citation',
                'url_citation': {'url': 'https://example.com', 'title': 'Example', 'start_index': 0, 'end_index': 10},
            },
        ],
    )
    choice = Choice.model_construct(index=0, message=message, finish_reason='stop', native_finish_reason='stop')
    response = ChatCompletion.model_construct(
        id='test', choices=[choice], created=0, object='chat.completion', model='test', provider='test'
    )

    # This should not raise a validation error
    result = model._process_response(response)  # type: ignore[reportPrivateUsage]
    text_part = cast(TextPart, result.parts[0])
    assert text_part.content == 'According to the source, this is the answer.'


async def test_openrouter_service_tier_completion(openrouter_api_key: str) -> None:
    """OpenRouter providers can return service_tier values outside the OpenAI Literal."""
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.5-flash', provider=provider)

    message = ChatCompletionMessage.model_construct(role='assistant', content='hi')
    choice = Choice.model_construct(index=0, message=message, finish_reason='stop', native_finish_reason='stop')
    response = ChatCompletion.model_construct(
        id='gen-123',
        choices=[choice],
        created=1234567890,
        object='chat.completion',
        model='google/gemini-2.5-flash',
        provider='Google',
        service_tier='standard',
    )

    result = model._process_response(response)  # type: ignore[reportPrivateUsage]
    text_part = cast(TextPart, result.parts[0])
    assert text_part.content == 'hi'


async def test_openrouter_service_tier_chunk() -> None:
    """OpenRouter streaming chunks can return service_tier values outside the OpenAI Literal."""
    data = {
        'id': 'gen-123',
        'choices': [
            {
                'index': 0,
                'delta': {'role': 'assistant', 'content': 'hi'},
                'finish_reason': 'stop',
                'native_finish_reason': 'stop',
            }
        ],
        'created': 1234567890,
        'model': 'google/gemini-2.5-flash',
        'object': 'chat.completion.chunk',
        'provider': 'Google',
        'service_tier': 'on_demand',
    }
    result = _OpenRouterChatCompletionChunk.model_validate(data)
    assert result.service_tier == 'on_demand'


async def test_openrouter_document_url_no_force_download(
    allow_model_requests: None, openrouter_api_key: str, vcr: Cassette
) -> None:
    """Test that OpenRouter passes DocumentUrl directly without downloading when force_download=False.

    OpenRouter supports file URLs directly in the Chat API, unlike native OpenAI which only
    supports base64-encoded data. This test verifies that when using OpenRouter, the URL
    is passed directly without being downloaded first.
    """
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('openai/gpt-4.1-mini', provider=provider)
    agent = Agent(model)

    pdf_url = 'https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf'
    document_url = DocumentUrl(url=pdf_url, force_download=False)

    result = await agent.run(['What is the main content of this document?', document_url])
    assert 'dummy' in result.output.lower() or 'pdf' in result.output.lower()

    # Verify URL was passed directly (not downloaded and base64-encoded)
    assert vcr is not None
    assert len(vcr.requests) == 1, 'Should only have one request (to OpenRouter, not a download)'  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    request_body = json.loads(vcr.requests[0].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    file_content = request_body['messages'][0]['content'][1]
    assert file_content == snapshot(
        {
            'file': {
                'file_data': 'https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf',
                'filename': 'filename.pdf',
            },
            'type': 'file',
        }
    )


async def test_openrouter_supported_builtin_tools() -> None:
    """Test that OpenRouterModel declares support for WebSearchTool."""
    supported = OpenRouterModel.supported_builtin_tools()
    assert WebSearchTool in supported


async def test_openrouter_web_search_prepare_request(openrouter_api_key: str) -> None:
    """Test that prepare_request injects web search plugins when WebSearchTool is present."""

    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('openai/gpt-4.1', provider=provider)

    model_request_parameters = ModelRequestParameters(
        builtin_tools=[WebSearchTool(search_context_size='high')],
    )

    new_settings, _ = model.prepare_request(None, model_request_parameters)

    assert new_settings is not None
    extra_body = cast(dict[str, Any], new_settings.get('extra_body', {}))
    assert 'plugins' in extra_body
    assert extra_body['plugins'] == [{'id': 'web'}]
    assert 'web_search_options' in extra_body
    assert extra_body['web_search_options'] == {'search_context_size': 'high'}


async def test_openrouter_no_web_search_without_tool(openrouter_api_key: str) -> None:
    """Test that no plugins are added when WebSearchTool is not present."""

    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('openai/gpt-4.1', provider=provider)

    model_request_parameters = ModelRequestParameters()

    new_settings, _ = model.prepare_request(None, model_request_parameters)

    assert new_settings is not None
    extra_body = cast(dict[str, Any], new_settings.get('extra_body', {}))
    assert 'plugins' not in extra_body
    assert 'web_search_options' not in extra_body


async def test_openrouter_settings_to_openai_settings_with_web_search() -> None:
    """Test _openrouter_settings_to_openai_settings when WebSearchTool is configured."""
    settings = OpenRouterModelSettings()
    model_request_parameters = ModelRequestParameters(
        builtin_tools=[WebSearchTool(search_context_size='high')],
    )

    result = _openrouter_settings_to_openai_settings(settings, model_request_parameters)

    extra_body = cast(dict[str, Any], result.get('extra_body', {}))
    assert 'plugins' in extra_body
    assert extra_body['plugins'] == [{'id': 'web'}]
    assert 'web_search_options' in extra_body
    assert extra_body['web_search_options'] == {'search_context_size': 'high'}


async def test_openrouter_prepare_request_loop_with_non_websearch_first(openrouter_api_key: str) -> None:
    """Test prepare_request loop continuation when first tool is not WebSearchTool."""
    from unittest.mock import Mock

    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('openai/gpt-4.1', provider=provider)

    non_web_tool = Mock(spec=[])
    web_tool = WebSearchTool(search_context_size='medium')

    model_request_parameters = ModelRequestParameters(
        builtin_tools=[non_web_tool, web_tool],
    )

    with patch.object(model.__class__.__bases__[0], 'prepare_request', return_value=({}, model_request_parameters)):
        new_settings, _ = model.prepare_request(None, model_request_parameters)

    assert new_settings is not None
    extra_body = cast(dict[str, Any], new_settings.get('extra_body', {}))
    assert 'plugins' in extra_body
    assert extra_body['plugins'] == [{'id': 'web'}]
    assert extra_body['web_search_options'] == {'search_context_size': 'medium'}


# ===== CachePoint in user messages =====


async def test_openrouter_cache_point_multiple_markers() -> None:
    """Test multiple CachePoint markers in a single prompt, including custom TTL."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=['First chunk', CachePoint(), 'Second chunk', CachePoint(ttl='1h'), 'Question'],
                )
            ]
        )
    ]

    mapped = await model._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    content = mapped[0].get('content')
    assert isinstance(content, list)
    assert content == snapshot(
        [
            {'type': 'text', 'text': 'First chunk', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}},
            {'type': 'text', 'text': 'Second chunk', 'cache_control': {'type': 'ephemeral', 'ttl': '1h'}},
            {'type': 'text', 'text': 'Question'},
        ]
    )


async def test_openrouter_cache_point_gemini_omits_ttl() -> None:
    """Test that CachePoint omits TTL for Gemini models."""
    model = OpenRouterModel('google/gemini-2.5-flash', provider=OpenRouterProvider(api_key='test-key'))

    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=['Context', CachePoint(), 'Question'])])]

    mapped = await model._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    content = mapped[0].get('content')
    assert isinstance(content, list)
    assert content == snapshot(
        [
            {'type': 'text', 'text': 'Context', 'cache_control': {'type': 'ephemeral'}},
            {'type': 'text', 'text': 'Question'},
        ]
    )


async def test_openrouter_cache_point_tilde_providers_use_downstream_cache_capabilities() -> None:
    """Test OpenRouter ``~provider`` aliases use the same cache controls as their downstream provider."""
    anthropic_model = OpenRouterModel(
        '~anthropic/claude-sonnet-latest', provider=OpenRouterProvider(api_key='test-key')
    )
    google_model = OpenRouterModel('~google/gemini-pro-latest', provider=OpenRouterProvider(api_key='test-key'))
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=['Context', CachePoint(ttl='1h'), 'Question'])])
    ]

    anthropic_mapped = await anthropic_model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, ModelRequestParameters()
    )
    google_mapped = await google_model._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]

    anthropic_content = anthropic_mapped[0].get('content')
    google_content = google_mapped[0].get('content')
    assert isinstance(anthropic_content, list)
    assert isinstance(google_content, list)
    assert cast(dict[str, Any], anthropic_content[0])['cache_control'] == {'type': 'ephemeral', 'ttl': '1h'}
    assert cast(dict[str, Any], google_content[0])['cache_control'] == {'type': 'ephemeral'}


async def test_openrouter_cache_point_first_content_raises_error(allow_model_requests: None) -> None:
    """Test that CachePoint as first content raises UserError."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))
    agent = Agent(model)

    with pytest.raises(
        UserError,
        match='CachePoint cannot be the first content in a user message - there must be previous content',
    ):
        await agent.run([CachePoint(), 'This should fail'])


async def test_openrouter_cache_point_unsupported_provider_ignored() -> None:
    """Test that CachePoint is silently ignored for unsupported downstream providers."""
    model = OpenRouterModel('openai/gpt-5', provider=OpenRouterProvider(api_key='test-key'))

    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=['Context', CachePoint(), 'Question'])])]

    mapped = await model._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    content = mapped[0].get('content')
    assert isinstance(content, list)
    # No cache_control should be present — CachePoint silently skipped
    assert content == snapshot(
        [
            {'type': 'text', 'text': 'Context'},
            {'type': 'text', 'text': 'Question'},
        ]
    )


async def test_openrouter_cache_point_string_content_unchanged() -> None:
    """Test that plain string content is not converted to list format by CachePoint logic."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))

    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='Just a plain string')])]

    mapped = await model._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    content = mapped[0].get('content')
    # String content should remain a string, not converted to list
    assert content == 'Just a plain string'


async def test_openrouter_cache_point_with_image_content() -> None:
    """Test CachePoint attaches cache_control to a preceding image content part."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        ImageUrl(url='https://example.com/image.jpg'),
                        CachePoint(),
                        'What is in this image?',
                    ]
                )
            ]
        )
    ]

    mapped = await model._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    content = mapped[0].get('content')
    assert isinstance(content, list)
    assert content == snapshot(
        [
            {
                'type': 'image_url',
                'image_url': {'url': 'https://example.com/image.jpg'},
                'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
            },
            {'type': 'text', 'text': 'What is in this image?'},
        ]
    )


async def test_openrouter_cache_messages_no_duplicate_with_explicit_cache_point() -> None:
    """Test that cache_messages doesn't conflict with an explicit CachePoint on different parts."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))
    settings = OpenRouterModelSettings(openrouter_cache_messages=True)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=['Long context to cache', CachePoint(), 'Now the question'],
                )
            ]
        )
    ]

    mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, ModelRequestParameters(), model_settings=settings
    )
    content = mapped[-1].get('content')
    assert isinstance(content, list)
    # CachePoint tags 'Long context to cache', cache_messages tags 'Now the question'
    assert content == snapshot(
        [
            {'type': 'text', 'text': 'Long context to cache', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}},
            {'type': 'text', 'text': 'Now the question', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}},
        ]
    )


# ===== openrouter_cache_instructions =====


async def test_openrouter_cache_instructions_adds_cache_control() -> None:
    """Test that openrouter_cache_instructions adds cache_control to system prompt."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))
    settings = OpenRouterModelSettings(openrouter_cache_instructions=True)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='You are a helpful assistant.'),
                UserPromptPart(content='Hello'),
            ]
        )
    ]

    mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, ModelRequestParameters(), model_settings=settings
    )

    # Find the system message
    system_msg = next(m for m in mapped if m.get('role') in ('system', 'developer'))
    content = system_msg.get('content')
    assert isinstance(content, list)
    assert content == snapshot(
        [
            {
                'type': 'text',
                'text': 'You are a helpful assistant.',
                'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
            }
        ]
    )


async def test_openrouter_cache_instructions_gemini_omits_ttl() -> None:
    """Test that openrouter_cache_instructions omits TTL for Gemini models."""
    model = OpenRouterModel('google/gemini-2.5-flash', provider=OpenRouterProvider(api_key='test-key'))
    settings = OpenRouterModelSettings(openrouter_cache_instructions=True)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='You are a helpful assistant.'),
                UserPromptPart(content='Hello'),
            ]
        )
    ]

    mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, ModelRequestParameters(), model_settings=settings
    )

    system_msg = next(m for m in mapped if m.get('role') in ('system', 'developer'))
    content = system_msg.get('content')
    assert isinstance(content, list)
    assert cast(dict[str, Any], content[0])['cache_control'] == snapshot({'type': 'ephemeral'})


async def test_openrouter_cache_instructions_unsupported_provider_ignored() -> None:
    """Test that openrouter_cache_instructions is silently ignored for unsupported providers."""
    model = OpenRouterModel('openai/gpt-5', provider=OpenRouterProvider(api_key='test-key'))
    settings = OpenRouterModelSettings(openrouter_cache_instructions=True)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='You are a helpful assistant.'),
                UserPromptPart(content='Hello'),
            ]
        )
    ]

    mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, ModelRequestParameters(), model_settings=settings
    )

    system_msg = next(m for m in mapped if m.get('role') in ('system', 'developer'))
    content = system_msg.get('content')
    # Should remain a plain string, not converted to list with cache_control
    assert isinstance(content, str)


async def test_openrouter_cache_instructions_no_system_message() -> None:
    """Test that cache_instructions is a no-op when no system message is present."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))
    settings = OpenRouterModelSettings(openrouter_cache_instructions=True)

    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='Hello')])]

    mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, ModelRequestParameters(), model_settings=settings
    )

    # No system message, so no cache_control should be added anywhere
    assert len(mapped) == 1
    assert mapped[0].get('role') == 'user'


async def test_openrouter_cache_instructions_ignores_user_role_profile() -> None:
    """Test that cache_instructions ignores profiles that map instructions to user messages."""
    model = OpenRouterModel(
        'anthropic/claude-sonnet-4.6',
        provider=OpenRouterProvider(api_key='test-key'),
        profile=OpenRouterModelProfile(openai_system_prompt_role='user', openrouter_supports_cache_control=True),
    )
    settings = OpenRouterModelSettings(openrouter_cache_instructions=True)
    params = ModelRequestParameters(instruction_parts=[InstructionPart(content='Static instructions.', dynamic=False)])

    mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        [ModelRequest(parts=[UserPromptPart(content='Hello')])], params, model_settings=settings
    )

    for message in mapped:
        content = message.get('content')
        assert isinstance(content, str)
        assert 'cache_control' not in content


@pytest.mark.parametrize(
    ('messages', 'instruction_parts', 'expected_system_messages'),
    [
        (
            [ModelRequest(parts=[UserPromptPart(content='Hello')])],
            [
                InstructionPart(content='Static instructions.', dynamic=False),
                InstructionPart(content='Dynamic instructions.', dynamic=True),
            ],
            [
                {
                    'role': 'system',
                    'content': [
                        {
                            'type': 'text',
                            'text': 'Static instructions.',
                            'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
                        }
                    ],
                },
                {'role': 'system', 'content': 'Dynamic instructions.'},
            ],
        ),
        (
            [
                ModelRequest(
                    parts=[
                        SystemPromptPart(content='Static system prompt.'),
                        UserPromptPart(content='Hello'),
                    ]
                )
            ],
            [InstructionPart(content='Dynamic instructions.', dynamic=True)],
            [
                {
                    'role': 'system',
                    'content': [
                        {
                            'type': 'text',
                            'text': 'Static system prompt.',
                            'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
                        }
                    ],
                },
                {'role': 'system', 'content': 'Dynamic instructions.'},
            ],
        ),
        (
            [ModelRequest(parts=[UserPromptPart(content='Hello')])],
            [InstructionPart(content='Dynamic instructions.', dynamic=True)],
            [{'role': 'system', 'content': 'Dynamic instructions.'}],
        ),
    ],
)
async def test_openrouter_cache_instructions_skips_dynamic_instruction_blocks(
    messages: list[ModelMessage],
    instruction_parts: list[InstructionPart],
    expected_system_messages: list[dict[str, Any]],
) -> None:
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))
    settings = OpenRouterModelSettings(openrouter_cache_instructions=True)
    params = ModelRequestParameters(instruction_parts=instruction_parts)

    mapped = await model._map_messages(messages, params, model_settings=settings)  # pyright: ignore[reportPrivateUsage]

    system_messages = [dict(m) for m in mapped if m.get('role') == 'system']
    assert system_messages == expected_system_messages


# ===== openrouter_cache_messages =====


async def test_openrouter_cache_messages_adds_cache_control() -> None:
    """Test that openrouter_cache_messages adds cache_control to the last message."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))
    settings = OpenRouterModelSettings(openrouter_cache_messages=True)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='System prompt.'),
                UserPromptPart(content='User message'),
            ]
        )
    ]

    mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, ModelRequestParameters(), model_settings=settings
    )

    # Last message should have cache_control
    last_msg = mapped[-1]
    content = last_msg.get('content')
    assert isinstance(content, list)
    assert content[-1] == snapshot(
        {'type': 'text', 'text': 'User message', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}}
    )

    # System message should NOT have cache_control
    system_msg = next(m for m in mapped if m.get('role') in ('system', 'developer'))
    system_content = system_msg.get('content')
    assert isinstance(system_content, str)


async def test_openrouter_cache_messages_empty_content_no_crash() -> None:
    """Test that openrouter_cache_messages does not crash when last message has empty content list."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))
    settings = OpenRouterModelSettings(openrouter_cache_messages=True)

    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=[])])]

    mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, ModelRequestParameters(), model_settings=settings
    )

    # Should not crash — empty content is left as-is
    last_msg = mapped[-1]
    content = last_msg.get('content')
    assert isinstance(content, list)
    assert content == []


# ===== openrouter_cache_tool_definitions =====


async def test_openrouter_cache_tool_definitions_anthropic() -> None:
    """Test that openrouter_cache_tool_definitions adds cache_control to the last tool for Anthropic."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))
    settings = OpenRouterModelSettings(openrouter_cache_tool_definitions=True)

    params = ModelRequestParameters(
        function_tools=[
            ToolDefinition(
                name='tool_one', description='First tool', parameters_json_schema={'type': 'object', 'properties': {}}
            ),
            ToolDefinition(
                name='tool_two', description='Second tool', parameters_json_schema={'type': 'object', 'properties': {}}
            ),
        ],
        allow_text_output=True,
    )

    tools = model._get_tools(params, model_settings=settings)  # pyright: ignore[reportPrivateUsage]

    assert len(tools) == 2
    # First tool should NOT have cache_control
    first_tool = cast(dict[str, Any], tools[0])
    assert 'cache_control' not in first_tool
    # Last tool SHOULD have cache_control
    last_tool = cast(dict[str, Any], tools[1])
    assert last_tool['cache_control'] == snapshot({'type': 'ephemeral', 'ttl': '5m'})


async def test_openrouter_cache_tool_definitions_gemini_ignored() -> None:
    """Test that openrouter_cache_tool_definitions has no effect for Gemini models."""
    model = OpenRouterModel('google/gemini-2.5-flash', provider=OpenRouterProvider(api_key='test-key'))
    settings = OpenRouterModelSettings(openrouter_cache_tool_definitions=True)

    params = ModelRequestParameters(
        function_tools=[
            ToolDefinition(
                name='my_tool', description='A tool', parameters_json_schema={'type': 'object', 'properties': {}}
            ),
        ],
        allow_text_output=True,
    )

    tools = model._get_tools(params, model_settings=settings)  # pyright: ignore[reportPrivateUsage]

    # Gemini should NOT get cache_control on tools
    last_tool = cast(dict[str, Any], tools[0])
    assert 'cache_control' not in last_tool


# ===== Combined settings =====


async def test_openrouter_cache_all_settings_combined() -> None:
    """Test that all cache settings work together without interfering."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))
    settings = OpenRouterModelSettings(
        openrouter_cache_instructions=True,
        openrouter_cache_messages=True,
        openrouter_cache_tool_definitions=True,
    )

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='System instructions.'),
                UserPromptPart(content='User message'),
            ]
        )
    ]

    params = ModelRequestParameters(
        function_tools=[
            ToolDefinition(
                name='my_tool', description='A tool', parameters_json_schema={'type': 'object', 'properties': {}}
            ),
        ],
        allow_text_output=True,
    )

    # Check tools
    tools = model._get_tools(params, model_settings=settings)  # pyright: ignore[reportPrivateUsage]
    last_tool = cast(dict[str, Any], tools[0])
    assert last_tool['cache_control'] == {'type': 'ephemeral', 'ttl': '5m'}

    # Check messages
    mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, params, model_settings=settings
    )

    # System message should have cache_control
    system_msg = next(m for m in mapped if m.get('role') in ('system', 'developer'))
    system_content = system_msg.get('content')
    assert isinstance(system_content, list)
    assert 'cache_control' in system_content[0]

    # Last message should have cache_control
    last_msg = mapped[-1]
    last_content = last_msg.get('content')
    assert isinstance(last_content, list)
    assert 'cache_control' in last_content[-1]


# ===== Cache point limit enforcement =====


async def test_openrouter_limit_cache_points_anthropic() -> None:
    """Anthropic models via OpenRouter are limited to 4 cache breakpoints per request.

    When more than 4 CachePoints are placed in user messages (with system + tool cache
    consuming some slots), excess breakpoints are removed from the oldest messages first.
    """
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))
    settings = OpenRouterModelSettings(
        openrouter_cache_instructions=True,
        openrouter_cache_tool_definitions=True,
    )

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='System instructions.'),
                UserPromptPart(content=['Old context', CachePoint(), 'Old question']),
            ]
        ),
        ModelRequest(
            parts=[
                UserPromptPart(content=['Middle context', CachePoint(), 'Middle question']),
            ]
        ),
        ModelRequest(
            parts=[
                UserPromptPart(content=['Recent context', CachePoint(), 'Recent question']),
            ]
        ),
    ]

    params = ModelRequestParameters(
        function_tools=[
            ToolDefinition(
                name='my_tool', description='A tool', parameters_json_schema={'type': 'object', 'properties': {}}
            ),
        ],
        allow_text_output=True,
    )

    mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, params, model_settings=settings
    )

    # System message uses 1 slot, tool def uses 1 slot → 2 remaining for messages.
    # 3 CachePoints in user messages, only the 2 newest should survive.
    system_msg = next(m for m in mapped if m.get('role') in ('system', 'developer'))
    system_content = system_msg.get('content')
    assert isinstance(system_content, list)
    assert 'cache_control' in system_content[0]

    user_messages = [m for m in mapped if m.get('role') == 'user']
    assert len(user_messages) == 3

    # Oldest user message — cache_control should have been removed
    old_content = user_messages[0].get('content')
    assert isinstance(old_content, list)
    assert all('cache_control' not in cast(dict[str, Any], p) for p in old_content)

    # Middle user message — cache_control should be preserved (2nd newest)
    mid_content = user_messages[1].get('content')
    assert isinstance(mid_content, list)
    assert any('cache_control' in cast(dict[str, Any], p) for p in mid_content)

    # Newest user message — cache_control should be preserved (newest)
    new_content = user_messages[2].get('content')
    assert isinstance(new_content, list)
    assert any('cache_control' in cast(dict[str, Any], p) for p in new_content)


async def test_openrouter_limit_cache_points_no_tool_budget_without_tools() -> None:
    """openrouter_cache_tool_definitions should only reserve a slot when tools exist."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))
    settings = OpenRouterModelSettings(
        openrouter_cache_instructions=True,
        openrouter_cache_tool_definitions=True,
    )

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='System instructions.'),
                UserPromptPart(content=['Old context', CachePoint(), 'Old question']),
            ]
        ),
        ModelRequest(
            parts=[
                UserPromptPart(content=['Middle context', CachePoint(), 'Middle question']),
            ]
        ),
        ModelRequest(
            parts=[
                UserPromptPart(content=['Recent context', CachePoint(), 'Recent question']),
            ]
        ),
    ]

    mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, ModelRequestParameters(), model_settings=settings
    )

    user_messages = [m for m in mapped if m.get('role') == 'user']
    assert len(user_messages) == 3

    for msg in user_messages:
        content = msg.get('content')
        assert isinstance(content, list)
        assert any('cache_control' in cast(dict[str, Any], p) for p in content)


async def test_openrouter_limit_cache_points_exceeds_budget() -> None:
    """When system + tool cache points alone exceed the limit, a UserError is raised."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))

    openai_messages: list[chat.ChatCompletionMessageParam] = [
        {
            'role': 'system',
            'content': [
                {'type': 'text', 'text': 'System 1', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}},
                {'type': 'text', 'text': 'System 2', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}},
                {'type': 'text', 'text': 'System 3', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}},
                {'type': 'text', 'text': 'System 4', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}},
            ],
        },  # type: ignore[typeddict-item]
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Hello'}]},
    ]

    with pytest.raises(UserError, match='Too many cache points for downstream provider'):
        model._limit_cache_points(openai_messages, has_tool_cache_point=True)  # pyright: ignore[reportPrivateUsage]


# ===== Cache E2E tests with cassettes =====


async def test_openrouter_cache_point_anthropic_e2e(
    allow_model_requests: None, openrouter_api_key: str, vcr: Cassette
) -> None:
    """Test CachePoint with Anthropic model via OpenRouter using real API."""
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=provider)
    agent = Agent(model)

    result = await agent.run(
        ['Here is some important context to cache.' * 20, CachePoint(), 'Summarize the context in one sentence.']
    )

    assert isinstance(result.output, str)
    assert len(result.output) > 0
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Here is some important context to cache.' * 20,
                            CachePoint(),
                            'Summarize the context in one sentence.',
                        ],
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='The context consists of a repeated phrase stating that there is "important context to cache," repeated 20 times, but contains no actual substantive information.'
                    )
                ],
                usage=RequestUsage(
                    input_tokens=176,
                    output_tokens=34,
                    details={'is_byok': False, 'audio_tokens': 0, 'reasoning_tokens': 0, 'image_tokens': 0},
                ),
                model_name='anthropic/claude-4.6-sonnet-20260217',
                timestamp=IsDatetime(),
                provider_name='openrouter',
                provider_url='https://openrouter.ai/api/v1',
                provider_details={
                    'finish_reason': 'stop',
                    'downstream_provider': 'Amazon Bedrock',
                    'cost': 0.001038,
                    'upstream_inference_cost': 0.001038,
                    'is_byok': False,
                    'timestamp': IsDatetime(),
                },
                provider_response_id='gen-1773009364-SjDV5yqNtQzbIiXso5JR',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    # Verify cache_control was in the request
    assert vcr is not None
    assert len(vcr.requests) >= 1  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    request_body = json.loads(vcr.requests[0].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    user_content = request_body['messages'][0]['content']
    # The first content part should have cache_control
    assert 'cache_control' in user_content[0]
    assert user_content[0]['cache_control'] == {'type': 'ephemeral', 'ttl': '5m'}


async def test_openrouter_cache_point_gemini_e2e(
    allow_model_requests: None, openrouter_api_key: str, vcr: Cassette
) -> None:
    """Test CachePoint with Gemini model via OpenRouter using real API."""
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.5-flash', provider=provider)
    agent = Agent(model)

    result = await agent.run(
        ['Here is some important context to cache.' * 20, CachePoint(), 'Summarize the context in one sentence.']
    )

    assert isinstance(result.output, str)
    assert len(result.output) > 0
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Here is some important context to cache.' * 20,
                            CachePoint(),
                            'Summarize the context in one sentence.',
                        ],
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The provided text repeatedly emphasizes the importance of caching context.')],
                usage=RequestUsage(
                    input_tokens=168,
                    output_tokens=11,
                    details={'is_byok': False, 'audio_tokens': 0, 'reasoning_tokens': 0, 'image_tokens': 0},
                ),
                model_name='google/gemini-2.5-flash',
                timestamp=IsDatetime(),
                provider_name='openrouter',
                provider_url='https://openrouter.ai/api/v1',
                provider_details={
                    'finish_reason': 'STOP',
                    'downstream_provider': 'Google',
                    'cost': 7.79e-05,
                    'upstream_inference_cost': 7.79e-05,
                    'is_byok': False,
                    'timestamp': IsDatetime(),
                },
                provider_response_id='gen-1773009367-3zFFs0yQRvCe01Kda6ID',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    # Verify cache_control was in the request (without TTL for Gemini)
    assert vcr is not None
    assert len(vcr.requests) >= 1  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    request_body = json.loads(vcr.requests[0].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    user_content = request_body['messages'][0]['content']
    assert 'cache_control' in user_content[0]
    assert user_content[0]['cache_control'] == {'type': 'ephemeral'}


async def test_openrouter_cache_instructions_e2e(
    allow_model_requests: None, openrouter_api_key: str, vcr: Cassette
) -> None:
    """Test openrouter_cache_instructions with Anthropic model via real API."""
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=provider)
    agent = Agent(
        model,
        instructions='You are a helpful assistant that specializes in caching. ' * 20,
        model_settings=OpenRouterModelSettings(openrouter_cache_instructions=True),
    )

    result = await agent.run('What do you specialize in? Answer in one sentence.')

    assert isinstance(result.output, str)
    assert len(result.output) > 0
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='What do you specialize in? Answer in one sentence.', timestamp=IsDatetime())
                ],
                timestamp=IsDatetime(),
                instructions=IsStr(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='I specialize in caching.')],
                usage=RequestUsage(
                    input_tokens=260,
                    output_tokens=10,
                    details={'is_byok': False, 'audio_tokens': 0, 'reasoning_tokens': 0, 'image_tokens': 0},
                ),
                model_name='anthropic/claude-4.6-sonnet-20260217',
                timestamp=IsDatetime(),
                provider_name='openrouter',
                provider_url='https://openrouter.ai/api/v1',
                provider_details={
                    'finish_reason': 'stop',
                    'downstream_provider': 'Amazon Bedrock',
                    'cost': 0.00093,
                    'upstream_inference_cost': 0.00093,
                    'is_byok': False,
                    'timestamp': IsDatetime(),
                },
                provider_response_id='gen-1773009369-muGga581yy6h5yw9o2yt',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    # Verify cache_control was added to system message
    assert vcr is not None
    request_body = json.loads(vcr.requests[0].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    system_msg = next(m for m in request_body['messages'] if m['role'] in ('system', 'developer'))
    content = system_msg['content']
    assert isinstance(content, list)
    assert 'cache_control' in content[-1]


async def test_openrouter_cache_messages_e2e(
    allow_model_requests: None, openrouter_api_key: str, vcr: Cassette
) -> None:
    """Test openrouter_cache_messages with Anthropic model via real API."""
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=provider)
    agent = Agent(
        model,
        instructions='Be helpful.',
        model_settings=OpenRouterModelSettings(openrouter_cache_messages=True),
    )

    result = await agent.run('Say hello in one word.')

    assert isinstance(result.output, str)
    assert len(result.output) > 0
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Say hello in one word.', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                instructions='Be helpful.',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Hello!')],
                usage=RequestUsage(
                    input_tokens=17,
                    output_tokens=5,
                    details={'is_byok': False, 'audio_tokens': 0, 'reasoning_tokens': 0, 'image_tokens': 0},
                ),
                model_name='anthropic/claude-4.6-sonnet-20260217',
                timestamp=IsDatetime(),
                provider_name='openrouter',
                provider_url='https://openrouter.ai/api/v1',
                provider_details={
                    'finish_reason': 'stop',
                    'downstream_provider': 'Amazon Bedrock',
                    'cost': 0.000126,
                    'upstream_inference_cost': 0.000126,
                    'is_byok': False,
                    'timestamp': IsDatetime(),
                },
                provider_response_id='gen-1773009372-jrp75p3HzyzBsmgfbEZn',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    # Verify cache_control was added to the last message
    assert vcr is not None
    request_body = json.loads(vcr.requests[0].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    last_msg = request_body['messages'][-1]
    content = last_msg['content']
    assert isinstance(content, list)
    assert 'cache_control' in content[-1]


async def test_openrouter_cache_tool_definitions_e2e(
    allow_model_requests: None, openrouter_api_key: str, vcr: Cassette
) -> None:
    """Test openrouter_cache_tool_definitions with Anthropic model via real API."""
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=provider)

    agent = Agent(
        model,
        model_settings=OpenRouterModelSettings(openrouter_cache_tool_definitions=True),
    )

    @agent.tool_plain
    def get_weather(city: str) -> str:  # pragma: no cover
        return f'Sunny in {city}'

    result = await agent.run('What tools do you have available? Just list them briefly.')

    assert isinstance(result.output, str)
    assert len(result.output) > 0
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What tools do you have available? Just list them briefly.', timestamp=IsDatetime()
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
I have one tool available:

- **get_weather**: Retrieves the current weather for a specified city.\
"""
                    )
                ],
                usage=RequestUsage(
                    input_tokens=566,
                    output_tokens=27,
                    details={'is_byok': False, 'audio_tokens': 0, 'reasoning_tokens': 0, 'image_tokens': 0},
                ),
                model_name='anthropic/claude-4.6-sonnet-20260217',
                timestamp=IsDatetime(),
                provider_name='openrouter',
                provider_url='https://openrouter.ai/api/v1',
                provider_details={
                    'finish_reason': 'stop',
                    'downstream_provider': 'Amazon Bedrock',
                    'cost': 0.002103,
                    'upstream_inference_cost': 0.002103,
                    'is_byok': False,
                    'timestamp': IsDatetime(),
                },
                provider_response_id='gen-1773009374-OCscsda5VF5I9PRwtZv1',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    # Verify cache_control was added to the last tool
    assert vcr is not None
    request_body = json.loads(vcr.requests[0].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    tools = request_body.get('tools', [])
    assert len(tools) >= 1
    assert 'cache_control' in tools[-1]
    assert tools[-1]['cache_control'] == {'type': 'ephemeral', 'ttl': '5m'}


async def test_openrouter_cache_messages_anthropic_real_api(
    allow_model_requests: None, openrouter_api_key: str
) -> None:
    """Test that openrouter_cache_messages produces cache write/read metrics for Anthropic via OpenRouter.

    Forces routing to the Anthropic provider directly (not Bedrock) to ensure cache locality.
    The first call populates the cache, and the second call reads from it.
    """
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=provider)
    agent = Agent(
        model,
        instructions='You are a helpful assistant.',
        model_settings=OpenRouterModelSettings(
            openrouter_cache_messages=True,
            openrouter_provider={'order': ['anthropic'], 'allow_fallbacks': False},
            max_tokens=100,
        ),
    )

    # Must exceed Claude Sonnet's cacheable prompt minimum.
    result1 = await agent.run(
        'Analyze the architectural patterns used in distributed database systems and their tradeoffs. ' * 200
    )
    usage1 = result1.usage()

    assert usage1.requests == 1
    assert usage1.input_tokens > 2000
    assert usage1.cache_write_tokens > 0
    assert usage1.output_tokens > 0

    # Second call continues the conversation — the previous cached message is still in the request
    result2 = await agent.run('Can you summarize that in one sentence?', message_history=result1.all_messages())
    usage2 = result2.usage()

    # cache_read_tokens > 0 proves caching actually worked end-to-end
    assert usage2.requests == 1
    assert usage2.cache_read_tokens > 0
    assert usage2.output_tokens > 0


async def test_openrouter_cache_instructions_gemini_real_api(
    allow_model_requests: None, openrouter_api_key: str
) -> None:
    """Test that openrouter_cache_instructions produces cache write/read metrics for Gemini via OpenRouter.

    Uses cache_instructions with a long system prompt so the cached content is identical across
    both calls. Forces routing to Google AI Studio to ensure cache locality.
    """
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.5-flash', provider=provider)

    long_instructions = (
        'You are a specialized assistant that helps with distributed systems design. '
        'You have deep expertise in consensus protocols, CAP theorem, and eventual consistency. '
    ) * 80  # Long enough to exceed Gemini's current cacheable prompt minimum.

    agent = Agent(
        model,
        instructions=long_instructions,
        model_settings=OpenRouterModelSettings(
            openrouter_cache_instructions=True,
            openrouter_provider={'order': ['google-ai-studio'], 'allow_fallbacks': False},
            max_tokens=300,
        ),
    )

    # First call populates the cache with the long system instructions
    result1 = await agent.run('What is the CAP theorem?')
    usage1 = result1.usage()

    assert usage1.requests == 1
    assert usage1.input_tokens > 1000
    assert usage1.cache_write_tokens > 0
    assert usage1.output_tokens > 0

    # Second call reuses the same system instructions — should hit cache
    result2 = await agent.run('What is eventual consistency?')
    usage2 = result2.usage()

    assert usage2.requests == 1
    assert usage2.cache_read_tokens > 0
    assert usage2.output_tokens > 0


async def test_openrouter_cache_streaming_e2e(
    allow_model_requests: None, openrouter_api_key: str, vcr: Cassette
) -> None:
    """Test that cache_control is correctly included in requests when using streaming."""
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=provider)
    agent = Agent(
        model,
        instructions='You are a helpful assistant that specializes in caching. ' * 20,
        model_settings=OpenRouterModelSettings(
            openrouter_cache_instructions=True,
            openrouter_cache_messages=True,
        ),
    )

    async with agent.run_stream('Say hello in one word.') as stream:
        result = await stream.get_output()

    assert isinstance(result, str)
    assert len(result) > 0
    assert stream.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Say hello in one word.', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                instructions=IsStr(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Hello!')],
                usage=RequestUsage(
                    input_tokens=254,
                    output_tokens=5,
                    details={'is_byok': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'image_tokens': 0},
                ),
                model_name='anthropic/claude-4.6-sonnet-20260217',
                timestamp=IsDatetime(),
                provider_name='openrouter',
                provider_url='https://openrouter.ai/api/v1',
                provider_details={
                    'timestamp': IsDatetime(),
                    'downstream_provider': 'Amazon Bedrock',
                    'finish_reason': 'stop',
                },
                provider_response_id='gen-1773012759-4u9w7As08eMtL75bWtu8',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    # Verify cache_control was included in the request
    assert vcr is not None
    request_body = json.loads(vcr.requests[0].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]

    # System message should have cache_control from cache_instructions
    system_msg = next(m for m in request_body['messages'] if m['role'] in ('system', 'developer'))
    system_content = system_msg['content']
    assert isinstance(system_content, list)
    assert 'cache_control' in system_content[-1]

    # Last user message should have cache_control from cache_messages
    last_msg = request_body['messages'][-1]
    last_content = last_msg['content']
    assert isinstance(last_content, list)
    assert 'cache_control' in last_content[-1]


async def test_openrouter_cache_all_settings_real_api(allow_model_requests: None, openrouter_api_key: str) -> None:
    """Test all cache settings combined with actual cache write+read metrics.

    Enables cache_instructions, cache_tool_definitions, and cache_messages together,
    forces Anthropic routing for cache locality, and verifies cache write/read metrics.
    """
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=provider)

    agent = Agent(
        model,
        instructions=(
            'You are an assistant that specializes in mathematics and calculations. '
            'Always show your work step by step. '
        )
        * 100,  # Long enough to exceed Claude Sonnet's cacheable prompt minimum.
        model_settings=OpenRouterModelSettings(
            openrouter_cache_instructions=True,
            openrouter_cache_tool_definitions=True,
            openrouter_cache_messages=True,
            openrouter_provider={'order': ['anthropic'], 'allow_fallbacks': False},
            max_tokens=100,
        ),
    )

    @agent.tool_plain
    def calculator(expression: str) -> str:
        """Evaluate a math expression."""
        return 'result'

    result1 = await agent.run('What is 123 * 456?')
    usage1 = result1.usage()

    assert usage1.requests >= 1
    assert usage1.input_tokens > 2000
    assert usage1.cache_write_tokens > 0
    assert usage1.output_tokens > 0

    # Second call with same agent — system instructions + tools should be cached
    result2 = await agent.run('What is 789 + 321?')
    usage2 = result2.usage()

    assert usage2.requests >= 1
    assert usage2.cache_read_tokens > 0
    assert usage2.output_tokens > 0


async def test_openrouter_limit_cache_points_e2e(
    allow_model_requests: None, openrouter_api_key: str, vcr: Cassette
) -> None:
    """Test that _limit_cache_points trims excess breakpoints so the request succeeds.

    Sends 5 CachePoint markers plus cache_instructions (6 total breakpoints) to an
    Anthropic model via OpenRouter. Without limiting, Anthropic would return a 400 error.
    Verifies the request succeeds and the cassette shows at most 4 cache_control breakpoints.
    """
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=provider)
    agent = Agent(
        model,
        instructions='You are a helpful assistant. ' * 50,
        model_settings=OpenRouterModelSettings(openrouter_cache_instructions=True),
    )

    result = await agent.run(
        [
            'Context block one. ' * 20,
            CachePoint(),
            'Context block two. ' * 20,
            CachePoint(),
            'Context block three. ' * 20,
            CachePoint(),
            'Context block four. ' * 20,
            CachePoint(),
            'Context block five. ' * 20,
            CachePoint(),
            'Summarize everything in one sentence.',
        ]
    )

    assert isinstance(result.output, str)
    assert len(result.output) > 0

    assert vcr is not None
    request_body = cast(
        dict[str, list[dict[str, list[dict[str, Any]]]]],
        json.loads(vcr.requests[0].body),  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    )

    cache_count = 0
    for msg in request_body['messages']:
        for block in msg['content']:
            if 'cache_control' in block:
                cache_count += 1

    assert cache_count <= 4


def test_openrouter_nested_provider_response() -> None:
    """OpenRouter sometimes nests the real response inside the 'provider' dict.

    Regression test for https://github.com/pydantic/pydantic-ai/issues/3994.
    """
    provider = OpenRouterProvider(api_key='test-key')
    model = OpenRouterModel('openai/gpt-4.1-mini', provider=provider)

    nested_completion = ChatCompletion.model_construct(
        id=None,
        choices=None,
        model=None,
        object=None,
        provider={
            'id': 'gen-123',
            'choices': [
                {
                    'index': 0,
                    'message': {'role': 'assistant', 'content': 'Hello from nested!'},
                    'finish_reason': 'stop',
                    'native_finish_reason': 'STOP',
                    'logprobs': None,
                }
            ],
            'model': 'google/gemini-3-flash-preview',
            'object': 'chat.completion',
            'provider': 'Google',
        },
        created=1234567890,
        usage=None,
    )

    model_response = model._process_response(nested_completion)  # type: ignore[reportPrivateUsage]

    assert model_response.parts == snapshot([TextPart(content='Hello from nested!')])
    assert model_response.provider_details == snapshot(
        {
            'downstream_provider': 'Google',
            'finish_reason': 'STOP',
            'timestamp': datetime.datetime(2009, 2, 13, 23, 31, 30, tzinfo=datetime.timezone.utc),
        }
    )


def test_openrouter_nested_provider_null_name() -> None:
    """Nested provider dict with provider=None falls back to 'unknown'."""
    provider = OpenRouterProvider(api_key='test-key')
    model = OpenRouterModel('openai/gpt-4.1-mini', provider=provider)

    completion = ChatCompletion.model_construct(
        id=None,
        choices=None,
        model=None,
        object=None,
        provider={
            'id': 'nested-gen-1',
            'choices': [
                {
                    'index': 0,
                    'message': {'role': 'assistant', 'content': 'Hi'},
                    'finish_reason': 'stop',
                    'native_finish_reason': 'STOP',
                    'logprobs': None,
                }
            ],
            'model': 'openai/gpt-4.1-mini',
            'object': 'chat.completion',
            'provider': None,
            'created': 1234567890,
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
        },
        created=1234567890,
    )

    result = model._process_response(completion)  # type: ignore[reportPrivateUsage]
    assert result.provider_details == snapshot(
        {
            'downstream_provider': 'unknown',
            'finish_reason': 'STOP',
            'timestamp': datetime.datetime(2009, 2, 13, 23, 31, 30, tzinfo=datetime.timezone.utc),
        }
    )


def test_openrouter_provider_dict_without_choices_raises() -> None:
    """Provider is a dict with no 'choices' key — no unwrap happens, validation fails."""
    provider = OpenRouterProvider(api_key='test-key')
    model = OpenRouterModel('openai/gpt-4.1-mini', provider=provider)

    completion = ChatCompletion.model_construct(
        id=None,
        choices=None,
        model=None,
        object=None,
        provider={'some_key': 'some_value'},
        created=1234567890,
    )

    with pytest.raises(UnexpectedModelBehavior):
        model._process_response(completion)  # type: ignore[reportPrivateUsage]


def test_openrouter_error_with_null_fields() -> None:
    """Error responses with null standard fields raise ModelHTTPError.

    Regression test for https://github.com/pydantic/pydantic-ai/issues/3994.
    """
    provider = OpenRouterProvider(api_key='test-key')
    model = OpenRouterModel('openai/gpt-4.1-mini', provider=provider)

    error_completion = ChatCompletion.model_construct(
        id=None,
        choices=None,
        model=None,
        object=None,
        provider=None,
        created=1234567890,
        usage=None,
        error={'code': 400, 'message': 'Invalid request parameters'},
    )

    with pytest.raises(ModelHTTPError) as exc_info:
        model._process_response(error_completion)  # type: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400
    assert 'Invalid request parameters' in str(exc_info.value)


def test_openrouter_malformed_error_fallthrough() -> None:
    """Malformed error data falls through to validation, surfacing as UnexpectedModelBehavior."""
    provider = OpenRouterProvider(api_key='test-key')
    model = OpenRouterModel('openai/gpt-4.1-mini', provider=provider)

    completion = ChatCompletion.model_construct(
        id=None,
        choices=None,
        model=None,
        object=None,
        provider=None,
        created=1234567890,
        usage=None,
        error='something went wrong',
    )

    with pytest.raises(UnexpectedModelBehavior):
        model._process_response(completion)  # type: ignore[reportPrivateUsage]


def test_openrouter_error_with_metadata() -> None:
    """Real-world error response with metadata field from #3994.

    OpenRouter returns error code 524 with extra metadata including the raw
    error and provider name. The extra fields should be ignored.
    """
    provider = OpenRouterProvider(api_key='test-key')
    model = OpenRouterModel('google/gemini-3-flash-preview', provider=provider)

    completion = ChatCompletion.model_construct(
        id=None,
        choices=None,
        created=1768361801,
        model=None,
        object=None,
        service_tier=None,
        system_fingerprint=None,
        usage=None,
        error={
            'message': 'Provider returned error',
            'code': 524,
            'metadata': {'raw': 'error code: 524', 'provider_name': 'Google'},
        },
        user_id='org_xxx',
    )

    with pytest.raises(ModelHTTPError) as exc_info:
        model._process_response(completion)  # type: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 524
    assert 'Provider returned error' in str(exc_info.value)
