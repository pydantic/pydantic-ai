from collections.abc import Sequence
from typing import Literal, cast

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    ModelHTTPError,
    ModelMessage,
    ModelRequest,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolDefinition,
    UnexpectedModelBehavior,
)
from pydantic_ai.direct import model_request, model_request_stream
from pydantic_ai.models import ModelRequestParameters

from ..conftest import try_import

with try_import() as imports_successful:
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion import Choice

    from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
    from pydantic_ai.providers.openrouter import OpenRouterProvider

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
    assert response.provider_details['native_finish_reason'] == 'stop'


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

        assert stream.provider_details == snapshot({'finish_reason': 'stop'})
        assert stream.finish_reason == snapshot('stop')


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

    mapped_messages = await model._map_messages([response])  # type: ignore[reportPrivateUsage]
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

    openai_messages = await model._map_messages(messages)  # type: ignore[reportPrivateUsage]

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


async def test_openrouter_errors_raised(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.0-flash-exp:free', provider=provider)
    agent = Agent(model, instructions='Be helpful.', retries=1)
    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Tell me a joke.')
    assert str(exc_info.value) == snapshot(
        "status_code: 429, model_name: google/gemini-2.0-flash-exp:free, body: {'code': 429, 'message': 'Provider returned error', 'metadata': {'provider_name': 'Google', 'raw': 'google/gemini-2.0-flash-exp:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations'}}"
    )


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


async def test_openrouter_map_messages_reasoning(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('anthropic/claude-3.7-sonnet:thinking', provider=provider)

    user_message = ModelRequest.user_text_prompt('Who are you. Think about it.')
    response = await model_request(model, [user_message])

    mapped_messages = await model._map_messages([user_message, response])  # type: ignore[reportPrivateUsage]

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
