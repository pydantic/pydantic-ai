from typing import cast

import pytest
from inline_snapshot import snapshot
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice

from pydantic_ai import Agent, ModelHTTPError, ModelRequest, TextPart, ThinkingPart, UnexpectedModelBehavior
from pydantic_ai.direct import model_request

from ..conftest import try_import

with try_import() as imports_successful:
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


async def test_openrouter_with_reasoning(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('z-ai/glm-4.6', provider=provider)
    response = await model_request(model, [ModelRequest.user_text_prompt('Who are you')])

    assert len(response.parts) == 2
    assert isinstance(thinking_part := response.parts[0], ThinkingPart)
    assert isinstance(response.parts[1], TextPart)
    assert thinking_part.content == snapshot(
        """\
Let me process this query about who I am. First, I should consider what the user really wants to know - they're likely seeking to understand my identity and capabilities as an AI assistant.

I need to be clear and accurate about my nature. I'm a GLM large language model developed by Zhipu AI, not a human. This distinction is fundamental to our interaction.

Looking at my core functions, I should highlight my ability to engage in natural conversations, answer questions, and assist with various tasks. My training involves processing vast amounts of text data, which enables me to understand and generate human-like responses.

It's important to mention my commitment to being helpful, harmless, and honest. These principles guide my interactions and ensure I provide appropriate assistance.

I should also emphasize my continuous learning aspect. While I don't store personal data, I'm regularly updated to improve my capabilities and knowledge base.

The response should be welcoming and encourage further questions about specific areas where I can help. This creates an open dialogue and shows my willingness to assist with various topics.

Let me structure this information in a clear, friendly manner that addresses the user's question while inviting further interaction.\
"""
    )
    assert response.provider_details is not None
    assert response.provider_details['reasoning_details'] == snapshot(
        [
            {
                'format': 'unknown',
                'index': 0,
                'text': """\
Let me process this query about who I am. First, I should consider what the user really wants to know - they're likely seeking to understand my identity and capabilities as an AI assistant.

I need to be clear and accurate about my nature. I'm a GLM large language model developed by Zhipu AI, not a human. This distinction is fundamental to our interaction.

Looking at my core functions, I should highlight my ability to engage in natural conversations, answer questions, and assist with various tasks. My training involves processing vast amounts of text data, which enables me to understand and generate human-like responses.

It's important to mention my commitment to being helpful, harmless, and honest. These principles guide my interactions and ensure I provide appropriate assistance.

I should also emphasize my continuous learning aspect. While I don't store personal data, I'm regularly updated to improve my capabilities and knowledge base.

The response should be welcoming and encourage further questions about specific areas where I can help. This creates an open dialogue and shows my willingness to assist with various topics.

Let me structure this information in a clear, friendly manner that addresses the user's question while inviting further interaction.\
""",
                'type': 'reasoning.text',
            }
        ]
    )


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
        model._process_response('This is not JSON!')

    assert str(exc_info.value) == snapshot(
        'Invalid response from OpenRouter chat completions endpoint, expected JSON data'
    )


async def test_openrouter_validate_error_response(openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.0-flash-exp:free', provider=provider)

    response = ChatCompletion.model_construct(model='test')
    response.error = {'message': 'This response has an error attribute', 'code': 200}

    with pytest.raises(ModelHTTPError) as exc_info:
        model._process_response(response)

    assert str(exc_info.value) == snapshot(
        'status_code: 200, model_name: test, body: This response has an error attribute'
    )


async def test_openrouter_validate_error_finish_reason(openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.0-flash-exp:free', provider=provider)

    choice = Choice.model_construct(finish_reason='error')
    response = ChatCompletion.model_construct(choices=[choice])

    with pytest.raises(UnexpectedModelBehavior) as exc_info:
        model._process_response(response)

    assert str(exc_info.value) == snapshot(
        'Invalid response from OpenRouter chat completions endpoint, error finish_reason without error data'
    )
