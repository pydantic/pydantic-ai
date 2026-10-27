"""Changing the thinking effort mid-conversation on the APIs that have a channel for it (#10385).

OpenAI Responses takes a `configuration_update` input item, Anthropic an effort-only `{'role': 'system'}` entry
with a per-message `output_config`, and OpenRouter a content-less `system` message carrying a `configuration_update`
(#11189). Either way the request-level effort stays what the conversation opened with,
so the prompt cache keeps its prefix: measured on 2026-10-23, a request-level change on `gpt-6.1-sol` read 0 of
~8,600 prompt tokens from cache where the item read 8,593. Every other case sends the change as the
request-level effort, as before.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, cast

import pytest

from pydantic_ai import (
    Agent,
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSettingsDeltaPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.direct import model_request
from pydantic_ai.exceptions import ModelHTTPError, UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentedModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings, ThinkingEffort

from .._inline_snapshot import snapshot
from ..conftest import RequestCapture, try_import

with try_import() as openai_imports_successful:
    from openai.types.chat import ChatCompletionMessage
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText
    from openai.types.responses.response_output_message import Content

    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    from .mock_openai import (
        MockOpenAI,
        MockOpenAIResponses,
        completion_message as chat_completion_message,
        get_mock_chat_completion_kwargs as get_mock_chat_kwargs,
        get_mock_responses_kwargs,
        response_message,
    )

with try_import() as anthropic_imports_successful:
    from anthropic import (
        AsyncAnthropicBedrock,
        AsyncAnthropicBedrockMantle,
        AsyncAnthropicFoundry,
        AsyncAnthropicVertex,
    )
    from anthropic.types.beta import BetaTextBlock, BetaUsage

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .test_anthropic import MockAnthropic, completion_message, get_mock_chat_completion_kwargs

pytestmark = pytest.mark.vcr

_PREFIX = 'You answer questions about the catalogue below in one short sentence.\n' + '\n'.join(
    f'Item {i:04d}: shelf {i % 37}, bin {i % 11}, revision {i}, status verified.' for i in range(400)
)
"""Long enough to be cached on both providers."""


def _history(change: ThinkingEffort = 'xhigh') -> list[ModelMessage]:
    """A conversation that opened at `low` effort, before a change to `change`."""
    return [
        ModelRequest(
            parts=[
                UserPromptPart('Where is item 12?'),
                ModelSettingsDeltaPart(settings={'thinking': 'low'}, top_level=True),
            ]
        ),
        ModelResponse(parts=[TextPart('Shelf 12, bin 1.')]),
        ModelRequest(parts=[UserPromptPart('And item 40?'), ModelSettingsDeltaPart(settings={'thinking': change})]),
    ]


# --- OpenAI Responses -----------------------------------------------------------------------------


def _responses_text(text: str) -> Any:
    return response_message(
        [
            ResponseOutputMessage(
                id='msg-1',
                content=cast(list[Content], [ResponseOutputText(text=text, type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            )
        ]
    )


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_openai_effort_change_keeps_the_cache(allow_model_requests: None, openai_api_key: str):
    """The second run sends the change as a `configuration_update` item and reads the first run's prefix from cache."""
    model = OpenAIResponsesModel('gpt-6.1-sol', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model, instructions=_PREFIX)
    settings = OpenAIResponsesModelSettings(openai_prompt_cache_key='pydantic-ai-effort-delta')
    first = await agent.run('Where is item 12?', model_settings={**settings, 'thinking': 'low'})
    second = await agent.run(
        'And item 40?', message_history=first.all_messages(), model_settings={**settings, 'thinking': 'xhigh'}
    )
    assert [
        part for message in second.all_messages() for part in message.parts if isinstance(part, ModelSettingsDeltaPart)
    ] == [
        ModelSettingsDeltaPart(settings={'thinking': 'low'}, top_level=True),
        ModelSettingsDeltaPart(settings={'thinking': 'xhigh'}),
    ]
    assert second.response.usage.cache_read_tokens >= first.response.usage.input_tokens - 100


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_openai_effort_change_is_a_configuration_update(allow_model_requests: None):
    mock_client = MockOpenAIResponses.create_mock(_responses_text('Shelf 40, bin 7.'))
    model = OpenAIResponsesModel('gpt-6.1-sol', provider=OpenAIProvider(openai_client=mock_client))

    await Agent(model).run(message_history=_history(), model_settings=ModelSettings(thinking='xhigh'))

    request = get_mock_responses_kwargs(mock_client)[0]
    assert request['reasoning'] == {'effort': 'low', 'context': 'all_turns'}
    assert request['input'] == snapshot(
        [
            {'role': 'user', 'content': 'Where is item 12?'},
            {'role': 'assistant', 'content': 'Shelf 12, bin 1.'},
            {'role': 'user', 'content': 'And item 40?'},
            {'type': 'configuration_update', 'reasoning': {'effort': 'xhigh'}},
        ]
    )


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_openai_without_the_opening_effort_sends_the_current_one(allow_model_requests: None):
    """A history processor that dropped the window's opening part leaves the current effort as the request-level one."""
    mock_client = MockOpenAIResponses.create_mock(_responses_text('Shelf 40, bin 7.'))
    model = OpenAIResponsesModel('gpt-6.1-sol', provider=OpenAIProvider(openai_client=mock_client))
    history: list[ModelMessage] = [
        ModelRequest(
            parts=[UserPromptPart('Where is item 12?'), ModelSettingsDeltaPart(settings={'thinking': 'medium'})]
        ),
        ModelResponse(parts=[TextPart('Shelf 12, bin 1.')]),
        ModelRequest(parts=[UserPromptPart('And item 40?')]),
    ]

    await Agent(model).run(message_history=history, model_settings=ModelSettings(thinking='xhigh'))

    request = get_mock_responses_kwargs(mock_client)[0]
    assert request['reasoning'] == {'effort': 'xhigh', 'context': 'all_turns'}
    assert request['input'][-1] == {'type': 'configuration_update', 'reasoning': {'effort': 'xhigh'}}


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
@pytest.mark.parametrize(
    ('settings', 'history'),
    [
        pytest.param({'openai_reasoning_effort': 'high'}, _history(), id='explicit-effort'),
        pytest.param({'openai_truncation': 'auto'}, _history(), id='truncation'),
        pytest.param(
            {'openai_context_management': [{'type': 'compaction'}]},
            _history(),
            id='server-compaction',
        ),
        pytest.param(
            {},
            [*_history()[:2], ModelResponse(parts=[CompactionPart(content='Summary.')]), *_history()[2:]],
            id='compacted-history',
        ),
    ],
)
def test_openai_falls_back_to_the_request_level_effort(
    settings: OpenAIResponsesModelSettings, history: list[ModelMessage]
):
    """Where OpenAI documents the item doesn't apply, the parts go and the current effort is sent at request level."""
    model = OpenAIResponsesModel('gpt-6.1-sol', provider=OpenAIProvider(api_key='test-key'))
    parameters = ModelRequestParameters(thinking='xhigh')
    sends_changes, effort_parameters = model._effort_delta_request(history, settings, parameters)  # pyright: ignore[reportPrivateUsage]
    assert sends_changes is False
    assert effort_parameters is parameters


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_openai_effort_parts_dont_break_server_side_continuation(allow_model_requests: None):
    """`openai_previous_response_id='auto'` matches the stored history, effort parts included, before they're removed."""
    mock_client = MockOpenAIResponses.create_mock([_responses_text('Shelf 12, bin 1.'), _responses_text('Shelf 40.')])
    model = OpenAIResponsesModel('gpt-6.1-sol', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(model)
    settings = OpenAIResponsesModelSettings(openai_previous_response_id='auto', thinking='low')

    first = await agent.run('Where is item 12?', model_settings=settings)
    await agent.run('And item 40?', message_history=first.all_messages(), model_settings=settings)

    request = get_mock_responses_kwargs(mock_client)[1]
    assert request['previous_response_id'] == '123'
    assert request['input'] == [{'role': 'user', 'content': 'And item 40?'}]


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
def test_openai_profile_gate():
    gpt_6, gpt_5 = OpenAIProvider.model_profile('gpt-6.1-sol'), OpenAIProvider.model_profile('gpt-5.6-sol')
    assert gpt_6 is not None and gpt_5 is not None
    assert gpt_6.get('supports_mid_conversation_effort') is True
    assert gpt_5.get('supports_mid_conversation_effort') is False


# --- OpenRouter -----------------------------------------------------------------------------------


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
@pytest.mark.parametrize(
    ('model_name', 'expected_request'),
    [
        pytest.param(
            'anthropic/claude-opus-5.5',
            snapshot(
                {
                    'reasoning': {'effort': 'low', 'enabled': True},
                    'messages': [
                        {'role': 'user', 'content': 'Where is item 12?'},
                        {
                            'role': 'assistant',
                            'content': 'Item 0012 is on shelf 12, bin 1 (revision 12, status verified).',
                        },
                        {
                            'role': 'user',
                            'content': [
                                {
                                    'type': 'text',
                                    'text': 'And item 40?',
                                    'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
                                }
                            ],
                        },
                        {'role': 'system', 'content': '', 'configuration_update': {'reasoning': {'effort': 'high'}}},
                    ],
                }
            ),
            id='anthropic',
        ),
        pytest.param(
            'openai/gpt-6.1-sol',
            snapshot(
                {
                    'reasoning': {'effort': 'low', 'enabled': True},
                    'messages': [
                        {'role': 'user', 'content': 'Where is item 12?'},
                        {'role': 'assistant', 'content': 'Item 0012 is on shelf 12, in bin 1.'},
                        {'role': 'user', 'content': 'And item 40?'},
                        {'role': 'system', 'content': '', 'configuration_update': {'reasoning': {'effort': 'high'}}},
                    ],
                }
            ),
            id='openai',
        ),
    ],
)
async def test_openrouter_effort_change_keeps_the_cache(
    allow_model_requests: None,
    openrouter_api_key: str,
    request_capture: RequestCapture,
    model_name: str,
    expected_request: Any,
):
    """The second run sends the change as a content-less `system` message and reads the first run's prefix from cache.

    Sent as the request-level `reasoning.effort` instead, the change read 0 of 8,058 prompt tokens from cache on
    `openai/gpt-6.1-sol` (recorded 2026-10-26). OpenRouter translates the message to the routed model's own form.
    """
    provider = OpenRouterProvider(api_key=openrouter_api_key, http_client=request_capture.http_client(timeout=120))
    agent = Agent(OpenRouterModel(model_name, provider=provider), instructions=_PREFIX)
    settings = OpenRouterModelSettings(openrouter_cache_instructions=True, openrouter_cache_messages=True)
    first = await agent.run('Where is item 12?', model_settings={**settings, 'thinking': 'low'})
    second = await agent.run(
        'And item 40?', message_history=first.all_messages(), model_settings={**settings, 'thinking': 'high'}
    )

    _, second_request = request_capture.bodies()
    # The instructions are left out: they're the long cached prefix.
    # Past the standing prompt: the `<system>` tag announcement and the catalogue.
    messages = cast(list[Any], second_request['messages'])[2:]
    assert {'reasoning': second_request['reasoning'], 'messages': messages} == expected_request
    assert second.response.usage.cache_read_tokens >= first.response.usage.input_tokens - 100


def _openrouter_text(text: str) -> Any:
    completion = chat_completion_message(ChatCompletionMessage(content=text, role='assistant'))
    return completion.model_copy(update={'provider': 'Anthropic'})


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_openrouter_earlier_effort_change_stays_in_place(allow_model_requests: None):
    """A change from an earlier request is replayed where it was, and the breakpoints skip over it."""
    mock_client = MockOpenAI.create_mock(_openrouter_text('Shelf 41.'))
    model = OpenRouterModel('anthropic/claude-opus-5.5', provider=OpenRouterProvider(openai_client=mock_client))
    history = [
        *_history('high'),
        ModelResponse(parts=[TextPart('Shelf 40, bin 7.')]),
        ModelRequest(parts=[UserPromptPart('And item 41?')]),
    ]

    await Agent(model).run(
        message_history=history,
        model_settings=OpenRouterModelSettings(thinking='high', openrouter_cache_messages=True),
    )

    request = get_mock_chat_kwargs(mock_client)[0]
    assert request['extra_body']['reasoning'] == {'effort': 'low', 'enabled': True}
    assert request['messages'] == snapshot(
        [
            {'role': 'user', 'content': 'Where is item 12?'},
            {'role': 'assistant', 'content': 'Shelf 12, bin 1.'},
            {'role': 'user', 'content': 'And item 40?'},
            {'role': 'system', 'content': '', 'configuration_update': {'reasoning': {'effort': 'high'}}},
            {'role': 'assistant', 'content': 'Shelf 40, bin 7.'},
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'And item 41?', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}}
                ],
            },
        ]
    )


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_openrouter_without_the_opening_effort_sends_the_current_one(allow_model_requests: None):
    """A history processor that dropped the window's opening part leaves the current effort as the request-level one."""
    mock_client = MockOpenAI.create_mock(_openrouter_text('Shelf 40, bin 7.'))
    model = OpenRouterModel('openai/gpt-6.1-sol', provider=OpenRouterProvider(openai_client=mock_client))

    history: list[ModelMessage] = [
        ModelRequest(
            parts=[UserPromptPart('Where is item 12?'), ModelSettingsDeltaPart(settings={'thinking': 'medium'})]
        ),
        ModelResponse(parts=[TextPart('Shelf 12, bin 1.')]),
        ModelRequest(parts=[UserPromptPart('And item 40?')]),
    ]

    await Agent(model).run(message_history=history, model_settings=ModelSettings(thinking='high'))

    request = get_mock_chat_kwargs(mock_client)[0]
    assert request['extra_body']['reasoning'] == {'effort': 'high', 'enabled': True}
    assert request['messages'][-1] == {
        'role': 'system',
        'content': '',
        'configuration_update': {'reasoning': {'effort': 'high'}},
    }


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
@pytest.mark.parametrize(
    ('model_name', 'settings'),
    [
        pytest.param('openai/gpt-6.1-sol', {'openrouter_reasoning': {'effort': 'medium'}}, id='explicit-reasoning'),
        pytest.param('openai/gpt-6.1-sol', {'openai_reasoning_effort': 'medium'}, id='explicit-effort'),
        pytest.param('anthropic/claude-sonnet-5', {}, id='unsupported-model'),
    ],
)
async def test_openrouter_falls_back_to_the_request_level_effort(
    allow_model_requests: None, model_name: str, settings: dict[str, Any]
):
    """A direct `request` skips `prepare_messages`, so the adapter itself removes the parts where it can't send them."""
    mock_client = MockOpenAI.create_mock(_openrouter_text('Shelf 40, bin 7.'))
    model = OpenRouterModel(model_name, provider=OpenRouterProvider(openai_client=mock_client))

    await model.request(
        _history('high'), cast(ModelSettings, {**settings, 'thinking': 'high'}), ModelRequestParameters()
    )

    request = get_mock_chat_kwargs(mock_client)[0]
    assert request['extra_body']['reasoning'] == settings.get(
        'openrouter_reasoning', {'effort': 'high', 'enabled': True}
    )
    assert not any('configuration_update' in message for message in request['messages'])


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
@pytest.mark.parametrize(
    ('model_name', 'supported'),
    [
        pytest.param('anthropic/claude-fable-5.1', True, id='fable-5-1'),
        pytest.param('anthropic/claude-opus-5', True, id='opus-5'),
        pytest.param('anthropic/claude-haiku-5.5', True, id='haiku-5-5'),
        pytest.param('anthropic/claude-fable-5', False, id='fable-5'),
        pytest.param('anthropic/claude-sonnet-5', False, id='sonnet-5'),
        pytest.param('openai/gpt-6-astra', True, id='gpt-6-astra'),
        pytest.param('openai/gpt-6-astra-pro', False, id='gpt-6-astra-pro'),
        pytest.param('openai/gpt-6.1-terra', False, id='gpt-6-1-terra'),
        pytest.param('openai/gpt-5.6', False, id='gpt-5-6'),
        pytest.param('google/gemini-3.5-flash', False, id='gemini'),
    ],
)
def test_openrouter_profile_gate(model_name: str, supported: bool):
    profile = OpenRouterProvider.model_profile(model_name)
    assert profile is not None
    assert profile.get('supports_mid_conversation_effort') is supported


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_openai_chat_direct_request_with_effort_parts_needs_prepare_messages(allow_model_requests: None):
    """Chat Completions has no mid-conversation channel; a direct `request` that skipped `prepare_messages` says so."""
    mock_client = MockOpenAI.create_mock(chat_completion_message(ChatCompletionMessage(content='ok', role='assistant')))
    model = OpenAIChatModel('gpt-6.1-sol', provider=OpenAIProvider(openai_client=mock_client))

    with pytest.raises(UserError, match='`ModelSettingsDeltaPart` cannot be rendered by this model'):
        await model.request(_history(), ModelSettings(thinking='xhigh'), ModelRequestParameters())


# --- Anthropic ------------------------------------------------------------------------------------


def _anthropic_text(text: str) -> Any:
    return completion_message([BetaTextBlock(text=text, type='text')], BetaUsage(input_tokens=5, output_tokens=10))


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_anthropic_effort_change_keeps_the_cache(allow_model_requests: None, anthropic_api_key: str):
    """The second run sends the change as an effort-only `system` entry and reads the first run's prefix from cache."""
    model = AnthropicModel('claude-opus-5-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(model, instructions=_PREFIX)
    settings = AnthropicModelSettings(anthropic_cache_instructions=True, anthropic_cache_messages=True)
    first = await agent.run('Where is item 12?', model_settings={**settings, 'thinking': 'low'})
    second = await agent.run(
        'And item 40?', message_history=first.all_messages(), model_settings={**settings, 'thinking': 'xhigh'}
    )
    assert second.response.usage.cache_read_tokens >= first.response.usage.cache_write_tokens


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_anthropic_bedrock_messages_api_takes_an_effort_change(
    allow_model_requests: None, request_capture: RequestCapture
):
    """The Bedrock Messages API takes the change as an effort-only `system` entry under its beta.

    Without the beta it rejects the entry with "Extra inputs are not permitted", so the header is what this pins.
    https://github.com/pydantic/pydantic-ai/issues/11481
    """
    client = AsyncAnthropicBedrockMantle(
        api_key=os.environ.get('AWS_BEARER_TOKEN_BEDROCK', 'test-bedrock-token'),
        aws_region=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'),
        http_client=request_capture.client,
    )
    model = AnthropicModel('anthropic.claude-opus-5-5', provider=AnthropicProvider(anthropic_client=client))
    agent = Agent(model, instructions='Answer in one short sentence.')
    first = await agent.run('Name a primary color.', model_settings={'thinking': 'low'})
    second = await agent.run(
        'Name another.', message_history=first.all_messages(), model_settings={'thinking': 'medium'}
    )

    assert second.output
    body = request_capture.bodies('/v1/messages')[-1]
    assert body['output_config'] == {'effort': 'low'}
    assert body['messages'] == snapshot(
        [
            {'role': 'user', 'content': [{'text': 'Name a primary color.', 'type': 'text'}]},
            {'role': 'assistant', 'content': [{'text': 'Red is a primary color.', 'type': 'text'}]},
            {'role': 'user', 'content': [{'text': 'Name another.', 'type': 'text'}]},
            {'role': 'system', 'content': [], 'output_config': {'effort': 'medium'}},
        ]
    )
    assert 'mid-conversation-output-config-2026-07-01' in request_capture.headers[-1]['anthropic-beta']


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_anthropic_effort_change_is_a_system_entry(allow_model_requests: None):
    mock_client = MockAnthropic.create_mock(_anthropic_text('Shelf 40, bin 7.'))
    model = AnthropicModel('claude-opus-5-5', provider=AnthropicProvider(anthropic_client=mock_client))

    await Agent(model).run(
        message_history=_history(),
        model_settings=AnthropicModelSettings(thinking='xhigh', anthropic_cache_messages=True),
    )

    request = get_mock_chat_completion_kwargs(mock_client)[0]
    assert request['output_config'] == {'effort': 'low'}
    assert 'mid-conversation-output-config-2026-07-01' in request['betas']
    # The entry has no block to carry the breakpoint, so it goes on the user turn before it.
    assert request['messages'] == snapshot(
        [
            {'role': 'user', 'content': [{'text': 'Where is item 12?', 'type': 'text'}]},
            {'role': 'assistant', 'content': [{'text': 'Shelf 12, bin 1.', 'type': 'text'}]},
            {
                'role': 'user',
                'content': [
                    {'text': 'And item 40?', 'type': 'text', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}}
                ],
            },
            {'role': 'system', 'content': [], 'output_config': {'effort': 'xhigh'}},
        ]
    )


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_anthropic_without_the_opening_effort_sends_the_current_one(allow_model_requests: None):
    """A history processor that dropped the window's opening part leaves the current effort as the request-level one."""
    mock_client = MockAnthropic.create_mock(_anthropic_text('Shelf 40, bin 7.'))
    model = AnthropicModel('claude-opus-5-5', provider=AnthropicProvider(anthropic_client=mock_client))
    history: list[ModelMessage] = [
        ModelRequest(
            parts=[UserPromptPart('Where is item 12?'), ModelSettingsDeltaPart(settings={'thinking': 'medium'})]
        ),
        ModelResponse(parts=[TextPart('Shelf 12, bin 1.')]),
        ModelRequest(parts=[UserPromptPart('And item 40?')]),
    ]

    await Agent(model).run(message_history=history, model_settings=AnthropicModelSettings(thinking='xhigh'))

    request = get_mock_chat_completion_kwargs(mock_client)[0]
    assert request['output_config'] == {'effort': 'xhigh'}
    assert request['messages'][-1] == {'role': 'system', 'content': [], 'output_config': {'effort': 'xhigh'}}


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_anthropic_effort_change_is_counted(allow_model_requests: None):
    mock_client = MockAnthropic.create_mock(_anthropic_text('unused'))
    model = AnthropicModel('claude-opus-5-5', provider=AnthropicProvider(anthropic_client=mock_client))
    messages = model.prepare_messages(_history())

    await model.count_tokens(
        messages, AnthropicModelSettings(thinking='xhigh'), ModelRequestParameters(thinking='xhigh')
    )

    request = get_mock_chat_completion_kwargs(mock_client)[0]
    assert request['messages'][-1] == {'role': 'system', 'content': [], 'output_config': {'effort': 'xhigh'}}
    assert request['output_config'] == {'effort': 'low'}


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
@pytest.mark.parametrize(
    'settings',
    [
        pytest.param({'anthropic_effort': 'high'}, id='explicit-effort'),
        pytest.param({'anthropic_thinking': {'type': 'disabled'}}, id='thinking-disabled'),
    ],
)
def test_anthropic_falls_back_to_the_request_level_effort(settings: AnthropicModelSettings):
    """Under non-adaptive thinking the API rejects a differing per-message effort, so it goes at request level."""
    model = AnthropicModel('claude-haiku-5-5', provider=AnthropicProvider(api_key='test-key'))
    parameters = ModelRequestParameters(thinking='xhigh')
    messages, effort_parameters = model._effort_delta_request(_history(), settings, parameters)  # pyright: ignore[reportPrivateUsage]
    assert effort_parameters is parameters
    assert not any(isinstance(part, ModelSettingsDeltaPart) for message in messages for part in message.parts)


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
@pytest.mark.parametrize(
    ('model_name', 'client', 'supported'),
    [
        pytest.param('claude-opus-5-5', None, True, id='claude-api'),
        pytest.param('claude-fable-5-1', None, True, id='fable-5-1'),
        pytest.param('claude-fable-5', None, False, id='fable-5'),
        pytest.param('claude-sonnet-5', None, False, id='sonnet-5'),
        pytest.param(
            'claude-haiku-5-5',
            lambda: AsyncAnthropicVertex(project_id='p', region='us-central1', access_token='x'),
            True,
            id='vertex',
        ),
        pytest.param(
            'us.anthropic.claude-opus-5-5',
            lambda: AsyncAnthropicBedrock(aws_access_key='x', aws_secret_key='y', aws_region='us-east-1'),
            True,
            id='bedrock-invoke-model',
        ),
        pytest.param(
            'us.anthropic.claude-sonnet-5-5',
            lambda: AsyncAnthropicBedrock(aws_access_key='x', aws_secret_key='y', aws_region='us-east-1'),
            False,
            id='bedrock-invoke-model-unlisted',
        ),
        pytest.param(
            'anthropic.claude-opus-5-5',
            lambda: AsyncAnthropicBedrockMantle(aws_region='us-east-1', aws_access_key='x', aws_secret_key='y'),
            True,
            id='bedrock-messages-api',
        ),
        pytest.param(
            'anthropic.claude-opus-5',
            lambda: AsyncAnthropicBedrockMantle(aws_region='us-east-1', aws_access_key='x', aws_secret_key='y'),
            False,
            id='bedrock-messages-api-unlisted',
        ),
        pytest.param(
            'claude-opus-5-5',
            lambda: AsyncAnthropicFoundry(api_key='x', base_url='https://example.com'),
            False,
            id='foundry',
        ),
    ],
)
def test_anthropic_profile_gate(model_name: str, client: Any, supported: bool):
    provider = AnthropicProvider(anthropic_client=client()) if client else AnthropicProvider(api_key='test-key')
    model = AnthropicModel(model_name, provider=provider)
    assert model.profile.get('supports_mid_conversation_effort') is supported


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_openai_explicit_effort_records_no_unified_level(allow_model_requests: None):
    """A request whose effort `openai_reasoning_effort` sets doesn't open a window at the unified level it overrides.

    Otherwise a later run would pin the request-level effort to that unified level, which the first request never
    sent, and move the cached prefix this exists to keep.
    """

    mock_client = MockOpenAIResponses.create_mock([_responses_text('Shelf 12, bin 1.'), _responses_text('Shelf 40.')])
    model = OpenAIResponsesModel('gpt-6.1-sol', provider=OpenAIProvider(openai_client=mock_client))
    # Both wrappers ask the model they wrap.
    agent = Agent(FallbackModel(InstrumentedModel(model)))

    first = await agent.run(
        'Where is item 12?',
        model_settings=OpenAIResponsesModelSettings(thinking='low', openai_reasoning_effort='high'),
    )
    second = await agent.run(
        'And item 40?', message_history=first.all_messages(), model_settings=ModelSettings(thinking='xhigh')
    )

    assert [
        part for message in second.all_messages() for part in message.parts if isinstance(part, ModelSettingsDeltaPart)
    ] == [ModelSettingsDeltaPart(settings={'thinking': 'xhigh'}, top_level=True)]
    first_request, second_request = get_mock_responses_kwargs(mock_client)
    assert first_request['reasoning']['effort'] == 'high'
    assert second_request['reasoning']['effort'] == 'xhigh'
    assert not any(item.get('type') == 'configuration_update' for item in second_request['input'])


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
def test_anthropic_explicit_effort_overrides_the_unified_level():
    model = AnthropicModel('claude-opus-5-5', provider=AnthropicProvider(api_key='test-key'))
    assert model._overrides_unified_effort({'anthropic_effort': 'high'}) is True  # pyright: ignore[reportPrivateUsage]
    assert model._overrides_unified_effort({'thinking': 'high'}) is False  # pyright: ignore[reportPrivateUsage]


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
def test_openai_chat_explicit_effort_overrides_the_unified_level():
    model = OpenAIChatModel('gpt-6.1-sol', provider=OpenAIProvider(api_key='test-key'))
    assert model._overrides_unified_effort({'openai_reasoning_effort': 'high'}) is True  # pyright: ignore[reportPrivateUsage]
    assert model._overrides_unified_effort({'thinking': 'high'}) is False  # pyright: ignore[reportPrivateUsage]


def _effort_parts_by_request(messages: list[ModelMessage]) -> list[list[ModelSettingsDeltaPart]]:
    return [
        [part for part in message.parts if isinstance(part, ModelSettingsDeltaPart)]
        for message in messages
        if isinstance(message, ModelRequest)
    ]


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_fallback_candidate_defaults_decide_the_recorded_effort(allow_model_requests: None):
    """The effort is the one the candidate sends, with its own `settings` merged in, not the wrapper's settings."""

    mock_client = MockOpenAIResponses.create_mock(
        [_responses_text('one'), _responses_text('two'), _responses_text('3')]
    )
    candidate = OpenAIResponsesModel(
        'gpt-6.1-sol', provider=OpenAIProvider(openai_client=mock_client), settings=ModelSettings(thinking='high')
    )
    agent = Agent(FallbackModel(candidate))

    first = await agent.run('Where is item 12?', model_settings=ModelSettings(thinking='low'))
    second = await agent.run(
        'And item 40?', message_history=first.all_messages(), model_settings=ModelSettings(thinking='high')
    )
    # The candidate's own default is still `'high'`, so leaving the run setting out changes nothing.
    third = await agent.run('And item 41?', message_history=second.all_messages())

    assert _effort_parts_by_request(third.all_messages()) == [
        [ModelSettingsDeltaPart(settings={'thinking': 'low'}, top_level=True)],
        [ModelSettingsDeltaPart(settings={'thinking': 'high'})],
        [],
    ]
    last_request = get_mock_responses_kwargs(mock_client)[2]
    assert last_request['reasoning']['effort'] == 'low'
    assert [item for item in last_request['input'] if item.get('type') == 'configuration_update'] == [
        {'type': 'configuration_update', 'reasoning': {'effort': 'high'}}
    ]


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_fallback_candidates_that_disagree_record_no_effort(allow_model_requests: None):
    """A request whose candidates would send different efforts claims none, so it goes out at request level.

    The first candidate overrides the unified effort with a provider-specific one and fails, so the second serves
    the request at the unified effort. Recording can't know which will serve it, so it opens a window naming no
    effort, and the serving candidate sends the current effort at request level with no markers.
    """

    class OverridingModel(FunctionModel):
        def _overrides_unified_effort(self, model_settings: Mapping[str, object]) -> bool:
            return True

    def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=500, model_name='overriding', body=None)

    mock_client = MockOpenAIResponses.create_mock([_responses_text('one'), _responses_text('two')])
    serving = OpenAIResponsesModel('gpt-6.1-sol', provider=OpenAIProvider(openai_client=mock_client))

    first = await Agent(serving).run('Where is item 12?', model_settings=ModelSettings(thinking='low'))
    second = await Agent(FallbackModel(OverridingModel(fail), serving)).run(
        'And item 40?', message_history=first.all_messages(), model_settings=ModelSettings(thinking='xhigh')
    )

    assert _effort_parts_by_request(second.all_messages()) == [
        [ModelSettingsDeltaPart(settings={'thinking': 'low'}, top_level=True)],
        [ModelSettingsDeltaPart(settings={'thinking': None}, top_level=True)],
    ]
    request = get_mock_responses_kwargs(mock_client)[1]
    assert request['reasoning']['effort'] == 'xhigh'
    assert not any(item.get('type') == 'configuration_update' for item in request['input'])


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_agent_history_works_with_direct_requests(allow_model_requests: None):
    """A history an agent run recorded effort parts in goes through `model_request`, at the effort it asks for.

    The parts are removed rather than replayed: a direct call records nothing, so an earlier change sent in place
    would override the effort it sets.
    """

    first = await Agent(TestModel()).run('Where is item 12?', model_settings=ModelSettings(thinking='low'))
    second = await Agent(TestModel()).run(
        'And item 40?', message_history=first.all_messages(), model_settings=ModelSettings(thinking='xhigh')
    )
    history = [*second.all_messages(), ModelRequest(parts=[UserPromptPart('And item 41?')])]

    chat_client = MockOpenAI.create_mock(chat_completion_message(ChatCompletionMessage(content='ok', role='assistant')))
    chat = OpenAIChatModel('gpt-6.1-sol', provider=OpenAIProvider(openai_client=chat_client))
    await model_request(chat, history, model_settings=ModelSettings(thinking='medium'))
    assert get_mock_chat_kwargs(chat_client)[0]['reasoning_effort'] == 'medium'

    responses_client = MockOpenAIResponses.create_mock(_responses_text('ok'))
    responses_model = OpenAIResponsesModel('gpt-6.1-sol', provider=OpenAIProvider(openai_client=responses_client))
    await model_request(responses_model, history, model_settings=ModelSettings(thinking='medium'))
    request = get_mock_responses_kwargs(responses_client)[0]
    assert request['reasoning']['effort'] == 'medium'
    assert not any(item.get('type') == 'configuration_update' for item in request['input'])


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_openai_direct_request_without_the_channel_sends_effort_at_request_level(allow_model_requests: None):
    """A direct `request` skips `prepare_messages`, so the adapter itself drops the parts on a model without the channel."""
    mock_client = MockOpenAIResponses.create_mock(_responses_text('Shelf 40, bin 7.'))
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    await model.request(_history(), ModelSettings(thinking='xhigh'), ModelRequestParameters())

    request = get_mock_responses_kwargs(mock_client)[0]
    assert request['reasoning']['effort'] == 'xhigh'
    assert not any(item.get('type') == 'configuration_update' for item in request['input'])


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_anthropic_direct_request_without_the_channel_sends_effort_at_request_level(allow_model_requests: None):
    mock_client = MockAnthropic.create_mock(_anthropic_text('Shelf 40, bin 7.'))
    model = AnthropicModel('claude-sonnet-5', provider=AnthropicProvider(anthropic_client=mock_client))

    await model.request(_history(), AnthropicModelSettings(thinking='xhigh'), ModelRequestParameters())

    request = get_mock_chat_completion_kwargs(mock_client)[0]
    assert request['output_config'] == {'effort': 'xhigh'}
    assert not any('output_config' in message for message in request['messages'])
    assert 'mid-conversation-output-config-2026-07-01' not in str(request.get('betas'))
