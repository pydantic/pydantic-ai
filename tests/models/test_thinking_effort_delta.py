"""Changing the thinking effort mid-conversation on the APIs that have a channel for it (#10385).

OpenAI Responses takes a `configuration_update` input item and Anthropic an effort-only `{'role': 'system'}` entry
with a per-message `output_config`. Either way the request-level effort stays what the conversation opened with,
so the prompt cache keeps its prefix: measured on 2026-10-23, a request-level change on `gpt-6.1-sol` read 0 of
~8,600 prompt tokens from cache where the item read 8,593. Every other case sends the change as the
request-level effort, as before.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from pydantic_ai import (
    Agent,
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingEffortDeltaPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as openai_imports_successful:
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText
    from openai.types.responses.response_output_message import Content

    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider

    from .mock_openai import MockOpenAIResponses, get_mock_responses_kwargs, response_message

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


def _history() -> list[ModelMessage]:
    """A conversation that opened at `low` effort, before a change to `xhigh`."""
    return [
        ModelRequest(
            parts=[UserPromptPart('Where is item 12?'), ThinkingEffortDeltaPart(effort='low', top_level=True)]
        ),
        ModelResponse(parts=[TextPart('Shelf 12, bin 1.')]),
        ModelRequest(parts=[UserPromptPart('And item 40?'), ThinkingEffortDeltaPart(effort='xhigh')]),
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
        part for message in second.all_messages() for part in message.parts if isinstance(part, ThinkingEffortDeltaPart)
    ] == [
        ThinkingEffortDeltaPart(effort='low', top_level=True),
        ThinkingEffortDeltaPart(effort='xhigh'),
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
        ModelRequest(parts=[UserPromptPart('Where is item 12?'), ThinkingEffortDeltaPart(effort='medium')]),
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
    messages, effort_parameters = model._effort_delta_request(history, settings, parameters)  # pyright: ignore[reportPrivateUsage]
    assert effort_parameters is parameters
    assert not any(isinstance(part, ThinkingEffortDeltaPart) for message in messages for part in message.parts)


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
def test_openai_profile_gate():
    gpt_6, gpt_5 = OpenAIProvider.model_profile('gpt-6.1-sol'), OpenAIProvider.model_profile('gpt-5.6-sol')
    assert gpt_6 is not None and gpt_5 is not None
    assert gpt_6.get('supports_mid_conversation_effort') is True
    assert gpt_5.get('supports_mid_conversation_effort') is False


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
    assert not any(isinstance(part, ThinkingEffortDeltaPart) for message in messages for part in message.parts)


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
            False,
            id='bedrock-messages-api',
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
