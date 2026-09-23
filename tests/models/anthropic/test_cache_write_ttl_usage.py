"""Anthropic cache-write durations survive usage extraction and streaming."""

from __future__ import annotations

from decimal import Decimal

import pytest

from pydantic_ai import Agent, ModelResponse

from ...conftest import try_import
from ..test_anthropic import MockAnthropic, completion_message

with try_import() as imports_successful:
    from anthropic.types.beta import (
        BetaMessage,
        BetaMessageDeltaUsage,
        BetaRawContentBlockStartEvent,
        BetaRawContentBlockStopEvent,
        BetaRawMessageDeltaEvent,
        BetaRawMessageStartEvent,
        BetaRawMessageStopEvent,
        BetaTextBlock,
        BetaUsage,
    )
    from anthropic.types.beta.beta_raw_message_delta_event import Delta

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
]


def _cache_usage() -> BetaUsage:
    return BetaUsage.model_validate(
        {
            'input_tokens': 100,
            'output_tokens': 20,
            'cache_creation_input_tokens': 30,
            'cache_read_input_tokens': 0,
            'cache_creation': {'ephemeral_5m_input_tokens': 10, 'ephemeral_1h_input_tokens': 20},
        }
    )


def _assert_cache_usage(response: ModelResponse, *, output_tokens: int, expected_cost: Decimal) -> None:
    assert response.usage.input_tokens == 130
    assert response.usage.cache_write_tokens == 30
    assert response.usage.output_tokens == output_tokens
    assert response.usage.details['cache_write_5m_tokens'] == 10
    assert response.usage.details['cache_write_1h_tokens'] == 20
    assert response.usage.opentelemetry_attributes()['gen_ai.usage.details.cache_write_1h_tokens'] == 20
    assert response.cost().total_price == expected_cost


async def test_anthropic_cache_write_duration_in_response(allow_model_requests: None) -> None:
    reply = completion_message([BetaTextBlock(text='done', type='text')], _cache_usage()).model_copy(
        update={'model': 'claude-opus-5-5'}
    )
    client = MockAnthropic.create_mock(reply)
    model = AnthropicModel('claude-opus-5-5', provider=AnthropicProvider(anthropic_client=client))

    result = await Agent(model).run('hello')

    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    _assert_cache_usage(response, output_tokens=20, expected_cost=Decimal('0.00101'))


async def test_anthropic_cache_write_duration_in_stream(allow_model_requests: None) -> None:
    stream = [
        BetaRawMessageStartEvent(
            type='message_start',
            message=BetaMessage(
                id='msg_1',
                model='claude-opus-5-5',
                role='assistant',
                type='message',
                content=[],
                stop_reason=None,
                usage=_cache_usage(),
            ),
        ),
        BetaRawContentBlockStartEvent(
            type='content_block_start', index=0, content_block=BetaTextBlock(type='text', text='done')
        ),
        BetaRawContentBlockStopEvent(type='content_block_stop', index=0),
        BetaRawMessageDeltaEvent(
            type='message_delta',
            delta=Delta(stop_reason='end_turn'),
            usage=BetaMessageDeltaUsage(input_tokens=100, output_tokens=25),
        ),
        BetaRawMessageStopEvent(type='message_stop'),
    ]
    client = MockAnthropic.create_stream_mock(stream)
    model = AnthropicModel('claude-opus-5-5', provider=AnthropicProvider(anthropic_client=client))

    async with Agent(model).run_stream('hello') as result:
        assert await result.get_output() == 'done'

    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    _assert_cache_usage(response, output_tokens=25, expected_cost=Decimal('0.00111'))
