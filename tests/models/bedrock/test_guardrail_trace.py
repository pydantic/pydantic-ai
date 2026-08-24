"""Tests for carrying the Bedrock guardrail `trace` through to `provider_details`.

Regression test for #7561: when the guardrail config sets `trace: "enabled"`,
Bedrock returns the per-policy assessment (which filter fired, at what
confidence, and the guardrail coverage) on the response's `trace` field —
both in the non-streaming response body and on the streaming `metadata`
event. It was previously discarded, so callers could tell *that* a guardrail
intervened but not *why*.
"""

from __future__ import annotations as _annotations

import pytest
from dirty_equals import IsPartialDict
from pytest_mock import MockerFixture

from pydantic_ai import Agent, TextPart

from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider


pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='bedrock not installed'),
]


GUARDRAIL_TRACE = {
    'guardrail': {
        'inputAssessment': {
            'guardrail-id-1': {
                'contentPolicy': {
                    'filters': [
                        {'type': 'PROMPT_ATTACK', 'confidence': 'HIGH', 'action': 'BLOCKED'},
                    ]
                },
            },
        },
    },
}


async def test_non_streaming_trace_in_provider_details(
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
    mocker: MockerFixture,
):
    """The guardrail assessment from the response body lands in `provider_details['trace']`."""
    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=bedrock_provider)
    agent = Agent(model)

    mock_converse = mocker.patch.object(model.client, 'converse')
    mock_converse.return_value = {
        'output': {'message': {'role': 'assistant', 'content': [{'text': 'hello'}]}},
        'stopReason': 'end_turn',
        'usage': {'inputTokens': 5, 'outputTokens': 10},
        'trace': GUARDRAIL_TRACE,
        'ResponseMetadata': {'HTTPStatusCode': 200},
    }

    result = await agent.run('hi')

    response = result.all_messages()[-1]
    assert response.provider_details == IsPartialDict(trace=GUARDRAIL_TRACE)
    assert response.provider_details['finish_reason'] == 'end_turn'
    assert TextPart(content='hello') in response.parts


async def test_non_streaming_no_trace_keeps_details_unchanged(
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
    mocker: MockerFixture,
):
    """Without a `trace` on the response, `provider_details` keeps its previous shape."""
    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=bedrock_provider)
    agent = Agent(model)

    mock_converse = mocker.patch.object(model.client, 'converse')
    mock_converse.return_value = {
        'output': {'message': {'role': 'assistant', 'content': [{'text': 'hello'}]}},
        'stopReason': 'end_turn',
        'usage': {'inputTokens': 5, 'outputTokens': 10},
        'ResponseMetadata': {'HTTPStatusCode': 200},
    }

    result = await agent.run('hi')

    response = result.all_messages()[-1]
    assert response.provider_details == {'finish_reason': 'end_turn'}


async def test_streaming_trace_in_provider_details(
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
    mocker: MockerFixture,
):
    """The guardrail assessment on the streaming `metadata` event lands in `provider_details['trace']`.

    Uses the metadata-after-messageStop order (the one Bedrock actually emits) but the merge
    logic must also survive the opposite order.
    """
    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=bedrock_provider)
    agent = Agent(model)

    def fake_event_stream():
        yield {'messageStart': {'role': 'assistant'}}
        yield {'contentBlockDelta': {'contentBlockIndex': 0, 'delta': {'text': 'hello'}}}
        yield {'contentBlockStop': {'contentBlockIndex': 0}}
        yield {'messageStop': {'stopReason': 'guardrail_intervened'}}
        yield {
            'metadata': {
                'usage': {'inputTokens': 5, 'outputTokens': 10},
                'trace': GUARDRAIL_TRACE,
            }
        }

    mock_converse_stream = mocker.patch.object(model.client, 'converse_stream')
    mock_converse_stream.return_value = {'stream': fake_event_stream()}

    async with agent.run_stream('hi') as result:
        chunks = [c async for c in result.stream_text(delta=True)]

    assert chunks == ['hello']
    response = result.all_messages()[-1]
    assert response.provider_details == IsPartialDict(
        finish_reason='guardrail_intervened',
        trace=GUARDRAIL_TRACE,
    )


async def test_streaming_trace_before_message_stop_preserved(
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
    mocker: MockerFixture,
):
    """A `metadata` event arriving *before* `messageStop` must not lose the trace when
    `messageStop` writes the finish reason."""
    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=bedrock_provider)
    agent = Agent(model)

    def fake_event_stream():
        yield {'messageStart': {'role': 'assistant'}}
        yield {'contentBlockDelta': {'contentBlockIndex': 0, 'delta': {'text': 'hi'}}}
        yield {'contentBlockStop': {'contentBlockIndex': 0}}
        yield {
            'metadata': {
                'usage': {'inputTokens': 5, 'outputTokens': 10},
                'trace': GUARDRAIL_TRACE,
            }
        }
        yield {'messageStop': {'stopReason': 'guardrail_intervened'}}

    mock_converse_stream = mocker.patch.object(model.client, 'converse_stream')
    mock_converse_stream.return_value = {'stream': fake_event_stream()}

    async with agent.run_stream('hi') as result:
        _ = [c async for c in result.stream_text(delta=True)]

    response = result.all_messages()[-1]
    assert response.provider_details == IsPartialDict(
        finish_reason='guardrail_intervened',
        trace=GUARDRAIL_TRACE,
    )
