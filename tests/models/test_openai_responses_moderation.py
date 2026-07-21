"""Tests for moderation results via OpenAI Responses API provider_details exposure."""

from __future__ import annotations

from typing import cast

import pytest
from openai.types import responses as resp
from openai.types.responses.response_output_message import Content, ResponseOutputMessage

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from ..conftest import message
from .mock_openai import MockOpenAIResponses

pytestmark = [pytest.mark.anyio]


async def test_openai_responses_moderation_non_streaming(allow_model_requests: None):
    """Moderation results from the Responses API are exposed in provider_details."""
    from openai.types.responses.response import (
        Moderation,
        ModerationInputModerationResult,
        ModerationOutputModerationResult,
    )

    moderation = Moderation(
        input=ModerationInputModerationResult(
            categories={'harassment': True},
            category_applied_input_types={'harassment': ['text']},
            category_scores={'harassment': 0.9},
            flagged=True,
            model='omni-moderation-2024-09-18',
            type='moderation_result',
        ),
        output=ModerationOutputModerationResult(
            categories={'harassment': False},
            category_applied_input_types={'harassment': ['text']},
            category_scores={'harassment': 0.1},
            flagged=False,
            model='omni-moderation-2024-09-18',
            type='moderation_result',
        ),
    )

    c = resp.Response(
        id='123',
        model='gpt-4o-123',
        object='response',
        created_at=1704067200,  # 2024-01-01
        output=[
            ResponseOutputMessage(
                id='output-1',
                content=cast(list[Content], [resp.ResponseOutputText(text='done', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            )
        ],
        parallel_tool_calls=True,
        tool_choice='auto',
        tools=[],
        moderation=moderation,
    )
    mock_client = MockOpenAIResponses.create_mock(c)
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))

    agent = Agent(model=model)
    result = await agent.run('Hello')

    response = message(result.all_messages(), ModelResponse, index=-1)
    assert response.provider_details is not None
    assert 'moderation' in response.provider_details
    assert response.provider_details['moderation'] == moderation.model_dump()
    assert response.provider_details['moderation']['input']['flagged'] is True
    # existing keys remain present alongside moderation
    assert 'timestamp' in response.provider_details


async def test_openai_responses_moderation_streaming(allow_model_requests: None):
    """Streaming Responses API moderation results are exposed in provider_details."""
    from openai.types.responses.response import (
        Moderation,
        ModerationInputModerationResult,
        ModerationOutputModerationResult,
    )

    moderation = Moderation(
        input=ModerationInputModerationResult(
            categories={'harassment': True},
            category_applied_input_types={'harassment': ['text']},
            category_scores={'harassment': 0.9},
            flagged=True,
            model='omni-moderation-2024-09-18',
            type='moderation_result',
        ),
        output=ModerationOutputModerationResult(
            categories={'harassment': False},
            category_applied_input_types={'harassment': ['text']},
            category_scores={'harassment': 0.1},
            flagged=False,
            model='omni-moderation-2024-09-18',
            type='moderation_result',
        ),
    )

    base_response = resp.Response(
        id='resp_001',
        model='gpt-4o',
        object='response',
        created_at=1704067200,
        output=[],
        parallel_tool_calls=True,
        tool_choice='auto',
        tools=[],
    )

    completed_response = base_response.model_copy(update={'status': 'completed', 'moderation': moderation})

    stream: list[resp.ResponseStreamEvent] = [
        resp.ResponseCreatedEvent(response=base_response, type='response.created', sequence_number=0),
        resp.ResponseInProgressEvent(response=base_response, type='response.in_progress', sequence_number=1),
        resp.ResponseOutputItemAddedEvent(
            item=ResponseOutputMessage(
                id='msg_001',
                content=[],
                role='assistant',
                status='in_progress',
                type='message',
            ),
            output_index=0,
            type='response.output_item.added',
            sequence_number=2,
        ),
        resp.ResponseContentPartAddedEvent(
            content_index=0,
            item_id='msg_001',
            output_index=0,
            part=resp.ResponseOutputText(text='', type='output_text', annotations=[]),
            type='response.content_part.added',
            sequence_number=3,
        ),
        resp.ResponseTextDeltaEvent(
            content_index=0,
            delta='Hello!',
            item_id='msg_001',
            output_index=0,
            type='response.output_text.delta',
            sequence_number=4,
            logprobs=[],
        ),
        resp.ResponseTextDoneEvent(
            content_index=0,
            item_id='msg_001',
            output_index=0,
            text='Hello!',
            type='response.output_text.done',
            sequence_number=5,
            logprobs=[],
        ),
        resp.ResponseOutputItemDoneEvent(
            item=ResponseOutputMessage(
                id='msg_001',
                content=cast(
                    list[Content], [resp.ResponseOutputText(text='Hello!', type='output_text', annotations=[])]
                ),
                role='assistant',
                status='completed',
                type='message',
            ),
            output_index=0,
            type='response.output_item.done',
            sequence_number=6,
        ),
        resp.ResponseCompletedEvent(
            response=completed_response,
            type='response.completed',
            sequence_number=7,
        ),
    ]

    mock_client = MockOpenAIResponses.create_mock_stream(stream)
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(model=model)

    async with agent.run_stream('Hello') as result:
        output = await result.get_output()

    assert output == 'Hello!'
    response = message(result.all_messages(), ModelResponse, index=-1)
    assert response.provider_details is not None
    assert 'moderation' in response.provider_details
    assert response.provider_details['moderation'] == moderation.model_dump()
    assert response.provider_details['moderation']['input']['flagged'] is True
    # existing keys remain present alongside moderation
    assert 'timestamp' in response.provider_details
    assert 'finish_reason' in response.provider_details
