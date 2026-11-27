"""xAI stored conversations: what `XaiModel` sends when continuing a stored response via `xai_previous_response_id`.

`MockXai` captures the `chat.create()` arguments as sent, so each test checks which messages follow the stored response
and whether the id is kept, using the `history_digest` recorded on the stored `ModelResponse`.
"""

from __future__ import annotations as _annotations

from dataclasses import replace

import pytest

from pydantic_ai import Agent, ModelRequest, ModelResponse, ToolReturnPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from .._inline_snapshot import snapshot
from ..conftest import try_import
from .mock_xai import MockXai, create_response, create_tool_call, get_grok_text_chunk, get_mock_chat_create_kwargs

with try_import() as imports_successful:
    from pydantic_ai.models.xai import XaiModel, XaiModelSettings
    from pydantic_ai.providers.xai import XaiProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='xai_sdk not installed')

XAI_NON_REASONING_MODEL = 'grok-4-fast-non-reasoning'


async def test_xai_previous_response_id_sends_only_what_follows_the_stored_response(allow_model_requests: None):
    """The stored response holds the history it was created from and the instructions, so neither is sent again."""
    mock_client = MockXai.create_mock([create_response(content='first'), create_response(content='second')])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, instructions='Be brief.', model_settings=XaiModelSettings(xai_store_messages=True))

    first = await agent.run('hello')
    [*_, stored] = first.all_messages()
    assert isinstance(stored, ModelResponse) and stored.history_digest is not None
    await agent.run(
        'again',
        message_history=first.all_messages(),
        model_settings=XaiModelSettings(xai_previous_response_id='grok-123'),
    )

    second_kwargs = get_mock_chat_create_kwargs(mock_client)[1]
    assert second_kwargs['messages'] == snapshot([{'content': [{'text': 'again'}], 'role': 'ROLE_USER'}])
    assert second_kwargs['previous_response_id'] == 'grok-123'


@pytest.mark.parametrize('change', ['instructions', 'history'])
async def test_xai_previous_response_id_starts_over_after_a_change(allow_model_requests: None, change: str):
    """Changed instructions or an edited history mean the server's copy is stale: send everything, without the id."""
    mock_client = MockXai.create_mock([create_response(content='first'), create_response(content='second')])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    settings = XaiModelSettings(xai_store_messages=True)
    first = await Agent(m, instructions='Be brief.', model_settings=settings).run('hello')
    history = first.all_messages()
    if change == 'history':
        request = history[0]
        assert isinstance(request, ModelRequest)
        history[0] = replace(request, parts=[UserPromptPart('hello, edited')])
    instructions = 'Be thorough.' if change == 'instructions' else 'Be brief.'

    await Agent(m, instructions=instructions, model_settings=settings).run(
        'again', message_history=history, model_settings=XaiModelSettings(xai_previous_response_id='grok-123')
    )

    second_kwargs = get_mock_chat_create_kwargs(mock_client)[1]
    assert 'previous_response_id' not in second_kwargs
    assert [message['role'] for message in second_kwargs['messages']] == snapshot(
        ['ROLE_SYSTEM', 'ROLE_USER', 'ROLE_ASSISTANT', 'ROLE_USER']
    )


async def test_xai_streamed_response_records_its_history_digest(allow_model_requests: None):
    """A streamed response stored on the server records its digest too, so a later request can continue from it."""
    stream = [get_grok_text_chunk('hi')]
    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, instructions='Be brief.', model_settings=XaiModelSettings(xai_store_messages=True))

    async with agent.run_stream('hello') as result:
        await result.get_output()
    [*_, stored] = result.all_messages()
    assert isinstance(stored, ModelResponse) and stored.history_digest is not None


async def test_xai_previous_response_id_starts_over_without_a_stored_digest(allow_model_requests: None):
    """A response stored by an earlier release has no digest to check against: send everything, without the id."""
    mock_client = MockXai.create_mock([create_response(content='first'), create_response(content='second')])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, instructions='Be brief.', model_settings=XaiModelSettings(xai_store_messages=True))

    first = await agent.run('hello')
    *history, stored = first.all_messages()
    assert isinstance(stored, ModelResponse)
    await agent.run(
        'again',
        message_history=[*history, replace(stored, history_digest=None)],
        model_settings=XaiModelSettings(xai_previous_response_id='grok-123'),
    )

    second_kwargs = get_mock_chat_create_kwargs(mock_client)[1]
    assert 'previous_response_id' not in second_kwargs
    assert [message['role'] for message in second_kwargs['messages']] == snapshot(
        ['ROLE_SYSTEM', 'ROLE_USER', 'ROLE_ASSISTANT', 'ROLE_USER']
    )


async def test_xai_previous_response_id_orders_tool_results_by_the_stored_responses_calls(allow_model_requests: None):
    """The stored response isn't sent, but its tool calls still set the order the results that follow it go out in.

    `Model.prepare_messages` already puts tool results in call order, so this sends the continued request unprepared
    to check `XaiModel`'s own ordering.
    """
    calls = [create_tool_call('call_a', 'a', {}), create_tool_call('call_b', 'b', {})]
    mock_client = MockXai.create_mock([create_response(tool_calls=calls), create_response(content='done')])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    params = ModelRequestParameters()
    history: list[ModelRequest | ModelResponse] = [ModelRequest(parts=[UserPromptPart('hello')])]
    stored = await m.request(m.prepare_messages(history, params), XaiModelSettings(xai_store_messages=True), params)
    assert stored.history_digest is not None
    results = ModelRequest(
        parts=[
            ToolReturnPart(tool_name='b', content='from b', tool_call_id='call_b'),
            ToolReturnPart(tool_name='a', content='from a', tool_call_id='call_a'),
        ]
    )

    await m.request(
        [*history, stored, results],
        XaiModelSettings(xai_store_messages=True, xai_previous_response_id='grok-123'),
        params,
    )

    second_kwargs = get_mock_chat_create_kwargs(mock_client)[1]
    assert second_kwargs['previous_response_id'] == 'grok-123'
    assert [message['tool_call_id'] for message in second_kwargs['messages']] == snapshot(['call_a', 'call_b'])
