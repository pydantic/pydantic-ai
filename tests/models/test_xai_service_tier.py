"""xAI's service tier: the tier `chat.create()` receives and the tier the response reports.

`MockXai` captures the `chat.create()` arguments as sent, and each test sets the tier on the response proto.
https://github.com/pydantic/pydantic-ai/issues/12071
"""

from __future__ import annotations

from typing import Literal

import pytest

from pydantic_ai import Agent, ModelResponse
from pydantic_ai.settings import ModelSettings, ServiceTier

from ..conftest import message, try_import
from .mock_xai import MockXai, create_response, get_grok_text_chunk, get_mock_chat_create_kwargs

with try_import() as imports_successful:
    from xai_sdk.proto import usage_pb2
    from xai_sdk.service_tier import service_tier_to_proto

    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.providers.xai import XaiProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='xai_sdk not installed')

MODEL_NAME = 'grok-4-fast-non-reasoning'


@pytest.mark.parametrize(
    'service_tier,sent',
    [
        ('default', {'service_tier': 'default'}),
        ('priority', {'service_tier': 'priority'}),
        ('auto', {}),
        ('flex', {}),
    ],
)
async def test_xai_service_tier(allow_model_requests: None, service_tier: ServiceTier, sent: dict[str, str]):
    """`default` and `priority` reach `chat.create`; `auto` and `flex` have no xAI tier and are omitted."""
    mock_client = MockXai.create_mock([create_response(content='test response')])
    m = XaiModel(MODEL_NAME, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, model_settings=ModelSettings(service_tier=service_tier))

    await agent.run('Hello')

    kwargs = get_mock_chat_create_kwargs(mock_client)[0]
    assert {key: value for key, value in kwargs.items() if key == 'service_tier'} == sent


@pytest.mark.parametrize('served_tier', ['default', 'priority'])
async def test_xai_service_tier_provider_details(
    allow_model_requests: None, served_tier: Literal['default', 'priority']
):
    """The tier xAI reports serving the response at lands in `provider_details`."""
    response = create_response(content='test response')
    response.proto.service_tier = service_tier_to_proto(served_tier)
    mock_client = MockXai.create_mock([response])
    agent = Agent(XaiModel(MODEL_NAME, provider=XaiProvider(xai_client=mock_client)))

    result = await agent.run('Hello')

    assert message(result.all_messages(), ModelResponse, index=-1).provider_details == {'service_tier': served_tier}


async def test_xai_stream_service_tier_provider_details(allow_model_requests: None):
    """A streamed response records the tier xAI reports on the accumulated response."""
    first_response, first_chunk = get_grok_text_chunk('Hello', finish_reason='')
    final_response, final_chunk = get_grok_text_chunk(' world')
    final_response.proto.service_tier = usage_pb2.SERVICE_TIER_PRIORITY
    mock_client = MockXai.create_mock_stream([[(first_response, first_chunk), (final_response, final_chunk)]])
    agent = Agent(XaiModel(MODEL_NAME, provider=XaiProvider(xai_client=mock_client)))

    async with agent.run_stream('Hello') as result:
        await result.get_output()

    assert message(result.all_messages(), ModelResponse, index=-1).provider_details == {'service_tier': 'priority'}
