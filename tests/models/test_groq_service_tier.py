"""Groq's service tier: the outgoing request body and the tier the response reports."""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx
import pytest

from pydantic_ai import Agent

from ..conftest import IsDatetime, try_import

with try_import() as imports_successful:
    from groq import AsyncGroq

    from pydantic_ai.models.groq import GroqModel, GroqModelSettings
    from pydantic_ai.providers.groq import GroqProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='groq not installed')


@dataclass(frozen=True)
class ServiceTierCase:
    id: str
    settings: GroqModelSettings
    sent: dict[str, object]
    """The request-body keys the tier merge writes, plus the user's own `extra_body` key."""
    served_tier: str = 'on_demand'


SERVICE_TIER_CASES = [
    ServiceTierCase(id='unified-auto', settings={'service_tier': 'auto'}, sent={'service_tier': 'auto'}),
    ServiceTierCase(id='unified-default', settings={'service_tier': 'default'}, sent={'service_tier': 'on_demand'}),
    ServiceTierCase(
        id='unified-flex', settings={'service_tier': 'flex'}, sent={'service_tier': 'flex'}, served_tier='flex'
    ),
    ServiceTierCase(id='unified-priority-omitted', settings={'service_tier': 'priority'}, sent={}),
    ServiceTierCase(
        id='groq-performance',
        settings={'groq_service_tier': 'performance'},
        sent={'service_tier': 'performance'},
        served_tier='performance',
    ),
    ServiceTierCase(
        id='groq-wins-over-unified',
        settings={'service_tier': 'flex', 'groq_service_tier': 'on_demand'},
        sent={'service_tier': 'on_demand'},
    ),
    ServiceTierCase(
        id='merged-with-extra-body-and-reasoning-effort',
        settings={'service_tier': 'flex', 'groq_reasoning_effort': 'low', 'extra_body': {'user': 'u1'}},
        sent={'service_tier': 'flex', 'reasoning_effort': 'low', 'user': 'u1'},
        served_tier='flex',
    ),
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in SERVICE_TIER_CASES])
async def test_service_tier(allow_model_requests: None, case: ServiceTierCase) -> None:
    """The tier rides in the request body, and the non-streamed response's tier lands in `provider_details`.

    A mock transport stands in for a cassette: no Groq key was available to record one, and the claim is the
    outgoing request body, which the transport captures as sent.
    https://github.com/pydantic/pydantic-ai/issues/12072
    """
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                'id': '123',
                'object': 'chat.completion',
                'created': 1704067200,
                'model': 'llama-3.3-70b-versatile',
                'choices': [
                    {'index': 0, 'message': {'role': 'assistant', 'content': 'world'}, 'finish_reason': 'stop'}
                ],
                'service_tier': case.served_tier,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AsyncGroq(api_key='test', base_url='http://localhost', max_retries=0, http_client=http_client)
        agent = Agent(GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(groq_client=client)))
        result = await agent.run('hello', model_settings=case.settings)

    [body] = bodies
    assert {key: body[key] for key in ('service_tier', 'reasoning_effort', 'user') if key in body} == case.sent
    assert result.response.provider_details == {
        'finish_reason': 'stop',
        'timestamp': IsDatetime(),
        'service_tier': case.served_tier,
    }


async def test_service_tier_streamed_request(allow_model_requests: None) -> None:
    """A streamed request carries the tier in its body too.

    A mock transport stands in for a cassette, for the same reason as in `test_service_tier`.
    """
    bodies: list[dict[str, object]] = []
    chunk = {
        'id': '1',
        'object': 'chat.completion.chunk',
        'created': 0,
        'model': 'llama-3.3-70b-versatile',
        'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'world'}, 'finish_reason': 'stop'}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=f'data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n'.encode(),
            headers={'content-type': 'text/event-stream'},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AsyncGroq(api_key='test', base_url='http://localhost', max_retries=0, http_client=http_client)
        agent = Agent(GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(groq_client=client)))
        async with agent.run_stream('hello', model_settings={'service_tier': 'flex'}) as result:
            assert await result.get_output() == 'world'

    [body] = bodies
    assert (body['stream'], body['service_tier']) == (True, 'flex')
