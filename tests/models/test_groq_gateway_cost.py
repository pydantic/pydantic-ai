"""Gateway cost and Groq token usage survive the real SDK's SSE parsing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal

import httpx
import pytest

from pydantic_ai import Agent, UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='Groq not installed')


@dataclass
class Case:
    id: str
    cost_blocks: dict[str, dict[str, str]] = field(default_factory=dict[str, dict[str, str]])
    top_level_usage: bool = True
    expected_cost: Decimal = Decimal('0.0000669')
    exceeds_limit: bool = False


@pytest.mark.parametrize(
    'case',
    [
        Case(
            id='canonical',
            cost_blocks={'logfire_ai_gateway': {'cost': '0.5'}, 'pydantic_ai_gateway': {'cost_estimate': '0.25'}},
            expected_cost=Decimal('0.5'),
            exceeds_limit=True,
        ),
        Case(
            id='legacy',
            cost_blocks={'pydantic_ai_gateway': {'cost_estimate': '0.25'}},
            expected_cost=Decimal('0.25'),
            exceeds_limit=True,
        ),
        Case(
            id='zero',
            cost_blocks={'logfire_ai_gateway': {'cost': '0'}, 'pydantic_ai_gateway': {'cost_estimate': '0.25'}},
            expected_cost=Decimal('0'),
        ),
        Case(id='invalid', cost_blocks={'logfire_ai_gateway': {'cost': 'invalid'}}),
        Case(id='no-cost'),
        Case(id='nested-only', top_level_usage=False),
    ],
    ids=lambda case: case.id,
)
async def test_groq_stream_gateway_cost(case: Case, allow_model_requests: None):
    """The Gateway writes cost to top-level usage while Groq supplies tokens in x_groq.usage.

    Stub HTTP underneath the real SDK to cover cost fields that native Groq cannot return.
    Distinct top-level token counts ensure the cost fix does not change token-source precedence.
    """
    chunk: dict[str, object] = {
        'id': 'chatcmpl-1',
        'object': 'chat.completion.chunk',
        'created': 1704067200,
        'model': 'llama-3.3-70b-versatile',
    }
    final_chunk: dict[str, object] = {
        **chunk,
        'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
        'x_groq': {
            'usage': {
                'prompt_tokens': 100,
                'completion_tokens': 10,
                'total_tokens': 110,
                'completion_tokens_details': {'reasoning_tokens': 2},
            }
        },
    }
    if case.top_level_usage:
        final_chunk['usage'] = {
            'prompt_tokens': 999,
            'completion_tokens': 999,
            'total_tokens': 1998,
            **case.cost_blocks,
        }
    events: list[dict[str, object]] = [
        {**chunk, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'hi'}}]},
        final_chunk,
    ]
    content = ''.join(f'data: {json.dumps(event)}\n\n' for event in events) + 'data: [DONE]\n\n'
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, content=content.encode(), headers={'content-type': 'text/event-stream'})
    )
    async with httpx.AsyncClient(transport=transport) as client:
        provider = GroqProvider(api_key='test', http_client=client)
        agent = Agent(GroqModel('llama-3.3-70b-versatile', provider=provider))
        async with agent.run_stream('hello') as result:
            assert await result.get_output() == 'hi'
        assert result.usage.cost == case.expected_cost
        assert (result.usage.input_tokens, result.usage.output_tokens, result.usage.details) == snapshot(
            (100, 10, {'reasoning_tokens': 2})
        )

        if case.exceeds_limit:
            with pytest.raises(UsageLimitExceeded, match='cost_limit'):
                async with agent.run_stream('hello', usage_limits=UsageLimits(cost_limit=Decimal('0.001'))) as limited:
                    await limited.get_output()
        else:
            async with agent.run_stream('hello', usage_limits=UsageLimits(cost_limit=Decimal('0.001'))) as limited:
                await limited.get_output()
            assert limited.usage.cost == case.expected_cost
