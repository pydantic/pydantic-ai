"""How `OpenAIChatModel` counts a Chat Completions `usage` whose `total_tokens` exceeds `prompt_tokens + completion_tokens`."""

from __future__ import annotations as _annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

import httpx2
import pytest

from pydantic_ai import Agent
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.usage import RequestUsage, RunUsage

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from openai import AsyncOpenAI

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')

GEMINI_OPENAI_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/openai/'


@pytest.mark.vcr
async def test_gemini_usage_counts_thinking_tokens_left_out_of_completion_tokens(
    allow_model_requests: None, gemini_api_key: str
):
    """Gemini's OpenAI-compatible endpoint counts thinking tokens in `total_tokens` but not in `completion_tokens`.

    The recording reports 8 prompt, 39 completion and 439 total tokens, and no `completion_tokens_details`. The
    392-token gap counts as output and reasoning tokens, as `completion_tokens_details` would report them.
    """
    agent = Agent(
        OpenAIChatModel(
            'gemini-3.5-flash', provider=OpenAIProvider(base_url=GEMINI_OPENAI_BASE_URL, api_key=gemini_api_key)
        )
    )

    result = await agent.run('Write three short sentences about tea.')

    assert result.usage == snapshot(
        RunUsage(
            details={'reasoning_tokens': 392},
            output_reasoning_tokens=392,
            input_tokens=8,
            output_tokens=431,
            cost=Decimal('0.003891'),
            requests=1,
        )
    )


@pytest.mark.vcr
async def test_gemini_stream_usage_counts_thinking_tokens_left_out_of_completion_tokens(
    allow_model_requests: None, gemini_api_key: str
):
    """Each streamed running total counts the thinking tokens Gemini reports only in `total_tokens`.

    The recording's three usage chunks report 8 prompt tokens and 18 to 36 completion tokens, with `total_tokens` 418
    above their sum and no `completion_tokens_details`. Each running total replaces the last, as in
    `test_gemini_stream_usage_replaces_running_totals`, and carries those 418 thinking tokens.
    """
    agent = Agent(
        OpenAIChatModel(
            'gemini-3.5-flash', provider=OpenAIProvider(base_url=GEMINI_OPENAI_BASE_URL, api_key=gemini_api_key)
        )
    )

    async with agent.run_stream('Write three short sentences about tea.') as result:
        usage_at_each_step = [response.usage async for response in result.stream_response(debounce_by=None)]

    assert usage_at_each_step == snapshot(
        [
            RequestUsage(
                details={'reasoning_tokens': 418}, input_tokens=8, output_tokens=436, output_reasoning_tokens=418
            ),
            RequestUsage(
                details={'reasoning_tokens': 418}, input_tokens=8, output_tokens=454, output_reasoning_tokens=418
            ),
            RequestUsage(
                details={'reasoning_tokens': 418},
                input_tokens=8,
                output_tokens=454,
                output_reasoning_tokens=418,
                cost=Decimal('0.004098'),
            ),
            RequestUsage(
                details={'reasoning_tokens': 418},
                input_tokens=8,
                output_tokens=454,
                output_reasoning_tokens=418,
                cost=Decimal('0.004098'),
            ),
        ]
    )
    assert result.usage == snapshot(
        RunUsage(
            details={'reasoning_tokens': 418},
            requests=1,
            output_reasoning_tokens=418,
            input_tokens=8,
            output_tokens=454,
            cost=Decimal('0.004098'),
        )
    )


THINKING_GAP_USAGE: dict[str, int] = {'prompt_tokens': 8, 'completion_tokens': 33, 'total_tokens': 423}


@dataclass(frozen=True)
class Case:
    id: str
    usage: Mapping[str, object] | None
    expected: RunUsage
    model_name: str = 'gemini-3.5-flash'
    profile: OpenAIModelProfile | None = None


CASES = [
    Case(
        id='reasoning-in-completion-tokens',
        usage={
            'prompt_tokens': 8,
            'completion_tokens': 415,
            'total_tokens': 423,
            'completion_tokens_details': {'reasoning_tokens': 382},
        },
        expected=snapshot(
            RunUsage(
                details={'reasoning_tokens': 382},
                output_tokens=415,
                output_reasoning_tokens=382,
                input_tokens=8,
                cost=Decimal('0.003747'),
                requests=1,
            )
        ),
    ),
    Case(
        id='thinking-gap',
        usage=THINKING_GAP_USAGE,
        expected=snapshot(
            RunUsage(
                details={'reasoning_tokens': 382},
                output_tokens=415,
                output_reasoning_tokens=382,
                input_tokens=8,
                cost=Decimal('0.003747'),
                requests=1,
            )
        ),
    ),
    Case(
        id='no-gap',
        usage={'prompt_tokens': 8, 'completion_tokens': 33, 'total_tokens': 41},
        expected=snapshot(RunUsage(output_tokens=33, input_tokens=8, cost=Decimal('0.000309'), requests=1)),
    ),
    Case(
        id='total-below-sum',
        usage={'prompt_tokens': 8, 'completion_tokens': 33, 'total_tokens': 30},
        expected=snapshot(RunUsage(output_tokens=33, input_tokens=8, cost=Decimal('0.000309'), requests=1)),
    ),
    Case(id='no-usage', usage=None, expected=snapshot(RunUsage(unmeasured_requests=1, requests=1))),
    Case(
        id='flag-off-by-name',
        usage=THINKING_GAP_USAGE,
        model_name='gpt-5-mini',
        expected=snapshot(RunUsage(output_tokens=33, input_tokens=8, cost=Decimal('0.000068'), requests=1)),
    ),
    Case(
        id='flag-off-by-profile',
        usage=THINKING_GAP_USAGE,
        profile=OpenAIModelProfile(openai_chat_completion_tokens_omit_reasoning=False),
        expected=snapshot(RunUsage(output_tokens=33, input_tokens=8, cost=Decimal('0.000309'), requests=1)),
    ),
]


@pytest.mark.parametrize('stream', [False, True], ids=['request', 'stream'])
@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in CASES])
async def test_chat_usage_total_gap_shapes(allow_model_requests: None, case: Case, stream: bool):
    """A gap between `total_tokens` and the sum of the other two counts as output only under the profile flag.

    `thinking-gap` and `reasoning-in-completion-tokens` carry the same 382 reasoning tokens, once only in `total_tokens`
    and once in `completion_tokens_details`, and report the same usage. With the flag off, or without a positive gap,
    `completion_tokens` stays the output. A mock transport stands in for a cassette because the live Gemini probes
    produced none of the other shapes while thinking.
    """
    message = {'role': 'assistant', 'content': 'Hello'}
    fields = {'id': '1', 'created': 0, 'model': case.model_name} | (
        {} if case.usage is None else {'usage': dict(case.usage)}
    )
    if stream:
        chunk = fields | {
            'object': 'chat.completion.chunk',
            'choices': [{'index': 0, 'delta': message, 'finish_reason': 'stop'}],
        }
        response = httpx2.Response(
            200,
            content=f'data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n'.encode(),
            headers={'content-type': 'text/event-stream'},
        )
    else:
        response = httpx2.Response(
            200,
            json=fields
            | {'object': 'chat.completion', 'choices': [{'index': 0, 'message': message, 'finish_reason': 'stop'}]},
        )

    async def handler(request: httpx2.Request) -> httpx2.Response:
        return response

    async with AsyncOpenAI(
        api_key='test',
        base_url=GEMINI_OPENAI_BASE_URL,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    ) as openai_client:
        agent = Agent(
            OpenAIChatModel(case.model_name, provider=OpenAIProvider(openai_client=openai_client), profile=case.profile)
        )
        if stream:
            async with agent.run_stream('Hi') as result:
                await result.get_output()
            usage = result.usage
        else:
            usage = (await agent.run('Hi')).usage

    assert usage == case.expected
